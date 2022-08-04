import logging
import sys
from typing import Any

import torch

import utils

sys.path.append(".")


from .resnet_models import BasicBlock, Bottleneck, L0ResNet, L0ResNet50


def l1_layerwise_prune_model(model, layerwise_target_density):

    # Turn on all gates for untrained model
    for layer in model.layers_dict["l0"]:

        # Compute L1 norm for each of the (output) convolutional kernels
        kernel_norms = layer.weight.data.norm(p=1, dim=[1, 2, 3])

        # Get the norm for which target_sparsity% of the values fall below
        # We will turn off all gates associated to kernels with norms BELOW this number
        norm_cutoff = torch.quantile(kernel_norms, q=(1.0 - layerwise_target_density))

        # Keep gates whose kernels had large-enough L1 norm
        on_gates_ix = (kernel_norms > norm_cutoff).float()
        # Turn to [-1, 1] values (since log-alpha expects negative values to turn off gates)
        on_gates_ix = 2 * (on_gates_ix - 0.5)

        # Force gates to be turned on or off with a large scale for log_alpha
        layer.weight_log_alpha.data = 5.0 * on_gates_ix


def load_module_from_pretrained(module, pre_module, pretrain_type=None):

    easy_copy_types = ["BatchNorm", "MaxPool"]
    if any([_ in type(module).__name__ for _ in easy_copy_types]):
        module.load_state_dict(pre_module.state_dict())

    if isinstance(module, torch.nn.Sequential):
        # These are the network layers containing Blocks
        for block, pretrained_block in zip(module, pre_module):
            load_module_from_pretrained(block, pretrained_block, pretrain_type)

    if isinstance(module, (Bottleneck, BasicBlock)):

        num_convs = 2 if isinstance(module, BasicBlock) else 3
        for conv_id in range(1, num_convs + 1):
            load_module_from_pretrained(
                getattr(module, f"conv{conv_id}"), getattr(pre_module, f"conv{conv_id}")
            )
            pre_module_bn_name = "bn" if pretrain_type == "torch" else "batch_norm"
            load_module_from_pretrained(
                getattr(module, f"batch_norm{conv_id}"),
                getattr(pre_module, pre_module_bn_name + str(conv_id)),
            )

        if module.has_downsampler:
            if pretrain_type == "torch":
                pre_shortcut_conv = pre_module.downsample[0]
                pre_shortcut_bn = pre_module.downsample[1]
            else:
                pre_shortcut_conv = pre_module.shortcut_conv
                pre_shortcut_bn = pre_module.shortcut_bn

            load_module_from_pretrained(module.shortcut_conv, pre_shortcut_conv)
            load_module_from_pretrained(module.shortcut_bn, pre_shortcut_bn)

    else:
        if hasattr(module, "weight"):
            module.weight.data.copy_(pre_module.weight.data)
        if hasattr(module, "bias") and module.bias is not None:
            assert hasattr(pre_module, "bias")
            if pre_module.bias is not None:
                module.bias.data.copy_(pre_module.bias.data)


def load_pretrained_ResNet50(args, dummy_model):

    # Download parameters with WandB api
    filters = {
        "$and": [
            {"config.model_type": args.model_type},
            {"config.task_type": "baseline"},
            {"state": "finished"},
            {"tags": "baseline"},
        ]
    }
    foo = utils.wandb_utils.get_baseline_model(
        "constrained_l0", args.dataset_name, filters
    )
    baseline_state, baseline_config, baseline_wandb_summary = foo

    train_top1 = baseline_wandb_summary["train/epoch/top1"]
    val_top1 = baseline_wandb_summary["val/top1"]
    logging.info(f"Imported model with Error (train/val): {train_top1} / {val_top1}")

    # This is a pretrained PurgedResNet
    dummy_model.load_state_dict(baseline_state)

    return dummy_model


def pretrained_as_l0_model(pretrained, pretrain_type, **kwargs: Any) -> L0ResNet:

    # Instantiate an L0ResNet50 model
    l0_model = L0ResNet50(
        num_classes=1000, l0_conv_ix=["conv1", "conv2", "conv3"], **kwargs
    )

    with torch.no_grad():

        # Load all parameters from the pretrained model
        load_module_from_pretrained(l0_model.conv1, pretrained.conv1)

        pre_bn = pretrained.bn1 if pretrain_type == "torch" else pretrained.batch_norm1
        load_module_from_pretrained(l0_model.batch_norm1, pre_bn)

        if l0_model.do_initial_maxpool:
            load_module_from_pretrained(l0_model.maxpool, pretrained.maxpool)

        for layer_id in range(1, 4 + 1):
            untrained_layer = getattr(l0_model, f"layer{layer_id}")
            pretrained_layer = getattr(pretrained, f"layer{layer_id}")
            # Must indicate pretrain_type to correctly load batch norm
            load_module_from_pretrained(
                untrained_layer, pretrained_layer, pretrain_type
            )

        load_module_from_pretrained(l0_model.avgpool, pretrained.avgpool)
        pre_fcout = pretrained.fc if pretrain_type == "torch" else pretrained.fcout
        load_module_from_pretrained(l0_model.fcout, pre_fcout)

    return l0_model
