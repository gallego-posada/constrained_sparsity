import contextlib
import copy
import dataclasses
import functools
import logging
from typing import Dict, Optional, Union

import cooper
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from ptflops import get_model_complexity_info

import sparse

from .basic_utils import AverageMeter, compute_accuracy
from .cmp_utils import BaselineProblem, ConstrainedL0Problem, PenalizedL0Problem
from .wandb_utils import collect_gates_hist, collect_l0_stats, collect_multipliers

OPTIM_DICT = {
    "None": None,
    "SGD": torch.optim.SGD,
    "SGDM": functools.partial(torch.optim.SGD, momentum=0.9),
    "Adam": torch.optim.Adam,
    "Adagrad": torch.optim.Adagrad,
    "RMSprop": torch.optim.RMSprop,
    # Optimizers with extrapolation, from cooper.optim
    "ExtraSGD": cooper.optim.ExtraSGD,
    "ExtraSGDM": functools.partial(cooper.optim.ExtraSGD, momentum=0.9),
    "ExtraAdam": cooper.optim.ExtraAdam,
}


def train(
    cmp,
    model,
    model_module,
    formulation,
    train_loader,
    constrained_optimizer,
    step_id,
    epoch,
    reg_config,
    use_wandb: bool,
    debug_batches=None,
    print_interval=150,
):
    """Train one epoch on the training set"""

    wandb_step_dict = {}

    # global total_steps, exp_flops, exp_l0, args, writer
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    for batch_id, (input_, target_) in enumerate(train_loader):

        step_id += 1

        # Force stop training after required number of batches
        if (debug_batches is not None) and batch_id > debug_batches:
            break

        if batch_id % print_interval == 0:
            logging.info("%s %s %s", epoch, batch_id, len(train_loader))

        if torch.cuda.is_available():
            target_ = target_.cuda()
            input_ = input_.cuda()

        constrained_optimizer.zero_grad()

        # We keep reg_config as a placeholder for baseline and mp runs
        lagrangian = formulation.composite_objective(
            cmp.closure,
            input_,
            target_,
            model,
            reg_config,
        )
        formulation.custom_backward(lagrangian)
        if hasattr(constrained_optimizer.primal_optimizer, "extrapolation"):
            constrained_optimizer.step(cmp.closure, model, input_, target_, reg_config)
        else:
            constrained_optimizer.step()

        if isinstance(model, sparse.BaseL0Model):
            # Clamp parameters for training stability
            [layer.clamp_parameters() for layer in model_module.layers_dict["l0"]]

            # Update constraint levels dynamically
            if reg_config.const_update == "dynamic":
                old_current = copy.deepcopy(reg_config.current_level)
                ineq_multipliers = formulation.state()[0]
                reg_config.update_current_level(ineq_multipliers)
                if not torch.allclose(reg_config.current_level, old_current):
                    logging.info("Const-Levels: {}".format(reg_config.current_level))

        outputs = cmp.state.misc["outputs"]
        loss = cmp.state.loss

        # Measure accuracy and update meters
        prec1 = compute_accuracy(outputs, target_, topk=(1,))[0]
        losses.update(loss.item(), input_.size(0))
        top1.update(100 - prec1.item(), input_.size(0))

        if use_wandb and (step_id % 50 == 0):  # Log every 50 steps

            # Log loss/acc stats
            log_dict = {
                "loss": loss.item(),
                "top1": 100.0 - prec1.item(),
                "lagrangian": lagrangian.item(),
            }

            if "reg_stats" in cmp.state.misc:
                # Collect L0 metrics
                reg_dict = collect_l0_stats(cmp.state.misc["reg_stats"])
                log_dict.update(reg_dict)

            # Collect multiplier values
            multiplier_dict = collect_multipliers(formulation)
            log_dict.update(multiplier_dict)

            # Re-name keys for grouping
            log_dict = {"train/batch/" + key: val for key, val in log_dict.items()}

            log_dict["epoch"] = epoch

            wandb_step_dict[step_id] = log_dict

    # Log aggregate loss over whole epoch
    epoch_log_dict = {"train/epoch/loss": losses.avg, "train/epoch/top1": top1.avg}
    if isinstance(model_module, sparse.BaseL0Model):
        hist_dict = collect_gates_hist(model_module.layers_dict["l0"])
        epoch_log_dict.update(hist_dict)

    epoch_log_dict["epoch"] = epoch

    return step_id, wandb_step_dict, epoch_log_dict


def validation_loop(
    epoch, val_loader, model, model_module, val_prop, cmp, do_purge_model
):

    model.eval()
    log_dict = {"epoch": epoch}

    if cmp.state.misc is not None and "reg_stats" in cmp.state.misc:
        # Obtain model L0, L0_full and L2 from current CMP state since these values
        # do not depend on the validation data
        reg_stats = cmp.state.misc["reg_stats"]
        log_dict["l0_model"] = reg_stats.l0_model
        log_dict["l0_full"] = reg_stats.l0_full
        log_dict["l2_model"] = reg_stats.l2_model

    if do_purge_model and isinstance(model_module, sparse.BaseL0Model):

        val_model = purge_model(model_module)

        # Log density (% purged parameters / original parameters) for sparsifiable layers
        layer_density_stats = {}
        for layer_path, layer_density in val_model.layer_densities.items():
            layer_density_stats[layer_path + "_dsty"] = layer_density
        log_dict.update(layer_density_stats)

    else:
        # Baseline runs and magnitude pruning runs do not allow for purging
        if isinstance(model, nn.parallel.DistributedDataParallel):
            val_model = model.module
        else:
            val_model = model

    val_model.eval()
    val_model_module = val_model

    if torch.cuda.is_available():
        val_model = val_model.cuda()
        if isinstance(model, nn.parallel.DistributedDataParallel):
            # TODO: validation loop could be distributed. This would require to sync
            # metrics across ranks. See https://github.com/pytorch/vision/blob/d6e39ff76c82c7510f68a7aa637f015e7a86f217/references/classification/train.py#L61
            val_model = nn.DataParallel(val_model)
            val_model_module = val_model.module

    # Only get MACS (~0.5 FLOPS) and params for purged models.
    # Ptflops library does not support L0XXX modules
    log_dict.update(get_macs_and_params(val_model_module))

    with torch.inference_mode():

        loss_meter = AverageMeter()
        top1_meter = AverageMeter()

        for step, (input_, target_) in enumerate(val_loader):
            # Only evaluate a proportion of the val_loader
            if step / len(val_loader) > val_prop:
                break

            if torch.cuda.is_available():
                target_ = target_.cuda()
                input_ = input_.cuda()

            output_ = model.forward(input_)
            if hasattr(model_module, "regularization"):
                loss, _ = cmp.loss_func(model, output_, target_)
            else:
                loss = cmp.loss_func(model, output_, target_)

            loss_meter.update(loss.item(), input_.size(0))
            prec1 = compute_accuracy(output_.data, target_, topk=(1,))[0]
            top1_meter.update(100 - prec1.item(), input_.size(0))

    log_dict.update({"loss": loss_meter.avg, "top1": top1_meter.avg})

    log_dict = {"val/" + key: val for key, val in log_dict.items()}

    return log_dict


@dataclasses.dataclass
class ModelKwargs:
    sparsity_type: str
    weight_decay: Optional[float]
    l2_detach_gates: Optional[bool]
    temperature: Optional[float]
    use_bias: Optional[bool]
    act_fn_module: Optional[torch.nn.Module]
    droprate_init: Optional[float]


def construct_model(args, num_classes, input_shape) -> torch.nn.Module:
    """Build model used for training based on argparser arguments, number of
    classes and input size."""

    # First, an L0 model is constructed
    model_kwargs = ModelKwargs(
        sparsity_type=args.sparsity_type,
        weight_decay=args.weight_decay,
        l2_detach_gates=args.l2_detach_gates,
        temperature=args.temp,
        use_bias=args.use_bias,
        act_fn_module=getattr(torch.nn, args.act_fn),
        droprate_init=args.droprate_init,
    )

    model_kwargs = dataclasses.asdict(model_kwargs)

    # First, construct L0 model. It will be purged afterwards in the case of
    # baseline/magnitude pruning tasks.
    if args.model_type == "MLP":
        in_size = np.prod(input_shape)
        model = sparse.L0MLP(
            in_size, num_classes, layer_dims=(300, 100), **model_kwargs
        )
    elif args.model_type == "LR":
        model = sparse.L0MLP(784, num_classes, layer_dims=(), **model_kwargs)
    elif args.model_type == "LeNet":
        assert len(input_shape) == 3
        # Expects (channels, height, width)
        model = sparse.L0LeNet5(
            num_classes,
            input_shape=(input_shape[0], input_shape[1], input_shape[2]),
            conv_dims=(20, 50),
            fc_dims=500,
            **model_kwargs,
        )
    elif args.model_type == "ResNet-28-10":
        model = sparse.L0WideResNet(
            num_classes,
            input_shape,
            depth=28,
            bn_type=args.bn_type,
            widen_factor=10,
            **model_kwargs,
        )
    elif args.model_type == "ResNet-16-8":
        model = sparse.L0WideResNet(
            num_classes,
            input_shape,
            depth=16,
            bn_type=args.bn_type,
            widen_factor=8,
            **model_kwargs,
        )
    elif args.model_type == "L0ResNet18":
        model = sparse.L0ResNet18(
            num_classes,
            bn_type=args.bn_type,
            l0_conv_ix=["conv1", "conv2"],  # All convs except shortcut are L0
            **model_kwargs,
        )
    elif args.model_type == "L0ResNet50":
        model = sparse.L0ResNet50(
            num_classes,
            bn_type=args.bn_type,
            l0_conv_ix=["conv1", "conv2", "conv3"],  # All convs except shortcut are L0
            **model_kwargs,
        )
    else:
        raise ValueError("Did not understand model_type")

    if args.task_type == "gated":
        # Return the constructed L0 model
        return model
    elif args.task_type == "magnitude_pruning":
        return magnitude_prune_model(args, model, model_kwargs)
    elif args.task_type == "baseline":
        return baseline_model(model)


def baseline_model(model):
    for layer in model.layers_dict["l0"]:
        # Ensure that all the gates are on. For log_alpha > 2.56, medians of
        # gates are 1 and thus purging results in a fully dense model without
        # fractional gates
        layer.weight_log_alpha.data.fill_(5.0)

    return purge_model(model)


def magnitude_prune_model(args, model, model_kwargs):

    if not isinstance(model, sparse.L0ResNet):
        raise ValueError("Only ResNet50 baselines supported for MP for now")
    if args.pretrain_type is None:
        raise ValueError(
            "Must specify which pretrained model to consider for MP experiments"
        )
    if args.sparsity_type != "structured":
        raise NotImplementedError("unstructured sparsity not supported for MP")

    if args.pretrain_type == "torch":
        # Download the pretrained weights from the PyTorch repository
        pretrained = tv_models.resnet50(pretrained=True, progress=True)
    else:
        # Baseline runs loaded from WandB saved the state_dict of a PurgedResNet
        # We now get a fully dense PurgedResNet dummy model. The pre-trained
        # parameters will be loaded into the dummy model.
        dummy_model = baseline_model(model)

        # Load the parameters of a fully dense pre-trained ResNet50 from WandB
        pretrained = sparse.load_pretrained_ResNet50(args, dummy_model)
        del dummy_model

    # For magnitude_pruning, we leverage the log_alpha gates of L0 models.
    # Then, we copy the weights of the pretrained model into an L0ResNet50
    l0_pretrained = sparse.pretrained_as_l0_model(
        pretrained, args.pretrain_type, **model_kwargs
    )

    # Weights and biases are the same for both models. We must guarantee
    # that the median gates of l0_pretrained are 1 to avoid failing checks due to
    # parameter shrinkage.
    with torch.no_grad():
        for layer in l0_pretrained.layers_dict["l0"]:
            layer.weight_log_alpha.fill_(5.0)
    compare_forwards(pretrained, l0_pretrained)
    del pretrained

    ## Commented block used for testing the purging
    # pruned = purge_model(l0_pretrained)
    # compare_forwards(pruned, l0_pretrained)

    logging.info("Pretrained ResNet50 loaded and tested successfully")

    # Magnitude prune the L0 baseline model via its log_alpha parameters
    assert len(args.target_density) == 1
    tdst = args.target_density[0]
    sparse.l1_layerwise_prune_model(l0_pretrained, tdst)

    # Purge the model to remove log_alpha parameters
    return purge_model(l0_pretrained)


def compare_forwards(model1, model2):
    model1.eval()
    model2.eval()
    with torch.inference_mode():
        for _ in range(10):
            # Assert complete forward matches
            x = torch.randn((10, 3, 224, 224))
            assert torch.allclose(model1.forward(x), model2.forward(x))


def construct_problem(
    loss_module: nn.Module = nn.CrossEntropyLoss(),
    task_type: str = "gated",
    target_density: list = None,
    lmbdas: list = None,
    weight_decay: float = None,
) -> Union[ConstrainedL0Problem, PenalizedL0Problem]:

    if task_type == "gated":
        if target_density is not None:
            return ConstrainedL0Problem(loss_module)
        elif lmbdas is not None:
            return PenalizedL0Problem(loss_module)
        else:
            raise ValueError("Must provide either target_density or lmbdas for 'gated'")
    elif task_type == "baseline":
        assert target_density is None
        assert lmbdas is None
        return BaselineProblem(loss_module, weight_decay)
    elif task_type == "magnitude_pruning":
        assert target_density is not None
        assert lmbdas is None
        return BaselineProblem(loss_module, weight_decay)
    else:
        raise ValueError("Did not understand task_type")


def purge_model(model):
    if isinstance(model, sparse.L0ResNet):
        return sparse.PurgedResNet(model)
    elif isinstance(model, sparse.L0WideResNet):
        return sparse.PurgedWideResNet(model)
    elif isinstance(model, (sparse.L0MLP, sparse.L0LeNet5)):
        return sparse.PurgedModel(model)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


def get_macs_and_params(model) -> Dict[str, int]:
    """Compute number of parameters of the model and FLOPS for forward
    computations. Leveraging ptflops library."""
    with contextlib.redirect_stdout(None):
        macs, params = get_model_complexity_info(
            model,
            model.input_shape,
            print_per_layer_stat=True,
            as_strings=False,
            ost=None,
        )

    return {"macs": macs, "params": params}
