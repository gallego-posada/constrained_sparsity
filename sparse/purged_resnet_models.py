from collections import OrderedDict
from copy import deepcopy
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .l0_layers import L0Conv2d
from .purged_models import (
    PrunedLayerInfo,
    create_purged_batch_norm,
    create_purged_conv,
    purge_conv_modules,
)
from .resnet_models import BasicBlock, Bottleneck, L0ResNet
from .utils import apply_bn_after_conv2d, get_block_nsp
from .wresnet_models import L0WideResNet, PreActivationBlock

# -----------------------------------------------------------------------------
#                               WideResNets
# -----------------------------------------------------------------------------


class PurgedPreActivationBlock(PreActivationBlock, nn.Module):
    """Implements a purged residual block."""

    def __init__(self, block: PreActivationBlock):
        """Construct a purged residual block based on a reference PreActivationBlock."""
        nn.Module.__init__(self)

        act_fn_module = block.act_fn_module

        # The first batchnorm layer is not pruned as it does not have an L0Conv2d
        # before it.
        self.batch_norm1 = deepcopy(block.batch_norm1)
        self.act1 = act_fn_module()

        if block.sparsity_type == "structured":
            layer_infos, mask_list = purge_conv_modules([block.conv1])
            conv1_params = layer_infos[0]
            mask = mask_list[0]  # Used for BN2
        else:
            conv1_params = block.conv1.get_params(do_sample=False)
            conv1_params = PrunedLayerInfo(
                conv1_params["weight"],
                conv1_params["bias"],
                block.conv1.conv_kwargs,
            )
            mask = None

        self.conv1 = create_purged_conv(conv1_params)
        self.act2 = act_fn_module()
        # Potentially mask BatchNorm 2 params as it follows a pruned conv layer.
        self.batch_norm2 = create_purged_batch_norm(block.batch_norm2, mask)

        _weight = block.conv2.weight.data
        if block.sparsity_type == "structured":
            # Previous layer was out-sparse. We need to mask inputs for the next conv2d.
            _weight = _weight[:, mask, ...]
        _bias = block.conv2.bias.data if block.conv2.bias is not None else None
        _kwargs = {
            "padding": block.conv2.padding,
            "dilation": block.conv2.dilation,
            "stride": block.conv2.stride,
            "groups": block.conv2.groups,
        }
        self.conv2 = create_purged_conv(PrunedLayerInfo(_weight, _bias, _kwargs))

        self.preserves_planes = block.preserves_planes
        if not self.preserves_planes:
            # The conv shortcut is not affected by sparsification. No need to
            # mask this module.
            self.shortcut_conv = deepcopy(block.shortcut_conv)


class PurgedWideResNet(nn.Module):
    """Implements a wide residual network composed exclusively of base Pytorch
    layers, obtained by pruning those of a reference L0WideResNet."""

    def __init__(self, model: L0WideResNet):

        nn.Module.__init__(self)

        self.input_shape = model.input_shape

        # 1st conv before any network block
        self.conv1 = deepcopy(model.conv1)
        self.activation = deepcopy(model.activation)

        self.layer_densities = OrderedDict()

        for net_layer_id in range(1, 4):  # layer{1, 2, 3}
            net_layer_name = f"layer{net_layer_id}"
            network_layer = getattr(model, net_layer_name)

            purged_block_list = []
            for block_id, inner_block in enumerate(network_layer):
                block_name = f"block{block_id + 1}"

                purged_block = PurgedPreActivationBlock(inner_block)
                purged_block_list.append(purged_block)

                # Get number of entries in weights and biases (does not count gates)
                original_nsp_dict = get_block_nsp(inner_block, all_blocks=False)
                purged_nsp_dict = get_block_nsp(purged_block, all_blocks=True)

                for layer_name, orig_nsp in original_nsp_dict.items():
                    layer_path = f"{net_layer_name}_{block_name}_{layer_name}"
                    purged_nsp = purged_nsp_dict[layer_name]
                    layer_density = (1.0 * purged_nsp) / orig_nsp
                    self.layer_densities[layer_path] = layer_density

            setattr(self, net_layer_name, nn.Sequential(*purged_block_list))

        self.batch_norm = deepcopy(model.batch_norm)
        self.fcout = deepcopy(model.fcout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)

        out = self.layer1.forward(out)
        out = self.layer2.forward(out)
        out = self.layer3.forward(out)

        out = self.activation(self.batch_norm(out))

        # To match fcout input dim, pool to a 1x1 matrix per channel.
        pool_k_size = (out.shape[2], out.shape[3])
        out = F.avg_pool2d(out, pool_k_size)
        out = out.view(out.size(0), -1)

        return self.fcout(out)


# -----------------------------------------------------------------------------
#                                 ResNets
# -----------------------------------------------------------------------------


class PurgedBasicBlock(BasicBlock, nn.Module):
    """
    Purges a BasicBlock for ResNet models.
    """

    def __init__(self, block: BasicBlock):
        nn.Module.__init__(self)

        self.act_fn = deepcopy(block.act_fn)

        conv_list = [block.conv1, block.conv2]
        assert all(
            [isinstance(conv, L0Conv2d) for conv in conv_list]
        ), "Current implementation of purging for ResNet Bottlenecks only supports L0Conv2d."

        if block.sparsity_type == "structured":
            # Get PrunedLayerInfo for conv1 and conv2 and their masks.
            layer_infos, mask_list = purge_conv_modules(conv_list)
            conv1_params, conv2_params = layer_infos
            bn1_mask, bn2_mask = mask_list
        else:
            layer_infos = []
            for conv in conv_list:
                conv_params = conv.get_params(do_sample=False)
                layer_infos.append(
                    PrunedLayerInfo(
                    conv_params["weight"],
                    conv_params["bias"],
                    conv.conv_kwargs,
                    )
                )
            conv1_params, conv2_params = layer_infos
            bn1_mask, bn2_mask = None, None

        self.conv1 = create_purged_conv(conv1_params)
        self.batch_norm1 = create_purged_batch_norm(block.batch_norm1, bn1_mask)

        self.conv2 = create_purged_conv(conv2_params)
        self.batch_norm2 = create_purged_batch_norm(block.batch_norm2, bn2_mask)

        self.pre_shortcut_mask = bn2_mask

        self.has_downsampler = False
        if block.has_downsampler:
            self.has_downsampler = True

            assert not isinstance(
                block.shortcut_conv, L0Conv2d
            ), "Current implementation of purging for ResNet BasicBlocks does not support L0Conv2d shortcuts."

            self.shortcut_conv = deepcopy(block.shortcut_conv)
            self.shortcut_bn = create_purged_batch_norm(block.shortcut_bn, mask=None)


class PurgedBottleneck(Bottleneck, nn.Module):
    """
    Purges a Bottleneck for ResNet models.
    """

    def __init__(self, block: Bottleneck):
        nn.Module.__init__(self)

        self.act_fn = deepcopy(block.act_fn)

        conv_list = [block.conv1, block.conv2, block.conv3]
        assert all(
            [isinstance(conv, L0Conv2d) for conv in conv_list]
        ), "Current implementation of purging for ResNet Bottlenecks only supports L0Conv2d."

        if block.sparsity_type == "structured":
            # Get PrunedLayerInfo for conv1 and conv2 and their masks.
            layer_infos, mask_list = purge_conv_modules(conv_list)
            conv1_params, conv2_params, conv3_params = layer_infos
            bn1_mask, bn2_mask, bn3_mask = mask_list
        else:
            layer_infos = []
            for conv in conv_list:
                conv_params = conv.get_params(do_sample=False)
                layer_infos.append(
                    PrunedLayerInfo(
                    conv_params["weight"],
                    conv_params["bias"],
                    conv.conv_kwargs,
                    )
                )
            conv1_params, conv2_params, conv3_params = layer_infos
            bn1_mask, bn2_mask, bn3_mask = None, None, None

        self.conv1 = create_purged_conv(conv1_params)
        self.batch_norm1 = create_purged_batch_norm(block.batch_norm1, bn1_mask)

        self.conv2 = create_purged_conv(conv2_params)
        self.batch_norm2 = create_purged_batch_norm(block.batch_norm2, bn2_mask)

        self.conv3 = create_purged_conv(conv3_params)
        self.batch_norm3 = create_purged_batch_norm(block.batch_norm3, bn3_mask)

        self.pre_shortcut_mask = bn3_mask

        self.has_downsampler = False
        if block.has_downsampler:
            self.has_downsampler = True

            assert not isinstance(
                block.shortcut_conv, L0Conv2d
            ), "Current implementation of purging for ResNet Bottlenecks does not support L0Conv2d shortcuts."

            self.shortcut_conv = deepcopy(block.shortcut_conv)
            self.shortcut_bn = create_purged_batch_norm(block.shortcut_bn, mask=None)


class PurgedResNet(nn.Module):
    """Class for purged ResNet models."""

    def __init__(self, model: L0ResNet):
        nn.Module.__init__(self)

        self.weight_decay = model.weight_decay
        self.input_shape = model.input_shape
        self.bn_type = model.bn_type

        self.conv1 = deepcopy(model.conv1)
        self.batch_norm1 = deepcopy(model.batch_norm1)
        self.act1 = deepcopy(model.act1)

        self.do_initial_maxpool = model.do_initial_maxpool
        if self.do_initial_maxpool:
            self.maxpool = deepcopy(model.maxpool)

        # Handle modules that require purging
        self.layer_densities = OrderedDict()

        for net_layer_id in range(1, 5):  # layer{1..4}
            net_layer_name = f"layer{net_layer_id}"
            model_layer = getattr(model, net_layer_name)

            purged_block_list = []
            for block_id, inner_block in enumerate(model_layer):
                block_name = f"block{block_id + 1}"

                purged_block = self.purge_block(inner_block)
                purged_block_list.append(purged_block)

                # Get number of entries in weights and biases (does not count gates)
                original_nsp_dict = get_block_nsp(inner_block, all_blocks=False)
                purged_nsp_dict = get_block_nsp(purged_block, all_blocks=True)

                for layer_name, orig_nsp in original_nsp_dict.items():
                    layer_path = f"{net_layer_name}_{block_name}_{layer_name}"
                    purged_nsp = purged_nsp_dict[layer_name]
                    layer_density = (1.0 * purged_nsp) / orig_nsp
                    self.layer_densities[layer_path] = layer_density

            setattr(self, net_layer_name, nn.Sequential(*purged_block_list))

        self.avgpool = deepcopy(model.avgpool)
        self.fcout = deepcopy(model.fcout)

    def purge_block(self, block) -> Union[PurgedBasicBlock, PurgedBottleneck]:
        if isinstance(block, BasicBlock):
            return PurgedBasicBlock(block)
        elif isinstance(block, Bottleneck):
            return PurgedBottleneck(block)
        else:
            raise ValueError(f"Unsupported block type: {type(block)}")

    def forward(self, x):

        out = apply_bn_after_conv2d(x, self.conv1, self.batch_norm1)
        out = self.act1(out)

        if self.do_initial_maxpool:
            out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fcout(out)

        return out
