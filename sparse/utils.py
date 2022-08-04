from collections import OrderedDict
from copy import deepcopy
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .l0_layers import L0BatchNorm2d, L0Conv2d
from .models import create_dense_conv2d_layer


def get_block_nsp(block, all_blocks: bool = False):
    """Get dict of nsp for each layer in a residual block

    Args:
        block: Block whose conv layers we are analyzing
        all_blocks: If False, we only consider L0 blocks.
    """
    block_nsps = OrderedDict()

    conv_names = ["conv1", "conv2", "conv3", "shortcut_conv"]

    for conv_name in conv_names:
        if hasattr(block, conv_name):
            layer = getattr(block, conv_name)

            if all_blocks or isinstance(layer, L0Conv2d):
                layer_nsp = layer.weight.data.nelement()
                if hasattr(layer, "bias") and layer.bias is not None:
                    layer_nsp += layer.bias.data.nelement()

                block_nsps[conv_name] = layer_nsp

    return block_nsps


def init_batch_norm(channels: int, bn_type: str):
    """Initializes a batch normalization layer give number of channels and type."""
    if bn_type == "identity":
        return torch.nn.Identity()
    elif bn_type == "regular":
        return torch.nn.BatchNorm2d(channels)
    elif bn_type == "L0":
        return L0BatchNorm2d(channels)
    else:
        raise ValueError("Did not understand BatchNorm type")


def apply_bn_after_conv2d(x: torch.Tensor, conv: nn.Module, bn: bool):
    """
    Apply a convolution and batch normalization to the input tensor.
    """
    if isinstance(bn, L0BatchNorm2d):
        # Make sure we are using an L0BatchNorm2d only after a L0Conv2d layer
        assert isinstance(conv, L0Conv2d)

    if isinstance(conv, L0Conv2d) and isinstance(bn, L0BatchNorm2d):
        out, mask = conv(x, return_mask=True)
        return bn(out, mask)
    else:
        # If we do not ask explicitly for the mask then conv only returns activation map
        return bn(conv(x))


def create_general_conv2d(
    in_planes: int,
    out_planes: int,
    kernel_size: Tuple[int, int],
    base_l0_kwargs: dict,
    is_sparsifiable: bool,
    use_bias: bool,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    sparsity_type: Optional[str] = None,
) -> Union[nn.Conv2d, L0Conv2d]:
    """
    Internal switch for creating Conv2d and L0Conv2d layers in a consistent way .
    """
    if is_sparsifiable:

        l0conv_kwargs = deepcopy(base_l0_kwargs)

        # Louizos et al. 2018 code does not use dropout, but rather weight decay.
        # "For the layers with the hard concrete gates we divided the weight decay
        # coefficient by (1 - droprate_init) to ensure that a-priori we assume the
        # same length-scale as the droprate_init dropout equivalent network."
        # Louizos et al. 2018 (p. 8) mentions the case when droprate_init = 0.3.
        initial_sigmoid = 1 - base_l0_kwargs["droprate_init"]
        l0conv_kwargs["weight_decay"] /= initial_sigmoid

        return L0Conv2d(
            in_planes,
            out_planes,
            use_bias=use_bias,
            sparsity_type=sparsity_type,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            base_l0_kwargs=l0conv_kwargs,
        )

    return create_dense_conv2d_layer(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_bias=use_bias,
    )
