"""ResNets with L0 regularization."""

import logging
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .l0_layers import L0Conv2d
from .models import BaseL0Model, create_dense_conv2d_layer, create_dense_linear_layer
from .utils import apply_bn_after_conv2d, init_batch_norm


class PreActivationBlock(nn.Module):
    """Basic residual block for building a residual net. Employs pre-activation.
    Its first convolution is sparsifiable, the second one and the convolutional
    shortcut are not. It does not employ linear layers."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        use_bias: bool,
        sparsity_type: str,
        kernel_size: int,
        stride: int,
        act_fn_module: nn.Module,
        bn_type: str,
        base_l0_kwargs: dict,
    ):

        super(PreActivationBlock, self).__init__()

        self.sparsity_type = sparsity_type
        self.stride = stride
        self.act_fn_module = act_fn_module
        self.bn_type = bn_type

        self.batch_norm1 = nn.BatchNorm2d(in_planes)
        self.act1 = act_fn_module()

        # Amlab code does not use dropout, rather weight decay.
        # "For the layers with the hard concrete gates we divided the weight decay
        # coefficient by 0.7 to ensure that a-priori we assume the same length-scale
        # as the 0.3 dropout equivalent network." (p. 8)
        conv1_kwargs = deepcopy(base_l0_kwargs)
        conv1_kwargs["weight_decay"] /= 0.7

        # This is the only L0 layer of the block.
        self.conv1 = L0Conv2d(
            in_planes,
            out_planes,
            use_bias,
            sparsity_type,
            (kernel_size, kernel_size),
            stride=1,
            padding=1,
            base_l0_kwargs=conv1_kwargs,
        )

        # Second layer. This one downsamples with its stride.
        self.batch_norm2 = init_batch_norm(out_planes, self.bn_type)

        self.act2 = act_fn_module()
        self.conv2 = create_dense_conv2d_layer(
            out_planes,
            out_planes,
            kernel_size=(kernel_size, kernel_size),
            stride=stride,
            padding=1,
            use_bias=use_bias,
        )

        # Does the block downsample in terms of channels/planes?
        self.preserves_planes = in_planes == out_planes
        if not self.preserves_planes:
            # Note that in Amlab code stride>1 only happens when preserves_planes==False
            # This code has not been tested for the case of preserves_planes==True and stride>1
            self.shortcut_conv = create_dense_conv2d_layer(
                in_planes,
                out_planes,
                kernel_size=(1, 1),
                stride=stride,
                padding=0,
                use_bias=use_bias,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of a residual block. The convolutional shortcut reshapes
        the input to match the output shape. Manually checked that refactor
        macthes logic in original Amlab code."""

        # Pre-activation employed. BatchNorm -> Act_fn -> Conv
        out = self.act1(self.batch_norm1(x))

        shortcut_output = x
        if not self.preserves_planes:
            # BN and activation function have already been applied.
            shortcut_output = self.shortcut_conv(out)

        out = apply_bn_after_conv2d(out, self.conv1, self.batch_norm2)
        out = self.conv2(self.act2(out))
        out = torch.add(out, shortcut_output)

        return out


class L0WideResNet(BaseL0Model):
    """Implementation of a Wide ResNet with some conv layers being L0 compatible.
    It has three network blocks, each with 6 layers."""

    def __init__(
        self,
        num_classes: int,
        input_shape: Tuple[int],
        use_bias: bool,
        sparsity_type: str,
        depth: int = 28,
        widen_factor: int = 10,
        kernel_sizes: Tuple[int, int, int, int] = (3, 3, 3, 3),
        droprate_init: float = 0.3,
        weight_decay: float = 5e-4,
        temperature: float = 2.0 / 3.0,
        act_fn_module: nn.Module = nn.ReLU,
        bn_type: str = "L0",
        l2_detach_gates: bool = False,
    ):

        super(L0WideResNet, self).__init__(weight_decay=weight_decay)

        # Preset number of channels. Could be increased with widen_factor
        num_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        assert (depth - 4) % 6 == 0
        self.net_block_depth = (
            depth - 4
        ) // 6  # number of basic blocks in network block.

        self.input_shape = input_shape

        if bn_type != "identity" and use_bias:
            Warning(
                "You have selected to use bias *and* batch normalization.\
                The BN layer will cancel the effect of the bias."
            )

        base_layer_kwargs = {
            "droprate_init": droprate_init,
            "weight_decay": self.weight_decay,
            "temperature": temperature,
            "l2_detach_gates": l2_detach_gates,
        }

        # 1st conv before any network block
        self.conv1 = create_dense_conv2d_layer(
            in_channels=self.input_shape[0],
            out_channels=num_channels[0],
            kernel_size=(kernel_sizes[0], kernel_sizes[0]),
            stride=1,
            padding=1,
            use_bias=use_bias,
        )

        # Network layers
        self.layer1 = self._make_layer(
            self.net_block_depth,
            num_channels[0],
            num_channels[1],
            use_bias,
            sparsity_type,
            kernel_size=kernel_sizes[1],
            stride=1,
            act_fn_module=act_fn_module,
            bn_type=bn_type,
            base_l0_kwargs=base_layer_kwargs,
        )
        self.layer2 = self._make_layer(
            self.net_block_depth,
            num_channels[1],
            num_channels[2],
            use_bias,
            sparsity_type,
            kernel_size=kernel_sizes[2],
            stride=2,
            act_fn_module=act_fn_module,
            bn_type=bn_type,
            base_l0_kwargs=base_layer_kwargs,
        )
        self.layer3 = self._make_layer(
            self.net_block_depth,
            num_channels[2],
            num_channels[3],
            use_bias,
            sparsity_type,
            kernel_size=kernel_sizes[3],
            stride=2,
            act_fn_module=act_fn_module,
            bn_type=bn_type,
            base_l0_kwargs=base_layer_kwargs,
        )

        # Batch Normalization, activation and output layer
        self.batch_norm = nn.BatchNorm2d(num_channels[3])
        self.activation = act_fn_module()

        # Hard-coded use_bias set to True for the final prediction layer
        self.fcout = create_dense_linear_layer(
            num_channels[3], num_classes, use_bias=True
        )

        self.layers_dict, self.params_dict = self.gather_layers_and_params()

    def _make_layer(
        self,
        num_res_blocks: int,
        in_planes: int,
        out_planes: int,
        use_bias: bool,
        sparsity_type: str,
        kernel_size: int,
        stride: int,
        act_fn_module: nn.Module,
        bn_type: str,
        base_l0_kwargs: dict,
    ) -> nn.Sequential:

        assert (
            num_res_blocks > 0
        ), "Number of residual blocks should be strictly positive"

        # First basic block of network block downsamples in_planes -> out_planes AND may do stride != 1
        layers = [
            PreActivationBlock(
                in_planes,
                out_planes,
                use_bias,
                sparsity_type,
                kernel_size,
                stride,
                act_fn_module,
                bn_type,
                base_l0_kwargs,
            )
        ]
        for _ in range(num_res_blocks - 1):
            layers.append(
                PreActivationBlock(
                    out_planes,
                    out_planes,
                    use_bias,
                    sparsity_type,
                    kernel_size,
                    1,
                    act_fn_module,
                    bn_type,
                    base_l0_kwargs,
                )
            )

        return nn.Sequential(*layers)

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
