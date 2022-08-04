import logging
from typing import Any, List, Tuple, Type, Union

import torch
import torch.nn as nn
import torchvision.models as tv_models

from .l0_layers import L0BatchNorm2d, L0Conv2d
from .models import BaseL0Model, create_dense_conv2d_layer, create_dense_linear_layer
from .utils import apply_bn_after_conv2d, create_general_conv2d, init_batch_norm


class Bottleneck(nn.Module):
    """
    Implements a Bottleneck block for ResNet models. The code in this class is an adaptation
    of the official Pytorch-vision ResNet code to allow for L0Conv2d layers.
    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    """

    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        use_bias: bool,
        sparsity_type: str,
        act_fn_module: nn.Module,
        bn_type: str,
        base_l0_kwargs: dict,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        l0_conv_ix: List[str] = [],
        is_downsample_l0: bool = False,
    ):
        super(Bottleneck, self).__init__()

        self.sparsity_type = sparsity_type
        self.bn_type = bn_type

        width = int(out_planes * (base_width / 64.0)) * groups

        self.stride = stride
        self.act_fn = act_fn_module()

        # Both self.conv2 and self.shortcut_conv layers downsample the input when stride != 1

        self.conv1 = create_general_conv2d(
            in_planes=in_planes,
            out_planes=width,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            is_sparsifiable="conv1" in l0_conv_ix,
            base_l0_kwargs=base_l0_kwargs,
            use_bias=use_bias,
            sparsity_type=sparsity_type,
        )
        self.batch_norm1 = init_batch_norm(width, self.bn_type)

        self.conv2 = create_general_conv2d(
            in_planes=width,
            out_planes=width,
            kernel_size=(3, 3),
            stride=stride,
            padding=dilation,
            groups=groups,
            dilation=dilation,
            is_sparsifiable="conv2" in l0_conv_ix,
            base_l0_kwargs=base_l0_kwargs,
            use_bias=use_bias,
            sparsity_type=sparsity_type,
        )

        self.batch_norm2 = init_batch_norm(width, self.bn_type)

        self.conv3 = create_general_conv2d(
            in_planes=width,
            out_planes=self.expansion * out_planes,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            is_sparsifiable="conv3" in l0_conv_ix,
            base_l0_kwargs=base_l0_kwargs,
            use_bias=use_bias,
            sparsity_type=sparsity_type,
        )
        self.batch_norm3 = init_batch_norm(self.expansion * out_planes, self.bn_type)

        self.has_downsampler = False
        if stride != 1 or in_planes != self.expansion * out_planes:
            self.has_downsampler = True

            self.shortcut_conv = create_general_conv2d(
                in_planes=in_planes,
                out_planes=self.expansion * out_planes,
                kernel_size=(1, 1),
                stride=stride,
                padding=0,
                is_sparsifiable=is_downsample_l0,
                base_l0_kwargs=base_l0_kwargs,
                use_bias=use_bias,
                sparsity_type=sparsity_type,
            )

            # If shortcut_conv is not L0, then we use a regular BN layer
            shortcut_bn_type = self.bn_type if is_downsample_l0 else "regular"
            self.shortcut_bn = init_batch_norm(
                self.expansion * out_planes, shortcut_bn_type
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        shortcut_output = x
        if self.has_downsampler:
            shortcut_output = apply_bn_after_conv2d(
                x, self.shortcut_conv, self.shortcut_bn
            )

        out = apply_bn_after_conv2d(x, self.conv1, self.batch_norm1)
        out = self.act_fn(out)

        out = apply_bn_after_conv2d(out, self.conv2, self.batch_norm2)
        out = self.act_fn(out)

        out = apply_bn_after_conv2d(out, self.conv3, self.batch_norm3)

        if hasattr(self, "pre_shortcut_mask") and self.pre_shortcut_mask is not None:
            # This allows us to inherit the forward for the PurgedBasicBlock
            # We keep the shortcut fully dense, but some of the convolutions
            # might have dropped channels for `out`.

            # `pre_shortcut_mask` tells us which channels survived the
            # convolutional layers.
            padded_out = torch.zeros_like(shortcut_output)
            padded_out[:, self.pre_shortcut_mask, ...] = out
            out = padded_out + shortcut_output
        else:
            out = out + shortcut_output

        out = self.act_fn(out)

        return out


class BasicBlock(nn.Module):
    """
    Implements a Basic block for ResNet models. The code in this class is an adaptation
    of the official Pytorch-vision ResNet code to allow for L0Conv2d layers.
    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    """

    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        sparsity_type: int,
        use_bias: bool,
        act_fn_module: nn.Module,
        bn_type: str,
        base_l0_kwargs: dict,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        l0_conv_ix: List[str] = [],
        is_downsample_l0: bool = False,
    ):
        super(BasicBlock, self).__init__()

        self.sparsity_type = sparsity_type
        self.bn_type = bn_type

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.stride = stride
        self.act_fn = act_fn_module()

        # Both self.conv1 and self.shortcut_conv layers downsample the input when stride != 1
        self.conv1 = create_general_conv2d(
            in_planes=in_planes,
            out_planes=out_planes,
            kernel_size=(3, 3),
            stride=stride,
            padding=dilation,
            groups=groups,
            dilation=dilation,
            is_sparsifiable="conv1" in l0_conv_ix,
            base_l0_kwargs=base_l0_kwargs,
            use_bias=use_bias,
            sparsity_type=sparsity_type,
        )

        self.batch_norm1 = init_batch_norm(out_planes, self.bn_type)

        self.conv2 = create_general_conv2d(
            in_planes=out_planes,
            out_planes=out_planes,
            kernel_size=(3, 3),
            stride=1,
            padding=dilation,
            groups=groups,
            dilation=dilation,
            is_sparsifiable="conv2" in l0_conv_ix,
            base_l0_kwargs=base_l0_kwargs,
            use_bias=use_bias,
            sparsity_type=sparsity_type,
        )

        self.batch_norm2 = init_batch_norm(out_planes, self.bn_type)

        self.has_downsampler = False
        if stride != 1 or in_planes != out_planes:
            self.has_downsampler = True

            self.shortcut_conv = create_general_conv2d(
                in_planes=in_planes,
                out_planes=out_planes,
                kernel_size=(1, 1),
                stride=stride,
                padding=0,
                is_sparsifiable=is_downsample_l0,
                base_l0_kwargs=base_l0_kwargs,
                use_bias=use_bias,
                sparsity_type=sparsity_type,
            )

            # If shortcut_conv is not L0, then we use a regular BN layer
            shortcut_bn_type = self.bn_type if is_downsample_l0 else "regular"
            self.shortcut_bn = init_batch_norm(out_planes, shortcut_bn_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        shortcut_output = x
        if self.has_downsampler:
            shortcut_output = apply_bn_after_conv2d(
                x, self.shortcut_conv, self.shortcut_bn
            )

        out = apply_bn_after_conv2d(x, self.conv1, self.batch_norm1)
        out = self.act_fn(out)

        out = apply_bn_after_conv2d(out, self.conv2, self.batch_norm2)

        if hasattr(self, "pre_shortcut_mask") and self.pre_shortcut_mask is not None:
            # This allows us to inherit the forward for the PurgedBasicBlock
            # We keep the shortcut fully dense, but some of the convolutions
            # might have dropped channels for `out`.

            # `pre_shortcut_mask` tells us which channels survived the
            # convolutional layers.
            padded_out = torch.zeros_like(shortcut_output)
            padded_out[:, self.pre_shortcut_mask, ...] = out
            out = padded_out + shortcut_output
        else:
            out = out + shortcut_output

        out = self.act_fn(out)

        return out


class L0ResNet(BaseL0Model):
    """
    Adaptation of the official Pytorch-vision ResNet code to allow for L0Conv2d
    layers in ResNet models.
    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py


    Args:
        l0_conv_ix: Strings indicating which Conv2d layers of each
            block are turned into an L0Conv2d layer. If this is empty, the model
            is fully dense. Add "conv1" (resp. 2 or 3) for the convolutional
            layers and "shortcut_conv" for the shortcut layers.
        bn_type: Type of batch normalization to use. Can be "regular" or "L0".
            If "L0", then the *previous* layer must be an L0Conv2d layer.
    """

    def __init__(
        self,
        block: Union[Type[Bottleneck], Type[BasicBlock]],
        layers: List[int],
        use_bias: bool,
        sparsity_type: str,
        num_classes: int = 1000,
        input_shape: Tuple[int, int, int] = (3, 224, 224),
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        weight_decay: float = 5e-4,
        temperature: float = 2.0 / 3.0,
        act_fn_module: nn.Module = nn.ReLU,
        bn_type: str = "L0",
        l2_detach_gates: bool = False,
        l0_conv_ix: List[str] = ["conv1", "conv2", "conv3", "shortcut_conv"],
        conv1_kwargs: dict = None,
        do_initial_maxpool: bool = True,
        droprate_init: float = 0.3,
    ):

        super(L0ResNet, self).__init__(weight_decay=weight_decay)

        self.input_shape = input_shape
        # The use_bias flag is ignored for ResNets. We hard-code to bias=False
        self.use_bias = False and use_bias

        self.bn_type = bn_type

        self.in_planes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = create_dense_conv2d_layer(
            in_channels=3,
            out_channels=self.in_planes,
            use_bias=use_bias,
            **conv1_kwargs,
        )

        self.batch_norm1 = nn.BatchNorm2d(self.in_planes)
        self.act1 = act_fn_module()

        self.do_initial_maxpool = do_initial_maxpool
        if self.do_initial_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Gather configurations to be used in the L0 layers for all blocks
        base_l0_kwargs = {
            "droprate_init": droprate_init,
            "weight_decay": self.weight_decay,
            "temperature": temperature,
            "l2_detach_gates": l2_detach_gates,
        }

        # Gather configurations for blocks in all layers
        block_kwargs = {
            "use_bias": use_bias,
            "sparsity_type": sparsity_type,
            "act_fn_module": act_fn_module,
            "bn_type": bn_type,
            "l0_conv_ix": l0_conv_ix,
            "base_l0_kwargs": base_l0_kwargs,
            # Check if we are using sparsifiable shortcut connections
            "is_downsample_l0": "shortcut_conv" in l0_conv_ix,
        }

        self.layer1 = self._make_layer(
            block, 64, layers[0], stride=1, block_kwargs=block_kwargs
        )
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, block_kwargs=block_kwargs
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, block_kwargs=block_kwargs
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, block_kwargs=block_kwargs
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Hard-coded use_bias set to True for the final prediction layer
        self.fcout = create_dense_linear_layer(
            512 * block.expansion, num_classes, use_bias=True
        )

        self.pytorch_weight_initialization(zero_init_residual=zero_init_residual)

        self.layers_dict, self.params_dict = self.gather_layers_and_params()

    def pytorch_weight_initialization(self, zero_init_residual: bool):
        """
        Initialize the weights following the scheme suggested in the Pytorch ResNet
        implementation.
        """

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, L0Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, L0BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Bottleneck],
        out_planes: int,
        blocks: int,
        stride: int,
        block_kwargs: dict,
    ) -> nn.Sequential:

        layers = []
        layers.append(
            block(
                in_planes=self.in_planes,
                out_planes=out_planes,
                stride=stride,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                **block_kwargs,
            )
        )

        # Set the in_planes of the next layer to the out_planes of the current layer
        self.in_planes = out_planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    in_planes=self.in_planes,
                    out_planes=out_planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    **block_kwargs,
                )
            )

        return nn.Sequential(*layers)

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


def L0ResNet18(num_classes: int = 200, **kwargs: Any) -> L0ResNet:
    """Constructs a ResNet-18 model."""

    conv1_kwargs = {"kernel_size": (3, 3), "stride": 2, "padding": 1}

    return L0ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        conv1_kwargs=conv1_kwargs,
        do_initial_maxpool=False,
        **kwargs,
    )


def L0ResNet50(num_classes: int = 1000, **kwargs: Any) -> L0ResNet:
    """Constructs a ResNet-50 model."""

    conv1_kwargs = {"kernel_size": (7, 7), "stride": 2, "padding": 3}

    return L0ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        conv1_kwargs=conv1_kwargs,
        do_initial_maxpool=True,
        **kwargs,
    )
