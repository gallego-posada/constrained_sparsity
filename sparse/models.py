"""
ConvNets, MLPs and Logistic Regressions leveraging L0-regularized layers.
Based on: C. Louizos, M. Welling, and D. P. Kingma. Learning Sparse Neural
Networks through L0 Regularization. In ICLR, 2018.
Major code re-use from: https://github.com/AMLab-Amsterdam/L0_regularization
"""
import dataclasses
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn

import utils

from .l0_layers import L0BatchNorm2d, L0Conv2d, L0Linear


@dataclasses.dataclass
class ModelRegStats:
    """
    A class to store the regularization statistics for a given model.
    """

    l0_layer: torch.Tensor
    l0_model: torch.Tensor
    l0_full: torch.Tensor
    l2_model: torch.Tensor


# ------------------------------------------------------------------------------
# Helper functions for creating "fully dense" Linear and Conv2D layers
# ------------------------------------------------------------------------------


def create_dense_conv2d_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int],
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    use_bias: bool = True,
) -> nn.Conv2d:
    """Create a vanilla Pytorch Conv2d and initialize its parameters."""

    conv2d = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=use_bias,
    )

    torch.nn.init.kaiming_normal_(conv2d.weight, mode="fan_out")
    if conv2d.bias is not None:
        conv2d.bias.data.normal_(0, 1e-2)

    return conv2d


def create_dense_linear_layer(
    in_features: int, out_features: int, use_bias: bool = True
) -> nn.Linear:
    """Create a vanilla Pytorch Linear layer and initialize its parameters."""

    linear = nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=use_bias,
    )

    torch.nn.init.kaiming_normal_(linear.weight, mode="fan_in")
    if linear.bias is not None:
        linear.bias.data.normal_(0, 1e-2)

    return linear


# ------------------------------------------------------------------------------


def layer_regularization(layer: nn.Module) -> dict:
    """
    Computes the regularization statistics (L0 and L2) for a given layer.
    """

    # List of supported modules without a custom regularization function
    no_custom_reg_layers = [nn.Conv2d, nn.Linear, nn.BatchNorm2d, L0BatchNorm2d]

    if hasattr(layer, "regularization") and callable(layer.regularization):
        return layer.regularization()
    else:
        # TODO: Should we do weight decay on BN layers?
        if any([isinstance(layer, _) for _ in no_custom_reg_layers]):

            # Compute L2 norm of layer parameters
            logpw = torch.sum(layer.weight.pow(2))
            logpb = torch.sum(layer.bias.pow(2)) if layer.bias is not None else 0.0
            l2_reg = 0.5 * (logpw + logpb)

            # Compute total number of trainable parameters in this layer
            num_params = layer.weight.numel()
            if layer.bias is not None:
                num_params += layer.bias.numel()

            return {"num_params": num_params, "l2_reg": l2_reg}

        else:
            raise RuntimeError("Layer type not supported to compute regularization.")


class BaseL0Model(nn.Module):
    """Base class for L0 models. Implements methods for regularization calculation
    which may be shared across different models with L0 layers"""

    def __init__(self, weight_decay: float = 0.0):
        self.weight_decay = weight_decay
        super().__init__()

    def gather_layers_and_params(
        self,
    ) -> Tuple[Dict[str, nn.Module], Dict[str, nn.Parameter]]:
        """
        Gather all l0-sparse and dense (non L0) layers and parameters of the model.
        Args:
            model: The model to gather layers and parameters from.
        Returns:
            layers_dict: All model layers grouped by "l0", "dense" and "bn".
            params_dict: All parameters of the model grouped by "net" (weights
                and biases), "gates" and "bn".
        """

        layers_dict: Dict[str, List] = {"l0": [], "dense": [], "bn": []}
        params_dict: Dict[str, List] = {"net": [], "gates": []}

        for m in self.modules():

            # Gather weights and biases for this module
            # Make sure this is not an activation/pooling layer
            if hasattr(m, "weight"):
                params_dict["net"].append(m.weight)

            if hasattr(m, "bias") and (m.bias is not None):
                params_dict["net"].append(m.bias)

            if isinstance(m, (L0Conv2d, L0Linear)):
                # Gather sparsifiable module and parameters for the gates
                layers_dict["l0"].append(m)
                params_dict["gates"].append(m.weight_log_alpha)
                if hasattr(m, "bias_log_alpha") and (m.bias_log_alpha is not None):
                    params_dict["gates"].append(m.bias_log_alpha)

            elif isinstance(m, (nn.Linear, nn.Conv2d)):
                # Gather non-sparsifiable module
                layers_dict["dense"].append(m)

            elif isinstance(m, (nn.BatchNorm2d, L0BatchNorm2d)):
                # Gather batch norm parameters
                layers_dict["bn"].append(m)

        return layers_dict, params_dict

    def regularization(self) -> ModelRegStats:
        """
        Compute surrogate L0 reg for each L0 layer in the model, and the
        model's global L0 regularization.
        Also, compute the model's L2 reg term.
        """

        # Extract all groups of layers for readability
        foo = [self.layers_dict[_] for _ in ["l0", "dense", "bn"]]
        l0_layers, dense_layers, bn_layers = foo

        # Compute (L0 and L2) regularization for each sparsifiable L0-layer
        l0_reg_dicts = list(map(layer_regularization, l0_layers))

        # Gather normalized L0 norm for each sparsifiable layer
        l0_per_layer = torch.stack([_["exp_l0"] for _ in l0_reg_dicts])

        # Compute L0 norm *only* for the sparsifiable layers
        nap_per_l0_layer = [_["exp_l0"] * _["num_params"] for _ in l0_reg_dicts]
        nsp_per_l0_layer = [_["num_params"] for _ in l0_reg_dicts]
        l0_model = sum(nap_per_l0_layer) / sum(nsp_per_l0_layer)

        dense_reg_dicts = list(map(layer_regularization, dense_layers))
        bn_reg_dicts = list(map(layer_regularization, bn_layers))

        # Expected L2 norm for the model
        l2_model = sum(
            [_["l2_reg"] for _ in l0_reg_dicts + dense_reg_dicts + bn_reg_dicts]
        )
        # Up until now, we had collected all L2 norms without the weight decay factor
        l2_model = self.weight_decay * l2_model

        # Keep track of the normalized L0 for the *whole* model. This includes
        # non sparsifiable layers. This is useful for comparing to other papers
        # which choose different layers to sparsify.
        non_l0_nsp = sum([_["num_params"] for _ in dense_reg_dicts + bn_reg_dicts])
        l0_full_active = sum(nap_per_l0_layer) + non_l0_nsp
        l0_full_total = sum(nsp_per_l0_layer) + non_l0_nsp
        l0_full_model = l0_full_active / l0_full_total

        return ModelRegStats(
            l0_layer=l0_per_layer,
            l0_model=l0_model,
            l0_full=l0_full_model,
            l2_model=l2_model,
        )


class L0MLP(BaseL0Model):
    """MLP with L0 regularization. If no hidden layers are specified, constructs
    a logistic regression model. It is built exclusively with L0Linear layers and
    activation functions."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        sparsity_type: str = "unstructured",
        layer_dims: Optional[Tuple[int, ...]] = (300, 100),
        weight_decay: float = 0.0,
        l2_detach_gates: bool = False,
        temperature: float = 2.0 / 3.0,
        droprate_init: float = 0.5,
        use_bias: bool = True,
        act_fn_module: Type[nn.ReLU] = nn.ReLU,
    ):
        super().__init__(weight_decay=weight_decay)

        self.input_shape = (1, input_dim)
        self.input_dim = input_dim
        self.layer_dims = layer_dims if layer_dims is not None else []
        self.num_classes = num_classes

        self.sparsity_type = sparsity_type
        self.use_bias = use_bias
        self.act_fn_module = act_fn_module

        # --------------------- Construct Linear Layers ---------------------
        l0linear_kwargs = {
            "weight_decay": weight_decay,
            "l2_detach_gates": l2_detach_gates,
            "temperature": temperature,
        }

        layers = []
        for i, dimh in enumerate(self.layer_dims):
            inp_dim = self.input_dim if i == 0 else self.layer_dims[i - 1]
            # Use different (low) sparsity for input layer
            droprate_init_ = 0.2 if i == 0 else droprate_init
            l0linear_kwargs["droprate_init"] = droprate_init_

            layer = L0Linear(inp_dim, dimh, use_bias, sparsity_type, l0linear_kwargs)
            layers += [layer, self.act_fn_module()]

        # Input dim for output layer is different for Logistic regression.
        last_hidden = layer_dims[-1] if len(layer_dims) > 0 else input_dim

        # Use provided droprate_init for output layer
        l0linear_kwargs["droprate_init"] = droprate_init
        layers.append(
            L0Linear(last_hidden, num_classes, use_bias, sparsity_type, l0linear_kwargs)
        )

        self.fcs = nn.Sequential(*layers)  # sequential object for forward.

        self.layers_dict, self.params_dict = self.gather_layers_and_params()
        # We need to set all_linear for purging functions
        self.all_linear = self.layers_dict["l0"]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.view(-1, self.input_dim)
        return self.fcs.forward(input)


class L0LeNet5(BaseL0Model):
    """Convnet with L0 regularization. It has two L0Conv2d layers, each followed
    by an activation and a maxpooling. The output of these layers is fed to two
    L0Linear fully connected layers.
    All the Conv and Linear layers of this model are sparsifiable.
    """

    def __init__(
        self,
        num_classes: int,
        sparsity_type: str = "unstructured",
        input_shape: Tuple[int, int, int] = (1, 28, 28),
        conv_dims: Tuple[int, int] = (20, 50),
        kernel_sizes: Tuple[int, int] = (5, 5),
        fc_dims: int = 500,
        weight_decay: float = 0.0,
        temperature: float = 2.0 / 3.0,
        use_bias: bool = True,
        act_fn_module: Type[nn.ReLU] = nn.ReLU,
        l2_detach_gates: bool = False,
        droprate_init: float = 0.5,
    ):

        assert len(conv_dims) == 2

        super().__init__(weight_decay=weight_decay)

        self.input_shape = input_shape
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.num_classes = num_classes

        self.sparsity_type = sparsity_type
        self.use_bias = use_bias
        self.act_fn_module = act_fn_module

        # --------------------- Construct Conv Layers ---------------------
        self.kernel_sizes = kernel_sizes
        l0conv_kwargs = {
            "droprate_init": droprate_init,
            "weight_decay": self.weight_decay,
            "temperature": temperature,
            "l2_detach_gates": l2_detach_gates,
        }

        conv_kwargs = {"stride": 1, "padding": 0, "dilation": 1, "groups": 1}

        # L0Conv2d parameters: in_channels, out_channels, kernel_size
        convs = [
            L0Conv2d(
                input_shape[0],
                conv_dims[0],
                use_bias,
                sparsity_type,
                (self.kernel_sizes[0], self.kernel_sizes[0]),
                base_l0_kwargs=l0conv_kwargs,
                **conv_kwargs
            ),
            self.act_fn_module(),
            nn.MaxPool2d(2),
            L0Conv2d(
                conv_dims[0],
                conv_dims[1],
                use_bias,
                sparsity_type,
                (self.kernel_sizes[1], self.kernel_sizes[1]),
                base_l0_kwargs=l0conv_kwargs,
                **conv_kwargs
            ),
            self.act_fn_module(),
            nn.MaxPool2d(2),
        ]

        self.convs = nn.Sequential(*convs)
        if torch.cuda.is_available():
            # Move convs to gpu to ensure tensors on same device for flat_fts calculation
            self.convs = self.convs.cuda()
        # Calculates feature dimensions coming out of convolutional layers
        self.flat_fts = int(
            np.prod(utils.basic_utils.get_final_features(input_shape, self.convs))
        )

        # --------------------- Construct Linear Layers ---------------------

        l0linear_kwargs = {
            "droprate_init": droprate_init,
            "weight_decay": self.weight_decay,
            "temperature": temperature,
            "l2_detach_gates": l2_detach_gates,
        }

        # L0Linear parameters: in_features, out_features
        fcs = [
            L0Linear(
                self.flat_fts, self.fc_dims, use_bias, sparsity_type, l0linear_kwargs
            ),
            self.act_fn_module(),
            L0Linear(
                self.fc_dims, num_classes, use_bias, sparsity_type, l0linear_kwargs
            ),
        ]

        self.fcs = nn.Sequential(*fcs)  # sequential object for forward

        self.layers_dict, self.params_dict = self.gather_layers_and_params()

        # The first two layers of LeNet5s are L0Conv2d layers and the following
        # two are L0Linear layers.
        self.all_convs = self.layers_dict["l0"][:2]
        self.all_linear = self.layers_dict["l0"][2:]

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        out = self.convs(input)
        out = out.view(out.size(0), -1)
        return self.fcs.forward(out)
