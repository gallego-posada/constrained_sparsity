from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

import utils

from .l0_layers import L0Conv2d, L0Linear
from .models import L0MLP, L0LeNet5


@dataclass
class PrunedLayerInfo:
    """Stores the main elements of a base Pytorch layer. This is used to
    construct a pruned layer from an L0 layer."""

    weight: torch.Tensor
    bias: Optional[torch.Tensor] = None
    kwargs: Optional[dict] = None

    def as_tuple(self) -> Tuple:
        return self.weight, self.bias, self.kwargs


def compute_layer_nsp(layer: Union[nn.Linear, nn.Conv2d]) -> int:
    """Count number of parameters in layer in terms of weight and bias."""
    num_par = layer.weight.numel()
    if layer.bias is not None:
        num_par += layer.bias.numel()
    return num_par


def purge_linear_modules(
    linear_layers: List[L0Linear],
    input_mask: Optional[torch.Tensor] = None,
) -> Tuple[List[PrunedLayerInfo], torch.Tensor]:
    """
    Purges a list of L0Linear modules. Given the state of a layer,
    some gates are deemed to be 0. Their corresponding parameters are removed
    from the weight term to produce pruned parameters.

    Returns a list of PrunedLayerInfo objects used for constructing pruned layers,
    and a mask with the size of the input to the first provided layer. This mask
    is used to indicate which inputs ought to be discarded before this sequence
    of linear layers..
    """

    input_mask_device = input_mask.device if input_mask is not None else None

    # Get a mask indicating active inputs for each linear layer.
    layer_gates = []
    for layer in linear_layers:
        assert layer.sparsity_type == "structured"

        # Ignore the bias gates as this function is only called for structured sparsity.
        gates, _ = layer.evaluation_gates()
        layer_gates.append(torch.flatten(gates).bool().to(input_mask_device))

    in_gates = layer_gates[0]  # for slicing inputs

    if input_mask is not None:
        # Account for the output sparsity of conv layer before a linear layer with input sparsity
        in_gates *= input_mask
        # Slicing output of convs and then slicing input of linear
        in_gates = in_gates[input_mask]

    purged_modules = []  # for storing pruned parameters of each layer

    for i, layer in enumerate(linear_layers[:-1]):

        my_gates = layer_gates[i]  # mask on input
        next_gates = layer_gates[i + 1]  # mask on output, the input to the next layer.

        # Get the test-time params, already multiplied by gates.
        dense_params = layer.get_params(do_sample=False)

        # Slice
        _weight = dense_params["weight"][:, my_gates][next_gates, :]
        _bias = (
            dense_params["bias"][next_gates]
            if dense_params["bias"] is not None
            else None
        )
        purged_modules.append(PrunedLayerInfo(_weight, _bias, {}))

    # Prediction layer. No slicing wrt its output.
    my_gates = layer_gates[-1]
    dense_params = linear_layers[-1].get_params(do_sample=False)

    _weight = dense_params["weight"][:, my_gates]
    _bias = dense_params["bias"] if dense_params["bias"] is not None else None
    purged_modules.append(PrunedLayerInfo(_weight, _bias, {}))

    return purged_modules, in_gates


def purge_conv_modules(
    conv_layers: List[L0Conv2d],
) -> Tuple[List[PrunedLayerInfo], torch.Tensor]:
    """
    Purges a list of L0Conv2d modules. Given the state of a layer,
    some gates are deemed to be 0. Their corresponding parameters are removed
    from the weight/bias terms to produce pruned parameters.

    Returns a list of PrunedLayerInfo objects, used for constructing pruned layers.
    Also returns a list for masking, indicating which outputs of the last provided
    layer were pruned, and should not be expected by subsequent linear layers.
    """

    # Store params like padding and stride for each layer.
    conv_kwargs_list = [_.conv_kwargs for _ in conv_layers]

    # Get a mask indicating active output neurons for each layer.
    layer_gates = []
    for layer in conv_layers:
        assert layer.sparsity_type == "structured"

        # Ignore the bias gates as this function is only called for structured sparsity.
        gates, _ = layer.evaluation_gates()
        layer_gates.append(torch.flatten(gates).bool())

    purged_modules = []

    # First layer. Does not have preceeding layers with sparse outputs.
    my_gates = layer_gates[0]
    dense_params = conv_layers[0].get_params(do_sample=False)

    # Slice parameters
    _weight = dense_params["weight"][my_gates, ...]
    _bias = dense_params["bias"][my_gates] if dense_params["bias"] is not None else None
    purged_modules.append(PrunedLayerInfo(_weight, _bias, conv_kwargs_list[0]))

    iterable = zip(conv_layers[1:], conv_kwargs_list[1:])
    for i, (layer, conv_kwargs) in enumerate(iterable, start=1):

        # output gates of previous layer, so input gates of current layer
        prev_gates = layer_gates[i - 1]
        my_gates = layer_gates[i]

        dense_params = layer.get_params(do_sample=False)

        _weight = dense_params["weight"][my_gates, ...][:, prev_gates, ...]
        _bias = (
            dense_params["bias"][my_gates] if dense_params["bias"] is not None else None
        )
        purged_modules.append(PrunedLayerInfo(_weight, _bias, conv_kwargs))

    return purged_modules, layer_gates


# --------------- Create purged layers --------------


def create_purged_conv(layer_info: PrunedLayerInfo) -> nn.Conv2d:
    """Creates an nn.Conv2d module from given PrunedLayerInfo.
    The latter refer to stride, padding, dilation, groups."""

    weight, bias, conv_kwargs = layer_info.as_tuple()

    use_bias = not (bias is None)
    out_channels, in_channels, kh, kw = weight.shape

    conv_layer = nn.Conv2d(
        in_channels, out_channels, (kh, kw), bias=use_bias, **conv_kwargs
    )

    conv_layer.weight.data = weight
    if use_bias:
        conv_layer.bias.data = bias

    return conv_layer


def create_purged_linear(layer_info: PrunedLayerInfo) -> nn.Linear:
    """Creates an nn.Linear module from given PrunedLayerInfo."""

    weight, bias, _ = layer_info.as_tuple()

    use_bias = not (bias is None)

    out_feats, in_feats = weight.shape
    linear_layer = nn.Linear(
        in_features=in_feats, out_features=out_feats, bias=use_bias
    )

    linear_layer.weight.data = weight
    if use_bias:
        linear_layer.bias.data = bias

    return linear_layer


def create_purged_batch_norm(layer, mask=None):
    if isinstance(layer, nn.Identity):
        return nn.Identity()
    else:
        if mask is None:
            mask = torch.ones(layer.num_features).bool()

        new_batch_norm = nn.BatchNorm2d(int(sum(mask).item()))

        new_batch_norm.weight.data = layer.weight.data[mask]
        new_batch_norm.bias.data = layer.bias.data[mask]
        new_batch_norm.running_mean = layer.running_mean[mask]
        new_batch_norm.running_var = layer.running_var[mask]
        new_batch_norm.num_batches_tracked = layer.num_batches_tracked

    return new_batch_norm


# --------------- Create purged L0 + LeNet models --------------
class PurgedModel(nn.Module):
    """Implements a model composed exclusively of base Pytorch layers,
    obtained by pruning those of a reference BaseL0Model."""

    def __init__(self, model: Union[L0MLP, L0LeNet5]):

        super().__init__()

        act_fn_module = model.act_fn_module
        self.input_shape = model.input_shape

        self.layer_densities = OrderedDict()

        # ----- Convolutional layers ------
        conv_modules, out_conv_mask = [], None
        if hasattr(model, "all_convs"):
            if model.sparsity_type == "unstructured":
                # No actual pruning happening here. Just create a list of modules
                # with the evaluation parameters.
                conv_modules = []
                for module in model.all_convs:
                    pruned_params = module.get_params(do_sample=False)
                    conv_modules.append(
                        PrunedLayerInfo(
                            pruned_params["weight"],
                            pruned_params["bias"],
                            module.conv_kwargs,
                        )
                    )
            else:
                foo = purge_conv_modules(model.all_convs)
                conv_modules, layer_gates = foo
                out_conv_mask = layer_gates[-1]

                # We need to update the layer densities after purging.
                self.update_layer_densities(model.all_convs, conv_modules, "conv")

        conv_layers = []
        for conv_module in conv_modules:
            new_conv = create_purged_conv(conv_module)
            conv_layers += [new_conv, nn.MaxPool2d(2), act_fn_module()]

        self.conv_seq = None
        conv_flat_mask = None
        if len(conv_modules) > 0:
            self.conv_seq = nn.Sequential(*conv_layers)

            if model.sparsity_type == "structured":
                # out_conv_mask has dim out_channels. Masking after flattening an
                # output means masking along all activation maps of each channel.

                # Get height and width of the activation maps after conv_layers.
                _, act_map_h, act_map_w = utils.basic_utils.get_final_features(
                    model.input_shape, nn.Sequential(*conv_layers)
                )
                # Expand the mask along the height and width dimensions.
                out_conv_mask = out_conv_mask.view(-1, 1, 1).repeat(
                    1, act_map_h, act_map_w
                )
                conv_flat_mask = torch.flatten(out_conv_mask)

        # ----- Linear layers ------
        linear_modules, in_lin_mask = [], None
        if hasattr(model, "all_linear"):
            if model.sparsity_type == "unstructured":
                for module in model.all_linear:
                    pruned_params = module.get_params(do_sample=False)
                    linear_modules.append(
                        PrunedLayerInfo(pruned_params["weight"], pruned_params["bias"])
                    )
                # Keep all outputs of first layer
                in_lin_mask = torch.ones(model.all_linear[0].in_features).bool()
            else:
                linear_modules, in_lin_mask = purge_linear_modules(
                    model.all_linear, input_mask=conv_flat_mask
                )
                # We need to update the layer densities after purging.
                self.update_layer_densities(model.all_linear, linear_modules, "fc")

        # Specifies features that are kept after convs and before fc layers.
        self.in_mask = in_lin_mask

        linear_layers = []
        for layer_id, linear_module in enumerate(linear_modules):
            new_linear = create_purged_linear(linear_module)
            linear_layers.append(new_linear)

            # Use activation function for all but the final layer
            if layer_id != len(linear_modules) - 1:
                linear_layers.append(act_fn_module())

        self.linear_seq = nn.Sequential(*linear_layers)

    def update_layer_densities(self, orig_list, purged_list, path_prefix):

        for mod_id, (orig_module, purged_module) in enumerate(
            zip(orig_list, purged_list)
        ):

            layer_path = f"{path_prefix}{mod_id}"
            orig_nsp = orig_module.num_sparsifiable_params

            purged_nsp = purged_module.weight.data.nelement()
            if hasattr(purged_module, "bias") and purged_module.bias is not None:
                if (
                    isinstance(orig_module, L0Linear)
                    and orig_module.sparsity_type == "structured"
                ):
                    # For input sparsity L0Linear, the bias is not sparsifiable.
                    # It is not counted in orig_nsp and so we do not counted in
                    # purged_nsp.
                    pass
                else:
                    purged_nsp += purged_module.bias.data.nelement()

            layer_density = (1.0 * purged_nsp) / orig_nsp
            self.layer_densities[layer_path] = layer_density

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward. First through convolutional layers if the model has any.
        Their output is then flattened and masked to be passed through linear
        layers."""
        out = input if self.conv_seq is None else self.conv_seq(input)
        out = out.view(input.shape[0], -1)

        if self.in_mask is not None:
            out = out[:, self.in_mask]

        out = self.linear_seq.forward(out)

        return out
