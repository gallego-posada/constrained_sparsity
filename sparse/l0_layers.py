"""
Module for sparsifiable ("L0") layers. Includes fully-connected, convolutional
and batchnorm layers.

Based on: C. Louizos, M. Welling, and D. P. Kingma. Learning Sparse Neural
Networks through L0 Regularization. In ICLR, 2018.
Major code re-use from: https://github.com/AMLab-Amsterdam/L0_regularization
"""

import math
from functools import partial
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _NormBase
from torch.nn.modules.utils import _pair as pair

# LIMIT_A = gamma; LIMIT_B = zeta -- 'stretching' parameters (Sect 4, p7)
LIMIT_A, LIMIT_B, EPS = -0.1, 1.1, 1e-6


class BaseL0Layer(nn.Module):
    """Base class for L0 layers. This class is not intended to be used directly."""

    def __init__(
        self,
        sparsity_type: str,
        use_bias: bool = True,
        weight_decay: float = 0.0,
        l2_detach_gates: bool = False,
        droprate_init: float = 0.5,
        temperature: float = 2.0 / 3.0,
    ):

        if weight_decay < 0.0:
            ValueError(
                "expected non-negative weight_decay. Got {}".format(weight_decay)
            )
        if droprate_init <= 0.0 or droprate_init >= 1.0:
            ValueError("expected droprate_init in (0,1). Got {}".format(droprate_init))
        if temperature <= 0.0 or temperature >= 1.0:
            ValueError("expected temperature in (0,1). Got {}".format(temperature))

        super(BaseL0Layer, self).__init__()

        self.sparsity_type = sparsity_type

        self.use_bias = use_bias
        self.weight_decay = weight_decay
        self.l2_detach_gates = l2_detach_gates

        self.temperature = temperature
        self.droprate_init = droprate_init

        # Number of parameters associated with each gate. Set in construct_gates().
        self.group_size: int = None

        # Params for the distribution of the gates, instantiated in each subclass.
        self.weight_log_alpha: torch.Parameter = None
        self.bias_log_alpha: Optional[torch.Parameter] = None

    def construct_gates(self):

        if self.sparsity_type == "unstructured":
            # One gate per parameter. This includes the bias if applicable.
            self.group_size = 1
            self.dim_z = self.weight.shape
            if self.use_bias:
                self.bias_log_alpha = nn.Parameter(torch.Tensor(self.bias.shape))

        elif self.sparsity_type == "structured":
            if isinstance(self, L0Linear):
                # Do input neuron sparsity for fully-connected layers
                self.group_size = self.out_features
                self.dim_z = (self.in_features,)

            elif isinstance(self, L0Conv2d):
                # Do output channel sparsity for convolutional layers
                self.group_size = self.weight[0, ...].data.nelement()
                if self.bias is not None:
                    # One bias per output channel
                    self.group_size += 1

                self.dim_z = (self.out_channels,)

        self.weight_log_alpha = nn.Parameter(torch.Tensor(*self.dim_z))

    def init_parameters(self, wmode: str):
        """Initialize layer parameters, including the {weight, bias}_log_alpha parameters.
        Use wmode="fan_in" for fully-connected layers, and "fan_out" for convolutional
        layers."""

        # Initialize weight and bias parameters
        nn.init.kaiming_normal_(self.weight, mode=wmode)
        if self.use_bias:
            self.bias.data.normal_(0, 1e-2)

        # Weights are scaled by their gates when computing forwards (see get_params()).
        # Thus, their effective intialization is shrunk by a factor of 1 - droprate_init.
        # In order to keep the initialization of L0 and non-L0 layers consistent,
        # we counter this shrinkage by adjusting the initial weights by the same amount.
        initial_sigmoid = 1 - self.droprate_init
        self.weight.data = self.weight.data / initial_sigmoid

        if self.use_bias and len(self.weight.shape) > 2:
            # In convolutional layers, biases are also multiplied by gates
            # (see get_params()). Thus, we also adjust them
            self.bias.data = self.bias.data / initial_sigmoid

        # Initialize gate parameters
        gate_mean_init = math.log((1 - self.droprate_init) / self.droprate_init)
        self.weight_log_alpha.data.normal_(gate_mean_init, 1e-2)
        if hasattr(self, "bias_log_alpha") and self.bias_log_alpha is not None:
            self.bias_log_alpha.data.normal_(gate_mean_init, 1e-2)

    def clamp_parameters(self):
        """Clamp weight_log_alpha parameters for numerical stability."""
        self.weight_log_alpha.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

        if hasattr(self, "bias_log_alpha") and self.bias_log_alpha is not None:
            self.bias_log_alpha.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def concrete_cdf(self, x: float, log_alpha: torch.Tensor) -> torch.Tensor:
        """Implements the CDF of the 'stretched' concrete distribution."""
        # 'Stretch' input to (gamma, zeta) -- Eq 25 (appendix).
        x_stretched = (x - LIMIT_A) / (LIMIT_B - LIMIT_A)

        # Eq 24 (appendix)
        logits = math.log(x_stretched / (1 - x_stretched))
        pre_clamp = torch.sigmoid(logits * self.temperature - log_alpha)

        return pre_clamp.clamp(min=EPS, max=1 - EPS)

    @torch.no_grad()
    def get_eps_noise(
        self, size: Tuple[int], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Sample uniform noise for the concrete distribution. If do_sample is
        False, return tensor of 0.5 to evaluate gate medians in concrete_quantile."""
        # Inverse CDF sampling from the concrete distribution. Then clamp for gates
        eps_noise = torch.rand(size, dtype=dtype, device=device)
        # Transform to interval (EPS, 1-EPS)
        eps_noise = EPS + (1 - 2 * EPS) * eps_noise

        return eps_noise

    def concrete_quantile(self, x: float, log_alpha: torch.Tensor) -> torch.Tensor:
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete
        distribution, given a uniform sample x (Eq. 10)."""

        y = torch.sigmoid((torch.log(x / (1 - x)) + log_alpha) / self.temperature)
        return y * (LIMIT_B - LIMIT_A) + LIMIT_A

    def sample_gates(self):
        """Obtain samples for the stochastic gates. Active gates may have
        fractional values (not necessarily binary 0/1).
        Args:
            distribution. If `False`, use the median of the distribution.
        """
        dtype, device = self.weight.dtype, self.weight.device

        weight_noise = self.get_eps_noise(self.dim_z, dtype, device)
        # Sample fractional gates in [0,1] based on sampled epsilon
        weight_z = self.concrete_quantile(weight_noise, self.weight_log_alpha)
        weight_z = torch.clamp(weight_z, min=0, max=1)

        bias_z = None
        if self.use_bias and self.sparsity_type == "unstructured":
            bias_noise = self.get_eps_noise(self.bias.shape, dtype, device)
            # Sample fractional gates in [0,1] based on sampled epsilon
            bias_z = self.concrete_quantile(bias_noise, self.bias_log_alpha)
            bias_z = torch.clamp(bias_z, min=0, max=1)

        return weight_z, bias_z

    def evaluation_gates(self):
        """
        Obtain medians for the stochastic gates, used for forwards. Active gates
        may have fractional values (not necessarily binary 0/1).
        """

        weight_z = torch.sigmoid(self.weight_log_alpha / self.temperature)
        weight_z = weight_z * (LIMIT_B - LIMIT_A) + LIMIT_A
        weight_z = torch.clamp(weight_z, min=0, max=1)

        bias_z = None
        if self.use_bias and self.sparsity_type == "unstructured":
            bias_z = torch.sigmoid(self.bias_log_alpha / self.temperature)
            bias_z = bias_z * (LIMIT_B - LIMIT_A) + LIMIT_A
            bias_z = torch.clamp(bias_z, min=0, max=1)

        return weight_z, bias_z

    def get_params(self, do_sample: bool) -> Dict[str, Union[torch.Tensor, None]]:
        """Sample the parameters of the layer based on an inner sampling of the
        gates. This function gets called when performing a forward pass.
        Args:
            do_sample: If `True`, sample the gates from the hard concrete.
                Otherwise, use the median of the distribution.
        """
        if do_sample:
            weight_z, bias_z = self.sample_gates()
        else:
            weight_z, bias_z = self.evaluation_gates()

        if self.sparsity_type == "unstructured":
            bias = None if not self.use_bias else bias_z * self.bias
        else:
            if isinstance(self, L0Linear):
                weight_z = weight_z.view(1, -1)
                # No need to alter bias if using input sparsity
                bias = self.bias
            elif isinstance(self, L0Conv2d):
                weight_z = weight_z.view(*self.dim_z, 1, 1, 1)
                bias_z = weight_z.view(-1)
                bias = None if not self.use_bias else bias_z * self.bias

        # Either in the structured or unstructured case, weights are multiplied
        # by their gates. In the structured case, the gates are 1-dimensional
        # and broadcasted to the shape of the weight tensor.
        weight = weight_z * self.weight

        return {"weight": weight, "bias": bias, "weight_z": weight_z, "bias_z": bias_z}

    def l2_penalty(self) -> torch.Tensor:
        """L2 regularization term for the layer's parameters."""

        # Probability of inactivity per weight gate
        weight_q0 = self.concrete_cdf(0, self.weight_log_alpha)
        if self.l2_detach_gates:
            weight_q0.detach_()

        if self.sparsity_type == "unstructured":
            # No need to group since each group corresponds to one weight.
            w_group_frob = self.weight.pow(2)
        else:
            if isinstance(self, L0Linear):
                # We do not sum yet across gates dimension=0 (with size in_features)
                w_group_frob = torch.sum(self.weight.pow(2), dim=1)
            elif isinstance(self, L0Conv2d):
                # We do not sum yet across gates dimension=0 (with size out_channels)
                w_group_frob = torch.sum(self.weight.pow(2), dim=(1, 2, 3))

        weight_exp_frob_norm = torch.sum((1 - weight_q0) * w_group_frob)

        bias_exp_frob_norm = 0.0
        if self.use_bias:
            if self.sparsity_type == "unstructured":
                # Probability of inactivity for each bias gate
                bias_q0 = self.concrete_cdf(0, self.bias_log_alpha)
                if self.l2_detach_gates:
                    bias_q0.detach_()
                bias_exp_frob_norm = torch.sum((1 - bias_q0) * self.bias.pow(2))
            else:
                if isinstance(self, L0Linear):
                    # Under input sparsity, all entries of bias contribute to L2 norm.
                    bias_exp_frob_norm = torch.sum(self.bias.pow(2))
                elif isinstance(self, L0Conv2d):
                    # Only the biases corresponding to "active" output channels
                    # contribute.
                    bias_exp_frob_norm = torch.sum((1 - weight_q0) * self.bias.pow(2))

        return 0.5 * (weight_exp_frob_norm + bias_exp_frob_norm)

    def expected_active_gates(self) -> Tuple[Union[torch.Tensor, None]]:
        """Return the expected number of active gates in the layer."""

        # Inactive probability per gate
        weight_q0 = self.concrete_cdf(0, self.weight_log_alpha)
        # Expected number of active gates
        weight_active_gates = torch.sum(1 - weight_q0)

        if hasattr(self, "bias_log_alpha") and self.bias_log_alpha is not None:
            bias_q0 = self.concrete_cdf(0, self.bias_log_alpha)
            bias_active_gates = torch.sum(1 - bias_q0)
        else:
            bias_active_gates = None

        return weight_active_gates, bias_active_gates

    def expected_active_parameters(self) -> torch.Tensor:
        """Return the expected number of active parameters in the layer."""

        weight_active_gates, bias_active_gates = self.expected_active_gates()

        # If using unstructured sparsity, group_size is 1
        weight_contrib = weight_active_gates * self.group_size

        if not self.use_bias:
            # If the layer does not use_bias, there is no contribution from the
            # bias to the active parameters.
            return weight_contrib

        elif self.sparsity_type == "structured":
            if isinstance(self, L0Linear):
                # Bias is kept fully dense, but it is not sparsifiable
                bias_contrib = 0.0
            elif isinstance(self, L0Conv2d):
                # Biases associated with active output channels DO contribute.
                # Nonetheless, we have setup self.group_size to account for the
                # biases in this case.
                bias_contrib = 0.0

            return weight_contrib + bias_contrib

        elif self.sparsity_type == "unstructured":
            assert bias_active_gates is not None
            # If using unstructured sparsity and self.use_bias, biases are also
            # sparsifiable. They have their own gates which contribute separately
            # to the active parameters.
            bias_contrib = bias_active_gates

            return weight_contrib + bias_contrib

    def regularization(self) -> Dict[str, Union[int, float]]:
        """Return the regularization terms for the layer's parameters."""

        # If weight_decay is 0, we skip the computation of the L2 norm
        exp_l2 = self.l2_penalty() if self.weight_decay != 0 else 0

        # Compute expected L0 norm as fraction of active parameters over total
        # sparsifiable parameters in this layer.
        nsp = self.num_sparsifiable_params
        exp_l0 = self.expected_active_parameters() / nsp

        return {"exp_l0": exp_l0, "num_params": nsp, "l2_reg": exp_l2}


class L0Linear(BaseL0Layer):
    """Implementation of a fully connected layer with L0 regularization for its
    input units."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool,
        sparsity_type: str,
        base_l0_kwargs: dict = {},
    ):

        # Call BaseL0Layer constructor.
        super(L0Linear, self).__init__(
            sparsity_type=sparsity_type, use_bias=use_bias, **base_l0_kwargs
        )

        self.out_features = out_features
        self.in_features = in_features

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        # Counts total number of sparsifiable parameters. This is not a count
        # of active params.
        self.num_sparsifiable_params = self.weight.data.nelement()

        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            if self.sparsity_type == "unstructured":
                # Biases are also sparsifiable under unstructured sparsity.
                self.num_sparsifiable_params += self.bias.data.nelement()
        else:
            self.bias = None

        self.construct_gates()
        self.init_parameters(wmode="fan_out")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer according to a sampled weight matrix.
        Can be used for inference, but consider using a PurgedModel from
        sparse.purged_models to avoid the overhead of using gates.
        """
        params_dict = self.get_params(do_sample=self.training)
        return F.linear(input, params_dict["weight"], params_dict["bias"])

    def __repr__(self):
        s = (
            "{name}({in_features} -> {out_features}, droprate_init={droprate_init}, "
            "temperature={temperature}, weight_decay={weight_decay}"
        )
        if not self.use_bias:
            s += ", bias=False"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)


class L0Conv2d(BaseL0Layer):
    """Implementation of a convolutional layer with L0 regularization for its
    output channels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_bias: bool,
        sparsity_type: str,
        kernel_size: Tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        base_l0_kwargs: dict = {},
    ):

        super(L0Conv2d, self).__init__(
            sparsity_type=sparsity_type, use_bias=use_bias, **base_l0_kwargs
        )

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.input_shape = None  # set during the first forward pass.

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Convolution-related parameters.
        self.kernel_size = pair(kernel_size)
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.dilation = pair(dilation)
        self.groups = groups
        self.conv_kwargs = {
            _: getattr(self, _) for _ in ["stride", "padding", "dilation", "groups"]
        }

        # Weight dimensions: (out_channels, in_channels, kernel_size, kernel_size).
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )

        # Number of params in weight matrix.
        self.num_sparsifiable_params = self.weight.data.nelement()
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            # Counting biases as they *are* 'sparsifiable' params under output
            # sparsity and under unstructured sparsity.
            self.num_sparsifiable_params += self.bias.data.nelement()
        else:
            self.bias = None

        # Pre-populate parameters for call to torch conv2d
        self.conv2d = partial(nn.functional.conv2d, **self.conv_kwargs)

        self.construct_gates()
        self.init_parameters(wmode="fan_in")

    def forward(self, input: torch.Tensor, return_mask: bool = False) -> torch.Tensor:
        """Forward pass of the layer according to a sampled weight and bias.
        Can be used for inference, but consider using a PurgedModel from
        sparse.purged_models to avoid the overhead of using gates."""
        if self.input_shape is None:
            self.input_shape = input.size()

        params_dict = self.get_params(do_sample=self.training)
        weight, bias, gates = [params_dict[_] for _ in ["weight", "bias", "weight_z"]]

        output = self.conv2d(input, weight, bias)

        if return_mask:
            if self.sparsity_type == "unstructured":
                # Unstructured sparsity does not perform purging. Therefore,
                # all channels are kept and we return a mask of True for every
                # output feature map.
                return output, torch.ones(output.shape[1]).bool()
            else:
                mask = torch.flatten(gates).bool()
            return output, mask

        return output

    def __repr__(self):
        s = (
            "{name}({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, "
            "droprate_init={droprate_init}, temperature={temperature}, weight_decay={weight_decay}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if not self.use_bias:
            s += ", bias=False"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)


class L0BatchNorm2d(_NormBase):
    """Implements a batch normalization layer with L0 sparsity."""

    def init(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
    ):

        factory_kwargs = {"device": device, "dtype": dtype}
        # _NormBase init handles the assignment of these attributes.
        super(L0BatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))

    def forward(
        self, input: torch.Tensor, gate_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """"""

        if gate_mask is None:
            gate_mask = torch.ones(self.num_features, device=input.device, dtype=bool)

        if all(gate_mask == 0):
            # If all gates are inactive, just return zeros.
            return torch.zeros_like(input, device=input.device)

        # --------------------------------------------------
        # This section is copied and unmodified from _BatchNorm.forward
        # --------------------------------------------------

        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # Pytorch TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    Warning(
                        "num_batches_tracked differs between out_channels, \
                        assuming all channels of all batches were tracked for EA_factor"
                    )
                    # TODO: could handle EA_factor as vector, but difficult for F.batch_norm call
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # ----------------- end<section> ------------------

        # --------------------------------------------------
        # Below, code was modified by to use the gate mask
        # --------------------------------------------------

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """

        if not self.training or self.track_running_stats:
            _running_mean = self.running_mean[gate_mask]
            _running_var = self.running_var[gate_mask]
        else:
            # If buffers are not to be tracked, ensure that they won't be updated
            _running_mean = None
            _running_var = None

        _input = input[:, gate_mask, ...]  # ignore sparse input channels
        _weight = self.weight[gate_mask, ...]
        _bias = self.bias[gate_mask]

        # Do BN on outputs from active units
        masked_output = F.batch_norm(
            _input,
            _running_mean,
            _running_var,
            _weight,
            _bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

        # _running_mean and _running_var were updated internally by F.batch_norm
        if self.training and self.track_running_stats:
            self.running_mean[gate_mask] = _running_mean
            self.running_var[gate_mask] = _running_var

        # Identity mapping for inactive units: 0s in forward but with a gradient
        output = input
        output[:, gate_mask, ...] = masked_output

        return output

    def get_nap(self, num_active_groups: int) -> int:
        """Get the number of active parameters for a given number of active groups."""

        # TODO: use num_active_groups from *previous layer* to determine
        # actual number of active parameters in the BN layer.
        # Currently, we naively count all params in weight and bias tensors.
        #
        # Results in paper fo not count parameters of BN layers towards l0_model.

        pass
