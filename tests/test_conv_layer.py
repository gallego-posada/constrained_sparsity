import pytest
import torch

from sparse import L0Conv2d


@pytest.fixture(params=["structured", "unstructured"])
def sparsity_type(request):
    return request.param


@pytest.fixture(params=[True, False])
def use_bias(request):
    return request.param


@pytest.fixture(
    params=[
        {"in_channels": 3, "out_channels": 20, "kernel_size": (5, 5), "stride": 1},
        {"in_channels": 10, "out_channels": 15, "kernel_size": (3, 3), "stride": 2},
    ]
)
def layer_kwargs(request):
    return request.param


@pytest.fixture()
def layer(layer_kwargs, use_bias, sparsity_type):
    conv_layer = L0Conv2d(
        use_bias=use_bias, sparsity_type=sparsity_type, **layer_kwargs
    )
    if torch.cuda.is_available():
        conv_layer = conv_layer.cuda()
    return conv_layer


def test_init(layer, sparsity_type, use_bias):
    """Sanity checks on the shape of tensors inside L0Conv2d"""

    # L0Conv2d weight dims match vanilla Pytorch's conv dims.
    out_channels, in_channels, kh, kw = layer.weight.shape

    if layer.use_bias:
        # out_channels should match the shape of the bias term
        assert layer.bias.shape == (out_channels,)
        if sparsity_type == "unstructured":
            assert layer.bias_log_alpha.shape == (out_channels,)

    if sparsity_type == "structured":
        # Conv layers have output feature map sparsity. log_alpha must match out_channels
        assert layer.weight_log_alpha.shape == (out_channels,)
    else:
        assert layer.weight_log_alpha.shape == (out_channels, in_channels, kh, kw)


def test_sparse_forward(layer, sparsity_type, use_bias):
    """Checks that forward passes are consistent when the layer is fully sparse"""

    # Make sure that all of the gates are very likely to be turned off
    layer.weight_log_alpha.data.fill_(-10.0)

    if sparsity_type == "unstructured" and use_bias:
        layer.bias_log_alpha.data.fill_(-10.0)

    # We consider eval mode as the medians of gates are selected deterministically
    # and for log_alpha < -2.56, they are 0. We therefore guarantee full sparsity
    layer.eval()

    in_channels = layer.weight.shape[1]
    t = torch.randn(100, in_channels, 28, 28)
    if torch.cuda.is_available():
        t = t.cuda()

    out = layer(t)
    # L0Conv2d layers employ output sparsity, so the output should be of zeros
    assert torch.allclose(out, torch.zeros_like(out))


def test_forward(layer, sparsity_type, use_bias):
    """Checks that forward passes apply output sparsity correctly on L0Conv2d layers"""

    # Set all gates to be on, except the first one. The first feature map of outputs
    # must then be of zeros, regardless of the input.
    layer.weight_log_alpha.data.fill_(10.0)
    layer.weight_log_alpha.data[0] = -10.0

    if sparsity_type == "unstructured" and use_bias:
        layer.bias_log_alpha.data.fill_(10.0)
        # biases have independent gates. Must garantee their sparsity as well
        layer.bias_log_alpha.data[0] = -10.0

    # We consider eval mode as the medians of gates are selected deterministically
    # For log_alpha < -2.56, they are 0; for log_alpha > 2.56, they are 1.
    layer.eval()

    out_channels = layer.weight.shape[1]
    input = torch.randn(100, out_channels, 28, 28)
    if torch.cuda.is_available():
        input = input.cuda()

    out = layer(input)
    out_first_channel = out[:, 0, :, :]
    assert torch.allclose(out_first_channel, torch.zeros_like(out_first_channel))


def test_bias_sampling(layer):
    """Ensures that the bias is sampled correctly: it should be affected by
    gate values as L0Conv2d considers output sparsity."""
    if not layer.use_bias:
        pytest.skip("Test considered for layers with bias")

    # Ensure that at least one gate is off, for which the bias is different to 0.
    # Then, the bias term before and after multiplying by gates should be different.
    layer.bias.data[0] = 1.0
    layer.weight_log_alpha.data[0] = -10.0

    param_dict = layer.get_params(do_sample=False)
    sampled_bias = param_dict["bias"]

    assert not torch.allclose(layer.bias, sampled_bias)


def test_gate_sampling(layer, sparsity_type, use_bias):
    """Test that gates are sampled matching the state of the log_alpha parameter"""
    in_dim = layer.weight.shape[0]
    if in_dim < 2:
        pytest.skip("Test considered for layers with at least 2 input features")

    layer.weight_log_alpha.data.fill_(10.0)
    layer.weight_log_alpha.data[0] = -10.0

    w_medians, b_medians = layer.evaluation_gates()

    assert torch.all(w_medians[0] == 0.0)
    assert torch.all(w_medians[1:] > 0.0)
    # Sanity check that medians do not exceed 1.
    assert torch.all(w_medians <= 1.0)

    if sparsity_type == "unstructured" and use_bias:
        # We do not change the bias_log_alpha. Nonetheless, it must exist for
        # unstructured sparsity and can be sanity checked.
        assert torch.all(b_medians > 0.0)
        assert torch.all(b_medians <= 1.0)
