import pytest
import torch

from sparse import L0Linear


@pytest.fixture(params=["structured", "unstructured"])
def sparsity_type(request):
    return request.param


@pytest.fixture(params=[True, False])
def use_bias(request):
    return request.param


@pytest.fixture(
    params=[
        {"in_features": 10, "out_features": 100},
        {"in_features": 130, "out_features": 2},
    ]
)
def layer_kwargs(request):
    return request.param


@pytest.fixture()
def layer(layer_kwargs, use_bias, sparsity_type):
    linear_layer = L0Linear(
        use_bias=use_bias, sparsity_type=sparsity_type, **layer_kwargs
    )
    if torch.cuda.is_available():
        linear_layer = linear_layer.cuda()
    return linear_layer


def test_init(layer, sparsity_type, use_bias):
    """Sanity checks on the shape of tensors inside L0Linear"""

    # L0Linear weight dims are currently (in_features, out_features), opposed to
    # (out_features, in_features) like in vanilla Pytorch.
    out_features, in_features = layer.weight.shape

    if layer.use_bias:
        # out_features should match the shape of the bias term
        assert layer.bias.shape == (out_features,)
        if sparsity_type == "unstructured":
            assert layer.bias_log_alpha.shape == (out_features,)

    if sparsity_type == "structured":
        # Linear layers have input neuron sparsity. log_alpha should match in_features
        assert layer.weight_log_alpha.shape == (in_features,)
    else:
        assert layer.weight_log_alpha.shape == (out_features, in_features)


def test_sparse_forward(layer, sparsity_type, use_bias):
    """Checks that forward passes are consistent with the given layer sparsity"""

    # Make sure that all of the gates are very likely to be turned off
    layer.weight_log_alpha.data.fill_(-10.0)

    if sparsity_type == "unstructured" and use_bias:
        layer.bias_log_alpha.data.fill_(-10.0)

    # We consider eval mode as the medians of gates are selected deterministically
    # and for log_alpha < -2.56, they are 0. We therefore guarantee full sparsity
    layer.eval()

    in_dim = layer.weight.shape[1]
    t = torch.randn(100, in_dim)
    if torch.cuda.is_available():
        t = t.cuda()

    out = layer(t)
    if layer.use_bias and sparsity_type == "structured":
        # With all gates at 0, out should match the layer's bias
        assert torch.allclose(out, layer.bias)
    else:
        # Alternatively, if the layer has no bias or the bias is off in unstructured
        # sparsity, the output should be 0
        assert torch.allclose(out, torch.zeros_like(out))


def test_forward(layer):
    """Checks that forward passes apply input sparsity correctly on L0Linear layers"""

    # Set all gates to be on, except the first one. The first feature of inputs
    # should be ignored by the layer.
    layer.weight_log_alpha.data.fill_(10.0)
    layer.weight_log_alpha.data[..., 0] = -10.0

    # We consider eval mode as the medians of gates are selected deterministically
    # For log_alpha < -2.56, they are 0; for log_alpha > 2.56, they are 1.
    layer.eval()

    if layer.use_bias:
        # Remove the bias influence on the output.
        layer.bias.data.fill_(0.0)

    out_dim, in_dim = layer.weight.shape
    # Input with zeros everywhere except on the first feature. If input sparsity
    # is applied correctly, the first feature should be ignored and the whole
    # output should consist of 0s.
    input = torch.zeros(100, in_dim)
    input[:, 0] = 1.0

    target = torch.zeros(100, out_dim)

    if torch.cuda.is_available():
        input = input.cuda()
        target = target.cuda()

    out = layer(input)
    assert torch.allclose(out, target)


def test_bias_sampling(layer, sparsity_type):
    """Ensures that the bias is sampled correctly: it should not be affected by
    gate values as L0Linear considers input sparsity."""
    if not layer.use_bias or sparsity_type == "unstructured":
        pytest.skip("Test considered for structured sparsity layers with bias")

    param_dict = layer.get_params(do_sample=False)
    sampled_bias = param_dict["bias"]

    assert torch.allclose(layer.bias, sampled_bias)


def test_gate_sampling(layer, sparsity_type, use_bias):
    """Test that gates are sampled matching the state of the log_alpha parameter"""
    in_dim = layer.weight.shape[1]
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
