import pytest
import torch

from sparse import L0MLP, L0LeNet5, L0WideResNet


@pytest.fixture(
    params=[
        # L0MLP model (input_dim, num_classes, layer_dims).
        (L0MLP, {"input_dim": 300, "num_classes": 2, "layer_dims": [100, 10]}),
        (
            L0MLP,
            {"input_dim": 784, "num_classes": 10, "layer_dims": []},
        ),  # Logistic regression
        (
            L0MLP,
            {
                "input_dim": 1000,
                "num_classes": 7,
                "layer_dims": [100, 10, 2, 10, 1, 100],
            },
        ),
        # L0LeNet5 model (input_shape, num_classes, conv_dims, fc_dims, kernel_sizes).
        (
            L0LeNet5,
            {
                "input_shape": (1, 28, 28),
                "num_classes": 10,
                "conv_dims": [20, 50],
                "fc_dims": 500,
                "kernel_sizes": (5, 5),
            },
        ),
        (
            L0LeNet5,
            {
                "input_shape": (3, 32, 32),
                "num_classes": 2,
                "conv_dims": [10, 5],
                "fc_dims": 1000,
                "kernel_sizes": (3, 3),
            },
        ),
        # WRN-28-5 on a CIFAR10-like dataset. Not using WRN-28-10 because it is too big for a 4gb GPU.
        (
            L0WideResNet,
            {
                "input_shape": (3, 32, 32),
                "depth": 28,
                "widen_factor": 5,
                "num_classes": 10,
                "weight_decay": 5e-4,
                "bn_type": "L0",
                "l2_detach_gates": True,
            },
        ),
        # WRN-16-8 on a tiny imagenet-like dataset.
        (
            L0WideResNet,
            {
                "input_shape": (3, 32, 32),
                "depth": 16,
                "widen_factor": 8,
                "num_classes": 200,
                "weight_decay": 5e-4,
                "bn_type": "L0",
                "l2_detach_gates": True,
            },
        ),
    ]
)
def config(request):
    return request.param


@pytest.fixture(params=["structured", "unstructured"])
def sparsity_type(request):
    return request.param


@pytest.fixture(params=[True, False])
def use_bias(request):
    return request.param


@pytest.fixture
def model(config, use_bias, sparsity_type):
    """Instantiate a model."""
    model_class, params = config

    if model_class == L0WideResNet and not torch.cuda.is_available():
        # Prevent any subsequent tests on WRN from happening on CPU-only devices.
        pytest.skip("We only perform WideResNet tests on GPUs.")

    model = model_class(use_bias=use_bias, sparsity_type=sparsity_type, **params)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


@pytest.fixture
def spec(config):
    """Extract the parameters used for model initialization."""
    _, param_dict = config
    return param_dict


def test_forward_shape(model, spec):
    """Verify dimensions of forward output."""
    input_dim = (
        (spec["input_dim"],) if isinstance(model, L0MLP) else spec["input_shape"]
    )
    num_classes = spec["num_classes"]

    t = torch.randn(100, *input_dim)
    if torch.cuda.is_available():
        t = t.cuda()

    model.eval()
    out = model.forward(t)
    assert out.shape == (100, num_classes)


def test_eval_forward(model, spec):
    """Verify that forwards in *eval* mode are deterministic operations."""
    input_dim = (
        (spec["input_dim"],) if isinstance(model, L0MLP) else spec["input_shape"]
    )
    t = torch.randn(100, *input_dim)
    if torch.cuda.is_available():
        t = t.cuda()

    model.eval()
    out1 = model.forward(t)
    out2 = model.forward(t)

    # Lenient tolerance is required for ResNet models.
    atol = 1e-4 if isinstance(model, L0WideResNet) else 1e-6
    assert torch.allclose(out1, out2, atol=atol)


def test_l0reg(model):
    """Test that the L0 regularization calculation is monotonic with respect to
    log_alpha."""

    lb = -10.0
    ub = 10.0

    for layer in model.layers_dict["l0"]:
        layer.weight_log_alpha.data.fill_(lb)
    low_reg = model.regularization()

    for layer in model.layers_dict["l0"]:
        layer.weight_log_alpha.data.fill_(ub)
    upper_reg = model.regularization()

    assert 0.0 <= low_reg.l0_model < upper_reg.l0_model <= 1.0

    assert all(0.0 <= low_reg.l0_layer)
    assert all(low_reg.l0_layer < upper_reg.l0_layer)
    assert all(upper_reg.l0_layer <= 1.0)


def test_weight_decay(model):
    """Test that the weight decay calculation is zero when weight_decay=0."""

    reg_dict = model.regularization()
    if model.weight_decay == 0.0:
        assert reg_dict.l2_model == 0.0
    else:
        assert reg_dict.l2_model > 0.0
