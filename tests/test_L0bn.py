from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sparse import L0BatchNorm2d, L0Conv2d
from utils.datasets import mnist


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.input_shape = (1, 1, 28, 28)
        # use_bias=False as the bias is neglected if followed by a BN.
        self.conv = L0Conv2d(1, 15, False, "structured", 3, stride=2)
        self.bn = L0BatchNorm2d(15)
        self.fc = nn.Linear(15, 10)

    def forward(self, x):

        out, mask = self.conv(x, return_mask=True)
        out = self.bn(out, gate_mask=mask)
        out = F.relu(out)

        pool_k_size = (out.shape[2], out.shape[3])
        out = F.avg_pool2d(out, pool_k_size)
        out = out.view(out.size(0), -1)

        return self.fc(out)


@pytest.fixture
def model():
    model = Model()
    if torch.cuda.is_available():
        model = model.cuda()

    # Set all gates to be on, except the first two.
    model.conv.weight_log_alpha.data[0:2] = -10.0
    model.conv.weight_log_alpha.data[2:] = 10.0

    return model


@pytest.fixture
def train_loader():
    train_loader, _, _ = mnist(100, 100)
    return train_loader


@pytest.fixture
def dummy_input():
    t = torch.randn(100, 1, 28, 28)
    if torch.cuda.is_available():
        t = t.cuda()
    return t


@pytest.fixture
def running_stats(model):
    """Extract the current running_stats from a trained model."""
    # Extract current running_mean and running_var
    running_mean = deepcopy(model.bn.running_mean)
    running_var = deepcopy(model.bn.running_var)

    return running_mean, running_var


def test_off_forward(model, dummy_input):
    """Test that off units remain off after batch norm."""

    # We consider eval mode as the medians of gates are selected deterministically
    # For log_alpha < -2.56, they are 0; for log_alpha > 2.56, they are 1.
    model.eval()

    conv_out, mask = model.conv(dummy_input, return_mask=True)
    assert all(mask[0:2] == 0)

    bn_out = model.bn(conv_out, gate_mask=mask)
    masked_out = bn_out[:, 0:2, ...]

    assert torch.allclose(masked_out, torch.zeros_like(masked_out))


def test_running_stats_train(model, running_stats, dummy_input):
    """Test that *train* mode forwards modify running_mean and running_var."""
    running_mean, running_var = running_stats

    model.train()
    _ = model(dummy_input)

    assert not torch.allclose(running_mean, model.bn.running_mean)
    assert not torch.allclose(running_var, model.bn.running_var)


def test_running_stats_eval(model, running_stats, dummy_input):
    """Test that *eval* mode forwards *do not* modify running stats."""

    running_mean, running_var = running_stats

    model.eval()
    _ = model(dummy_input)

    assert torch.allclose(running_mean, model.bn.running_mean)
    assert torch.allclose(running_var, model.bn.running_var)


def test_sparse_running_stats(model, running_stats, dummy_input):
    """Test that the running_stats are not modified in the position of off units."""

    running_mean, running_var = running_stats

    model.train()
    _ = model(dummy_input)

    assert torch.allclose(running_mean[0:2], model.bn.running_mean[0:2])
    assert torch.allclose(running_var[0:2], model.bn.running_var[0:2])


@pytest.fixture
def model_with_gradients(model, dummy_input):
    model.train()
    _ = model(dummy_input)

    dummy_target = torch.randint(0, 10, (100,)).to(dummy_input.device)
    loss = F.cross_entropy(model(dummy_input), dummy_target)
    model.zero_grad()
    loss.backward()

    return model


def test_bn_gradients(model_with_gradients):
    """BN params (weight, bias) associated with *off* units should have zero gradient"""

    masked_w_grad = model_with_gradients.bn.weight.grad[0:2]
    masked_b_grad = model_with_gradients.bn.bias.grad[0:2]

    assert torch.allclose(masked_w_grad, torch.zeros_like(masked_w_grad))
    assert torch.allclose(masked_b_grad, torch.zeros_like(masked_b_grad))


def test_conv_gradients(model_with_gradients):
    """Conv params associated with *off* units should have *non-zero* gradients
    Test that the backward through the BN is propagated correctly."""

    masked_w_grad = model_with_gradients.conv.weight.grad[:, 0:2, ...]

    assert not torch.allclose(masked_w_grad, torch.zeros_like(masked_w_grad))
