from argparse import Namespace
import re

import numpy as np
import pytest
import torch

import sparse
from utils.exp_utils import construct_model, get_macs_and_params, purge_model


def purge_and_compare(model, input_shape):

    # Induce sparsity by setting the log_alpha of all convolutional layers
    for layer in model.layers_dict["l0"]:
        num_gates = layer.weight_log_alpha.shape[0]
        layer.weight_log_alpha.data.fill_(10.0)

        if isinstance(layer, sparse.L0Linear):
            layer.weight_log_alpha.data[0 : num_gates // 3, ...] = -10.0
        else:
            layer.weight_log_alpha.data[2 * num_gates // 3 :, ...] = -10.0

    # Purge the model
    purged_model = purge_model(model)

    if torch.cuda.is_available():
        model = model.cuda()
        purged_model = purged_model.cuda()

    model.eval()
    purged_model.eval()

    with torch.inference_mode():

        # Compute several forwards to increase certainty in equivalence between
        # the outputs of purged and original models
        for batch_id in range(10):
            x = torch.randn((10, *input_shape)) / np.sqrt(np.prod(input_shape))
            if torch.cuda.is_available():
                x = x.cuda()

            base_output = model(x)
            purged_output = purged_model(x)

            assert purged_output.shape == base_output.shape
            assert torch.allclose(base_output, purged_output, atol=1e-4)
              
            # ptflops does not support the modules in non-purged model
            # Only computing flops for purged model since it has "usual" Pytorch
            # modules (Conv2d as opposed to L0Conv2d)
            print(get_macs_and_params(purged_model))


@pytest.fixture(params=["unstructured", "structured"])
def sparsity_type(request):
    return request.param


@pytest.fixture(params=[True, False])
def use_bias(request):
    return request.param


@pytest.mark.parametrize(
    "model_type", ["MLP", "LeNet", "L0ResNet18", "L0ResNet50", "ResNet-28-10"]
)
def test_purging(model_type, sparsity_type, use_bias):
    """Test the purging L0-sparsifiable models."""

    # Creating a Namespace to simulate execution of core_exp
    args = Namespace(
        weight_decay=0.0,
        l2_detach_gates=True,
        temp=2.0 / 3.0,
        droprate_init=0.3,
        sparsity_type=sparsity_type,
        use_bias=use_bias,
        act_fn="ReLU",
        model_type=model_type,
        bn_type="L0",
        task_type="gated",
    )

    if model_type in ["MLP", "LeNet"]:
        input_shape = (1, 28, 28)
        num_classes = 10
    elif model_type == "L0ResNet18":
        input_shape = (3, 64, 64)
        num_classes = 200
        # l0_conv_ix = ["conv1", "conv2"] # This is the hard-coded default
    elif model_type == "L0ResNet50":
        input_shape = (3, 224, 224)
        num_classes = 1000
        # l0_conv_ix = ["conv1", "conv2"] # This is the hard-coded default
    elif model_type == "ResNet-28-10":
        input_shape = (3, 32, 32)
        num_classes = 100
    else:
        raise ValueError("Model type not understood")

    model = construct_model(args, num_classes, input_shape)

    purge_and_compare(model, input_shape)
