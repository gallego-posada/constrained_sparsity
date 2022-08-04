import pytest
import torch

import sparse


@pytest.fixture(params=["structured", "unstructured"])
def sparsity_type(request):
    return request.param

@pytest.mark.parametrize("depth", [18, 50])
def test_resnet_creation(depth, sparsity_type):
    """Test the instantiation of L0ResNet18 and L0ResNet50 model."""

    assert depth in [18, 50]

    kwargs = {
        "weight_decay": 0.0,
        "temperature": 2.0 / 3.0,
        "l2_detach_gates": True,
        "use_bias": False,
        "sparsity_type": sparsity_type,
    }

    # ResNet50 for Imagenet (3, 224, 224) in_shape and ResNet18 for
    # tiny_ImageNet (3, 32, 32) in_shape
    if depth == 18:
        input_shape = (3, 64, 64)
        num_classes = 200
        l0_conv_ix = ["conv1", "conv2"]
    if depth == 50:
        input_shape = (3, 224, 224)
        num_classes = 1000
        l0_conv_ix = ["conv1", "conv2", "conv3"]

    # Dummy inputs of the appropriate size.
    x = torch.randn((1, *input_shape))

    model_class = getattr(sparse, f"L0ResNet{depth}")
    resnet_model = model_class(
        num_classes=num_classes,
        input_shape=input_shape,
        l0_conv_ix=l0_conv_ix,
        **kwargs,
    )

    if torch.cuda.is_available():
        resnet_model = resnet_model.cuda()
        x = x.cuda()

    # ImageNet dataset has 1000 classes, tiny-ImageNet has 200 classes
    assert resnet_model(x).shape == (1, num_classes)

    for layer_id in [1, 2, 3, 4]:

        current_layer = getattr(resnet_model, f"layer{layer_id}")

        # Since "shortcut_conv" is not an entry in l0_conv_ix, the shortcut should
        # be a regular nn.Conv2d layer.
        # Only the first block in a layer has a shortcut.
        if hasattr(current_layer[0], "shortcut_conv"):
            assert isinstance(current_layer[0].shortcut_conv, torch.nn.Conv2d)

        # Since "conv1" *is* an entry in l0_conv_ix, this should be an L0Conv2d layer.
        for block in current_layer:
            assert isinstance(block.conv1, sparse.L0Conv2d)
