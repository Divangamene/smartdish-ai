import torch

from smartdish.models.cnn import SimpleCNN


def test_simple_cnn_output_shape():
    model = SimpleCNN(num_classes=5)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 5)
