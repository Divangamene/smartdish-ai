"""Verifica se as bibliotecas principais estão instaladas e funcionais.

Uso:  uv run python scripts/check_env.py
"""
import importlib
import sys

LIBS = [
    "torch", "torchvision", "torchaudio", "torchmetrics", "lightning", "timm",
    "tensorflow", "keras", "numpy", "pandas", "scipy", "sklearn", "skimage",
    "PIL", "cv2", "albumentations", "matplotlib", "seaborn", "plotly",
    "tensorboard", "onnx", "onnxruntime", "gradio", "datasets", "smartdish",
]

print(f"Python {sys.version.split()[0]}\n")
for name in LIBS:
    try:
        mod = importlib.import_module(name)
        print(f"  OK   {name:<15} {getattr(mod, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        print(f"  FALHA {name:<15} {e}")

import torch  # noqa: E402

x = torch.randn(1, 3, 224, 224)
from smartdish.models.cnn import SimpleCNN  # noqa: E402

print(f"\nSimpleCNN forward: {tuple(SimpleCNN(10)(x).shape)} | CUDA: {torch.cuda.is_available()}")
