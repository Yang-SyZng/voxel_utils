# utils/__init__.py
from typing import Final
import torch

DOUBLE: Final = torch.double
FLOAT64: Final = torch.float64
FLOAT32: Final = torch.float32
HASH_P: Final = 116101
MAX_N: Final = 10000000000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"