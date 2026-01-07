"""Configuration settings for tensor-toolbox."""
import torch

# Device configuration. We also use Apple MPS if available,
# as we are implementing on a macOS system with an M3 chip.
mps_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
if mps_available:
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# All the tensors are assumed real:
DTYPE = torch.float32
