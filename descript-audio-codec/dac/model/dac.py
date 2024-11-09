import math
from typing import List
from typing import Union

import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from torch import nn

from .base import CodecMixin
from dac.nn.layers import Snake1d
from dac.nn.layers import WNConv1d
from dac.nn.layers import WNConvTranspose1d
from dac.nn.quantize import ResidualVectorQuantize

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

# Simplified ResidualUnit block
class ResidualUnit(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return x + self.relu(self.conv2(self.relu(self.conv1(x))))

# Simplified Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim: int = 64):
        super().__init__()
        self.conv = nn.Conv1d(1, input_dim, kernel_size=3, padding=1)
        self.residual_unit = ResidualUnit(input_dim)

    def forward(self, x):
        x = self.conv(x)
        return self.residual_unit(x)

# Simplified Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim: int = 1):
        super().__init__()
        self.residual_unit = ResidualUnit(output_dim)
        self.conv = nn.Conv1d(output_dim, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.residual_unit(x)
        return self.conv(x)

# Simplified DAC model
class DAC(nn.Module):
    def __init__(self, encoder_dim: int = 64):
        super().__init__()
        self.encoder = Encoder(encoder_dim)
        self.decoder = Decoder(encoder_dim)
        self.apply(init_weights)

    def forward(self, x):
        # Encode and decode the signal
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    @classmethod
    def load(cls, path: str, map_to_cpu: bool = True):
        """Load the model state from a checkpoint with option to map to CPU."""
        if map_to_cpu:
            checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=True)
        else:
            checkpoint = torch.load(path, weights_only=True)
        model = cls()  # Initialize the model
        model.load_state_dict(checkpoint["model_state_dict"])  # Load state_dict
        return model

def load_model(load_path: str):
    """Load a DAC model from the specified checkpoint path."""
    generator = DAC.load(load_path)
    return generator

# Example usage
if __name__ == "__main__":
    load_path = "path/to/checkpoint.pth"  # Replace with the actual path to your checkpoint
    model = load_model(load_path).to("cpu")  # Load the model from a checkpoint
    x = torch.randn(1, 1, 44100)  # Example input tensor
    output = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)