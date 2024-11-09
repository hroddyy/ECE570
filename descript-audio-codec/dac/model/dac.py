import math
from typing import List, Union
import torch
import torch.nn as nn
import numpy as np
from audiotools.ml import BaseModel  # Import BaseModel
from .base import CodecMixin  # Import CodecMixin

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=1)
        )

    def forward(self, x):
        y = self.block(x)
        return x + y

class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2),
            ResidualUnit(dim // 2),
            nn.Conv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2))
        )

    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    def __init__(self, d_model: int = 64, strides: List[int] = [2, 4]):
        super().__init__()
        self.block = [nn.Conv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            self.block.append(EncoderBlock(d_model, stride=stride))
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose1d(input_dim, output_dim, kernel_size=4, stride=2),
            ResidualUnit(output_dim),
            ResidualUnit(output_dim)
        )

    def forward(self, x):
        return self.block(x)

class Decoder(nn.Module):
    def __init__(self, input_channel: int = 64):
        super().__init__()
        layers = [nn.Conv1d(input_channel, input_channel // 2, kernel_size=7)]
        for _ in range(2):  # Two decoder blocks for simplicity
            layers.append(DecoderBlock(input_channel // (2 ** _), input_channel // (2 ** (_ + 1))))
        layers.append(nn.Conv1d(input_channel // (2 ** 2), 1, kernel_size=7))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DAC(BaseModel, CodecMixin):  # Inherit from BaseModel and CodecMixin
    def __init__(self,
                 encoder_dim: int = 64,
                 encoder_rates: List[int] = [2, 4],
                 latent_dim: int = None,
                 decoder_dim: int = 64,
                 sample_rate: int = 44100):
        
        super().__init__()  # Call the parent class constructor
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.sample_rate = sample_rate
        
        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        
        self.latent_dim = latent_dim
        
        # Initialize encoder and decoder components
        self.encoder = Encoder(encoder_dim)
        self.decoder = Decoder(decoder_dim)

    @classmethod
    def load(cls, load_path: str):
        """Load the DAC model from a checkpoint file."""
        model = cls()  
        model.load_state_dict(torch.load(load_path))  
        model.eval()  
        return model

    def encode(self, audio_data: torch.Tensor):
        return self.encoder(audio_data)

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def forward(self,
                audio_data: torch.Tensor,
                sample_rate: int = None):

        length = audio_data.shape[-1]
        
        audio_data = nn.functional.pad(audio_data, (0, length % self.sample_rate))  # Example padding
        
        z = self.encode(audio_data)
        
        audio_output = self.decode(z)
        
        return {
            "audio": audio_output,
            "z": z,
            "length": length
        }

if __name__ == "__main__":
    model = DAC().to("cpu")
    
    length = 44100 * 2 
    x = torch.randn(1, 1, length).to(model.device)
    
    out = model(x)["audio"]
    
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)