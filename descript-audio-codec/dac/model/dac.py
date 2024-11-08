import torch
from torch import nn
import math
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from dac.nn.quantize import ResidualVectorQuantize
from dac.nn.layers import Snake1d, WNConv1d, WNConvTranspose1d

class DAC(BaseModel):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: list = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: list = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        sample_rate: int = 44100,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = math.prod(encoder_rates)

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.encoder = self._create_encoder(encoder_dim, encoder_rates, latent_dim)
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
        )
        self.decoder = self._create_decoder(latent_dim, decoder_dim, decoder_rates)

    def _create_encoder(self, dim, rates, latent_dim):
        layers = [WNConv1d(1, dim, kernel_size=7, padding=3)]
        for rate in rates:
            dim *= 2
            layers.append(self._encoder_block(dim, rate))
        layers.extend([
            Snake1d(dim),
            WNConv1d(dim, latent_dim, kernel_size=3, padding=1),
        ])
        return nn.Sequential(*layers)

    def _encoder_block(self, dim, stride):
        return nn.Sequential(
            self._residual_unit(dim // 2),
            self._residual_unit(dim // 2),
            self._residual_unit(dim // 2),
            Snake1d(dim // 2),
            WNConv1d(dim // 2, dim, kernel_size=2*stride, stride=stride, padding=stride//2),
        )

    def _residual_unit(self, dim):
        return nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, padding=3),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def _create_decoder(self, input_dim, dim, rates):
        layers = [WNConv1d(input_dim, dim, kernel_size=7, padding=3)]
        for i, rate in enumerate(rates):
            layers.append(self._decoder_block(dim // (2**i), dim // (2**(i+1)), rate))
        layers.extend([
            Snake1d(dim // (2**len(rates))),
            WNConv1d(dim // (2**len(rates)), 1, kernel_size=7, padding=3),
            nn.Tanh(),
        ])
        return nn.Sequential(*layers)

    def _decoder_block(self, input_dim, output_dim, stride):
        return nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(input_dim, output_dim, kernel_size=2*stride, stride=stride, padding=stride//2),
            self._residual_unit(output_dim),
            self._residual_unit(output_dim),
            self._residual_unit(output_dim),
        )

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))
        return audio_data

    def encode(self, audio_data: torch.Tensor, n_quantizers: int = None):
        z = self.encoder(audio_data)
        z, codes, latents, commitment_loss, codebook_loss = self.quantizer(z, n_quantizers)
        return z, codes, latents, commitment_loss, codebook_loss

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def forward(self, audio_data: torch.Tensor, sample_rate: int = None, n_quantizers: int = None):
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        z, codes, latents, commitment_loss, codebook_loss = self.encode(audio_data, n_quantizers)
        x = self.decode(z)
        return {
            "audio": x[..., :length],
            "z": z,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss,
        }

    def compress(self, audio: AudioSignal, **kwargs):
        audio_data = audio.audio_data
        sample_rate = audio.sample_rate
        result = self.forward(audio_data, sample_rate, **kwargs)
        return result["codes"]

    def decompress(self, compressed_data, **kwargs):
        z = self.quantizer.decode(compressed_data)
        audio = self.decode(z)
        return AudioSignal(audio, sample_rate=self.sample_rate)