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
from torch.nn.utils.parametrizations import weight_norm
import tqdm

def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)

class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y

class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2, dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    def __init__(
        self, d_model: int = 64, strides: list = [2, 4, 8, 8], d_latent: int = 64,
    ):
        super().__init__()
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)

class Decoder(nn.Module):
    def __init__(
        self, input_channel, channels, rates, d_out: int = 1,
    ):
        super().__init__()
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class DAC(BaseModel, CodecMixin):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.latent_dim = latent_dim
        self.hop_length = np.prod(encoder_rates)

        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )
        self.decoder = Decoder(
            latent_dim, decoder_dim, decoder_rates,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)
        self.delay = self.get_delay()


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

    @torch.no_grad()
    def compress(
        self,
        audio_path_or_signal: Union[str, Path, AudioSignal],
        win_duration: float = 1.0,
        verbose: bool = False,
        normalize_db: float = -16,
        n_quantizers: int = None,
    ) -> DACFile:
        audio_signal = audio_path_or_signal if isinstance(audio_signal, AudioSignal) else AudioSignal.load_from_file_with_ffmpeg(str(audio_path_or_signal))
        
        self.eval()
        original_padding = self.padding
        original_device = audio_signal.device
        audio_signal = audio_signal.clone()
        original_sr = audio_signal.sample_rate
        original_length = audio_signal.signal_length

        # Resample and normalize
        audio_signal.resample(self.sample_rate)
        input_db = audio_signal.loudness()
        if normalize_db is not None:
            audio_signal.normalize(normalize_db)
        audio_signal.ensure_max_of_audio()

        nb, nac, nt = audio_signal.audio_data.shape
        audio_signal.audio_data = audio_signal.audio_data.reshape(nb * nac, 1, nt)

        # Determine processing strategy (chunked or unchunked)
        if audio_signal.signal_duration <= win_duration:
            self.padding = True
            n_samples = nt
            hop = nt
        else:
            self.padding = False
            audio_signal.zero_pad(self.delay, self.delay)
            n_samples = int(win_duration * self.sample_rate)
            n_samples = int(math.ceil(n_samples / self.hop_length) * self.hop_length)
            hop = self.get_output_length(n_samples)

        # Process audio in chunks
        codes = []
        range_fn = tqdm.trange if verbose else range
        for i in range_fn(0, nt, hop):
            x = audio_signal[..., i : i + n_samples]
            x = x.zero_pad(0, max(0, n_samples - x.shape[-1]))
            audio_data = x.audio_data.to(self.device)
            audio_data = self.preprocess(audio_data, self.sample_rate)
            _, c, _, _, _ = self.encode(audio_data, n_quantizers)
            codes.append(c.to(original_device))

        chunk_length = c.shape[-1]
        codes = torch.cat(codes, dim=-1)

        dac_file = DACFile(
            codes=codes,
            chunk_length=chunk_length,
            original_length=original_length,
            input_db=input_db,
            channels=nac,
            sample_rate=original_sr,
            padding=self.padding,
            dac_version=SUPPORTED_VERSIONS[-1],
        )

        if n_quantizers is not None:
            codes = codes[:, :n_quantizers, :]

        self.padding = original_padding
        return dac_file

    def decompress(self, compressed_data, verbose: bool = False, **kwargs):
        z = self.quantizer.decode(compressed_data)
        audio = self.decode(z)
        
        if verbose:
            print(f"Decompressed audio shape: {audio.shape}")
        
        return AudioSignal(audio, sample_rate=self.sample_rate)