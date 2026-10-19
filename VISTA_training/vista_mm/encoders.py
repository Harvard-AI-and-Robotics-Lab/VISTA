"""Pretrained sensor encoders for question-conditioned temporal resampling.

Audio channels are encoded independently and averaged; IMU inputs share a PRIMUS
encoder. Weights are loaded explicitly during model initialization."""

from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .primus import PrimusEncoder


def build_beats(config):
    from .third_party.beats.BEATs import BEATs, BEATsConfig

    if not config or config.get("input_patch_size", -1) <= 0:
        raise ValueError(
            "BEATs architecture must be resolved from the verified asset manifest"
        )
    if config.get("finetuned_model", False):
        raise ValueError(
            "Use a BEATs pretrained representation checkpoint, not a classifier"
        )
    if config.get("encoder_layerdrop", 0) != 0:
        raise ValueError(
            "LayerDrop must be zero for deterministic parameter participation"
        )
    return BEATs(BEATsConfig(config))


def beats_features(network, fbank):
    """Official extract_features trunk, with canonical fbank computed during prepare."""
    features = network.patch_embedding(fbank.unsqueeze(1))
    features = features.reshape(features.shape[0], features.shape[1], -1).transpose(
        1, 2
    )
    features = network.layer_norm(features)
    if network.post_extract_proj is not None:
        features = network.post_extract_proj(features)
    return network.encoder(network.dropout_input(features), padding_mask=None)[0]


class NativeFrontend(nn.Module):
    def __init__(self, modality, config):
        super().__init__()
        self.modality, self.config = modality, config
        self.network = (
            build_beats(config.beats_config) if modality == "audio" else PrimusEncoder()
        )
        self.set_trainable(config.finetune_encoders)

    def set_trainable(self, enabled):
        self.network.requires_grad_(enabled)
        self.network.train(self.training and enabled)

    def train(self, mode=True):
        super().train(mode)
        if not self.config.finetune_encoders:
            self.network.eval()
        return self

    def _encode(self, values):
        def run(x):
            return (
                beats_features(self.network, x)
                if self.modality == "audio"
                else self.network(x)
            )

        chunks = []
        context = nullcontext if self.config.finetune_encoders else torch.no_grad
        # Load in float32; a distributed engine may cast the entire model later.
        # Match its actual parameter dtype explicitly, including the GRU input.
        with context(), torch.autocast(device_type=values.device.type, enabled=False):
            dtype = next(self.network.parameters()).dtype
            for x in values.to(dtype).split(self.config.encoder_microbatch):
                if (
                    self.training
                    and self.config.finetune_encoders
                    and self.config.encoder_checkpointing
                ):
                    chunks.append(checkpoint(run, x, use_reentrant=False))
                else:
                    chunks.append(run(x))
        return torch.cat(chunks)

    def forward(self, data):
        x, valid = data["native_values"], data["native_mask"].bool()
        batch, windows, devices = x.shape[:3]
        flat_valid = valid.reshape(-1)
        flat = x.flatten(0, 2)
        selected = flat_valid.nonzero().flatten()
        # One masked dummy keeps every trunk parameter in the graph even for an
        # all-missing minibatch. No all-masked attention or conditional DDP parameters.
        if not len(selected):
            selected = torch.zeros(1, dtype=torch.long, device=x.device)
        clean = flat.index_select(0, selected)
        clean = clean.masked_fill(~flat_valid[selected, None, None], 0)
        if not torch.isfinite(clean).all():
            raise ValueError("Nonfinite observed native encoder input")
        encoded = self._encode(clean)
        encoded = encoded.masked_fill(
            ~flat_valid[selected].reshape(-1, *([1] * (encoded.ndim - 1))), 0
        )
        result = encoded.new_zeros((len(flat), *encoded.shape[1:])).index_copy(
            0, selected, encoded
        )
        starts = data["native_times"].float()
        if self.modality == "audio":
            # [B, windows, microphones, time*frequency, 768]
            result = result.reshape(batch, windows, devices, *encoded.shape[1:])
            result = result.sum(2) / valid.sum(2).clamp_min(1)[..., None, None]
            patch = self.network.input_patch_size
            nt, nf = x.shape[-2] // patch, x.shape[-1] // patch
            # Kaldi centers are 12.5 ms + k*10 ms, then averaged within each patch.
            offsets = (
                torch.arange(nt, device=x.device) * patch + (patch - 1) / 2
            ) * 0.01 + 0.0125
            times = (starts[:, :, None, None] + offsets[None, None, :, None]).expand(
                -1, -1, -1, nf
            )
            positions = x.new_zeros(batch, windows, nt, nf, 3)
            positions[..., 0] = (torch.arange(nf, device=x.device) + 0.5) / nf
            present = valid.any(2)[..., None].expand(-1, -1, nt * nf)
            quality = valid.float().mean(2)[..., None].expand_as(present)
        else:
            result = result.reshape(batch, windows, devices, -1)
            times = (starts[:, :, None] + 2.5).expand(-1, -1, devices)
            positions = x.new_zeros(batch, windows, devices, 3)
            positions[..., 0] = torch.arange(devices, device=x.device) * 2 - 1
            present, quality = valid, valid.float()
        result = result.reshape(batch, -1, result.shape[-1])
        return {
            "values": result,
            "mask": present.reshape(batch, -1, 1).expand_as(result),
            "quality": quality.reshape(batch, -1),
            "times": times.reshape(batch, -1),
            "positions": positions.reshape(batch, -1, 3),
            "duration": data["duration"],
        }
