"""Question-conditioned sensor fusion with temporal alignment and missing-data masks.

Temporal adapters produce eight tokens per stream. Fusion supports shared
bottleneck exchange and one-way cross-attention, with reliability gates learned
through supervised answer loss."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .bottleneck import BottleneckBlock
from .config import DIMS, SensorConfig
from .encoders import NativeFrontend


def sinusoidal(times, width):
    freq = torch.exp(
        torch.arange(0, width, 2, device=times.device, dtype=torch.float32)
        * (-math.log(10000) / width)
    )
    angle = times.float().unsqueeze(-1) * freq
    result = torch.stack([angle.sin(), angle.cos()], -1).flatten(-2)
    return result[..., :width]


class CrossBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads = heads
        self.qnorm = nn.LayerNorm(width)
        self.knorm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * width, width),
            nn.Dropout(dropout),
        )

    def forward(self, queries, memory, valid, log_prior=None):
        # Float attention bias combines validity with quality; avoid dtype-mismatched masks.
        bias = torch.zeros(
            (memory.shape[0], memory.shape[1]),
            device=memory.device,
            dtype=queries.dtype,
        )
        if log_prior is not None:
            bias = bias + log_prior.to(bias.dtype)
        bias = bias.masked_fill(~valid, float("-inf"))
        bias = (
            bias[:, None, None, :]
            .expand(-1, self.heads, queries.shape[1], -1)
            .reshape(-1, queries.shape[1], memory.shape[1])
        )
        key = self.knorm(memory)
        x = self.attn(
            self.qnorm(queries), key, key, attn_mask=bias, need_weights=False
        )[0]
        queries = queries + x
        return queries + self.ff(queries)


class ModalityAdapter(nn.Module):
    def __init__(self, dim, config, positions=False):
        super().__init__()
        d = config.width
        self.norm = nn.LayerNorm(dim)
        self.project = nn.Linear(dim, d)
        self.mask_project = nn.Linear(dim, d, bias=False)
        self.position_project = nn.Linear(3, d, bias=False) if positions else None
        self.quality_project = (
            nn.Linear(1, d, bias=False) if config.use_quality else None
        )
        self.queries = nn.Parameter(torch.randn(config.tokens_per_modality, d) * 0.02)
        self.null = nn.Parameter(torch.randn(1, d) * 0.02)
        self.blocks = nn.ModuleList(
            [
                CrossBlock(d, config.heads, config.dropout)
                for _ in range(config.adapter_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d)

    def forward(self, values, mask, quality, times, duration, positions=None):
        values = values.to(self.project.weight.dtype)
        mask = mask.bool()
        values = values.masked_fill(~mask, 0)
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite observed sensor input")
        relative_time = times.float() / duration[:, None].clamp_min(1e-6)
        memory = self.project(self.norm(values)) + self.mask_project(
            mask.to(values.dtype)
        )
        if self.position_project is not None:
            if positions is None:
                raise ValueError(
                    "Native modality tokens require spatial/device coordinates"
                )
            memory = memory + self.position_project(positions.to(values.dtype))
        memory = memory + sinusoidal(relative_time * 2 * math.pi, memory.shape[-1]).to(
            memory.dtype
        )
        if self.quality_project is not None:
            memory = memory + self.quality_project(
                quality.to(values.dtype).unsqueeze(-1)
            )
        valid = mask.any(-1)
        present = valid.any(-1)
        # A learned null key is enabled only when the entire modality is absent.
        memory = torch.cat([memory, self.null[None].expand(len(memory), -1, -1)], 1)
        valid = torch.cat([valid, ~present[:, None]], 1)
        anchors = torch.linspace(
            0, 2 * math.pi, len(self.queries), device=values.device
        )
        queries = self.queries + sinusoidal(anchors, self.queries.shape[-1]).to(
            values.dtype
        )
        queries = queries[None].expand(len(values), -1, -1)
        for block in self.blocks:
            queries = block(queries, memory, valid)
        return self.output_norm(queries), present


class QuestionEncoder(nn.Module):
    def __init__(self, hidden, config):
        super().__init__()
        self.project = nn.Linear(hidden, config.width)
        layer = nn.TransformerEncoderLayer(
            config.width,
            config.heads,
            config.width * 4,
            config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, config.question_layers, enable_nested_tensor=False
        )
        self.score = nn.Linear(config.width, 1)
        self.norm = nn.LayerNorm(config.width)

    def forward(self, embeddings, mask):
        if not mask.any(1).all():
            raise ValueError("Each example needs nonempty question text")
        x = self.project(embeddings.to(self.project.weight.dtype))
        pos = torch.arange(x.shape[1], device=x.device)
        x = x + sinusoidal(pos, x.shape[-1]).to(x.dtype)
        x = self.encoder(x, src_key_padding_mask=~mask.bool())
        weights = (
            self.score(x)
            .squeeze(-1)
            .float()
            .masked_fill(~mask.bool(), float("-inf"))
            .softmax(-1)
        )
        return self.norm((x * weights.to(x.dtype).unsqueeze(-1)).sum(1))


class ReliabilityGates(nn.Module):
    """Joint-context sigmoid gates, inspired by GMU's multi-input conditioning.

    Inputs contain question text and observed sensor features only. The joint
    context never reads dropped/absent values. Gates are calculated before any
    question-conditioned exchange, keeping the no-question-gates ablation valid.
    Scores are routing parameters, not calibrated confidence or causal importance.
    """

    def __init__(self, config: SensorConfig):
        super().__init__()
        self.config = config
        joint = config.gate_context == "joint"
        self.context = None
        if joint:
            context_width = len(config.modalities) * (config.width + 4)
            self.context = nn.Sequential(
                nn.LayerNorm(context_width),
                nn.Linear(context_width, config.width),
                nn.GELU(),
            )
        input_width = config.width * (3 if joint else 2) + 3
        self.gates = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.LayerNorm(input_width),
                    nn.Linear(input_width, config.width),
                    nn.GELU(),
                    nn.Linear(config.width, 1),
                )
                for m in config.modalities
            }
        )

    def forward(self, tokens, question, info, present):
        summary = tokens.mean(dim=2).masked_fill(~present[:, :, None], 0)
        info = info.masked_fill(~present[:, :, None], 0).to(summary.dtype)
        if not self.config.use_quality:
            info = torch.zeros_like(info)
        question = (
            question if self.config.use_question_gates else torch.zeros_like(question)
        )
        context = None
        if self.context is not None:
            joint = torch.cat(
                [summary, info, present[:, :, None].to(summary.dtype)], dim=-1
            )
            context = self.context(joint.flatten(1))
        scores = []
        for index, modality in enumerate(self.config.modalities):
            parts = [question, summary[:, index], info[:, index]]
            if context is not None:
                parts.append(context)
            scores.append(self.gates[modality](torch.cat(parts, dim=-1)).squeeze(-1))
        logits = torch.stack(scores, dim=1).float() / self.config.gate_temperature
        return torch.sigmoid(logits).masked_fill(~present, 0)


def mixture_weights(reliability, present, learned=True):
    """Normalize routing mass, including an explicit no-update/null alternative.

    The ungated control averages available streams exactly, as in MBT, and uses
    the fallback only for an all-missing sample. Compute in float32 under autocast.
    """
    reliability = reliability.float().masked_fill(~present, 0)
    if learned:
        fallback = (1 - reliability.max(dim=1, keepdim=True).values).clamp_min(1e-4)
    else:
        reliability = present.float()
        fallback = (~present.any(dim=1, keepdim=True)).float()
    weights = torch.cat([reliability, fallback], dim=1)
    return weights / weights.sum(dim=1, keepdim=True)


class SensorFusion(nn.Module):
    def __init__(self, hidden_size, config: SensorConfig):
        super().__init__()
        if not config.modalities:
            raise ValueError(
                "SensorFusion requires modalities; use native Qwen for RGB-only"
            )
        self.config = config
        self.question = QuestionEncoder(hidden_size, config)
        dims = dict(DIMS)
        native = []
        if config.audio_encoder == "beats" and "audio" in config.modalities:
            dims["audio"] = config.beats_config.get("encoder_embed_dim", 768)
            native.append("audio")
        if config.imu_encoder == "primus" and "imu" in config.modalities:
            dims["imu"] = 512
            native.append("imu")
        self.frontends = nn.ModuleDict({m: NativeFrontend(m, config) for m in native})
        if config.depth_encoder == "spatial":
            dims["metric3d"] = 6
        self.adapters = nn.ModuleDict(
            {
                m: ModalityAdapter(
                    dims[m],
                    config,
                    m in native
                    or (m == "metric3d" and config.depth_encoder == "spatial"),
                )
                for m in config.modalities
            }
        )
        self.type_embeddings = nn.Parameter(
            torch.randn(len(config.modalities), config.width) * 0.02
        )
        self.gating = ReliabilityGates(config) if config.use_reliability_gates else None
        self.queries = nn.Parameter(
            torch.randn(config.fused_tokens, config.width) * 0.02
        )
        if config.fusion == "bottleneck":
            self.register_parameter("null", None)
            self.blocks = nn.ModuleList(
                [BottleneckBlock(config) for _ in range(config.fusion_layers)]
            )
        else:
            self.null = nn.Parameter(torch.randn(1, config.width) * 0.02)
            self.blocks = nn.ModuleList(
                [
                    CrossBlock(config.width, config.heads, config.dropout)
                    for _ in range(config.fusion_layers)
                ]
            )
        self.output = nn.Sequential(
            nn.LayerNorm(config.width), nn.Linear(config.width, hidden_size)
        )
        self.last_gate_weights = None
        self.last_mixture_weights = None
        self.last_modality_available = None

    def _cross_attention(self, tokens, queries, present, mixture):
        cfg = self.config
        memory = torch.cat(
            [tokens.flatten(1, 2), self.null[None].expand(len(queries), -1, -1)], dim=1
        )
        valid = torch.cat(
            [
                present.repeat_interleave(cfg.tokens_per_modality, dim=1),
                torch.ones(len(queries), 1, device=queries.device, dtype=torch.bool),
            ],
            dim=1,
        )
        # Correct token multiplicity so eight keys do not multiply prior mass.
        prior = torch.cat(
            [
                mixture[:, :-1].repeat_interleave(cfg.tokens_per_modality, dim=1)
                / cfg.tokens_per_modality,
                mixture[:, -1:],
            ],
            dim=1,
        )
        log_prior = prior.clamp_min(1e-8).log().masked_fill(prior == 0, float("-inf"))
        for block in self.blocks:
            queries = block(queries, memory, valid, log_prior)
        return queries

    def forward(self, sensors, question_embeddings, question_mask):
        cfg = self.config
        question = self.question(question_embeddings, question_mask)
        tokens, summaries, present = [], [], []
        for modality in cfg.modalities:
            d = sensors[modality]
            if modality in self.frontends:
                d = self.frontends[modality](d)
            mask = d["mask"].bool()
            observed = mask.any(-1)
            quality = d["quality"].float().masked_fill(~observed, 0)
            if not torch.isfinite(quality).all():
                raise ValueError("Nonfinite observed sensor quality")
            tok, availability = self.adapters[modality](
                d["values"],
                mask,
                quality,
                d["times"],
                d["duration"],
                d.get("positions"),
            )
            coverage = mask.float().mean((1, 2))
            qmean = (quality * observed).sum(1) / observed.sum(1).clamp_min(1)
            # Explicit third summary is validity of the final observation bin.
            info = torch.stack([coverage, qmean, observed[:, -1].float()], -1)
            if self.training and cfg.modality_dropout:
                availability = availability & (
                    torch.rand(len(tok), device=tok.device) >= cfg.modality_dropout
                )
            tokens.append(tok)
            summaries.append(info)
            present.append(availability)
        present = torch.stack(present, 1)
        tokens = torch.stack(tokens, 1)
        info = torch.stack(summaries, 1)
        reliability = (
            self.gating(tokens, question, info, present)
            if self.gating is not None
            else present.float()
        )
        mixture = mixture_weights(reliability, present, cfg.use_reliability_gates)
        tokens = (tokens + self.type_embeddings[None, :, None]).masked_fill(
            ~present[:, :, None, None], 0
        )
        # Retain stream-specific evidence before the shared bottleneck mixes it.
        # This optional path uses the existing projection; it adds no parameters.
        readout = None
        if cfg.readout_tokens_per_modality:
            b, m, k, d = tokens.shape
            pooled = (
                F.adaptive_avg_pool1d(
                    tokens.reshape(b * m, k, d).transpose(1, 2),
                    cfg.readout_tokens_per_modality,
                )
                .transpose(1, 2)
                .reshape(b, m, cfg.readout_tokens_per_modality, d)
            )
            readout = self.output(pooled)
            readout = readout * reliability[:, :, None, None].to(readout.dtype)
            readout = readout.masked_fill(~present[:, :, None, None], 0).flatten(1, 2)
        queries = self.queries[None].expand(len(question), -1, -1) + question[:, None]
        if cfg.fusion == "bottleneck":
            for block in self.blocks:
                tokens, queries = block(tokens, queries, present, mixture)
        else:
            queries = self._cross_attention(tokens, queries, present, mixture)
        self.last_gate_weights = reliability.detach()
        self.last_mixture_weights = mixture.detach()
        self.last_modality_available = present.detach()
        result = self.output(queries)
        # All missing or all dropped: zero sensor contribution, no NaN or fabricated evidence.
        result = result.masked_fill(~present.any(1)[:, None, None], 0)
        return torch.cat([result, readout], dim=1) if readout is not None else result
