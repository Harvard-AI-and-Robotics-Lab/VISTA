"""Qwen3-VL integration that retains native video tokens, mRoPE and DeepStack.

A scoped embedding hook replaces only dedicated sensor slots. input_ids still go
through upstream forward/generation, so video positions never need reimplementation.
Pinned to the inspected Transformers 4.57.6 implementation; eager execution only.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3VLForConditionalGeneration

from .config import SensorConfig
from .fusion import ModalityAdapter, SensorFusion

SENSOR_TOKEN = "<|vista_sensor|>"


class VistaQwen3VL(Qwen3VLForConditionalGeneration):
    @staticmethod
    def _fix_state_dict_key_on_load(key):
        # BEATs intentionally retains the original weight_norm module.
        # Transformers otherwise rewrites its keys to a different parametrization.
        if key.startswith("sensor_encoder.frontends.audio.network.") and key.endswith(
            ("weight_g", "weight_v")
        ):
            return key, False
        return Qwen3VLForConditionalGeneration._fix_state_dict_key_on_load(key)

    def __init__(self, config):
        super().__init__(config)
        cfg = SensorConfig(**getattr(config, "vista_sensor", {}))
        self.sensor_encoder = SensorFusion(config.text_config.hidden_size, cfg)
        self.sensor_encoder.apply(self._init_weights)
        if self.sensor_encoder.frontends:
            # Set after Qwen post_init: these modules do not exist during super().__init__.
            self._keep_in_fp32_modules_strict = ["frontends"]
            self._tied_weights_keys = list(self._tied_weights_keys or []) + [
                r"sensor_encoder.frontends.audio.network.encoder.layers.\d+.self_attn.relative_attention_bias.weight"
            ]

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
        if isinstance(module, (SensorFusion, ModalityAdapter)):
            for parameter in module.parameters(recurse=False):
                nn.init.normal_(parameter, mean=0, std=0.02)

    def _sensor_embeddings(self, sensors, question_input_ids, question_attention_mask):
        if question_input_ids is None or question_attention_mask is None:
            raise ValueError(
                "Question-only tokens are required for sensor conditioning"
            )
        # Direct lookup avoids the scoped replacement hook and never reads answer tokens.
        question_embeddings = F.embedding(
            question_input_ids, self.get_input_embeddings().weight
        )
        return self.sensor_encoder(
            sensors, question_embeddings, question_attention_mask
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        logits_to_keep=0,
        sensors=None,
        question_input_ids=None,
        question_attention_mask=None,
        **kwargs,
    ):
        config = self.sensor_encoder.config
        common = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        if config.sensor_token_id is None:
            raise ValueError("Sensor token ID has not been registered")
        slots = input_ids.eq(config.sensor_token_id) if input_ids is not None else None
        has_slots = slots is not None and bool(slots.any())
        if sensors is None:
            if has_slots:
                raise ValueError("Sensor slots were supplied without sensor tensors")
            return super().forward(**common)
        if inputs_embeds is not None or input_ids is None:
            raise ValueError(
                "Sensor injection requires input_ids, preserving native multimodal positions"
            )
        if not torch.all(slots.sum(1) == config.input_tokens):
            raise ValueError(
                f"Each prompt must contain exactly {config.input_tokens} sensor slots"
            )
        embeddings = self._sensor_embeddings(
            sensors, question_input_ids, question_attention_mask
        )

        def inject(module, args, output):
            ids = args[0]
            if ids.shape != input_ids.shape or not torch.equal(ids, input_ids):
                raise RuntimeError(
                    "Unexpected Qwen embedding call; check pinned Transformers version"
                )
            result = output.clone()
            result[slots] = embeddings.to(output.dtype).reshape(-1, output.shape[-1])
            return result

        handle = self.get_input_embeddings().register_forward_hook(inject)
        try:
            return super().forward(**common)
        finally:
            handle.remove()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        sensors=None,
        question_input_ids=None,
        question_attention_mask=None,
        **kwargs,
    ):
        prepared = super().prepare_inputs_for_generation(input_ids, **kwargs)
        if (prepared["input_ids"] == self.sensor_encoder.config.sensor_token_id).any():
            prepared.update(
                sensors=sensors,
                question_input_ids=question_input_ids,
                question_attention_mask=question_attention_mask,
            )
        return prepared

    def generate(self, *args, **kwargs):
        cfg = kwargs.get("generation_config", self.generation_config)
        if (
            kwargs.get("num_beams", cfg.num_beams) != 1
            or kwargs.get("num_return_sequences", cfg.num_return_sequences) != 1
        ):
            raise ValueError(
                "This audited sensor generation path supports one beam and one return sequence"
            )
        return super().generate(*args, **kwargs)
