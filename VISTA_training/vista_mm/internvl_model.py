"""VISTA sensor slots for the native Transformers InternVL3 implementation.

Keep InternVL's image/video embedding and cached decoding paths intact. Only
dedicated sensor token embeddings are replaced, before native visual injection.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import InternVLForConditionalGeneration

from .config import SensorConfig
from .fusion import ModalityAdapter, SensorFusion


class VistaInternVL(InternVLForConditionalGeneration):
    @staticmethod
    def _fix_state_dict_key_on_load(key):
        # Preserve BEATs' original weight_norm keys when restoring checkpoints.
        if key.startswith("sensor_encoder.frontends.audio.network.") and key.endswith(
            ("weight_g", "weight_v")
        ):
            return key, False
        return InternVLForConditionalGeneration._fix_state_dict_key_on_load(key)

    def __init__(self, config):
        super().__init__(config)
        cfg = SensorConfig(**getattr(config, "vista_sensor", {}))
        self.sensor_encoder = SensorFusion(config.text_config.hidden_size, cfg)
        self.sensor_encoder.apply(self._init_weights)
        if self.sensor_encoder.frontends:
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

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        labels=None,
        cache_position=None,
        logits_to_keep=0,
        image_sizes=None,
        sensors=None,
        question_input_ids=None,
        question_attention_mask=None,
        **kwargs,
    ):
        common = dict(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            labels=labels,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            image_sizes=image_sizes,
            **kwargs,
        )
        cfg = self.sensor_encoder.config
        if cfg.sensor_token_id is None:
            raise ValueError("Sensor token ID has not been registered")
        slots = input_ids.eq(cfg.sensor_token_id) if input_ids is not None else None
        if sensors is None:
            if slots is not None and bool(slots.any()):
                raise ValueError("Sensor slots were supplied without sensor tensors")
            return super().forward(**common)
        if inputs_embeds is not None or input_ids is None:
            raise ValueError("Sensor injection requires input_ids")
        if not torch.all(slots.sum(1) == cfg.input_tokens):
            raise ValueError(
                f"Each prompt must contain exactly {cfg.input_tokens} sensor slots"
            )
        if question_input_ids is None or question_attention_mask is None:
            raise ValueError(
                "Question-only tokens are required for sensor conditioning"
            )
        question = F.embedding(question_input_ids, self.get_input_embeddings().weight)
        embeddings = self.sensor_encoder(sensors, question, question_attention_mask)

        def inject(module, args, output):
            if args[0].shape != input_ids.shape or not torch.equal(args[0], input_ids):
                raise RuntimeError("Unexpected InternVL embedding call")
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
        ids = prepared.get("input_ids")
        if ids is not None and bool(
            ids.eq(self.sensor_encoder.config.sensor_token_id).any()
        ):
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
                "Sensor generation supports one beam and one return sequence"
            )
        return super().generate(*args, **kwargs)
