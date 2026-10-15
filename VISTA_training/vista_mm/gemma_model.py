"""VISTA slots injected before Gemma 3's native image merge and causal masks."""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Gemma3ForConditionalGeneration
from .config import SensorConfig
from .fusion import ModalityAdapter, SensorFusion


class VistaGemma3(Gemma3ForConditionalGeneration):
    @staticmethod
    def _fix_state_dict_key_on_load(key):
        if key.startswith("sensor_encoder.frontends.audio.network.") and key.endswith(
            ("weight_g", "weight_v")
        ):
            return key, False
        return Gemma3ForConditionalGeneration._fix_state_dict_key_on_load(key)

    def __init__(self, config):
        super().__init__(config)
        cfg = SensorConfig(**config.vista_sensor)
        self.sensor_encoder = SensorFusion(config.text_config.hidden_size, cfg)
        for module in self.sensor_encoder.modules():
            module._vista_sensor_module = True
        self.sensor_encoder.apply(self._init_sensor_weights)
        if self.sensor_encoder.frontends:
            self._keep_in_fp32_modules_strict = ["frontends"]
            self._tied_weights_keys = list(self._tied_weights_keys or []) + [
                r"sensor_encoder.frontends.audio.network.encoder.layers.\d+.self_attn.relative_attention_bias.weight"
            ]

    def _init_weights(self, module):
        if getattr(module, "_vista_sensor_module", False):
            self._init_sensor_weights(module)
        else:
            super()._init_weights(module)

    @staticmethod
    def _init_sensor_weights(module):
        # Same 0.02 sensor initialization as VISTA's other backbone families;
        # Gemma's much larger text initializer does not apply to sensor adapters.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
        if isinstance(module, (SensorFusion, ModalityAdapter)):
            for p in module.parameters(recurse=False):
                nn.init.normal_(p, mean=0, std=0.02)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        token_type_ids=None,
        inputs_embeds=None,
        logits_to_keep=0,
        sensors=None,
        question_input_ids=None,
        question_attention_mask=None,
        **kwargs,
    ):
        common = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        cfg = self.sensor_encoder.config
        slots = input_ids.eq(cfg.sensor_token_id) if input_ids is not None else None
        if sensors is None:
            if slots is not None and slots.any():
                raise ValueError("Sensor slots require sensor tensors")
            return super().forward(**common)
        if input_ids is None or inputs_embeds is not None:
            raise ValueError("Sensor injection requires input_ids")
        if not torch.all(slots.sum(1) == cfg.input_tokens):
            raise ValueError("Wrong number of sensor slots")
        if question_input_ids is None or question_attention_mask is None:
            raise ValueError("Question-only tokens required")
        embedding = self.get_input_embeddings()
        question = F.embedding(question_input_ids, embedding.weight)
        values = self.sensor_encoder(sensors, question, question_attention_mask)

        def inject(module, args, output):
            expected = input_ids
            if self.config.image_token_id >= self.model.vocab_size:
                expected = input_ids.masked_fill(
                    input_ids.eq(self.config.image_token_id), 0
                )
            if not torch.equal(args[0], expected):
                raise RuntimeError("Unexpected Gemma embedding call")
            result = output.clone()
            scale = getattr(module, "embed_scale", 1.0)
            result[slots] = (values.to(output.dtype) * scale).reshape(
                -1, output.shape[-1]
            )
            return result

        handle = embedding.register_forward_hook(inject)
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
        result = super().prepare_inputs_for_generation(input_ids, **kwargs)
        ids = result.get("input_ids")
        if ids is not None and ids.eq(self.sensor_encoder.config.sensor_token_id).any():
            result.update(
                sensors=sensors,
                question_input_ids=question_input_ids,
                question_attention_mask=question_attention_mask,
            )
        return result

    def generate(self, *args, **kwargs):
        cfg = kwargs.get("generation_config", self.generation_config)
        if (
            kwargs.get("num_beams", cfg.num_beams) != 1
            or kwargs.get("num_return_sequences", cfg.num_return_sequences) != 1
        ):
            raise ValueError(
                "Sensor decoding supports one beam and one return sequence"
            )
        return super().generate(*args, **kwargs)
