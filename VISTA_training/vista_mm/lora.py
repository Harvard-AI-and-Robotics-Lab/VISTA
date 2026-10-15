"""Compact adapters reuse immutable backbone and native encoder assets."""

import json
import copy
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file

from peft import LoraConfig, PeftModelForCausalLM
from torch.nn.utils.weight_norm import WeightNorm

from .encoders import NativeFrontend


def prepare_native_copy(model):
    """Make weight-norm caches copyable by PEFT's modules_to_save.

    The actual weight_g/weight_v parameters are unchanged. WeightNorm recomputes
    the derived weight at every forward; its cached non-leaf tensor otherwise
    prevents deepcopy of a freshly initialized BEATs frontend.
    """
    for frontend in model.modules():
        if isinstance(frontend, NativeFrontend):
            for module in frontend.modules():
                for hook in module._forward_pre_hooks.values():
                    if isinstance(hook, WeightNorm):
                        setattr(module, hook.name, getattr(module, hook.name).detach())


class VistaPeftModel(PeftModelForCausalLM):
    @classmethod
    def from_pretrained(cls, model, model_id, is_trainable=False):
        # PEFT's factory dispatches by task_type and otherwise drops subclasses.
        config = LoraConfig.from_pretrained(model_id)
        config.inference_mode = not is_trainable
        prepare_native_copy(model)
        wrapped = cls(model, config)
        compact = Path(model_id) / "compact.json"
        if compact.exists():
            initialized = get_peft_model_state_dict(
                wrapped, adapter_name="default", save_embedding_layers=False
            )
            weights = load_file(str(Path(model_id) / "adapter_model.safetensors"))
            expected = {k for k in initialized if ".frontends." not in k}
            if set(weights) != expected:
                raise ValueError(
                    f"Compact adapter keys differ: missing={expected - set(weights)}, unexpected={set(weights) - expected}"
                )
            weights.update({k: v for k, v in initialized.items() if ".frontends." in k})
            result = set_peft_model_state_dict(wrapped, weights, adapter_name="default")
            missing = [
                k
                for k in result.missing_keys
                if "lora_" in k or ".modules_to_save.default." in k
            ]
            wrapped._freeze_frontends()
            if not is_trainable:
                wrapped.eval()
        else:
            result = wrapped.load_adapter(
                model_id, "default", is_trainable=is_trainable, torch_device="cpu"
            )
            missing = list(result.missing_keys)
        if compact.exists():
            metadata = json.loads(compact.read_text())
            if metadata["format"] != "vista-frozen-assets-v1":
                raise ValueError("Unknown compact adapter format")
            missing = [k for k in missing if ".frontends." not in k]
            if metadata["sensor_token_id"] is not None:
                rows = load_file(str(Path(model_id) / "sensor_token.safetensors"))
                indices = metadata["embedding_indices"]
                core = wrapped.get_base_model()
                with torch.no_grad():
                    for name, layer in (
                        ("input", core.get_input_embeddings()),
                        ("output", core.get_output_embeddings()),
                    ):
                        layer.weight[indices] = rows[name].to(layer.weight)
        if missing or result.unexpected_keys:
            raise ValueError(
                f"Incomplete LoRA checkpoint: missing={missing}, unexpected={result.unexpected_keys}"
            )
        return wrapped

    def _freeze_frontends(self):
        for module in self.modules():
            if isinstance(module, NativeFrontend):
                module.set_trainable(False)

    def set_adapter(self, *args, **kwargs):
        result = super().set_adapter(*args, **kwargs)
        self._freeze_frontends()
        return result

    def load_adapter(self, *args, **kwargs):
        result = super().load_adapter(*args, **kwargs)
        self._freeze_frontends()
        return result

    def save_pretrained(self, save_directory, *args, **kwargs):
        core = self.get_base_model()
        for name, parameter in self.named_parameters():
            if (
                ".frontends." in name or "embed_tokens" in name or "lm_head" in name
            ) and parameter.requires_grad:
                raise ValueError(f"Compact saves require frozen shared weights: {name}")
        state = kwargs.pop("state_dict", None)
        if state is None:
            state = self.state_dict()
        if not kwargs.get("is_main_process", True):
            return
        if set(self.peft_config) != {"default"}:
            raise ValueError("Compact checkpoint requires exactly one default adapter")
        state = get_peft_model_state_dict(
            self, state_dict=state, adapter_name="default", save_embedding_layers=False
        )
        state = {
            k: v.detach().cpu().contiguous()
            for k, v in state.items()
            if ".frontends." not in k
        }
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        save_file(
            state, str(root / "adapter_model.safetensors"), metadata={"format": "pt"}
        )
        adapter_config = copy.deepcopy(self.peft_config["default"])
        adapter_config.inference_mode = True
        adapter_config.save_pretrained(root)
        if kwargs.get("is_main_process", True):
            token = getattr(core.config, "vista_sensor", {}).get("sensor_token_id")
            indices = []
            if token is not None:
                old_size = getattr(
                    core.config,
                    "vista_original_vocab_size",
                    len(core.get_input_embeddings().weight),
                )
                indices = sorted(
                    set(
                        [token]
                        + list(range(old_size, len(core.get_input_embeddings().weight)))
                    )
                )
                save_file(
                    {
                        "input": core.get_input_embeddings()
                        .weight[indices]
                        .detach()
                        .cpu()
                        .clone()
                        .contiguous(),
                        "output": core.get_output_embeddings()
                        .weight[indices]
                        .detach()
                        .cpu()
                        .clone()
                        .contiguous(),
                    },
                    str(root / "sensor_token.safetensors"),
                )
            (root / "compact.json").write_text(
                json.dumps(
                    {
                        "format": "vista-frozen-assets-v1",
                        "sensor_token_id": token,
                        "embedding_indices": indices,
                        "reused": "Pinned pretrained backbone and verified BEATs/PRIMUS assets; embeddings frozen, all added token rows saved separately",
                    },
                    indent=2,
                )
            )
            core.config.save_pretrained(root)
            core.generation_config.save_pretrained(root)
