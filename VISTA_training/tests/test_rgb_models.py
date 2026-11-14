"""Small real architectures exercise RGB gradients and adapter reloads offline."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import (
    InternVLConfig, InternVLForConditionalGeneration,
    Qwen3VLConfig, Qwen3VLForConditionalGeneration,
    Gemma3Config, Gemma3ForConditionalGeneration,
    TrainingArguments,
)

from vista_mm.config import Experiment
from vista_mm.train import load_backbone, configure_trainable, VistaTrainer


def tiny_backbone(family):
    text = dict(vocab_size=128, hidden_size=64, intermediate_size=128,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                head_dim=16, max_position_embeddings=256, pad_token_id=0,
                bos_token_id=1, eos_token_id=2)
    vision = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                  num_attention_heads=4, image_size=28, patch_size=14)
    if family == "internvl":
        config = InternVLConfig(text_config=dict(text, model_type="qwen2"),
                                vision_config=vision, image_token_id=120, image_seq_length=1)
        return InternVLForConditionalGeneration(config), config
    if family == "qwen3_vl":
        text["rope_scaling"] = dict(rope_type="default", mrope_section=[2, 2, 4])
        vision = dict(depth=1, hidden_size=32, intermediate_size=64, out_hidden_size=64,
                      num_heads=4, patch_size=2, temporal_patch_size=2,
                      spatial_merge_size=2, deepstack_visual_indexes=[], num_position_embeddings=16)
        config = Qwen3VLConfig(text_config=text, vision_config=vision,
                               image_token_id=120, video_token_id=121,
                               vision_start_token_id=122, vision_end_token_id=123)
        return Qwen3VLForConditionalGeneration(config), config
    text.update(sliding_window=32, sliding_window_pattern=2)
    config = Gemma3Config(text_config=text, vision_config=vision, mm_tokens_per_image=4,
                          image_token_index=120, boi_token_index=121, eoi_token_index=122)
    return Gemma3ForConditionalGeneration(config), config


def visual_batch(family, device):
    if family == "internvl":
        ids = [1] + [120] * 8 + [4, 5, 6, 2]
        visual = dict(pixel_values=torch.randn(8, 3, 28, 28, device=device, dtype=torch.bfloat16))
    elif family == "qwen3_vl":
        ids = [1] + [122, 120, 123] * 8 + [4, 5, 6, 2]
        visual = dict(pixel_values=torch.randn(32, 24, device=device, dtype=torch.bfloat16),
                      image_grid_thw=torch.tensor([[1, 2, 2]] * 8, device=device))
    else:
        ids = [1] + ([121] + [120] * 4 + [122]) * 8 + [4, 5, 6, 2]
        visual = dict(pixel_values=torch.randn(8, 3, 28, 28, device=device, dtype=torch.bfloat16),
                      token_type_ids=torch.tensor([[int(i == 120) for i in ids]], device=device))
    inputs = torch.tensor([ids], device=device)
    labels = torch.full_like(inputs, -100)
    labels[:, -3:] = inputs[:, -3:]
    return dict(input_ids=inputs, attention_mask=torch.ones_like(inputs), labels=labels, **visual)


@pytest.mark.gpu
@pytest.mark.parametrize("family", ["internvl", "qwen3_vl", "gemma3"])
def test_rgb_updates_only_lora_and_reloads_for_stage_two(family, tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the BF16 training path")
    import vista_mm.train as training
    base, _ = tiny_backbone(family)
    base_path = tmp_path / "base"
    base.save_pretrained(base_path)
    del base
    tokenizer = SimpleNamespace(pad_token_id=0, padding_side="right")
    processor = SimpleNamespace(tokenizer=tokenizer)
    monkeypatch.setattr(training.AutoProcessor, "from_pretrained", lambda *a, **kw: processor)
    monkeypatch.setattr(training.SynchronizedQwenProcessor, "from_pretrained", lambda *a, **kw: processor)
    cfg = Experiment(model_id=str(base_path), revision="local-fixture", input_mode="rgb",
                     train_mode="lora", sensor={"modalities": []}, lora_rank=4, lora_alpha=8,
                     lora_dropout=0, gradient_checkpointing=False, max_steps=2)
    model, _ = load_backbone(cfg)
    model = configure_trainable(model, cfg).cuda()
    assert not hasattr(model.get_base_model(), "sensor_encoder")
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("language_model.layers." in n and "lora_" in n for n in trainable)
    assert not any(p.requires_grad for n, p in model.named_parameters()
                   if "vision" in n or "visual" in n or "projector" in n)
    before = {n: p.detach().clone() for n, p in trainable.items()}
    args = TrainingArguments(output_dir=str(tmp_path / "trainer"), report_to=[], bf16=True)
    trainer = VistaTrainer(model=model, args=args, sensor_lr=3e-4)
    optimizer = trainer.create_optimizer()
    batch = visual_batch(family, "cuda")
    model.train()
    for _ in range(2):
        optimizer.zero_grad()
        loss = trainer.compute_loss(model, batch)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        optimizer.step()
    assert any(not torch.equal(before[n], p) for n, p in trainable.items())
    model.eval()
    forward = {k: v for k, v in batch.items() if k != "labels"}
    with torch.no_grad():
        expected = model(**forward).logits.cpu()
    adapter = tmp_path / "stage1/adapter"
    model.save_pretrained(adapter)
    assert not (adapter / "sensor_token.safetensors").exists()
    del model, trainer, optimizer
    torch.cuda.empty_cache()
    restored, _ = load_backbone(cfg, checkpoint=adapter, training=False)
    restored = restored.cuda().eval()
    with torch.no_grad():
        actual = restored(**forward).logits.cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    assert not any(p.requires_grad for p in restored.parameters())
    del restored
    torch.cuda.empty_cache()
    cfg.initial_adapter = str(adapter)
    stage2, _ = load_backbone(cfg, training=True)
    stage2 = configure_trainable(stage2, cfg).cuda()
    loss = stage2(**forward, labels=batch["labels"]).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for n, p in stage2.named_parameters() if "lora_" in n)
