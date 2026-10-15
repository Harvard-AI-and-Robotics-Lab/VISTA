# RGB + SLAM ablation

The frozen models receive 8 RGB frames followed by 4 left/right SLAM pairs (16 images total). Both SLAM cameras use the same four uniformly spaced positions on their common exported-video timeline. The original RGB sampler and task prompts are reused; an input description identifies the auxiliary cameras and ordering.

Expected layout for each clip:

```
test/TC01/TC01_001/TC01_001.mp4
test/TC01/TC01_001/vrs/slam_left.mp4
test/TC01/TC01_001/vrs/slam_right.mp4
```

The sampler checks FPS, frame availability, and decoded presentation timestamps. These timestamps describe the exported MP4s: without device-clock SLAM timestamp sidecars, matching them does not prove capture-time synchronization if records were dropped. The manifest records both frame counts and this limitation.

## Run

Use the model environments described in the benchmark README. For this SLAM runner, use OpenCV 4.13.0 (validated on the exported videos) and run the preflight in each model environment. OpenCV 4.11.0 returned negative presentation timestamps for these MP4s; the sampler correctly rejects those timestamps instead of claiming synchronization. The queue expects `VISTA_VENVS_ROOT/<environment>/bin/python` (defaults to `VISTA_benchmark/venvs`). InternVL3-9B uses `venv_internvl3_9b`.

```bash
cd VISTA_benchmark
python launch_rgb_slam.py \
  --dataset_root /path/to/VISTA/test \
  --csv_path /path/to/VISTA/test/test.csv \
  --output_dir outputs/rgb8_slam8
```

The default queue includes Qwen3-VL 2B/4B/8B, InternVL3 1B/2B/8B/9B, and VideoLLaMA3-7B. LLaVA is not included in this queue. `--models` selects a subset; `--limit` enables a small run. `--max_parallel`, `--min_free_mib`, and `--poll_seconds` configure GPU scheduling. The scheduler requires Linux, NVIDIA tools, and Python environments installed before launch.

```bash
python launch_rgb_slam.py --output_dir outputs/rgb8_slam8 --status
python run_rgb_slam.py --dataset_root /path/to/VISTA/test --preflight_only
```

The queue supports resumption, tracks workers, and validates prediction coverage. Outputs and logs are separate from RGB-only results. Never combine outputs generated with different code or decoding settings: use a new output directory for each protocol. This checkout uses the shared decoding settings in `models.py`; historical runs from the older local checkout used greedy decoding and are a different experiment.

## LLaVA zero-shot input correction

The RGB-only LLaVA-Video path now uses the `qwen_1_5` conversation template, a single visual marker, one tensor containing all eight RGB frames, and `modalities=["video"]`. The checkpoint remains `lmms-lab/LLaVA-Video-7B-Qwen2`. Existing repository decoding settings are preserved. Predictions and evaluation scores from the previous multi-image input path must be regenerated before comparison with this corrected path.
