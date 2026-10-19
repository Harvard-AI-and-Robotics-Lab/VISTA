# VISTA Open-Source Model Setup

Use Linux/Bash, Python 3.12, and a CUDA-capable NVIDIA GPU. Run these commands
from the repository root. For paid API models, use the main README's
[`venv_api` setup](../README.md#environment-setup).

## Environment Setup

Install one environment for your selected model:

| Models | Environment | Requirements |
|---|---|---|
| Qwen3-VL and Gemma 3 (vLLM) | `venv_qwen` | [requirements_qwen.txt](requirements/requirements_qwen.txt) |
| InternVL3 1B / 2B / 8B / 14B | `venv_internvl3` | [requirements_internvl3.txt](requirements/requirements_internvl3.txt) |
| InternVL3 9B | `venv_internvl3_9b` | [requirements_internvl3_9b.txt](requirements/requirements_internvl3_9b.txt) |
| VideoLLaMA3 | `venv_videollama` | [requirements_videollama.txt](requirements/requirements_videollama.txt) |
| LLaVA-Video | `venv_llava` | See below |

Set the environment and requirements from the table (Qwen example):

```bash
VISTA_ENV=venv_qwen
VISTA_REQUIREMENTS=requirements_qwen.txt
python3.12 -m venv "VISTA_benchmark/venvs/$VISTA_ENV"
"VISTA_benchmark/venvs/$VISTA_ENV/bin/python" -m pip install -U pip setuptools wheel
"VISTA_benchmark/venvs/$VISTA_ENV/bin/python" -m pip install \
  -r "VISTA_benchmark/requirements/$VISTA_REQUIREMENTS" httpx
```

Configure `.env` and launch your model using the [main run instructions](../README.md#run).
The launcher selects its environment automatically. Set `HF_TOKEN` for gated
weights after accepting their access terms; `CUDA_VISIBLE_DEVICES` selects GPUs.

The historical `qwen3vl_32b` CLI key loads `Qwen/Qwen3-VL-30B-A3B-Instruct`.
See the [model table](../README.md#supported-models) for all CLI keys.

## LLaVA-Video

```bash
python3.12 -m venv VISTA_benchmark/venvs/venv_llava
source VISTA_benchmark/venvs/venv_llava/bin/activate
```

Install a compatible [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) version
using its LLaVA-Video instructions, then add the runner dependencies:

```bash
python -m pip install httpx opencv-python-headless pandas Pillow requests tqdm
```
