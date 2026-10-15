#model.py vista

import os
import base64
import hashlib
import json
import random
import re
import sys
import time
import cv2
import httpx
import numpy as np
import requests
from PIL import Image


MAX_OUTPUT_TOKENS = 4096

# ---------------------------------------------------------------------------------------
# DECODING PROTOCOL -- one rule for every benchmarked model, restated at every call site.
#
#   temperature = 0.7,  top_p = 0.8,  repetition_penalty = 1.05,
#   seed = 0 PER ITEM (hashed with clip_id|task_type; see SEED_MODE),  max output tokens = 4096
#
# Restated everywhere because the defaults disagree, and inheriting them silently produced
# three regimes inside one results table: vLLM's SamplingParams defaults to temperature=1.0 /
# top_p=1.0 / top_k=0 unseeded AND overrides the model's own generation_config, while
# transformers falls back to that config -- do_sample=True at 0.7/0.8 for VideoLLaMA3 and
# Qwen3-VL, unset (so greedy) for InternVL3.
#
# Chosen by measurement, not by argument. Qwen3-VL degenerates into repetition loops that run
# to the token cap, and the rate depends on temperature MONOTONICALLY in the unhelpful
# direction -- lower temperature loops more (qwen3vl_4b, share of outputs at the cap):
#
#                        no context   full sensor context
#   greedy                    --              23%
#   T=0.7 p=0.8              9.6%             14.0%
#   T=1.0 p=1.0              0.0%              5.5%
#   T=0.7 p=0.8 rep=1.05     0.0%              1.0%   (and 0% of those are loops)
#   T=0.7 p=0.8 rep=1.10     0.0%              0.0%
#
# So greedy is the worst option here, not the safe one, and no temperature alone fixes it on
# the context-injected arms. The repetition penalty does. 1.05 is the weakest setting that
# removes every loop, chosen over 1.10 because this task has legitimate repetition (naming the
# same object across successive guidance steps) and the mildest sufficient intervention
# distorts it least. The per-item seed is what makes sampling reproducible.
# See VISTA_mm/docs/DECODING.md.
# Overridable from the environment so the protocol can be swept without editing code -- the
# repetition behaviour below had to be measured across several settings before one was chosen.
TEMPERATURE = float(os.environ.get("VISTA_TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("VISTA_TOP_P", "0.8"))
SEED = int(os.environ.get("VISTA_SEED", "0"))
# 1.05, the weakest setting measured to eliminate Qwen3-VL's repetition loops entirely.
REPETITION_PENALTY = float(os.environ.get("VISTA_REPETITION_PENALTY", "1.05"))

# SEED MODE -- how the base SEED becomes the seed an individual item is sampled with.
#
#   "item"   (default) seed = hash(SEED, clip_id|task_type). Reproducible, independent across
#            items, and identical for the same item in every ablation arm, so arm deltas stay
#            paired on common random numbers.
#   "shared" every item sampled with SEED itself. This is what the first full campaign ran
#            with; it is kept ONLY so those frozen results can be reproduced exactly.
#
# Why "shared" was wrong: reseeding every item to one value hands every item the same random
# draws, so sampling decisions are coupled across the whole benchmark rather than independent.
# A run can then land on one side of a model's output distribution everywhere at once.
# Measured on VideoLLaMA3 (20 clips, 79 items, same prompts): refusals 62.0% with a shared seed
# vs 20.3% with per-item seeds -- and 19.0% unseeded. The repetition penalty made no difference
# (59.5% shared without it). See VISTA_mm/docs/DECODING.md.
SEED_MODE = os.environ.get("VISTA_SEED_MODE", "item").strip().lower()
if SEED_MODE not in ("item", "shared"):
    raise ValueError(f"VISTA_SEED_MODE must be 'item' or 'shared', got {SEED_MODE!r}")

_warned_no_key = False


def item_seed(seed_key=None) -> int:
    """The seed to sample one item with, under SEED_MODE.

    hashlib, not hash(): Python's hash() of a str is salted per process, so it would give a
    resumed run -- a new process -- different seeds for the same items.
    """
    global _warned_no_key
    if SEED_MODE == "shared":
        return SEED
    if seed_key is None:
        if not _warned_no_key:
            import sys as _sys
            print("[WARNING] run_model() called without seed_key under VISTA_SEED_MODE=item: "
                  "falling back to one shared seed, which couples sampling across items. "
                  "Pass seed_key=f'{clip_id}|{task_type}'.", file=_sys.stderr, flush=True)
            _warned_no_key = True
        return SEED
    digest = hashlib.sha256(f"{SEED}|{seed_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def decoding_signature() -> dict:
    """Everything that determines what a model generates, as recorded next to its predictions."""
    return {"temperature": TEMPERATURE, "top_p": TOP_P, "repetition_penalty": REPETITION_PENALTY,
            "seed": SEED, "seed_mode": SEED_MODE, "max_output_tokens": MAX_OUTPUT_TOKENS}


def guard_decoding(out_dir: str, name: str, n_existing: int) -> None:
    """Refuse to append to predictions that were generated under different decoding.

    Resume appends to an existing predictions file, so a run started with changed settings
    would silently finish someone else's file and produce a row generated under two protocols.
    A `decoding_<name>.json` sidecar records the settings the file was started with:

      * sidecar present and different  -> exit
      * no sidecar but predictions exist -> exit (cannot prove what made them)
      * fresh file                       -> write the sidecar
    """
    path = os.path.join(out_dir, f"decoding_{name}.json")
    sig = decoding_signature()
    if os.path.exists(path):
        with open(path) as f:
            prev = json.load(f)
        if prev != sig:
            raise SystemExit(f"[decoding guard] {path} was generated with {prev}, this run uses "
                             f"{sig}. Refusing to mix protocols in one file -- write to a "
                             f"different output directory (e.g. set VISTA_RUN_TAG).")
    elif n_existing > 0:
        raise SystemExit(f"[decoding guard] {n_existing} predictions for {name} already exist in "
                         f"{out_dir} with no {os.path.basename(path)} recording how they were "
                         f"made. Refusing to resume into them under {sig}.")
    else:
        os.makedirs(out_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(sig, f, indent=2)


def _seed_torch(seed: int):
    """Seed immediately before each generate() on the transformers paths.

    Seeding once at startup would make every item's text depend on how many items ran before
    it, so a resumed run -- which processes fewer -- would produce different output for the
    same clip and quietly break the resume guarantee. Per call, with the item's own seed, each
    item is reproducible from its identity alone. The vLLM paths pass the same per-item seed
    in SamplingParams.
    """
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


############################################
# FRAME EXTRACTION
############################################

def extract_frames(video_path, num_frames=8):
    """Extract num_frames uniformly sampled frames from the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * total_frames / (num_frames - 1)) for i in range(num_frames)]
    indices = [min(i, total_frames - 1) for i in indices]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

############################################
# UTILS
############################################

def frame_to_pil(frame):
    """Convert a BGR numpy frame to a PIL Image."""
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def frame_to_jpeg_bytes(frame):
    """Convert a BGR numpy frame to JPEG bytes."""
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise ValueError("Failed to encode frame as JPEG")
    return buffer.tobytes()

def frame_to_base64(frame):
    """Convert a BGR numpy frame to a base64-encoded JPEG string."""
    return base64.b64encode(frame_to_jpeg_bytes(frame)).decode("utf-8")

############################################
# CLOSED SOURCE MODELS
############################################

GEMINI_MODEL_IDS = {
    "gemini_3_1_flash_lite": "gemini-3.1-flash-lite",
}
GEMINI_RETRY_ATTEMPTS = 8
GEMINI_RETRY_STATUS_CODES = [408, 429, 500, 502, 503, 504]
GEMINI_RETRY_INITIAL_DELAY = 2.0
GEMINI_RETRY_MAX_DELAY = 60.0


class GeminiDailyQuotaExhaustedError(RuntimeError):
    pass


def _gemini_daily_quota_exhausted(error):
    if getattr(error, "code", None) != 429:
        return False
    details = str(getattr(error, "details", "")).lower()
    return "perday" in details or "per day" in details


def _gemini_server_retry_delay(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() == "retrydelay" and isinstance(item, str):
                match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)s", item)
                if match:
                    return float(match.group(1))
            delay = _gemini_server_retry_delay(item)
            if delay is not None:
                return delay
    elif isinstance(value, list):
        for item in value:
            delay = _gemini_server_retry_delay(item)
            if delay is not None:
                return delay
    return None


def _gemini_retry_delay(error, attempt):
    exponential_delay = min(
        GEMINI_RETRY_INITIAL_DELAY * (2 ** (attempt - 1)),
        GEMINI_RETRY_MAX_DELAY,
    )
    server_delay = _gemini_server_retry_delay(getattr(error, "details", None))
    base_delay = max(exponential_delay, server_delay or 0.0)
    return min(base_delay + random.uniform(0.0, 1.0), GEMINI_RETRY_MAX_DELAY)

def run_gemini(frames, prompt, api_key, model_name="gemini-3.1-flash-lite"):
    if not api_key:
        raise ValueError("A Gemini API key is required for Gemini models")

    from google import genai
    from google.genai import errors, types

    parts = [
        types.Part.from_bytes(data=frame_to_jpeg_bytes(frame), mime_type="image/jpeg")
        for frame in frames
    ]
    parts.append(types.Part.from_text(text=prompt))

    client = genai.Client(api_key=api_key)
    try:
        for attempt in range(1, GEMINI_RETRY_ATTEMPTS + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=parts,
                    config=types.GenerateContentConfig(
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                    ),
                )
                break
            except errors.APIError as error:
                if _gemini_daily_quota_exhausted(error):
                    raise GeminiDailyQuotaExhaustedError(
                        "Gemini daily request quota is exhausted; resume after the "
                        "quota resets or the API project is moved to a paid tier"
                    ) from error
                if error.code not in GEMINI_RETRY_STATUS_CODES:
                    raise
                retry_error = error
            except (httpx.TransportError, requests.exceptions.RequestException) as error:
                retry_error = error

            if attempt == GEMINI_RETRY_ATTEMPTS:
                raise retry_error
            delay = _gemini_retry_delay(retry_error, attempt)
            print(
                f"[WARN] Gemini request failed ({retry_error}); retrying in "
                f"{delay:.1f}s ({attempt + 1}/{GEMINI_RETRY_ATTEMPTS})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)

        result = response.text.strip() if response.text else ""
        if not result:
            raise RuntimeError("Gemini response completed without text output")
        return result
    finally:
        client.close()

def _receive_realtime_event(websocket, expected_type=None, timeout=120):
    """Receive Realtime events until the requested event type arrives."""
    while True:
        event = json.loads(websocket.recv(timeout=timeout))
        if event.get("type") == "error":
            error = event.get("error", {})
            raise RuntimeError(error.get("message") or json.dumps(error))
        if expected_type is None or event.get("type") == expected_type:
            return event

def _extract_realtime_text(response_event, text_deltas):
    """Return streamed text, falling back to the completed response object."""
    streamed_text = "".join(text_deltas).strip()
    if streamed_text:
        return streamed_text

    text_parts = []
    for output_item in response_event.get("response", {}).get("output", []):
        for content in output_item.get("content", []):
            text = content.get("text") or content.get("transcript")
            if text:
                text_parts.append(text)
    return "".join(text_parts).strip()

def run_gpt_realtime(frames, prompt, api_key, model_name="gpt-realtime-mini"):
    """Run one isolated multimodal Realtime request and return text output."""
    if not api_key:
        raise ValueError("An OpenAI API key is required for gpt_realtime_mini")

    from websockets.sync.client import connect

    url = f"wss://api.openai.com/v1/realtime?model={model_name}"
    with connect(
        url,
        additional_headers={"Authorization": f"Bearer {api_key}"},
        open_timeout=30,
        close_timeout=10,
        max_size=None,
    ) as websocket:
        _receive_realtime_event(websocket, "session.created")

        session = {
            "type": "realtime",
            "model": model_name,
            "output_modalities": ["text"],
        }
        websocket.send(json.dumps({"type": "session.update", "session": session}))
        _receive_realtime_event(websocket, "session.updated")

        content = [
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{frame_to_base64(frame)}",
            }
            for frame in frames
        ]
        content.append({"type": "input_text", "text": prompt})
        websocket.send(json.dumps({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user", "content": content},
        }))

        websocket.send(json.dumps({
            "type": "response.create",
            "response": {
                "output_modalities": ["text"],
                "max_output_tokens": MAX_OUTPUT_TOKENS,
            },
        }))

        text_deltas = []
        while True:
            event = _receive_realtime_event(websocket)
            event_type = event.get("type")
            if event_type == "response.output_text.delta":
                text_deltas.append(event.get("delta", ""))
            elif event_type == "response.done":
                response = event.get("response", {})
                if response.get("status") != "completed":
                    details = response.get("status_details") or response.get("error") or response
                    raise RuntimeError(f"Realtime response failed: {details}")
                result = _extract_realtime_text(event, text_deltas)
                if not result:
                    raise RuntimeError("Realtime response completed without text output")
                return result

############################################
# OPEN SOURCE MODELS
############################################

_loaded_model = None
_loaded_processor = None
_loaded_model_name = None

# Gemma 3 settings. NUM_FRAMES must match extract_frames' default: it is what vLLM is told to
# allow per prompt, and a mismatch is rejected at scheduling time rather than silently ignored.
NUM_FRAMES = 8
GEMMA_BACKEND = os.environ.get("VISTA_GEMMA_BACKEND", "vllm").lower()

def load_model(model_key, hf_token=None):
    import torch
    global _loaded_model, _loaded_processor, _loaded_model_name

    model_map = {
        "qwen3vl_2b":    "Qwen/Qwen3-VL-2B-Instruct",
        "qwen3vl_4b":    "Qwen/Qwen3-VL-4B-Instruct",
        "qwen3vl_8b":    "Qwen/Qwen3-VL-8B-Instruct",
        "qwen3vl_32b":   "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "internvl3_1b":  "OpenGVLab/InternVL3-1B",
        "internvl3_2b":  "OpenGVLab/InternVL3-2B",
        "internvl3_8b":  "OpenGVLab/InternVL3-8B",
        "internvl3_9b":  "OpenGVLab/InternVL3-9B",
        "internvl3_14b": "OpenGVLab/InternVL3-14B",
        "llava_video":   "lmms-lab/LLaVA-Video-7B-Qwen2",
        "llama3_video":  "DAMO-NLP-SG/VideoLLaMA3-7B",
        # Gemma 3. These repos are gated: the account must have accepted Google's license and
        # HF_TOKEN must reach the process. scripts/vista_env.sh exports it from the standard
        # login file -- necessary because setting HF_HOME also moves huggingface_hub's token
        # lookup, which otherwise makes an authorised account look like an anonymous one.
        "gemma3_4b":     "google/gemma-3-4b-it",
        "gemma3_12b":    "google/gemma-3-12b-it",
        "gemma3_27b":    "google/gemma-3-27b-it",
    }

    if model_key not in model_map:
        raise ValueError(f"Unknown model key: {model_key}")

    hf_name = model_map[model_key]
    print(f"[INFO] Loading model: {hf_name}")

    max_memory = {}
    for i in range(torch.cuda.device_count()):
        try:
            total    = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            free     = total - reserved
            if free >= 5:
                max_memory[i] = f"{int(free * 0.9)}GiB"
        except:
            pass
    max_memory["cpu"] = "100GiB"

    from transformers import AutoProcessor, AutoModelForCausalLM, AutoModel
    print(f"[INFO] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"[INFO] torch.cuda.device_count() = {torch.cuda.device_count()}")

    if model_key == "llava_video":
        from llava.model.builder import load_pretrained_model
        n_gpus = torch.cuda.device_count()
        mm = {i: "10GiB" for i in range(n_gpus)}
        mm["cpu"] = "100GiB"
        tokenizer, model, image_processor, _ = load_pretrained_model(
            hf_name, None, "llava_qwen", device_map="auto",
            attn_implementation="eager",
            max_memory=mm
        )
        _loaded_model = model
        _loaded_processor = (tokenizer, image_processor)

    elif model_key in ("qwen3vl_2b", "qwen3vl_4b", "qwen3vl_8b", "qwen3vl_32b"):
        from vllm import LLM

        TP_CONSTRAINTS = {
            "qwen3vl_2b":  1,
            "qwen3vl_4b":  1,
            "qwen3vl_8b":  1,
            "qwen3vl_32b": 2,
        }
        n_gpus = torch.cuda.device_count()
        tp = min(TP_CONSTRAINTS[model_key], n_gpus)
        print(f"[INFO] tensor_parallel_size = {tp} (GPUs visibles: {n_gpus})")

        _loaded_processor = AutoProcessor.from_pretrained(hf_name, token=hf_token, trust_remote_code=True)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_gb = free_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)
        usable_gb = free_gb - 4.0
        gpu_mem_util = max(0.3, min(0.85, usable_gb / total_gb))
        print(f"[INFO] VRAM libre: {free_gb:.1f}GB → gpu_memory_utilization = {gpu_mem_util:.2f}")

        _loaded_model = LLM(
            hf_name, trust_remote_code=True, dtype="bfloat16",
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=32768,
            tensor_parallel_size=tp,
        )

    elif model_key in ("gemma3_4b", "gemma3_12b", "gemma3_27b"):
        # Two backends, selected explicitly rather than by whatever happens to be importable.
        #   vllm (default, needs venv_qwen)      -- ~8x faster, used for the real runs
        #   hf   (needs venv_internvl3)          -- reference implementation, used to validate
        # They were checked to agree item-by-item on the smoke set; see docs/GEMMA3.md.
        if GEMMA_BACKEND == "vllm":
            # vLLM resolves gated repos through huggingface_hub's own credential lookup and
            # takes no token argument, so the token has to be in the environment before LLM()
            # is constructed. Without this a gated repo fails with a bare 401 inside the
            # engine-core subprocess, where the traceback says nothing about credentials.
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token
            from vllm import LLM
            n_gpus = torch.cuda.device_count()
            # 27B is 51.1 GiB in bf16 against a 47.4 GiB A6000, so it MUST shard over two cards.
            # min(needed, n_gpus) would silently degrade that to tp=1 whenever the process sees
            # one GPU -- which is exactly what a campaign worker does, since it pins
            # CUDA_VISIBLE_DEVICES to its own single card. The result is an OOM hours into a run
            # with nothing pointing at the cause, so refuse up front instead.
            needed = 2 if model_key == "gemma3_27b" else 1
            if n_gpus < needed:
                raise RuntimeError(
                    f"{model_key} needs {needed} GPUs (weights exceed one card) but only "
                    f"{n_gpus} visible. Set CUDA_VISIBLE_DEVICES to {needed} devices; a "
                    f"one-GPU-per-worker queue cannot run this model."
                )
            tp = needed
            _loaded_processor = AutoProcessor.from_pretrained(hf_name, token=hf_token)
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            usable_gb = free_bytes / (1024**3) - 4.0
            gpu_mem_util = max(0.3, min(0.85, usable_gb / (total_bytes / (1024**3))))
            print(f"[INFO] gemma3 backend=vllm tp={tp} gpu_memory_utilization={gpu_mem_util:.2f}")
            _loaded_model = LLM(
                hf_name, dtype="bfloat16", gpu_memory_utilization=gpu_mem_util,
                max_model_len=16384, tensor_parallel_size=tp,
                # vLLM caps multimodal items per prompt at 1 by default; the benchmark sends 8
                # frames, and without this the extra 7 are rejected at scheduling time.
                limit_mm_per_prompt={"image": NUM_FRAMES},
            )
        else:
            from transformers import Gemma3ForConditionalGeneration
            # eager attention: Gemma 3 interleaves sliding-window and full-attention layers
            # (pattern of 6). Asking for eager explicitly keeps the mask handling on the
            # reference path rather than depending on what kernel happens to be installed.
            print("[INFO] gemma3 backend=hf (reference implementation)")
            _loaded_processor = AutoProcessor.from_pretrained(hf_name, token=hf_token)
            _loaded_model = Gemma3ForConditionalGeneration.from_pretrained(
                hf_name, token=hf_token, torch_dtype=torch.bfloat16,
                device_map="auto", attn_implementation="eager", max_memory=max_memory
            ).eval()

    elif model_key in ("internvl3_1b", "internvl3_2b", "internvl3_8b", "internvl3_9b", "internvl3_14b"):
        from transformers import AutoTokenizer
        _loaded_processor = AutoTokenizer.from_pretrained(hf_name, token=hf_token, trust_remote_code=True)
        _loaded_model = AutoModelForCausalLM.from_pretrained(
            hf_name, token=hf_token, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True
        )

    else:
        _loaded_processor = AutoProcessor.from_pretrained(hf_name, token=hf_token, trust_remote_code=True)
        _loaded_model = AutoModelForCausalLM.from_pretrained(
            hf_name, token=hf_token, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True, max_memory=max_memory
        )

    _loaded_model_name = model_key
    print(f"[INFO] Model loaded: {hf_name}")

def free_model():
    """Free the loaded model from GPU memory."""
    global _loaded_model, _loaded_processor, _loaded_model_name
    if _loaded_model is not None:
        import torch

        del _loaded_model
        del _loaded_processor
        torch.cuda.empty_cache()
        _loaded_model = None
        _loaded_processor = None
        _loaded_model_name = None
        print("[INFO] Model freed from memory.")

def run_open_source(frames, prompt, seed_key=None):
    import torch
    """Run inference with the currently loaded open source model.

    seed_key identifies the item (``f"{clip_id}|{task_type}"``) so it is sampled with its own
    seed; see SEED_MODE.
    """
    global _loaded_model, _loaded_processor, _loaded_model_name

    if _loaded_model is None:
        raise RuntimeError("No model loaded. Call load_model() first.")

    seed = item_seed(seed_key)

    pil_images = [frame_to_pil(f) for f in frames]

    if _loaded_model_name == "llama3_video":
        pil_images = [img.resize((384, 384)) for img in pil_images]
        image_contents = [{"type": "image"} for _ in pil_images]
        image_contents.append({"type": "text", "text": prompt})
        conversation = [{"role": "user", "content": image_contents}]
        inputs = _loaded_processor(
            conversation=conversation,
            images=[("video", pil_images)],
            return_tensors="pt",
        )
        inputs = {k: v.to("cuda:0", dtype=torch.bfloat16) if hasattr(v, 'to') and hasattr(v, 'dtype') and v.dtype in (torch.float32, torch.float16) else v.to("cuda:0") if hasattr(v, 'to') else v for k, v in inputs.items()}
        _seed_torch(seed)
        with torch.no_grad():
            output = _loaded_model.generate(
                **inputs, max_new_tokens=MAX_OUTPUT_TOKENS,
                do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
            )
        return _loaded_processor.decode(output[0], skip_special_tokens=True).strip()

    elif _loaded_model_name == "llava_video":
        import copy
        tokenizer, image_processor = _loaded_processor
        from llava.mm_utils import tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates

        # One video containing all sampled frames: one visual token, one tensor.
        conversation = copy.deepcopy(conv_templates["qwen_1_5"])
        conversation.append_message(conversation.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
        conversation.append_message(conversation.roles[1], None)
        input_ids = tokenizer_image_token(
            conversation.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(_loaded_model.device)
        video = image_processor.preprocess(pil_images, return_tensors="pt")["pixel_values"]
        video = video.to(_loaded_model.device, dtype=_loaded_model.dtype)
        _seed_torch(seed)
        with torch.inference_mode():
            output = _loaded_model.generate(
                input_ids, images=[video], modalities=["video"],
                max_new_tokens=MAX_OUTPUT_TOKENS,
                do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
            )
        return tokenizer.decode(output[0], skip_special_tokens=True).strip()

    elif _loaded_model_name in ("qwen3vl_2b", "qwen3vl_4b", "qwen3vl_8b", "qwen3vl_32b"):
        from vllm import SamplingParams
        from io import BytesIO
        sampling_params = SamplingParams(max_tokens=MAX_OUTPUT_TOKENS,
                                         temperature=TEMPERATURE, top_p=TOP_P, seed=seed,
                                         repetition_penalty=REPETITION_PENALTY)
        image_contents = []
        for img in pil_images:
            buf = BytesIO()
            img.save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            image_contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        image_contents.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": image_contents}]
        text_prompt = _loaded_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = _loaded_model.generate([{
            "prompt": text_prompt,
            "multi_modal_data": {"image": pil_images}
        }], sampling_params)
        return outputs[0].outputs[0].text.strip()

    elif _loaded_model_name in ("gemma3_4b", "gemma3_12b", "gemma3_27b"):
        # Gemma 3 must go through apply_chat_template, NOT processor(images=..., text=...).
        # The processor only expands an image into its 256 soft tokens where it finds a
        # <start_of_image> marker in the text; a raw prompt has none, so the pixel values are
        # computed, passed in, and then never attended to. The model answers from the text alone
        # and looks like a poor vision model instead of erroring. The chat template inserts the
        # markers, one per {"type": "image"} entry.
        if GEMMA_BACKEND == "vllm":
            from vllm import SamplingParams
            content = [{"type": "image"} for _ in pil_images]
            content.append({"type": "text", "text": prompt})
            text_prompt = _loaded_processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False, add_generation_prompt=True,
            )
            n_markers = text_prompt.count("<start_of_image>")
            if n_markers != len(pil_images):
                raise RuntimeError(
                    f"Gemma3 prompt carries {n_markers} image markers for {len(pil_images)} "
                    f"frames -- the frames are not reaching the model."
                )
            # Gemma 3's chat template emits a literal <bos>, and vLLM re-tokenizes a string
            # prompt with add_special_tokens=True (vllm/multimodal/processing/context.py:331),
            # which prepends another one: input_ids start [2, 2, ...] instead of [2, ...].
            # A duplicated BOS is off-distribution for the model and degrades output quality
            # silently -- nothing errors, the answers are just worse. Drop ours and let vLLM
            # add exactly one. The HF backend is unaffected: it tokenizes the template itself.
            bos = getattr(_loaded_processor.tokenizer, "bos_token", None)
            if bos and text_prompt.startswith(bos):
                text_prompt = text_prompt[len(bos):]
            outputs = _loaded_model.generate([{
                "prompt": text_prompt,
                "multi_modal_data": {"image": pil_images},
            }], SamplingParams(max_tokens=MAX_OUTPUT_TOKENS,
                              temperature=TEMPERATURE, top_p=TOP_P, seed=seed,
                              repetition_penalty=REPETITION_PENALTY))
            return outputs[0].outputs[0].text.strip()

        content = [{"type": "image", "image": img} for img in pil_images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = _loaded_processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        device = next(_loaded_model.parameters()).device
        inputs = {
            k: (v.to(device=device, dtype=torch.bfloat16)
                if hasattr(v, "is_floating_point") and v.is_floating_point()
                else v.to(device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
        input_len = inputs["input_ids"].shape[-1]
        # NUM_FRAMES x 256 = 2048 image tokens, so this is well inside the 128k context.
        n_img = int((inputs["input_ids"] == _loaded_model.config.image_token_id).sum())
        if n_img != len(pil_images) * _loaded_model.config.mm_tokens_per_image:
            raise RuntimeError(
                f"Gemma3 image-token count is {n_img}, expected "
                f"{len(pil_images) * _loaded_model.config.mm_tokens_per_image} -- the frames are "
                f"not reaching the model."
            )
        _seed_torch(seed)
        with torch.no_grad():
            output = _loaded_model.generate(
                **inputs,
                max_new_tokens=MAX_OUTPUT_TOKENS,
                do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
            )
        # generate() returns the prompt followed by the continuation; decoding the whole thing
        # would prepend 2048 image placeholders and the full instruction to every answer.
        return _loaded_processor.decode(output[0][input_len:], skip_special_tokens=True).strip()

    elif _loaded_model_name in ("internvl3_1b", "internvl3_2b", "internvl3_8b", "internvl3_9b", "internvl3_14b"):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        
        transform = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        pixel_values = torch.stack([transform(img) for img in pil_images]).to(_loaded_model.device, dtype=torch.bfloat16)
        num_patches_list = [len(pil_images)]

        _seed_torch(seed)
        response = _loaded_model.chat(
            tokenizer=_loaded_processor,
            pixel_values=pixel_values,
            question=prompt,
            num_patches_list=num_patches_list,
            generation_config={"max_new_tokens": MAX_OUTPUT_TOKENS, "do_sample": True,
                               "temperature": TEMPERATURE, "top_p": TOP_P,
                               "repetition_penalty": REPETITION_PENALTY}
        )
        return response

    else:
        inputs = _loaded_processor(images=pil_images, text=prompt, return_tensors="pt")

    device = next(_loaded_model.parameters()).device
    inputs = {
        k: v.to(device=device, dtype=torch.bfloat16) if hasattr(v, "to") and v.dtype == torch.float32
        else v.to(device) if hasattr(v, "to")
        else v
        for k, v in inputs.items()
    }
    _seed_torch(seed)
    with torch.no_grad():
        output = _loaded_model.generate(
            **inputs, max_new_tokens=MAX_OUTPUT_TOKENS,
            do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
        )
    return _loaded_processor.decode(output[0], skip_special_tokens=True).strip()

############################################
# UNIFIED INTERFACE
############################################


CLOSED_SOURCE_MODELS = set(GEMINI_MODEL_IDS) | {"gpt_realtime_mini"}
OPEN_SOURCE_MODELS = {"qwen3vl_2b", "qwen3vl_4b", "qwen3vl_8b", "qwen3vl_32b", "internvl3_1b", "internvl3_2b", "internvl3_8b", "internvl3_9b", "internvl3_14b", "llava_video", "llama3_video", "gemma3_4b", "gemma3_12b", "gemma3_27b"}

def run_model(frames, prompt, model_key, gemini_api_key=None, openai_api_key=None,
              hf_token=None, seed_key=None):
    """Unified inference function for all models."""
    if model_key in GEMINI_MODEL_IDS:
        return run_gemini(
            frames,
            prompt,
            gemini_api_key,
            model_name=GEMINI_MODEL_IDS[model_key],
        )
    elif model_key == "gpt_realtime_mini":
        return run_gpt_realtime(frames, prompt, openai_api_key)
    elif model_key in OPEN_SOURCE_MODELS:
        return run_open_source(frames, prompt, seed_key=seed_key)
    else:
        raise ValueError(f"Unknown model key: {model_key}")
