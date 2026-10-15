import os
import base64
import json
import random
import re
import sys
import time
import cv2
import httpx
import numpy as np
import requests
import torch
from PIL import Image


LLAMA3_VIDEO_RESIZE = 640
MAX_TOKENS = 4096
GEMMA3_4B_MODEL = os.environ.get(
    "EGOB_GEMMA3_4B_MODEL", "google/gemma-3-4b-it"
)
GEMMA3_4B_REVISION = os.environ.get("EGOB_GEMMA3_4B_REVISION") or None


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


def run_gemini(frames, question, api_key, model_name="gemini-3.1-flash-lite"):
    """Run inference with Gemini via the Google Gen AI SDK."""
    if not api_key:
        raise ValueError("A Gemini API key is required for Gemini models")

    from google import genai
    from google.genai import errors, types

    parts = [
        types.Part.from_bytes(
            data=frame_to_jpeg_bytes(frame),
            mime_type="image/jpeg",
        )
        for frame in frames
    ]
    parts.append(types.Part.from_text(text=question))

    client = genai.Client(api_key=api_key)
    try:
        for attempt in range(1, GEMINI_RETRY_ATTEMPTS + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=parts,
                    config=types.GenerateContentConfig(
                        max_output_tokens=MAX_TOKENS,
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

GPT_REALTIME_RETRY_ATTEMPTS = 5
GPT_REALTIME_RETRY_INITIAL_DELAY = 2.0
GPT_REALTIME_RETRY_MAX_DELAY = 30.0


def _receive_realtime_event(websocket, expected_type=None, timeout=120):
    while True:
        event = json.loads(websocket.recv(timeout=timeout))
        if event.get("type") == "error":
            error = event.get("error", {})
            raise RuntimeError(error.get("message") or json.dumps(error))
        if expected_type is None or event.get("type") == expected_type:
            return event


def _extract_realtime_text(response_event, text_deltas):
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


def _run_gpt_realtime_once(frames, question, api_key, model_name):
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
        websocket.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": model_name,
                "output_modalities": ["text"],
            },
        }))
        _receive_realtime_event(websocket, "session.updated")

        content = [
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{frame_to_base64(frame)}",
            }
            for frame in frames
        ]
        content.append({"type": "input_text", "text": question})
        websocket.send(json.dumps({
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user", "content": content},
        }))
        websocket.send(json.dumps({
            "type": "response.create",
            "response": {
                "output_modalities": ["text"],
                "max_output_tokens": MAX_TOKENS,
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


def run_gpt_realtime(frames, question, api_key, model_name="gpt-realtime-mini"):
    if not api_key:
        raise ValueError("An OpenAI API key is required for gpt_realtime_mini")

    for attempt in range(1, GPT_REALTIME_RETRY_ATTEMPTS + 1):
        try:
            return _run_gpt_realtime_once(frames, question, api_key, model_name)
        except Exception as error:
            if attempt == GPT_REALTIME_RETRY_ATTEMPTS:
                raise
            delay = min(
                GPT_REALTIME_RETRY_INITIAL_DELAY * (2 ** (attempt - 1))
                + random.uniform(0.0, 1.0),
                GPT_REALTIME_RETRY_MAX_DELAY,
            )
            print(
                f"[WARN] Realtime request failed ({error}); retrying in "
                f"{delay:.1f}s ({attempt + 1}/{GPT_REALTIME_RETRY_ATTEMPTS})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)

############################################
# OPEN SOURCE MODELS
############################################

_loaded_model = None
_loaded_processor = None
_loaded_model_name = None

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
        "internvl3_14b": "OpenGVLab/InternVL3-14B",
        "llava_video":   "lmms-lab/LLaVA-Video-7B-Qwen2",
        "llama3_video":  "DAMO-NLP-SG/VideoLLaMA3-7B",
        "gemma3_4b":     GEMMA3_4B_MODEL,
    }

    if model_key not in model_map:
        raise ValueError(f"Unknown model key: {model_key}")

    hf_name = model_map[model_key]
    print(f"[INFO] Loading model: {hf_name}")

    from transformers import AutoProcessor, AutoModelForCausalLM, AutoModel
    print(f"[INFO] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"[INFO] torch.cuda.device_count() = {torch.cuda.device_count()}")

    if model_key == "llava_video":
        from llava.model.builder import load_pretrained_model
        n_gpus = torch.cuda.device_count()
        mm = {i: "42GiB" for i in range(n_gpus)}
        mm["cpu"] = "100GiB"
        dtype_name = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
        tokenizer, model, image_processor, _ = load_pretrained_model(
            hf_name, None, "llava_qwen", torch_dtype=dtype_name, device_map="auto",
            attn_implementation="eager",
            max_memory=mm
        )
        meta_params = [name for name, parameter in model.named_parameters() if parameter.is_meta]
        if meta_params:
            print(f"[WARNING] Model still has {len(meta_params)} meta parameters after loading.")
            print("[WARNING] First few meta params:", meta_params[:20])
        else:
            print("[INFO] No meta parameters remain after loading.")
        _loaded_model = model.eval()
        _loaded_processor = (tokenizer, image_processor)

    elif model_key in ("qwen3vl_2b", "qwen3vl_4b", "qwen3vl_8b", "qwen3vl_32b"):
        from vllm import LLM

        # Attention heads par modèle — tp doit être un diviseur
        TP_CONSTRAINTS = {
            "qwen3vl_2b":  1,   # 16 heads
            "qwen3vl_4b":  1,
            "qwen3vl_8b":  1,
            "qwen3vl_32b": 2,
        }
        n_gpus = torch.cuda.device_count()
        tp = min(TP_CONSTRAINTS[model_key], n_gpus)
        print(f"[INFO] tensor_parallel_size = {tp} (GPUs visibles: {n_gpus})")

        _loaded_processor = AutoProcessor.from_pretrained(hf_name, token=hf_token, trust_remote_code=True)
        # Calcule gpu_memory_utilization dynamiquement
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_gb = free_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)
        # Garde 4 GB de marge pour le modèle + NCCL
        usable_gb = free_gb - 4.0
        gpu_mem_util = max(0.3, min(0.85, usable_gb / total_gb))
        print(f"[INFO] VRAM libre: {free_gb:.1f}GB → gpu_memory_utilization = {gpu_mem_util:.2f}")

        _loaded_model = LLM(
            hf_name, trust_remote_code=True, dtype="bfloat16",
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=32768,
            tensor_parallel_size=tp,
        )

    elif model_key == "gemma3_4b":
        from transformers import Gemma3ForConditionalGeneration

        print("[INFO] Gemma 3 backend: Hugging Face Transformers")
        _loaded_processor = AutoProcessor.from_pretrained(
            hf_name, token=hf_token, revision=GEMMA3_4B_REVISION
        )
        _loaded_model = Gemma3ForConditionalGeneration.from_pretrained(
            hf_name,
            token=hf_token,
            revision=GEMMA3_4B_REVISION,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        ).eval()

    elif model_key in ("internvl3_1b", "internvl3_2b", "internvl3_8b", "internvl3_14b"):
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
            device_map="auto", trust_remote_code=True
        )

    _loaded_model_name = model_key
    print(f"[INFO] Model loaded: {hf_name}")

def free_model():
    import torch
    """Free the loaded model from GPU memory."""
    global _loaded_model, _loaded_processor, _loaded_model_name
    if _loaded_model is not None:
        del _loaded_model
        del _loaded_processor
        torch.cuda.empty_cache()
        _loaded_model = None
        _loaded_processor = None
        _loaded_model_name = None
        print("[INFO] Model freed from memory.")

def run_open_source(frames, question, llama3_video_resize=LLAMA3_VIDEO_RESIZE):
    import torch
    """Run inference with the currently loaded open source model."""
    global _loaded_model, _loaded_processor, _loaded_model_name

    if _loaded_model is None:
        raise RuntimeError("No model loaded. Call load_model() first.")

    pil_images = [frame_to_pil(f) for f in frames]

    if _loaded_model_name == "llama3_video":
        if llama3_video_resize is None:
            llama3_video_resize = LLAMA3_VIDEO_RESIZE
        if llama3_video_resize and llama3_video_resize > 0:
            pil_images = [
                img.resize((llama3_video_resize, llama3_video_resize))
                for img in pil_images
            ]
        image_contents = [{"type": "image"} for _ in pil_images]
        image_contents.append({"type": "text", "text": question})
        conversation = [{"role": "user", "content": image_contents}]
        inputs = _loaded_processor(
            conversation=conversation,
            images=[("video", pil_images)],
            return_tensors="pt",
        )
        inputs = {k: v.to("cuda:0", dtype=torch.bfloat16) if hasattr(v, 'to') and hasattr(v, 'dtype') and v.dtype in (torch.float32, torch.float16) else v.to("cuda:0") if hasattr(v, 'to') else v for k, v in inputs.items()}
        with torch.no_grad():
            output = _loaded_model.generate(**inputs, max_new_tokens=MAX_TOKENS)
        return _loaded_processor.decode(output[0], skip_special_tokens=True).strip()

    elif _loaded_model_name == "llava_video":
        tokenizer, image_processor = _loaded_processor
        import copy
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token

        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        video_np = np.stack([np.asarray(img.convert("RGB")) for img in pil_images])
        video_tensor = image_processor.preprocess(video_np, return_tensors="pt")["pixel_values"]
        video_tensor = video_tensor.to(device=device, dtype=image_dtype)

        instruction = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

        with torch.no_grad():
            output = _loaded_model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=[video_tensor],
                modalities=["video"],
                do_sample=False,
                temperature=0,
                max_new_tokens=MAX_TOKENS,
                pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip()

    elif _loaded_model_name in ("qwen3vl_2b", "qwen3vl_4b", "qwen3vl_8b", "qwen3vl_32b"):
        from vllm import SamplingParams
        import base64
        from io import BytesIO

        sampling_params = SamplingParams(max_tokens=MAX_TOKENS)

        image_contents = []
        for img in pil_images:
            buf = BytesIO()
            img.save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        image_contents.append({"type": "text", "text": question})

        messages = [{"role": "user", "content": image_contents}]
        prompt = _loaded_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        outputs = _loaded_model.generate([{
            "prompt": prompt,
            "multi_modal_data": {"image": pil_images}
        }], sampling_params=sampling_params)

        return outputs[0].outputs[0].text.strip()

    elif _loaded_model_name == "gemma3_4b":
        content = [{"type": "image", "image": image} for image in pil_images]
        content.append({"type": "text", "text": question})
        inputs = _loaded_processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = next(_loaded_model.parameters()).device
        inputs = {
            key: (
                value.to(device=device, dtype=torch.bfloat16)
                if hasattr(value, "is_floating_point") and value.is_floating_point()
                else value.to(device) if hasattr(value, "to") else value
            )
            for key, value in inputs.items()
        }
        input_length = inputs["input_ids"].shape[-1]
        image_tokens = int(
            (inputs["input_ids"] == _loaded_model.config.image_token_id).sum()
        )
        expected_image_tokens = (
            len(pil_images) * _loaded_model.config.mm_tokens_per_image
        )
        if image_tokens != expected_image_tokens:
            raise RuntimeError(
                f"Gemma 3 prompt contains {image_tokens} image tokens; expected "
                f"{expected_image_tokens} for {len(pil_images)} frames"
            )
        with torch.no_grad():
            output = _loaded_model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                do_sample=False,
            )
        return _loaded_processor.decode(
            output[0][input_length:], skip_special_tokens=True
        ).strip()
    
    elif _loaded_model_name in ("internvl3_1b", "internvl3_2b", "internvl3_8b", "internvl3_14b"):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        
        transform = T.Compose([
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        pixel_values = torch.stack([transform(img) for img in pil_images]).to(_loaded_model.device, dtype=torch.bfloat16)
        num_patches_list = [len(pil_images)]  # toutes les frames = 1 entrée avec N patches

        eos_token_id = getattr(_loaded_processor, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(getattr(_loaded_model, "generation_config", None), "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(getattr(_loaded_model, "config", None), "eos_token_id", None)

        pad_token_id = getattr(_loaded_processor, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(getattr(_loaded_model, "generation_config", None), "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(getattr(_loaded_model, "config", None), "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = eos_token_id

        generation_config = {
            "max_new_tokens": MAX_TOKENS,
            "do_sample": False,
        }
        if eos_token_id is not None:
            generation_config["eos_token_id"] = eos_token_id
        if pad_token_id is not None:
            generation_config["pad_token_id"] = pad_token_id

        response = _loaded_model.chat(
            tokenizer=_loaded_processor,
            pixel_values=pixel_values,
            question=question,
            num_patches_list=num_patches_list,
            generation_config=generation_config,
        )
        return response

    else:
        inputs = _loaded_processor(images=pil_images, text=question, return_tensors="pt")

    # Move all inputs to GPU and cast to bfloat16
    inputs = {
        k: v.to(device=_loaded_model.device, dtype=torch.bfloat16)
        if hasattr(v, 'to') and v.dtype == torch.float32
        else v.to(_loaded_model.device)
        if hasattr(v, 'to')
        else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        output = _loaded_model.generate(**inputs, max_new_tokens=MAX_TOKENS)
    return _loaded_processor.decode(output[0], skip_special_tokens=True).strip()

############################################
# UNIFIED INTERFACE
############################################

CLOSED_SOURCE_MODELS = set(GEMINI_MODEL_IDS) | {"gpt_realtime_mini"}
OPEN_SOURCE_MODELS = {"qwen3vl_2b", "qwen3vl_4b", "qwen3vl_8b", "qwen3vl_32b", "internvl3_1b", "internvl3_2b", "internvl3_8b", "internvl3_14b", "llava_video", "llama3_video", "gemma3_4b"}

def run_model(frames, question, model_key, gemini_api_key=None, openai_api_key=None,
              hf_token=None, llama3_video_resize=LLAMA3_VIDEO_RESIZE):
    """Unified inference function for all models."""
    if model_key in GEMINI_MODEL_IDS:
        return run_gemini(
            frames,
            question,
            gemini_api_key,
            model_name=GEMINI_MODEL_IDS[model_key],
        )
    elif model_key == "gpt_realtime_mini":
        return run_gpt_realtime(frames, question, openai_api_key)
    elif model_key in OPEN_SOURCE_MODELS:
        return run_open_source(frames, question, llama3_video_resize=llama3_video_resize)
    else:
        raise ValueError(f"Unknown model key: {model_key}")
