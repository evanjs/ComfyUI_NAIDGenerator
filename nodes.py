import copy
import os
from pathlib import Path
import time
from contextlib import contextmanager
import folder_paths
import zipfile
import json as _json
import copy as _copy

from .get_image_metadata import GetImageMetadata
from .character_concatenate_nai import CharacterConcatenateNAI
from .character_nai import CharacterNAI

from .utils import *
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import torch
import numpy as np
from PIL import Image as PILImage


TOOLTIP_LIMIT_OPUS_FREE = "Limit image size and steps for free generation by Opus."

@contextmanager
def _naid_profile_step(enabled: bool, label: str):
    if not enabled:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[NovelAI][profile] {label}: {elapsed:.3f}s")

# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------

# Accepted canvas sizes (per CR guidance); we will letterbox/pad to one of these
ACCEPTED_CR_SIZES = [(1024, 1536), (1536, 1024), (1472, 1472)]

def _get_user_data(access_token, timeout=120, retry=3):
    """Fetches user data to check Anlas balance. Now a global helper."""
    USER_API_BASE_URL = "https://api.novelai.net"

    req_mod = requests
    if retry is not None and retry > 1:
        retries = Retry(
            total=retry,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retries))
        req_mod = session

    response = req_mod.get(
        f"{USER_API_BASE_URL}/user/data",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=timeout
    )

    response.raise_for_status()
    return response.json()

def _choose_cr_canvas(w, h):
    """Select the accepted CR canvas size whose aspect ratio is closest to the source image."""
    aspect = w / h
    best = None
    best_diff = 9e9
    for cw, ch in ACCEPTED_CR_SIZES:
        diff = abs((cw / ch) - aspect)
        if diff < best_diff:
            best_diff = diff
            best = (cw, ch)
    return best

def pad_image_to_canvas(tensor_image, target_size):
    """
    Letterbox the given tensor image [1,H,W,C] into target_size (W,H) with black padding,
    preserving aspect ratio.
    """
    _, H, W, C = tensor_image.shape
    tw, th = target_size
    arr = (tensor_image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    mode = "RGBA" if (C == 4) else "RGB"
    pil = PILImage.fromarray(arr)

    scale = min(tw / W, th / H)
    new_w = max(1, int(W * scale))
    new_h = max(1, int(H * scale))
    pil_resized = pil.resize((new_w, new_h), PILImage.LANCZOS)

    if mode == "RGBA":
        canvas = PILImage.new("RGBA", (tw, th), (0, 0, 0, 0))
    else:
        canvas = PILImage.new("RGB", (tw, th), (0, 0, 0))
    offset = ((tw - new_w) // 2, (th - new_h) // 2)
    canvas.paste(pil_resized, offset)

    out = np.array(canvas).astype(np.float32) / 255.0
    return torch.from_numpy(out)[None,]

# -------------------------------------------------
# Core simple prompt conversion / utility nodes
# -------------------------------------------------

class PromptToNAID:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": {
            "text": ("STRING", { "forceInput":True, "multiline": True, "dynamicPrompts": False,}),
            "weight_per_brace": ("FLOAT", { "default": 0.05, "min": 0.05, "max": 0.10, "step": 0.05 }),
            "syntax_mode": (["brace", "numeric"], { "default": "brace" }),
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert"
    CATEGORY = "NovelAI/utils"
    def convert(self, text, weight_per_brace, syntax_mode):
        nai_prompt = prompt_to_nai(text, weight_per_brace, syntax_mode)
        return (nai_prompt,)

class ImageToNAIMask:
    @classmethod
    def INPUT_TYPES(s):
        return { "required": { "image": ("IMAGE",) } }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "NovelAI/utils"
    def convert(self, image):
        s = resize_to_naimask(image)
        return (s,)

class ModelOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ([
                    "nai-diffusion-2",
                    "nai-diffusion-furry-3",
                    "nai-diffusion-3",
                    "nai-diffusion-4-curated-preview",
                    "nai-diffusion-4-full",
                    "nai-diffusion-4-5-curated",
                    "nai-diffusion-4-5-full"
                ], { "default": "nai-diffusion-4-5-full" }),
            },
            "optional": { "option": ("NAID_OPTION",) },
        }

    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"
    def set_option(self, model, option=None):
        option = copy.deepcopy(option) if option else {}
        option["model"] = model
        return (option,)

class Img2ImgOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", { "default": 0.70, "min": 0.01, "max": 0.99, "step": 0.01, "display": "number" }),
                "noise": ("FLOAT", { "default": 0.00, "min": 0.00, "max": 0.99, "step": 0.02, "display": "number" }),
            },
        }

    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"
    def set_option(self, image, strength, noise):
        option = {}
        option["img2img"] = (image, strength, noise)
        return (option,)

class InpaintingOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
                "add_original_image": ("BOOLEAN", { "default": True }),
            },
        }

    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"
    def set_option(self, image, mask, add_original_image):
        option = {}
        option["infill"] = (image, mask, add_original_image)
        return (option,)

class VibeTransferOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "information_extracted": ("FLOAT", { "default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "display": "number" }),
                "strength": ("FLOAT", { "default": 0.6, "min": 0.01, "max": 1.0, "step": 0.01, "display": "number" }),
            },
            "optional": { "option": ("NAID_OPTION",) },
        }
    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"
    def set_option(self, image, information_extracted, strength, option=None):
        option = copy.deepcopy(option) if option else {}
        if "vibe" not in option:
            option["vibe"] = []
        option["vibe"].append((image, information_extracted, strength))
        return (option,)

class NetworkOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ignore_errors": ("BOOLEAN", { "default": True }),
                "timeout_sec": ("INT", { "default": 120, "min": 30, "max": 3000, "step": 1, "display": "number" }),
                "retry": ("INT", { "default": 3, "min": 1, "max": 100, "step": 1, "display": "number" }),
            },
            "optional": { "option": ("NAID_OPTION",) },
        }
    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"
    def set_option(self, ignore_errors, timeout_sec, retry, option=None):
        option = copy.deepcopy(option) if option else {}
        option["ignore_errors"] = ignore_errors
        option["timeout_sec"] = timeout_sec
        option["retry"] = retry
        return (option,)


class AutosaveOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "autosave_format": (["png", "webp"], { "default": "png" }),
                "webp_quality": ("INT", { "default": 85, "min": 1, "max": 100, "step": 1, "display": "number" }),
            },
            "optional": { "option": ("NAID_OPTION",) },
        }

    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"

    def set_option(self, autosave_format, webp_quality, option=None):
        option = copy.deepcopy(option) if option else {}
        option["autosave_format"] = autosave_format
        option["webp_quality"] = webp_quality
        return (option,)

# -------------------------------------------------
# Character Reference (Single Image)
# -------------------------------------------------

class CharacterReferenceOption:
    INFO_EXTRACT_DEFAULT = 1.0
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "style_aware": ("BOOLEAN", {"default": True, "tooltip": "Copy style along with identity."}),
                "fidelity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "number", "tooltip": "How strictly to match the character (and style if enabled)."}),
            },
            "optional": {"option": ("NAID_OPTION",),}
        }
    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"
    def set_option(self, image, style_aware, fidelity, option=None):
        option = copy.deepcopy(option) if option else {}
        fidelity = max(0.0, min(1.0, fidelity))
        option["character_reference_single"] = {
            "image": image,
            "style_aware": style_aware,
            "fidelity": fidelity,
            "info_extracted": self.INFO_EXTRACT_DEFAULT,
        }
        return (option,)

# -------------------------------------------------
# Generation Node
# -------------------------------------------------

class GenerateNAID:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "limit_opus_free": ("BOOLEAN", {"default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE}),
                "width": ("INT", {"default": 832, "min": 64, "max": 1600, "step": 64, "display": "number"}),
                "height": ("INT", {"default": 1216, "min": 64, "max": 1600, "step": 64, "display": "number"}),
                "positive": ("STRING", {"default": ", best quality, amazing quality, very aesthetic, absurdres", "multiline": True, "dynamicPrompts": False}),
                "negative": ("STRING", {"default": "lowres", "multiline": True, "dynamicPrompts": False}),
                "use_coords": ("BOOLEAN", {"default": True, "label": "Manual_positioning", "tooltip": "Manually specify character positions"}),
                "use_order": ("BOOLEAN", {"default": True, "label": "Keep_order", "tooltip": "Preserve character order"}),
                "legacy_uc": ("BOOLEAN", {"default": False, "label": "Legacy_UC", "tooltip": "Use legacy UC logic"}),
                "steps": ("INT", {"default": 28, "min": 0, "max": 50, "step": 1, "display": "number"}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 10.0, "step": 0.1, "display": "number"}),
                "variety": ("BOOLEAN", {"default": False}),
                "decrisper": ("BOOLEAN", {"default": False}),
                "smea": (["none", "SMEA", "SMEA+DYN"], {"default": "none"}),
                "sampler": (["k_euler", "k_euler_ancestral", "k_dpmpp_2s_ancestral", "k_dpmpp_2m_sde", "k_dpmpp_2m", "k_dpmpp_sde", "ddim"], {"default": "k_euler"}),
                "scheduler": (["native", "karras", "exponential", "polyexponential"], {"default": "native"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 9999999999, "step": 1, "display": "number"}),
                "uncond_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05, "display": "number"}),
                "cfg_rescale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.02, "display": "number"}),
                "keep_alpha": ("BOOLEAN", {"default": True, "tooltip": "Disable to further process output images locally"}),
            },
            "optional": {"option": ("NAID_OPTION",)},
        }

    RETURN_TYPES = ("IMAGE","METADATA",)
    FUNCTION = "generate"
    CATEGORY = "NovelAI"

    @staticmethod
    def _post_image(access_token, prompt, model, action, parameters, timeout=None, retry=None):
        data = {"input": prompt, "model": model, "action": action, "parameters": parameters}

        req_mod = requests
        if retry is not None and retry > 1:
            retries = Retry(total=retry, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["POST"])
            session = requests.Session()
            session.mount("https://", HTTPAdapter(max_retries=retries))
            req_mod = session

        response = req_mod.post(f"{BASE_URL}/ai/generate-image", json=data, headers={"Authorization": f"Bearer {access_token}"}, timeout=timeout)

        if response.status_code >= 400:
            print("RAW ERROR STATUS:", response.status_code)
            print("RAW ERROR BODY:", response.text)
            try:
                dbg = _copy.deepcopy(data)
                p = dbg.get("parameters", {})
                if "director_reference_images" in p: p["director_reference_images"] = [i[:60] + "...(trunc)" for i in p["director_reference_images"]]
                if "reference_image_multiple" in p: p["reference_image_multiple"] = [i[:60] + "...(trunc)" for i in p["reference_image_multiple"]]
                dbg["parameters"] = p
                print("OUTGOING PAYLOAD (sanitized):", _json.dumps(dbg)[:2000])
            except Exception as e:
                print("Payload debug failed:", e)

        response.raise_for_status()
        return response.content

    def generate(self, limit_opus_free, width, height, positive, negative,
                 steps, cfg, decrisper, variety, smea, sampler, scheduler,
                 seed, uncond_scale, cfg_rescale, keep_alpha, use_coords, use_order, legacy_uc, option=None):
        profile_enabled = True if option is None else option.get("profile", True)
        total_start = time.perf_counter()

        with _naid_profile_step(profile_enabled, "calculate_resolution"):
            width, height = calculate_resolution(width * height, (width, height))

        with _naid_profile_step(profile_enabled, "build base params"):
            params = {
                "params_version": 1, "width": width, "height": height, "scale": cfg, "sampler": sampler, "steps": steps,
                "seed": seed, "n_samples": 1, "ucPreset": 3, "qualityToggle": False,
                "sm": (smea == "SMEA" or smea == "SMEA+DYN") and sampler != "ddim",
                "sm_dyn": (smea == "SMEA+DYN") and sampler != "ddim",
                "dynamic_thresholding": decrisper, "controlnet_strength": 1.0, "add_original_image": False,
                "cfg_rescale": cfg_rescale, "noise_schedule": scheduler, "legacy_v3_extend": False,
                "uncond_scale": uncond_scale, "negative_prompt": negative, "prompt": positive,
                "reference_image_multiple": [], "reference_information_extracted_multiple": [], "reference_strength_multiple": [],
                "extra_noise_seed": seed,
                "v4_prompt": {"use_coords": use_coords, "use_order": use_order,
                              "legacy_uc": legacy_uc, "caption": {"base_caption": positive, "char_captions": []}},
                "v4_negative_prompt": {"use_coords": False, "use_order": False, "caption": {"base_caption": negative, "char_captions": []}}
            }

            model = "nai-diffusion-4-5-full"
            action = "generate"

            if sampler == "k_euler_ancestral" and scheduler != "native":
                params["deliberate_euler_ancestral_bug"] = False
                params["prefer_brownian"] = True

        if option:
            with _naid_profile_step(profile_enabled, "apply option payloads"):
                if "img2img" in option:
                    action = "img2img"
                    image, strength, noise = option["img2img"]
                    with _naid_profile_step(profile_enabled, "img2img resize + base64"):
                        params["image"] = image_to_base64(resize_image(image, (width, height)))
                    params["strength"] = strength
                    params["noise"] = noise
                elif "infill" in option:
                    action = "infill"
                    image, mask, add_original_image = option["infill"]
                    with _naid_profile_step(profile_enabled, "infill image resize + base64"):
                        params["image"] = image_to_base64(resize_image(image, (width, height)))
                    with _naid_profile_step(profile_enabled, "infill mask resize + base64"):
                        params["mask"] = naimask_to_base64(resize_to_naimask(mask, (width, height), "4" in model))
                    params["add_original_image"] = add_original_image

                if "vibe" in option:
                    with _naid_profile_step(profile_enabled, "vibe resize + base64 all"):
                        for vibe in option["vibe"]:
                            vimg, information_extracted, strength = vibe
                            params["reference_image_multiple"].append(image_to_base64(resize_image(vimg, (width, height))))
                            params["reference_information_extracted_multiple"].append(information_extracted)
                            params["reference_strength_multiple"].append(strength)

                if "model" in option:
                    model = option["model"]

                # NOTE
                # Updating dictionaries in Python will clobber/overwrite existing values
                # Instead, recursively merge the dictionaries so that we can compose base and character prompts

                # Handle V4 options
                if "v4_prompt" in option:
                    params["v4_prompt"] = merge_dicts_non_empty(params["v4_prompt"], option["v4_prompt"])

                if "v4_negative_prompt" in option:
                    params["v4_negative_prompt"] = merge_dicts_non_empty(params["v4_negative_prompt"], option["v4_negative_prompt"])

                if "character_reference_single" in option:
                    with _naid_profile_step(profile_enabled, "character reference pad + base64"):
                        ref = option["character_reference_single"]
                        base_caption = "character&style" if ref["style_aware"] else "character"
                        ref_img = ref["image"]
                        _, h_raw, w_raw, _ = ref_img.shape
                        canvas_w, canvas_h = _choose_cr_canvas(w_raw, h_raw)
                        padded = pad_image_to_canvas(ref_img, (canvas_w, canvas_h))
                        params["director_reference_images"] = [image_to_base64(padded)]
                        params["director_reference_descriptions"] = [{"use_coords": False, "use_order": False, "legacy_uc": False, "caption": {"base_caption": base_caption, "char_captions": []}}]
                        params["director_reference_strength_values"] = [1.0]
                        params["director_reference_secondary_strength_values"] = [1.0 - ref["fidelity"]]
                        params["director_reference_information_extracted"] = [1.0]

        with _naid_profile_step(profile_enabled, "final param adjustments"):
            timeout = option.get("timeout", 120) if option else 120
            retry = option.get("retry", 3) if option else 3

            if limit_opus_free:
                pixel_limit = 1024 * 1024
                if width * height > pixel_limit:
                    params["width"], params["height"] = calculate_resolution(pixel_limit, (width, height))
                if steps > 28: params["steps"] = 28

            if variety: params["skip_cfg_above_sigma"] = calculate_skip_cfg_above_sigma(params["width"], params["height"], model)
            if sampler == "ddim" and "nai-diffusion-2" not in model: params["sampler"] = "ddim_v3"
            if action == "infill" and "nai-diffusion-2" not in model: model = f"{model}-inpainting"

        start_anlas = None
        try:
            with _naid_profile_step(profile_enabled, "pre-gen anlas /user/data"):
                user_data = _get_user_data(self.access_token, timeout, retry)
                start_anlas = user_data.get("subscription", {}).get("trainingStepsLeft")
                if start_anlas is not None: print(f"[NovelAI] Anlas (pre-gen): {start_anlas}")
        except Exception as e: print(f"[NovelAI] Anlas tracking failed (pre-gen): {e}")

        image = blank_image()
        metadata = {}
        try:
            with _naid_profile_step(profile_enabled, "NovelAI generate-image POST"):
                zipped_bytes = self._post_image(self.access_token, positive, model, action, params, timeout, retry)

            with _naid_profile_step(profile_enabled, "zip read image bytes"):
                with zipfile.ZipFile(io.BytesIO(zipped_bytes)) as zipped:
                    image_bytes = zipped.read(zipped.infolist()[0])

            ## save original png to comfy output dir
            ## use basic logic to determine whether we should be saving to `img2img` or `txt2img` directory
            with _naid_profile_step(profile_enabled, "autosave path setup"):
                save_type = "img2img" if action in ("img2img", "infill") else "txt2img"
                output_type_dir = os.path.join(self.output_dir, save_type)
                full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                    "NAI_autosave", output_type_dir)

                autosave_format = option.get("autosave_format", "png") if option else "png"
                webp_quality = option.get("webp_quality", 85) if option else 85
                file_ext = "webp" if autosave_format == "webp" else "png"

                file = f"{filename}_{counter:05}_.{file_ext}"
                d = Path(full_output_folder)
                d.mkdir(exist_ok=True, parents=True)

            with _naid_profile_step(profile_enabled, f"save image with metadata ({autosave_format}, quality={webp_quality})"):
                save_image_with_metadata(image_bytes, d / file, autosave_format, webp_quality)

            if start_anlas is not None:
                try:
                    with _naid_profile_step(profile_enabled, "post-gen anlas /user/data"):
                        user_data_final = _get_user_data(self.access_token, timeout, retry)
                        final_anlas = user_data_final.get("subscription", {}).get("trainingStepsLeft")
                        if final_anlas is not None:
                            print(f"[NovelAI] Generation cost: {start_anlas - final_anlas} Anlas")
                            print(f"[NovelAI] Anlas (post-gen): {final_anlas}")
                except Exception as e: print(f"[NovelAI] Anlas tracking failed (post-gen): {e}")

            # Save metadata JSON
            with _naid_profile_step(profile_enabled, "extract metadata"):
                metadata = get_metadata(image_bytes)

            ## save image metadata to a sidecar file to make it easier to import with services such as Hydrus
            with _naid_profile_step(profile_enabled, "save metadata sidecar JSON"):
                save_metadata_json(action, d, file, metadata, model, params)

            with _naid_profile_step(profile_enabled, "bytes_to_image"):
                image = bytes_to_image(image_bytes, keep_alpha)
        except Exception as e:
            if option and option.get("ignore_errors", False): print("ignore error:", e)
            else: raise e
        finally:
            if profile_enabled:
                total_elapsed = time.perf_counter() - total_start
                print(f"[NovelAI][profile] TOTAL GenerateNAID.generate: {total_elapsed:.3f}s")

        return (image,metadata,)

# -------------------------------------------------
# Director Tool Augment Nodes
# -------------------------------------------------

def base_augment(access_token, output_dir, limit_opus_free, ignore_errors, req_type, image, options=None):
    w, h = image.shape[2], image.shape[1]
    if limit_opus_free and w * h > 1024 * 1024:
        w, h = calculate_resolution(1024 * 1024, (w, h))
            
    start_anlas = None
    try:
        user_data = _get_user_data(access_token)
        start_anlas = user_data.get("subscription", {}).get("trainingStepsLeft")
        if start_anlas is not None: print(f"[NovelAI] Anlas (pre-augment): {start_anlas}")
    except Exception as e: print(f"[NovelAI] Anlas tracking failed (pre-augment): {e}")
            
    base64_image = image_to_base64(resize_image(image, (w, h)))
    result_image = blank_image()
    try:
        zipped_bytes = augment_image(access_token, req_type, w, h, base64_image, options=options)
        with zipfile.ZipFile(io.BytesIO(zipped_bytes)) as zipped:
            image_bytes = zipped.read(zipped.infolist()[0])

        full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path("NAI_autosave", output_dir)
        file = f"{filename}_{counter:05}_.png"
        d = Path(full_output_folder)
        d.mkdir(exist_ok=True)
        (d / file).write_bytes(image_bytes)

        if start_anlas is not None:
            try:
                user_data_final = _get_user_data(access_token)
                final_anlas = user_data_final.get("subscription", {}).get("trainingStepsLeft")
                if final_anlas is not None:
                    print(f"[NovelAI] Augment cost: {start_anlas - final_anlas} Anlas")
                    print(f"[NovelAI] Anlas (post-augment): {final_anlas}")
            except Exception as e: print(f"[NovelAI] Anlas tracking failed (post-augment): {e}")

        result_image = bytes_to_image(image_bytes)
    except Exception as e:
        if ignore_errors: print("ignore error:", e)
        else: raise e

    return (result_image,)

class RemoveBGAugment:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), "limit_opus_free": ("BOOLEAN", { "default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE }), "ignore_errors": ("BOOLEAN", { "default": False }),}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "augment"
    CATEGORY = "NovelAI/director_tools"
    def augment(self, image, limit_opus_free, ignore_errors):
        return base_augment(self.access_token, self.output_dir, limit_opus_free, ignore_errors, "bg-removal", image)

class LineArtAugment:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), "limit_opus_free": ("BOOLEAN", { "default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE }), "ignore_errors": ("BOOLEAN", { "default": False }),}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "augment"
    CATEGORY = "NovelAI/director_tools"
    def augment(self, image, limit_opus_free, ignore_errors):
        return base_augment(self.access_token, self.output_dir, limit_opus_free, ignore_errors, "lineart", image)

class SketchAugment:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), "limit_opus_free": ("BOOLEAN", { "default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE }), "ignore_errors": ("BOOLEAN", { "default": False }),}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "augment"
    CATEGORY = "NovelAI/director_tools"
    def augment(self, image, limit_opus_free, ignore_errors):
        return base_augment(self.access_token, self.output_dir, limit_opus_free, ignore_errors, "sketch", image)

class ColorizeAugment:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), "limit_opus_free": ("BOOLEAN", { "default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE }), "ignore_errors": ("BOOLEAN", { "default": False }), "defry": ("INT", { "default": 0, "min": 0, "max": 5, "step": 1, "display": "number" }), "prompt": ("STRING", { "default": "", "multiline": True, "dynamicPrompts": False }),}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "augment"
    CATEGORY = "NovelAI/director_tools"
    def augment(self, image, limit_opus_free, ignore_errors, defry, prompt):
        return base_augment(self.access_token, self.output_dir, limit_opus_free, ignore_errors, "colorize", image, options={ "defry": defry, "prompt": prompt })

class EmotionAugment:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()
    strength_list = ["normal", "slightly_weak", "weak", "even_weaker", "very_weak", "weakest"]
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), "limit_opus_free": ("BOOLEAN", { "default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE }), "ignore_errors": ("BOOLEAN", { "default": False }), "mood": (["neutral", "happy", "sad", "angry", "scared", "surprised", "tired", "excited", "nervous", "thinking", "confused", "shy", "disgusted", "smug", "bored", "laughing", "irritated", "aroused", "embarrassed", "worried", "love", "determined", "hurt", "playful"], { "default": "neutral" }), "strength": (s.strength_list, { "default": "normal" }), "prompt": ("STRING", { "default": "", "multiline": True, "dynamicPrompts": False }),}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "augment"
    CATEGORY = "NovelAI/director_tools"
    def augment(self, image, limit_opus_free, ignore_errors, mood, strength, prompt):
        prompt = f"{mood};;{prompt}"
        defry = EmotionAugment.strength_list.index(strength)
        return base_augment(self.access_token, self.output_dir, limit_opus_free, ignore_errors, "emotion", image, options={ "defry": defry, "prompt": prompt })

class DeclutterAugment:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), "limit_opus_free": ("BOOLEAN", { "default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE }), "ignore_errors": ("BOOLEAN", { "default": False }),}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "augment"
    CATEGORY = "NovelAI/director_tools"
    def augment(self, image, limit_opus_free, ignore_errors):
        return base_augment(self.access_token, self.output_dir, limit_opus_free, ignore_errors, "declutter", image)

# -------------------------------------------------
# Anlas Tracker (Visual Node)
# -------------------------------------------------

class AnlasTrackerNAID:
    def __init__(self):
        self.access_token = get_access_token()
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": { "trigger": ("*",) } # Allows chaining to control execution order
        }

    RETURN_TYPES = ("INT", "STRING",)
    RETURN_NAMES = ("anlas_int", "anlas_string",)
    FUNCTION = "get_anlas"
    CATEGORY = "NovelAI/utils"
    
    def get_anlas(self, trigger=None):
        anlas_count = 0
        try:
            user_data = _get_user_data(self.access_token)
            anlas_count = user_data.get("subscription", {}).get("trainingStepsLeft", 0)
            print(f"[NovelAI] Current Anlas Balance: {anlas_count}")
        except Exception as e:
            print(f"[NovelAI] Failed to fetch Anlas balance: {e}")
            return (0, "Error fetching Anlas")
            
        return (anlas_count, f"{anlas_count} Anlas")

# -------------------------------------------------
# V4 Base / Negative Prompt nodes
# -------------------------------------------------

class V4BasePrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"base_caption": ("STRING", { "multiline": True }),}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert"
    CATEGORY = "NovelAI/v4"
    def convert(self, base_caption):
        return (base_caption,)

class V4NegativePrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"negative_caption": ("STRING", { "multiline": True }),}}
    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert"
    CATEGORY = "NovelAI/v4"
    def convert(self, negative_caption):
        return (negative_caption,)


# GenerateNAID node extension: character prompt slots added
# Add 'characters' to INPUT_TYPES of existing GenerateNAID and internally assemble char_captions in a metadata.yaml structure.
# Replace existing GenerateNAID.INPUT_TYPES

old_generate_naid_input_types = GenerateNAID.INPUT_TYPES

def new_generate_naid_input_types(s):
    types = old_generate_naid_input_types()
    types["required"]["characters"] = ("CHARACTER_LIST_NAI", {"default": [], "forceInput": False, "multiline": True})
    return types
GenerateNAID.INPUT_TYPES = classmethod(new_generate_naid_input_types)

# Add 'characters' to INPUT_TYPES of existing GenerateNAID and internally assemble char_captions in a metadata.yaml structure.

old_generate_naid_generate = GenerateNAID.generate

def new_generate_naid_generate(self, *args, **kwargs):
    # characters is always the last argument
    import inspect
    sig = inspect.signature(old_generate_naid_generate)
    params = list(sig.parameters.keys())
    # Existing arguments + characters + option
    option = kwargs.get('option', None)
    characters = kwargs.get('characters', None)
    if characters is None and len(args) >= len(params):
        characters = args[len(params)-1]
    # Build v4_prompt
    if option is None:
        option = {}
    if characters:
        # Same structure as metadata.yaml
        char_captions = []
        for idx, c in enumerate(characters):
            centers = c.get("centers", [])
            # Warn if centers is empty or None
            if not centers or centers is None or not isinstance(centers, list) or not centers:
                print(f"[WARN] Character index {idx} centers is empty or invalid: {centers}")
            else:
                # If centers is not in [{"x":..., "y":...}] format, fix it
                if isinstance(centers, dict):
                    centers = [centers]
                elif isinstance(centers, list):
                    # If it's a list but not a list of dicts, fix it
                    if centers and not isinstance(centers[0], dict):
                        print(f"[WARN] Unexpected format for centers: {centers}")
                        centers = [{"x": centers[0], "y": centers[1]}] if len(centers) == 2 else []
            char_captions.append({
                "char_caption": c.get("char_caption", ""),
                "centers": centers
            })
        if char_captions:
            if "v4_prompt" not in option:
                option["v4_prompt"] = {"caption": {}}
            if "caption" not in option["v4_prompt"]:
                option["v4_prompt"]["caption"] = {}
            option["v4_prompt"]["caption"]["char_captions"] = char_captions
    # Print option contents before sending to API
    print("[DEBUG] API送信前 option:", option)
    kwargs['option'] = option
    # Remove characters from kwargs
    if 'characters' in kwargs:
        del kwargs['characters']
    return old_generate_naid_generate(self, *args[:len(params)-1], **kwargs)

GenerateNAID.generate = new_generate_naid_generate

# -------------------------------------------------
# Registration
# -------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "GenerateNAID": GenerateNAID,
    "ModelOptionNAID": ModelOption,
    "Img2ImgOptionNAID": Img2ImgOption,
    "InpaintingOptionNAID": InpaintingOption,
    "VibeTransferOptionNAID": VibeTransferOption,
    "NetworkOptionNAID": NetworkOption,
    "AutosaveOptionNAID": AutosaveOption,
    "CharacterReferenceOptionNAID": CharacterReferenceOption,
    "AnlasTrackerNAID": AnlasTrackerNAID, # New node
    "MaskImageToNAID": ImageToNAIMask,
    "PromptToNAID": PromptToNAID,
    "RemoveBGNAID": RemoveBGAugment,
    "LineArtNAID": LineArtAugment,
    "SketchNAID": SketchAugment,
    "ColorizeNAID": ColorizeAugment,
    "EmotionNAID": EmotionAugment,
    "DeclutterNAID": DeclutterAugment,
    "CharacterNAI": CharacterNAI,
    "CharacterConcatenateNAI": CharacterConcatenateNAI,
    "GetImageMetadata": GetImageMetadata,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GenerateNAID": "Generate ✒️🅝🅐🅘",
    "ModelOptionNAID": "ModelOption ✒️🅝🅐🅘",
    "Img2ImgOptionNAID": "Img2ImgOption ✒️🅝🅐🅘",
    "InpaintingOptionNAID": "InpaintingOption ✒️🅝🅐🅘",
    "VibeTransferOptionNAID": "VibeTransferOption ✒️🅝🅐🅘",
    "NetworkOptionNAID": "NetworkOption ✒️🅝🅐🅘",
    "AutosaveOptionNAID": "AutosaveOption ✒️🅝🅐🅘",
    "CharacterReferenceOptionNAID": "Character Reference ✒️🅝🅐🅘",
    "AnlasTrackerNAID": "Anlas Tracker ✒️🅝🅐🅘", # New node
    "MaskImageToNAID": "Convert Mask Image ✒️🅝🅐🅘",
    "PromptToNAID": "Convert Prompt ✒️🅝🅐🅘",
    "RemoveBGNAID": "Remove BG ✒️🅝🅐🅘",
    "LineArtNAID": "LineArt ✒️🅝🅐🅘",
    "SketchNAID": "Sketch ✒️🅝🅐🅘",
    "ColorizeNAID": "Colorize ✒️🅝🅐🅘",
    "EmotionNAID": "Emotion ✒️🅝🅐🅘",
    "DeclutterNAID": "Declutter ✒️🅝🅐🅘",
    "CharacterNAI": "Character ✒️🅝🅐🅘",
    "CharacterConcatenateNAI": "CharacterConcatenate ✒️🅝🅐🅘",
    "GetImageMetadata": "Get Image Metadata ✒️🅝🅐🅘",
}
