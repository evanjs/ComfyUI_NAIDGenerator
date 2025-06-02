from hashlib import blake2b
import argon2

import base64
import dotenv
from os import environ as env
import io
import re
import requests
from requests.adapters import HTTPAdapter, Retry
import comfy.utils

import torch
import numpy as np
from PIL import Image, ImageOps, ExifTags

# cherry-picked from novelai_api.utils
def argon_hash(email: str, password: str, size: int, domain: str) -> str:
    pre_salt = f"{password[:6]}{email}{domain}"
    blake = blake2b(digest_size=16)
    blake.update(pre_salt.encode())
    salt = blake.digest()
    raw = argon2.low_level.hash_secret_raw(password.encode(), salt, 2, int(2000000 / 1024), 1, size, argon2.low_level.Type.ID,)
    hashed = base64.urlsafe_b64encode(raw).decode()
    return hashed

def get_access_key(email: str, password: str) -> str:
    return argon_hash(email, password, 64, "novelai_data_access_key")[:64]


def login(key) -> str:
    response = requests.post(f"https://api.novelai.net/user/login", json={ "key": key })
    response.raise_for_status()
    return response.json()["accessToken"]

def get_access_token():
    dotenv.load_dotenv()
    if "NAI_ACCESS_TOKEN" in env:
        access_token = env["NAI_ACCESS_TOKEN"]
    elif "NAI_ACCESS_KEY" in env:
        print("ComfyUI_NAIDGenerator: NAI_ACCESS_KEY is deprecated. use NAI_ACCESS_TOKEN instead.")
        access_key = env["NAI_ACCESS_KEY"]
    elif "NAI_USERNAME" in env and "NAI_PASSWORD" in env:
        print("ComfyUI_NAIDGenerator: NAI_USERNAME is deprecated. use NAI_ACCESS_TOKEN instead.")
        username = env["NAI_USERNAME"]
        password = env["NAI_PASSWORD"]
        access_key = get_access_key(username, password)
    else:
        raise RuntimeError("Please ensure that NAI_API_TOKEN is set in ComfyUI/.env file.")

    if not access_token:
        access_token = login(access_key)
    return access_token


BASE_URL="https://image.novelai.net"
def generate_image(access_token, prompt, model, action, parameters, timeout=None, retry=None):
    data = { "input": prompt, "model": model, "action": action, "parameters": parameters }

    request = requests
    if retry is not None and retry > 1:
        retries = Retry(total=retry, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["POST"])
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retries))
        request = session

    response = request.post(f"{BASE_URL}/ai/generate-image", json=data, headers={ "Authorization": f"Bearer {access_token}" }, timeout=timeout)
    response.raise_for_status()
    return response.content

def augment_image(access_token, req_type, width, height, image, options={}, timeout=None, retry=None):
    data = { "req_type": req_type, "width": width, "height": height, "image": image }
    if options:
        data.update(options)

    request = requests
    if retry is not None and retry > 1:
        retries = Retry(total=retry, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["POST"])
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retries))
        request = session

    response = request.post(f"{BASE_URL}/ai/augment-image", json=data, headers={ "Authorization": f"Bearer {access_token}" }, timeout=timeout)
    response.raise_for_status()
    return response.content


def image_to_base64(image):
    i = 255. * image[0].cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    image_bytesIO = io.BytesIO()
    img.save(image_bytesIO, format="png")
    return base64.b64encode(image_bytesIO.getvalue()).decode()

def naimask_to_base64(image):
    i = 255. * image[0].cpu().numpy()
    i = np.clip(i, 0, 255).astype(np.uint8)
    alpha = np.sum(i, axis=-1) > 0
    alpha = np.uint8(alpha * 255)
    rgba = np.dstack((i, alpha))
    img = Image.fromarray(rgba)
    image_bytesIO = io.BytesIO()
    img.save(image_bytesIO, format="png")
    return base64.b64encode(image_bytesIO.getvalue()).decode()

def bytes_to_image(image_bytes, keep_alpha=True):
    i = Image.open(io.BytesIO(image_bytes))
    i = ImageOps.exif_transpose(i)
    if not keep_alpha:
        i = i.convert("RGB")
    image = np.array(i).astype(np.float32) / 255.0
    return torch.from_numpy(image)[None,]

def blank_image():
    return torch.tensor([[[0]]])

def resize_image(image, size_to):
    samples = image.movedim(-1,1)
    w, h = size_to
    s = comfy.utils.common_upscale(samples, w, h, "bilinear", "disabled")
    s = s.movedim(1,-1)
    return s

def resize_to_naimask(mask, image_size=None, is_v4=False):
    samples = mask.movedim(-1,1)
    w, h = (samples.shape[3], samples.shape[2]) if not image_size else image_size
    width = int(np.ceil(w / 64) * 8)
    height = int(np.ceil(h / 64) * 8)
    s = comfy.utils.common_upscale(samples, width, height, "nearest-exact", "disabled")
    if is_v4:
        s = comfy.utils.common_upscale(s, width*8, height*8, "nearest-exact", "disabled")
    s = s.movedim(1,-1)
    return s

def calculate_resolution(pixel_count, aspect_ratio):
    pixel_count = pixel_count / 4096
    w, h = aspect_ratio
    k = (pixel_count * w / h) ** 0.5
    width = int(np.floor(k) * 64)
    height = int(np.floor(k * h / w) * 64)
    return width, height

def calculate_skip_cfg_above_sigma(w, h, model):
    if model == "nai-diffusion-4-5-full":
        return 58
    
    # 832 * 1216
    return (w * h / 1011712) ** 0.5 * 19


def prompt_to_stack(sentence):
    result = []
    current_str = ""
    stack = [{ "weight": 1.0, "data": result }]

    for i, c in enumerate(sentence):
        if c in '()':
            # current_str = current_str.strip()
            if c == '(':
                if current_str: stack[-1]["data"].append(current_str)
                stack[-1]["data"].append({ "weight": 1.0, "data": [] });
                stack.append(stack[-1]["data"][-1])
            elif c == ')':
                searched = re.search(r"^(.*):(-?[0-9\.]+)$", current_str)
                current_str, weight = searched.groups() if searched else (current_str, 1.1)
                if current_str: stack[-1]["data"].append(current_str)
                stack[-1]["weight"] = float(weight)
                if stack[-1]["data"] != result:
                    stack.pop()
                else: # no more to pop
                    print("error  :", sentence);
                    print(f"col {i:>3}:", " " * i + "^")
                    # raise Exception('Error durring parsing parentheses', sentence, i, c)
            current_str = ""
        else:
            current_str += c

    if current_str:
        stack[-1]["data"].append(current_str)

    return result

def prompt_stack_to_nai(l, weight_per_brace=0.05, syntax_mode="brace"):
    result = ""
    for el in l:
        if isinstance(el, dict):
            weight = el["weight"]
            prompt = prompt_stack_to_nai(el["data"], weight_per_brace, syntax_mode)
            if weight < 0:
                syntax_mode = "numeric"
            if syntax_mode == "brace":
                brace_count = round((weight - 1.0) / weight_per_brace)
                result += "{" * brace_count + "[" * -brace_count + prompt + "}" * brace_count + "]" * -brace_count
            elif syntax_mode == "numeric":
                result += f"{weight:g}::{prompt} ::"
        else:
            result += el
    return result

def prompt_to_nai(prompt, weight_per_brace=0.05, syntax_mode="brace"):
    return prompt_stack_to_nai(prompt_to_stack(prompt.replace("\(", "（").replace("\)", "）")), weight_per_brace, syntax_mode).replace("（", "(").replace("）",")")


def get_metadata(image):
    if isinstance(image, bytes):
        # Handle bytes input
        i = Image.open(io.BytesIO(image))
    elif isinstance(image, (list, tuple)) and len(image) > 0:
        img_data = image[0]
        if hasattr(img_data, 'cpu') and hasattr(img_data, 'numpy'):
            # Handle tensor input
            i = Image.fromarray(np.uint8(255 * img_data.cpu().numpy()))
        elif isinstance(img_data, np.ndarray):
            # Handle numpy array input
            i = Image.fromarray(np.uint8(255 * img_data))
        else:
            # Assume it's already a PIL image
            i = img_data
    else:
        i = image

    metadata = {}
    if hasattr(i, '_getexif') and i._getexif() is not None:
        try:
            metadata = {ExifTags.TAGS[k]: str(v) for k, v in i._getexif().items()
                        if k in ExifTags.TAGS and isinstance(v, (str, int, float, bool))}
        except:
            pass

    if hasattr(i, 'info'):
        if "Comment" in i.info:
            metadata["Comment"] = i.info["Comment"]
        else:
            for key, value in i.info.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value

    return (str(metadata),)


def merge_dicts_non_empty(dict1, dict2):
    """Merges two dictionaries recursively, prioritizing non-None and non-empty values."""
    merged = {}

    # Use a simple union of keys instead of set operations to avoid hashing issues
    all_keys = list(dict1.keys()) + [k for k in dict2.keys() if k not in dict1]

    for k in all_keys:
        val1 = dict1.get(k)
        val2 = dict2.get(k)

        if isinstance(val1, dict) and isinstance(val2, dict):
            merged[k] = merge_dicts_non_empty(val1, val2)
        elif isinstance(val1, list) and isinstance(val2, list):
            # Handle list merging safely without dict.fromkeys()
            combined_list = []
            seen = set()

            # Only add items to the result if they're hashable and not already seen
            for item in val1 + val2:
                try:
                    item_hash = hash(item)
                    if item_hash not in seen:
                        seen.add(item_hash)
                        combined_list.append(item)
                except TypeError:
                    # If item isn't hashable (like a dict), just add it
                    combined_list.append(item)

            merged[k] = combined_list
        elif val1 and val2:
            merged[k] = val1
        elif val1:
            merged[k] = val1
        elif val2:
            merged[k] = val2
        else:
            pass
    return merged
