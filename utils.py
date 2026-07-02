import json
from hashlib import blake2b
import argon2
from xml.sax.saxutils import escape, unescape

import base64
import struct
import zlib
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


# Constants for skip CFG calculation
REFERENCE_RESOLUTION = 1011712  # 832 * 1216
NAI_V4_5_SIGMA_MULTIPLIER = 58
DEFAULT_SIGMA_MULTIPLIER = 19


def calculate_skip_cfg_above_sigma(w, h, model):
    if model == "nai-diffusion-4-5-full":
        return (w * h / REFERENCE_RESOLUTION) ** 0.5 * NAI_V4_5_SIGMA_MULTIPLIER

    return (w * h / REFERENCE_RESOLUTION) ** 0.5 * DEFAULT_SIGMA_MULTIPLIER


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


_NAI_METADATA_KEY_ALIASES = {
    "comment": "Comment",
    "description": "Description",
    "generation_time": "Generation_time",
    "software": "Software",
    "source": "Source",
    "title": "Title",
    "documentname": "Title",
    "imagedescription": "Description",
}

_EXIF_METADATA_SKIP_KEYS = {
    "ExifOffset",
}


def _canonical_metadata_key(key):
    key = str(key)
    return _NAI_METADATA_KEY_ALIASES.get(key.lower(), key)


def _decode_metadata_value(value, encoding="utf-8"):
    if isinstance(value, bytes):
        return value.decode(encoding, errors="replace")
    if isinstance(value, (str, int, float, bool)):
        return value
    return None


def _store_metadata_value(metadata, key, value, overwrite=True):
    key = _canonical_metadata_key(key)

    if key in {"exif", "icc_profile"}:
        return

    value = _decode_metadata_value(value)
    if value is None:
        return

    if overwrite or key not in metadata:
        metadata[key] = value


def _clean_xmp_value(value):
    value = value.strip()
    if value.startswith("<![CDATA[") and value.endswith("]]>"):
        value = value[len("<![CDATA["):-len("]]>")]
    return unescape(value.strip())


def _extract_nai_xmp_metadata(xmp):
    xmp_text = _decode_metadata_value(xmp)
    if not xmp_text:
        return {}

    metadata = {}

    for match in re.finditer(
        r"<(?:[\w.-]+:)?([A-Za-z_][\w.-]*)\b[^>]*>(.*?)</(?:[\w.-]+:)?\1>",
        xmp_text,
        re.DOTALL,
    ):
        key = _canonical_metadata_key(match.group(1))
        if key in {"Comment", "Description", "Generation_time", "Software", "Source", "Title"}:
            metadata[key] = _clean_xmp_value(match.group(2))

    for match in re.finditer(
        r"\b(?:[\w.-]+:)?([A-Za-z_][\w.-]*)=(['\"])(.*?)\2",
        xmp_text,
        re.DOTALL,
    ):
        key = _canonical_metadata_key(match.group(1))
        if key in {"Comment", "Description", "Generation_time", "Software", "Source", "Title"}:
            metadata.setdefault(key, _clean_xmp_value(match.group(3)))

    return metadata


def _extract_png_text_metadata(image_bytes):
    if not isinstance(image_bytes, bytes):
        return {}

    if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return {}

    metadata = {}
    offset = 8

    while offset + 8 <= len(image_bytes):
        length = struct.unpack(">I", image_bytes[offset:offset + 4])[0]
        chunk_type = image_bytes[offset + 4:offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        data = image_bytes[data_start:data_end]

        if data_end + 4 > len(image_bytes):
            break

        try:
            if chunk_type == b"tEXt":
                key, value = data.split(b"\x00", 1)
                _store_metadata_value(
                    metadata,
                    key.decode("latin-1", errors="replace"),
                    value.decode("latin-1", errors="replace"),
                )

            elif chunk_type == b"zTXt":
                key, rest = data.split(b"\x00", 1)
                compression_method = rest[0]
                compressed_value = rest[1:]

                if compression_method == 0:
                    _store_metadata_value(
                        metadata,
                        key.decode("latin-1", errors="replace"),
                        zlib.decompress(compressed_value).decode("latin-1", errors="replace"),
                    )

            elif chunk_type == b"iTXt":
                key, rest = data.split(b"\x00", 1)
                compression_flag = rest[0]
                compression_method = rest[1]
                rest = rest[2:]

                language_tag, rest = rest.split(b"\x00", 1)
                translated_key, text = rest.split(b"\x00", 1)

                if compression_flag == 1 and compression_method == 0:
                    text = zlib.decompress(text)

                _store_metadata_value(
                    metadata,
                    key.decode("utf-8", errors="replace"),
                    text.decode("utf-8", errors="replace"),
                )
        except Exception:
            pass

        offset = data_end + 4

        if chunk_type == b"IEND":
            break

    return metadata


def _decode_exif_user_comment(value):
    if isinstance(value, str):
        return value

    if not isinstance(value, bytes):
        return None

    if value.startswith(b"ASCII\x00\x00\x00"):
        return value[8:].decode("ascii", errors="replace").rstrip("\x00")

    if value.startswith(b"UNICODE\x00"):
        return value[8:].decode("utf-16", errors="replace").rstrip("\x00")

    if value.startswith(b"JIS\x00\x00\x00\x00\x00"):
        return value[8:].decode("shift_jis", errors="replace").rstrip("\x00")

    return value.decode("utf-8", errors="replace").rstrip("\x00")


def _extract_exif_metadata(image):
    metadata = {}

    if not hasattr(image, "_getexif"):
        return metadata

    try:
        exif = image._getexif()
    except Exception:
        return metadata

    if not exif:
        return metadata

    for key, value in exif.items():
        tag = ExifTags.TAGS.get(key, str(key))

        if tag in _EXIF_METADATA_SKIP_KEYS:
            continue

        if tag == "UserComment":
            comment = _decode_exif_user_comment(value)
            if comment:
                metadata.setdefault("Comment", comment)
            continue

        canonical_key = _canonical_metadata_key(tag)
        decoded_value = _decode_metadata_value(value)

        if decoded_value is None:
            continue

        if tag == "Software" and isinstance(decoded_value, str) and decoded_value.startswith("NovelAI Diffusion"):
            metadata.setdefault("Source", decoded_value)
            metadata.setdefault("Software", "NovelAI")
            continue

        if canonical_key != tag:
            metadata.setdefault(canonical_key, decoded_value)
            continue

        if canonical_key in {"Comment", "Description", "Generation_time", "Software", "Source", "Title"}:
            metadata.setdefault(canonical_key, decoded_value)

    return metadata


def _normalize_nai_comment_metadata(metadata):
    comment = metadata.get("Comment")

    if not isinstance(comment, str):
        return metadata

    try:
        parsed_comment = json.loads(comment)
    except json.JSONDecodeError:
        return metadata

    if not isinstance(parsed_comment, dict):
        return metadata

    nested_comment = parsed_comment.get("Comment")

    if isinstance(nested_comment, str):
        for key, value in parsed_comment.items():
            if key == "Comment":
                continue

            _store_metadata_value(metadata, key, value, overwrite=False)

        try:
            metadata["Comment"] = json.loads(nested_comment)
        except json.JSONDecodeError:
            metadata["Comment"] = nested_comment
    else:
        metadata["Comment"] = parsed_comment

    return metadata


def get_metadata(image):
    raw_image_bytes = image if isinstance(image, bytes) else None

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

    metadata = {}

    if raw_image_bytes:
        try:
            metadata.update(_extract_png_text_metadata(raw_image_bytes))
        except Exception as e:
            print(f"Warning: Could not parse PNG text metadata: {e}")

    if hasattr(i, "info"):
        for key, value in i.info.items():
            if key in {"exif", "icc_profile"}:
                continue

            if key in {"xmp", "XML:com.adobe.xmp"}:
                for xmp_key, xmp_value in _extract_nai_xmp_metadata(value).items():
                    _store_metadata_value(metadata, xmp_key, xmp_value)
                continue

            _store_metadata_value(metadata, key, value)

    if hasattr(i, "text"):
        try:
            for key, value in i.text.items():
                _store_metadata_value(metadata, key, value)
        except Exception:
            pass

    for key, value in _extract_exif_metadata(i).items():
        metadata.setdefault(key, value)

    _normalize_nai_comment_metadata(metadata)

    return (json.dumps(metadata, ensure_ascii=False),)


def get_nai_comment(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return image.info.get("Comment")


def build_nai_xmp(comment):
    if not comment:
        return None

    escaped_comment = escape(str(comment))
    return f"""<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about="" xmlns:nai="https://novelai.net/ns/1.0/">
      <nai:Comment>{escaped_comment}</nai:Comment>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>""".encode("utf-8")


def save_image_with_metadata(image_bytes, output_path, output_format="png", webp_quality=85):
    output_format = (output_format or "png").lower()

    if output_format == "png":
        output_path.write_bytes(image_bytes)
        return

    image = Image.open(io.BytesIO(image_bytes))
    xmp = build_nai_xmp(get_nai_comment(image_bytes))

    save_kwargs = {
        "format": "WEBP",
        "quality": int(webp_quality),
        "method": 6,
    }

    if xmp:
        save_kwargs["xmp"] = xmp

    if "exif" in image.info:
        save_kwargs["exif"] = image.info["exif"]

    if "icc_profile" in image.info:
        save_kwargs["icc_profile"] = image.info["icc_profile"]

    image.save(output_path, **save_kwargs)


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

def save_metadata_json(action, d, file, metadata, model, params):
    metadata_dict = {"metadata": {}}
    try:
        # Extract the metadata string from the tuple
        if isinstance(metadata, tuple) and len(metadata) > 0:
            metadata_str = metadata[0]
        else:
            metadata_str = str(metadata)

        parsed_metadata = None

        # First, convert the string representation to an actual dictionary
        if isinstance(metadata_str, str):
            try:
                parsed_metadata = json.loads(metadata_str)
            except json.JSONDecodeError:
                # Extract the Comment value - this is the JSON string we want to parse
                import ast
                try:
                    # Convert the string representation of a dict to an actual dict
                    parsed_metadata = ast.literal_eval(metadata_str)
                except (SyntaxError, ValueError):
                    parsed_metadata = None

        if isinstance(parsed_metadata, dict):
            comment = parsed_metadata.get("Comment")

            if isinstance(comment, dict):
                metadata_dict["metadata"] = comment
            elif isinstance(comment, str):
                try:
                    metadata_dict["metadata"] = json.loads(comment)
                except json.JSONDecodeError:
                    metadata_dict["metadata"] = parsed_metadata
            else:
                metadata_dict["metadata"] = parsed_metadata
        else:
            # Handle case where metadata isn't a string or doesn't contain Comment
            metadata_dict["metadata"] = {"raw": metadata_str}

    except Exception as e:
        print(f"Warning: Could not parse metadata: {e}")
        if isinstance(metadata, tuple) and len(metadata) > 0:
            metadata_dict["metadata"] = {"raw": metadata[0]}
        else:
            metadata_dict["metadata"] = {"raw": str(metadata)}
    # Add comfyui_data as before
    metadata_dict["comfyui_data"] = {
        "workflow": {
            "model": model,
            "action": action,
            "parameters": params
        }
    }
    metadata_file = f"{file}.json"
    (d / metadata_file).write_text(json.dumps(metadata_dict, indent=2, ensure_ascii=False))
