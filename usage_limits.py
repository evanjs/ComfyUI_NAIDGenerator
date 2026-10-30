"""Helpers for NovelAI Diffusion V5 Opus usage-limit reporting."""

# User-provided account sample: 93% allowance corresponded to about 1,609
# standard-resolution generations, with a 6,048-second interval per 1% refill.
SAMPLE_USAGE_PERCENT = 93
SAMPLE_REMAINING_IMAGES = 1609
SAMPLE_SECONDS_PER_PERCENT = 6048
SECONDS_PER_DAY = 24 * 60 * 60


def get_subscription(user_data):
    """Return subscription data from a wrapped or direct subscription response."""
    if not isinstance(user_data, dict):
        return {}
    subscription = user_data.get("subscription")
    if isinstance(subscription, dict):
        return subscription
    if "tier" in user_data or "trainingStepsLeft" in user_data or "usage" in user_data:
        return user_data
    return {}


def get_opus_usage(user_data):
    """Normalize the subscription usage payload into displayable estimates."""
    subscription = get_subscription(user_data)
    usage = subscription.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}

    raw_percent = usage.get("percent")
    percent = int(raw_percent) if isinstance(raw_percent, (int, float)) and not isinstance(raw_percent, bool) else None
    if percent is not None:
        percent = max(0, min(100, percent))

    seconds_to_next_percent = usage.get("timeUntilNextPercent")
    if not isinstance(seconds_to_next_percent, (int, float)) or isinstance(seconds_to_next_percent, bool) or seconds_to_next_percent <= 0:
        seconds_to_next_percent = None

    estimated_images = None
    refill_percent_per_day = None
    refill_images_per_day = None
    if percent is not None:
        estimated_images = round(percent * SAMPLE_REMAINING_IMAGES / SAMPLE_USAGE_PERCENT)
    if seconds_to_next_percent is not None:
        refill_percent_per_day = SECONDS_PER_DAY / seconds_to_next_percent
        estimated_capacity = SAMPLE_REMAINING_IMAGES * 100 / SAMPLE_USAGE_PERCENT
        refill_images_per_day = round(refill_percent_per_day * estimated_capacity / 100)

    return {
        "percent": percent,
        "is_negative": bool(usage.get("isNegative", False)),
        "time_until_next_percent": seconds_to_next_percent,
        "estimated_images": estimated_images,
        "refill_percent_per_day": refill_percent_per_day,
        "refill_images_per_day": refill_images_per_day,
    }


def format_opus_usage(usage):
    """Format the usage tracker output; image counts are sample-based estimates."""
    percent = usage.get("percent")
    if percent is None:
        return "Opus V5 usage unavailable"

    estimated_images = usage.get("estimated_images")
    image_text = f" (~{estimated_images} images)" if estimated_images is not None else ""
    lines = [f"{percent}% remaining{image_text}"]

    refill_percent = usage.get("refill_percent_per_day")
    refill_images = usage.get("refill_images_per_day")
    if refill_percent is not None:
        refill_text = f" (~{refill_images} images)" if refill_images is not None else ""
        lines.append(f"Currently refills at {refill_percent:.1f}% per day{refill_text}.")

    lines.append("Image counts are approximate, based on the supplied standard-resolution sample.")
    return "\n".join(lines)


def opus_usage_result(user_data):
    """Return ComfyUI's node result plus text for its inline preview widget."""
    usage = get_opus_usage(user_data)
    percent = usage["percent"]
    info = format_opus_usage(usage)
    return {"ui": {"text": [info]}, "result": (percent if percent is not None else 0, info)}


def should_block_v5_generation(subscription, model):
    """Whether an exhausted active Opus usage pool blocks a V5 request."""
    if not isinstance(subscription, dict) or not isinstance(model, str):
        return False
    if subscription.get("tier") != 3 or subscription.get("active") is not True:
        return False
    if "nai-diffusion-5" not in model.lower():
        return False

    usage = subscription.get("usage")
    if not isinstance(usage, dict):
        return False
    percent = usage.get("percent")
    return isinstance(percent, (int, float)) and not isinstance(percent, bool) and percent <= 0
