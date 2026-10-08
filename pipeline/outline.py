"""Photo -> black-and-white outline drawing with the OpenAI image edit API.
From the robo-replicator backend, trimmed to the GPT image models.

Needs OPENAI_API_KEY. OPENAI_IMAGE_MODEL (default gpt-image-1) and OUTLINE_PROMPT
can be overridden in the environment.
"""
import base64
import os
from io import BytesIO

from openai import APIStatusError, OpenAI
from PIL import Image

OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1").strip()

OUTLINE_PROMPT = os.getenv(
    "OUTLINE_PROMPT",
    (
        "Convert this into a very simple black-and-white outline drawing of the main person "
        "in the foreground — the person closest to the camera and centered in the frame. "
        "Draw only that person. Ignore and omit the background entirely: no walls, furniture, "
        "objects, scenery, patterns, clutter, or other people. "
        "Use clean, minimal black line art on a pure white background. "
        "No grayscale, no shading, no color, no fills, and no textures. "
        "Preserve the person's main contours and recognizable silhouette with as few strokes as possible. "
        "The result should look like a cartoon line drawing of the person alone, suitable for a pen plotter robot."
    ),
)


class MissingApiKeyError(RuntimeError):
    """Raised when OPENAI_API_KEY is not configured."""


class OutlineGenerationError(RuntimeError):
    """Raised when OpenAI fails to produce an outline image."""


def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise MissingApiKeyError("OPENAI_API_KEY is not set. Add it to .env (see .env.example).")
    return api_key


def generate_outline(data: bytes, filename: str) -> bytes:
    """Image bytes -> PNG bytes of a black line drawing on white."""
    client = OpenAI(api_key=get_openai_api_key())
    try:
        response = client.images.edit(
            model=OPENAI_IMAGE_MODEL,
            image=("source.png", _prepare(data), "image/png"),
            prompt=OUTLINE_PROMPT,
            n=1,
            size="1024x1024",
            background="opaque",
            output_format="png",
        )
    except APIStatusError as exc:
        raise OutlineGenerationError(
            f"OpenAI rejected the request for {filename!r} (model={OPENAI_IMAGE_MODEL}): {exc.message}"
        ) from exc
    except Exception as exc:
        raise OutlineGenerationError(f"OpenAI image generation failed for {filename!r}: {exc}") from exc

    b64 = response.data[0].b64_json if response.data else None
    if not b64:
        raise OutlineGenerationError(f"OpenAI returned no image data for {filename!r}.")
    raw = base64.b64decode(b64)
    with Image.open(BytesIO(raw)) as decoded:
        decoded.verify()
    return raw


def _prepare(data: bytes) -> bytes:
    """PNG bytes, at most 2048 px on the longest side."""
    with Image.open(BytesIO(data)) as opened:
        rgba = opened.convert("RGBA")
    longest = max(rgba.width, rgba.height, 1)
    if longest > 2048:
        scale = 2048 / longest
        rgba = rgba.resize(
            (max(1, int(rgba.width * scale)), max(1, int(rgba.height * scale))),
            Image.Resampling.LANCZOS,
        )
    buffer = BytesIO()
    rgba.save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue()
