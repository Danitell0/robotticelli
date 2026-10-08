"""Vin: turn a portrait photo into a simple smiley-style cartoon SVG.

The photo is sent to Kimi-K2.6 on Nebius Token Factory, which replies with a
playful line drawing as SVG: stroke-only paths, few lines, so the arm can trace
it with a pen. From robo-replicator's scripts/portrait_cartoon.py.

Needs NEBIUS_API_KEY. NEBIUS_MODEL overrides the model.
"""
import base64
import os
import re
from datetime import datetime
from pathlib import Path

import cv2
from openai import OpenAI

from config import IMAGES_DIR

CAMERA_DIR = IMAGES_DIR / "camera"  # webcam captures (git-ignored)
CARTOON_DIR = IMAGES_DIR / "processed"

NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_MODEL = "moonshotai/Kimi-K2.6"

# 1080p webcam; the driver falls back to its nearest supported mode.
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080

CARTOON_PROMPT = """\
You are a cartoonist drawing for a robot arm that holds a single pen.

Look at this portrait photo and draw the person as a funny, super simple
cartoon, like a smiley-face doodle. Exaggerate for fun. Keep it friendly.

First write ONE line starting with "Features:" naming the 2 or 3 things that
make this person recognizable (hair shape and length, glasses, beard, hat...).
Then write the SVG.

Layout guide for the 512x512 canvas (y grows downward):
- Head: one closed oval centered near (256, 260), about 220 wide, 260 tall.
- Hair: outline it around the top and sides of the head, matching the photo.
- Eyes: two small dots or happy arcs at y = 240, x = 210 and x = 302.
- Glasses (only if worn): one ring around EACH eye plus a short bridge between.
- Nose: optional, one tiny curve near (256, 285).
- Mouth: a big smiling arc around y = 320, from x = 200 to x = 312.
- Do not draw a neck, body, or background.

SVG rules (strict):
- <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
- Use only <path> elements. Straight lines (M, L) and Bezier curves (C, Q) are fine.
- Every path: fill="none" stroke="black" stroke-width="4" stroke-linecap="round" stroke-linejoin="round".
- No fills, no text, no gradients, no images, no transforms, no <style>.
- At most 15 paths. Fewer, longer, continuous strokes are better.
- No markdown code fences.
"""


class CartoonError(RuntimeError):
    """Raised when a photo cannot be saved or turned into a cartoon."""


def save_frame(frame) -> Path:
    """Write a BGR frame to images/camera/portrait_<timestamp>.jpg."""
    CAMERA_DIR.mkdir(parents=True, exist_ok=True)
    path = CAMERA_DIR / f"portrait_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
    if not cv2.imwrite(str(path), frame):
        raise CartoonError(f"Could not write photo: {path}")
    return path


def encode_image(path: Path, max_side: int = 1024) -> str:
    """Downscale to keep the request small, return a JPEG data URL."""
    image = cv2.imread(str(path))
    if image is None:
        raise CartoonError(f"Could not read image: {path}")
    scale = max_side / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise CartoonError(f"Could not encode image: {path}")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode()


def extract_svg(text: str) -> str:
    match = re.search(r"<svg\b.*?</svg>", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        raise CartoonError("The model reply contained no <svg> element.")
    return match.group(0)


def cartoonize(photo: Path, model: str | None = None) -> Path:
    """Send the photo to Kimi and save the cartoon as images/processed/<photo stem>.svg."""
    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        raise CartoonError("Set NEBIUS_API_KEY in .env (see .env.example) or in your environment.")

    client = OpenAI(base_url=NEBIUS_BASE_URL, api_key=api_key)
    response = client.chat.completions.create(
        model=model or os.environ.get("NEBIUS_MODEL", DEFAULT_MODEL),
        max_tokens=8000,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": encode_image(photo)}},
                    {"type": "text", "text": CARTOON_PROMPT},
                ],
            }
        ],
        # Kimi otherwise spends the whole token budget reasoning before the SVG.
        extra_body={"chat_template_kwargs": {"thinking": False}},
    )
    choice = response.choices[0]
    reply = choice.message.content or ""
    try:
        svg = extract_svg(reply)
    except CartoonError as error:
        raise CartoonError(f"{error} (finish_reason={choice.finish_reason})") from error

    CARTOON_DIR.mkdir(parents=True, exist_ok=True)
    path = CARTOON_DIR / f"{photo.stem}.svg"
    path.write_text(svg + "\n", encoding="utf-8")
    return path
