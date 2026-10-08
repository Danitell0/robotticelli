"""Take a photo and turn it into strokes the arm can draw.

Built on pieces from the team web app (see pipeline/):
  - the "Vin" cartoonist, pipeline/vin.py (photos land in images/camera/)
  - the OpenAI outline, vectorizer and SVG parser, through image_draw.py

Ways to turn the photo into lines:
  vin    Vin (Kimi-K2.6 on Nebius) draws a smiley-style cartoon portrait as an SVG.
         Needs NEBIUS_API_KEY in .env.
  ai     OpenAI gpt-image-1 redraws it as an outline, then it gets vectorized.
         Needs OPENAI_API_KEY in .env.
  edges  trace the photo's edges with OpenCV. Free, rougher.

Command line, from the repo root:
    python photo.py                      # take a photo, Vin cartoon
    python photo.py --method edges
    python photo.py --photo portrait_x.jpg --method vin
"""
import argparse
import os
import time
from pathlib import Path

from image_draw import (PROCESSED_DIR, ImageDrawError, image_to_strokes, preview, render_paths, simplify,
                        svg_to_paths)

METHODS = {
    "vin": "Vin cartoon        smiley-style portrait by Kimi on Nebius",
    "ai": "AI outline         OpenAI redraws it, then it gets vectorized",
    "edges": "Trace the edges    free, no AI, rougher",
}


class PhotoError(RuntimeError):
    pass


_vin = None


def vin():
    """pipeline/vin.py, imported on first use so the menu starts without openai installed."""
    global _vin
    if _vin is None:
        from pipeline import vin as module
        _vin = module
    return _vin


def camera_dir() -> Path:
    return vin().CAMERA_DIR


def recent_photos(limit: int = 6) -> list[Path]:
    """Newest photos first, from the same folder the web app saves to."""
    folder = camera_dir()
    if not folder.is_dir():
        return []
    photos = [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    return sorted(photos, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]


def take_photo(camera_index: int | None = None, countdown: int = 3) -> Path:
    """Count down in the terminal, then snap. No preview window: the arm's venv has
    opencv-python-headless (from lerobot), which can't open windows, so Vin's
    SPACE-to-snap preview would crash here."""
    import cv2

    v = vin()
    index = camera_index if camera_index is not None else int(os.environ.get("CAMERA_INDEX", 0))
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            raise PhotoError(f"Could not open camera {index}. Plugged in? Try CAMERA_INDEX=1 or 2.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, v.CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, v.CAPTURE_HEIGHT)
        frame = None
        for n in range(countdown, 0, -1):
            print(f"  {n}...", flush=True)
            end = time.time() + 1.0
            while time.time() < end:  # keep reading so exposure settles and the frame is fresh
                ok, frame = cap.read()
        ok, frame = cap.read()
        if not ok or frame is None:
            raise PhotoError("The camera opened but sent no picture.")
        print("  Snap!")
        return v.save_frame(frame)
    except cv2.error as e:
        raise PhotoError(f"Camera error: {e}") from e
    finally:
        cap.release()


def show(path: Path) -> None:
    """Open an image in the desktop's viewer without blocking."""
    import subprocess
    subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def methods() -> dict[str, tuple[bool, str]]:
    """method -> (usable, reason if not)."""
    from image_draw import ai_available
    nebius = bool(os.environ.get("NEBIUS_API_KEY"))
    ai_ok, ai_reason = ai_available()
    return {
        "vin": (nebius, "" if nebius else "NEBIUS_API_KEY is not set (.env)"),
        "ai": (ai_ok, ai_reason),
        "edges": (True, ""),
    }


def photo_to_strokes(photo: Path, method: str, detail: str = "medium") -> tuple[list, Path]:
    """Photo -> strokes in 0..1 and a preview PNG path."""
    if method == "vin":
        print("Vin is sketching the portrait (can take a minute)...")
        try:
            svg_path = Path(vin().cartoonize(photo))
        except vin().CartoonError as e:
            raise PhotoError(str(e)) from e
        paths, w, h = svg_to_paths(svg_path.read_bytes())
        outline_png = render_paths(paths, w, h)
        # Vin's cartoon is already clean vector art: keep it, don't simplify it away.
        strokes = simplify(paths, w, h, "vector")
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        return strokes, preview(photo, outline_png, strokes, "vin")
    if method in ("ai", "edges"):
        return image_to_strokes(photo, method, detail)
    raise PhotoError(f"Unknown method {method!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--photo", help="use this photo (file in images/camera/ or a path) instead of the camera")
    p.add_argument("--method", choices=METHODS, default="vin")
    p.add_argument("--detail", choices=["simple", "medium", "detailed"], default="medium")
    p.add_argument("--camera-index", type=int)
    args = p.parse_args()
    try:
        if args.photo:
            photo = Path(args.photo)
            if not photo.exists():
                photo = camera_dir() / args.photo
        else:
            photo = take_photo(args.camera_index)
        strokes, prev = photo_to_strokes(photo, args.method, args.detail)
    except (PhotoError, ImageDrawError) as e:
        raise SystemExit(str(e))
    print(f"{len(strokes)} strokes. Preview: {prev}")


if __name__ == "__main__":
    main()
