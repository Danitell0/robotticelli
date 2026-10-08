"""Turn an image from the images/ folder into pen strokes the arm can draw.

Steps 1-2 come from the team web app's backend (see pipeline/):
  1. outline    - get black line art from the image, one of:
                    ai     OpenAI gpt-image-1 redraws it as a cartoon outline (needs
                           OPENAI_API_KEY in .env, costs per image)
                    lines  the image already is a line drawing, use it as is
                    edges  trace the edges of a photo with OpenCV (free, rougher)
  2. vectorize  - skeletonize + skan centerlines -> SVG paths
  3. simplify   - ours: join touching paths, drop scraps, smooth, keep the longest,
                  and order them so the pen travels as little as possible

Outlines are cached in images/processed/ so the AI is only paid for once per image,
and a side-by-side preview PNG is written there too.

Command line, from the repo root:
    python image_draw.py cat.jpg --method edges --detail medium
"""
import argparse
import math
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import IMAGES_DIR

PROCESSED_DIR = IMAGES_DIR / "processed"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg"}

METHODS = {
    "ai": "AI outline (OpenAI, best for photos, costs per image)",
    "lines": "Already a line drawing (use it as is)",
    "edges": "Trace the edges (free, rougher on photos)",
}

# detail -> (join gap, smoothing, shortest line kept, max strokes), sizes as a fraction of the image
DETAIL = {
    "simple": (0.015, 0.006, 0.06, 15),
    "medium": (0.012, 0.004, 0.035, 30),
    "detailed": (0.010, 0.003, 0.02, 60),
    "vector": (0.0, 0.0005, 0.0, 500),  # already-clean vector art (Claude, Vin, SVG files): keep every line
}


class ImageDrawError(RuntimeError):
    pass


# ---------- outline and vectorize ----------

def ai_available() -> tuple[bool, str]:
    """(usable, reason) for the AI outline step."""
    try:
        from pipeline.outline import get_openai_api_key
        get_openai_api_key()
        return True, ""
    except Exception as e:
        return False, str(e).split(". ")[0]


def list_images() -> list[Path]:
    IMAGES_DIR.mkdir(exist_ok=True)
    return sorted(p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ImageDrawError("Could not encode image")
    return buf.tobytes()


def _load_gray(path: Path, longest: int) -> np.ndarray:
    with Image.open(path) as im:
        rgb = np.array(im.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    scale = longest / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray


def make_outline(path: Path, method: str) -> bytes:
    """Black line art on white, as PNG bytes. Cached per image and method."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cache = PROCESSED_DIR / f"{path.stem}-{method}-outline.png"
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return cache.read_bytes()

    if method == "ai":
        from pipeline.outline import OutlineGenerationError, generate_outline
        print("Asking OpenAI for an outline (this can take 30-60 s)...")
        try:
            png = generate_outline(path.read_bytes(), path.name)
        except (OutlineGenerationError, RuntimeError) as e:
            raise ImageDrawError(str(e)) from e
    elif method == "lines":
        png = _png_bytes(_load_gray(path, 1024))
    elif method == "edges":
        gray = _load_gray(path, 600)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
        png = _png_bytes(255 - edges)  # black lines on white, like Lucas's outlines
    else:
        raise ImageDrawError(f"Unknown method {method!r}")
    cache.write_bytes(png)
    return png


def vectorize(outline_png: bytes, name: str) -> tuple[list, float, float]:
    """Lucas's centerline vectorizer. Returns (paths in pixels, width, height)."""
    from pipeline.svg_parser import parse_svg_paths
    from pipeline.vectorize import VectorizationError, bitmap_to_svg
    try:
        svg = bitmap_to_svg(outline_png, name)
    except VectorizationError as e:
        raise ImageDrawError(str(e)) from e
    parsed = parse_svg_paths(svg)
    return parsed.paths, parsed.width, parsed.height


# ---------- simplify for the arm ----------

def _length(p: list) -> float:
    return sum(math.dist(a, b) for a, b in zip(p, p[1:]))


def _join(paths: list, gap: float) -> list:
    """Merge paths whose ends are within `gap` of each other (skan splits lines at every junction)."""
    paths = [list(p) for p in paths]
    merged = True
    while merged:
        merged = False
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                a, b = paths[i], paths[j]
                for a_end, b_end in ((-1, 0), (-1, -1), (0, 0), (0, -1)):
                    if math.dist(a[a_end], b[b_end]) <= gap:
                        a = a if a_end == -1 else a[::-1]
                        b = b if b_end == 0 else b[::-1]
                        paths[i] = a + b[1:]
                        del paths[j]
                        merged = True
                        break
                if merged:
                    break
            if merged:
                break
    return paths


def _order(paths: list) -> list:
    """Greedy nearest-neighbour ordering from the top-left, flipping paths when shorter."""
    left, out, pos = list(paths), [], (0.0, 0.0)
    while left:
        best = min(range(len(left)), key=lambda k: min(math.dist(pos, left[k][0]), math.dist(pos, left[k][-1])))
        p = left.pop(best)
        if math.dist(pos, p[-1]) < math.dist(pos, p[0]):
            p = p[::-1]
        out.append(p)
        pos = p[-1]
    return out


def simplify(paths: list, width: float, height: float, detail: str) -> list:
    """Pixel paths -> a small set of strokes in 0..1 paper coordinates, centred, aspect kept."""
    gap, eps, min_len, max_strokes = DETAIL[detail]
    size = max(width, height)
    paths = _join(paths, gap * size)
    smoothed = []
    for p in paths:
        # A loop (ends where it starts) must be smoothed as closed, or approxPolyDP drops part of it.
        closed = len(p) > 3 and math.dist(p[0], p[-1]) < 1e-6
        pts = np.array(p[:-1] if closed else p, np.float32).reshape(-1, 1, 2)
        p = [tuple(map(float, q[0])) for q in cv2.approxPolyDP(pts, eps * size, closed)]
        if closed:
            p.append(p[0])
        if len(p) >= 2 and _length(p) >= min_len * size:
            smoothed.append(p)
    smoothed.sort(key=_length, reverse=True)
    kept = smoothed[:max_strokes]
    if not kept:
        raise ImageDrawError("Nothing left to draw after simplifying. Try more detail or another method.")
    xs = [x for p in kept for x, _ in p]
    ys = [y for p in kept for _, y in p]
    x0, y0 = min(xs), min(ys)
    span = max(max(xs) - x0, max(ys) - y0, 1e-9)
    ox, oy = (1 - (max(xs) - x0) / span) / 2, (1 - (max(ys) - y0) / span) / 2
    norm = [[(ox + (x - x0) / span, oy + (y - y0) / span) for x, y in p] for p in kept]
    return _order(norm)


def split_subpaths(svg: bytes) -> bytes:
    """One <path> per subpath. The SVG parser joins every M inside a path into one line,
    which would draw lines between e.g. separate whiskers or eyes."""
    def split(match: re.Match) -> str:
        parts = [p.strip() for p in re.split(r"(?=M)", match.group(1)) if p.strip()]
        return "".join(f'<path fill="none" stroke="black" d="{p}"/>' for p in parts)
    text = re.sub(r'<path\b[^>]*?\sd="([^"]*)"[^>]*/?>(?:</path>)?', split, svg.decode())
    return text.encode()


def svg_to_paths(svg: bytes) -> tuple[list, float, float]:
    """Clean vector art (from Claude or Vin) -> (paths, width, height) via pipeline/svg_parser.py."""
    from pipeline.svg_parser import parse_svg_paths
    try:
        parsed = parse_svg_paths(split_subpaths(svg))
    except ValueError as e:
        raise ImageDrawError(f"Couldn't read the SVG: {e}") from e
    return parsed.paths, parsed.width, parsed.height


def render_paths(paths: list, w: float, h: float) -> bytes:
    """Draw vector paths into a PNG, to use as the 'outline' panel of the preview."""
    side = 512
    img = np.full((side, side), 255, np.uint8)
    s = side / max(w, h)
    for p in paths:
        cv2.polylines(img, [np.array([(int(x * s), int(y * s)) for x, y in p], np.int32)], False, 0, 2, cv2.LINE_AA)
    return cv2.imencode(".png", img)[1].tobytes()


def preview(path: Path, outline_png: bytes, strokes: list, tag: str) -> Path:
    """Original | outline | what the arm will draw, side by side."""
    side = 400
    tiles = []
    outline = cv2.imdecode(np.frombuffer(outline_png, np.uint8), cv2.IMREAD_GRAYSCALE)
    original = outline if path.suffix.lower() == ".svg" else _load_gray(path, side)  # PIL can't open SVG
    for img in (original, outline):
        tile = np.full((side, side), 255, np.uint8)
        scale = side / max(img.shape)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        y, x = (side - img.shape[0]) // 2, (side - img.shape[1]) // 2
        tile[y:y + img.shape[0], x:x + img.shape[1]] = img
        tiles.append(tile)
    drawn = np.full((side, side), 255, np.uint8)
    for s in strokes:
        cv2.polylines(drawn, [np.array([(int(x * (side - 1)), int(y * (side - 1))) for x, y in s], np.int32)],
                      False, 0, 1, cv2.LINE_AA)
    tiles.append(drawn)
    sheet = np.hstack([np.pad(t, 4, constant_values=200) for t in tiles])
    out = PROCESSED_DIR / f"{path.stem}-{tag}-preview.png"
    cv2.imwrite(str(out), sheet)
    return out


def svg_file_to_strokes(path: Path) -> tuple[list, Path]:
    """An SVG is already vector line art (e.g. a Vin cartoon): read its paths and keep them as drawn."""
    paths, w, h = svg_to_paths(path.read_bytes())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    strokes = simplify(paths, w, h, "vector")
    return strokes, preview(path, render_paths(paths, w, h), strokes, "svg")


def image_to_strokes(path: Path, method: str, detail: str) -> tuple[list, Path]:
    """The whole pipeline. Returns (strokes in 0..1, preview png path). SVGs skip straight to their paths."""
    if path.suffix.lower() == ".svg":
        return svg_file_to_strokes(path)
    outline_png = make_outline(path, method)
    paths, w, h = vectorize(outline_png, path.name)
    strokes = simplify(paths, w, h, detail)
    return strokes, preview(path, outline_png, strokes, f"{method}-{detail}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", help="file name in images/ (or a path)")
    p.add_argument("--method", choices=METHODS, default="edges")
    p.add_argument("--detail", choices=DETAIL, default="medium")
    args = p.parse_args()
    path = Path(args.image)
    if not path.exists():
        path = IMAGES_DIR / args.image
    strokes, prev = image_to_strokes(path, args.method, args.detail)
    print(f"{len(strokes)} strokes. Preview: {prev}")


if __name__ == "__main__":
    main()
