"""Black line art (PNG) -> centerline SVG paths. From the robo-replicator backend."""
import logging
import os
from io import BytesIO
from xml.etree.ElementTree import Element, SubElement, tostring

import cv2
import numpy as np
from PIL import Image
from skan import Skeleton
from skimage.morphology import skeletonize

from pipeline.svg_parser import SVG_PATH_PADDING, content_bounds_with_padding

logger = logging.getLogger(__name__)

# Strokes are skeletonized to a 1-pixel medial axis, then split into branch paths with
# skan (graph traversal). OpenCV findContours on a skeleton retraces loops and junction
# pixels and tends to produce doubled or overlapping segments.
SIMPLIFY_EPSILON = float(os.getenv("VECTORIZE_SIMPLIFY_EPSILON", "1.5"))
MIN_PATH_LENGTH = float(os.getenv("VECTORIZE_MIN_PATH_LENGTH", "8"))


class VectorizationError(RuntimeError):
    """Raised when outline bitmap vectorization fails."""


def bitmap_to_svg(outline_png: bytes, source_filename: str) -> bytes:
    gray = _load_grayscale(outline_png)
    height, width = gray.shape

    binary = _binarize_outline(gray)
    skeleton = _skeletonize(binary)
    contours = _extract_skeleton_paths(skeleton)

    if not contours:
        raise VectorizationError(
            f"No drawable outlines found in the bitmap for source file {source_filename!r}."
        )

    contours, crop_width, crop_height = _crop_contours_to_bounds(contours, SVG_PATH_PADDING)
    svg_bytes = _contours_to_svg(contours, crop_width, crop_height, source_filename)
    logger.info(
        "Vectorized outline for %r into %d centerline paths (cropped viewBox %dx%d from %dx%d)",
        source_filename,
        len(contours),
        int(round(crop_width)),
        int(round(crop_height)),
        width,
        height,
    )
    return svg_bytes


def _load_grayscale(outline_png: bytes) -> np.ndarray:
    with Image.open(BytesIO(outline_png)) as image:
        rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _binarize_outline(gray: np.ndarray) -> np.ndarray:
    """Return a binary image where line pixels are True (foreground)."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Close tiny gaps so skeletons stay connected; open removes isolated noise.
    close_kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    open_kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)
    return binary > 0


def _skeletonize(binary: np.ndarray) -> np.ndarray:
    """Reduce stroke regions to 1-pixel-wide centerlines."""
    return skeletonize(binary)


def _extract_skeleton_paths(skeleton: np.ndarray) -> list[np.ndarray]:
    skeleton_graph = Skeleton(skeleton)
    path_lengths = skeleton_graph.path_lengths()

    filtered: list[np.ndarray] = []
    for path_index, length in enumerate(path_lengths):
        if length < MIN_PATH_LENGTH:
            continue

        # skan returns (row, col); OpenCV contours use (x, y).
        coordinates = skeleton_graph.path_coordinates(path_index)
        points = np.array(
            [[int(column), int(row)] for row, column in coordinates],
            dtype=np.int32,
        ).reshape(-1, 1, 2)

        simplified = cv2.approxPolyDP(points, epsilon=SIMPLIFY_EPSILON, closed=False)
        if len(simplified) >= 2:
            filtered.append(simplified)

    filtered.sort(key=lambda contour: cv2.arcLength(contour, False), reverse=True)
    return filtered


def _contour_bounds(contours: list[np.ndarray]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for contour in contours:
        for point in contour:
            xs.append(float(point[0][0]))
            ys.append(float(point[0][1]))
    return min(xs), min(ys), max(xs), max(ys)


def _crop_contours_to_bounds(
    contours: list[np.ndarray],
    padding_fraction: float,
) -> tuple[list[np.ndarray], float, float]:
    min_x, min_y, max_x, max_y = _contour_bounds(contours)
    crop_x, crop_y, crop_width, crop_height = content_bounds_with_padding(
        min_x, min_y, max_x, max_y, padding_fraction
    )
    offset_x = int(round(crop_x))
    offset_y = int(round(crop_y))
    cropped: list[np.ndarray] = []
    for contour in contours:
        points = [
            [[int(round(point[0][0])) - offset_x, int(round(point[0][1])) - offset_y]]
            for point in contour
        ]
        cropped.append(np.array(points, dtype=np.int32))
    return cropped, crop_width, crop_height


def _contours_to_svg(
    contours: list[np.ndarray],
    width: float,
    height: float,
    source_filename: str,
) -> bytes:
    view_width = max(1, int(round(width)))
    view_height = max(1, int(round(height)))
    svg = Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "viewBox": f"0 0 {view_width} {view_height}",
            "width": str(view_width),
            "height": str(view_height),
        },
    )

    title = SubElement(svg, "title")
    title.text = f"Centerline paths derived from {source_filename}"

    group = SubElement(
        svg,
        "g",
        {
            "fill": "none",
            "stroke": "#000000",
            "stroke-width": "1",
            "stroke-linecap": "round",
            "stroke-linejoin": "round",
        },
    )

    for contour in contours:
        path_data = _contour_to_path_data(contour)
        SubElement(group, "path", {"d": path_data})

    xml_body = tostring(svg, encoding="utf-8", xml_declaration=True)
    return xml_body


def _contour_to_path_data(contour: np.ndarray) -> str:
    points = [(int(point[0][0]), int(point[0][1])) for point in contour]
    start_x, start_y = points[0]
    commands = [f"M {start_x} {start_y}"]

    for x, y in points[1:]:
        commands.append(f"L {x} {y}")

    return " ".join(commands)
