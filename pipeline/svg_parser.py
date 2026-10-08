"""Parse SVG path data (M, L, C, Q, Z) into polylines. From the robo-replicator backend."""
import math
import os
import re
from dataclasses import dataclass
from xml.etree import ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"

# Points sampled along each Bezier curve.
BEZIER_SAMPLES = int(os.getenv("ROBOT_BEZIER_SAMPLES", "12"))

# Padding around path bounds as a fraction of content width/height (per axis).
SVG_PATH_PADDING = float(os.getenv("SVG_PATH_PADDING", "0.05"))


class SvgParseError(ValueError):
    """Raised when SVG path data cannot be parsed."""


@dataclass(frozen=True)
class ParsedSvg:
    width: float
    height: float
    paths: list[list[tuple[float, float]]]
    source_width: float | None = None
    source_height: float | None = None
    crop_x: float = 0.0
    crop_y: float = 0.0


def parse_svg_paths(svg_bytes: bytes) -> ParsedSvg:
    try:
        root = ET.fromstring(svg_bytes)
    except ET.ParseError as exc:
        raise SvgParseError("Invalid SVG XML.") from exc

    width, height = _viewbox_size(root)
    path_elements = root.findall(f".//{{{SVG_NS}}}path")
    if not path_elements:
        path_elements = root.findall(".//path")

    paths: list[list[tuple[float, float]]] = []
    for element in path_elements:
        path_data = element.get("d")
        if not path_data:
            continue
        points = _path_data_to_polyline(path_data.strip())
        if len(points) >= 2:
            paths.append(points)

    if not paths:
        raise SvgParseError("SVG contains no drawable path elements.")

    source_width, source_height = width, height
    crop_x = 0.0
    crop_y = 0.0
    if _needs_content_crop(paths, width, height, SVG_PATH_PADDING):
        paths, width, height, crop_x, crop_y = crop_paths_to_bounds(paths, SVG_PATH_PADDING)
    return ParsedSvg(
        width=width,
        height=height,
        paths=paths,
        source_width=source_width,
        source_height=source_height,
        crop_x=crop_x,
        crop_y=crop_y,
    )


def _needs_content_crop(
    paths: list[list[tuple[float, float]]],
    viewbox_width: float,
    viewbox_height: float,
    padding_fraction: float,
) -> bool:
    min_x, min_y, max_x, max_y = paths_bounds(paths)
    if viewbox_width <= 0 or viewbox_height <= 0:
        return True

    edge_tolerance = padding_fraction + 0.01
    left_gap = min_x / viewbox_width
    top_gap = min_y / viewbox_height
    right_gap = (viewbox_width - max_x) / viewbox_width
    bottom_gap = (viewbox_height - max_y) / viewbox_height
    if max(left_gap, top_gap, right_gap, bottom_gap) > edge_tolerance:
        return True

    used_width = (max_x - min_x) / viewbox_width
    used_height = (max_y - min_y) / viewbox_height
    return used_width < 0.98 or used_height < 0.98


def paths_bounds(paths: list[list[tuple[float, float]]]) -> tuple[float, float, float, float]:
    xs = [point[0] for path in paths for point in path]
    ys = [point[1] for path in paths for point in path]
    return min(xs), min(ys), max(xs), max(ys)


def content_bounds_with_padding(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    padding_fraction: float,
) -> tuple[float, float, float, float]:
    content_width = max(max_x - min_x, 1.0)
    content_height = max(max_y - min_y, 1.0)
    pad_x = content_width * padding_fraction
    pad_y = content_height * padding_fraction
    crop_x = min_x - pad_x
    crop_y = min_y - pad_y
    crop_width = content_width + 2.0 * pad_x
    crop_height = content_height + 2.0 * pad_y
    return crop_x, crop_y, crop_width, crop_height


def crop_paths_to_bounds(
    paths: list[list[tuple[float, float]]],
    padding_fraction: float,
) -> tuple[list[list[tuple[float, float]]], float, float, float, float]:
    min_x, min_y, max_x, max_y = paths_bounds(paths)
    crop_x, crop_y, crop_width, crop_height = content_bounds_with_padding(
        min_x, min_y, max_x, max_y, padding_fraction
    )
    cropped_paths = [
        [(x - crop_x, y - crop_y) for x, y in path] for path in paths
    ]
    return cropped_paths, crop_width, crop_height, crop_x, crop_y


def _viewbox_size(root: ET.Element) -> tuple[float, float]:
    viewbox = root.get("viewBox")
    if viewbox:
        parts = [float(value) for value in viewbox.replace(",", " ").split()]
        if len(parts) == 4:
            return parts[2], parts[3]

    width = float(root.get("width", "0"))
    height = float(root.get("height", "0"))
    if width <= 0 or height <= 0:
        raise SvgParseError("SVG is missing a usable viewBox or width/height.")
    return width, height


def _path_data_to_polyline(path_data: str) -> list[tuple[float, float]]:
    tokens = _tokenize_path_data(path_data)
    points: list[tuple[float, float]] = []
    current = (0.0, 0.0)
    subpath_start = (0.0, 0.0)
    index = 0
    active_command = "M"

    while index < len(tokens):
        token = tokens[index]
        if token in "MLCQZmlcqz":
            active_command = token.upper()
            index += 1
            if active_command == "Z":
                if points and points[-1] != subpath_start:
                    points.append(subpath_start)
                current = subpath_start
            continue

        if active_command == "M":
            current, subpath_start, index = _read_point(tokens, index)
            points.append(current)
            active_command = "L"
        elif active_command == "L":
            current, _, index = _read_point(tokens, index)
            points.append(current)
        elif active_command == "C":
            if index + 5 >= len(tokens):
                raise SvgParseError("Incomplete cubic Bézier command in path data.")
            control_1 = (float(tokens[index]), float(tokens[index + 1]))
            control_2 = (float(tokens[index + 2]), float(tokens[index + 3]))
            end = (float(tokens[index + 4]), float(tokens[index + 5]))
            index += 6
            points.extend(_sample_cubic(current, control_1, control_2, end)[1:])
            current = end
        elif active_command == "Q":
            if index + 3 >= len(tokens):
                raise SvgParseError("Incomplete quadratic Bézier command in path data.")
            control = (float(tokens[index]), float(tokens[index + 1]))
            end = (float(tokens[index + 2]), float(tokens[index + 3]))
            index += 4
            points.extend(_sample_quadratic(current, control, end)[1:])
            current = end
        else:
            raise SvgParseError(f"Unsupported SVG path command: {active_command}")

    deduped: list[tuple[float, float]] = []
    for point in points:
        if not deduped or point != deduped[-1]:
            deduped.append(point)
    return deduped


def _tokenize_path_data(path_data: str) -> list[str]:
    return re.findall(r"[MLCQZmlcqz]|-?\d*\.?\d+(?:[eE][-+]?\d+)?", path_data)


def _read_point(
    tokens: list[str], index: int
) -> tuple[tuple[float, float], tuple[float, float], int]:
    if index + 1 >= len(tokens):
        raise SvgParseError("Incomplete coordinate pair in path data.")
    point = (float(tokens[index]), float(tokens[index + 1]))
    return point, point, index + 2


def _sample_cubic(
    start: tuple[float, float],
    control_1: tuple[float, float],
    control_2: tuple[float, float],
    end: tuple[float, float],
) -> list[tuple[float, float]]:
    samples = max(BEZIER_SAMPLES, 2)
    return [_cubic_point(start, control_1, control_2, end, step / (samples - 1)) for step in range(samples)]


def _sample_quadratic(
    start: tuple[float, float],
    control: tuple[float, float],
    end: tuple[float, float],
) -> list[tuple[float, float]]:
    samples = max(BEZIER_SAMPLES, 2)
    return [_quadratic_point(start, control, end, step / (samples - 1)) for step in range(samples)]


def _cubic_point(
    start: tuple[float, float],
    control_1: tuple[float, float],
    control_2: tuple[float, float],
    end: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    u = 1.0 - t
    x = (
        u**3 * start[0]
        + 3 * u**2 * t * control_1[0]
        + 3 * u * t**2 * control_2[0]
        + t**3 * end[0]
    )
    y = (
        u**3 * start[1]
        + 3 * u**2 * t * control_1[1]
        + 3 * u * t**2 * control_2[1]
        + t**3 * end[1]
    )
    return x, y


def _quadratic_point(
    start: tuple[float, float],
    control: tuple[float, float],
    end: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    u = 1.0 - t
    x = u**2 * start[0] + 2 * u * t * control[0] + t**2 * end[0]
    y = u**2 * start[1] + 2 * u * t * control[1] + t**2 * end[1]
    return x, y


def path_length(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for start, end in zip(points, points[1:]):
        total += math.hypot(end[0] - start[0], end[1] - start[1])
    return total
