"""Recover the drawing from a robot replay dataset made by the team web app (the
"...-draw" folders / zips it downloads), and save it as an SVG in images/.

Those downloads hold only the arm's joint angles, not the SVG they were made from. This
runs the SO-101's forward kinematics (from assets/so101_new_calib.urdf) on every frame,
keeps the frames where the tool is down on the paper, and turns each run of them into a
line. The backend squashes the drawing's height to fit its canvas model, so the result is
stretched back to the source SVG's size recorded in meta/draw_source.json.

Usage (from the repo root):
    python tools/dataset_to_svg.py ~/Downloads/webcam-capture-...-draw
    python tools/dataset_to_svg.py ~/Downloads/some-draw.zip --name my_portrait
"""
import argparse
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ARM_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ARM_REPO))
from config import ASSETS_DIR  # noqa: E402

URDF = ASSETS_DIR / "so101_new_calib.urdf"
TIP = "gripper_frame_link"


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


class ForwardKinematics:
    """Tool-tip position from joint angles, using the URDF's joint chain (numpy only)."""

    def __init__(self, urdf: Path, tip: str = TIP):
        root = ET.parse(urdf).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain, link = [], tip
        while link in by_child:
            chain.append(by_child[link])
            link = by_child[link].find("parent").get("link")
        self.chain = chain[::-1]

    def tip(self, q_rad: dict) -> np.ndarray:
        t = np.eye(4)
        for j in self.chain:
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()]
            a = np.eye(4)
            a[:3, :3], a[:3, 3] = _rpy(*rpy), xyz
            t = t @ a
            if j.get("type") == "revolute":
                r = np.eye(4)
                r[:3, :3] = _axis_angle([float(v) for v in j.find("axis").get("xyz").split()], q_rad.get(j.get("name"), 0.0))
                t = t @ r
        return t[:3, 3]


def recover(folder: Path) -> tuple[str, int]:
    """Dataset folder -> (svg text, number of lines)."""
    src = json.loads((folder / "meta" / "draw_source.json").read_text())
    names = [n.removesuffix(".pos") for n in json.loads((folder / "meta" / "info.json").read_text())
             ["features"]["action"]["names"]]
    actions = np.stack(pd.read_parquet(folder / "data" / "chunk-000" / "file-000.parquet")["action"].to_numpy())
    fk = ForwardKinematics(URDF)
    tips = np.array([fk.tip({n: np.radians(v) for n, v in zip(names, row)}) for row in actions]) * 1000  # mm

    plane, lift = src["planar_canvas"]["plane_z_mm"], src["motion"]["travel_lift_mm"]
    down = np.abs(tips[:, 2] - plane) < lift * 0.3
    # Seen from the far side of the paper, so the page comes out the right way up, not mirrored.
    page = np.stack([tips[:, 1], tips[:, 0]], 1)
    strokes, cur = [], []
    for i in range(len(tips)):
        if down[i]:
            cur.append(page[i])
        elif cur:
            if len(cur) > 1:
                strokes.append(np.array(cur))
            cur = []
    if len(cur) > 1:
        strokes.append(np.array(cur))
    if not strokes:
        raise SystemExit("No pen-down movement found in this dataset.")

    pts = np.concatenate(strokes)
    lo, span = pts.min(0), np.ptp(pts, 0)
    w, h = src.get("source_svg_width") or 1000.0, src.get("source_svg_height") or 1000.0
    paths = []
    for s in strokes:
        xy = (s - lo) / span * [w, h]  # undo the backend's squashed canvas depth
        paths.append('  <path fill="none" stroke="black" stroke-width="3" d="M '
                     + " L ".join(f"{x:.1f} {y:.1f}" for x, y in xy) + '"/>')
    title = f"Recovered from robot dataset ({src.get('source_filename', folder.name)})"
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}">\n'
           f"  <title>{title}</title>\n" + "\n".join(paths) + "\n</svg>\n")
    return svg, len(paths)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path, help="the downloaded ...-draw folder or .zip")
    p.add_argument("--name", help="file name in images/ (default: from the dataset)")
    args = p.parse_args()
    src = args.dataset.expanduser()
    with tempfile.TemporaryDirectory() as tmp:
        if src.suffix == ".zip":
            zipfile.ZipFile(src).extractall(tmp)
            src = Path(tmp)
        svg, n = recover(src)
    name = args.name or args.dataset.name.removesuffix(".zip").removesuffix("-draw")
    out = ARM_REPO / "images" / f"{name}.svg"
    out.write_text(svg)
    print(f"{n} lines -> {out}")


if __name__ == "__main__":
    main()
