"""Make the SO-101 follower arm draw shapes and words with a pen, no leader arm needed.

There's no kinematic model installed, so instead you teach the paper by hand:
a grid of points with the pen touching (3x3 by default: corners, edge
middles and centre), plus one pose hovering above the centre. Every point on
the paper is then blended from the nearest taught points. More grid points
means the pen follows the paper more closely, especially on a big sheet.

1. Clamp a pen in the gripper, then teach the paper (torque is off):
    python draw.py --teach            # 3x3 grid, 9 points
    python draw.py --teach --grid 4   # 4x4 grid, 16 points
   Saved to data/draw_area.json.

2. Draw:
    python draw.py --list
    python draw.py --text HELLO
    python draw.py --shape star
    python draw.py --shape circle --size 0.5 --speed 0.1
    python draw.py --strokes my_drawing.json
    python draw.py --shape heart --dry-run   # check, don't move
    python draw.py --shape heart --air       # trace in the air
   If the pen presses too hard, raise --pen-height (0 = exactly the taught
   touch poses, 1 = the hover height). If lines fade out, lower it; it can go
   negative to press harder.

   A strokes file is a list of strokes, each a list of [x, y] points from
   0 to 1, with (0, 0) the top-left corner you taught:
       [[[0.2, 0.2], [0.8, 0.2]], [[0.5, 0.2], [0.5, 0.8]]]

Ctrl+C lifts the pen and returns the arm to its starting pose.
"""
import argparse
import json
import math
import time
from pathlib import Path

from config import DATA_DIR
from dance import FPS, JOINTS, PORT, glide, make_robot

AREA_FILE = DATA_DIR / "draw_area.json"


# ---------- shapes, in paper coordinates (0..1, origin top-left) ----------

def polygon(points: int, step: int = 1, phase: float = -math.pi / 2) -> list:
    pts = [(0.5 + 0.5 * math.cos(phase + 2 * math.pi * i * step / points),
            0.5 + 0.5 * math.sin(phase + 2 * math.pi * i * step / points)) for i in range(points + 1)]
    return [pts]


def heart() -> list:
    pts = []
    for i in range(121):
        t = 2 * math.pi * i / 120
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append((0.5 + x / 34, 0.5 - y / 34))
    return [pts]


def spiral(turns: int = 4) -> list:
    n = 60 * turns
    return [[(0.5 + 0.5 * (i / n) * math.cos(2 * math.pi * turns * i / n),
              0.5 + 0.5 * (i / n) * math.sin(2 * math.pi * turns * i / n)) for i in range(n + 1)]]


def smiley() -> list:
    face = [(0.5 + 0.5 * math.cos(2 * math.pi * i / 72), 0.5 + 0.5 * math.sin(2 * math.pi * i / 72)) for i in range(73)]
    mouth = [(0.5 + 0.28 * math.cos(math.pi * (0.15 + 0.7 * i / 30)), 0.52 + 0.28 * math.sin(math.pi * (0.15 + 0.7 * i / 30)))
             for i in range(31)]
    eye_l = [(0.35, 0.33), (0.35, 0.42)]
    eye_r = [(0.65, 0.33), (0.65, 0.42)]
    return [face, eye_l, eye_r, mouth]


def circle(cx: float, cy: float, r: float, n: int = 36) -> list:
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n + 1)]


def swirl(cx: float, cy: float, r: float, turns: float, direction: int = 1, phase: float = 0.0) -> list:
    n = int(40 * turns)
    return [(cx + r * (i / n) * math.cos(phase + direction * 2 * math.pi * turns * i / n),
             cy + r * (i / n) * math.sin(phase + direction * 2 * math.pi * turns * i / n)) for i in range(n + 1)]


def starry_night() -> list:
    """Line-art take on Van Gogh's The Starry Night: cypress on the left, swirling
    sky, stars, crescent moon top right, hills and a village with a church."""
    strokes = []
    # Wind lines flowing across the sky.
    for y0, amp, x0, x1 in [(0.08, 0.02, 0.25, 0.75), (0.47, 0.025, 0.25, 0.95)]:
        strokes.append([(x0 + (x1 - x0) * i / 40, y0 + amp * math.sin(i / 40 * 3 * math.pi)) for i in range(41)])
    # The big double swirl in the middle of the sky.
    strokes.append(swirl(0.40, 0.28, 0.11, 2.0))
    strokes.append(swirl(0.58, 0.31, 0.08, 1.75, direction=-1, phase=math.pi))
    # Stars: a small swirl with a halo.
    for cx, cy, r in [(0.30, 0.12, 0.03), (0.62, 0.13, 0.035), (0.75, 0.27, 0.03),
                      (0.27, 0.40, 0.025), (0.70, 0.43, 0.025), (0.92, 0.38, 0.03)]:
        strokes.append(swirl(cx, cy, r, 1.5))
        strokes.append(circle(cx, cy, r * 1.6, 24))
    # Crescent moon with a halo.
    outer = [(0.86 + 0.06 * math.cos(a), 0.14 + 0.06 * math.sin(a))
             for a in [math.radians(60 + 240 * i / 30) for i in range(31)]]
    inner = [(0.89 + 0.05 * math.cos(a), 0.13 + 0.05 * math.sin(a))
             for a in [math.radians(285 - 210 * i / 30) for i in range(31)]]
    strokes.append(outer + inner + [outer[0]])
    strokes.append(circle(0.86, 0.14, 0.095, 36))
    # Rolling hills.
    strokes.append([(0.22 + 0.78 * i / 40, 0.62 - 0.04 * math.sin(i / 40 * 2 * math.pi) - 0.03 * i / 40)
                    for i in range(41)])
    # Village: houses (base-left, walls, roof peak) and the church with its spire.
    for x, w, h in [(0.32, 0.07, 0.06), (0.42, 0.06, 0.05), (0.70, 0.07, 0.06), (0.80, 0.06, 0.05), (0.89, 0.07, 0.06)]:
        base = 0.86
        strokes.append([(x, base), (x, base - h), (x + w / 2, base - h - 0.035), (x + w, base - h), (x + w, base), (x, base)])
    strokes.append([(0.52, 0.86), (0.52, 0.74), (0.555, 0.74), (0.555, 0.62), (0.5675, 0.52), (0.58, 0.62),
                    (0.58, 0.74), (0.62, 0.74), (0.62, 0.86), (0.52, 0.86)])
    strokes.append([(0.22, 0.92), (1.0, 0.92)])
    # Cypress tree: a tall flame, with two inner flame lines.
    left = [(0.13 - 0.06 * (1 - t) ** 0.8 + 0.012 * math.sin(t * 9 * math.pi), 0.95 - 0.85 * t) for t in [i / 40 for i in range(41)]]
    right = [(0.15 + 0.06 * (1 - t) ** 0.8 + 0.012 * math.sin(t * 9 * math.pi + 1), 0.95 - 0.85 * t) for t in [i / 40 for i in range(41)]]
    strokes.append(left + right[::-1])
    for dx in (-0.02, 0.02):
        strokes.append([(0.14 + dx * (1 - t) + 0.008 * math.sin(t * 7 * math.pi), 0.9 - 0.7 * t) for t in [i / 30 for i in range(31)]])
    return strokes


SHAPES = {
    "square": lambda: [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]],
    "triangle": lambda: polygon(3),
    "circle": lambda: polygon(72),
    "star": lambda: polygon(5, step=2),
    "heart": heart,
    "spiral": spiral,
    "smiley": smiley,
    "starry_night": starry_night,
}


# ---------- text: a single-stroke block font ----------
# Each letter is 0.6 wide and 1 tall, y pointing down.

O = [(0, 0), (0.6, 0), (0.6, 1), (0, 1), (0, 0)]
P = [(0, 1), (0, 0), (0.6, 0), (0.6, 0.5), (0, 0.5)]
FONT = {
    "A": [[(0, 1), (0.3, 0), (0.6, 1)], [(0.12, 0.6), (0.48, 0.6)]],
    "B": [[(0, 1), (0, 0), (0.45, 0), (0.6, 0.12), (0.6, 0.38), (0.45, 0.5), (0, 0.5)],
          [(0.45, 0.5), (0.6, 0.62), (0.6, 0.88), (0.45, 1), (0, 1)]],
    "C": [[(0.6, 0), (0, 0), (0, 1), (0.6, 1)]],
    "D": [[(0, 0), (0, 1), (0.4, 1), (0.6, 0.75), (0.6, 0.25), (0.4, 0), (0, 0)]],
    "E": [[(0.6, 0), (0, 0), (0, 1), (0.6, 1)], [(0, 0.5), (0.45, 0.5)]],
    "F": [[(0.6, 0), (0, 0), (0, 1)], [(0, 0.5), (0.45, 0.5)]],
    "G": [[(0.6, 0), (0, 0), (0, 1), (0.6, 1), (0.6, 0.5), (0.3, 0.5)]],
    "H": [[(0, 0), (0, 1)], [(0.6, 0), (0.6, 1)], [(0, 0.5), (0.6, 0.5)]],
    "I": [[(0.3, 0), (0.3, 1)], [(0.1, 0), (0.5, 0)], [(0.1, 1), (0.5, 1)]],
    "J": [[(0.6, 0), (0.6, 1), (0, 1), (0, 0.7)]],
    "K": [[(0, 0), (0, 1)], [(0.6, 0), (0, 0.5), (0.6, 1)]],
    "L": [[(0, 0), (0, 1), (0.6, 1)]],
    "M": [[(0, 1), (0, 0), (0.3, 0.5), (0.6, 0), (0.6, 1)]],
    "N": [[(0, 1), (0, 0), (0.6, 1), (0.6, 0)]],
    "O": [O],
    "P": [P],
    "Q": [O, [(0.35, 0.7), (0.6, 1)]],
    "R": [P, [(0.2, 0.5), (0.6, 1)]],
    "S": [[(0.6, 0), (0, 0), (0, 0.5), (0.6, 0.5), (0.6, 1), (0, 1)]],
    "T": [[(0, 0), (0.6, 0)], [(0.3, 0), (0.3, 1)]],
    "U": [[(0, 0), (0, 1), (0.6, 1), (0.6, 0)]],
    "V": [[(0, 0), (0.3, 1), (0.6, 0)]],
    "W": [[(0, 0), (0.15, 1), (0.3, 0.5), (0.45, 1), (0.6, 0)]],
    "X": [[(0, 0), (0.6, 1)], [(0.6, 0), (0, 1)]],
    "Y": [[(0, 0), (0.3, 0.5), (0.6, 0)], [(0.3, 0.5), (0.3, 1)]],
    "Z": [[(0, 0), (0.6, 0), (0, 1), (0.6, 1)]],
    "!": [[(0.3, 0), (0.3, 0.7)], [(0.3, 0.92), (0.3, 1)]],
    " ": [],
}
LETTER_GAP = 0.3


def text_strokes(text: str) -> list:
    """Lay the text out on one line, scaled to the paper width and centred."""
    text = text.upper()
    unknown = sorted(set(text) - set(FONT))
    if unknown:
        raise SystemExit(f"Can't write {''.join(unknown)!r}. Letters available: {''.join(sorted(FONT)).strip()}")
    strokes, x = [], 0.0
    for ch in text:
        strokes += [[(x + px, py) for px, py in s] for s in FONT[ch]]
        x += 0.6 + LETTER_GAP
    width = x - LETTER_GAP
    s = 1 / max(width, 1)
    return [[(0.5 + (px - width / 2) * s, 0.5 + (py - 0.5) * s) for px, py in st] for st in strokes]


# ---------- signature ----------

SIGN_BOX = (0.75, 0.905)  # top-left of the signature, in paper coordinates
SIGN_GAP = 0.015  # keep the drawing at least this far above / left of the signature


def clear_signature_corner(strokes: list) -> tuple[list, float]:
    """If the drawing reaches into the bottom-right corner where the signature goes, shrink it
    evenly around the paper centre until it clears. Returns (strokes, scale used)."""
    sx, sy = SIGN_BOX
    corner = [(x, y) for st in strokes for x, y in st if x > sx - SIGN_GAP and y > sy - SIGN_GAP]
    if not corner:
        return strokes, 1.0
    # Each corner point clears by moving up OR left, whichever needs less shrinking (the larger
    # allowed scale); the drawing must use a scale that works for every one of them.
    k = min(1.0, min(max((sy - SIGN_GAP - 0.5) / (y - 0.5), (sx - SIGN_GAP - 0.5) / (x - 0.5)) for x, y in corner))
    return [[(0.5 + (x - 0.5) * k, 0.5 + (y - 0.5) * k) for x, y in st] for st in strokes], k

def _smooth(points: list, per_segment: int = 10) -> list:
    """Catmull-Rom curve through the points, sampled as a polyline."""
    pts = [points[0]] + list(points) + [points[-1]]
    out = [points[0]]
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        for k in range(1, per_segment + 1):
            t = k / per_segment
            t2, t3 = t * t, t * t * t
            out.append(tuple(0.5 * (2 * p1[d] + (-p0[d] + p2[d]) * t + (2 * p0[d] - 5 * p1[d] + 4 * p2[d] - p3[d]) * t2
                                    + (-p0[d] + 3 * p1[d] - 3 * p2[d] + p3[d]) * t3) for d in (0, 1)))
    return out


def signature(x0: float = 0.75, y0: float = 0.905, w: float = 0.21, h: float = 0.065) -> list:
    """A quick 'Robotticelli' scribble, like a painter's signature: a capital R that sweeps into
    a run of cursive humps and loops (tall ones for the t's and l's), slanted like handwriting,
    with an underline flourish. Not meant to be readable. Fits the box (x0, y0, w, h) in
    paper coordinates; u, v below run 0..1 across and down that box."""
    def place(u, v):
        u += 0.12 * (1.0 - v)  # forward slant
        return (x0 + u * w, y0 + v * h)

    # Capital R: stem up, round bowl, then the leg sweeps down and right into the name.
    r = _smooth([(0.03, 0.92), (0.04, 0.5), (0.06, 0.06), (0.13, 0.0), (0.18, 0.12), (0.15, 0.33),
                 (0.07, 0.42), (0.13, 0.6), (0.19, 0.86), (0.23, 0.72)], 8)
    # 'obotticelli': (height, width, loop) per letter. loop: how much the stroke curls back on itself.
    letters = [(0.30, 0.060, 0.012), (0.70, 0.050, 0.020), (0.28, 0.058, 0.012), (0.62, 0.040, 0.016),
               (0.66, 0.040, 0.016), (0.26, 0.045, 0.006), (0.30, 0.055, 0.010), (0.27, 0.052, 0.014),
               (0.78, 0.040, 0.020), (0.74, 0.040, 0.020), (0.24, 0.050, 0.006)]
    run, u = [], 0.23
    for height, width, loop in letters:
        for i in range(1, 11):
            t = (i / 10) * 2 * math.pi
            run.append((u + width * i / 10 - loop * math.sin(t), 0.9 - height * (0.5 - 0.5 * math.cos(t))))
        u += width
    body = [place(a, b) for a, b in r + run]
    # Flourish: from the last letter dip down, then one long swoosh back under the name, flicking up.
    flourish = _smooth([(u, 0.88), (u + 0.05, 1.02), (0.80, 1.14), (0.48, 1.12), (0.20, 1.16), (0.06, 1.08),
                        (0.02, 0.98)], 10)
    return [body, [place(a, b) for a, b in flourish]]


def fit(strokes: list, size: float) -> list:
    """Shrink strokes around the paper centre: size 1 fills the taught area."""
    return [[(0.5 + (x - 0.5) * size, 0.5 + (y - 0.5) * size) for x, y in s] for s in strokes]


def densify(stroke: list, step: float) -> list:
    """Add points so consecutive points are at most `step` apart."""
    out = [stroke[0]]
    for (x0, y0), (x1, y1) in zip(stroke, stroke[1:]):
        n = max(1, math.ceil(math.hypot(x1 - x0, y1 - y0) / step))
        out += [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n) for i in range(1, n + 1)]
    return out


# ---------- paper coordinates -> joint poses ----------

TRAVEL_LIFT = 4.0  # travel height, as a multiple of each point's taught lift


class Paper:
    def __init__(self, area: dict, pen_height: float = 0.0, bottom_lift: float = 0.0, centre_lift: float = 0.0,
                 lift_back: float = 0.0):
        if "grid" in area:
            self.grid = area["grid"]
        else:  # older 4-corner file
            c = area["corners"]
            self.grid = [[c["top_left"], c["top_right"]], [c["bottom_left"], c["bottom_right"]]]
        # Newer files hold a hover pose above every grid point, so lifting is straight up
        # everywhere. Older ones have a single hover above the centre, which reused across a
        # big sheet pulls the tool backwards near the edges.
        self.hover_grid = area.get("hover_grid")
        centre = self._blend(0.5, 0.5)
        self.lift = {j: area["hover"][j] - centre[j] for j in JOINTS}
        self.lift["gripper"] = 0.0  # keep the same grip on the pen when lifting
        self.pen_height = pen_height
        # Towards the bottom of the paper the arm sits closer to it and presses harder,
        # so lift the tool a bit more the further down the page it goes (y: 0 top, 1 bottom).
        self.bottom_lift = bottom_lift  # negative = press harder towards the bottom
        # Extra height that is full at the centre of the paper and fades to 0 at the corners.
        self.centre_lift = centre_lift
        # Extra backwards move on every lift, in paper-widths. "Back" is towards the bottom
        # of the paper, which is the edge nearest the arm's base.
        self.lift_back = lift_back

    def _blend(self, x: float, y: float, grid: list | None = None) -> dict:
        """Bilinear blend inside the grid cell that contains (x, y)."""
        grid = grid or self.grid
        n = len(grid) - 1
        col, row = min(int(x * n), n - 1), min(int(y * n), n - 1)
        u, v = x * n - col, y * n - row
        tl, tr = grid[row][col], grid[row][col + 1]
        bl, br = grid[row + 1][col], grid[row + 1][col + 1]
        return {j: (1 - u) * (1 - v) * tl[j] + u * (1 - v) * tr[j] + u * v * br[j] + (1 - u) * v * bl[j]
                for j in JOINTS}

    def lift_at(self, x: float, y: float) -> dict:
        """Joint change that lifts the tool off the paper at (x, y): the taught lift,
        plus lift_back paper-widths towards the base."""
        if self.hover_grid:
            p, up = self._blend(x, y), self._blend(x, y, self.hover_grid)
            lift = {j: 0.0 if j == "gripper" else up[j] - p[j] for j in JOINTS}
        else:
            lift = dict(self.lift)
        if self.lift_back:
            back = self.towards_base(x, y)
            lift = {j: lift[j] + self.lift_back * back[j] for j in JOINTS}
        return lift

    def towards_base(self, x: float, y: float) -> dict:
        """Joint change per paper-width of moving towards the bottom of the paper (the base)."""
        y0, y1 = max(y - 0.05, 0.0), min(y + 0.05, 1.0)
        a, b = self._blend(x, y0), self._blend(x, y1)
        return {j: 0.0 if j == "gripper" else (b[j] - a[j]) / (y1 - y0) for j in JOINTS}

    def height_at(self, x: float, y: float) -> float:
        """Tool height at (x, y): 0 = taught touch pose, 1 = hover. Negative presses harder."""
        centre = max(0.0, 1 - math.hypot(x - 0.5, y - 0.5) / math.hypot(0.5, 0.5))
        h = self.pen_height + self.bottom_lift * y + self.centre_lift * centre
        return min(max(h, -0.6), 0.95)  # never at or above hover height, never pushing through

    def touch(self, x: float, y: float) -> dict:
        p, lift = self._blend(x, y), self.lift_at(x, y)
        h = self.height_at(x, y)
        return {j: p[j] + h * lift[j] for j in JOINTS}

    def hover(self, x: float, y: float) -> dict:
        p, lift = self._blend(x, y), self.lift_at(x, y)
        return {j: p[j] + lift[j] for j in JOINTS}

    def travel(self, x: float, y: float) -> dict:
        """Well above (x, y), for moving across the paper. Taught lifts can be under a
        centimetre, and a joint-space move between two hover poses dips in the middle,
        so going across at hover height drags the tool over the paper."""
        p, lift = self._blend(x, y), self.lift_at(x, y)
        return clamp({j: p[j] + TRAVEL_LIFT * lift[j] for j in JOINTS})


def travel_to(robot, paper: Paper, current: dict, src, dst, seconds: float = 1.0) -> dict:
    """Lift straight up from src (if the tool is on the paper there), cross at travel
    height, then come straight down to hover over dst. src/dst are paper (x, y)."""
    if src is not None:
        current = glide(robot, current, paper.travel(*src), 0.6, False)
    current = glide(robot, current, paper.travel(*dst), seconds, False)
    return glide(robot, current, paper.hover(*dst), 0.6, False)


# ---------- robot ----------

def read_pose(robot) -> dict:
    obs = robot.get_observation()
    return {j: obs[f"{j}.pos"] for j in JOINTS}


def describe(row: int, col: int, n: int) -> str:
    """e.g. 'top left corner', 'left edge', 'centre' (3x3), or a fraction for bigger grids."""
    v = "top" if row == 0 else "bottom" if row == n else None
    h = "left" if col == 0 else "right" if col == n else None
    if v and h:
        return f"{v} {h} corner"
    if n == 2:
        return f"{v or h} edge middle" if (v or h) else "centre"
    across, down = f"{col}/{n} of the way across", f"{row}/{n} of the way down"
    if v:
        return f"{v} edge, {across}"
    if h:
        return f"{h} edge, {down}"
    return f"inside: {across}, {down}"


def teach(robot, size: int):
    n = size - 1
    robot.bus.disable_torque()
    print("Torque OFF: hold the arm and move it by hand.")
    print(f"Teaching a {size}x{size} grid, row by row from the top-left. At each point:")
    print("  1. tip touching the paper LIGHTLY, Enter")
    print("  2. lift it ~2 cm straight AWAY from the surface, at right angles to it")
    print("     (on a flat table that's straight up; on an angled canvas, tilt with the canvas), Enter\n")
    grid, hover_grid = [], []
    for row in range(size):
        grid.append([])
        hover_grid.append([])
        for col in range(size):
            where = describe(row, col, n)
            input(f"Tip ON the paper at the {where}, then Enter: ")
            grid[row].append(read_pose(robot))
            input(f"  Now lift ~2 cm straight away from the surface at the {where}, then Enter: ")
            hover_grid[row].append(read_pose(robot))
    hover = hover_grid[n // 2][n // 2]
    AREA_FILE.write_text(json.dumps({"grid": grid, "hover_grid": hover_grid, "hover": hover}, indent=2))
    print(f"\nSaved to {AREA_FILE}")
    input("Rest the arm in its parking pose, then Enter to finish.")


# Calibrated range of each joint (of our arm) less a 5 degree margin.
SAFE = {"shoulder_pan": 113, "shoulder_lift": 101, "elbow_flex": 93, "wrist_flex": 96, "wrist_roll": 175}
TILT_WARN = 90  # wrist_flex degrees beyond which the pen starts to tilt


def clamp(pose: dict) -> dict:
    return {j: max(-SAFE[j], min(SAFE[j], v)) if j in SAFE else v for j, v in pose.items()}


# Jump detection: a jump is when a joint suddenly lags its command by much more than usual.
LEFT_ZONE = 0.35  # paper x below this is near the wrist limit: always half speed there
JUMP_MIN_DEG = 3.0  # never call anything smaller than this a jump
JUMP_FACTOR = 3.0  # ...or less than this many times the recent average lag
CALM_SECONDS = 1.5  # smooth drawing needed before speeding back up


class Pace:
    """Pen speed that halves on every detected jump and recovers slowly when calm."""

    def __init__(self, base: float):
        self.base = base
        self.speed = base
        self.avg_lag = None
        self.calm = 0.0
        self.jumps = []
        self.xy = (0.5, 0.5)  # where on the paper the tool was last sent

    def update(self, lag: float, joint: str, x: float, y: float) -> bool:
        if self.avg_lag is None:
            self.avg_lag = lag
        jump = lag > max(JUMP_MIN_DEG, JUMP_FACTOR * self.avg_lag)
        if jump:
            self.speed = max(self.speed / 2, self.base / 10)
            self.calm = 0.0
            self.jumps.append((x, y, joint, lag))
            print(f"  JUMP at x={x:.2f} y={y:.2f}: {joint} lagged {lag:.1f} deg -> slowing to {self.speed:.3f}")
        else:
            self.avg_lag = 0.95 * self.avg_lag + 0.05 * lag
            self.calm += 1 / FPS
            if self.calm > CALM_SECONDS:
                self.speed = min(self.speed * 1.1, self.base)
                self.calm = 0.0
        return jump

    def limit(self, x: float) -> float:
        return min(self.speed, self.base / 2) if x < LEFT_ZONE else self.speed


def follow(robot, pen, stroke: list, pace: Pace):
    """Draw one stroke, watching the real joint positions and slowing down on jumps."""
    pts = densify(stroke, 0.002)
    dist = [0.0]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        dist.append(dist[-1] + math.hypot(x1 - x0, y1 - y0))
    s, k, last = 0.0, 0, None
    while True:
        tick = time.perf_counter()
        while k < len(pts) - 1 and dist[k + 1] < s:
            k += 1
        if k >= len(pts) - 1:
            x, y = pts[-1]
        else:
            t = (s - dist[k]) / max(dist[k + 1] - dist[k], 1e-9)
            x = pts[k][0] + (pts[k + 1][0] - pts[k][0]) * t
            y = pts[k][1] + (pts[k + 1][1] - pts[k][1]) * t
        cmd = pen(x, y)
        pace.xy = (x, y)
        robot.send_action({f"{j}.pos": v for j, v in cmd.items()})
        if last is not None:
            real = read_pose(robot)
            joint, lag = max(((j, abs(real[j] - last[j])) for j in JOINTS if j != "gripper"), key=lambda e: e[1])
            if pace.update(lag, joint, x, y):
                time.sleep(0.3)  # let the pen settle before carrying on
        last = cmd
        if s >= dist[-1]:
            break
        s = min(s + pace.limit(x) / FPS, dist[-1])
        time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - tick)))


def draw(robot, paper: Paper, strokes: list, speed: float, air: bool = False):
    pen = paper.hover if air else paper.touch
    home = read_pose(robot)
    pace = Pace(speed)
    try:
        current, prev = home, None
        for i, stroke in enumerate(strokes, 1):
            print(f"Stroke {i}/{len(strokes)}")
            # Up, across at travel height, straight down onto the start of the stroke.
            current = travel_to(robot, paper, current, prev, stroke[0], 3.0 if i == 1 else 1.0)
            current = glide(robot, current, pen(*stroke[0]), 0.5, False)
            follow(robot, pen, stroke, pace)
            current = pen(*stroke[-1])
            current = glide(robot, current, paper.hover(*stroke[-1]), 0.5, False)
            prev = stroke[-1]
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if pace.jumps:
            left = sum(1 for x, *_ in pace.jumps if x < LEFT_ZONE)
            print(f"{len(pace.jumps)} jumps detected ({left} on the left side)")
        else:
            print("No jumps detected")
        print("Lifting pen and returning to start pose")
        current = read_pose(robot)
        # Lift well clear first so the tool doesn't drag across the paper on the way home.
        lift = paper.lift_at(*pace.xy)
        lifted = clamp({j: current[j] + TRAVEL_LIFT * lift[j] for j in JOINTS})
        glide(robot, current, lifted, 0.8, False)
        glide(robot, lifted, home, 3.0, False)


class Plan:
    """A drawing checked against the taught paper, ready to draw."""

    def __init__(self, raw: list, size: float, speed: float, pen_height: float, bottom_lift: float = 0.0,
                 centre_lift: float = 0.0, lift_back: float = 0.0, vscale: float = 1.0, sign: bool = False):
        if not AREA_FILE.exists():
            raise ValueError(f"No {AREA_FILE.name} yet. Teach the paper first.")
        self.strokes = fit(raw, min(max(size, 0.05), 1.0))
        # Squash (vscale < 1) or stretch the drawing vertically around the paper centre. On a
        # tilted canvas drawings come out too tall; compressing them keeps circles round.
        self.strokes = [[(x, 0.5 + (y - 0.5) * vscale) for x, y in st] for st in self.strokes]
        self.shrunk = 1.0
        if sign:
            self.strokes, self.shrunk = clear_signature_corner(self.strokes)
            self.strokes += signature()
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for s in self.strokes for x, y in s):
            raise ValueError("Some points fall outside the paper (0..1). Use a smaller size.")
        self.paper = Paper(json.loads(AREA_FILE.read_text()), pen_height, bottom_lift, centre_lift, lift_back)
        self.speed = speed
        self.seconds = sum(len(densify(s, speed / FPS)) for s in self.strokes) / FPS
        # Near the end of its range the wrist can't keep the pen upright, so the pen tilts.
        # This happens far from the centre of the paper, mostly towards the top edge.
        self.worst_wrist = max(abs(self.paper.touch(x, y)["wrist_flex"])
                               for s in self.strokes for x, y in densify(s, 0.02))

    def summary(self) -> str:
        return f"{len(self.strokes)} strokes, about {self.seconds:.0f}s of drawing"

    def tilt_warning(self) -> str | None:
        if self.worst_wrist <= TILT_WARN:
            return None
        return (f"WARNING: wrist_flex reaches {self.worst_wrist:.0f} deg (limit ~{SAFE['wrist_flex']}), "
                "the pen will tilt. Use a smaller size, or move the paper closer to the arm and re-teach.")

    def run(self, robot, air: bool = False):
        draw(robot, self.paper, self.strokes, self.speed, air)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default=PORT)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--teach", action="store_true", help="teach the paper by hand")
    mode.add_argument("--shape", choices=SHAPES, help="draw a built-in shape")
    mode.add_argument("--text", help="write text in block capitals")
    mode.add_argument("--strokes", metavar="FILE", help="draw strokes from a JSON file")
    mode.add_argument("--list", action="store_true", help="list built-in shapes and letters")
    p.add_argument("--grid", type=int, default=3, help="with --teach: points per side to teach (2 = corners only)")
    p.add_argument("--size", type=float, default=0.8, help="fraction of the paper to use (0..1)")
    p.add_argument("--speed", type=float, default=0.15, help="pen speed in paper-widths per second")
    p.add_argument("--pen-height", type=float, default=0.15,
                   help="raise the pen this fraction of the way to hover height (0 = taught touch poses)")
    p.add_argument("--bottom-lift", type=float, default=0.0,
                   help="extra pen height towards the bottom of the paper (0 at the top edge); negative presses harder")
    p.add_argument("--centre-lift", type=float, default=0.0,
                   help="extra pen height at the centre of the paper, fading to 0 at the corners")
    p.add_argument("--lift-back", type=float, default=0.0,
                   help="paper-widths to also move back (towards the base) when lifting the pen")
    p.add_argument("--vscale", type=float, default=1.0,
                   help="vertical scale of the drawing, e.g. 0.8 to squash it on a tilted canvas")
    p.add_argument("--sign", action="store_true", help="sign the drawing in the bottom-right corner")
    p.add_argument("--air", action="store_true", help="trace the drawing at hover height, pen never touches")
    p.add_argument("--dry-run", action="store_true", help="check the drawing without moving the arm")
    args = p.parse_args()

    if args.list:
        print("Shapes:  " + ", ".join(SHAPES))
        print("Letters: " + "".join(sorted(FONT)).strip())
        return

    if not args.teach:
        if args.text:
            raw = text_strokes(args.text)
        else:
            raw = SHAPES[args.shape]() if args.shape else json.loads(Path(args.strokes).read_text())
        try:
            plan = Plan(raw, args.size, args.speed, args.pen_height, args.bottom_lift, args.centre_lift,
                        args.lift_back, args.vscale, args.sign)
        except ValueError as e:
            raise SystemExit(str(e))
        paper, strokes = plan.paper, plan.strokes
        print(plan.summary())
        if plan.tilt_warning():
            print(plan.tilt_warning())
        if args.dry_run:
            return

    robot = make_robot(args.port)
    robot.connect(calibrate=False)
    try:
        if args.teach:
            teach(robot, max(2, args.grid))
        else:
            draw(robot, paper, strokes, args.speed, args.air)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
