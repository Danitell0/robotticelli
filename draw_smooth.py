"""draw.py with smoother arm motion. Same options as draw.py; draw.py itself is untouched.

Why draw.py looks jerky
-----------------------
draw.follow() moves the pen along each stroke at one constant speed. So the speed
jumps from 0 to full at the start of a stroke and back to 0 at the end, the arm
turns sharp corners at full speed, and a detected "jump" halves the speed at once
and pauses for 0.3 s. On top of that, draw.Paper blends the taught grid points
bilinearly per cell, so joint velocities have a kink wherever the pen crosses from
one grid cell into the next. Each of these is a sudden change in joint speed,
which the servos turn into a visible jolt.

The idea: smoothstep
--------------------
smoothstep(t) = 10t^3 - 15t^4 + 6t^5 (the same curve as backend/scripts/line3.py)
goes from 0 to 1 with zero speed AND zero acceleration at both ends. Here it shapes
the pen SPEED along a stroke: every place where the speed has to drop (stroke start,
stroke end, a corner, a grid crossing) gets a smoothstep dip instead of a step, so
the arm never has to change speed abruptly. The path itself is unchanged: the pen
traces exactly the same points as draw.py, only the timing along them differs.

What changes while drawing a stroke:
- The pen speeds up and slows down with a smoothstep curve at the start and end of
  every stroke, instead of jumping straight to full speed.
- It slows down before sharp corners (square, letters, houses) and speeds up after.
- It eases off a little where the pen crosses from one taught grid cell to the next:
  the joint blend has a kink there (draw.Paper is bilinear per cell).
- When a jump is detected the speed still drops, but it eases down and back up
  instead of halving instantly and pausing for 0.3 s.
- Pen-up travel and pen-down/up use the same smoothstep ease (zero speed AND zero
  acceleration at both ends) instead of a cosine ease.

How it plugs in
---------------
draw.draw() calls follow() and glide() through the draw module, so at the bottom of
this file those two names are swapped for the smooth versions and then draw.main()
runs as usual. The swap only happens in a process started with this script; running
draw.py directly, or importing draw elsewhere, still gets the original behaviour.

Tuning
------
The constants below are all in paper-widths (the taught area is 1 wide). Bigger
RAMP / CORNER_RADIUS = gentler but slower; smaller = snappier. In a simulated run
(fake arm, 0.8 size, speed 0.15) peak joint acceleration dropped by roughly 50-65%
on square, star and starry_night, at the cost of 14-66% more drawing time. If it is
too slow, lower RAMP first; raising --speed also works, but acceleration grows with
the square of the speed, so go up in small steps.

Try it in this order:
    python draw_smooth.py --shape starry_night --dry-run
    python draw_smooth.py --shape starry_night --air
    python draw_smooth.py --shape starry_night
The original is still there to compare: draw.py takes the same options.
"""
import math
import time

import draw
from dance import FPS, JOINTS

# ---------- tuning ----------

RAMP = 0.04  # paper-widths to speed up at the start of a stroke and slow down at the end
CORNER_RADIUS = 0.03  # paper-widths before/after a sharp corner where the pen slows down
CORNER_MIN_DEG = 20  # turns gentler than this (curves) keep full speed
GRID_FACTOR = 0.6  # speed fraction where the pen crosses a taught grid line
MIN_FACTOR = 0.12  # never crawl slower than this fraction of the pace speed
SPEED_SMOOTHING = 0.15  # 0..1 per frame: how quickly the actual speed follows the target
CURVE_RADIUS = 0.06  # paper-widths: curves tighter than this radius slow down in proportion


# ---------- easing ----------

def smoothstep(t: float) -> float:
    """Quintic smoothstep: 0 -> 1 with zero speed and zero acceleration at both ends."""
    t = min(max(t, 0.0), 1.0)
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def ease(a: dict, b: dict, t: float) -> dict:
    """Blend two joint poses with smoothstep, t in [0, 1]. Replaces dance.ease (cosine),
    whose acceleration is not zero at the ends, so starts and stops were a little abrupt."""
    k = smoothstep(t)
    return {j: a[j] + (b[j] - a[j]) * k for j in JOINTS}


def glide(robot, start: dict, end: dict, seconds: float, dry_run: bool) -> dict:
    """Same as dance.glide (pen-up travel, lowering and lifting the pen), using the smoothstep ease."""
    steps = max(1, int(seconds * FPS))
    for i in range(1, steps + 1):
        tick = time.perf_counter()
        pose = ease(start, end, i / steps)
        if not dry_run:
            robot.send_action({f"{j}.pos": v for j, v in pose.items()})
        time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - tick)))
    return end


# ---------- where to slow down along a stroke ----------

def corners(stroke: list) -> list:
    """(distance along the stroke, speed factor) for each turn sharper than CORNER_MIN_DEG.

    The sharper the turn, the lower the factor: (1 + cos(turn)) / 2 gives 1 for a straight
    line, 0.5 for a right angle and MIN_FACTOR for a U-turn. Curves are drawn as many
    small turns below CORNER_MIN_DEG, so they keep full speed.
    """
    out, s = [], 0.0
    for (x0, y0), (x1, y1), (x2, y2) in zip(stroke, stroke[1:], stroke[2:]):
        s += math.hypot(x1 - x0, y1 - y0)
        a1, a2 = math.atan2(y1 - y0, x1 - x0), math.atan2(y2 - y1, x2 - x1)
        turn = abs((a2 - a1 + math.pi) % (2 * math.pi) - math.pi)
        if math.degrees(turn) > CORNER_MIN_DEG:
            out.append((s, max(MIN_FACTOR, (1 + math.cos(turn)) / 2)))  # 90 deg -> 0.5, U-turn -> MIN
    return out


def grid_crossings(pts: list, dist: list, cells: int) -> list:
    """(distance along the stroke, GRID_FACTOR) wherever the pen moves into another grid cell.

    draw.Paper blends joints bilinearly inside each cell of the taught grid, so the joint
    velocity bends at cell borders even on a straight line. Easing off there hides that kink.
    """
    cell = [(min(int(x * cells), cells - 1), min(int(y * cells), cells - 1)) for x, y in pts]
    return [(dist[i], GRID_FACTOR) for i in range(1, len(pts)) if cell[i] != cell[i - 1]]


def speed_factor(s: float, length: float, bends: list) -> float:
    """Fraction of the pace speed to use at distance s along a stroke of the given length.

    Starts from the ramp at both ends of the stroke, then takes the lowest of all the
    corner and grid-crossing dips around s. Each dip is a smoothstep from its low factor
    at the bend back to 1 at CORNER_RADIUS away. Never below MIN_FACTOR, so the pen keeps moving.
    """
    ramp = min(RAMP, length / 2)
    f = 1.0
    if ramp > 0:
        f = min(smoothstep(s / ramp), smoothstep((length - s) / ramp))
    for c, low in bends:
        f = min(f, low + (1 - low) * smoothstep(abs(s - c) / CORNER_RADIUS))
    return max(f, MIN_FACTOR)


def curve_limits(pts: list, dist: list) -> list:
    """Speed factor at each point from how tight the curve is there: radius / CURVE_RADIUS.

    Corner detection only catches single sharp turns. A small circle (an icon dot, the bowl
    of a letter) is made of many gentle turns, so the pen used to whip round it at full speed
    and the servos, lagging behind, rounded it into a blob. The radius comes from the circle
    through points a little before and after; the result is then spread over CORNER_RADIUS so
    the pen has slowed down before it reaches the curve."""
    n, k = len(pts), 4  # neighbours 4 points (0.008 paper-widths) either side
    raw = [1.0] * n
    for i in range(k, n - k):
        (x0, y0), (x1, y1), (x2, y2) = pts[i - k], pts[i], pts[i + k]
        a, b, c = math.dist((x0, y0), (x1, y1)), math.dist((x1, y1), (x2, y2)), math.dist((x0, y0), (x2, y2))
        area2 = abs((x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0))
        if area2 > 1e-12:
            radius = a * b * c / (2 * area2)
            raw[i] = max(MIN_FACTOR, min(1.0, radius / CURVE_RADIUS))
    # Look ahead and behind: take the lowest limit within CORNER_RADIUS of each point.
    out, j0, j1 = [], 0, 0
    for i in range(n):
        while dist[i] - dist[j0] > CORNER_RADIUS:
            j0 += 1
        while j1 < n - 1 and dist[j1 + 1] - dist[i] <= CORNER_RADIUS:
            j1 += 1
        out.append(min(raw[j0:j1 + 1]))
    return out


# ---------- drawing ----------

def follow(robot, pen, stroke: list, pace: draw.Pace):
    """Like draw.follow, but with a smooth speed profile along the stroke.

    Same loop as the original: once per frame, find the point at distance s along the
    stroke, send the joint pose for it, and compare the real joints with the previous
    command to detect jumps. The only difference is how far s advances each frame:
    pace speed (eased) x speed_factor(s) instead of a constant pace speed.
    """
    # Points every 0.002 paper-widths, and the distance from the stroke start to each.
    pts = draw.densify(stroke, 0.002)
    dist = [0.0]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        dist.append(dist[-1] + math.hypot(x1 - x0, y1 - y0))
    cells = len(pen.__self__.grid) - 1  # pen is paper.touch or paper.hover
    length, bends = dist[-1], corners(stroke) + grid_crossings(pts, dist, cells)
    curve = curve_limits(pts, dist)
    # s: distance drawn so far, k: index of the point just before s,
    # last: previous command (for jump detection), base: eased pace speed.
    s, k, last, base = 0.0, 0, None, pace.limit(pts[0][0])
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
        pace.xy = (x, y)  # where the tool is, so Ctrl+C lifts it straight up from here
        robot.send_action({f"{j}.pos": val for j, val in cmd.items()})
        if last is not None:
            real = draw.read_pose(robot)
            joint, lag = max(((j, abs(real[j] - last[j])) for j in JOINTS if j != "gripper"), key=lambda e: e[1])
            pace.update(lag, joint, x, y)  # a jump lowers pace.speed; base eases down to it below
        last = cmd
        if s >= length:
            break
        # The stroke profile is exact (no lag, so corners slow down in time);
        # only the jump-driven pace changes are eased.
        base += (pace.limit(x) - base) * SPEED_SMOOTHING
        s = min(s + base * min(speed_factor(s, length, bends), curve[k]) / FPS, length)
        time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - tick)))


# draw.draw() looks these up in the draw module, so swapping them is all it takes.
# This only affects the process running this script; draw.py on its own is unchanged.
draw.follow = follow
draw.glide = glide

if __name__ == "__main__":
    draw.main()
