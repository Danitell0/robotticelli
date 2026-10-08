"""Interactive terminal menu for the SO-101 follower arm.

Run from the repo root, with lerobot installed:
    python arm.py

To try the menu without the arm, add --sim: a simulated arm accepts every
command at real speed, and paper setup writes to a scratch copy so the real
draw_area.json is never touched.

Pick a mode, then its options. Ctrl+C during a mode stops it and the arm
returns to its start pose; Ctrl+C at a menu goes back (or quits at the top).
The arm connects the first time a mode needs it and stays connected until you quit.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import types

import dance as dance_module
from dance import PORT, dance, glide, idle, make_robot, teach_up_pose, wave
import draw_smooth  # noqa: F401  swaps in smooth stroke following and easing for every drawing
from draw import FONT, SHAPES, Paper, Plan, describe, read_pose, teach, text_strokes, travel_to
from image_draw import DETAIL, IMAGES_DIR, METHODS, ImageDrawError, ai_available, image_to_strokes, list_images
from luck import LuckError, engines, lucky_subject, prompt_to_strokes
from moves import angry_cat, load_moves
from photo import METHODS as PHOTO_METHODS, PhotoError, methods as photo_methods, photo_to_strokes, recent_photos, show, take_photo

BOLD, DIM, CYAN, YELLOW, RED, RESET = "\033[1m", "\033[2m", "\033[36m", "\033[33m", "\033[31m", "\033[0m"

BANNER = rf"""{CYAN}{BOLD}
  ░█▀▄░█▀█░█▀▄░█▀█░▀█▀░▀█▀░▀█▀░█▀▀░█▀▀░█░░░█░░░▀█▀
  ░█▀▄░█░█░█▀▄░█░█░░█░░░█░░░█░░█░░░█▀▀░█░░░█░░░░█░
  ░▀░▀░▀▀▀░▀▀░░▀▀▀░░▀░░░▀░░▀▀▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀
{RESET}{DIM}  La Machine: a robotics hackathon by Tech Makers{RESET}
"""


CREDITS = "  A project made by Antoine, Daniel, Ilai, Lucas, Shaikha and Takako."


class SimRobot:
    """Stands in for the SO-101 with --sim: remembers the last pose it was sent."""

    def __init__(self):
        from dance import JOINTS
        self.config = types.SimpleNamespace(max_relative_target=10.0)
        self.bus = types.SimpleNamespace(enable_torque=lambda *a, **k: None, disable_torque=lambda *a, **k: None)
        self.pose = {j: 0.0 for j in JOINTS}
        self.commands = 0

    def get_observation(self):
        return {f"{j}.pos": v for j, v in self.pose.items()}

    def send_action(self, action):
        self.commands += 1
        self.pose = {j: action.get(f"{j}.pos", v) for j, v in self.pose.items()}
        return action

    def disconnect(self):
        print(f"{DIM}[sim] {self.commands} commands sent to the simulated arm{RESET}")


class Arm:
    """Connects to the arm on first use and keeps the connection open."""

    def __init__(self, sim: bool = False):
        self.robot = None
        self.sim = sim

    def get(self):
        if self.robot is None and self.sim:
            self.robot = SimRobot()
        if self.robot is None:
            if not os.path.exists(PORT):
                raise ConnectionError(f"Arm not found at {PORT}. Is the USB cable plugged in?")
            print(f"{DIM}Connecting to the arm...{RESET}")
            robot = make_robot()
            robot.connect(calibrate=False)
            self.robot = robot
        return self.robot

    def close(self):
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None


# ---------- small input helpers ----------

def clear():
    print("\033[2J\033[H", end="")


def choose(title: str, options: list[str], back: str = "Back", footer: str = "") -> int | None:
    """Show a numbered menu. Returns the chosen index, or None for back/quit."""
    print(f"{BOLD}{title}{RESET}\n")
    for i, label in enumerate(options, 1):
        print(f"  {CYAN}{i}{RESET}  {label}")
    print(f"  {CYAN}0{RESET}  {back}\n")
    if footer:
        print(f"{DIM}{footer}{RESET}\n")
    while True:
        try:
            raw = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return None
        if raw == "0":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(f"{YELLOW}Type a number from 0 to {len(options)}.{RESET}")


def ask_number(prompt: str, default: float, lo: float, hi: float, integer: bool = False) -> float:
    while True:
        raw = input(f"{prompt} {DIM}[{default:g}]{RESET}: ").strip()
        if not raw:
            return default
        try:
            value = int(raw) if integer else float(raw)
        except ValueError:
            print(f"{YELLOW}Not a number.{RESET}")
            continue
        if lo <= value <= hi:
            return value
        print(f"{YELLOW}Pick something between {lo:g} and {hi:g}.{RESET}")


def pause():
    try:
        input(f"\n{DIM}Press Enter to go back to the menu.{RESET}")
    except (KeyboardInterrupt, EOFError):
        print()


# ---------- modes ----------

# (label, loops, speed, scale)
DANCE_PRESETS = [
    ("Gentle   small, slow moves - good first run", 1, 0.7, 0.5),
    ("Groove   the normal dance, twice", 2, 1.0, 1.0),
    ("Party    bigger and faster, until Ctrl+C", 0, 1.3, 1.2),
]


def dance_mode(arm: Arm):
    while True:
        clear()
        print(BANNER)
        labels = [p[0] for p in DANCE_PRESETS] + ["Custom   choose loops, speed and size",
                                                  "Idle     stretch up and sway like a snake, until Ctrl+C",
                                                  "Wave     stretch up and wave hello"]
        pick = choose("DANCE MODE", labels)
        if pick is None:
            return
        if pick == len(DANCE_PRESETS) + 1:
            idle_mode(arm)
            continue
        if pick == len(DANCE_PRESETS) + 2:
            up_pose_mode(arm, "WAVE", "Wave stretches the arm fully up and waves hello for 10 seconds.", wave)
            continue
        if pick < len(DANCE_PRESETS):
            _, loops, speed, scale = DANCE_PRESETS[pick]
        else:
            print()
            loops = ask_number("How many times? (0 = until Ctrl+C)", 2, 0, 100, integer=True)
            speed = ask_number("Speed (0.3 slow - 1.5 fast)", 1.0, 0.3, 1.5)
            scale = ask_number("Size of the moves (0.3 small - 1.3 big)", 1.0, 0.3, 1.3)
        print(f"\n{BOLD}Dancing!{RESET} {DIM}Ctrl+C to stop.{RESET}\n")
        dance(arm.get(), loops, speed, scale)
        pause()


def idle_mode(arm: Arm):
    up_pose_mode(arm, "IDLE", "Idle stretches the arm fully up, then sways it like a snake with the tip facing up.",
                 idle)


def up_pose_mode(arm: Arm, title: str, about: str, run):
    """Menu for the dances that start from the taught 'fully up' pose (Idle, Wave)."""
    clear()
    print(BANNER)
    have_pose = dance_module.UP_POSE_FILE.exists()
    options = ["Start"] if have_pose else []
    options.append("Teach the up pose" if not have_pose else "Re-teach the up pose")
    print(about)
    print("It needs to know what 'fully up' looks like: you show it once by hand (shared by Idle and Wave).\n")
    pick = choose(title, options)
    if pick is None:
        return
    robot = arm.get()
    if options[pick].endswith("up pose"):
        print()
        teach_up_pose(robot)
        pause()
        return
    print(f"\n{BOLD}{title.capitalize()}!{RESET} {DIM}Ctrl+C to stop.{RESET}\n")
    run(robot)
    pause()


# (label, rounds, speed)
CAT_PRESETS = [
    ("Grumpy   5 rounds, a little slower", 5, 0.75),
    ("Furious  full speed, until Ctrl+C", 0, 1.0),
]
CAT_MAX_STEP = 25.0  # the sweep needs a looser per-step cap to swipe hard enough


def angry_cat_mode(arm: Arm):
    while True:
        clear()
        print(BANNER)
        labels = [p[0] for p in CAT_PRESETS] + ["Custom   choose rounds and speed"]
        pick = choose("ANGRY CAT MODE  -  stare... then swipe things off the table", labels)
        if pick is None:
            return
        if pick < len(CAT_PRESETS):
            _, rounds, speed = CAT_PRESETS[pick]
        else:
            print()
            rounds = ask_number("How many rounds? (0 = until Ctrl+C)", 5, 0, 100, integer=True)
            speed = ask_number("Speed (0.3 slow - 1.0 full)", 0.75, 0.3, 1.0)
        moves = load_moves()
        robot = arm.get()
        print(f"\n{BOLD}Mrrrow.{RESET} {DIM}Ctrl+C to stop.{RESET}\n")
        robot.config.max_relative_target = CAT_MAX_STEP
        try:
            angry_cat(robot, moves, rounds, speed)
        finally:
            robot.config.max_relative_target = 10.0
        pause()


# Per-tool drawing settings. height: 0 = the taught touch poses, 1 = hover height, higher = lighter.
# bottom_lift: extra height added towards the bottom of the paper, where the arm sits closer
# to it and presses harder. The brush needs a lighter touch: pressing spreads its bristles.
TOOLS = {
    "pen": {"height": 0.5, "bottom_lift": 0.0, "centre_lift": 0.0, "speed": 0.03},
    # Brush: presses harder towards the bottom of the page and lighter near the centre.
    "brush": {"height": 0.0, "bottom_lift": -0.5, "centre_lift": 0.6, "speed": 0.2},
}

# How the paper sits. lift_back: paper-widths the tool also moves back towards the base when it
# lifts off. vscale: vertical scale of drawings (a tilted canvas stretches them vertically).
SURFACES = {
    "table": {"name": "flat on the table", "lift_back": 0.0, "vscale": 1.0},
    "upright": {"name": "upright (vertical)", "lift_back": 0.0, "vscale": 1.0},
    "tilted": {"name": "a bit tilted", "lift_back": 0.04, "vscale": 0.8},
}
DRAW = {"tool": "brush", "surface": None, "sign": True}  # surface is read from the taught paper on first use


def tool() -> dict:
    return TOOLS[DRAW["tool"]]


def surface() -> dict:
    """The paper surface: the one saved with the taught paper, unless changed this session."""
    if DRAW["surface"] is None:
        import draw
        try:
            saved = json.loads(draw.AREA_FILE.read_text()).get("surface")
        except (OSError, ValueError):
            saved = None
        DRAW["surface"] = saved if saved in SURFACES else "table"
    return SURFACES[DRAW["surface"]]


def save_surface():
    """Remember the surface in the paper file, so it comes back next time."""
    import draw
    if draw.AREA_FILE.exists():
        area = json.loads(draw.AREA_FILE.read_text())
        area["surface"] = DRAW["surface"]
        draw.AREA_FILE.write_text(json.dumps(area, indent=2))


def tool_summary() -> str:
    t = tool()
    return (f"{DRAW['tool']}, height {t['height']:g}, bottom {t['bottom_lift']:+g}, "
            f"centre {t['centre_lift']:+g}, speed {t['speed']:g}")


def paper_summary() -> str:
    sf = surface()
    return f"{sf['name']}, lift back {sf['lift_back']:g}, vertical scale {sf['vscale']:g}"
SIMPLE_SHAPES = ["square", "circle", "heart", "star"]
# Size names -> fraction of the paper. Small and central keeps the pen from tilting.
SHAPE_SIZES = [("Small", 0.25), ("Medium", 0.35), ("Large", 0.5), ("Full page", 0.9)]
WORD_SIZES = [("Small", 0.4), ("Medium", 0.55), ("Large", 0.7), ("Full page", 0.9)]
MAX_WORD = 6


def pick_size(sizes: list) -> float | None:
    pick = choose("How big?", [name for name, _ in sizes])
    return None if pick is None else sizes[pick][1]


def open_file(path) -> None:
    """Open a file in the desktop's default viewer without blocking the menu."""
    subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def confirm_and_draw(arm: Arm, raw: list, size: float, title: str, preview=None):
    """Check the drawing against the paper, then draw it (optionally tracing in the air first)."""
    try:
        t = tool()
        sf = surface()
        plan = Plan(raw, size, t["speed"], t["height"], t["bottom_lift"], t["centre_lift"],
                    sf["lift_back"], sf["vscale"], sign=DRAW["sign"])
    except ValueError as e:
        print(f"\n{RED}{e}{RESET}")
        pause()
        return
    while True:
        clear()
        print(BANNER)
        print(f"{BOLD}{title}{RESET}  {DIM}{plan.summary()}{RESET}")
        print(f"{DIM}Tool: {tool_summary()}{RESET}")
        print(f"{DIM}Paper: {paper_summary()}{RESET}")
        if DRAW["sign"]:
            note = f", drawing shrunk to {plan.shrunk:.0%} to make room" if plan.shrunk < 1 else ""
            print(f"{DIM}Signed 'Robotticelli' in the bottom-right corner{note}{RESET}")
        if plan.tilt_warning():
            print(f"{YELLOW}{plan.tilt_warning()}{RESET}")
        print()
        slow = min(plan.speed, SLOW_SPEED)
        options = ["Draw it",
                   f"Draw it slowly      speed {slow:g}, for detailed drawings (logos, small letters)",
                   "Trace it in the air first (pen never touches)"]
        if preview:
            print(f"{DIM}Preview: {preview}{RESET}\n")
            options.append("Open the preview image")
        pick = choose("Ready?", options)
        if pick is None:
            return
        if pick == 3:
            open_file(preview)
            continue
        air = pick == 2
        print(f"\n{BOLD}{'Tracing in the air' if air else 'Drawing'}...{RESET} {DIM}Ctrl+C to stop and lift the pen.{RESET}\n")
        speed = plan.speed
        if pick == 1:
            plan.speed = slow
        try:
            plan.run(arm.get(), air=air)
        finally:
            plan.speed = speed
        if not air:
            pause()
            return
        pause()


SLOW_SPEED = 0.05  # "Draw it slowly": small details need the servos to keep up


def simple_drawing(arm: Arm):
    clear()
    print(BANNER)
    pick = choose("SIMPLE DRAWING", [s.capitalize() for s in SIMPLE_SHAPES])
    if pick is None:
        return
    print()
    size = pick_size(SHAPE_SIZES)
    if size is None:
        return
    name = SIMPLE_SHAPES[pick]
    confirm_and_draw(arm, SHAPES[name](), size, name.capitalize())


def write_word(arm: Arm):
    clear()
    print(BANNER)
    print(f"{BOLD}WRITE A WORD{RESET}  {DIM}up to {MAX_WORD} letters, A-Z and !  (empty = back){RESET}\n")
    while True:
        try:
            word = input("Word: ").strip().upper()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if not word:
            return
        bad = sorted(set(word) - set(FONT))
        if bad:
            print(f"{YELLOW}Can't write {''.join(bad)!r}. Use letters A-Z and !.{RESET}")
        elif len(word.replace(" ", "")) > MAX_WORD:
            print(f"{YELLOW}Too long - {MAX_WORD} letters max.{RESET}")
        else:
            break
    print()
    size = pick_size(WORD_SIZES)
    if size is None:
        return
    confirm_and_draw(arm, text_strokes(word), size, f'"{word}"')


IMAGE_SIZES = [("Small", 0.3), ("Medium", 0.45), ("Large", 0.6), ("Full page", 0.9)]


def draw_image(arm: Arm):
    clear()
    print(BANNER)
    images = list_images()
    if not images:
        print(f"{BOLD}DRAW AN IMAGE{RESET}\n")
        print(f"No images yet. Put a .jpg, .png or .svg in:\n  {IMAGES_DIR}")
        pause()
        return
    pick = choose(f"DRAW AN IMAGE  {DIM}from {IMAGES_DIR}{RESET}", [p.name for p in images])
    if pick is None:
        return
    image = images[pick]
    method, detail = "lines", "medium"
    if image.suffix.lower() != ".svg":  # an SVG is already vector lines: no conversion to choose
        method, detail = choose_image_conversion()
        if method is None:
            return
    print()
    size = pick_size(IMAGE_SIZES)
    if size is None:
        return

    print(f"\n{DIM}Turning {image.name} into lines...{RESET}")
    try:
        strokes, preview = image_to_strokes(image, method, detail)
    except (ImageDrawError, RuntimeError, OSError) as e:
        print(f"\n{RED}{e}{RESET}")
        pause()
        return
    confirm_and_draw(arm, strokes, size, image.name, preview=preview)


def choose_image_conversion() -> tuple[str | None, str]:
    """Ask how to turn a bitmap into lines, and how much detail. (None, "") if backed out."""
    ai_ok, ai_reason = ai_available()
    names = list(METHODS)
    labels = [METHODS[m] + ("" if m != "ai" or ai_ok else f"  {DIM}- not set up: {ai_reason}{RESET}") for m in names]
    print()
    pick = choose("How should it turn the image into lines?", labels)
    if pick is None:
        return None, ""
    method = names[pick]
    if method == "ai" and not ai_ok:
        print(f"\n{YELLOW}AI outline isn't set up: {ai_reason}.")
        print(f"Add OPENAI_API_KEY to .env, or pick another way.{RESET}")
        pause()
        return None, ""

    print()
    details = ["simple", "medium", "detailed"]
    pick = choose("How much detail?", [f"{d.capitalize():9s} up to {DETAIL[d][3]} lines" for d in details])
    if pick is None:
        return None, ""
    return method, details[pick]


LUCK_ENGINES = [
    ("claude", "Claude draws it    clean vector lines, about 10-20 s"),
    ("openai", "OpenAI paints it   then it gets vectorized, about 30-60 s"),
]


def try_your_luck(arm: Arm):
    clear()
    print(BANNER)
    status = engines()
    labels = [label + ("" if status[e][0] else f"  {DIM}- not set up: {status[e][1]}{RESET}") for e, label in LUCK_ENGINES]
    pick = choose("TRY YOUR LUCK!  -  who makes the picture?", labels)
    if pick is None:
        return
    engine = LUCK_ENGINES[pick][0]
    if not status[engine][0]:
        key = "ANTHROPIC_API_KEY" if engine == "claude" else "OPENAI_API_KEY"
        print(f"\n{YELLOW}{engine.capitalize()} isn't set up: {status[engine][1]}.")
        print(f"Export {key}, or add it to .env.{RESET}")
        pause()
        return

    print()
    try:
        subject = input(f"What should it draw? {DIM}(Enter = surprise me){RESET}: ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return
    if not subject:
        subject = lucky_subject()
        print(f"\n{BOLD}Your luck: {subject}!{RESET}")
    print()
    size = pick_size(IMAGE_SIZES)
    if size is None:
        return

    print()
    try:
        strokes, preview = prompt_to_strokes(subject, engine)
    except (LuckError, ImageDrawError, RuntimeError, OSError) as e:
        print(f"\n{RED}{e}{RESET}")
        pause()
        return
    confirm_and_draw(arm, strokes, size, subject, preview=preview)


def test_paper(arm: Arm):
    """Tap the tool down at each taught point with the current height settings."""
    import draw
    clear()
    print(BANNER)
    print(f"{BOLD}TEST THE PAPER{RESET}  {DIM}{tool_summary()}{RESET}\n")
    print("The tool visits every taught point and touches down at each, using the current")
    print("height settings. Watch for points where it presses too hard or doesn't reach.\n")
    if not draw.AREA_FILE.exists():
        print(f"{RED}No paper taught yet - use Set up the paper first.{RESET}")
        pause()
        return
    area = json.loads(draw.AREA_FILE.read_text())
    if "hover_grid" not in area:
        print(f"{YELLOW}This paper was taught the old way (one hover pose), so lifts can pull the")
        print(f"tool backwards near the edges. Re-teach it with Set up the paper.{RESET}\n")
    if choose("Start?", ["Yes, tap every taught point"]) is None:
        return
    t = tool()
    paper = Paper(area, t["height"], t["bottom_lift"], t["centre_lift"], surface()["lift_back"])
    robot = arm.get()
    home = read_pose(robot)
    current, prev = home, None
    n = len(paper.grid) - 1
    try:
        for row in range(n + 1):
            for col in range(n + 1):
                x, y = col / n, row / n
                print(f"  {describe(row, col, n)}")
                current = travel_to(robot, paper, current, prev, (x, y), 3.0 if prev is None else 1.5)
                current = draw_smooth.glide(robot, current, paper.touch(x, y), 0.8, False)
                time.sleep(1.0)
                current = draw_smooth.glide(robot, current, paper.hover(x, y), 0.5, False)
                prev = (x, y)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("Returning to start pose")
        glide(robot, read_pose(robot), home, 3.0, False)
    pause()


def draw_photo(arm: Arm):
    """Take a photo (or reuse one, e.g. from the web app), turn it into lines, draw it."""
    import datetime
    clear()
    print(BANNER)
    try:
        photos = recent_photos()
    except PhotoError as e:
        print(f"{RED}{e}{RESET}")
        pause()
        return
    labels = ["Take a new photo   3-2-1 countdown, then it snaps"]
    labels += [f"Use {p.name}  {DIM}{datetime.datetime.fromtimestamp(p.stat().st_mtime):%H:%M}{RESET}" for p in photos]
    pick = choose("TAKE A PHOTO", labels)
    if pick is None:
        return
    if pick == 0:
        while True:
            print(f"\n{BOLD}Look at the camera!{RESET}")
            try:
                photo = take_photo()
            except PhotoError as e:
                print(f"\n{RED}{e}{RESET}")
                pause()
                return
            show(photo)
            print(f"{DIM}Opened {photo.name} in the image viewer.{RESET}\n")
            keep = choose("Keep this photo?", ["Keep it", "Retake"], back="Cancel")
            if keep is None:
                return
            if keep == 0:
                break
    else:
        photo = photos[pick - 1]

    status = photo_methods()
    keys = list(PHOTO_METHODS)
    print()
    pick = choose("How should it turn the photo into lines?",
                  [PHOTO_METHODS[k] + ("" if status[k][0] else f"  {DIM}- not set up: {status[k][1]}{RESET}")
                   for k in keys])
    if pick is None:
        return
    method = keys[pick]
    if not status[method][0]:
        print(f"\n{YELLOW}That way isn't set up: {status[method][1]}.{RESET}")
        pause()
        return
    detail = "medium"
    if method != "vin":  # Vin's cartoon is already clean vector art
        print()
        details = ["simple", "medium", "detailed"]
        pick = choose("How much detail?", [f"{d.capitalize():9s} up to {DETAIL[d][3]} lines" for d in details])
        if pick is None:
            return
        detail = details[pick]
    print()
    size = pick_size(IMAGE_SIZES)
    if size is None:
        return

    print(f"\n{DIM}Turning {photo.name} into lines...{RESET}")
    try:
        strokes, preview = photo_to_strokes(photo, method, detail)
    except (PhotoError, ImageDrawError, RuntimeError, OSError) as e:
        print(f"\n{RED}{e}{RESET}")
        pause()
        return
    confirm_and_draw(arm, strokes, size, photo.name, preview=preview)


def set_up_paper(arm: Arm):
    clear()
    print(BANNER)
    print(f"{BOLD}SET UP THE PAPER{RESET}\n")
    print("Do this whenever the paper or the tool has moved. At each point (9 or 16, spread over")
    print("the paper) you touch the paper lightly, then lift about 2 cm straight away")
    print("from the surface, at right angles to it. On a flat table that's straight up; on an")
    print("angled canvas, follow the angle, so the brush comes off cleanly instead of sliding.\n")
    keys = list(SURFACES)
    pick = choose("Where is the paper?", [SURFACES[k]["name"].capitalize() for k in keys])
    if pick is None:
        return
    print()
    g = choose("How many points?", ["9 points (3 x 3)   quicker",
                                     "16 points (4 x 4)  more accurate, best for detailed or full-page drawings"])
    if g is None:
        return
    grid = (3, 4)[g]
    print()
    if choose("Start?", ["Yes, turn the torque off - I'm holding the arm"]) is None:
        return
    robot = arm.get()
    try:
        teach(robot, grid)
    finally:
        robot.bus.enable_torque()  # hold the parking pose again
    DRAW["surface"] = keys[pick]
    save_surface()
    pause()


def paper_surface(arm: Arm):
    keys = list(SURFACES)
    while True:
        clear()
        print(BANNER)
        surface()  # load the saved choice
        sf = SURFACES[DRAW["surface"]]
        print("Lift back: how far the tool also moves back towards the base when it lifts off.")
        print("Vertical scale: below 1 squashes drawings vertically (use it when they come out too tall).\n")
        pick = choose(f"PAPER SURFACE  {DIM}now: {sf['name']}{RESET}", [
            *[SURFACES[k]["name"].capitalize() + ("  (selected)" if k == DRAW["surface"] else "") for k in keys],
            f"Lift back        now {sf['lift_back']:g}  (paper-widths)",
            f"Vertical scale   now {sf['vscale']:g}",
        ])
        if pick is None:
            return
        if pick < len(keys):
            DRAW["surface"] = keys[pick]
            save_surface()
        elif pick == len(keys):
            sf["lift_back"] = ask_number("Lift back (0 = straight off, 0.04 = a bit back)", sf["lift_back"], 0.0, 0.15)
        else:
            sf["vscale"] = ask_number("Vertical scale (0.8 = 20% shorter)", sf["vscale"], 0.5, 1.5)


def tool_and_pressure(arm: Arm):
    while True:
        clear()
        print(BANNER)
        t = tool()
        print("Height: 0 = the taught touch poses, 1 = hover height (in the air). Higher = lighter.")
        print("Faint or missing lines -> lower it. Thick or pressed lines -> raise it.")
        print("Bottom lift: added towards the bottom of the paper. Negative = press harder there.")
        print("Centre lift: added at the centre, fading out to the corners. Positive = lighter there.\n")
        pick = choose(f"TOOL & PRESSURE  {DIM}now: {tool_summary()}{RESET}", [
            "Use the pen" + ("    (selected)" if DRAW["tool"] == "pen" else ""),
            "Use the brush" + ("  (selected)" if DRAW["tool"] == "brush" else ""),
            f"Height           now {t['height']:g}",
            f"Bottom lift      now {t['bottom_lift']:+g}",
            f"Centre lift      now {t['centre_lift']:+g}",
            f"Speed            now {t['speed']:g}  (paper-widths per second)",
        ])
        if pick is None:
            return
        if pick in (0, 1):
            DRAW["tool"] = ["pen", "brush"][pick]
        elif pick == 2:
            t["height"] = ask_number(f"{DRAW['tool'].capitalize()} height", t["height"], -0.3, 0.95)
        elif pick == 3:
            t["bottom_lift"] = ask_number("Bottom lift (negative = press harder)", t["bottom_lift"], -0.6, 0.6)
        elif pick == 4:
            t["centre_lift"] = ask_number("Centre lift (positive = lighter)", t["centre_lift"], -0.6, 0.6)
        else:
            t["speed"] = ask_number("Speed (0.01 slow - 0.2 fast)", t["speed"], 0.01, 0.2)


def toggle_signature(arm: Arm):
    DRAW["sign"] = not DRAW["sign"]


def drawing_mode(arm: Arm):
    options = [
        ("Simple drawing    square, circle, heart or star", simple_drawing),
        (f"Write a word      up to {MAX_WORD} letters", write_word),
        ("Draw an image     from the images/ folder", draw_image),
        ("Take a photo      snap it, turn it into lines, draw it", draw_photo),
        ("Try your luck!    AI draws whatever you ask", try_your_luck),
        ("Set up the paper  teach it where the paper is", set_up_paper),
        ("Tool & pressure", tool_and_pressure),
        ("Test the paper    tap the tool on every taught point", test_paper),
        ("Paper surface", paper_surface),
        ("Signature", toggle_signature),
    ]
    while True:
        clear()
        print(BANNER)
        labels = [label for label, _ in options]
        labels[6] = f"Tool & pressure   now: {tool_summary()}"
        labels[8] = f"Paper surface     now: {paper_summary()}"
        labels[9] = f"Signature         {'on' if DRAW['sign'] else 'off'}  (Robotticelli, bottom-right corner)"
        pick = choose("DRAWING MODE", labels)
        if pick is None:
            return
        options[pick][1](arm)


MODES = [
    ("Dance mode", dance_mode),
    ("Angry Cat mode", angry_cat_mode),
    ("Drawing mode", drawing_mode),
]


def main():
    sim = "--sim" in sys.argv
    if sim:
        global BANNER
        BANNER += f"{YELLOW}{BOLD}  SIMULATION - no arm connected{RESET}\n"
        # Paper setup in sim would save the simulated arm's poses as the paper: use a scratch copy.
        import draw
        sim_area = IMAGES_DIR / "processed" / "sim_draw_area.json"
        sim_area.parent.mkdir(parents=True, exist_ok=True)
        if draw.AREA_FILE.exists():  # fresh copy of the real paper each run
            shutil.copy(draw.AREA_FILE, sim_area)
        draw.AREA_FILE = sim_area
        dance_module.UP_POSE_FILE = sim_area.with_name("sim_idle_up.json")  # same for the idle pose
    arm = Arm(sim)
    try:
        while True:
            clear()
            print(BANNER)
            pick = choose("What should the arm do?", [m[0] for m in MODES], back="Quit", footer=CREDITS)
            if pick is None:
                break
            try:
                MODES[pick][1](arm)
            except (ConnectionError, ValueError) as e:
                print(f"\n{RED}{e}{RESET}")
                pause()
    finally:
        arm.close()
        print("Bye!")


if __name__ == "__main__":
    sys.exit(main())
