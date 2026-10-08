"""Calibrate the follower arm, with or without the gripper motor (follows HAS_GRIPPER in dance.py).

Same steps as lerobot-calibrate, which fails when motor 6 is missing:
  1. move the arm to the middle of its range (the pose lerobot shows) and press Enter
  2. move every joint except wrist_roll through its full range, then press Enter

The previous calibration file is backed up next to it first, and its gripper entry
is kept, so reinstalling the gripper later doesn't need a new gripper calibration.

Usage (from the repo root):
    python tools/calibrate.py
"""
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dance import HAS_GRIPPER, make_robot  # noqa: E402

robot = make_robot(max_relative_target=None)
path = robot.calibration_fpath
old = json.loads(path.read_text()) if path.is_file() else {}
if path.is_file():
    backup = path.with_name(f"{path.stem}.backup-{time.strftime('%Y%m%d-%H%M%S')}.json")
    shutil.copy(path, backup)
    print(f"Backed up the current calibration to {backup}")

print(f"Calibrating {'with' if HAS_GRIPPER else 'without'} the gripper motor: {', '.join(robot.bus.motors)}\n")
robot.bus.connect()
try:
    robot.calibration = {}  # always run a fresh calibration instead of offering the old file
    robot.calibrate()
finally:
    robot.bus.disconnect(disable_torque=True)

if not HAS_GRIPPER and "gripper" in old:
    saved = json.loads(path.read_text())
    saved["gripper"] = old["gripper"]
    path.write_text(json.dumps(saved, indent=4))
    print("Kept the old gripper calibration in the file for when the gripper goes back on.")
