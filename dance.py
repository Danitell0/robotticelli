"""Make the SO-101 follower arm dance on its own, no leader arm needed.

The arm eases from wherever it is into a short choreography of keyframe poses,
loops it, then eases back to where it started before releasing torque.

Joint values are in degrees, where 0 is the middle of each joint's calibrated
range. The gripper is 0 (closed) to 100 (open).

Usage (from the repo root, with lerobot installed):
    python dance.py
    python dance.py --loops 3 --speed 1.5 --scale 0.6
    python dance.py --dry-run   # print poses, don't touch the arm

Ctrl+C stops at any time and returns the arm to its starting pose.
"""
import argparse
import json
import math
import time

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from config import ARM_PORT, DATA_DIR, ROBOT_ID

PORT = ARM_PORT
FPS = 50

# The gripper motor (ID 6) is installed. Set to False if it's removed again
# (at the hackathon the pen sat on the fixed jaw with no motor 6).
HAS_GRIPPER = True

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class GripperlessFollower(SO101Follower):
    """SO-101 follower without motor 6. Everything else in these scripts still passes
    a "gripper.pos" around; it's accepted and ignored, and reads back as the last value sent."""

    def __init__(self, config):
        super().__init__(config)
        self.calibration.pop("gripper", None)
        motors = {name: m for name, m in self.bus.motors.items() if name != "gripper"}
        self.bus = FeetechMotorsBus(port=config.port, motors=motors, calibration=self.calibration)
        self._gripper = 0.0

    def get_observation(self):
        return {**super().get_observation(), "gripper.pos": self._gripper}

    def send_action(self, action):
        self._gripper = action.get("gripper.pos", self._gripper)
        sent = super().send_action({k: v for k, v in action.items() if k != "gripper.pos"})
        return {**sent, "gripper.pos": self._gripper}


def make_robot(port: str = PORT, max_relative_target: float | None = 10.0):
    """The follower arm, with or without the gripper motor depending on HAS_GRIPPER."""
    cls = SO101Follower if HAS_GRIPPER else GripperlessFollower
    return cls(SO101FollowerConfig(port=port, id=ROBOT_ID, max_relative_target=max_relative_target))

# Hard limits in degrees, kept well inside the calibrated ranges
# (our calibration file gave roughly +/-118, 106, 98, 101, 180).
LIMITS = {
    "shoulder_pan": (-60, 60),
    "shoulder_lift": (-50, 50),
    "elbow_flex": (-50, 50),
    "wrist_flex": (-60, 60),
    "wrist_roll": (-90, 90),
    "gripper": (0, 80),
}

# (pose, seconds to reach it). Poses are offsets from the centre pose.
CHOREOGRAPHY = [
    # sway left and right
    ({"shoulder_pan": 35, "wrist_roll": 45, "gripper": 10}, 1.0),
    ({"shoulder_pan": -35, "wrist_roll": -45, "gripper": 10}, 1.2),
    ({"shoulder_pan": 35, "wrist_roll": 45, "gripper": 10}, 1.2),
    ({"shoulder_pan": 0, "wrist_roll": 0}, 0.8),
    # nod twice
    ({"shoulder_lift": -20, "elbow_flex": 25, "wrist_flex": 30}, 0.6),
    ({"shoulder_lift": 15, "elbow_flex": -15, "wrist_flex": -25}, 0.6),
    ({"shoulder_lift": -20, "elbow_flex": 25, "wrist_flex": 30}, 0.6),
    ({"shoulder_lift": 15, "elbow_flex": -15, "wrist_flex": -25}, 0.6),
    # clap the gripper
    ({"gripper": 70}, 0.3),
    ({"gripper": 5}, 0.3),
    ({"gripper": 70}, 0.3),
    ({"gripper": 5}, 0.3),
    # big wrist twirl
    ({"wrist_roll": 85, "wrist_flex": 20}, 0.9),
    ({"wrist_roll": -85, "wrist_flex": -20}, 1.5),
    ({"wrist_roll": 0, "wrist_flex": 0}, 0.9),
    # back to centre
    ({}, 1.0),
]


def full_pose(offsets: dict, scale: float) -> dict:
    pose = {}
    for j in JOINTS:
        v = offsets.get(j, 0.0)
        if j != "gripper":
            v *= scale
        lo, hi = LIMITS[j]
        pose[j] = min(max(v, lo), hi)
    return pose


def ease(a: dict, b: dict, t: float) -> dict:
    """Cosine ease-in-out between two poses, t in [0, 1]."""
    k = (1 - math.cos(math.pi * t)) / 2
    return {j: a[j] + (b[j] - a[j]) * k for j in JOINTS}


def glide(robot, start: dict, end: dict, seconds: float, dry_run: bool) -> dict:
    steps = max(1, int(seconds * FPS))
    for i in range(1, steps + 1):
        tick = time.perf_counter()
        pose = ease(start, end, i / steps)
        if not dry_run:
            robot.send_action({f"{j}.pos": v for j, v in pose.items()})
        time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - tick)))
    return end


def dance(robot, loops: int = 2, speed: float = 1.0, scale: float = 1.0):
    """Play the choreography `loops` times (0 = until Ctrl+C), then return to the start pose."""
    obs = robot.get_observation()
    home = {j: obs[f"{j}.pos"] for j in JOINTS}
    print("Start pose: " + "  ".join(f"{j}={v:.1f}" for j, v in home.items()))

    current = home
    try:
        print("Moving to centre pose")
        current = glide(robot, current, full_pose({}, scale), 3.0, False)
        loop = 0
        while loops == 0 or loop < loops:
            loop += 1
            print(f"Dance loop {loop}")
            for offsets, secs in CHOREOGRAPHY:
                target = full_pose(offsets, scale)
                current = glide(robot, current, target, secs / speed, False)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("Returning to start pose")
        # Read the real position: Ctrl+C may have landed mid-glide.
        obs = robot.get_observation()
        current = {j: obs[f"{j}.pos"] for j in JOINTS}
        glide(robot, current, home, 3.0, False)


# ---------- idle: stretch up, then sway like a snake ----------

UP_POSE_FILE = DATA_DIR / "idle_up.json"
# Calibrated range of each joint less a 5 degree margin (as in draw.py).
SAFE = {"shoulder_pan": 113, "shoulder_lift": 101, "elbow_flex": 93, "wrist_flex": 96, "wrist_roll": 175}


def read_pose(robot) -> dict:
    obs = robot.get_observation()
    return {j: obs[f"{j}.pos"] for j in JOINTS}


def teach_up_pose(robot):
    """Hold the arm fully stretched straight up by hand; saved to idle_up.json."""
    robot.bus.disable_torque()
    try:
        input("Torque OFF. Hold the arm fully stretched straight UP, tip pointing up, then Enter: ")
        UP_POSE_FILE.write_text(json.dumps(read_pose(robot), indent=2))
        print(f"Saved to {UP_POSE_FILE}")
    finally:
        robot.bus.enable_torque()  # holds the up pose


def snake_pose(up: dict, t: float, amount: float) -> dict:
    """The up pose plus a travelling wave through shoulder and elbow. The wrist cancels their
    tilt (shoulder + elbow + wrist stays constant - checked on the taught paper poses), so the
    tip keeps pointing up while the body sways. amount 0..1 fades the wave in."""
    shoulder = amount * 12 * math.sin(2 * math.pi * t / 3.0)
    elbow = amount * 20 * math.sin(2 * math.pi * t / 3.0 - math.pi / 2)
    pose = dict(up)
    pose["shoulder_lift"] += shoulder
    pose["elbow_flex"] += elbow
    pose["wrist_flex"] -= shoulder + elbow
    pose["shoulder_pan"] += amount * 25 * math.sin(2 * math.pi * t / 7.0)  # slow look around
    pose["wrist_roll"] += amount * 15 * math.sin(2 * math.pi * t / 4.5)
    return {j: max(-SAFE[j], min(SAFE[j], v)) if j in SAFE else v for j, v in pose.items()}


def idle(robot, speed: float = 1.0):
    """Stretch fully up, then sway like a snake with the tip facing up, until Ctrl+C."""
    if not UP_POSE_FILE.exists():
        raise ValueError("No up pose yet - teach it first (Idle -> teach the up pose).")
    up = json.loads(UP_POSE_FILE.read_text())
    home = read_pose(robot)
    current = home
    try:
        print("Stretching up...")
        current = glide(robot, current, up, 3.0, False)
        print("Swaying. Ctrl+C to stop.")
        start = time.perf_counter()
        while True:
            tick = time.perf_counter()
            t = (tick - start) * speed
            current = snake_pose(up, t, min(1.0, t / 2.0))  # fade the wave in over 2 s
            robot.send_action({f"{j}.pos": v for j, v in current.items()})
            time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - tick)))
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("Returning to start pose")
        current = glide(robot, read_pose(robot), up, 1.5, False)
        glide(robot, current, home, 3.0, False)


def wave(robot, seconds: float = 10.0):
    """Stretch up and wave hello, over the top: a big snake sway, wide fast wrist flicks like a
    frantic waving hand, and a big swing of the base, all pulsing in bursts of excitement.
    The flicks tilt the tip around 'pointing up'; the sway itself keeps it up."""
    if not UP_POSE_FILE.exists():
        raise ValueError("No up pose yet - teach it first (Idle -> teach the up pose).")
    up = json.loads(UP_POSE_FILE.read_text())
    home = read_pose(robot)
    current = home
    try:
        print("Stretching up...")
        current = glide(robot, current, up, 3.0, False)
        print("Waving!")
        start = time.perf_counter()
        while (t := time.perf_counter() - start) < seconds:
            tick = time.perf_counter()
            # Fade in over 1 s and out over the last 1 s, so the wave starts and ends at "up".
            amount = min(1.0, t, seconds - t)
            # Bursts of excitement: everything swells and calms every 2.5 s.
            burst = 1 + 0.35 * math.sin(2 * math.pi * t / 2.5)
            # As big and fast as the wrist can follow: tuned so it stays under the 10 deg/frame
            # safety cap and off its joint limits.
            pose = snake_pose(up, t * 1.4, 1.2 * amount * burst)  # big, quick body sway
            pose["wrist_flex"] += amount * burst * 35 * math.sin(2 * math.pi * t / 0.85)  # the frantic hand wave
            pose["shoulder_pan"] += amount * 15 * math.sin(2 * math.pi * t / 1.75)  # extra side-to-side swing
            current = {j: max(-SAFE[j], min(SAFE[j], v)) if j in SAFE else v for j, v in pose.items()}
            robot.send_action({f"{j}.pos": v for j, v in current.items()})
            time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - tick)))
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("Returning to start pose")
        current = glide(robot, read_pose(robot), up, 1.5, False)
        glide(robot, current, home, 3.0, False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default=PORT)
    p.add_argument("--loops", type=int, default=2, help="times to repeat the dance (0 = forever)")
    p.add_argument("--speed", type=float, default=1.0, help=">1 is faster")
    p.add_argument("--scale", type=float, default=1.0, help="shrink (<1) or grow moves; limits still apply")
    p.add_argument("--dry-run", action="store_true", help="print the poses without connecting")
    args = p.parse_args()

    if args.dry_run:
        for offsets, secs in CHOREOGRAPHY:
            pose = full_pose(offsets, args.scale)
            print(f"{secs / args.speed:4.2f}s  " + "  ".join(f"{j}={v:6.1f}" for j, v in pose.items()))
        return

    # max_relative_target caps each step at 10 deg so a bad pose can't make the arm jump.
    robot = make_robot(args.port)
    robot.connect(calibrate=False)
    try:
        dance(robot, args.loops, args.speed, args.scale)
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()
