"""Teach the SO-101 follower arm named moves by hand, then play them back,
one by name or picked at random. No leader arm needed.

1. Teach a move (torque is off, you move the arm by hand):
    python moves.py --teach sweep
    python moves.py --teach wave
   At each waypoint, press Enter to record it and give the seconds to reach it.
   Type 'd' + Enter when done. Each move is saved to data/moves/<name>.json;
   teaching an existing name overwrites it.

   A sweep that knocks things off the table is 4 waypoints:
     a. raised, above one side of the table          (1.5 s)
     b. lowered, gripper just above the surface      (1.0 s)
     c. same height, swung across to the other side  (0.4 s  <- the fast stroke)
     d. raised again                                 (1.0 s)

2. Play:
    python moves.py --list
    python moves.py --move sweep --loops 3
    python moves.py --random --loops 5     # random move each round
    python moves.py --random --loops 0     # forever
   Add --speed 0.5 for a slow test run.

Ctrl+C stops at any time and returns the arm to its starting pose.
"""
import argparse
import json
import random
import time

from config import DATA_DIR
from dance import JOINTS, PORT, glide, make_robot

MOVES_DIR = DATA_DIR / "moves"


def read_pose(robot) -> dict:
    obs = robot.get_observation()
    return {j: obs[f"{j}.pos"] for j in JOINTS}


def load_moves() -> dict:
    return {f.stem: json.loads(f.read_text()) for f in sorted(MOVES_DIR.glob("*.json"))}


def teach(robot, name: str):
    robot.bus.disable_torque()
    print(f"Teaching '{name}'. Torque OFF: hold the arm and move it by hand.\n")
    waypoints = []
    while True:
        cmd = input(f"Move to waypoint {len(waypoints) + 1}, then Enter to record (or 'd' + Enter when done): ")
        if cmd.strip().lower() == "d":
            break
        pose = read_pose(robot)
        secs = input("  seconds to reach this pose [1.0]: ").strip()
        waypoints.append({"pose": pose, "seconds": float(secs) if secs else 1.0})
        print("  recorded: " + "  ".join(f"{j}={v:.1f}" for j, v in pose.items()))

    if waypoints:
        MOVES_DIR.mkdir(exist_ok=True)
        path = MOVES_DIR / f"{name}.json"
        path.write_text(json.dumps(waypoints, indent=2))
        print(f"\nSaved {len(waypoints)} waypoints to {path}")
    input("Rest the arm in its parking pose, then Enter to finish.")


def pick(moves: dict, name: str | None, last: str | None) -> str:
    if name:
        return name
    # Avoid playing the same move twice in a row when there's a choice.
    choices = [m for m in moves if m != last] or list(moves)
    return random.choice(choices)


def perform(robot, waypoints: list, current: dict, speed: float) -> dict:
    """Play a move's waypoints after the first. Call once the arm is at waypoints[0]."""
    for wp in waypoints[1:]:
        current = glide(robot, current, wp["pose"], wp["seconds"] / speed, False)
    return current


ANGRY_CAT_MOVES = {"sweep": 0.7, "anger": 0.3}  # move -> chance of picking it


def angry_cat(robot, moves: dict, rounds: int, speed: float, stare: tuple = (1.5, 3.5)):
    """Cat on a table: stare at something, then swipe it off (sweep) or throw a fit (anger)."""
    missing = [m for m in ANGRY_CAT_MOVES if m not in moves]
    if missing:
        raise ValueError(f"Angry Cat needs these taught moves: {', '.join(missing)}. Teach with moves.py --teach NAME.")
    home = read_pose(robot)
    current = home
    try:
        n = 0
        while rounds == 0 or n < rounds:
            n += 1
            name = random.choices(list(ANGRY_CAT_MOVES), weights=list(ANGRY_CAT_MOVES.values()))[0]
            waypoints = moves[name]
            # Creep into the move's first pose, then stare with the odd tail-twitch of the wrist.
            print(f"[{n}] ...stares at the table...")
            current = glide(robot, current, waypoints[0]["pose"], 2.0, False)
            current = stare_at(robot, current, random.uniform(*stare))
            print(f"[{n}] {'SWIPE!' if name == 'sweep' else 'HISSSS!'}")
            current = perform(robot, waypoints, current, speed)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("Returning to start pose")
        glide(robot, read_pose(robot), home, 3.0, False)


def stare_at(robot, pose: dict, seconds: float) -> dict:
    """Hold still for `seconds`, with one or two small wrist twitches."""
    end = time.time() + seconds
    for _ in range(random.randint(1, 2)):
        time.sleep(max(0.0, random.uniform(0.3, 0.6) * (end - time.time())))
        twitch = {**pose, "wrist_roll": pose["wrist_roll"] + random.choice((-6, 6))}
        glide(robot, pose, twitch, 0.12, False)
        glide(robot, twitch, pose, 0.12, False)
    time.sleep(max(0.0, end - time.time()))
    return pose


def play(robot, moves: dict, name: str | None, loops: int, speed: float):
    home = read_pose(robot)
    current = home
    last = None
    try:
        loop = 0
        while loops == 0 or loop < loops:
            loop += 1
            last = pick(moves, name, last)
            waypoints = moves[last]
            print(f"Round {loop}: {last}")
            current = glide(robot, current, waypoints[0]["pose"], 2.0 / speed, False)
            current = perform(robot, waypoints, current, speed)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("Returning to start pose")
        # Read the real position: Ctrl+C may have landed mid-glide.
        glide(robot, read_pose(robot), home, 3.0, False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default=PORT)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--teach", metavar="NAME", help="record a new move by moving the arm by hand")
    mode.add_argument("--move", metavar="NAME", help="play one move")
    mode.add_argument("--random", action="store_true", help="play a random move each round")
    mode.add_argument("--list", action="store_true", help="list taught moves")
    p.add_argument("--loops", type=int, default=1, help="rounds to play (0 = forever)")
    p.add_argument("--speed", type=float, default=1.0, help=">1 is faster")
    p.add_argument("--max-step", type=float, default=15.0, help="max degrees per step (safety cap)")
    args = p.parse_args()

    moves = load_moves()
    if args.list:
        for m, wps in moves.items():
            print(f"{m:15s} {len(wps)} waypoints, {sum(w['seconds'] for w in wps):.1f}s")
        if not moves:
            print("No moves yet. Teach one with --teach NAME.")
        return
    if not args.teach:
        if not moves:
            raise SystemExit("No moves yet. Teach one with --teach NAME.")
        if args.move and args.move not in moves:
            raise SystemExit(f"Unknown move '{args.move}'. Taught: {', '.join(moves)}")

    robot = make_robot(args.port, args.max_step)
    robot.connect(calibrate=False)
    try:
        if args.teach:
            teach(robot, args.teach)
        else:
            play(robot, moves, args.move, args.loops, args.speed)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
