"""Assign ID + baudrate to ONE motor, on either arm.

Refuses to run unless exactly one motor is on the bus. That condition is
what avoids the broadcast_ping collision that crashes lerobot-setup-motors
with 'IndexError: list index out of range'.

Usage:
    python tools/setup_motor.py <follower|leader> <joint> [port]

Joints (in both arms):
    shoulder_pan  shoulder_lift  elbow_flex  wrist_flex  wrist_roll  gripper

Example (replacement elbow motor on the follower):
    python tools/setup_motor.py follower elbow_flex /dev/ttyACM1
"""
import sys
import scservo_sdk as scs

BAUDRATES = [1000000, 500000, 250000, 128000, 115200, 57600, 38400, 19200]

if len(sys.argv) < 3:
    sys.exit(__doc__)

arm, joint = sys.argv[1].lower(), sys.argv[2]
port = sys.argv[3] if len(sys.argv) > 3 else ("/dev/ttyACM1" if arm == "follower" else "/dev/ttyACM0")

if arm == "follower":
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    dev = SO101Follower(SO101FollowerConfig(port=port, id="setup_tmp"))
elif arm == "leader":
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
    dev = SO101Leader(SO101LeaderConfig(port=port, id="setup_tmp"))
else:
    sys.exit(f"First arg must be 'follower' or 'leader', got '{arm}'")

if joint not in dev.bus.motors:
    sys.exit(f"Unknown joint '{joint}'. Valid: {list(dev.bus.motors)}")
target_id = dev.bus.motors[joint].id

# ---- preflight: how many motors are on this bus? ----
ph = scs.PortHandler(port)
if not ph.openPort():
    sys.exit(f"Could not open {port}. Is the board connected via USB?")
pk = scs.PacketHandler(0)
found = []
for br in BAUDRATES:
    ph.setBaudRate(br)
    hits = [i for i in range(scs.MAX_ID + 1) if pk.ping(ph, i)[1] == scs.COMM_SUCCESS]
    if hits:
        found = [(br, i) for i in hits]
        break
ph.closePort()

if not found:
    sys.exit(
        "No motor responding.\n"
        "  - is the board's POWER SUPPLY connected and the LED lit?\n"
        "  - is the 3-pin cable seated at both ends?\n"
        "  - Waveshare board: both jumpers on the 'B' (USB) channel?"
    )
if len(found) > 1:
    ids = ", ".join(str(i) for _, i in found)
    sys.exit(
        f"{len(found)} motors on the bus (ids: {ids}).\n"
        f"Connect ONLY the new '{joint}' motor -- unplug the others at the BOARD end.\n"
        f"Running with several attached is what causes the IndexError."
    )

br, found_id = found[0]
print(f"Found exactly one motor: id={found_id} @ {br} baud")
if found_id == target_id and br == 1000000:
    print(f"It is ALREADY set to '{joint}' (id={target_id}) at 1000000 baud. Nothing to do.")
    sys.exit(0)

print(f"Writing: '{joint}' -> id={target_id}, baudrate=1000000 ...")
dev.bus.setup_motor(joint, initial_baudrate=br, initial_id=found_id)
print(f"DONE: '{joint}' is now id={target_id} @ 1000000 baud")
print("\nNEXT: this motor's zero position differs from the burned one,")
print("      so you MUST recalibrate this arm before teleoperating.")
