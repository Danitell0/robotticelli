"""Configure ONE named leader motor, with a pre-flight bus check.

Refuses to run unless exactly one motor is on the bus, which is the
condition that avoids the broadcast_ping collision / IndexError.

Usage: python tools/setup_one.py <motor_name> [port]
  e.g. python tools/setup_one.py elbow_flex
"""
import sys
import scservo_sdk as scs
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

BAUDRATES = [1000000, 500000, 250000, 128000, 115200, 57600, 38400, 19200]

if len(sys.argv) < 2:
    sys.exit(__doc__)
motor = sys.argv[1]
port = sys.argv[2] if len(sys.argv) > 2 else "/dev/ttyACM0"

leader = SO101Leader(SO101LeaderConfig(port=port, id="setup_tmp"))
if motor not in leader.bus.motors:
    sys.exit(f"Unknown motor '{motor}'. Valid: {list(leader.bus.motors)}")
target_id = leader.bus.motors[motor].id

# --- pre-flight: count motors on the bus, pinging each id individually ---
ph = scs.PortHandler(port)
if not ph.openPort():
    sys.exit(f"Could not open {port}")
pk = scs.PacketHandler(0)
present = []
for br in BAUDRATES:
    ph.setBaudRate(br)
    hits = [i for i in range(scs.MAX_ID + 1)
            if pk.ping(ph, i)[1] == scs.COMM_SUCCESS]
    if hits:
        present = [(br, i) for i in hits]
        break
ph.closePort()

if len(present) == 0:
    sys.exit("No motor responding. Check the 3-pin cable and that power is connected.")
if len(present) > 1:
    ids = ", ".join(str(i) for _, i in present)
    sys.exit(
        f"{len(present)} motors on the bus (ids: {ids}).\n"
        f"Disconnect all but '{motor}' at the BOARD end, then retry.\n"
        f"Running with several motors attached is what causes the IndexError."
    )

br, found_id = present[0]
print(f"Exactly one motor found: id={found_id} at baudrate={br}")
print(f"Setting it to '{motor}' -> id={target_id}, baudrate=1000000")

leader.bus.setup_motor(motor, initial_baudrate=br, initial_id=found_id)
print(f"DONE: '{motor}' motor id set to {target_id}")
