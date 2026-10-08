"""Quick health check: are both arms' motor buses alive?

Read-only. Pings ids 1-6 at 1Mbaud on each arm.
Usage: python tools/power_check.py
"""
import scservo_sdk as scs

ARMS = (("LEADER", "/dev/ttyACM0"), ("FOLLOWER", "/dev/ttyACM1"))
ok = True

for name, port in ARMS:
    ph = scs.PortHandler(port)
    if not ph.openPort():
        print(f"{name:9} {port}  PORT WILL NOT OPEN (board unplugged from USB?)")
        ok = False
        continue
    ph.setBaudRate(1000000)
    pk = scs.PacketHandler(0)
    alive = [i for i in range(1, 7) if pk.ping(ph, i)[1] == scs.COMM_SUCCESS]
    ph.closePort()

    if len(alive) == 6:
        print(f"{name:9} {port}  OK - all 6 motors alive")
    elif alive:
        missing = [i for i in range(1, 7) if i not in alive]
        print(f"{name:9} {port}  PARTIAL - alive {alive}, MISSING {missing}")
        ok = False
    else:
        print(f"{name:9} {port}  NO MOTORS - motor power is off")
        print(f"{'':9} -> check barrel jack + board LED; if the PSU latched into")
        print(f"{'':9}    protection, unplug it from the mains for 30s to reset")
        ok = False

print()
print("READY for teleoperate" if ok else "NOT READY - fix the above first")
