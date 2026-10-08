"""Scan the Feetech bus: ping every ID across every baudrate.

Read-only. Writes nothing to motor EEPROM.
Usage: python tools/scan_bus.py [port]
"""
import sys
import scservo_sdk as scs

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
BAUDRATES = [1000000, 500000, 250000, 128000, 115200, 57600, 38400, 19200]
STS3215 = 777

ph = scs.PortHandler(PORT)
if not ph.openPort():
    sys.exit(f"FAIL: could not open {PORT}")
print(f"Port {PORT} open.\n")

pk = scs.PacketHandler(0)
found_any = False

for br in BAUDRATES:
    ph.setBaudRate(br)
    hits = []
    for mid in range(scs.MAX_ID + 1):
        model, comm, err = pk.ping(ph, mid)
        if comm == scs.COMM_SUCCESS:
            tag = "sts3215" if model == STS3215 else f"UNKNOWN model {model}"
            hits.append((mid, model, tag))
    if hits:
        found_any = True
        print(f"baudrate {br}:")
        for mid, model, tag in hits:
            print(f"    id={mid:<3} model={model:<5} {tag}")

ph.closePort()

if not found_any:
    print("NOTHING FOUND on any baudrate.")
    print("  -> no motor is responding: check the 3-pin cable, the motor, and that")
    print("     the power supply is still connected to the controller board.")
else:
    print("\nIf more than one id is listed, you have multiple motors on the bus.")
    print("Disconnect all but the one you are setting up.")
