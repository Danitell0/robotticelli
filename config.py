"""Settings that differ from one machine to the next, read from the environment or .env.

Copy .env.example to .env and fill it in. Nothing secret lives in the code: API keys
are only ever read from the environment (or a .env file, which git ignores).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IMAGES_DIR = ROOT / "images"
ASSETS_DIR = ROOT / "assets"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")  # a real environment variable wins over .env
except ImportError:
    pass

# Serial port of the SO-101 follower arm. /dev/serial/by-id/... is stable across replugs on Linux.
ARM_PORT = os.getenv("ARM_PORT", "/dev/ttyACM0")
# LeRobot calibration id: calibration is stored under ~/.cache/huggingface/lerobot/calibration/.
ROBOT_ID = os.getenv("ROBOT_ID", "robotticelli_follower")
