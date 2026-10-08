# Robotticelli

**La Machine**, a robotics hackathon by Tech Makers #1 winner!

An SO-101 robot arm that holds a pen and draws: shapes, words, pictures, photos of you,
or whatever you ask an AI to sketch. It can also dance and throw a tantrum.

Built in a weekend by Tech Makers, by Antoine,
Daniel, Ilai, Lucas, Shaikha and Takako.

```
  ░█▀▄░█▀█░█▀▄░█▀█░▀█▀░▀█▀░▀█▀░█▀▀░█▀▀░█░░░█░░░▀█▀
  ░█▀▄░█░█░█▀▄░█░█░░█░░░█░░░█░░█░░░█▀▀░█░░░█░░░░█░
  ░▀░▀░▀▀▀░▀▀░░▀▀▀░░▀░░░▀░░▀▀▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀░▀▀▀
```

## What it does

`arm.py` opens a terminal menu with three modes:

| Mode | What happens |
|------|--------------|
| **Dance** | Built-in choreographies, a hello wave, or an idle snake-like sway |
| **Angry Cat** | Plays moves taught by hand (sweeping things off the table, double takes...) at random |
| **Drawing** | Shapes, words, images, webcam photos and AI drawings, on paper the arm was taught by hand |

Drawing mode options:

- **Simple drawing**: square, triangle, circle, star, heart, spiral, smiley, and a Starry Night
- **Write a word**: a built-in single-stroke font
- **Draw an image**: any PNG/JPG/WebP/SVG from `images/`
- **Take a photo**: snap a webcam portrait and draw it as a cartoon
- **Try your luck!**: type a prompt (or get a random one) and Claude or OpenAI draws it
- **Set up / test the paper**, tool and pen pressure, and an optional *Robotticelli* signature

## How it works

**No inverse kinematics, just teaching by hand.** With torque off, you move the pen to a
grid of points on the paper (3×3 by default) plus one hover pose. Any point on the page is
then blended from the nearest taught joint poses. It's quick to set up and adapts to
whatever table, paper and pen holder you have.

**Image → strokes pipeline:**

1. **Outline**: get black line art from the input, via OpenAI `gpt-image-1`, OpenCV edge
   tracing, or the image as-is if it's already a line drawing. SVGs skip this step.
2. **Vectorize**: skeletonize the outline into centerline SVG paths (scikit-image + skan).
3. **Simplify**: join touching paths, drop scraps, smooth, keep the longest strokes, and
   order them nearest-neighbour so the pen travels as little as possible.
4. **Draw**: strokes are followed with smoothstep speed profiles that slow down for
   corners and grid-cell crossings, so the servos don't jolt (`draw_smooth.py`).

Every conversion writes a side-by-side preview (original | outline | what the arm will draw)
to `images/processed/`.

## Hardware

- [SO-101](https://github.com/TheRobotStudio/SO-ARM100) follower arm (Feetech STS3215 servos), calibrated with [LeRobot](https://github.com/huggingface/lerobot)
- A pen clamped in the gripper
- A webcam, for the photo mode

## Setup

Python 3.12+.

```bash
git clone <this repo> robotticelli
cd robotticelli

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # set ARM_PORT, ROBOT_ID and any API keys
```

Then calibrate the follower with LeRobot (`lerobot-calibrate`, or `tools/calibrate.py` if
the gripper motor is removed), using the same id as `ROBOT_ID`.

### API keys

All optional; each one unlocks a feature. Keys are only read from the environment or the
git-ignored `.env` file, never from the code. Everything else (dancing, moves, shapes,
words, SVGs, edge tracing) works without any key.

| Key | Used for |
|-----|----------|
| `ANTHROPIC_API_KEY` | Try your luck with Claude |
| `OPENAI_API_KEY` | AI outlines of images and photos, Try your luck with OpenAI |
| `NEBIUS_API_KEY` | Vin portrait cartoons (Kimi on Nebius Token Factory) |

## Usage

```bash
python arm.py          # the menu
python arm.py --sim    # the menu with a simulated arm, no hardware needed
```

Each mode is also a standalone script:

```bash
python draw.py --teach                 # teach the paper (3x3 grid)
python draw.py --shape star --dry-run  # check without moving
python draw.py --text HELLO
python draw_smooth.py --shape starry_night --air   # trace in the air first

python dance.py --loops 3 --speed 1.5
python moves.py --teach sweep          # record a move by hand
python moves.py --random --loops 5

python image_draw.py cat-photo.jpg --method edges --detail medium
python image_draw.py grumpy-cat.svg
python photo.py --method vin
python luck.py "a cat wearing a hat" --engine claude
```

`Ctrl+C` always stops the arm and returns it to its starting pose.

## Project layout

```
arm.py            interactive menu, entry point
dance.py          arm connection, easing helpers, dance and idle poses
moves.py          teach and play named moves (Angry Cat mode)
draw.py           paper model, shapes, font, stroke following
draw_smooth.py    smoothstep motion profiles, patched into draw.py
image_draw.py     image -> strokes pipeline
photo.py          webcam capture and photo -> strokes
luck.py           prompt -> AI drawing -> strokes
config.py         machine-specific settings from .env

pipeline/         OpenAI outline, centerline vectorizer, SVG parser, Vin portraits
assets/           SO-101 URDF, for tools/dataset_to_svg.py

data/             taught paper grid, idle pose and moves (specific to our arm and table)
images/           pictures to draw; previews go to images/processed/ and webcam
                  photos to images/camera/ (both git-ignored)
tools/            motor setup, bus scan, power check, calibration without a gripper,
                  and recovering an SVG from a robot replay dataset
```

## Troubleshooting

- **"Arm not found"**: check `ARM_PORT` in `.env` (`ls /dev/serial/by-id/`).
- **Motor not responding**: `python tools/power_check.py` and `python tools/scan_bus.py <port>`.
- **Pen presses too hard / lines fade**: adjust *Tool & pressure* in the menu, or
  `--pen-height` on `draw.py`.
- **The arm draws in the wrong place**: re-teach the paper. `data/draw_area.json` is the
  calibration of our table at the hackathon and won't match yours.

## Credits

- `pipeline/` is adapted from the team's web app,
  [robo-replicator](https://github.com/Lucas1981/robo-replicator), which turns a photo into
  SVG paths and a LeRobot replay dataset.
- `assets/so101_new_calib.urdf` is the SO-101 model from
  [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100).
- Thanks to Tech Makers for organising La Machine.

## License

[MIT](LICENSE)
