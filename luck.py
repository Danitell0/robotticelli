"""Try your luck: turn a prompt (or a random subject) into a drawing for the arm.

Two ways to get the picture:
  claude  Claude draws the line art directly as an SVG (Claude Opus 5.5).
          Needs ANTHROPIC_API_KEY in .env.
  openai  OpenAI gpt-image-1 paints a line-art image, then Lucas's pipeline
          vectorizes it. Needs OPENAI_API_KEY in .env.

Either way the result goes through the same simplify step as image_draw.py,
and the picture + a preview are saved in images/processed/.

Command line, from the repo root:
    python luck.py "a cat wearing a hat" --engine claude
"""
import argparse
import base64
import os
import random
import re
from pathlib import Path

from image_draw import (PROCESSED_DIR, ImageDrawError, preview, render_paths, simplify,
                        svg_to_paths, vectorize)

LUCKY_SUBJECTS = [
    "a cat sitting", "a rocket ship", "a sailboat on waves", "a smiling sun", "a little house with a tree",
    "a robot waving", "a fish", "a flower in a pot", "a snail", "a hot air balloon", "an owl on a branch",
    "a cupcake", "a guitar", "a turtle", "a lighthouse", "a dinosaur", "a bicycle", "a mushroom", "a whale",
    "a cactus", "a teapot", "a butterfly", "a castle", "a penguin",
]

CLAUDE_MODEL = "claude-opus-5-5"
CLAUDE_SYSTEM = """You draw simple line art for a robot arm holding a pen.

Reply with exactly one <svg> element and nothing else.
- viewBox="0 0 100 100".
- Only <path> elements, with fill="none" and stroke="black".
- Path data may use ONLY absolute uppercase commands M, L, Q, C and Z. No lowercase
  (relative) commands, no H, V, S, T or A, no transforms, no other elements, no text.
- At most 20 paths. Prefer long continuous lines over many short ones.
- A clean, recognisable cartoon outline that fills most of the canvas. No shading or hatching."""

OPENAI_PROMPT = (
    "A very simple black-and-white cartoon line drawing of {subject}. Clean, minimal black outlines "
    "on a pure white background. No shading, no color, no fills, no textures, no text. "
    "As few strokes as possible, suitable for a pen plotter robot."
)


class LuckError(RuntimeError):
    pass


def engines() -> dict[str, tuple[bool, str]]:
    """engine -> (usable, reason if not)."""
    claude_ok = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
                     or (Path.home() / ".config" / "anthropic").is_dir())
    return {
        "claude": (claude_ok, "" if claude_ok else "ANTHROPIC_API_KEY is not set"),
        "openai": (bool(os.getenv("OPENAI_API_KEY")), "" if os.getenv("OPENAI_API_KEY") else "OPENAI_API_KEY is not set"),
    }


def lucky_subject() -> str:
    return random.choice(LUCKY_SUBJECTS)


def _claude_svg(subject: str) -> bytes:
    import anthropic

    client = anthropic.Anthropic()
    try:
        response = client.beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=16000,
            output_config={"effort": "low"},  # a quick doodle, not a hard problem
            # If a safety classifier declines, the API retries on a fallback model in the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=CLAUDE_SYSTEM,
            messages=[{"role": "user", "content": f"Draw {subject}."}],
        )
    except anthropic.AuthenticationError as e:
        raise LuckError("Claude rejected the API key (ANTHROPIC_API_KEY).") from e
    except anthropic.RateLimitError as e:
        raise LuckError("Claude is rate limiting us - wait a moment and try again.") from e
    except anthropic.APIStatusError as e:
        raise LuckError(f"Claude API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LuckError("Can't reach the Claude API - check the internet connection.") from e

    if response.stop_reason == "refusal":
        raise LuckError(f"Claude declined to draw {subject!r}. Try another prompt.")
    text = "".join(b.text for b in response.content if b.type == "text")
    match = re.search(r"<svg\b.*?</svg>", text, re.S)
    if not match:
        raise LuckError("Claude didn't return an SVG. Try again.")
    svg = match.group(0)
    if re.search(r'\sd="[^"]*[mlhvcsqtaz]', svg) or re.search(r'\sd="[^"]*[HVSTA]', svg):
        raise LuckError("Claude used SVG commands the parser can't follow. Try again.")
    return svg.encode()


def _openai_png(subject: str) -> bytes:
    from pipeline.outline import OPENAI_IMAGE_MODEL
    from openai import OpenAI, OpenAIError

    try:
        result = OpenAI().images.generate(
            model=OPENAI_IMAGE_MODEL, prompt=OPENAI_PROMPT.format(subject=subject), size="1024x1024", n=1
        )
    except OpenAIError as e:
        raise LuckError(f"OpenAI image generation failed: {e}") from e
    b64 = result.data[0].b64_json if result.data else None
    if not b64:
        raise LuckError("OpenAI returned no image.")
    return base64.b64decode(b64)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "luck"


def prompt_to_strokes(subject: str, engine: str, detail: str = "medium") -> tuple[list, Path]:
    """Prompt -> strokes in 0..1 and a preview PNG path."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stem = PROCESSED_DIR / f"luck-{_slug(subject)}-{engine}"
    if engine == "claude":
        print(f"Asking Claude to draw {subject}...")
        svg = _claude_svg(subject)
        stem.with_suffix(".svg").write_bytes(svg)
        try:
            paths, w, h = svg_to_paths(svg)
        except ImageDrawError as e:
            raise LuckError(f"Couldn't read Claude's drawing: {e}") from e
        # Claude's art is already simple: render it as the "outline" image for the preview.
        outline_png = render_paths(paths, w, h)
    elif engine == "openai":
        print(f"Asking OpenAI to paint {subject} (30-60 s)...")
        outline_png = _openai_png(subject)
        stem.with_suffix(".png").write_bytes(outline_png)
        paths, w, h = vectorize(outline_png, stem.name)
    else:
        raise LuckError(f"Unknown engine {engine!r}")
    try:
        # Claude's SVG is already clean line art; only the traced OpenAI image needs simplifying.
        strokes = simplify(paths, w, h, "vector" if engine == "claude" else detail)
    except ImageDrawError as e:
        raise LuckError(str(e)) from e
    source = stem.with_suffix(".png")
    source.write_bytes(outline_png)
    return strokes, preview(source, outline_png, strokes, detail)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prompt", nargs="?", help="what to draw (empty = random)")
    p.add_argument("--engine", choices=["claude", "openai"], default="claude")
    p.add_argument("--detail", choices=["simple", "medium", "detailed"], default="medium")
    args = p.parse_args()
    subject = args.prompt or lucky_subject()
    strokes, prev = prompt_to_strokes(subject, args.engine, args.detail)
    print(f"{subject}: {len(strokes)} strokes. Preview: {prev}")


if __name__ == "__main__":
    main()
