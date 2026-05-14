#!/usr/bin/env python3
"""Generate raw sprite/asset images via the OpenAI gpt-image-1 model.

This is the image generation tool for the generate2dsprite skill. It produces
the raw PNG that scripts/generate2dsprite.py then postprocesses (chroma-key
cleanup, frame splitting, alignment, QC, transparent export).

Write the creative prompt yourself (see references/prompt-rules.md); this
script only calls the API and saves the result. Keep the skill's solid flat
magenta (#FF00FF) background convention in the prompt so the local processor
can chroma-key it.

Requires the `openai` package and an OPENAI_API_KEY environment variable.

Examples:
  # plain generation
  python scripts/gen_image.py --prompt "<hand-written prompt>" \\
    --size 1024x1024 --output run/raw-sheet.png

  # reference-guided generation / edit (also used to pass a layout guide)
  python scripts/gen_image.py --prompt "<hand-written prompt>" \\
    --reference run/references/ref.png --output run/raw-sheet.png
"""

from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path

VALID_SIZES = ["1024x1024", "1536x1024", "1024x1536", "auto"]
VALID_QUALITY = ["low", "medium", "high", "auto"]


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        text = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    else:
        text = (args.prompt or "").strip()
    if not text:
        raise SystemExit("Empty prompt. Pass --prompt or --prompt-file.")
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--prompt", help="Creative prompt text. Write it yourself.")
    parser.add_argument("--prompt-file", type=Path, help="Read the prompt from a file instead of --prompt.")
    parser.add_argument("--output", required=True, type=Path, help="Where to save the generated PNG.")
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help="Reference or layout-guide image. Repeatable. When given, uses the image-edit endpoint.",
    )
    parser.add_argument(
        "--size",
        default="1024x1024",
        choices=VALID_SIZES,
        help="Output size. Square grids -> 1024x1024, wide strips -> 1536x1024, tall sheets -> 1024x1536.",
    )
    parser.add_argument("--quality", default="high", choices=VALID_QUALITY)
    parser.add_argument("--model", default="gpt-image-1.5", help="Image model id (default: gpt-image-1.5).")
    parser.add_argument("--n", type=int, default=1, help="Number of images. >1 writes -1/-2 suffixes.")
    parser.add_argument(
        "--no-save-prompt",
        action="store_true",
        help="Do not write the <output>.prompt.txt sidecar.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prompt = read_prompt(args)

    if args.n < 1:
        raise SystemExit("--n must be >= 1.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Export it before running this script.")

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise SystemExit("The 'openai' package is not installed. Run: pip install openai") from exc

    client = OpenAI()

    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.reference:
        missing = [str(p) for p in args.reference if not Path(p).exists()]
        if missing:
            raise SystemExit(f"Reference image(s) not found: {', '.join(missing)}")
        handles = [open(p, "rb") for p in args.reference]
        try:
            result = client.images.edit(
                model=args.model,
                image=handles if len(handles) > 1 else handles[0],
                prompt=prompt,
                size=args.size,
                quality=args.quality,
                n=args.n,
            )
        finally:
            for handle in handles:
                handle.close()
    else:
        result = client.images.generate(
            model=args.model,
            prompt=prompt,
            size=args.size,
            quality=args.quality,
            n=args.n,
        )

    saved: list[Path] = []
    for index, item in enumerate(result.data):
        if args.n > 1:
            target = output.with_name(f"{output.stem}-{index + 1}{output.suffix}")
        else:
            target = output
        if not item.b64_json:
            raise SystemExit("API response did not include image data.")
        target.write_bytes(base64.b64decode(item.b64_json))
        saved.append(target)

    if not args.no_save_prompt:
        for target in saved:
            Path(f"{target}.prompt.txt").write_text(prompt, encoding="utf-8")

    for target in saved:
        print(target.resolve())


if __name__ == "__main__":
    main()
