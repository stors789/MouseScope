#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


def build_icns(source_path: Path, output_path: Path) -> None:
    source = Image.open(source_path).convert("RGBA")
    fitted = ImageOps.contain(source, (1024, 1024), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    canvas.alpha_composite(
        fitted,
        ((canvas.width - fitted.width) // 2, (canvas.height - fitted.height) // 2),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="ICNS")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the MouseScope macOS ICNS asset.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_icns(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
