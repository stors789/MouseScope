#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageChops, ImageOps


ICON_SIZE = 1024
ARTWORK_SIZE = 760
BACKGROUND_COLOR = (242, 246, 249, 255)


def _extract_artwork(source: Image.Image) -> Image.Image:
    """Remove the scanned white page while preserving the original logo pixels."""
    rgb = source.convert("RGB")
    white = Image.new("RGB", rgb.size, "white")
    difference = ImageChops.difference(rgb, white).convert("L")

    # Ignore the slight off-white noise in the source image when finding the logo.
    content_mask = difference.point(lambda value: 255 if value >= 12 else 0)
    content_bounds = content_mask.getbbox()
    if content_bounds is None:
        raise ValueError("The icon source does not contain visible artwork")

    rgb = rgb.crop(content_bounds)
    difference = difference.crop(content_bounds)

    # A soft white-key removes the rectangular page without redrawing the logo.
    alpha = difference.point(
        lambda value: 0
        if value <= 6
        else 255
        if value >= 28
        else round((value - 6) * 255 / 22)
    )
    artwork = rgb.convert("RGBA")
    artwork.putalpha(alpha)
    return artwork


def build_icns(source_path: Path, output_path: Path) -> None:
    source = Image.open(source_path)
    artwork = _extract_artwork(source)
    fitted = ImageOps.contain(
        artwork,
        (ARTWORK_SIZE, ARTWORK_SIZE),
        method=Image.Resampling.LANCZOS,
    )

    # Keep the background opaque and full-bleed. macOS applies the platform
    # rounded-rectangle mask; transparent legacy artwork gets an extra gray plate.
    canvas = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), BACKGROUND_COLOR)
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
