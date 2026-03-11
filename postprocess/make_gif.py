#!/usr/bin/env python3
"""Combine PNG frames from a directory into an animated GIF.

Usage
-----
  python postprocess/make_gif.py <sim_dir> [--fps 10] [--output movie.gif] [--resize 0.5]
  python postprocess/make_gif.py <sim_dir> --dir strings/3d --pattern "strings3d_step_*.png"
  python postprocess/make_gif.py <png_directory> --direct --pattern "*.png"

If --output is not given, the GIF is saved as animation.gif in the source directory.
"""

import argparse
import glob
import os
import re
import sys

from PIL import Image


def natural_sort_key(path):
    """Sort filenames by embedded numbers (floats, scientific notation, or integers)."""
    base = os.path.basename(path)
    # 1) Scientific notation  (e.g. 3.562500e-01)
    m = re.search(r"[-+]?\d+\.?\d*[eE][-+]?\d+", base)
    if m:
        return (0, float(m.group()))
    # 2) Plain float  (e.g. t_0.625.png  ->  0.625)
    m = re.search(r"[-+]?\d+\.\d+", base)
    if m:
        return (0, float(m.group()))
    # 3) Fall back to integer-group splitting for step-based names
    parts = re.split(r"(\d+)", base)
    return (1, [int(p) if p.isdigit() else p.lower() for p in parts])


def main():
    parser = argparse.ArgumentParser(
        description="Create animated GIF from PNG snapshots.",
    )
    parser.add_argument(
        "sim_dir", help="Simulation directory or PNG directory (with --direct)"
    )
    parser.add_argument(
        "--fps", type=float, default=10, help="Frames per second (default: 10)"
    )
    parser.add_argument("--output", type=str, default=None, help="Output GIF path")
    parser.add_argument(
        "--resize",
        type=float,
        default=1.0,
        help="Scale factor for frames (e.g. 0.5 = half size, saves disk). Default: 1.0",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="Glob pattern for PNG files (default: *.png)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Subdirectory inside sim_dir to look for PNGs (default: scan sim_dir itself)",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Treat sim_dir as the PNG directory directly (no subdirectory lookup)",
    )
    parser.add_argument(
        "--step_min",
        type=int,
        default=None,
        help="Start index (Python slice). E.g. 10 = skip first 10 files.",
    )
    parser.add_argument(
        "--step_max",
        type=int,
        default=None,
        help="End index (Python slice). E.g. -75 = exclude last 75 files.",
    )
    args = parser.parse_args()

    if args.direct:
        png_dir = args.sim_dir
    elif args.dir is not None:
        png_dir = os.path.join(args.sim_dir, args.dir)
    else:
        png_dir = args.sim_dir

    if not os.path.isdir(png_dir):
        print(f"Error: directory not found: {png_dir}")
        sys.exit(1)

    all_png = sorted(
        glob.glob(os.path.join(png_dir, args.pattern)),
        key=natural_sort_key,
    )
    png_files = all_png[args.step_min : args.step_max]

    if not png_files:
        print(f"Error: no files matching '{args.pattern}' in {png_dir}")
        print(f"  (total before slicing: {len(all_png)})")
        sys.exit(1)

    output_path = args.output or os.path.join(png_dir, "animation.gif")
    duration_ms = int(1000 / args.fps)

    print(f"Directory: {png_dir}")
    print(f"Pattern  : {args.pattern}")
    print(f"Total    : {len(all_png)} matched, {len(png_files)} after slicing "
          f"[{args.step_min}:{args.step_max}]")
    print(f"FPS      : {args.fps}  ({duration_ms} ms/frame)")
    print(f"Resize   : {args.resize:.2f}x")
    print(f"Output   : {output_path}")
    n_show = min(5, len(png_files))
    print(f"First {n_show}:")
    for f in png_files[:n_show]:
        print(f"    {os.path.basename(f)}")
    if len(png_files) > 2 * n_show:
        print(f"    ... ({len(png_files) - 2*n_show} more) ...")
    if len(png_files) > n_show:
        print(f"Last {min(n_show, len(png_files) - n_show)}:")
        for f in png_files[-n_show:]:
            print(f"    {os.path.basename(f)}")

    frames = []
    for i, path in enumerate(png_files):
        img = Image.open(path).convert("RGBA")
        if args.resize != 1.0:
            new_w = int(img.width * args.resize)
            new_h = int(img.height * args.resize)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        # GIF doesn't support RGBA; composite onto white background
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img)
        frames.append(bg.convert("RGB"))
        if (i + 1) % 20 == 0 or (i + 1) == len(png_files):
            print(f"  loaded {i + 1}/{len(png_files)}")

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Done. {size_mb:.1f} MB -> {output_path}")


if __name__ == "__main__":
    main()
