#!/usr/bin/env python3
"""Convert an ordered range of photos into a video.

Usage examples:
  python photo2video_range.py -i /path/to/images -o out.mp4 -r 30
  python photo2video_range.py -i /path/to/images -o 100_200.mp4 --start 100 --end 200

The script sorts files using natural sort so names like img1.jpg,img2.jpg,img10.jpg order correctly.
It uses OpenCV (`cv2`) to write an mp4 video. Install with `pip install opencv-python` if missing.
Range options use zero-based, inclusive indices after sorting, so --start 100 --end 200 writes 101 frames.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import List, Optional


def natural_key(s: str):
    # Split string into list of ints and text for natural sorting
    _nsre = re.compile(r"(\d+)")
    return [int(text) if text.isdigit() else text.lower() for text in _nsre.split(s)]


def collect_images(folder: str, exts: List[str], recursive: bool) -> List[str]:
    pattern = "**/*" if recursive else "*"
    files: List[str] = []
    for ext in exts:
        g = os.path.join(folder, f"{pattern}.{ext}")
        files.extend(glob.glob(g, recursive=recursive))
    files = [f for f in files if os.path.isfile(f)]
    files.sort(key=natural_key)
    return files


def select_images(images: List[str], start: int, end: Optional[int], step: int) -> List[str]:
    stop = None if end is None else end + 1
    return images[start:stop:step]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Assemble a selected range of photos into a video")
    parser.add_argument("-i", "--input-dir", required=True, help="Folder containing images")
    parser.add_argument("-o", "--output", default="output.mp4", help="Output video file (mp4)")
    parser.add_argument("-r", "--fps", type=int, default=3, help="Frames per second")
    parser.add_argument("--exts", default="png", help="Comma-separated image extensions to include")
    parser.add_argument("--recursive", action="store_true", help="Recursively search subfolders")
    parser.add_argument("--resize", action="store_true", help="Resize images to match first image if sizes differ")
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based inclusive start index after natural sorting. Default: 0",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Zero-based inclusive end index after natural sorting. Default: last image",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Keep every Nth selected image. Default: 1",
    )
    args = parser.parse_args(argv)

    if args.start < 0:
        parser.error("--start must be >= 0")
    if args.end is not None and args.end < 0:
        parser.error("--end must be >= 0")
    if args.step < 1:
        parser.error("--step must be >= 1")
    if args.end is not None and args.end < args.start:
        parser.error("--end must be greater than or equal to --start")

    try:
        import cv2
    except Exception:
        print("OpenCV (cv2) is required. Install with: pip install opencv-python", file=sys.stderr)
        raise

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]
    images = collect_images(input_dir, exts, args.recursive)
    if not images:
        print(f"No images found in {input_dir} with extensions {exts}", file=sys.stderr)
        sys.exit(1)

    original_total = len(images)
    selected_end = args.end if args.end is not None else original_total - 1
    images = select_images(images, args.start, args.end, args.step)
    if not images:
        print(
            f"No images selected from {original_total} sorted images "
            f"with start={args.start}, end={selected_end}, step={args.step}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Read first image to get size
    first = cv2.imread(images[0])
    if first is None:
        print(f"Failed to read first image: {images[0]}", file=sys.stderr)
        sys.exit(1)
    height, width = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.abspath(args.output)
    writer = cv2.VideoWriter(out_path, fourcc, float(args.fps), (width, height))
    if not writer.isOpened():
        print(f"Failed to open video writer for {out_path}", file=sys.stderr)
        sys.exit(1)

    total = len(images)
    print(
        f"Writing {total}/{original_total} frames "
        f"(start={args.start}, end={selected_end}, step={args.step}) "
        f"-> {out_path} at {args.fps} FPS"
    )
    print(f"First frame: {os.path.basename(images[0])}")
    print(f"Last frame : {os.path.basename(images[-1])}")
    skipped = 0
    for idx, img_path in enumerate(images, start=1):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not read {img_path}; skipping", file=sys.stderr)
            skipped += 1
            continue
        h, w = img.shape[:2]
        if (h, w) != (height, width):
            if args.resize:
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            else:
                # Crop or pad: center-crop if larger, pad with black if smaller
                if h > height or w > width:
                    # center-crop
                    top = max(0, (h - height) // 2)
                    left = max(0, (w - width) // 2)
                    img = img[top:top + height, left:left + width]
                    if img.shape[0] != height or img.shape[1] != width:
                        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                else:
                    # pad
                    pad_top = (height - h) // 2
                    pad_bottom = height - h - pad_top
                    pad_left = (width - w) // 2
                    pad_right = width - w - pad_left
                    img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)

        writer.write(img)
        if idx % 50 == 0 or idx == total:
            print(f"  wrote {idx}/{total} frames", end="\r")

    writer.release()
    print()
    print(f"Done. Wrote {total - skipped} frames (skipped {skipped}) to {out_path}")


if __name__ == "__main__":
    main()
