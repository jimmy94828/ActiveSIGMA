#!/usr/bin/env python3
"""Batch render BEV images for per-step params.npz files.

Example:
  python tools/helper/render_bev_by_step.py \
    --results_root results/Replica \
    --method SemanticHeat \
    --stride 100 \
    --dataset Replica \
    --coord-system sim \
    --render-mode gaussian \
    --remove-front-percent 25 \
    --render-gt-semantic
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional


def parse_step_from_name(name: str) -> Optional[int]:
    match = re.search(r"(\d+)$", name)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", name)
    if match:
        return int(match.group(1))
    return None


def iter_scene_dirs(results_root: Path, scenes: Optional[List[str]]) -> Iterable[Path]:
    if scenes:
        for scene in scenes:
            yield results_root / scene
        return
    for entry in sorted(results_root.iterdir()):
        if entry.is_dir():
            yield entry


def build_command(args: argparse.Namespace, scene: str, npz_path: Path, out_path: Path) -> List[str]:
    cmd = [
        sys.executable,
        "src/visualization/render_rgb_sem_entro_bev.py",
        "--npz",
        str(npz_path),
        "--dataset",
        args.dataset,
        "--scene",
        scene,
        "--out",
        str(out_path),
        "--coord-system",
        args.coord_system,
        "--render-mode",
        args.render_mode,
        "--remove-front-percent",
        str(args.remove_front_percent),
    ]
    if args.render_gt_semantic:
        cmd.append("--render-gt-semantic")
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch render BEV maps every N steps.")
    parser.add_argument("--results_root", default="results/Replica", help="Root results dir, e.g. results/Replica")
    parser.add_argument("--method", default="SemanticHeat", help="Method dir name under scene")
    parser.add_argument("--run_glob", default="run_*", help="Run dir glob under method, e.g. run_*")
    parser.add_argument("--scenes", nargs="*", default=None, help="Scene names to include, default: all scenes")
    parser.add_argument("--npz_glob", default="splatam/params*.npz", help="Glob for step npz files")
    parser.add_argument("--stride", type=int, default=100, help="Only render steps divisible by this value")
    parser.add_argument("--include_nonstep", action="store_true", help="Also render params without step digits")
    parser.add_argument("--out_subdir", default="visualization", help="Output subdir under run dir")
    parser.add_argument("--out_prefix", default="bev_step", help="Output filename prefix")
    parser.add_argument("--dataset", default="Replica", help="Dataset name for renderer")
    parser.add_argument("--coord-system", default="sim", choices=["sim", "slam"], help="Coordinate system")
    parser.add_argument("--render-mode", default="gaussian", choices=["gaussian", "nearest"], help="Render mode")
    parser.add_argument("--remove-front-percent", type=float, default=25.0, help="Remove nearest N percent of points")
    parser.add_argument("--render-gt-semantic", action="store_true", help="Also render GT semantic BEV")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    if not results_root.exists():
        raise SystemExit(f"results_root not found: {results_root}")

    for scene_dir in iter_scene_dirs(results_root, args.scenes):
        if not scene_dir.is_dir():
            continue
        scene = scene_dir.name
        method_dir = scene_dir / args.method
        if not method_dir.is_dir():
            continue

        for run_dir in sorted(method_dir.glob(args.run_glob)):
            if not run_dir.is_dir():
                continue

            npz_paths = sorted(run_dir.glob(args.npz_glob))
            for npz_path in npz_paths:
                step = parse_step_from_name(npz_path.stem)
                if step is None and not args.include_nonstep:
                    continue
                if step is not None and args.stride > 0 and (step % args.stride != 0):
                    continue

                if step is None:
                    out_name = f"{args.out_prefix}_{npz_path.stem}.png"
                else:
                    out_name = f"{args.out_prefix}_{step:06d}.png"
                out_path = run_dir / args.out_subdir / out_name

                cmd = build_command(args, scene, npz_path, out_path)
                if args.dry_run:
                    print(" ".join(cmd))
                    continue
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
