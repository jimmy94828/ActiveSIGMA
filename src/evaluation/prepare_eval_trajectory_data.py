import argparse
import errno
import os
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an eval_data_dir for trajectory evaluation by pairing a new "
            "traj.txt with an existing results_habitat folder."
        )
    )
    parser.add_argument(
        "--source-eval-data-dir",
        type=Path,
        default=None,
        help="Existing eval data directory containing results_habitat/.",
    )
    parser.add_argument(
        "--source-results-habitat",
        type=Path,
        default=None,
        help="Existing results_habitat directory. Overrides --source-eval-data-dir/results_habitat.",
    )
    parser.add_argument(
        "--traj-file",
        type=Path,
        required=True,
        help="New trajectory text file. Values are parsed as one or more 4x4 poses.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output eval data directory to create/update.",
    )
    parser.add_argument(
        "--copy-results-habitat",
        action="store_true",
        help="Copy results_habitat instead of creating a symlink.",
    )
    parser.add_argument(
        "--overwrite-traj",
        action="store_true",
        help="Overwrite out_dir/traj.txt if it already exists.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def load_pose_blocks(path: Path) -> np.ndarray:
    poses: List[np.ndarray] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                values = np.asarray([float(token) for token in line.split()], dtype=np.float64)
            except ValueError:
                continue
            if values.size % 16 != 0:
                raise ValueError(
                    f"{path}:{line_no} has {values.size} numeric values; expected a multiple of 16."
                )
            poses.extend(values.reshape(-1, 4, 4))

    if not poses:
        raise ValueError(f"No 4x4 poses were parsed from {path}")

    stacked = np.stack(poses, axis=0)
    if not np.all(np.isfinite(stacked)):
        raise ValueError(f"Trajectory contains non-finite values: {path}")
    return stacked


def write_traj_file(path: Path, poses: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite-traj to replace it.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pose in poses:
            flat = pose.reshape(-1)
            f.write(" ".join(f"{value:.9g}" for value in flat))
            f.write("\n")


def resolve_results_habitat(
    source_eval_data_dir: Optional[Path],
    source_results_habitat: Optional[Path],
) -> Path:
    if source_results_habitat is not None:
        candidate = resolve_path(source_results_habitat)
    elif source_eval_data_dir is not None:
        candidate = resolve_path(source_eval_data_dir) / "results_habitat"
    else:
        raise ValueError("Pass --source-eval-data-dir or --source-results-habitat.")

    if not candidate.is_dir():
        raise FileNotFoundError(f"results_habitat directory not found: {candidate}")
    return candidate


def same_target(path: Path, target: Path) -> bool:
    try:
        return path.resolve() == target.resolve()
    except FileNotFoundError:
        return False


def ensure_results_habitat(src: Path, dst: Path, copy_tree: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if same_target(dst, src):
            return
        raise FileExistsError(
            f"{dst} already exists and does not point to {src}. "
            "Choose a new --out-dir or remove the existing results_habitat manually."
        )

    if copy_tree:
        shutil.copytree(src, dst)
    else:
        try:
            os.symlink(src, dst, target_is_directory=True)
        except OSError as exc:
            if exc.errno == errno.ENOSYS:
                raise OSError(
                    "The output filesystem does not support symlinks. "
                    "Choose an --out-dir on a filesystem that supports symlinks, "
                    "for example /tmp/..., or pass --copy-results-habitat."
                ) from exc
            raise


def main() -> None:
    args = parse_args()
    traj_file = resolve_path(args.traj_file)
    out_dir = resolve_path(args.out_dir)
    results_src = resolve_results_habitat(args.source_eval_data_dir, args.source_results_habitat)

    if not traj_file.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {traj_file}")

    poses = load_pose_blocks(traj_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_traj_file(out_dir / "traj.txt", poses, args.overwrite_traj)
    ensure_results_habitat(results_src, out_dir / "results_habitat", args.copy_results_habitat)

    link_mode = "copied" if args.copy_results_habitat else "linked"
    print(f"Prepared eval data : {out_dir}")
    print(f"Trajectory poses   : {poses.shape[0]}")
    print(f"results_habitat    : {link_mode} from {results_src}")


if __name__ == "__main__":
    main()
