import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

cv2 = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RGB/semantic evaluation on the same trajectory GT for multiple methods."
    )
    parser.add_argument(
        "--eval_data_dir",
        required=True,
        help="Trajectory GT folder containing traj.txt and results_habitat/.",
    )
    parser.add_argument(
        "--dataset",
        choices=["MP3D", "Replica"],
        default=None,
        help="Dataset name. Used to colorize semantic .npy maps when semantic RGB files are absent.",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=CFG=RESULT_DIR",
        help="Method label, config path, and result directory. Can be repeated.",
    )
    parser.add_argument(
        "--stage",
        default="final",
        help="Checkpoint stage/folder to load, e.g. final or exploration_stage_1.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Optional params step. RGB uses eval_rgb_by_step.py; semantic uses load_params_by_step.",
    )
    parser.add_argument(
        "--metrics",
        choices=["rgb", "semantic", "both"],
        default="both",
        help="Which metrics to run.",
    )
    parser.add_argument(
        "--eval_suffix",
        default=None,
        help="Output suffix for eval directories. Defaults to <stage>_traj or <stage>_step_<step>_traj.",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="CUDA_VISIBLE_DEVICES value for evaluator subprocesses.",
    )
    parser.add_argument(
        "--enable_vis",
        default="0",
        help="Forwarded to evaluators.",
    )
    parser.add_argument(
        "--summary_csv",
        default=None,
        help="Optional CSV path for the final summary.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip an evaluator when its output file already exists.",
    )
    parser.add_argument(
        "--plot_eval_data",
        dest="plot_eval_data",
        action="store_true",
        default=True,
        help="Save RGB/depth/semantic panels for the evaluation trajectory and rendered semantic comparison panels.",
    )
    parser.add_argument(
        "--no_plot_eval_data",
        dest="plot_eval_data",
        action="store_false",
        help="Disable saving trajectory panels and rendered semantic comparison panels.",
    )
    parser.add_argument(
        "--plot_dir",
        default=None,
        help="Optional output directory for evaluation trajectory plots. Defaults to each run's splatam/eval_<suffix>/trajectory_rgbd_semantic/.",
    )
    parser.add_argument(
        "--plot_stride",
        type=int,
        default=1,
        help="Save one trajectory plot every N frames.",
    )
    return parser.parse_args()


def parse_run(value: str) -> Tuple[str, str, str]:
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"--run must be LABEL=CFG=RESULT_DIR, got: {value}")
    return parts[0], parts[1], parts[2]


def run_command(cmd: List[str], env: dict, cwd: Path) -> None:
    print("\n==> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def eval_dir_for(result_dir: str, eval_suffix: str) -> Path:
    return Path(result_dir) / "splatam" / f"eval_{eval_suffix}"


def ensure_cv2() -> None:
    global cv2
    if cv2 is not None:
        return
    try:
        import cv2 as cv2_module
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Trajectory plotting requires OpenCV. Install cv2 or pass --no_plot_eval_data."
        ) from exc
    cv2 = cv2_module


def natural_key(path: Path) -> List[object]:
    key: List[object] = []
    digits = ""
    for char in path.name:
        if char.isdigit():
            digits += char
        else:
            if digits:
                key.append(int(digits))
                digits = ""
            key.append(char)
    if digits:
        key.append(int(digits))
    return key


def frame_index(path: Path) -> int:
    digits = "".join(char for char in path.stem if char.isdigit())
    return int(digits) if digits else -1


def find_rgb_paths(results_dir: Path) -> List[Path]:
    paths = sorted(results_dir.glob("frame*.jpg"), key=natural_key)
    if not paths:
        paths = sorted((results_dir / "rgb").glob("color*.jpg"), key=natural_key)
    if not paths:
        paths = sorted(results_dir.glob("frame*.png"), key=natural_key)
    return paths


def find_depth_paths(results_dir: Path) -> List[Path]:
    return sorted(results_dir.glob("depth*.png"), key=natural_key)


def find_semantic_rgb_paths(results_dir: Path) -> List[Path]:
    semantic_dir = results_dir / "semantic"
    paths = sorted(semantic_dir.glob("semantic_rgb*.png"), key=natural_key)
    if not paths:
        paths = sorted(semantic_dir.glob("*rgb*.png"), key=natural_key)
    return paths


def find_semantic_npy_paths(results_dir: Path) -> List[Path]:
    semantic_dir = results_dir / "semantic"
    paths = sorted(semantic_dir.glob("semantic_map*.npy"), key=natural_key)
    if not paths:
        paths = [
            path for path in sorted(semantic_dir.glob("semantic*.npy"), key=natural_key)
            if "obj" not in path.stem
        ]
    return paths


def find_rendered_semantic_paths(eval_dir: Path) -> List[Path]:
    return sorted((eval_dir / "rendered_semantic").glob("gs_*.png"), key=natural_key)


def build_index(paths: List[Path]) -> dict:
    return {frame_index(path): path for path in paths if frame_index(path) >= 0}


def semantic_num_classes(dataset: Optional[str]) -> int:
    if dataset and dataset.lower() == "mp3d":
        return 41
    return 102


def semantic_colormap(num_classes: int) -> np.ndarray:
    try:
        from imgviz import label_colormap
    except ModuleNotFoundError:
        raise RuntimeError(
            "imgviz is required to match ActiveMapping's semantic label_colormap."
        )

    colormap = label_colormap(num_classes)[:, :3]
    return colormap.astype(np.uint8)


def semantic_npy_to_bgr(path: Path, dataset: Optional[str]) -> np.ndarray:
    semantic = np.load(path)
    if semantic.ndim == 3 and semantic.shape[0] == 1:
        semantic = semantic[0]
    semantic = np.squeeze(semantic)
    if semantic.ndim != 2:
        raise ValueError(f"Semantic map must be 2D, got shape {semantic.shape}: {path}")

    num_classes = semantic_num_classes(dataset)
    semantic = semantic.astype(np.int64, copy=False)
    semantic = semantic.copy()
    semantic[(semantic < 0) | (semantic >= num_classes)] = 0
    semantic_rgb = semantic_colormap(num_classes)[semantic]
    return cv2.cvtColor(semantic_rgb, cv2.COLOR_RGB2BGR)


def resize_to_height(image: np.ndarray, height: int, interpolation: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == height:
        return image
    width = max(1, int(round(w * height / h)))
    return cv2.resize(image, (width, height), interpolation=interpolation)


def colorize_depth(depth_path: Path) -> np.ndarray:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth image: {depth_path}")
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    valid = depth > 0
    if np.any(valid):
        lo = float(np.percentile(depth[valid], 1))
        hi = float(np.percentile(depth[valid], 99))
        if hi <= lo:
            hi = float(depth[valid].max())
            lo = float(depth[valid].min())
    else:
        lo, hi = 0.0, 1.0

    if hi <= lo:
        hi = lo + 1.0
    normalized = np.clip((depth.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    depth_u8 = (normalized * 255).astype(np.uint8)
    depth_u8[~valid] = 0
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)


def make_missing_panel(size: Tuple[int, int], text: str) -> np.ndarray:
    width, height = size
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(panel, text, (20, max(40, height // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
    return panel


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 34), (0, 0, 0), thickness=-1)
    cv2.putText(labeled, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return labeled


def read_or_missing(path: Optional[Path], size: Tuple[int, int], text: str) -> np.ndarray:
    if path is None:
        return make_missing_panel(size, text)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return make_missing_panel(size, text)
    return image


def read_semantic_panel(
    idx: int,
    semantic_rgb_by_idx: Dict[int, Path],
    semantic_npy_by_idx: Dict[int, Path],
    size: Tuple[int, int],
    dataset: Optional[str],
) -> np.ndarray:
    semantic = read_semantic_image(idx, semantic_rgb_by_idx, semantic_npy_by_idx, dataset)
    if semantic is not None:
        return semantic

    return make_missing_panel(size, "missing semantic")


def read_semantic_image(
    idx: int,
    semantic_rgb_by_idx: Dict[int, Path],
    semantic_npy_by_idx: Dict[int, Path],
    dataset: Optional[str],
) -> Optional[np.ndarray]:
    # Prefer label maps so comparison plots use the same label_colormap as ActiveMapping eval.
    semantic_npy = semantic_npy_by_idx.get(idx)
    if semantic_npy is not None:
        try:
            return semantic_npy_to_bgr(semantic_npy, dataset)
        except Exception as exc:
            print(f"[warn] Failed to colorize semantic map {semantic_npy}: {exc}", flush=True)

    semantic_path = semantic_rgb_by_idx.get(idx)
    if semantic_path is not None:
        semantic = cv2.imread(str(semantic_path), cv2.IMREAD_COLOR)
        if semantic is not None:
            return semantic

    return None


def plot_dirs_for_runs(runs: List[Tuple[str, str, str]], eval_suffix: str, plot_dir: str = None) -> List[Path]:
    if plot_dir:
        return [Path(plot_dir).expanduser()]

    plot_dirs = []
    seen = set()
    for _, _, result_dir in runs:
        output_dir = eval_dir_for(result_dir, eval_suffix) / "trajectory_rgbd_semantic"
        key = str(output_dir.resolve() if output_dir.is_absolute() else output_dir)
        if key not in seen:
            plot_dirs.append(output_dir)
            seen.add(key)
    return plot_dirs


def save_eval_data_plots(eval_data_dir: Path, output_dir: Path, stride: int, dataset: Optional[str]) -> None:
    if stride < 1:
        raise ValueError(f"--plot_stride must be >= 1, got: {stride}")

    results_dir = eval_data_dir / "results_habitat"
    rgb_paths = find_rgb_paths(results_dir)
    depth_by_idx = build_index(find_depth_paths(results_dir))
    semantic_by_idx = build_index(find_semantic_rgb_paths(results_dir))
    semantic_npy_by_idx = build_index(find_semantic_npy_paths(results_dir))
    if not rgb_paths:
        print(f"[warn] Skip trajectory plotting: no RGB frames in {results_dir}", flush=True)
        return

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    overview_tiles = []
    sample_interval = max(1, len(rgb_paths) // 12)
    saved_count = 0

    for rgb_path in rgb_paths[::stride]:
        idx = frame_index(rgb_path)
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            print(f"[warn] Failed to read RGB frame: {rgb_path}", flush=True)
            continue

        depth_path = depth_by_idx.get(idx)
        if depth_path is not None:
            depth = colorize_depth(depth_path)
        else:
            depth = make_missing_panel((rgb.shape[1], rgb.shape[0]), "missing depth")

        semantic = read_semantic_panel(
            idx,
            semantic_by_idx,
            semantic_npy_by_idx,
            (rgb.shape[1], rgb.shape[0]),
            dataset,
        )

        target_h = 256
        rgb_small = resize_to_height(rgb, target_h, cv2.INTER_AREA)
        depth_small = resize_to_height(depth, target_h, cv2.INTER_NEAREST)
        semantic_small = resize_to_height(semantic, target_h, cv2.INTER_NEAREST)
        target_w = rgb_small.shape[1]
        depth_small = cv2.resize(depth_small, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        semantic_small = cv2.resize(semantic_small, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        panel = np.hstack(
            [
                add_label(rgb_small, f"RGB {idx:04d}"),
                add_label(depth_small, "Depth"),
                add_label(semantic_small, "Semantic"),
            ]
        )
        cv2.imwrite(str(frames_dir / f"frame_{idx:04d}.png"), panel)
        saved_count += 1

        if len(overview_tiles) < 12 and (saved_count == 1 or saved_count % sample_interval == 0):
            overview_tiles.append(cv2.resize(panel, (panel.shape[1] // 2, panel.shape[0] // 2), interpolation=cv2.INTER_AREA))

    if overview_tiles:
        width = max(tile.shape[1] for tile in overview_tiles)
        padded = []
        for tile in overview_tiles:
            if tile.shape[1] < width:
                pad = np.zeros((tile.shape[0], width - tile.shape[1], 3), dtype=np.uint8)
                tile = np.hstack([tile, pad])
            padded.append(tile)
        cv2.imwrite(str(output_dir / "overview.png"), np.vstack(padded))

    print(f"Saved {saved_count} trajectory RGBD/semantic plots to {frames_dir}", flush=True)


def find_eval_output_frame_indices(eval_dir: Path, stride: int) -> List[int]:
    for relative_dir in ("plots", "rendered_rgb", "rendered_depth", "rendered_semantic"):
        image_paths = sorted((eval_dir / relative_dir).glob("*.png"), key=natural_key)
        frame_indices = sorted(build_index(image_paths).keys())
        if frame_indices:
            return frame_indices[::stride]
    return []


def save_gt_semantic_plots(
    eval_data_dir: Path,
    runs: List[Tuple[str, str, str]],
    eval_suffix: str,
    stride: int,
    dataset: Optional[str],
) -> None:
    if stride < 1:
        raise ValueError(f"--plot_stride must be >= 1, got: {stride}")

    results_dir = eval_data_dir / "results_habitat"
    rgb_by_idx = build_index(find_rgb_paths(results_dir))
    semantic_rgb_by_idx = build_index(find_semantic_rgb_paths(results_dir))
    semantic_npy_by_idx = build_index(find_semantic_npy_paths(results_dir))
    semantic_indices = sorted(set(semantic_rgb_by_idx.keys()) | set(semantic_npy_by_idx.keys()))
    if not semantic_indices:
        print(f"[warn] Skip semantic_plots GT export: no semantic maps in {results_dir / 'semantic'}", flush=True)
        return

    fallback_indices = sorted((set(rgb_by_idx.keys()) & set(semantic_indices)) or set(semantic_indices))

    for label, _, result_dir in runs:
        eval_dir = eval_dir_for(result_dir, eval_suffix)
        frame_indices = find_eval_output_frame_indices(eval_dir, stride)
        if frame_indices:
            frame_indices = [idx for idx in frame_indices if idx in semantic_indices]
        else:
            frame_indices = fallback_indices[::stride]

        if not frame_indices:
            print(f"[warn] Skip semantic_plots GT export for {label}: no matching semantic frames", flush=True)
            continue

        plot_dir = eval_dir / "semantic_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        saved_count = 0

        for idx in frame_indices:
            semantic = read_semantic_image(idx, semantic_rgb_by_idx, semantic_npy_by_idx, dataset)
            if semantic is None:
                continue

            rgb_path = rgb_by_idx.get(idx)
            if rgb_path is not None:
                rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                if rgb is not None and semantic.shape[:2] != rgb.shape[:2]:
                    semantic = cv2.resize(semantic, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

            cv2.imwrite(str(plot_dir / f"{idx:04d}_gt.png"), semantic)
            saved_count += 1

        print(f"Saved {saved_count} GT semantic plots for {label} to {plot_dir}", flush=True)


def save_rendered_semantic_plots(
    eval_data_dir: Path,
    runs: List[Tuple[str, str, str]],
    eval_suffix: str,
    output_dir: Path,
    stride: int,
    dataset: Optional[str],
) -> None:
    if stride < 1:
        raise ValueError(f"--plot_stride must be >= 1, got: {stride}")

    rendered_by_run: List[Tuple[str, Dict[int, Path]]] = []
    for label, _, result_dir in runs:
        eval_dir = eval_dir_for(result_dir, eval_suffix)
        rendered_paths = find_rendered_semantic_paths(eval_dir)
        if not rendered_paths:
            print(f"[warn] No rendered semantic frames for {label}: {eval_dir / 'rendered_semantic'}", flush=True)
        rendered_by_run.append((label, build_index(rendered_paths)))

    frame_indices = sorted({idx for _, paths in rendered_by_run for idx in paths.keys()})
    if not frame_indices:
        print("[warn] Skip rendered semantic comparison: no rendered semantic frames were found.", flush=True)
        return

    results_dir = eval_data_dir / "results_habitat"
    rgb_by_idx = build_index(find_rgb_paths(results_dir))
    semantic_rgb_by_idx = build_index(find_semantic_rgb_paths(results_dir))
    semantic_npy_by_idx = build_index(find_semantic_npy_paths(results_dir))

    frames_dir = output_dir / "rendered_semantic_comparison" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    overview_tiles = []
    sample_interval = max(1, len(frame_indices) // 12)
    saved_count = 0

    for idx in frame_indices[::stride]:
        reference = None
        for _, rendered in rendered_by_run:
            rendered_path = rendered.get(idx)
            if rendered_path is None:
                continue
            image = cv2.imread(str(rendered_path), cv2.IMREAD_COLOR)
            if image is not None:
                reference = image
                break
        if reference is None:
            reference = make_missing_panel((1200, 680), "missing render")

        size = (reference.shape[1], reference.shape[0])
        rgb = read_or_missing(rgb_by_idx.get(idx), size, f"missing RGB {idx:04d}")
        semantic = read_semantic_panel(idx, semantic_rgb_by_idx, semantic_npy_by_idx, size, dataset)

        target_h = 256
        panels = [
            add_label(resize_to_height(rgb, target_h, cv2.INTER_AREA), f"GT RGB {idx:04d}"),
            add_label(resize_to_height(semantic, target_h, cv2.INTER_NEAREST), "GT Semantic"),
        ]
        target_w = panels[0].shape[1]
        panels[1] = cv2.resize(panels[1], (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        for label, rendered in rendered_by_run:
            render = read_or_missing(rendered.get(idx), size, f"missing {label}")
            render_small = resize_to_height(render, target_h, cv2.INTER_NEAREST)
            render_small = cv2.resize(render_small, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            panels.append(add_label(render_small, f"{label} Semantic"))

        panel = np.hstack(panels)
        cv2.imwrite(str(frames_dir / f"frame_{idx:04d}.png"), panel)
        saved_count += 1

        if len(overview_tiles) < 12 and (saved_count == 1 or saved_count % sample_interval == 0):
            overview_tiles.append(cv2.resize(panel, (panel.shape[1] // 2, panel.shape[0] // 2), interpolation=cv2.INTER_AREA))

    overview_dir = frames_dir.parent
    if overview_tiles:
        width = max(tile.shape[1] for tile in overview_tiles)
        padded = []
        for tile in overview_tiles:
            if tile.shape[1] < width:
                pad = np.zeros((tile.shape[0], width - tile.shape[1], 3), dtype=np.uint8)
                tile = np.hstack([tile, pad])
            padded.append(tile)
        cv2.imwrite(str(overview_dir / "overview.png"), np.vstack(padded))

    print(f"Saved {saved_count} rendered semantic comparison plots to {frames_dir}", flush=True)


def main() -> None:
    args = parse_args()
    repo_dir = Path(__file__).resolve().parents[2]
    eval_data_dir = Path(args.eval_data_dir).expanduser()
    if not eval_data_dir.is_absolute():
        eval_data_dir = (Path.cwd() / eval_data_dir).resolve()
    if not (eval_data_dir / "traj.txt").exists():
        raise FileNotFoundError(f"Missing traj.txt in eval_data_dir: {eval_data_dir}")
    if not (eval_data_dir / "results_habitat").is_dir():
        raise FileNotFoundError(f"Missing results_habitat/ in eval_data_dir: {eval_data_dir}")

    if args.eval_suffix is not None:
        eval_suffix = args.eval_suffix
    elif args.step is None:
        eval_suffix = f"{args.stage}_traj"
    else:
        eval_suffix = f"{args.stage}_step_{args.step}_traj"

    runs = [parse_run(run) for run in args.run]

    plot_dirs = plot_dirs_for_runs(runs, eval_suffix, args.plot_dir)

    if args.plot_eval_data:
        ensure_cv2()
        for plot_dir in plot_dirs:
            if not plot_dir.is_absolute():
                plot_dir = (Path.cwd() / plot_dir).resolve()
            save_eval_data_plots(eval_data_dir, plot_dir, args.plot_stride, args.dataset)

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    for label, cfg, result_dir in runs:
        print(f"\n### Evaluating {label}", flush=True)
        common = [
            "--cfg", cfg,
            "--result_dir", result_dir,
            "--enable_vis", args.enable_vis,
            "--stage", args.stage,
            "--eval_data_dir", str(eval_data_dir),
            "--eval_suffix", eval_suffix,
        ]

        if args.metrics in {"rgb", "both"}:
            rgb_output = eval_dir_for(result_dir, eval_suffix) / "render_result.txt"
            if args.skip_existing and rgb_output.exists():
                print(f"Skip RGB eval for {label}: {rgb_output} exists", flush=True)
            else:
                if args.step is None:
                    script = "src/evaluation/eval_nvs_result.py"
                    cmd = [sys.executable, script, *common]
                else:
                    script = "src/evaluation/eval_rgb_by_step.py"
                    cmd = [sys.executable, script, *common, "--step", str(args.step)]
                run_command(cmd, env, repo_dir)

        if args.metrics in {"semantic", "both"}:
            semantic_output = eval_dir_for(result_dir, eval_suffix) / "semantic_result.txt"
            if args.skip_existing and semantic_output.exists():
                print(f"Skip semantic eval for {label}: {semantic_output} exists", flush=True)
            else:
                semantic_step = 0 if args.step is None else args.step
                cmd = [
                    sys.executable,
                    "src/evaluation/eval_semantic.py",
                    *common,
                    "--step",
                    str(semantic_step),
                ]
                run_command(cmd, env, repo_dir)

    summary_cmd = [
        sys.executable,
        "src/evaluation/summarize_eval_results.py",
        "--eval_suffix",
        eval_suffix,
    ]
    for label, _, result_dir in runs:
        summary_cmd.extend(["--run", f"{label}={result_dir}"])
    if args.summary_csv:
        summary_cmd.extend(["--out_csv", args.summary_csv])
    run_command(summary_cmd, env, repo_dir)

    if args.plot_eval_data:
        ensure_cv2()
        save_gt_semantic_plots(eval_data_dir, runs, eval_suffix, args.plot_stride, args.dataset)
        for plot_dir in plot_dirs:
            if not plot_dir.is_absolute():
                plot_dir = (Path.cwd() / plot_dir).resolve()
            save_rendered_semantic_plots(eval_data_dir, runs, eval_suffix, plot_dir, args.plot_stride, args.dataset)


if __name__ == "__main__":
    main()
