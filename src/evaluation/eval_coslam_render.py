import argparse
import glob
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(os.getcwd())

try:
    import cv2
except Exception:  # pragma: no cover - optional frame export dependency
    cv2 = None

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    import imageio

try:
    from pytorch_msssim import ms_ssim
except Exception:  # pragma: no cover - keep geometry eval usable if the package is missing
    ms_ssim = None

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
except Exception:  # pragma: no cover
    LearnedPerceptualImagePatchSimilarity = None

from src.naruto.cfg_loader import load_cfg
from src.simulator import init_simulator
from src.slam import init_SLAM_model
from src.utils.general_utils import InfoPrinter, fix_random_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate NARUTO/Co-SLAM rendering metrics on a shared eval trajectory or active trajectory."
    )
    parser.add_argument("--cfg", required=True, help="NARUTO config used by the run.")
    parser.add_argument("--result_dir", required=True, help="Run result directory, e.g. results/Replica/office0/NARUTO/run_0.")
    parser.add_argument("--ckpt", required=True, help="Co-SLAM checkpoint path, e.g. ckpt_2000_final.pt.")
    parser.add_argument("--eval_dir", default=None, help="Directory for psnr.txt, render_result.txt, and optional frames.")
    parser.add_argument("--result_txt", default=None, help="Optional aggregate eval_result.txt to append render metrics to.")
    parser.add_argument("--eval_every", type=int, default=5, help="Evaluate every N frames. Match SplaTAM config['eval_every'] for fair comparison.")
    parser.add_argument("--max_frames", type=int, default=0, help="Optional cap on evaluated frames; 0 means no cap.")
    parser.add_argument("--chunk_size", type=int, default=8192, help="Ray chunk size for rendering.")
    parser.add_argument("--skip_first", action="store_true", help="Skip the first eval frame when averaging metrics.")
    parser.add_argument("--save_frames", action="store_true", help="Save rendered and GT RGB/depth frames.")
    parser.add_argument("--disable_lpips", action="store_true", help="Skip LPIPS calculation.")
    parser.add_argument("--lpips_downsample", type=int, default=1, help="Downsample factor for LPIPS only.")
    parser.add_argument("--device", default="cuda", help="Torch device for Co-SLAM rendering.")
    parser.add_argument("--eval_data_dir", default=None, help="Shared eval folder containing traj.txt and results_habitat/.")
    parser.add_argument("--eval_data_basedir", default=None, help="Shared eval basedir; used with --eval_sequence.")
    parser.add_argument("--eval_sequence", default=None, help="Sequence name under --eval_data_basedir.")
    parser.add_argument(
        "--eval_pose_mode",
        choices=["absolute", "relative"],
        default="absolute",
        help="absolute uses poses from traj.txt directly; relative mimics GradSLAMDataset relative_pose=True.",
    )
    return parser.parse_args()


def make_cfg_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        cfg=args.cfg,
        setting=None,
        result_dir=args.result_dir,
        seed=None,
        enable_vis=0,
        stage="final",
        eval_data_dir=args.eval_data_dir,
        eval_data_basedir=args.eval_data_basedir,
        eval_sequence=args.eval_sequence,
        eval_suffix=None,
    )


def prepare_cfg(args: argparse.Namespace):
    cfg = load_cfg(make_cfg_args(args))
    cfg.dirs.result_dir = args.result_dir
    cfg.visualizer.vis_rgbd = False
    if hasattr(cfg.visualizer, "enable_all_vis"):
        cfg.visualizer.enable_all_vis = False
    cfg.slam.enable_active_planning = False
    cfg.slam.enable_active_ray = False
    cfg.slam.active_ray = False
    return cfg


def load_checkpoint(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu")


def natural_key(path: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(path))]


def resolve_eval_data_dir(args: argparse.Namespace, main_cfg) -> Optional[Path]:
    if args.eval_data_dir:
        return Path(args.eval_data_dir)
    if args.eval_data_basedir and args.eval_sequence:
        return Path(args.eval_data_basedir) / args.eval_sequence
    basedir = main_cfg.slam.get("dataset_eval_basedir", None)
    sequence = main_cfg.slam.get("dataset_eval_sequence", None)
    if basedir and sequence:
        return Path(basedir) / sequence
    return None


def load_poses(path: Path, pose_mode: str) -> List[torch.Tensor]:
    poses = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            values = np.asarray([float(token) for token in line.split()], dtype=np.float32)
            if values.size != 16:
                raise ValueError(f"{path}:{line_no} has {values.size} values; expected 16 for one 4x4 pose.")
            poses.append(torch.from_numpy(values.reshape(4, 4)).float())

    if not poses:
        raise ValueError(f"No poses found in {path}")

    if pose_mode == "relative":
        first_inv = torch.linalg.inv(poses[0])
        poses = [first_inv @ pose for pose in poses]
    return poses


def read_rgb(path: Path) -> torch.Tensor:
    color = np.asarray(imageio.imread(path), dtype=np.float32)
    if color.ndim == 2:
        color = np.repeat(color[..., None], 3, axis=-1)
    if color.shape[-1] > 3:
        color = color[..., :3]
    return torch.from_numpy(color / 255.0).float()


def read_depth(path: Path, png_depth_scale: float) -> torch.Tensor:
    if path.suffix.lower() == ".exr":
        raise NotImplementedError("EXR depth is not supported by this evaluator yet.")
    depth = np.asarray(imageio.imread(path), dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return torch.from_numpy(depth / float(png_depth_scale)).float()


def resize_color_depth(color: torch.Tensor, depth: torch.Tensor, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if tuple(color.shape[:2]) == (height, width) and tuple(depth.shape[:2]) == (height, width):
        return color, depth

    color_chw = color.permute(2, 0, 1).unsqueeze(0).float()
    color = F.interpolate(color_chw, size=(height, width), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)

    depth_nchw = depth.unsqueeze(0).unsqueeze(0).float()
    depth = F.interpolate(depth_nchw, size=(height, width), mode="nearest")[0, 0]
    return color, depth


def load_eval_data_frames(
    eval_data_dir: Path,
    height: int,
    width: int,
    png_depth_scale: float,
    pose_mode: str,
) -> List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    eval_data_dir = eval_data_dir.expanduser()
    if not eval_data_dir.is_absolute():
        eval_data_dir = Path.cwd() / eval_data_dir
    traj_path = eval_data_dir / "traj.txt"
    results_dir = eval_data_dir / "results_habitat"
    if not traj_path.is_file():
        raise FileNotFoundError(f"Missing traj.txt in eval_data_dir: {eval_data_dir}")
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Missing results_habitat/ in eval_data_dir: {eval_data_dir}")

    color_paths = sorted(glob.glob(str(results_dir / "frame*.jpg")) + glob.glob(str(results_dir / "frame*.png")), key=natural_key)
    depth_paths = sorted(glob.glob(str(results_dir / "depth*.png")), key=natural_key)
    if len(color_paths) != len(depth_paths):
        raise ValueError(f"Color/depth count mismatch in {results_dir}: {len(color_paths)} vs {len(depth_paths)}")
    if not color_paths:
        raise ValueError(f"No frame*.jpg/png files found in {results_dir}")

    poses = load_poses(traj_path, pose_mode)
    if len(poses) < len(color_paths):
        raise ValueError(f"Pose count ({len(poses)}) is smaller than image count ({len(color_paths)}) in {eval_data_dir}")

    frames = []
    for idx, (color_path, depth_path) in enumerate(zip(color_paths, depth_paths)):
        color = read_rgb(Path(color_path))
        depth = read_depth(Path(depth_path), png_depth_scale)
        color, depth = resize_color_depth(color, depth, height, width)
        frames.append((idx, poses[idx], color, depth))
    return frames


def sorted_pose_items(poses: Dict) -> List[Tuple[int, torch.Tensor]]:
    items = []
    for key, pose in poses.items():
        try:
            frame_id = int(key)
        except Exception:
            frame_id = key
        pose_t = pose.detach().cpu().float() if torch.is_tensor(pose) else torch.as_tensor(pose, dtype=torch.float32)
        items.append((frame_id, pose_t))
    return sorted(items, key=lambda item: item[0])


def select_items(items, eval_every: int, max_frames: int, skip_first: bool):
    eval_every = max(1, int(eval_every))
    selected = []
    for idx, item in enumerate(items):
        if idx != 0 and (idx + 1) % eval_every != 0:
            continue
        if skip_first and idx == 0:
            continue
        selected.append(item)
        if max_frames > 0 and len(selected) >= max_frames:
            break
    return selected


def render_image(slam, c2w: torch.Tensor, target_depth: torch.Tensor, chunk_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    device = slam.device
    height, width = slam.H, slam.W
    rays_d_cam = slam.rays_d.reshape(-1, 3).to(device)
    target_depth_flat = target_depth.reshape(-1, 1).to(device)
    c2w = c2w.to(device)

    rgb_chunks = []
    depth_chunks = []
    chunk_size = max(1, int(chunk_size))

    with torch.no_grad():
        for start in range(0, rays_d_cam.shape[0], chunk_size):
            end = min(start + chunk_size, rays_d_cam.shape[0])
            rays_d_cam_chunk = rays_d_cam[start:end]
            rays_o = c2w[:3, 3].reshape(1, 3).expand(rays_d_cam_chunk.shape[0], 3)
            rays_d = torch.sum(rays_d_cam_chunk[..., None, :] * c2w[:3, :3], -1)
            target_d = target_depth_flat[start:end]
            ret = slam.model.render_rays(rays_o, rays_d, target_d=target_d)
            rgb_chunks.append(ret["rgb"].detach().cpu())
            depth_chunks.append(ret["depth"].detach().cpu())

    rgb = torch.cat(rgb_chunks, dim=0).reshape(height, width, 3).clamp(0.0, 1.0)
    depth = torch.cat(depth_chunks, dim=0).reshape(height, width)
    return rgb, depth


def psnr_from_mse(mse: torch.Tensor) -> float:
    return float((-10.0 * torch.log10(mse.clamp_min(1e-10))).item())


def compute_metrics(
    pred_rgb: torch.Tensor,
    pred_depth: torch.Tensor,
    gt_rgb: torch.Tensor,
    gt_depth: torch.Tensor,
    depth_trunc: float,
    ssim_fn,
    lpips_fn,
    lpips_downsample: int,
) -> Dict[str, float]:
    valid = torch.isfinite(gt_depth) & (gt_depth > 0.0) & (gt_depth < depth_trunc)
    if valid.sum().item() == 0:
        raise ValueError("No valid depth pixels for metric calculation.")

    rgb_mse = torch.mean((pred_rgb[valid] - gt_rgb[valid]) ** 2)
    depth_diff = pred_depth[valid] - gt_depth[valid]

    metrics = {
        "psnr": psnr_from_mse(rgb_mse),
        "l1(cm)": float(torch.mean(torch.abs(depth_diff)).item() * 100.0),
        "rmse": float(torch.sqrt(torch.mean(depth_diff ** 2)).item()),
    }

    mask = valid.float().unsqueeze(0)
    pred_chw = pred_rgb.permute(2, 0, 1) * mask
    gt_chw = gt_rgb.permute(2, 0, 1) * mask

    if ssim_fn is not None:
        metrics["ssim"] = float(ssim_fn(pred_chw.unsqueeze(0), gt_chw.unsqueeze(0), data_range=1.0, size_average=True).item())
    else:
        metrics["ssim"] = math.nan

    if lpips_fn is not None:
        pred_lpips = pred_chw.unsqueeze(0).clamp(0.0, 1.0)
        gt_lpips = gt_chw.unsqueeze(0).clamp(0.0, 1.0)
        if lpips_downsample > 1:
            scale = 1.0 / float(lpips_downsample)
            pred_lpips = F.interpolate(pred_lpips, scale_factor=scale, mode="bilinear", align_corners=False)
            gt_lpips = F.interpolate(gt_lpips, scale_factor=scale, mode="bilinear", align_corners=False)
        lpips_device = next(lpips_fn.parameters()).device
        metrics["lpips"] = float(lpips_fn(pred_lpips.to(lpips_device), gt_lpips.to(lpips_device)).item())
    else:
        metrics["lpips"] = math.nan

    return metrics


def save_frame_pair(eval_dir: Path, frame_id: int, pred_rgb: torch.Tensor, pred_depth: torch.Tensor, gt_rgb: torch.Tensor, gt_depth: torch.Tensor) -> None:
    if cv2 is None:
        return
    for name in ["rendered_rgb", "rendered_depth", "rgb", "depth"]:
        (eval_dir / name).mkdir(parents=True, exist_ok=True)

    pred_rgb_np = (pred_rgb.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    gt_rgb_np = (gt_rgb.numpy() * 255.0).clip(0, 255).astype(np.uint8)

    def depth_color(depth: torch.Tensor) -> np.ndarray:
        arr = depth.numpy()
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr / 6.0, 0.0, 1.0)
        return cv2.applyColorMap((arr * 255.0).astype(np.uint8), cv2.COLORMAP_JET)

    cv2.imwrite(str(eval_dir / "rendered_rgb" / f"gs_{frame_id:04d}.png"), cv2.cvtColor(pred_rgb_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(eval_dir / "rgb" / f"gt_{frame_id:04d}.png"), cv2.cvtColor(gt_rgb_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(eval_dir / "rendered_depth" / f"gs_{frame_id:04d}.png"), depth_color(pred_depth))
    cv2.imwrite(str(eval_dir / "depth" / f"gt_{frame_id:04d}.png"), depth_color(gt_depth))


def write_metric_outputs(eval_dir: Path, frame_ids: List[int], metric_lists: Dict[str, List[float]], result_txt: Optional[str]) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    for key, values in metric_lists.items():
        filename = "l1.txt" if key == "l1(cm)" else f"{key}.txt"
        np.savetxt(eval_dir / filename, np.asarray(values, dtype=np.float64))

    averages = {}
    for key, values in metric_lists.items():
        arr = np.asarray(values, dtype=np.float64)
        averages[key] = math.nan if arr.size == 0 or np.isnan(arr).all() else float(np.nanmean(arr))
    np.savetxt(eval_dir / "frame_ids.txt", np.asarray(frame_ids, dtype=np.int64), fmt="%d")

    with (eval_dir / "render_result.txt").open("w", encoding="utf-8") as f:
        for key in ["psnr", "ssim", "lpips", "l1(cm)", "rmse"]:
            f.write(f"{key}: {averages.get(key, math.nan)}\n")
        f.write(f"num_frames: {len(frame_ids)}\n")

    if result_txt:
        Path(result_txt).parent.mkdir(parents=True, exist_ok=True)
        with open(result_txt, "a", encoding="utf-8") as f:
            for key in ["psnr", "ssim", "lpips", "l1(cm)", "rmse"]:
                f.write(f"{key},{averages.get(key, math.nan)}\n")
            f.write(f"render_num_frames,{len(frame_ids)}\n")


def build_lpips(disable_lpips: bool, device: str):
    if disable_lpips or LearnedPerceptualImagePatchSimilarity is None:
        return None
    try:
        return LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device).eval()
    except Exception as exc:
        print(f"[WARN] LPIPS is disabled because initialization failed: {exc}")
        return None


def main() -> None:
    args = parse_args()
    if args.eval_every < 1:
        raise ValueError("--eval_every must be >= 1")

    eval_dir = Path(args.eval_dir) if args.eval_dir else Path(args.result_dir) / "coslam" / "eval_final"
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    main_cfg = prepare_cfg(args)
    info_printer = InfoPrinter("NARUTO-Eval")
    fix_random_seed(int(main_cfg.general.get("seed", 0)))
    info_printer.update_scene(main_cfg.general.dataset + " - " + main_cfg.general.scene)

    print(f"==> Loading checkpoint: {args.ckpt}")
    ckpt = load_checkpoint(args.ckpt)

    print("==> Initializing Co-SLAM model")
    slam = init_SLAM_model(main_cfg, info_printer)
    slam.device = torch.device(device)
    slam.model.load_state_dict(ckpt["model"])
    slam.model.to(slam.device).eval()
    slam.config["training"]["perturb"] = 0.0

    eval_data_dir = resolve_eval_data_dir(args, main_cfg)
    selected_eval_frames = None
    sim = None
    if eval_data_dir is not None:
        print(f"==> Loading shared eval trajectory: {eval_data_dir} (pose_mode={args.eval_pose_mode})")
        frames = load_eval_data_frames(
            eval_data_dir,
            slam.H,
            slam.W,
            float(slam.config["cam"]["png_depth_scale"]),
            args.eval_pose_mode,
        )
        selected_eval_frames = select_items(frames, args.eval_every, args.max_frames, args.skip_first)
        source_msg = "shared eval trajectory"
    else:
        print("==> No shared eval trajectory was provided; falling back to active checkpoint poses and simulator GT")
        pose_items = sorted_pose_items(ckpt["pose"])
        selected_eval_frames = select_items(pose_items, args.eval_every, args.max_frames, args.skip_first)
        print("==> Initializing simulator")
        sim = init_simulator(main_cfg, info_printer)
        source_msg = "checkpoint active trajectory"

    if not selected_eval_frames:
        raise RuntimeError("No frames selected for render evaluation.")
    info_printer.update_total_step(len(selected_eval_frames))

    lpips_fn = build_lpips(args.disable_lpips, device)
    metric_lists = {"psnr": [], "ssim": [], "lpips": [], "l1(cm)": [], "rmse": []}
    frame_ids = []
    depth_trunc = float(slam.config["cam"].get("depth_trunc", 100.0))

    print(
        f"==> Evaluating {len(selected_eval_frames)} frames from {source_msg} "
        f"(eval_every={args.eval_every}, chunk_size={args.chunk_size})"
    )
    for item in tqdm(selected_eval_frames, desc="Render eval"):
        if eval_data_dir is not None:
            frame_id, c2w_cpu, color, depth = item
        else:
            frame_id, c2w_cpu = item
            color, depth = sim.simulate(c2w_cpu.numpy(), no_print=True)
            color, depth = resize_color_depth(color.float(), depth.float(), slam.H, slam.W)

        pred_rgb, pred_depth = render_image(slam, c2w_cpu, depth, args.chunk_size)
        metrics = compute_metrics(
            pred_rgb,
            pred_depth,
            color,
            depth,
            depth_trunc,
            ms_ssim,
            lpips_fn,
            max(1, args.lpips_downsample),
        )

        for key, value in metrics.items():
            metric_lists[key].append(value)
        frame_ids.append(int(frame_id))

        if args.save_frames:
            save_frame_pair(eval_dir, int(frame_id), pred_rgb, pred_depth, color, depth)

        del pred_rgb, pred_depth, color, depth
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_metric_outputs(eval_dir, frame_ids, metric_lists, args.result_txt)
    print(f"==> Render metrics saved to {eval_dir / 'render_result.txt'}")


if __name__ == "__main__":
    main()
