"""
MIT License

Copyright (c) 2024 OPPO

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw


sys.path.append(os.getcwd())
from src.visualization.render_rgb_sem_entro_bev import (  # noqa: E402
    compute_bounds,
    convert_points_to_sim,
    gaussian_splat_features,
    load_npz_fields,
    project_points_to_bev,
    save_rgb,
    top_surface_indices,
    visible_opacity,
    visible_scales_px,
)


def argument_parsing() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render RGB BEV from params.npz and overlay trajectory."
    )
    parser.add_argument("--mesh_file", type=str, default="", help="deprecated; ignored")
    parser.add_argument("--traj_file", type=str, default="", help="params.npz file")
    parser.add_argument("--anchor_traj_file", type=str, default="", help="dataset traj.txt used for SLAM->sim alignment")
    parser.add_argument("--out_dir", type=str, default=None, help="output directory")
    parser.add_argument("--with_interact", type=int, default=0, help="deprecated; ignored")

    parser.add_argument("--dataset", type=str, default="Replica", help="dataset name")
    parser.add_argument("--scene", type=str, default="office3", help="scene name")
    parser.add_argument("--method", type=str, default="SemanticHeat", help="method name")
    parser.add_argument("--slam", type=str, default="splatam", help="SLAM folder name")
    parser.add_argument("--run_version", type=str, default="run_0", help="run folder name")
    parser.add_argument("--coord_system", choices=["sim", "slam"], default="sim", help="RGB BEV coordinate system")

    parser.add_argument("--res", type=float, default=0.01, help="meters per pixel")
    parser.add_argument("--pad", type=float, default=0.05, help="XY bound padding ratio")
    parser.add_argument("--remove_front_percent", type=float, default=0.5, help="near-side z percent to remove")
    parser.add_argument("--min_height", type=float, default=None, help="optional lower z bound")
    parser.add_argument("--max_height", type=float, default=None, help="optional upper z bound")
    parser.add_argument("--flip", action="store_true", help="use bottom-up surface instead of top-down")
    parser.add_argument("--min_sigma_px", type=float, default=0.75, help="minimum Gaussian sigma in pixels")
    parser.add_argument("--max_sigma_px", type=float, default=3.0, help="maximum Gaussian sigma in pixels")
    parser.add_argument("--max_kernel_radius", type=int, default=4, help="maximum Gaussian kernel radius")
    parser.add_argument(
        "--interpolation",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
        default="lanczos",
        help="image interpolation used when saving the BEV image",
    )
    parser.add_argument("--dpi", type=int, default=500, help="output DPI")

    parser.add_argument("--skip_step", type=int, default=5, help="trajectory pose stride")
    parser.add_argument("--line_width", type=int, default=3, help="trajectory line width in pixels")
    parser.add_argument("--draw_pose_points", action="store_true", help="draw pose samples on the trajectory")
    return parser.parse_args()


def load_trajectory_pose(line: str) -> np.ndarray:
    pose = np.array(list(map(float, line.split())), dtype=np.float32).reshape(4, 4)
    return pose


def load_first_pose(traj_txt: str) -> np.ndarray:
    with open(traj_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return load_trajectory_pose(line)
    raise ValueError(f"Empty trajectory file: {traj_txt}")


def load_cam_traj(params_path: str) -> dict:
    params = dict(np.load(params_path, allow_pickle=True))
    return {
        "w2cs": np.asarray(params["gt_w2c_all_frames"], dtype=np.float32),
        "intrinsic": np.asarray(params["intrinsics"], dtype=np.float32),
        "height": params["org_height"],
        "width": params["org_width"],
    }


def convert_rel2world(start_c2w_rel: np.ndarray, rel_c2w_slam: np.ndarray) -> np.ndarray:
    return start_c2w_rel @ rel_c2w_slam


def resolve_default_paths(args: argparse.Namespace) -> argparse.Namespace:
    home = "/media/phudh/X9_Pro/undergraduate"
    proj_dir = f"{home}/ActiveMapping"
    result_dir = f"{proj_dir}/results_all"
    gt_data_dir = f"{proj_dir}/data/{args.dataset}"

    if not args.traj_file:
        args.traj_file = (
            f"{result_dir}/{args.dataset}/{args.scene}/{args.method}/"
            f"{args.run_version}/{args.slam}/final/params.npz"
        )
    if not args.anchor_traj_file:
        args.anchor_traj_file = f"{gt_data_dir}/{args.scene}/traj.txt"
    if args.out_dir is None:
        args.out_dir = f"{result_dir}/{args.dataset}/TRAJ/"
    return args


def render_rgb_bev(
    params_file: str,
    dataset: str,
    coord_system: str,
    traj_path: str,
    res: float,
    pad: float,
    remove_front_percent: float,
    min_height,
    max_height,
    flip: bool,
    min_sigma_px: float,
    max_sigma_px: float,
    max_kernel_radius: int,
) -> tuple[np.ndarray, tuple, dict]:
    """RGB-only path matching render_rgb_sem_entro_bev.py."""
    fields = load_npz_fields(params_file)
    points_sim = convert_points_to_sim(
        fields["points"],
        dataset=dataset,
        coord_system=coord_system,
        traj_path=traj_path,
    )

    keep = np.ones(points_sim.shape[0], dtype=bool)
    front_cutoff = None
    view_nearness = points_sim[:, 2] if not flip else -points_sim[:, 2]
    if remove_front_percent is not None:
        if not (0.0 <= remove_front_percent < 100.0):
            raise ValueError("--remove_front_percent must be in [0, 100)")
        front_cutoff = float(np.percentile(view_nearness, 100.0 - float(remove_front_percent)))
        keep &= view_nearness <= front_cutoff
    if min_height is not None:
        keep &= points_sim[:, 2] >= float(min_height)
    if max_height is not None:
        keep &= points_sim[:, 2] <= float(max_height)

    points_sim = points_sim[keep]
    if points_sim.shape[0] == 0:
        raise ValueError("No RGB splat points remain after filtering.")

    colors = fields["colors"]
    if colors is None:
        raise ValueError("RGB colors are missing from params.npz")
    colors = colors[keep]

    scales = fields["scales"]
    if scales is not None:
        scales = scales[keep]
    opacities = fields["opacities"]
    if opacities is not None:
        opacities = opacities[keep]

    bounds = compute_bounds(points_sim[:, :2], res=res, pad=pad)
    px, py, _, _, linear = project_points_to_bev(points_sim, bounds=bounds, res=res)
    chosen_idx = top_surface_indices(linear, points_sim[:, 2], flip=flip)
    sigma_px = visible_scales_px(
        scales,
        chosen_idx,
        res=res,
        min_sigma_px=min_sigma_px,
        max_sigma_px=max_sigma_px,
    )
    alpha = visible_opacity(opacities, chosen_idx)

    rgb_img = gaussian_splat_features(
        colors[chosen_idx].astype(np.float32) / 255.0,
        px[chosen_idx],
        py[chosen_idx],
        sigma_px,
        alpha,
        bounds,
        max_kernel_radius=max_kernel_radius,
        background_value=1.0,
    )
    rgb_img = np.clip(rgb_img * 255.0, 0, 255).astype(np.uint8)

    stats = {
        "points_kept": int(points_sim.shape[0]),
        "visible_splats": int(chosen_idx.shape[0]),
        "front_cutoff": front_cutoff,
    }
    return rgb_img, bounds, stats


def load_world_trajectory_points(params_file: str, anchor_traj_file: str, skip_step: int) -> np.ndarray:
    start_c2w = load_first_pose(anchor_traj_file)
    camera_trajectory = load_cam_traj(params_file)
    stride = max(1, int(skip_step))
    w2c_subset = camera_trajectory["w2cs"][::stride]
    c2ws = [convert_rel2world(start_c2w, np.linalg.inv(w2c)) for w2c in w2c_subset]
    return np.asarray([c2w[:3, 3] for c2w in c2ws], dtype=np.float32)


def draw_trajectory_overlay(
    image: np.ndarray,
    trajectory_points: np.ndarray,
    bounds,
    res: float,
    line_width: int,
    draw_pose_points: bool,
) -> np.ndarray:
    _, _, _, _, _, height = bounds
    px, py, _, _, _ = project_points_to_bev(trajectory_points, bounds=bounds, res=res)
    pixels = np.stack([px, (height - 1) - py], axis=1)
    pixel_tuples = [tuple(np.rint(p).astype(int).tolist()) for p in pixels]

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    if len(pixel_tuples) >= 2:
        draw.line(pixel_tuples, fill=(255, 140, 0), width=max(1, int(line_width)))

    radius = max(3, int(line_width) + 2)
    if draw_pose_points:
        for point in pixel_tuples[1:-1]:
            draw.ellipse(
                (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
                outline=(0, 255, 0),
                width=max(1, int(line_width)),
            )
    if pixel_tuples:
        start = pixel_tuples[0]
        end = pixel_tuples[-1]
        draw.ellipse(
            (start[0] - radius, start[1] - radius, start[0] + radius, start[1] + radius),
            fill=(255, 0, 0),
        )
        draw.ellipse(
            (end[0] - radius, end[1] - radius, end[0] + radius, end[1] + radius),
            fill=(0, 0, 255),
        )
    return np.asarray(pil_image, dtype=np.uint8)


def main() -> None:
    args = resolve_default_paths(argument_parsing())
    if args.coord_system != "sim":
        raise ValueError("Trajectory overlay currently expects --coord_system sim for alignment.")

    os.makedirs(args.out_dir, exist_ok=True)
    rgb_img, bounds, stats = render_rgb_bev(
        params_file=args.traj_file,
        dataset=args.dataset,
        coord_system=args.coord_system,
        traj_path=args.anchor_traj_file,
        res=args.res,
        pad=args.pad,
        remove_front_percent=args.remove_front_percent,
        min_height=args.min_height,
        max_height=args.max_height,
        flip=args.flip,
        min_sigma_px=args.min_sigma_px,
        max_sigma_px=args.max_sigma_px,
        max_kernel_radius=args.max_kernel_radius,
    )
    trajectory_points = load_world_trajectory_points(
        args.traj_file,
        args.anchor_traj_file,
        skip_step=args.skip_step,
    )
    out_image = draw_trajectory_overlay(
        rgb_img,
        trajectory_points,
        bounds,
        res=args.res,
        line_width=args.line_width,
        draw_pose_points=args.draw_pose_points,
    )

    render_filepath = os.path.join(args.out_dir, f"vis_trajectory{args.scene}_{args.run_version}.png")
    save_rgb(
        render_filepath,
        out_image,
        dpi=args.dpi,
        interpolation=args.interpolation,
    )

    _, _, _, _, width, height = bounds
    print(f"Input npz        : {args.traj_file}")
    print(f"Trajectory pose  : {args.anchor_traj_file}")
    print(f"BEV size         : {height} x {width}")
    print(f"RGB splats kept  : {stats['points_kept']}")
    print(f"Visible splats   : {stats['visible_splats']}")
    if stats["front_cutoff"] is not None:
        print(f"Front cutoff     : {stats['front_cutoff']:.4f}")
    print(f"Trajectory poses : {trajectory_points.shape[0]}")
    print(f"Interpolation    : {args.interpolation}")
    print(f"DPI              : {args.dpi}")
    print(f"Saved trajectory : {render_filepath}")


if __name__ == "__main__":
    main()
