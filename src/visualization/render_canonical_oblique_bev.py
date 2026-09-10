"""
Render canonical top-down and four-direction oblique Gaussian-map views.

Each oblique view uses a slightly tilted orthographic camera after canonical
scene alignment. RGB output contains a bright-green exploration trajectory and
sampled camera frustums. Semantic colors exactly follow the evaluation palette,
and every PNG uses splat coverage as a transparent alpha channel.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import typing

import numpy as np
from PIL import Image, ImageDraw


if __package__ in (None, ""):
    import sys

    sys.path.append(os.getcwd())

from src.visualization.render_rgb_sem_entro_bev import (  # noqa: E402
    Bounds,
    compute_bounds,
    convert_points_to_sim,
    eval_label_colormap,
    gaussian_splat_features,
    load_first_traj_pose,
    load_npz_fields,
    project_points_to_bev,
    top_surface_indices,
    visible_opacity,
    visible_scales_px,
)


RGB = typing.Tuple[int, int, int]


def rotation_z(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _deterministic_sample(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    stride = max(1, points.shape[0] // max_points)
    return points[::stride][:max_points]


def estimate_canonical_transform(
    points: np.ndarray,
    camera_c2ws: np.ndarray,
    labels: typing.Optional[np.ndarray] = None,
    wall_label: int = 93,
    max_sample_points: int = 120000,
    angle_step_deg: float = 0.5,
) -> typing.Tuple[np.ndarray, dict]:
    """Align the dominant floor-plan axis to +X and place the start near -Y."""
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("points must have shape (N, 3)")

    all_xy = np.asarray(points[:, :2], dtype=np.float32)
    low, high = np.percentile(all_xy, [0.5, 99.5], axis=0)
    center_xy = ((low + high) * 0.5).astype(np.float32)

    source_xy = all_xy
    source_name = "all_non_ceiling_points"
    if labels is not None and labels.shape[0] == points.shape[0]:
        wall_mask = np.asarray(labels == int(wall_label))
        if int(np.count_nonzero(wall_mask)) >= 1000:
            source_xy = all_xy[wall_mask]
            source_name = f"semantic_wall_{wall_label}"

    source_xy = _deterministic_sample(source_xy, max_sample_points)
    src_low, src_high = np.percentile(source_xy, [0.5, 99.5], axis=0)
    inlier_mask = np.all((source_xy >= src_low) & (source_xy <= src_high), axis=1)
    source_xy = source_xy[inlier_mask] - center_xy[None]

    if source_xy.shape[0] < 3:
        angle_rad = 0.0
    else:
        best_area = float("inf")
        angle_rad = 0.0
        for angle_deg in np.arange(0.0, 90.0, float(angle_step_deg)):
            candidate = math.radians(float(angle_deg))
            c = math.cos(candidate)
            s = math.sin(candidate)
            x_rot = c * source_xy[:, 0] - s * source_xy[:, 1]
            y_rot = s * source_xy[:, 0] + c * source_xy[:, 1]
            width = float(x_rot.max() - x_rot.min())
            height = float(y_rot.max() - y_rot.min())
            area = max(width, 1e-6) * max(height, 1e-6)
            if area < best_area:
                best_area = area
                angle_rad = candidate

    centered_xy = all_xy - center_xy[None]
    initial_rotation = rotation_z(angle_rad)
    rotated_xy = (initial_rotation[:2, :2] @ centered_xy.T).T
    rot_low, rot_high = np.percentile(rotated_xy, [0.5, 99.5], axis=0)
    extents = rot_high - rot_low
    if extents[1] > extents[0]:
        angle_rad += math.pi * 0.5

    rotation = rotation_z(angle_rad)
    start_xy = camera_c2ws[0, :2, 3] - center_xy
    start_rotated = rotation[:2, :2] @ start_xy
    first_forward = -camera_c2ws[0, :3, 2]
    forward_rotated = rotation @ first_forward

    # Resolve the 180-degree ambiguity consistently across scenes.
    if start_rotated[1] > 0.05 or (
        abs(float(start_rotated[1])) <= 0.05 and forward_rotated[1] < 0.0
    ):
        angle_rad += math.pi
        rotation = rotation_z(angle_rad)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    center_world = np.asarray([center_xy[0], center_xy[1], 0.0], dtype=np.float32)
    transform[:3, 3] = -(rotation @ center_world)

    final_xy = (rotation[:2, :2] @ centered_xy.T).T
    final_low, final_high = np.percentile(final_xy, [0.5, 99.5], axis=0)
    metadata = {
        "rotation_degrees": float(math.degrees(angle_rad) % 360.0),
        "center_xy_world": [float(center_xy[0]), float(center_xy[1])],
        "alignment_source": source_name,
        "robust_xy_extent": [
            float(final_high[0] - final_low[0]),
            float(final_high[1] - final_low[1]),
        ],
    }
    return transform, metadata


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    points_h = np.concatenate([np.asarray(points, dtype=np.float32), ones], axis=1)
    return (transform @ points_h.T).T[:, :3].astype(np.float32)


def transform_camera_poses(c2ws: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.einsum("ij,njk->nik", transform, c2ws).astype(np.float32)


def load_world_camera_poses(params_path: str, anchor_traj_path: str) -> np.ndarray:
    start_c2w = load_first_traj_pose(anchor_traj_path)
    with np.load(params_path, allow_pickle=True) as data:
        if "gt_w2c_all_frames" not in data:
            raise ValueError("params.npz does not contain gt_w2c_all_frames")
        w2cs = np.asarray(data["gt_w2c_all_frames"], dtype=np.float32)
    if w2cs.ndim != 3 or w2cs.shape[1:] != (4, 4):
        raise ValueError(f"Unexpected gt_w2c_all_frames shape: {w2cs.shape}")
    relative_c2ws = np.linalg.inv(w2cs)
    return np.einsum("ij,njk->nik", start_c2w, relative_c2ws).astype(np.float32)


def project_fixed_view(
    points: np.ndarray,
    view_mode: str,
    elevation_deg: float,
    azimuth_deg: float = 0.0,
) -> np.ndarray:
    """Return projected (screen_x, screen_y, camera_nearness)."""
    points = np.asarray(points, dtype=np.float32)
    if view_mode == "topdown":
        return points[:, [0, 1, 2]].astype(np.float32)
    if view_mode != "oblique":
        raise ValueError(f"Unsupported view mode: {view_mode}")

    elevation = math.radians(float(elevation_deg))
    sin_e = math.sin(elevation)
    cos_e = math.cos(elevation)
    azimuth = math.radians(float(azimuth_deg))
    cos_a = math.cos(azimuth)
    sin_a = math.sin(azimuth)

    view_x = cos_a * points[:, 0] + sin_a * points[:, 1]
    view_y = -sin_a * points[:, 0] + cos_a * points[:, 1]
    screen_x = view_x
    screen_y = sin_e * view_y + cos_e * points[:, 2]
    nearness = -cos_e * view_y + sin_e * points[:, 2]
    return np.stack([screen_x, screen_y, nearness], axis=1).astype(np.float32)


def gaussian_splat_labels(
    labels: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    sigma_px: np.ndarray,
    alpha: np.ndarray,
    bounds: Bounds,
    palette: np.ndarray,
    max_kernel_radius: int,
    background: RGB = (0, 0, 0),
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Rasterize exact label colors and a smooth splat coverage channel."""
    _, _, _, _, width, height = bounds
    num_pixels = int(width * height)
    best_weight = np.zeros(num_pixels, dtype=np.float32)
    label_image = np.full(num_pixels, -1, dtype=np.int32)

    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    base_x = np.floor(px).astype(np.int32)
    base_y = np.floor(py).astype(np.int32)
    radius_px = np.clip(
        np.ceil(3.0 * sigma_px).astype(np.int32),
        1,
        int(max_kernel_radius),
    )
    inv_two_sigma_sq = 0.5 / np.maximum(sigma_px * sigma_px, 1e-8)

    for dy in range(-max_kernel_radius, max_kernel_radius + 1):
        gy = base_y + dy
        valid_y = (gy >= 0) & (gy < height) & (abs(dy) <= radius_px)
        if not np.any(valid_y):
            continue
        dy_center = (gy.astype(np.float32) + 0.5) - py

        for dx in range(-max_kernel_radius, max_kernel_radius + 1):
            gx = base_x + dx
            valid = (
                valid_y
                & (gx >= 0)
                & (gx < width)
                & (abs(dx) <= radius_px)
            )
            if not np.any(valid):
                continue

            valid_idx = np.flatnonzero(valid)
            delta_x = (gx[valid].astype(np.float32) + 0.5) - px[valid]
            delta_y = dy_center[valid]
            weights = alpha[valid] * np.exp(
                -(delta_x * delta_x + delta_y * delta_y)
                * inv_two_sigma_sq[valid]
            )
            target = gy[valid].astype(np.int64) * width + gx[valid].astype(np.int64)

            np.maximum.at(best_weight, target, weights)
            is_winner = weights >= (best_weight[target] - 1e-8)
            if np.any(is_winner):
                winning_target = target[is_winner]
                label_image[winning_target] = labels[valid_idx[is_winner]]

    image = np.empty((height * width, 3), dtype=np.uint8)
    image[:] = np.asarray(background, dtype=np.uint8)
    valid = label_image >= 0
    if np.any(valid):
        mapped = np.clip(label_image[valid], 0, palette.shape[0] - 1)
        image[valid] = palette[mapped]

    coverage = np.clip(best_weight * 4.0, 0.0, 1.0)
    coverage = np.rint(coverage * 255.0).astype(np.uint8)
    return (
        np.flipud(image.reshape(height, width, 3)),
        np.flipud(coverage.reshape(height, width)),
    )


def render_projected_view(
    points: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
    scales: typing.Optional[np.ndarray],
    opacities: typing.Optional[np.ndarray],
    view_mode: str,
    elevation_deg: float,
    azimuth_deg: float,
    res: float,
    pad: float,
    min_sigma_px: float,
    max_sigma_px: float,
    max_kernel_radius: int,
    palette: np.ndarray,
) -> typing.Tuple[np.ndarray, np.ndarray, Bounds, dict]:
    projected = project_fixed_view(
        points,
        view_mode,
        elevation_deg,
        azimuth_deg=azimuth_deg,
    )
    bounds = compute_bounds(projected[:, :2], res=res, pad=pad)
    px, py, _, _, linear = project_points_to_bev(
        projected,
        bounds=bounds,
        res=res,
    )
    chosen_idx = top_surface_indices(linear, projected[:, 2], flip=False)
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
        background_value=0.0,
    )
    rgb_img = np.clip(rgb_img * 255.0, 0, 255).astype(np.uint8)

    semantic_img, coverage = gaussian_splat_labels(
        labels=labels[chosen_idx],
        px=px[chosen_idx],
        py=py[chosen_idx],
        sigma_px=sigma_px,
        alpha=alpha,
        bounds=bounds,
        palette=palette,
        max_kernel_radius=max_kernel_radius,
    )
    rgb_img[coverage == 0] = 0
    rgb_rgba = np.dstack([rgb_img, coverage])
    semantic_rgba = np.dstack([semantic_img, coverage])
    metadata = {
        "view_mode": view_mode,
        "azimuth_degrees": float(azimuth_deg) % 360.0,
        "bev_size": [int(bounds[5]), int(bounds[4])],
        "visible_splats": int(chosen_idx.shape[0]),
    }
    return rgb_rgba, semantic_rgba, bounds, metadata


def project_to_image_pixels(
    points: np.ndarray,
    view_mode: str,
    elevation_deg: float,
    azimuth_deg: float,
    bounds: Bounds,
    res: float,
) -> np.ndarray:
    projected = project_fixed_view(
        points,
        view_mode,
        elevation_deg,
        azimuth_deg=azimuth_deg,
    )
    xmin, _, ymin, _, _, height = bounds
    px = (projected[:, 0] - xmin) / res
    py = (projected[:, 1] - ymin) / res
    return np.stack([px, (height - 1) - py], axis=1).astype(np.float32)


def camera_frustum_world(
    c2w: np.ndarray,
    intrinsics: typing.Optional[np.ndarray],
    depth: float,
) -> np.ndarray:
    if intrinsics is not None:
        fx = max(float(intrinsics[0, 0]), 1e-6)
        fy = max(float(intrinsics[1, 1]), 1e-6)
        cx = max(float(intrinsics[0, 2]), 1.0)
        cy = max(float(intrinsics[1, 2]), 1.0)
        half_width = depth * cx / fx
        half_height = depth * cy / fy
    else:
        half_width = depth * 0.55
        half_height = depth * 0.40

    local = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-half_width, -half_height, -depth],
            [half_width, -half_height, -depth],
            [half_width, half_height, -depth],
            [-half_width, half_height, -depth],
        ],
        dtype=np.float32,
    )
    return (c2w[:3, :3] @ local.T).T + c2w[:3, 3][None]


def draw_dashed_polyline(
    draw: ImageDraw.ImageDraw,
    points: typing.Sequence[typing.Tuple[int, int]],
    fill: typing.Tuple[int, int, int, int],
    width: int,
    dash_length: float,
    gap_length: float,
) -> None:
    """Draw one dashed path while preserving dash phase across segments."""
    if len(points) < 2:
        return

    dash_length = max(float(dash_length), 1.0)
    gap_length = max(float(gap_length), 1.0)
    drawing = True
    remaining = dash_length

    for start, end in zip(points[:-1], points[1:]):
        start_xy = np.asarray(start, dtype=np.float32)
        end_xy = np.asarray(end, dtype=np.float32)
        delta = end_xy - start_xy
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-6:
            continue

        direction = delta / distance
        position = 0.0
        while position < distance - 1e-6:
            step = min(remaining, distance - position)
            if drawing and step > 1e-6:
                dash_start = start_xy + direction * position
                dash_end = start_xy + direction * (position + step)
                draw.line(
                    [tuple(dash_start), tuple(dash_end)],
                    fill=fill,
                    width=max(1, int(width)),
                )
            position += step
            remaining -= step
            if remaining <= 1e-6:
                drawing = not drawing
                remaining = dash_length if drawing else gap_length


def draw_trajectory_and_cameras(
    image: np.ndarray,
    camera_c2ws: np.ndarray,
    intrinsics: typing.Optional[np.ndarray],
    view_mode: str,
    elevation_deg: float,
    azimuth_deg: float,
    bounds: Bounds,
    res: float,
    pose_stride: int,
    frustum_depth: float,
    trajectory_width: int,
    trajectory_dash_length: float,
    trajectory_gap_length: float,
    frustum_width: int,
) -> typing.Tuple[np.ndarray, int]:
    base = Image.fromarray(image).convert("RGBA")
    if camera_c2ws.shape[0] == 0:
        return np.asarray(base), 0

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    trajectory_color = (255, 0, 0, 255)
    frustum_color = (40, 255, 70, 255)

    centers = camera_c2ws[:, :3, 3]
    path_px = project_to_image_pixels(
        centers,
        view_mode=view_mode,
        elevation_deg=elevation_deg,
        azimuth_deg=azimuth_deg,
        bounds=bounds,
        res=res,
    )
    path_tuples = [tuple(np.rint(point).astype(int)) for point in path_px]
    draw_dashed_polyline(
        draw,
        path_tuples,
        fill=trajectory_color,
        width=trajectory_width,
        dash_length=trajectory_dash_length,
        gap_length=trajectory_gap_length,
    )

    stride = max(1, int(pose_stride))
    pose_indices = list(range(0, camera_c2ws.shape[0], stride))
    if pose_indices[-1] != camera_c2ws.shape[0] - 1:
        pose_indices.append(camera_c2ws.shape[0] - 1)

    frustum_edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 1),
    ]
    for pose_idx in pose_indices:
        frustum = camera_frustum_world(
            camera_c2ws[pose_idx],
            intrinsics=intrinsics,
            depth=float(frustum_depth),
        )
        frustum_px = project_to_image_pixels(
            frustum,
            view_mode=view_mode,
            elevation_deg=elevation_deg,
            azimuth_deg=azimuth_deg,
            bounds=bounds,
            res=res,
        )
        frustum_tuples = [
            tuple(np.rint(point).astype(int))
            for point in frustum_px
        ]
        for start_idx, end_idx in frustum_edges:
            draw.line(
                [frustum_tuples[start_idx], frustum_tuples[end_idx]],
                fill=frustum_color,
                width=max(1, int(frustum_width)),
            )
        origin_x, origin_y = frustum_tuples[0]
        draw.ellipse(
            (origin_x - 1, origin_y - 1, origin_x + 1, origin_y + 1),
            fill=frustum_color,
        )

    composited = Image.alpha_composite(base, overlay)
    return np.asarray(composited), len(pose_indices)

def fit_to_square_canvas(

    image: np.ndarray,
    canvas_size: int,
    nearest: bool,
    margin_ratio: float = 0.04,
) -> np.ndarray:
    image_pil = Image.fromarray(image).convert("RGBA")
    margin = max(1, int(round(canvas_size * margin_ratio)))
    available = max(1, canvas_size - 2 * margin)
    scale = min(available / image_pil.width, available / image_pil.height)
    new_size = (
        max(1, int(round(image_pil.width * scale))),
        max(1, int(round(image_pil.height * scale))),
    )
    if nearest:
        resized = image_pil.resize(new_size, Image.Resampling.NEAREST)
    else:
        resized = (
            image_pil.convert("RGBa")
            .resize(new_size, Image.Resampling.LANCZOS)
            .convert("RGBA")
        )
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    offset = (
        (canvas_size - resized.width) // 2,
        (canvas_size - resized.height) // 2,
    )
    canvas.paste(resized, offset)
    return np.asarray(canvas, dtype=np.uint8)


def save_png(path: str, image: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(image).save(path, format="PNG", optimize=True)


def azimuth_suffix(azimuth_deg: float) -> str:
    normalized = float(azimuth_deg) % 360.0
    if abs(normalized - round(normalized)) < 1e-6:
        return f"{int(round(normalized)):03d}"
    return f"{normalized:06.2f}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render aligned top-down and four-direction oblique BEV views "
            "with transparent backgrounds."
        )
    )
    parser.add_argument("--npz", required=True, help="final params.npz")
    parser.add_argument("--traj", required=True, help="dataset traj.txt anchor")
    parser.add_argument("--scene", required=True, help="scene name")
    parser.add_argument("--out-dir", required=True, help="output directory")
    parser.add_argument("--dataset", default="Replica", help="dataset name")
    parser.add_argument("--res", type=float, default=0.01, help="meters per pixel")
    parser.add_argument("--pad", type=float, default=0.06, help="view padding ratio")
    parser.add_argument(
        "--elevation",
        type=float,
        default=80.0,
        help="oblique view elevation in degrees",
    )
    parser.add_argument(
        "--azimuths",
        type=float,
        nargs="+",
        default=[0.0, 90.0, 180.0, 270.0],
        help="horizontal view directions in degrees",
    )
    parser.add_argument(
        "--remove-labels",
        type=int,
        nargs="+",
        default=[31, 47],
        help="semantic classes removed from every view",
    )
    parser.add_argument(
        "--ceiling-label",
        type=int,
        default=31,
        help="semantic class used to estimate the ceiling height",
    )
    parser.add_argument(
        "--ceiling-clearance",
        type=float,
        default=0.15,
        help="remove geometry this many meters below the median ceiling height",
    )
    parser.add_argument(
        "--wall-label",
        type=int,
        default=93,
        help="semantic wall class used for axis alignment",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=102,
        help="number of evaluation semantic classes",
    )
    parser.add_argument(
        "--pose-stride",
        type=int,
        default=10,
        help="draw one camera frustum every N poses",
    )
    parser.add_argument(
        "--frustum-depth",
        type=float,
        default=0.18,
        help="camera frustum depth in meters",
    )
    parser.add_argument(
        "--trajectory-width",
        type=int,
        default=2,
        help="trajectory width in render pixels",
    )
    parser.add_argument(
        "--trajectory-dash-length",
        type=float,
        default=4.0,
        help="black trajectory dash length in render pixels",
    )
    parser.add_argument(
        "--trajectory-gap-length",
        type=float,
        default=3.0,
        help="black trajectory gap length in render pixels",
    )
    parser.add_argument(
        "--frustum-width",
        type=int,
        default=1,
        help="frustum width in render pixels",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=3200,
        help="square output size in pixels",
    )
    parser.add_argument("--min-sigma-px", type=float, default=0.9)
    parser.add_argument("--max-sigma-px", type=float, default=4.0)
    parser.add_argument("--max-kernel-radius", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fields = load_npz_fields(args.npz)
    camera_c2ws = load_world_camera_poses(args.npz, args.traj)

    points_world = convert_points_to_sim(
        fields["points"],
        dataset=args.dataset,
        coord_system="sim",
        traj_path=args.traj,
    )
    colors = fields["colors"]
    labels = fields["labels"]
    if colors is None:
        raise ValueError("RGB colors are missing from params.npz")
    if labels is None:
        raise ValueError("Semantic labels are missing from params.npz")

    ceiling_points = points_world[
        labels == int(args.ceiling_label),
        2,
    ]
    ceiling_height = (
        float(np.median(ceiling_points))
        if ceiling_points.size
        else float(np.percentile(points_world[:, 2], 95.0))
    )
    ceiling_cutoff = ceiling_height - max(-3.0, float(args.ceiling_clearance))
    remove_labels = np.asarray(args.remove_labels, dtype=np.int32)
    label_keep = ~np.isin(labels, remove_labels)
    height_keep = points_world[:, 2] < ceiling_cutoff
    keep = label_keep & height_keep
    if not np.any(keep):
        raise ValueError("No points remain after removing upper geometry")

    input_point_count = int(points_world.shape[0])
    removed_label_count = int(np.count_nonzero(~label_keep))
    removed_height_count = int(np.count_nonzero(label_keep & ~height_keep))
    points_world = points_world[keep]
    colors = colors[keep]
    labels = labels[keep]
    scales = fields["scales"]
    if scales is not None:
        scales = scales[keep]
    opacities = fields["opacities"]
    if opacities is not None:
        opacities = opacities[keep]

    canonical_transform, alignment_metadata = estimate_canonical_transform(
        points_world,
        camera_c2ws,
        labels=labels,
        wall_label=args.wall_label,
    )
    points_canonical = transform_points(points_world, canonical_transform)
    cameras_canonical = transform_camera_poses(camera_c2ws, canonical_transform)

    azimuths = []
    for value in args.azimuths:
        normalized = float(value) % 360.0
        if not any(abs(normalized - existing) < 1e-6 for existing in azimuths):
            azimuths.append(normalized)
    if not azimuths:
        raise ValueError("At least one azimuth is required")

    palette = eval_label_colormap(args.num_classes)
    outputs = {}
    oblique_views = []
    num_frustums = 0
    for azimuth_deg in azimuths:
        oblique_rgb, oblique_semantic, oblique_bounds, view_metadata = render_projected_view(
            points=points_canonical,
            colors=colors,
            labels=labels,
            scales=scales,
            opacities=opacities,
            view_mode="oblique",
            elevation_deg=args.elevation,
            azimuth_deg=azimuth_deg,
            res=args.res,
            pad=args.pad,
            min_sigma_px=args.min_sigma_px,
            max_sigma_px=args.max_sigma_px,
            max_kernel_radius=args.max_kernel_radius,
            palette=palette,
        )
        oblique_rgb_without_trajectory = fit_to_square_canvas(
            oblique_rgb,
            canvas_size=args.canvas_size,
            nearest=False,
        )
        oblique_rgb, view_num_frustums = draw_trajectory_and_cameras(
            image=oblique_rgb,
            camera_c2ws=cameras_canonical,
            intrinsics=fields["intrinsics"],
            view_mode="oblique",
            elevation_deg=args.elevation,
            azimuth_deg=azimuth_deg,
            bounds=oblique_bounds,
            res=args.res,
            pose_stride=args.pose_stride,
            frustum_depth=args.frustum_depth,
            trajectory_width=args.trajectory_width,
            trajectory_dash_length=args.trajectory_dash_length,
            trajectory_gap_length=args.trajectory_gap_length,
            frustum_width=args.frustum_width,
        )
        num_frustums = max(num_frustums, view_num_frustums)
        oblique_rgb = fit_to_square_canvas(
            oblique_rgb,
            canvas_size=args.canvas_size,
            nearest=False,
        )
        oblique_semantic = fit_to_square_canvas(
            oblique_semantic,
            canvas_size=args.canvas_size,
            nearest=True,
        )

        suffix = azimuth_suffix(azimuth_deg)
        rgb_without_trajectory_path = os.path.join(
            args.out_dir,
            f"bev_rgb_oblique_azimuth_{suffix}.png",
        )
        rgb_path = os.path.join(
            args.out_dir,
            f"bev_rgb_trajectory_oblique_azimuth_{suffix}.png",
        )
        semantic_path = os.path.join(
            args.out_dir,
            f"bev_semantic_oblique_azimuth_{suffix}.png",
        )
        save_png(rgb_without_trajectory_path, oblique_rgb_without_trajectory)
        save_png(rgb_path, oblique_rgb)
        save_png(semantic_path, oblique_semantic)
        outputs[f"rgb_oblique_azimuth_{suffix}"] = rgb_without_trajectory_path
        outputs[f"rgb_trajectory_oblique_azimuth_{suffix}"] = rgb_path
        outputs[f"semantic_oblique_azimuth_{suffix}"] = semantic_path
        oblique_views.append(
            {
                **view_metadata,
                "elevation_degrees": float(args.elevation),
                "rgb_without_trajectory_output": rgb_without_trajectory_path,
                "rgb_output": rgb_path,
                "semantic_output": semantic_path,
            }
        )

    _, topdown_semantic, _, topdown_metadata = render_projected_view(
        points=points_canonical,
        colors=colors,
        labels=labels,
        scales=scales,
        opacities=opacities,
        view_mode="topdown",
        elevation_deg=90.0,
        azimuth_deg=0.0,
        res=args.res,
        pad=args.pad,
        min_sigma_px=args.min_sigma_px,
        max_sigma_px=args.max_sigma_px,
        max_kernel_radius=args.max_kernel_radius,
        palette=palette,
    )
    topdown_semantic = fit_to_square_canvas(
        topdown_semantic,
        canvas_size=args.canvas_size,
        nearest=True,
    )
    semantic_topdown_path = os.path.join(
        args.out_dir,
        "bev_semantic_topdown.png",
    )
    metadata_path = os.path.join(args.out_dir, "bev_metadata.json")
    save_png(semantic_topdown_path, topdown_semantic)
    outputs["semantic_topdown"] = semantic_topdown_path

    metadata = {
        "scene": args.scene,
        "input_npz": args.npz,
        "trajectory_anchor": args.traj,
        "coordinate_system": "canonical Replica world",
        "alignment": alignment_metadata,
        "oblique_views": oblique_views,
        "topdown": topdown_metadata,
        "camera_poses_total": int(camera_c2ws.shape[0]),
        "camera_frustums_drawn_per_view": int(num_frustums),
        "camera_pose_stride": int(args.pose_stride),
        "trajectory_style": {
            "color_rgba": [0, 0, 0, 255],
            "line": "dashed",
            "dash_length_render_px": float(args.trajectory_dash_length),
            "gap_length_render_px": float(args.trajectory_gap_length),
            "width_render_px": int(args.trajectory_width),
        },
        "camera_frustum_style": {
            "color_rgba": [40, 255, 70, 255],
            "width_render_px": int(args.frustum_width),
        },
        "input_point_count": input_point_count,
        "retained_point_count": int(points_world.shape[0]),
        "removed_semantic_labels": [int(label) for label in remove_labels],
        "removed_by_label_count": removed_label_count,
        "removed_above_cutoff_count": removed_height_count,
        "ceiling_height_m": float(ceiling_height),
        "ceiling_cutoff_m": float(ceiling_cutoff),
        "ceiling_clearance_m": float(args.ceiling_clearance),
        "meters_per_pixel": float(args.res),
        "semantic_palette": f"imgviz.label_colormap({args.num_classes})",
        "semantic_interpolation": "nearest",
        "background": "transparent RGBA",
        "canvas_size": [int(args.canvas_size), int(args.canvas_size)],
        "outputs": outputs,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Scene              : {args.scene}")
    print(f"Input npz          : {args.npz}")
    print(f"Canonical rotation : {alignment_metadata['rotation_degrees']:.2f} deg")
    print(f"Alignment source   : {alignment_metadata['alignment_source']}")
    print(f"Oblique elevation  : {args.elevation:.2f} deg")
    print(f"Oblique azimuths   : {', '.join(str(value) for value in azimuths)}")
    print(f"Camera poses       : {camera_c2ws.shape[0]}")
    print(f"Camera frustums    : {num_frustums} per view")
    print(f"Ceiling cutoff     : {ceiling_cutoff:.3f} m")
    print(f"Semantic palette   : imgviz.label_colormap({args.num_classes})")
    print(f"Saved views        : {len(outputs)}")
    print(f"Saved Metadata     : {metadata_path}")


if __name__ == "__main__":
    main()
