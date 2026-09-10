"""
Evaluate semantic mIoU from six orthographic scene views.

The evaluator reads a reconstructed SemSplaTAM/SplaTAM `params.npz`, projects
the reconstructed semantic labels and the dataset semantic mesh into aligned
orthographic label maps, then computes mIoU per view.
"""

import argparse
import json
import math
import os
import sys
import typing

os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())

from src.visualization.render_rgb_sem_entro_bev import (  # noqa: E402
    Bounds,
    compute_bounds,
    convert_points_to_sim,
    label_colors,
    load_gt_semantic,
    load_mp3d_label_map,
    load_npz_fields,
    normalize_label_map,
    visible_opacity,
    visible_scales_px,
)


ViewSpec = typing.NamedTuple(
    "ViewSpec",
    [
        ("key", str),
        ("name", str),
        ("u_axis", int),
        ("v_axis", int),
        ("depth_axis", int),
        ("flip", bool),
    ],
)


ORTHO_VIEWS = [
    ViewSpec("bird_eye", "Bird Eye View", 0, 1, 2, False),
    ViewSpec("front_elevation", "Front Elevation View", 0, 2, 1, False),
    ViewSpec("back_elevation", "Back Elevation View", 0, 2, 1, True),
    ViewSpec("upward_elevation", "Upward Elevation View", 0, 1, 2, True),
    ViewSpec("left_side_elevation", "Left-side Elevation View", 1, 2, 0, True),
    ViewSpec("right_side_elevation", "Right-side Elevation View", 1, 2, 0, False),
]


def finite_or_none(value: float) -> typing.Optional[float]:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def resolve_traj_path(dataset: str, scene: typing.Optional[str], explicit_traj: typing.Optional[str]) -> typing.Optional[str]:
    if explicit_traj is not None:
        return explicit_traj
    if not scene:
        return None

    dataset_lower = dataset.lower()
    if dataset_lower == "replica":
        candidate = os.path.join("data", "Replica", scene, "traj.txt")
    elif dataset_lower == "mp3d":
        candidate = os.path.join("data", "mp3d_sim_nvs_v2", scene, "traj.txt")
    else:
        candidate = None

    if candidate is not None and os.path.exists(candidate):
        return candidate
    return None


def view_uv_depth(points: np.ndarray, view: ViewSpec) -> typing.Tuple[np.ndarray, np.ndarray]:
    uv = points[:, [view.u_axis, view.v_axis]].astype(np.float32)
    depth = points[:, view.depth_axis].astype(np.float32)
    return uv, depth


def project_uv_to_bounds(uv: np.ndarray, bounds: Bounds, res: float) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xmin, _, ymin, _, width, height = bounds
    px = (uv[:, 0] - xmin) / res
    py = (uv[:, 1] - ymin) / res
    ix = np.clip(np.floor(px).astype(np.int64), 0, width - 1)
    iy = np.clip(np.floor(py).astype(np.int64), 0, height - 1)
    linear = iy * width + ix
    return px, py, ix, iy, linear


def visible_indices(linear: np.ndarray, depth: np.ndarray, flip: bool) -> np.ndarray:
    order = np.argsort(depth if flip else -depth)
    linear_sorted = linear[order]
    _, first = np.unique(linear_sorted, return_index=True)
    return order[first]


def filter_points_for_view(
    points: np.ndarray,
    view: ViewSpec,
    remove_front_percent: float,
    min_height: typing.Optional[float],
    max_height: typing.Optional[float],
) -> typing.Tuple[np.ndarray, typing.Optional[float]]:
    keep = np.ones(points.shape[0], dtype=bool)
    cutoff = None

    if remove_front_percent > 0.0:
        if not (0.0 <= remove_front_percent < 100.0):
            raise ValueError("--remove-front-percent must be in [0, 100)")
        depth = points[:, view.depth_axis]
        nearness = -depth if view.flip else depth
        cutoff = float(np.percentile(nearness, 100.0 - remove_front_percent))
        keep &= nearness <= cutoff

    if min_height is not None:
        keep &= points[:, 2] >= float(min_height)
    if max_height is not None:
        keep &= points[:, 2] <= float(max_height)

    return keep, cutoff


def splat_label_map(
    labels: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    sigma_px: np.ndarray,
    alpha: np.ndarray,
    bounds: Bounds,
    max_kernel_radius: int,
) -> np.ndarray:
    _, _, _, _, width, height = bounds
    num_pixels = width * height

    labels = labels.astype(np.int32)
    label_flat = np.zeros(num_pixels, dtype=np.int32)
    best_weight = np.zeros(num_pixels, dtype=np.float32)

    base_x = np.floor(px).astype(np.int32)
    base_y = np.floor(py).astype(np.int32)
    radius_px = np.clip(np.ceil(3.0 * sigma_px).astype(np.int32), 1, max_kernel_radius)
    inv_two_sigma_sq = 0.5 / np.maximum(sigma_px * sigma_px, 1e-8)

    for dy in range(-max_kernel_radius, max_kernel_radius + 1):
        gy = base_y + dy
        valid_y = (gy >= 0) & (gy < height) & (np.abs(dy) <= radius_px)
        if not np.any(valid_y):
            continue
        dy_center = (gy.astype(np.float32) + 0.5) - py
        for dx in range(-max_kernel_radius, max_kernel_radius + 1):
            gx = base_x + dx
            valid = valid_y & (gx >= 0) & (gx < width) & (np.abs(dx) <= radius_px)
            if not np.any(valid):
                continue

            delta_x = (gx[valid].astype(np.float32) + 0.5) - px[valid]
            delta_y = dy_center[valid]
            d2 = delta_x * delta_x + delta_y * delta_y
            weight = alpha[valid] * np.exp(-d2 * inv_two_sigma_sq[valid])
            target = gy[valid].astype(np.int64) * width + gx[valid].astype(np.int64)
            np.maximum.at(best_weight, target, weight.astype(np.float32))

    for dy in range(-max_kernel_radius, max_kernel_radius + 1):
        gy = base_y + dy
        valid_y = (gy >= 0) & (gy < height) & (np.abs(dy) <= radius_px)
        if not np.any(valid_y):
            continue
        dy_center = (gy.astype(np.float32) + 0.5) - py
        for dx in range(-max_kernel_radius, max_kernel_radius + 1):
            gx = base_x + dx
            valid = valid_y & (gx >= 0) & (gx < width) & (np.abs(dx) <= radius_px)
            if not np.any(valid):
                continue

            delta_x = (gx[valid].astype(np.float32) + 0.5) - px[valid]
            delta_y = dy_center[valid]
            d2 = delta_x * delta_x + delta_y * delta_y
            weight = alpha[valid] * np.exp(-d2 * inv_two_sigma_sq[valid])
            target = gy[valid].astype(np.int64) * width + gx[valid].astype(np.int64)
            update = weight >= (best_weight[target] - 1e-8)
            if np.any(update):
                label_flat[target[update]] = labels[valid][update]

    label_img = label_flat.reshape(height, width)
    return np.flipud(label_img)


def render_label_projection(
    points: np.ndarray,
    labels: np.ndarray,
    view: ViewSpec,
    bounds: Bounds,
    res: float,
    mode: str,
    scales: typing.Optional[np.ndarray],
    opacities: typing.Optional[np.ndarray],
    min_sigma_px: float,
    max_sigma_px: float,
    max_kernel_radius: int,
    fixed_sigma_px: float,
) -> typing.Tuple[np.ndarray, int]:
    uv, depth = view_uv_depth(points, view)
    px, py, ix, iy, linear = project_uv_to_bounds(uv, bounds=bounds, res=res)
    chosen_idx = visible_indices(linear, depth, flip=view.flip)

    _, _, _, _, width, height = bounds
    if mode == "nearest":
        image = np.zeros((height, width), dtype=np.int32)
        image[iy[chosen_idx], ix[chosen_idx]] = labels[chosen_idx].astype(np.int32)
        return np.flipud(image), int(chosen_idx.shape[0])

    if scales is None:
        sigma_px = np.full(chosen_idx.shape[0], fixed_sigma_px, dtype=np.float32)
    else:
        sigma_px = visible_scales_px(
            scales,
            chosen_idx,
            res=res,
            min_sigma_px=min_sigma_px,
            max_sigma_px=max_sigma_px,
        )
    alpha = visible_opacity(opacities, chosen_idx)
    label_img = splat_label_map(
        labels=labels[chosen_idx],
        px=px[chosen_idx],
        py=py[chosen_idx],
        sigma_px=sigma_px,
        alpha=alpha,
        bounds=bounds,
        max_kernel_radius=max_kernel_radius,
    )
    return label_img, int(chosen_idx.shape[0])


def calc_miou_np(
    pred: np.ndarray,
    target: np.ndarray,
    ignore_label: int = 0,
) -> typing.Tuple[float, typing.Dict[int, float]]:
    pred_flat = pred.reshape(-1).astype(np.int64)
    target_flat = target.reshape(-1).astype(np.int64)
    valid = target_flat != int(ignore_label)
    pred_valid = pred_flat[valid]
    target_valid = target_flat[valid]

    if target_valid.size == 0:
        return float("nan"), {}

    classes = np.unique(np.concatenate([pred_valid, target_valid]))
    classes = classes[classes != int(ignore_label)]
    ious = {}
    for cls in classes:
        pred_cls = pred_valid == cls
        target_cls = target_valid == cls
        intersection = np.logical_and(pred_cls, target_cls).sum(dtype=np.float64)
        union = np.logical_or(pred_cls, target_cls).sum(dtype=np.float64)
        if union > 0:
            ious[int(cls)] = float(intersection / union)

    if not ious:
        return float("nan"), {}
    return float(np.mean(list(ious.values()))), ious


def colorize_label_image(label_img: np.ndarray, label_map: typing.Optional[dict]) -> np.ndarray:
    labels = np.unique(label_img.astype(np.int32))
    lut = label_colors(labels, label_map)
    image = np.full((*label_img.shape, 3), 255, dtype=np.uint8)
    for label in labels:
        label = int(label)
        if label == 0:
            continue
        image[label_img == label] = lut[label]
    return image


def save_label_preview(path: str, label_img: np.ndarray, label_map: typing.Optional[dict], dpi: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = colorize_label_image(label_img, label_map)
    plt.figure(figsize=(8, 8))
    plt.imshow(image, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate semantic mIoU over six orthographic views.")
    parser.add_argument("--npz", required=True, help="path to params.npz")
    parser.add_argument("--dataset", default="Replica", help="dataset name")
    parser.add_argument("--scene", required=True, help="scene name used to load GT semantic mesh")
    parser.add_argument("--traj", default=None, help="optional trajectory txt used by exploration-map coordinates")
    parser.add_argument("--out-dir", required=True, help="directory for orthoview_miou.{json,txt}")
    parser.add_argument("--coord-system", choices=["sim", "slam"], default="sim", help="coordinate system of rendered points")
    parser.add_argument("--render-mode", choices=["gaussian", "nearest"], default="gaussian", help="label projection mode")
    parser.add_argument("--res", type=float, default=0.01, help="meters per pixel")
    parser.add_argument("--pad", type=float, default=0.05, help="orthographic bound padding ratio")
    parser.add_argument("--remove-front-percent", type=float, default=0.0, help="remove nearest N percent along each view direction before projection")
    parser.add_argument("--min-height", type=float, default=None, help="optional lower z bound after coordinate conversion")
    parser.add_argument("--max-height", type=float, default=None, help="optional upper z bound after coordinate conversion")
    parser.add_argument("--label-map", default=None, help="JSON label->RGB mapping for saved previews")
    parser.add_argument("--mp3d-palette", choices=["high-contrast", "mpcat40", "eval"], default="high-contrast", help="MP3D palette when --label-map is not provided")
    parser.add_argument("--min-sigma-px", type=float, default=0.75, help="minimum reconstructed Gaussian label splat sigma")
    parser.add_argument("--max-sigma-px", type=float, default=3.0, help="maximum reconstructed Gaussian label splat sigma")
    parser.add_argument("--max-kernel-radius", type=int, default=4, help="maximum reconstructed Gaussian label splat radius")
    parser.add_argument("--gt-sigma-px", type=float, default=1.0, help="GT semantic mesh label splat sigma")
    parser.add_argument("--gt-kernel-radius", type=int, default=3, help="GT semantic mesh label splat radius")
    parser.add_argument("--ignore-label", type=int, default=0, help="target label ignored when computing mIoU")
    parser.add_argument("--save-views", action="store_true", help="save pred/GT semantic preview PNGs for each view")
    parser.add_argument("--dpi", type=int, default=300, help="preview image DPI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.coord_system != "sim":
        raise ValueError("Orthoview mIoU compares against GT semantic meshes in sim/world coordinates; use --coord-system sim")

    fields = load_npz_fields(args.npz)
    if fields["labels"] is None:
        raise ValueError("Semantic labels are missing; expected `seman_cls_ids` or `semantic_logits` in params.npz")

    traj_path = resolve_traj_path(args.dataset, args.scene, args.traj)
    pred_points = convert_points_to_sim(
        fields["points"],
        dataset=args.dataset,
        coord_system=args.coord_system,
        traj_path=traj_path,
    )
    pred_labels = fields["labels"].astype(np.int32)
    pred_scales = fields["scales"]
    pred_opacities = fields["opacities"]

    gt_points, gt_labels = load_gt_semantic(args.dataset, args.scene)
    gt_labels = gt_labels.astype(np.int32)

    label_map = None
    dataset_lower = args.dataset.lower()
    if args.label_map is not None:
        with open(args.label_map, "r", encoding="utf-8") as f:
            label_map = normalize_label_map(json.load(f))
    elif dataset_lower == "mp3d":
        label_map = load_mp3d_label_map(args.mp3d_palette)

    results = {
        "npz": args.npz,
        "dataset": args.dataset,
        "scene": args.scene,
        "coord_system": args.coord_system,
        "traj": traj_path,
        "render_mode": args.render_mode,
        "res": args.res,
        "ignore_label": args.ignore_label,
        "views": {},
    }

    miou_values = []
    for view in ORTHO_VIEWS:
        pred_keep, pred_cutoff = filter_points_for_view(
            pred_points,
            view,
            remove_front_percent=args.remove_front_percent,
            min_height=args.min_height,
            max_height=args.max_height,
        )
        gt_keep, _ = filter_points_for_view(
            gt_points,
            view,
            remove_front_percent=args.remove_front_percent,
            min_height=args.min_height,
            max_height=args.max_height,
        )
        if not np.any(pred_keep):
            raise ValueError(f"No reconstructed points remain for {view.name}")
        if not np.any(gt_keep):
            raise ValueError(f"No GT semantic points remain for {view.name}")

        pred_view_points = pred_points[pred_keep]
        pred_view_labels = pred_labels[pred_keep]
        gt_view_points = gt_points[gt_keep]
        gt_view_labels = gt_labels[gt_keep]

        pred_view_scales = pred_scales[pred_keep] if pred_scales is not None else None
        pred_view_opacities = pred_opacities[pred_keep] if pred_opacities is not None else None

        pred_uv, _ = view_uv_depth(pred_view_points, view)
        gt_uv, _ = view_uv_depth(gt_view_points, view)
        bounds = compute_bounds(np.concatenate([pred_uv, gt_uv], axis=0), res=args.res, pad=args.pad)

        pred_label_img, pred_visible = render_label_projection(
            points=pred_view_points,
            labels=pred_view_labels,
            view=view,
            bounds=bounds,
            res=args.res,
            mode=args.render_mode,
            scales=pred_view_scales,
            opacities=pred_view_opacities,
            min_sigma_px=args.min_sigma_px,
            max_sigma_px=args.max_sigma_px,
            max_kernel_radius=args.max_kernel_radius,
            fixed_sigma_px=args.min_sigma_px,
        )
        gt_label_img, gt_visible = render_label_projection(
            points=gt_view_points,
            labels=gt_view_labels,
            view=view,
            bounds=bounds,
            res=args.res,
            mode="gaussian",
            scales=None,
            opacities=None,
            min_sigma_px=args.gt_sigma_px,
            max_sigma_px=args.gt_sigma_px,
            max_kernel_radius=args.gt_kernel_radius,
            fixed_sigma_px=args.gt_sigma_px,
        )

        miou, class_ious = calc_miou_np(pred_label_img, gt_label_img, ignore_label=args.ignore_label)
        if math.isfinite(miou):
            miou_values.append(miou)

        view_result = {
            "name": view.name,
            "miou": finite_or_none(miou),
            "miou_percent": finite_or_none(miou * 100.0),
            "valid_target_pixels": int(np.count_nonzero(gt_label_img != args.ignore_label)),
            "pred_visible_points": int(pred_visible),
            "gt_visible_points": int(gt_visible),
            "image_size": {"height": int(bounds[5]), "width": int(bounds[4])},
            "front_cutoff": finite_or_none(pred_cutoff),
            "class_iou": {str(k): finite_or_none(v) for k, v in class_ious.items()},
        }
        results["views"][view.key] = view_result

        if args.save_views:
            save_label_preview(
                os.path.join(args.out_dir, f"{view.key}_pred.png"),
                pred_label_img,
                label_map,
                dpi=args.dpi,
            )
            save_label_preview(
                os.path.join(args.out_dir, f"{view.key}_gt.png"),
                gt_label_img,
                label_map,
                dpi=args.dpi,
            )

    mean_miou = float(np.mean(miou_values)) if miou_values else float("nan")
    results["mean_miou"] = finite_or_none(mean_miou)
    results["mean_miou_percent"] = finite_or_none(mean_miou * 100.0)

    json_path = os.path.join(args.out_dir, "orthoview_miou.json")
    txt_path = os.path.join(args.out_dir, "orthoview_miou.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"npz: {args.npz}\n")
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"scene: {args.scene}\n")
        f.write(f"render_mode: {args.render_mode}\n")
        f.write(f"mean_miou: {results['mean_miou_percent']}\n")
        for view in ORTHO_VIEWS:
            item = results["views"][view.key]
            f.write(
                f"{item['name']}: miou={item['miou_percent']} "
                f"valid_target_pixels={item['valid_target_pixels']} "
                f"pred_visible_points={item['pred_visible_points']} "
                f"gt_visible_points={item['gt_visible_points']}\n"
            )

    print(f"Saved JSON: {json_path}")
    print(f"Saved TXT : {txt_path}")
    print(f"Mean mIoU : {results['mean_miou_percent']}")
    for view in ORTHO_VIEWS:
        item = results["views"][view.key]
        print(f"{item['name']}: {item['miou_percent']}")


if __name__ == "__main__":
    main()
