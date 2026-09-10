#!/usr/bin/env python3
"""Metrics-only MP3D semantic evaluation for the paper's Table S.5 protocol."""

import argparse
import csv
import json
import math
import os
import runpy
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from channel_rasterization import GaussianRasterizer as SemanticRenderer
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "third_parties/splatam"))

from datasets.gradslam_datasets import load_dataset_config  # noqa: E402
from src.slam.semsplatam.modified_ver.scripts.splatam import get_dataset  # noqa: E402
from src.slam.semsplatam.modified_ver.splatam.splatam import (  # noqa: E402
    setup_camera,
    transformed_params2semrendervar,
)
from third_parties.splatam.utils.slam_external import build_rotation  # noqa: E402
from third_parties.splatam.utils.slam_helpers import transformed_params2rendervar  # noqa: E402


PAPER_CLASSES = [17, 37, 15, 14, 26, 5]
PAPER_CLASS_NAMES = ["ceiling", "appliances", "sink", "plant", "counter", "table"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-basedir",
        type=Path,
        default=REPO_ROOT / "data/mp3d_sim_nvs_v2",
    )
    parser.add_argument(
        "--room-config",
        type=Path,
        default=REPO_ROOT / "configs/MP3D/mp3d_splatam_s.py",
    )
    parser.add_argument(
        "--class-info",
        type=Path,
        default=REPO_ROOT / "configs/MP3D/class_info_file.json",
    )
    parser.add_argument("--experiment", default="")
    parser.add_argument("--stage", default="")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_eval_dataset(args: argparse.Namespace):
    room_config = runpy.run_path(str(args.room_config))["config"]
    data_config = room_config["data"]
    gradslam_config_path = Path(data_config["gradslam_data_cfg"])
    if not gradslam_config_path.is_absolute():
        gradslam_config_path = REPO_ROOT / gradslam_config_path
    gradslam_config = load_dataset_config(str(gradslam_config_path))

    return get_dataset(
        config_dict=gradslam_config,
        basedir=str(args.dataset_basedir),
        sequence=args.scene,
        start=data_config["start"],
        end=data_config["end"],
        stride=data_config["stride"],
        desired_height=data_config["desired_image_height"],
        desired_width=data_config["desired_image_width"],
        device=args.device,
        relative_pose=True,
        ignore_bad=data_config.get("ignore_bad", False),
        use_train_split=data_config.get("use_train_split", True),
        load_semantics=True,
    )


def load_checkpoint(path: Path, device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    required = {
        "means3D",
        "unnorm_rotations",
        "logit_opacities",
        "log_scales",
        "semantic_logits",
        "seman_cls_ids",
        "rgb_colors",
    }
    with np.load(path, allow_pickle=True) as checkpoint:
        missing = sorted(required - set(checkpoint.files))
        if missing:
            raise ValueError(f"{path} is missing required keys: {missing}")
        params = {
            key: torch.as_tensor(checkpoint[key], device=device, dtype=torch.float32)
            for key in required
            if key != "seman_cls_ids"
        }
        seman_cls_ids = torch.as_tensor(checkpoint["seman_cls_ids"], device=device, dtype=torch.long)

    if seman_cls_ids.ndim == 1:
        seman_cls_ids = seman_cls_ids[:, None]
    if params["semantic_logits"].ndim == 1:
        params["semantic_logits"] = params["semantic_logits"][:, None]
    if seman_cls_ids.shape != params["semantic_logits"].shape:
        raise ValueError(
            "seman_cls_ids and semantic_logits must have identical [N, top-k] shapes; "
            f"got {tuple(seman_cls_ids.shape)} and {tuple(params['semantic_logits'].shape)}"
        )

    variables = {
        "seman_cls_ids": seman_cls_ids,
        "n_cls": 41,
    }
    return params, variables


def transform_to_frame(
    params: Dict[str, torch.Tensor],
    rel_w2c: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    points = params["means3D"]
    rotations = params["unnorm_rotations"]
    points_h = torch.cat(
        (points, torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)),
        dim=1,
    )
    transformed = {
        "means3D": (rel_w2c @ points_h.T).T[:, :3],
    }
    if params["log_scales"].shape[1] == 1:
        transformed["unnorm_rotations"] = rotations
    else:
        camera_rotation = build_rotation(matrix_to_quaternion(rel_w2c[:3, :3])[None])[0]
        transformed["unnorm_rotations"] = quaternion_multiply(camera_rotation, F.normalize(rotations))
    return transformed


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    # MP3D checkpoints are isotropic; this branch supports anisotropic checkpoints defensively.
    trace = matrix.trace()
    qw = torch.sqrt(torch.clamp(1.0 + trace, min=1e-8)) / 2.0
    scale = 4.0 * qw.clamp_min(1e-8)
    return torch.stack(
        (
            qw,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )
    )


def quaternion_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def iou_for_class(pred: torch.Tensor, target: torch.Tensor, class_id: int) -> float:
    pred_class = pred == class_id
    target_class = target == class_id
    union = torch.count_nonzero(pred_class | target_class).item()
    if union == 0:
        return float("nan")
    intersection = torch.count_nonzero(pred_class & target_class).item()
    return float(intersection / union)


def frame_miou(pred: torch.Tensor, target: torch.Tensor) -> float:
    classes = torch.unique(torch.cat((pred, target)))
    classes = classes[(classes > 0) & (classes <= 40)]
    values = [iou_for_class(pred, target, int(class_id)) for class_id in classes.tolist()]
    return float(np.nanmean(values)) if values else float("nan")


def safe_nanmean(values: List[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(array)) if np.any(np.isfinite(array)) else float("nan")


def load_class_names(path: Path) -> Dict[int, str]:
    with path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    names = {0: "unknown"}
    for class_id, value in metadata.items():
        names[int(class_id)] = str(value.get("name", class_id))
    return names


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the semantic Gaussian rasterizer")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    dataset = load_eval_dataset(args)
    params, variables = load_checkpoint(args.params, device)
    class_names = load_class_names(args.class_info)

    first_color, _, first_intrinsics, first_pose = dataset[0]
    first_w2c = torch.linalg.inv(first_pose)
    camera = setup_camera(
        first_color.shape[1],
        first_color.shape[0],
        first_intrinsics[:3, :3].detach().cpu().numpy(),
        first_w2c.detach().cpu().numpy(),
        num_channels=41,
    )

    frame_mious: List[float] = []
    mapped_frame_mious: List[float] = []
    per_class_frame: Dict[int, List[float]] = {class_id: [] for class_id in range(1, 41)}
    totals = {
        class_id: {"tp": 0, "fp": 0, "fn": 0, "frames_with_gt": 0, "frames_with_union": 0}
        for class_id in range(1, 41)
    }
    evaluated_frames = 0

    for frame_id in tqdm(range(len(dataset)), desc=f"{args.experiment}/{args.scene}/{args.stage}"):
        if frame_id != 0 and (frame_id + 1) % args.eval_every != 0:
            continue
        _, _, _, pose = dataset[frame_id]
        gt = dataset.get_semantic_map(frame_id)[0].long()
        valid = gt != 0
        if not torch.any(valid):
            continue

        transformed = transform_to_frame(params, torch.linalg.inv(pose))
        rgb_render_variables = transformed_params2rendervar(params, transformed)
        _, radius, _ = Renderer(raster_settings=camera)(**rgb_render_variables)
        seen = radius > 0
        render_variables = transformed_params2semrendervar(params, variables, transformed, seen)
        rendered, _ = SemanticRenderer(raster_settings=camera)(**render_variables)
        rendered = torch.nan_to_num(rendered, nan=0.0).clamp_min_(0.0)
        pred = rendered.permute(1, 2, 0).argmax(dim=-1)

        pred_valid = pred[valid]
        gt_valid = gt[valid]
        frame_mious.append(frame_miou(pred_valid, gt_valid))

        # SGS-SLAM protocol used for the paper's Avg. column: map predictions
        # to the ground-truth categories that occur in this test view.
        candidate_ids = torch.unique(gt)
        mapped_pred = candidate_ids[rendered[candidate_ids].argmax(dim=0)]
        mapped_frame_mious.append(frame_miou(mapped_pred[valid], gt_valid))
        evaluated_frames += 1

        for class_id in range(1, 41):
            pred_class = pred_valid == class_id
            gt_class = gt_valid == class_id
            tp = int(torch.count_nonzero(pred_class & gt_class).item())
            fp = int(torch.count_nonzero(pred_class & ~gt_class).item())
            fn = int(torch.count_nonzero(~pred_class & gt_class).item())
            union = tp + fp + fn
            if torch.any(gt_class):
                totals[class_id]["frames_with_gt"] += 1
            if union:
                totals[class_id]["frames_with_union"] += 1
                per_class_frame[class_id].append(tp / union)
            else:
                per_class_frame[class_id].append(float("nan"))
            totals[class_id]["tp"] += tp
            totals[class_id]["fp"] += fp
            totals[class_id]["fn"] += fn

        del transformed, rgb_render_variables, render_variables, rendered, pred

    rows = []
    for class_id in range(1, 41):
        counts = totals[class_id]
        denominator = counts["tp"] + counts["fp"] + counts["fn"]
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_names.get(class_id, str(class_id)),
                "frame_mean_iou": safe_nanmean(per_class_frame[class_id]),
                "pooled_iou": counts["tp"] / denominator if denominator else float("nan"),
                **counts,
            }
        )

    row_by_id = {row["class_id"]: row for row in rows}
    paper_values = {
        name: row_by_id[class_id]["frame_mean_iou"]
        for class_id, name in zip(PAPER_CLASSES, PAPER_CLASS_NAMES)
    }
    pooled_class_ious = [row["pooled_iou"] for row in rows]
    metrics = {
        "experiment": args.experiment,
        "scene": args.scene,
        "stage": args.stage,
        "checkpoint": str(args.params.resolve()),
        "dataset_basedir": str(args.dataset_basedir.resolve()),
        "num_dataset_frames": len(dataset),
        "num_evaluated_frames": evaluated_frames,
        "eval_every": args.eval_every,
        "num_gaussians": int(params["means3D"].shape[0]),
        "protocol": {
            "avg": "frame-mean mIoU after mapping predictions to GT categories present in each view",
            "per_class": "frame-mean raw rendered-vs-GT IoU",
            "mpcat40": "frame-mean raw rendered-vs-GT mIoU over MPCAT40 labels",
        },
        "table_s5": {
            "avg": safe_nanmean(mapped_frame_mious),
            **paper_values,
            "mpcat40": safe_nanmean(frame_mious),
        },
        "pooled": {
            "avg_common": safe_nanmean(
                [row_by_id[class_id]["pooled_iou"] for class_id in PAPER_CLASSES]
            ),
            **{
                name: row_by_id[class_id]["pooled_iou"]
                for class_id, name in zip(PAPER_CLASSES, PAPER_CLASS_NAMES)
            },
            "mpcat40": safe_nanmean(pooled_class_ious),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, allow_nan=True)
    with (args.output_dir / "per_class_iou.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "table_s5.txt").open("w", encoding="utf-8") as file:
        for key, value in metrics["table_s5"].items():
            file.write(f"{key}: {value * 100:.6f}\n")

    print(json.dumps(metrics["table_s5"], indent=2))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    evaluate(parse_args())
