import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_replica_pose(line: str) -> np.ndarray:
    c2w = np.array(list(map(float, line.split())), dtype=np.float32).reshape(4, 4)
    c2w[:3, 1] *= -1
    c2w[:3, 2] *= -1
    return c2w


def load_first_traj_pose(transform_traj: Path) -> np.ndarray:
    with transform_traj.open("r") as f:
        first_line = f.readline().strip()
    if not first_line:
        raise ValueError(f"Empty trajectory file: {transform_traj}")
    values = np.array(list(map(float, first_line.split())), dtype=np.float32)
    if values.size != 16:
        raise ValueError(f"Expected 16 values in first trajectory pose, got {values.size}: {transform_traj}")
    return values.reshape(4, 4)


def load_slam_to_replica_world(transform_traj: Path) -> np.ndarray:
    with transform_traj.open("r") as f:
        first_line = f.readline().strip()
    if not first_line:
        raise ValueError(f"Empty trajectory file: {transform_traj}")
    slam2sim = load_replica_pose(first_line)
    slam2sim[:3, 1] *= -1
    slam2sim[:3, 2] *= -1
    return slam2sim


def load_slam_to_world(dataset: str, transform_traj: Path) -> np.ndarray:
    dataset_lower = dataset.lower()
    if dataset_lower == "replica":
        return load_slam_to_replica_world(transform_traj)
    if dataset_lower == "mp3d":
        return load_first_traj_pose(transform_traj)
    raise ValueError(f"Unsupported dataset for trajectory transform: {dataset}")


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.ones((len(points), 4), dtype=np.float32)
    homogeneous[:, :3] = points.astype(np.float32)
    transformed = homogeneous @ transform.T
    return (transformed[:, :3] / transformed[:, 3:4]).astype(np.float32)


def load_replica_id_to_label(info_path: Path) -> Tuple[np.ndarray, Dict[int, str]]:
    with info_path.open("r") as f:
        info = json.load(f)

    raw_id_to_label = info.get("id_to_label")
    if raw_id_to_label is None:
        raise ValueError(f"{info_path} does not contain id_to_label")

    if isinstance(raw_id_to_label, dict):
        max_id = max(int(k) for k in raw_id_to_label.keys())
        id_to_label = np.full(max_id + 1, -1, dtype=np.int64)
        for k, v in raw_id_to_label.items():
            id_to_label[int(k)] = int(v)
    else:
        id_to_label = np.asarray(raw_id_to_label, dtype=np.int64)

    class_names = {}
    for item in info.get("classes", []):
        class_names[int(item["id"])] = item["name"]
    return id_to_label, class_names


def load_mp3d_class_names(class_info_path: Path) -> Dict[int, str]:
    with class_info_path.open("r") as f:
        info = json.load(f)
    class_names = {}
    for key, value in info.items():
        try:
            class_id = int(key)
        except ValueError:
            continue
        if isinstance(value, dict):
            class_names[class_id] = str(value.get("name", class_id))
        else:
            class_names[class_id] = str(value)
    return class_names


def load_mp3d_instance_to_mpcat40(
    semseg_path: Path,
    category_mapping_path: Path,
    instance_map_path: Optional[Path],
) -> Dict[int, int]:
    if instance_map_path is not None and instance_map_path.is_file():
        with instance_map_path.open("r") as f:
            raw = json.load(f)
        return {int(k): int(v) for k, v in raw.items()}

    if not semseg_path.is_file():
        raise FileNotFoundError(f"MP3D semseg JSON not found: {semseg_path}")
    if not category_mapping_path.is_file():
        raise FileNotFoundError(f"MP3D category mapping TSV not found: {category_mapping_path}")

    label_to_mpcat40 = {}
    with category_mapping_path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                label_index = int(row["index"])
                mpcat40_index = int(row["mpcat40index"])
            except Exception:
                continue
            if mpcat40_index == 41:
                mpcat40_index = 0
            label_to_mpcat40[label_index] = mpcat40_index

    with semseg_path.open("r") as f:
        semseg_data = json.load(f)

    instance_to_mpcat40 = {}
    for group in semseg_data.get("segGroups", []):
        try:
            instance_id = int(group["id"])
            label_index = int(group["label_index"])
        except Exception:
            continue
        instance_to_mpcat40[instance_id] = int(label_to_mpcat40.get(label_index, 0))
    return instance_to_mpcat40


def load_gt_semantic_mesh(mesh_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ply = PlyData.read(mesh_path)
    vertex = ply["vertex"].data
    vertices = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)

    face_data = ply["face"].data
    if "object_id" not in face_data.dtype.names:
        raise ValueError(f"{mesh_path} has no face object_id property")

    raw_faces = face_data["vertex_indices"]
    raw_object_ids = np.asarray(face_data["object_id"], dtype=np.int64)
    faces = []
    object_ids = []
    skipped = 0
    for face, object_id in zip(raw_faces, raw_object_ids):
        face = np.asarray(face, dtype=np.int64)
        if len(face) < 3:
            skipped += 1
            continue
        for i in range(1, len(face) - 1):
            faces.append([face[0], face[i], face[i + 1]])
            object_ids.append(object_id)
    if skipped:
        print(f"Skipping {skipped} degenerate faces from {mesh_path}")

    faces = np.asarray(faces, dtype=np.int64)
    object_ids = np.asarray(object_ids, dtype=np.int64)
    return vertices, faces, object_ids


def map_object_ids_to_labels(object_ids: np.ndarray, id_to_label: np.ndarray) -> np.ndarray:
    labels = np.full(object_ids.shape, -1, dtype=np.int64)
    valid = (object_ids >= 0) & (object_ids < len(id_to_label))
    labels[valid] = id_to_label[object_ids[valid]]
    return labels


def map_object_ids_with_dict(object_ids: np.ndarray, object_to_label: Dict[int, int], default: int = 0) -> np.ndarray:
    labels = np.full(object_ids.shape, default, dtype=np.int64)
    for idx, object_id in enumerate(object_ids):
        labels[idx] = int(object_to_label.get(int(object_id), default))
    return labels


def load_gt_semantics(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    vertices, faces, face_object_ids = load_gt_semantic_mesh(args.gt_mesh)
    dataset_lower = args.dataset.lower()

    if dataset_lower == "replica":
        if args.semantic_info is None:
            raise ValueError("Replica evaluation requires --semantic_info")
        id_to_label, class_names = load_replica_id_to_label(args.semantic_info)
        face_labels = map_object_ids_to_labels(face_object_ids, id_to_label)
        return vertices, faces, face_labels, class_names

    if dataset_lower == "mp3d":
        missing = []
        for name in ("class_info", "semseg_json", "category_mapping"):
            if getattr(args, name) is None:
                missing.append("--" + name)
        if missing:
            raise ValueError("MP3D evaluation requires " + ", ".join(missing))
        class_names = load_mp3d_class_names(args.class_info)
        instance_to_mpcat40 = load_mp3d_instance_to_mpcat40(
            semseg_path=args.semseg_json,
            category_mapping_path=args.category_mapping,
            instance_map_path=args.instance_map,
        )
        face_labels = map_object_ids_with_dict(face_object_ids, instance_to_mpcat40, default=0)
        return vertices, faces, face_labels, class_names

    raise ValueError(f"Unsupported dataset: {args.dataset}")


def sample_labeled_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_labels: np.ndarray,
    num_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    tri = vertices[faces]
    edge_a = tri[:, 1] - tri[:, 0]
    edge_b = tri[:, 2] - tri[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
    valid_area = areas > 0
    if not np.any(valid_area):
        raise ValueError("GT mesh has no positive-area triangles")

    tri = tri[valid_area]
    face_labels = face_labels[valid_area]
    areas = areas[valid_area]
    probs = areas / areas.sum()

    rng = np.random.default_rng(seed)
    face_indices = rng.choice(len(tri), size=num_samples, replace=True, p=probs)
    chosen = tri[face_indices]

    r1 = rng.random(num_samples, dtype=np.float32)
    r2 = rng.random(num_samples, dtype=np.float32)
    sqrt_r1 = np.sqrt(r1)
    w0 = 1.0 - sqrt_r1
    w1 = sqrt_r1 * (1.0 - r2)
    w2 = sqrt_r1 * r2

    samples = (
        chosen[:, 0] * w0[:, None]
        + chosen[:, 1] * w1[:, None]
        + chosen[:, 2] * w2[:, None]
    ).astype(np.float32)
    labels = face_labels[face_indices].astype(np.int64)
    return samples, labels


def load_predicted_gaussians(
    params_path: Path,
    opacity_threshold: float,
    ignore_labels: Iterable[int],
) -> Tuple[np.ndarray, np.ndarray]:
    params = np.load(params_path)
    required_base = {"means3D", "logit_opacities"}
    missing_base = sorted(required_base.difference(params.files))
    if missing_base:
        raise ValueError(f"{params_path} is missing required keys: {missing_base}")

    xyz = params["means3D"].astype(np.float32)

    if "semantic_logits" in params.files and "seman_cls_ids" in params.files:
        semantic_logits = params["semantic_logits"]
        seman_cls_ids = params["seman_cls_ids"]
        top_idx = np.argmax(semantic_logits, axis=1)
        pred_labels = seman_cls_ids[np.arange(len(seman_cls_ids)), top_idx].astype(np.int64)
    elif "semantic_ids" in params.files:
        pred_labels = np.rint(params["semantic_ids"].reshape(-1)).astype(np.int64)
    else:
        raise ValueError(
            f"{params_path} is missing semantic labels. Expected either "
            "['semantic_logits', 'seman_cls_ids'] or ['semantic_ids']."
        )

    if len(pred_labels) != len(xyz):
        raise ValueError(
            f"Semantic label count ({len(pred_labels)}) does not match means3D count ({len(xyz)}) in {params_path}"
        )

    opacities = sigmoid(params["logit_opacities"].reshape(-1))
    finite = np.isfinite(xyz).all(axis=1)
    keep = finite & (opacities >= opacity_threshold)
    for label in ignore_labels:
        keep &= pred_labels != int(label)

    return xyz[keep], pred_labels[keep]


def query_in_chunks(tree: cKDTree, points: np.ndarray, chunk_size: int) -> Tuple[np.ndarray, np.ndarray]:
    distances = np.empty(len(points), dtype=np.float32)
    indices = np.empty(len(points), dtype=np.int64)
    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        d, i = tree.query(points[start:end], k=1, workers=-1)
        distances[start:end] = d.astype(np.float32)
        indices[start:end] = i.astype(np.int64)
    return distances, indices


def compute_class_metrics(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    class_ids: np.ndarray,
    class_names: Dict[int, str],
) -> Tuple[Dict[str, float], list]:
    rows = []
    ious = []
    accs = []
    supports = []

    for class_id in class_ids:
        gt_is_class = gt_labels == class_id
        pred_is_class = pred_labels == class_id
        tp = int(np.count_nonzero(gt_is_class & pred_is_class))
        fp = int(np.count_nonzero(~gt_is_class & pred_is_class))
        fn = int(np.count_nonzero(gt_is_class & ~pred_is_class))
        support = int(np.count_nonzero(gt_is_class))
        denom = tp + fp + fn
        iou = float(tp / denom) if denom else float("nan")
        acc = float(tp / support) if support else float("nan")
        if support > 0:
            ious.append(iou)
            accs.append(acc)
            supports.append(support)
        rows.append(
            {
                "class_id": int(class_id),
                "class_name": class_names.get(int(class_id), str(class_id)),
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "iou": iou,
                "acc": acc,
            }
        )

    total = len(gt_labels)
    correct = int(np.count_nonzero(gt_labels == pred_labels))
    ious_arr = np.asarray(ious, dtype=np.float64)
    accs_arr = np.asarray(accs, dtype=np.float64)
    supports_arr = np.asarray(supports, dtype=np.float64)
    metrics = {
        "gt_domain_overall_acc": float(correct / total) if total else float("nan"),
        "gt_domain_miou": float(np.nanmean(ious_arr)) if len(ious_arr) else float("nan"),
        "gt_domain_macc": float(np.nanmean(accs_arr)) if len(accs_arr) else float("nan"),
        "gt_domain_frequency_weighted_iou": float(np.nansum(ious_arr * supports_arr) / supports_arr.sum())
        if supports_arr.sum()
        else float("nan"),
        "num_eval_classes_present": int(len(ious_arr)),
        "num_valid_gt_samples": int(total),
    }
    return metrics, rows


def write_outputs(output_dir: Path, metrics: Dict[str, object], rows: list) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    fieldnames = ["class_id", "class_name", "support", "tp", "fp", "fn", "iou", "acc"]
    with (output_dir / "per_class_iou.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "summary.txt").open("w") as f:
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.6f}\n")
            else:
                f.write(f"{key}: {value}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate 3D semantic labels for Replica or MP3D SplaTAM outputs.")
    parser.add_argument("--dataset", choices=["Replica", "MP3D"], default="Replica")
    parser.add_argument("--scene", default=None, help="Scene name for metadata output")
    parser.add_argument("--params", type=Path, required=True, help="Path to SplaTAM params*.npz")
    parser.add_argument("--gt_mesh", type=Path, required=True, help="GT semantic mesh with face object_id")
    parser.add_argument("--semantic_info", type=Path, default=None, help="Replica habitat/info_semantic.json")
    parser.add_argument("--class_info", type=Path, default=None, help="MP3D class_info_file.json")
    parser.add_argument("--semseg_json", type=Path, default=None, help="MP3D house_segmentations/<scene>.semseg.json")
    parser.add_argument("--category_mapping", type=Path, default=None, help="MP3D category_mapping.tsv")
    parser.add_argument("--instance_map", type=Path, default=None, help="Optional MP3D instance_to_mpcat40.json")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for metrics")
    parser.add_argument("--num_samples", type=int, default=200000, help="Area-weighted GT surface samples")
    parser.add_argument("--distance_threshold", type=float, default=0.05, help="Match threshold in meters")
    parser.add_argument("--opacity_threshold", type=float, default=0.05, help="Minimum Gaussian opacity")
    parser.add_argument(
        "--transform_traj",
        type=Path,
        default=None,
        help="traj.txt used to transform SplaTAM local points into GT world coordinates",
    )
    parser.add_argument("--ignore_labels", type=int, nargs="*", default=[-1, 0], help="Semantic label ids to ignore")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=500000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ignore_labels = set(int(i) for i in args.ignore_labels)

    vertices, faces, face_labels, class_names = load_gt_semantics(args)
    gt_points, gt_labels = sample_labeled_surface(vertices, faces, face_labels, args.num_samples, args.seed)

    gt_valid = np.ones(len(gt_labels), dtype=bool)
    for label in ignore_labels:
        gt_valid &= gt_labels != int(label)
    gt_points = gt_points[gt_valid]
    gt_labels = gt_labels[gt_valid]

    pred_points, pred_labels = load_predicted_gaussians(args.params, args.opacity_threshold, ignore_labels)
    if args.transform_traj is not None:
        pred_points = transform_points(load_slam_to_world(args.dataset, args.transform_traj), pred_points)
    if len(pred_points) == 0:
        raise ValueError("No predicted points remain after filtering")
    if len(gt_points) == 0:
        raise ValueError("No GT points remain after filtering")

    pred_tree = cKDTree(pred_points)
    gt_to_pred_dist, gt_to_pred_idx = query_in_chunks(pred_tree, gt_points, args.chunk_size)
    gt_matched = gt_to_pred_dist <= args.distance_threshold
    gt_domain_pred = np.full(len(gt_points), -1, dtype=np.int64)
    gt_domain_pred[gt_matched] = pred_labels[gt_to_pred_idx[gt_matched]]

    class_ids = np.asarray(sorted(set(class_names.keys()) - ignore_labels), dtype=np.int64)
    metrics, rows = compute_class_metrics(gt_labels, gt_domain_pred, class_ids, class_names)

    semantic_recall_hits = gt_matched & (gt_domain_pred == gt_labels)
    metrics.update(
        {
            "dataset": args.dataset,
            "scene": args.scene or "",
            "distance_threshold": float(args.distance_threshold),
            "opacity_threshold": float(args.opacity_threshold),
            "num_gt_samples_requested": int(args.num_samples),
            "num_pred_points_after_filter": int(len(pred_points)),
            "gt_geometry_recall_at_threshold": float(np.mean(gt_matched)) if len(gt_matched) else float("nan"),
            "gt_semantic_recall_at_threshold": float(np.mean(semantic_recall_hits))
            if len(semantic_recall_hits)
            else float("nan"),
        }
    )

    gt_tree = cKDTree(gt_points)
    pred_to_gt_dist, pred_to_gt_idx = query_in_chunks(gt_tree, pred_points, args.chunk_size)
    pred_close = pred_to_gt_dist <= args.distance_threshold
    semantic_precision_hits = pred_close & (pred_labels == gt_labels[pred_to_gt_idx])
    semantic_precision = float(np.mean(semantic_precision_hits)) if len(semantic_precision_hits) else float("nan")
    semantic_recall = metrics["gt_semantic_recall_at_threshold"]
    if semantic_precision + semantic_recall > 0:
        semantic_fscore = 2.0 * semantic_precision * semantic_recall / (semantic_precision + semantic_recall)
    else:
        semantic_fscore = float("nan")

    metrics.update(
        {
            "pred_geometry_precision_at_threshold": float(np.mean(pred_close)) if len(pred_close) else float("nan"),
            "pred_semantic_precision_at_threshold": semantic_precision,
            "semantic_fscore_at_threshold": semantic_fscore,
        }
    )

    write_outputs(args.output_dir, metrics, rows)

    print(f"Wrote 3D semantic evaluation to {args.output_dir}")
    print(f"mIoU: {metrics['gt_domain_miou']:.4f}")
    print(f"mAcc: {metrics['gt_domain_macc']:.4f}")
    print(f"Semantic precision@{args.distance_threshold}: {semantic_precision:.4f}")
    print(f"Semantic recall@{args.distance_threshold}: {semantic_recall:.4f}")
    print(f"Semantic F-score@{args.distance_threshold}: {semantic_fscore:.4f}")


if __name__ == "__main__":
    main()
