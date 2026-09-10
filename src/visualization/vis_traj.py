import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def load_scene_from_ply(
    semantic_ply: str,
    color_fields: Optional[str] = None,
    max_points: int = 200000,
    scene_flip_yz: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from plyfile import PlyData  # type: ignore
    except Exception as e:
        raise ImportError("Please install plyfile: pip install plyfile") from e

    ply_path = Path(semantic_ply)
    if not ply_path.exists():
        raise FileNotFoundError(f"Scene PLY not found: {ply_path}")

    ply = PlyData.read(str(ply_path))
    if "vertex" not in ply:
        raise ValueError(f"PLY has no 'vertex' element: {ply_path}")

    v = ply["vertex"].data
    names = list(v.dtype.names or [])
    lower_map = {n.lower(): n for n in names}

    for k in ["x", "y", "z"]:
        if k not in lower_map:
            raise ValueError(f"PLY vertex missing '{k}' field. fields={names}")

    xyz = np.stack(
        [
            np.asarray(v[lower_map["x"]], dtype=np.float32),
            np.asarray(v[lower_map["y"]], dtype=np.float32),
            np.asarray(v[lower_map["z"]], dtype=np.float32),
        ],
        axis=1,
    )

    if scene_flip_yz:
        xyz[:, 1] *= -1.0
        xyz[:, 2] *= -1.0

    if color_fields is not None:
        req = [s.strip() for s in color_fields.split(",") if s.strip()]
        if len(req) != 3:
            raise ValueError("--scene_color_fields must be 3 fields, e.g. f_dc_0,f_dc_1,f_dc_2")
        keys = []
        for r in req:
            key = lower_map.get(r.lower())
            if key is None:
                raise ValueError(f"field '{r}' not found. fields={names}")
            keys.append(key)
        rgb = np.stack([np.asarray(v[k], dtype=float) for k in keys], axis=1)
        rgb = _normalize_rgb(rgb)
    elif all(k in lower_map for k in ["red", "green", "blue"]):
        rgb = np.stack(
            [
                np.asarray(v[lower_map["red"]], dtype=float),
                np.asarray(v[lower_map["green"]], dtype=float),
                np.asarray(v[lower_map["blue"]], dtype=float),
            ],
            axis=1,
        )
        if np.nanmax(rgb) > 1.0:
            rgb = np.clip(rgb / 255.0, 0.0, 1.0)
    elif all(k in lower_map for k in ["f_dc_0", "f_dc_1", "f_dc_2"]):
        rgb = np.stack(
            [
                np.asarray(v[lower_map["f_dc_0"]], dtype=float),
                np.asarray(v[lower_map["f_dc_1"]], dtype=float),
                np.asarray(v[lower_map["f_dc_2"]], dtype=float),
            ],
            axis=1,
        )
        rgb = _normalize_rgb(rgb)
    else:
        rgb = np.full((xyz.shape[0], 3), 0.7, dtype=float)

    if max_points is not None and xyz.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(xyz.shape[0], size=max_points, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]

    return xyz, rgb


def visualize_scene_with_trajectory(
    scene_xyz: np.ndarray,
    scene_rgb: np.ndarray,
    positions: np.ndarray,
    out_path: str,
    title: str = "Scene + Trajectory",
    scene_point_size: float = 0.2,
    scene_alpha: float = 0.7,
) -> None:
    xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        scene_xyz[:, 0], scene_xyz[:, 1], scene_xyz[:, 2],
        c=scene_rgb, s=scene_point_size, alpha=scene_alpha, linewidths=0
    )

    ax.plot(xs, ys, zs, linewidth=2.0, color="yellow", label="trajectory")
    ax.scatter(xs[0], ys[0], zs[0], s=80, marker="o", color="red", label="start")
    ax.scatter(xs[-1], ys[-1], zs[-1], s=80, marker="^", color="cyan", label="end")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    all_xyz = np.vstack([scene_xyz, positions])
    mins = all_xyz.min(axis=0)
    maxs = all_xyz.max(axis=0)
    center = (mins + maxs) * 0.5
    max_range = np.max(maxs - mins)
    if max_range == 0:
        max_range = 1.0
    half = max_range * 0.5

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=240)
    plt.close(fig)

def _to_4x4(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose)
    if pose.shape == (4, 4):
        return pose.astype(np.float64)
    if pose.shape == (3, 4):
        T = np.eye(4, dtype=np.float64)
        T[:3, :4] = pose
        return T
    raise ValueError(f"Pose is not matrix form (3x4/4x4): {pose.shape}")


def _apply_flip_yz_to_translation(t: np.ndarray, flip_yz: bool) -> np.ndarray:
    if not flip_yz:
        return t
    out = t.copy()
    out[1] *= -1.0
    out[2] *= -1.0
    return out


def extract_translation(
    pose: np.ndarray,
    invert_pose: bool = False,
    flip_yz: bool = False,
    traj_scale: float = 1.0,
) -> np.ndarray:
    pose = np.asarray(pose)

    # matrix pose
    if pose.shape in [(4, 4), (3, 4)]:
        T = _to_4x4(pose)
        if invert_pose:
            T = np.linalg.inv(T)
        t = T[:3, 3].astype(np.float64)
        t = _apply_flip_yz_to_translation(t, flip_yz)
        return t * float(traj_scale)

    # vector pose
    if pose.ndim == 1 and pose.shape[0] >= 3:
        t = pose[:3].astype(np.float64)
        t = _apply_flip_yz_to_translation(t, flip_yz)
        return t * float(traj_scale)

    raise ValueError(f"Unsupported pose shape: {pose.shape}")


def extract_semantic_value(sem: np.ndarray) -> float:
    sem = np.asarray(sem)

    if sem.ndim == 0:
        return float(sem)
    if sem.ndim == 1:
        if sem.size == 0:
            raise ValueError("Empty semantic array.")
        if sem.size == 1:
            return float(sem[0])
        return float(np.argmax(sem))
    return float(np.argmax(sem.reshape(-1)))


def load_positions(
    pose_dir: str,
    invert_pose: bool = False,
    flip_yz: bool = False,
    traj_scale: float = 1.0,
) -> Tuple[np.ndarray, List[Path]]:
    pose_dir = Path(pose_dir)
    if not pose_dir.exists() or not pose_dir.is_dir():
        raise FileNotFoundError(f"Pose directory not found: {pose_dir}")

    npy_files: List[Path] = sorted(pose_dir.glob("*.npy"), key=natural_key)
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in: {pose_dir}")

    positions = []
    for f in npy_files:
        pose = np.load(str(f), allow_pickle=False)
        t = extract_translation(
            pose,
            invert_pose=invert_pose,
            flip_yz=flip_yz,
            traj_scale=traj_scale,
        )
        positions.append(t)

    return np.stack(positions, axis=0), npy_files


def load_semantics_for_poses(pose_files: List[Path], semantic_dir: str) -> np.ndarray:
    semantic_dir = Path(semantic_dir)
    if not semantic_dir.exists() or not semantic_dir.is_dir():
        raise FileNotFoundError(f"Semantic directory not found: {semantic_dir}")

    sem_values = []
    missing = []
    for pf in pose_files:
        sf = semantic_dir / pf.name
        if not sf.exists():
            missing.append(sf.name)
            sem_values.append(np.nan)
            continue
        sem = np.load(str(sf), allow_pickle=False)
        sem_values.append(extract_semantic_value(sem))

    if missing:
        print(f"[WARN] Missing semantic files: {len(missing)} (will be ignored in color overlay)")

    return np.array(sem_values, dtype=float)


def _nearest_neighbor_values(
    query_xyz: np.ndarray,
    cloud_xyz: np.ndarray,
    cloud_values: np.ndarray,
    max_nn_dist: Optional[float] = None,
) -> np.ndarray:
    # 優先用 scipy cKDTree
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(cloud_xyz)
        dists, idxs = tree.query(query_xyz, k=1)
        out = cloud_values[idxs].astype(float)
        if max_nn_dist is not None:
            out[dists > max_nn_dist] = np.nan
        return out
    except Exception:
        # fallback: numpy brute force (資料很大會慢)
        out = np.empty((query_xyz.shape[0],), dtype=float)
        for i, q in enumerate(query_xyz):
            d2 = np.sum((cloud_xyz - q[None, :]) ** 2, axis=1)
            j = int(np.argmin(d2))
            dist = float(np.sqrt(d2[j]))
            out[i] = float(cloud_values[j])
            if max_nn_dist is not None and dist > max_nn_dist:
                out[i] = np.nan
        return out


def _nearest_neighbor_indices(
    query_xyz: np.ndarray,
    cloud_xyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.spatial import cKDTree  # type: ignore
        tree = cKDTree(cloud_xyz)
        dists, idxs = tree.query(query_xyz, k=1)
        return idxs.astype(np.int64), dists.astype(float)
    except Exception:
        idxs = np.empty((query_xyz.shape[0],), dtype=np.int64)
        dists = np.empty((query_xyz.shape[0],), dtype=float)
        for i, q in enumerate(query_xyz):
            d2 = np.sum((cloud_xyz - q[None, :]) ** 2, axis=1)
            j = int(np.argmin(d2))
            idxs[i] = j
            dists[i] = float(np.sqrt(d2[j]))
        return idxs, dists


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    lo = np.percentile(rgb, 1, axis=0)
    hi = np.percentile(rgb, 99, axis=0)
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def load_semantic_from_single_ply(
    positions: np.ndarray,
    semantic_ply: str,
    semantic_field: Optional[str] = None,
    max_nn_dist: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    回傳:
      - semantic_values: (N,) 連續值/類別值
      - semantic_rgb:    (N,3) RGB
    兩者只會有一個非 None
    """
    try:
        from plyfile import PlyData  # type: ignore
    except Exception as e:
        raise ImportError("Please install plyfile: pip install plyfile") from e

    ply_path = Path(semantic_ply)
    if not ply_path.exists():
        raise FileNotFoundError(f"Semantic PLY not found: {ply_path}")

    ply = PlyData.read(str(ply_path))
    if "vertex" not in ply:
        raise ValueError(f"PLY has no 'vertex' element: {ply_path}")

    v = ply["vertex"].data
    names = list(v.dtype.names or [])
    lower_map = {n.lower(): n for n in names}

    for k in ["x", "y", "z"]:
        if k not in lower_map:
            raise ValueError(f"PLY vertex missing '{k}' field. fields={names}")

    xyz = np.stack(
        [
            np.asarray(v[lower_map["x"]], dtype=np.float32),
            np.asarray(v[lower_map["y"]], dtype=np.float32),
            np.asarray(v[lower_map["z"]], dtype=np.float32),
        ],
        axis=1,
    )

    idxs, dists = _nearest_neighbor_indices(positions, xyz)
    invalid = (dists > max_nn_dist) if (max_nn_dist is not None) else np.zeros_like(dists, dtype=bool)

    # 1) 使用者指定欄位
    if semantic_field is not None:
        req = [s.strip() for s in semantic_field.split(",") if s.strip()]
        if len(req) == 1:
            key = lower_map.get(req[0].lower())
            if key is None:
                raise ValueError(f"semantic_field='{semantic_field}' not found. fields={names}")
            vals = np.asarray(v[key], dtype=float)[idxs]
            vals[invalid] = np.nan
            print(f"[INFO] semantic field from PLY: {key}")
            return vals, None

        if len(req) == 3:
            keys = []
            for r in req:
                k = lower_map.get(r.lower())
                if k is None:
                    raise ValueError(f"semantic_field='{r}' not found. fields={names}")
                keys.append(k)
            rgb_cloud = np.stack([np.asarray(v[k], dtype=float) for k in keys], axis=1)
            rgb_cloud = _normalize_rgb(rgb_cloud)
            rgb = rgb_cloud[idxs]
            rgb[invalid] = np.nan
            print(f"[INFO] RGB fields from PLY: {keys}")
            return None, rgb

        raise ValueError("semantic_field must be 1 field or 3 fields separated by comma.")

    # 2) 自動模式：優先 f_dc_0/1/2
    if all(k in lower_map for k in ["f_dc_0", "f_dc_1", "f_dc_2"]):
        rgb_cloud = np.stack(
            [
                np.asarray(v[lower_map["f_dc_0"]], dtype=float),
                np.asarray(v[lower_map["f_dc_1"]], dtype=float),
                np.asarray(v[lower_map["f_dc_2"]], dtype=float),
            ],
            axis=1,
        )
        rgb_cloud = _normalize_rgb(rgb_cloud)
        rgb = rgb_cloud[idxs]
        rgb[invalid] = np.nan
        print("[INFO] Auto color by fields: f_dc_0,f_dc_1,f_dc_2")
        return None, rgb

    # 3) 後備：單一欄位
    for c in ["opacity", "scale_0", "scale_1", "scale_2"]:
        if c in lower_map:
            vals = np.asarray(v[lower_map[c]], dtype=float)[idxs]
            vals[invalid] = np.nan
            print(f"[INFO] Auto semantic field from PLY: {c}")
            return vals, None

    raise ValueError(f"No usable semantic/color fields found in PLY. fields={names}")


def visualize_trajectory(
    positions: np.ndarray,
    out_path: str,
    title: str = "Trajectory",
    semantic_values: Optional[np.ndarray] = None,
    semantic_rgb: Optional[np.ndarray] = None,
    cmap: str = "viridis",
) -> None:
    xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(xs, ys, zs, linewidth=1.5, color="lightgray", label="trajectory")

    if semantic_rgb is not None:
        valid = ~np.isnan(semantic_rgb).any(axis=1)
        if np.any(valid):
            ax.scatter(xs[valid], ys[valid], zs[valid], c=semantic_rgb[valid], s=18, label="semantic(rgb)")
        else:
            print("[WARN] No valid RGB semantic values to render.")
    elif semantic_values is not None:
        valid = ~np.isnan(semantic_values)
        if np.any(valid):
            sc = ax.scatter(xs[valid], ys[valid], zs[valid], c=semantic_values[valid], cmap=cmap, s=18, label="semantic")
            cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.8)
            cbar.set_label("semantic value / class id")
        else:
            print("[WARN] No valid semantic values to render.")

    ax.scatter(xs[0], ys[0], zs[0], s=70, marker="o", color="red", label="start")
    ax.scatter(xs[-1], ys[-1], zs[-1], s=70, marker="^", color="blue", label="end")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    max_range = np.array([xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()]).max()
    if max_range == 0:
        max_range = 1.0
    mid_x = (xs.max() + xs.min()) * 0.5
    mid_y = (ys.max() + ys.min()) * 0.5
    mid_z = (zs.max() + zs.min()) * 0.5
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=220)
    plt.close(fig)

"""
Usage:
python src/visualization/vis_traj.py \
  --pose_dir results/Replica/office2/SemanticHeat/run_0_v2/visualization/pose \
  --semantic_ply results/Replica/office2/SemanticHeat/run_0_v2/splatam/semantic_GS_0900.ply \
  --semantic_field label \
  --out results/Replica/office2/SemanticHeat/run_0_v2/visualization/trajectory_semantic.png
"""

"""
python /media/phudh/X9_Pro/undergraduate/ActiveMapping/src/visualization/vis_traj.py \
  --pose_dir results/Replica/office2/SemanticHeat/run_0_v2/visualization/pose \
  --semantic_ply results/Replica/office2/SemanticHeat/run_0_v2/splatam/semantic_GS_0900.ply \
  --scene_color_fields f_dc_0,f_dc_1,f_dc_2 \
  --invert_pose \
  --out results/Replica/office2/SemanticHeat/run_0_v2/visualization/trajectory_semantic.png
"""

"""
python /media/phudh/X9_Pro/undergraduate/ActiveMapping/src/visualization/vis_traj.py \
  --pose_dir /media/phudh/X9_Pro/undergraduate/ActiveMapping/results/Replica/office2/SemanticHeat/run_0_v2/visualization/pose \
  --semantic_ply /media/phudh/X9_Pro/undergraduate/ActiveMapping/results/Replica/office2/SemanticHeat/run_0_v2/splatam/semantic_GS_0900.ply \
  --scene_color_fields f_dc_0,f_dc_1,f_dc_2 \
  --auto_search \
  --out /media/phudh/X9_Pro/undergraduate/ActiveMapping/results/Replica/office2/SemanticHeat/run_0_v2/visualization/scene_with_traj_auto.png
"""

def _alignment_score(positions: np.ndarray, scene_xyz: np.ndarray) -> Tuple[float, float]:
    # 用 trajectory 到 scene 的最近鄰距離評分（越小越好）
    if positions.shape[0] > 3000:
        idx = np.linspace(0, positions.shape[0] - 1, 3000).astype(np.int64)
        q = positions[idx]
    else:
        q = positions
    _, dists = _nearest_neighbor_indices(q, scene_xyz)
    return float(np.median(dists)), float(np.mean(dists))


def auto_search_alignment(
    pose_dir: str,
    semantic_ply: str,
    scene_color_fields: Optional[str],
    scene_sample: int,
    scale_candidates: List[float],
) -> Tuple[bool, bool, bool, float]:
    best = None

    for scene_flip in [False, True]:
        scene_xyz, _ = load_scene_from_ply(
            semantic_ply=semantic_ply,
            color_fields=scene_color_fields,
            max_points=scene_sample,
            scene_flip_yz=scene_flip,
        )

        for invert_pose in [False, True]:
            for flip_yz in [False, True]:
                for scale in scale_candidates:
                    positions, _ = load_positions(
                        pose_dir=pose_dir,
                        invert_pose=invert_pose,
                        flip_yz=flip_yz,
                        traj_scale=scale,
                    )
                    med, mean = _alignment_score(positions, scene_xyz)
                    cur = (med, mean, invert_pose, flip_yz, scene_flip, scale)
                    if best is None or cur[:2] < best[:2]:
                        best = cur

    assert best is not None
    med, mean, invert_pose, flip_yz, scene_flip, scale = best
    print(
        f"[AUTO] best: median={med:.4f}, mean={mean:.4f}, "
        f"invert_pose={invert_pose}, flip_yz={flip_yz}, scene_flip_yz={scene_flip}, traj_scale={scale}"
    )
    return invert_pose, flip_yz, scene_flip, scale


def main():
    parser = argparse.ArgumentParser(description="Visualize trajectory and overlay semantic values")
    parser.add_argument("--pose_dir", type=str, required=True, help="內含 pose .npy 的資料夾")
    parser.add_argument("--semantic_dir", type=str, default=None, help="(舊模式) 內含 semantic .npy 的資料夾")
    parser.add_argument("--semantic_ply", type=str, default=None, help="(新模式) 單一 semantic .ply 檔案")
    parser.add_argument("--semantic_field", type=str, default=None, help="PLY 裡的 semantic 欄位名（如 label/class）")
    parser.add_argument("--max_nn_dist", type=float, default=None, help="最近鄰最大距離，超過則設為 NaN")
    parser.add_argument("--scene_color_fields", type=str, default=None, help="場景顏色欄位，例: f_dc_0,f_dc_1,f_dc_2")
    parser.add_argument("--scene_sample", type=int, default=200000, help="場景點雲最多取樣點數")
    parser.add_argument("--scene_point_size", type=float, default=0.2, help="場景點大小")
    parser.add_argument("--scene_alpha", type=float, default=0.7, help="場景透明度")
    parser.add_argument("--out", type=str, default="trajectory.png", help="輸出圖片路徑")
    parser.add_argument("--title", type=str, default="Trajectory + Semantic", help="圖標題")
    parser.add_argument("--cmap", type=str, default="viridis", help="semantic colormap")
    parser.add_argument("--invert_pose", action="store_true", help="把每個矩陣 pose 取反矩陣 (常用於 Tcw -> Twc)")
    parser.add_argument("--flip_yz", action="store_true", help="對 trajectory 做 y,z 取負 (OpenGL/Replica 座標修正常用)")
    parser.add_argument("--scene_flip_yz", action="store_true", help="對 scene ply 做 y,z 取負")
    parser.add_argument("--traj_scale", type=float, default=1.0, help="軌跡尺度倍率 (單眼SLAM可調)")
    parser.add_argument("--auto_search", action="store_true", help="自動搜尋最佳 invert/flip/scale 組合")
    parser.add_argument(
        "--scale_candidates",
        type=str,
        default="0.1,0.2,0.5,1,2,5,10",
        help="auto_search 用的 scale 候選，逗號分隔",
    )
    args = parser.parse_args()

    if args.auto_search and args.semantic_ply is not None:
        scales = [float(x.strip()) for x in args.scale_candidates.split(",") if x.strip()]
        inv, ftraj, fscene, s = auto_search_alignment(
            pose_dir=args.pose_dir,
            semantic_ply=args.semantic_ply,
            scene_color_fields=args.scene_color_fields,
            scene_sample=args.scene_sample,
            scale_candidates=scales,
        )
        args.invert_pose = inv
        args.flip_yz = ftraj
        args.scene_flip_yz = fscene
        args.traj_scale = s

    positions, pose_files = load_positions(
        args.pose_dir,
        invert_pose=args.invert_pose,
        flip_yz=args.flip_yz,
        traj_scale=args.traj_scale,
    )

    if args.semantic_ply is not None:
        scene_xyz, scene_rgb = load_scene_from_ply(
            semantic_ply=args.semantic_ply,
            color_fields=args.scene_color_fields,
            max_points=args.scene_sample,
            scene_flip_yz=args.scene_flip_yz,
        )
        visualize_scene_with_trajectory(
            scene_xyz=scene_xyz,
            scene_rgb=scene_rgb,
            positions=positions,
            out_path=args.out,
            title=args.title,
            scene_point_size=args.scene_point_size,
            scene_alpha=args.scene_alpha,
        )
        print(f"[OK] Saved scene+trajectory image to: {args.out}")
        return

    # 舊模式保留
    sem_values = None
    sem_rgb = None
    if args.semantic_dir is not None:
        sem_values = load_semantics_for_poses(pose_files, args.semantic_dir)

    visualize_trajectory(
        positions=positions,
        out_path=args.out,
        title=args.title,
        semantic_values=sem_values,
        semantic_rgb=sem_rgb,
        cmap=args.cmap,
    )
    print(f"[OK] Saved trajectory image to: {args.out}")


if __name__ == "__main__":
    main()