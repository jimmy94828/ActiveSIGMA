"""Estimate the transform between Replica GT mesh and reconstructed Gaussian PLY.

This tool is intended for pairs such as:
  - target: `data/Replica/<scene>_mesh.ply`
  - source: `results/Replica/<scene>/<method>/run_<id><ver>/<slam>/rgb_GS_1000.ply`

Workflow:
1. Use the first pose in `data/Replica/<scene>/traj.txt` as the initial
   SLAM-world -> Replica-sim transform.
2. Optionally refine with point-to-point ICP on sampled point clouds.
3. Save both the initial and final 4x4 matrices as JSON.

Example:
  python tools/helper/estimate_replica_ply_transform.py \
    --scene office4 --method SemanticHeat --run-version _v6 --use-icp
"""

import argparse
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import trimesh
from plyfile import PlyData, PlyElement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate Replica GT mesh <-> Gaussian PLY transform."
    )
    parser.add_argument("--scene", default="office4", help="Replica scene name")
    parser.add_argument("--method", default="SemanticHeat", help="experiment method name")
    parser.add_argument("--run-id", type=int, default=0, help="run index")
    parser.add_argument("--run-version", default="_v6", help="run version suffix, e.g. _v6")
    parser.add_argument("--slam", default="splatam", help="SLAM subdirectory name")
    parser.add_argument("--source-ply", default=None, help="source Gaussian PLY path")
    parser.add_argument("--target-ply", default=None, help="target GT mesh/PLY path")
    parser.add_argument("--traj-txt", default=None, help="Replica trajectory txt path")
    parser.add_argument("--sample-points", type=int, default=200000, help="max sampled points per cloud")
    parser.add_argument("--seed", type=int, default=0, help="random seed for sampling")
    parser.add_argument("--use-icp", action="store_true", help="refine initial transform with ICP")
    parser.add_argument("--icp-threshold", type=float, default=0.10, help="ICP correspondence threshold in meters")
    parser.add_argument("--save-json", default=None, help="optional output JSON path")
    parser.add_argument("--save-aligned-ply", default=None, help="optional aligned source point cloud path")
    return parser.parse_args()


def default_paths(args: argparse.Namespace) -> Tuple[str, str, str]:
    project_dir = os.getcwd()
    run_name = f"run_{args.run_id}{args.run_version}"
    source_ply = args.source_ply or os.path.join(
        project_dir,
        "results",
        "Replica",
        args.scene,
        args.method,
        run_name,
        args.slam,
        "rgb_GS_1000.ply",
    )
    target_ply = args.target_ply or os.path.join(
        project_dir,
        "data",
        "Replica",
        f"{args.scene}_mesh.ply",
    )
    traj_txt = args.traj_txt or os.path.join(
        project_dir,
        "data",
        "Replica",
        args.scene,
        "traj.txt",
    )
    return source_ply, target_ply, traj_txt


def ensure_exists(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def load_first_pose(traj_txt: str) -> np.ndarray:
    with open(traj_txt, "r", encoding="utf-8") as f:
        line = f.readline().strip()
    values = np.fromstring(line, sep=" ", dtype=np.float32)
    if values.size != 16:
        raise ValueError(f"Expected 16 values in first pose, got {values.size}: {traj_txt}")
    return values.reshape(4, 4)


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    points_h = np.concatenate([points, ones], axis=1)
    transformed = (transform @ points_h.T).T
    return transformed[:, :3].astype(np.float32)


def random_sample(points: np.ndarray, sample_points: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] <= sample_points:
        return points.astype(np.float32)
    indices = rng.choice(points.shape[0], size=sample_points, replace=False)
    return points[indices].astype(np.float32)


def concat_scene(scene: trimesh.Scene) -> trimesh.Trimesh:
    if len(scene.geometry) == 0:
        raise ValueError("Empty trimesh scene")
    return trimesh.util.concatenate(
        tuple(
            trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
            for g in scene.geometry.values()
        )
    )


def load_target_points(path: str, sample_points: int, rng: np.random.Generator) -> np.ndarray:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = concat_scene(mesh)

    if isinstance(mesh, trimesh.Trimesh) and mesh.faces is not None and len(mesh.faces) > 0:
        sampled, _ = trimesh.sample.sample_surface_even(mesh, sample_points)
        return sampled.astype(np.float32)

    if hasattr(mesh, "vertices") and len(mesh.vertices) > 0:
        return random_sample(np.asarray(mesh.vertices), sample_points, rng)

    raise ValueError(f"Unsupported target geometry for sampling: {path}")


def load_source_points(path: str, sample_points: int, rng: np.random.Generator) -> np.ndarray:
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    points = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float32)
    return random_sample(points, sample_points, rng)


def points_to_open3d(points: np.ndarray):
    import open3d as o3d

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return point_cloud


def run_icp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    threshold: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    import open3d as o3d

    source = points_to_open3d(source_points)
    target = points_to_open3d(target_points)
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        np.eye(4, dtype=np.float64),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    stats = {
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
    }
    return np.asarray(result.transformation, dtype=np.float32), stats


def save_point_cloud_ply(path: str, points: np.ndarray) -> None:
    vertex = np.empty(points.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    PlyData([PlyElement.describe(vertex, "vertex")]).write(path)


def bounds(points: np.ndarray) -> Dict[str, list]:
    return {
        "min": points.min(axis=0).astype(float).tolist(),
        "max": points.max(axis=0).astype(float).tolist(),
        "mean": points.mean(axis=0).astype(float).tolist(),
    }


def main() -> None:
    args = parse_args()
    source_ply, target_ply, traj_txt = default_paths(args)
    ensure_exists(source_ply, "source_ply")
    ensure_exists(target_ply, "target_ply")
    ensure_exists(traj_txt, "traj_txt")

    rng = np.random.default_rng(args.seed)
    source_points = load_source_points(source_ply, args.sample_points, rng)
    target_points = load_target_points(target_ply, args.sample_points, rng)

    initial_transform = load_first_pose(traj_txt)
    source_points_initial = apply_transform(source_points, initial_transform)

    icp_transform = np.eye(4, dtype=np.float32)
    icp_stats: Dict[str, float] = {}
    final_transform = initial_transform.copy()
    aligned_points = source_points_initial

    if args.use_icp:
        icp_transform, icp_stats = run_icp(
            source_points_initial,
            target_points,
            threshold=args.icp_threshold,
        )
        final_transform = icp_transform @ initial_transform
        aligned_points = apply_transform(source_points, final_transform)

    result = {
        "scene": args.scene,
        "source_ply": source_ply,
        "target_ply": target_ply,
        "traj_txt": traj_txt,
        "sample_points": int(args.sample_points),
        "initial_transform": initial_transform.astype(float).tolist(),
        "icp_transform": icp_transform.astype(float).tolist(),
        "final_transform": final_transform.astype(float).tolist(),
        "source_bounds_slam": bounds(source_points),
        "source_bounds_after_initial": bounds(source_points_initial),
        "target_bounds": bounds(target_points),
    }
    if icp_stats:
        result["icp_stats"] = icp_stats

    print(json.dumps(result, indent=2))

    if args.save_json:
        save_dir = os.path.dirname(args.save_json)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    if args.save_aligned_ply:
        save_dir = os.path.dirname(args.save_aligned_ply)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        save_point_cloud_ply(args.save_aligned_ply, aligned_points)


if __name__ == "__main__":
    main()
