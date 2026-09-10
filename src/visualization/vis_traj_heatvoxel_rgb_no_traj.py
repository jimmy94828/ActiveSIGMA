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
import typing

import numpy as np


sys.path.append(os.getcwd())

C0 = 0.28209479177387814

# Not using WebRTC server in this script — use local Open3D visualizer instead


def argument_parsing() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize reconstructed RGB point cloud without trajectory overlays."
    )
    parser.add_argument("--mesh_file", type=str, default="",
                        help="unused; kept for backward compatibility")
    parser.add_argument("--traj_file", type=str, default="",
                        help="params.npz file used to load reconstructed RGB point cloud")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="unused; kept for backward compatibility")
    parser.add_argument("--with_interact", type=int, default=1,
                        help="with interaction for visualization")
    parser.add_argument("--scene", type=str, default="room2",
                        help="scene name used for default paths and z threshold preset")
    parser.add_argument("--dataset", type=str, default="Replica",
                        help="dataset name used for default paths")
    parser.add_argument("--method", type=str, default="SemanticHeat",
                        help="method name used for default paths")
    parser.add_argument("--run_version", type=str, default="run_0",
                        help="run directory name")
    parser.add_argument("--slam", type=str, default="splatam",
                        help="SLAM output directory name")
    parser.add_argument("--traj_txt", type=str, default="",
                        help="simulation trajectory txt used as the coordinate anchor")
    parser.add_argument("--z_thres", type=float, default=None,
                        help="keep points with z <= z_thres; default follows the scene preset")
    parser.add_argument("--point_size", type=float, default=3.2,
                        help="Open3D point size")
    args = parser.parse_args()
    return args


def import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("Open3D is required for this visualizer. Install open3d first.") from exc
    return o3d


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return (points @ rotation.T + translation).astype(np.float32)


def decode_rgb_values(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors)
    if colors.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if np.issubdtype(colors.dtype, np.floating):
        return np.clip(colors.astype(np.float32), 0.0, 1.0)
    return np.clip(colors.astype(np.float32) / 255.0, 0.0, 1.0)


def decode_rgb_from_fdc(f_dc: np.ndarray) -> np.ndarray:
    f_dc = np.asarray(f_dc, dtype=np.float32)
    return decode_rgb_values(f_dc[:, :3] * C0 + 0.5)


def load_rgb_points(npz_path: str) -> typing.Tuple[np.ndarray, np.ndarray, str]:
    params = np.load(npz_path, allow_pickle=True)
    if "means3D" in params:
        points = np.asarray(params["means3D"], dtype=np.float32)
    elif "points" in params:
        points = np.asarray(params["points"], dtype=np.float32)
    else:
        raise ValueError(f"No `means3D` or `points` found in {npz_path}")

    color_source = None
    if "rgb_colors" in params:
        colors = decode_rgb_values(np.asarray(params["rgb_colors"], dtype=np.float32))
        color_source = "rgb_colors"
    elif "features_dc" in params:
        f_dc = np.asarray(params["features_dc"], dtype=np.float32).reshape(points.shape[0], -1)
        if f_dc.shape[1] < 3:
            raise ValueError(f"`features_dc` must have at least 3 channels, got {f_dc.shape[1]}")
        colors = decode_rgb_from_fdc(f_dc)
        color_source = "features_dc (SH DC -> RGB)"
    elif all(key in params for key in ("f_dc_0", "f_dc_1", "f_dc_2")):
        f_dc = np.stack(
            [
                np.asarray(params["f_dc_0"], dtype=np.float32),
                np.asarray(params["f_dc_1"], dtype=np.float32),
                np.asarray(params["f_dc_2"], dtype=np.float32),
            ],
            axis=1,
        )
        colors = decode_rgb_from_fdc(f_dc)
        color_source = "f_dc_0/f_dc_1/f_dc_2 (SH DC -> RGB)"
    else:
        raise ValueError(
            f"RGB colors are missing in {npz_path}; expected `rgb_colors`, `features_dc`, or `f_dc_0..2`"
        )

    if points.shape[0] != colors.shape[0]:
        raise ValueError(f"Point/color count mismatch: {points.shape[0]} points vs {colors.shape[0]} colors")

    finite_mask = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
    if not np.all(finite_mask):
        print(f"[WARN] Removed {int(np.count_nonzero(~finite_mask))} non-finite points.")
        points = points[finite_mask]
        colors = colors[finite_mask]

    return points.astype(np.float32), colors.astype(np.float32), color_source


def load_rgb_point_cloud(
    npz_path: str,
    transform: typing.Optional[np.ndarray] = None,
):
    o3d = import_open3d()
    points, colors, color_source = load_rgb_points(npz_path)
    if transform is not None:
        points = apply_transform(points, transform)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud, color_source


def load_Replica_pose(line: str):
    """Load the first Replica pose used only as a coordinate anchor."""
    c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
    return c2w


if __name__ == '__main__':
    ### arguments ###
    args = argument_parsing()
    "/media/phudh/X9_Pro/undergraduate/ActiveMapping/src/visualization/vis_traj_heatvoxel.py"
    # set params ##
    HOME = "/media/phudh/X9_Pro/undergraduate"
    PROJ_DIR = f"{HOME}/ActiveMapping"
    DATASET = args.dataset
    RESULT_DIR = f'{PROJ_DIR}/results_all'
    GT_DATA_DIR = f"{PROJ_DIR}/data/{DATASET}"

    scene = args.scene
    method = args.method
    slam = args.slam
    run_version = args.run_version

    if not args.traj_file:
        args.traj_file = f'{RESULT_DIR}/{DATASET}/{scene}/{method}/{run_version}/{slam}/final/params.npz'
    traj_txt = args.traj_txt or f'{GT_DATA_DIR}/{scene}/traj.txt'

    with open(traj_txt, 'r') as f:
        lines = f.readlines()
        poses = [load_Replica_pose(line) for line in lines]

    transform = poses[0]
    window_hw = (1024, 1024)

    mesh, color_source = load_rgb_point_cloud(args.traj_file, transform=transform)
    print(f"[INFO] RGB color source: {color_source}")

    # Same clipping flow as the semantic no-trajectory visualizer.
    vertices = np.asarray(mesh.points)
    thres = args.z_thres
    if thres is None:
        thres = 1.0 if scene != "room2" else -0.75  # default 1.0, room2 -0.75
    mask = vertices[:, 2] <= thres
    print(f"[INFO] z filter: keep {int(mask.sum())}/{len(mask)} points with z <= {thres:.3f}")

    filtered_vertices = vertices[mask]
    filtered_colors = np.asarray(mesh.colors)[mask]

    o3d = import_open3d()
    filtered_mesh = o3d.geometry.PointCloud()
    filtered_mesh.points = o3d.utility.Vector3dVector(filtered_vertices)
    filtered_mesh.colors = o3d.utility.Vector3dVector(filtered_colors)

    ### initialize window ###
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=window_hw[1], height=window_hw[0])
    vis.add_geometry(filtered_mesh)

    render_option = vis.get_render_option()
    if render_option is not None:
        render_option.point_size = float(args.point_size)

    vis.poll_events()
    vis.update_renderer()

    print('Running local Open3D visualizer with reconstructed RGB and without trajectory overlays.')
    if args.with_interact:
        vis.run()
    vis.destroy_window()
