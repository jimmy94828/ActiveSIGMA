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
import numpy as np
import open3d as o3d
import os, sys
import typing


sys.path.append(os.getcwd())
from src.visualization.render_rgb_sem_entro_bev import (
    label_colors,
    load_mp3d_label_map,
)

# Not using WebRTC server in this script — use local Open3D visualizer instead


def argument_parsing() -> argparse.Namespace:
    """parse arguments

    Returns:
        args: arguments
        
    """
    parser = argparse.ArgumentParser(
            description="Arguments to visualize a semantic point cloud without trajectory overlays."
        )
    parser.add_argument("--mesh_file", type=str, default="", 
                        help="mesh file")
    parser.add_argument("--traj_file", type=str, default="",
                        help="params.npz file used to load the semantic point cloud")
    parser.add_argument("--out_dir", type=str, default=None, 
                        help="unused; kept for backward compatibility")
    parser.add_argument("--with_interact", type=int, default=1,
                        help="with interaction for visualization")
    args = parser.parse_args()
    return args


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return (points @ rotation.T + translation).astype(np.float32)


def load_semantic_points(npz_path: str) -> typing.Tuple[np.ndarray, np.ndarray]:
    params = np.load(npz_path, allow_pickle=True)
    if "means3D" in params:
        points = np.asarray(params["means3D"], dtype=np.float32)
    elif "points" in params:
        points = np.asarray(params["points"], dtype=np.float32)
    else:
        raise ValueError(f"No `means3D` or `points` found in {npz_path}")

    semantic_logits = None
    if "semantic_logits" in params:
        semantic_logits = np.asarray(params["semantic_logits"], dtype=np.float32)

    if "seman_cls_ids" in params:
        raw_labels = np.asarray(params["seman_cls_ids"])
        if raw_labels.ndim == 2 and raw_labels.shape[1] > 1:
            if semantic_logits is not None and semantic_logits.shape == raw_labels.shape:
                topk_idx = np.argmax(semantic_logits, axis=1)
                labels = raw_labels[np.arange(raw_labels.shape[0]), topk_idx].astype(np.int32)
            else:
                labels = raw_labels[:, 0].astype(np.int32)
        else:
            labels = raw_labels.reshape(-1).astype(np.int32)
    elif semantic_logits is not None:
        labels = np.argmax(semantic_logits, axis=1).astype(np.int32)
    else:
        raise ValueError(f"Semantic labels are missing in {npz_path}; expected `seman_cls_ids` or `semantic_logits`")

    if points.shape[0] != labels.shape[0]:
        raise ValueError(f"Point/label count mismatch: {points.shape[0]} points vs {labels.shape[0]} labels")
    return points, labels


def semantic_colors(labels: np.ndarray, label_map: typing.Optional[dict] = None) -> np.ndarray:
    lut = label_colors(labels, label_map)
    colors = np.empty((labels.shape[0], 3), dtype=np.float32)
    for label, color in lut.items():
        colors[labels == int(label)] = np.asarray(color, dtype=np.float32) / 255.0
    return colors


def load_semantic_point_cloud(
    npz_path: str,
    transform: typing.Optional[np.ndarray] = None,
    label_map: typing.Optional[dict] = None,
) -> o3d.geometry.PointCloud:
    points, labels = load_semantic_points(npz_path)
    if transform is not None:
        points = apply_transform(points, transform)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(semantic_colors(labels, label_map))
    return point_cloud


def resolve_label_map(dataset: str) -> typing.Optional[dict]:
    if dataset.lower() == "mp3d":
        return load_mp3d_label_map("high-contrast")
    return None


def load_Replica_pose(line: str):
    """ load the first Replica pose used only as a coordinate anchor

    Args:
        line (str): pose data as txt line. Format: camera-to-world, RUB

    Returns:
        c2w (np.ndarry, [4,4]): pose. Format: camera-to-world, RDF
    """
    c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
    return c2w

if __name__ == '__main__':
    ### arguments ###
    args = argument_parsing()
    "/media/phudh/X9_Pro/undergraduate/ActiveMapping/src/visualization/vis_traj_heatvoxel.py"
    # set params ##
    HOME = "/media/phudh/X9_Pro/undergraduate"
    PROJ_DIR = f"{HOME}/ActiveMapping"
    # PROJ_DIR = f"{HOME}/ActiveSGM"
    DATASET = "Replica"
    RESULT_DIR =f'{PROJ_DIR}/results_all'
    GT_DATA_DIR = f"{PROJ_DIR}/data/{DATASET}"

    scene = "room2"
    seed = 0
    method = "SemanticHeat"
    # method = "ActiveSem"
    slam = "splatam"
    run_version = "run_0" # 

    # Semantic point cloud colors are generated from params.npz below.
    if not args.traj_file:
        args.traj_file =f'{RESULT_DIR}/{DATASET}/{scene}/{method}/{run_version}/{slam}/final/params.npz'
    traj_txt=f'{GT_DATA_DIR}/{scene}/traj.txt'

    with open(traj_txt, 'r') as f:
        lines = f.readlines()
        poses = [load_Replica_pose(line) for line in lines]

    transform = poses[0]

    window_hw = (1024, 1024)

    label_map = resolve_label_map(DATASET)
    mesh = load_semantic_point_cloud(args.traj_file, transform=transform, label_map=label_map)
    # modify semantic point cloud
    vertices = np.asarray(mesh.points)
    thres = 1.0 if scene != "room2" else -0.75  #  default 1.0, room2 -0.75
    mask = vertices[:, 2] <= thres

    filtered_vertices = vertices[mask]
    filtered_colors = np.asarray(mesh.colors)[mask]
    filtered_mesh = o3d.geometry.PointCloud()
    filtered_mesh.points = points=o3d.utility.Vector3dVector(filtered_vertices)
    filtered_mesh.colors=o3d.utility.Vector3dVector(filtered_colors)

    ### initialize window ###
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=window_hw[1], height=window_hw[0])

    # register or add mesh to the visualizer window
    # webrtc_server.register_object("mesh", filtered_mesh)
    vis.add_geometry(filtered_mesh)
    vis.poll_events()
    vis.update_renderer()

    ### Add mesh ###
    # vis.draw_geometries([mesh])
    # vis.add_geometry(mesh)
    # vis.add_geometry(filtered_mesh)

    print('Running local Open3D visualizer without trajectory overlays.')
    if args.with_interact:
        vis.run()
    vis.destroy_window()
