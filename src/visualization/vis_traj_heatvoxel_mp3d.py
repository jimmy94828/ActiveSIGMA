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
import glob
import time


sys.path.append(os.getcwd())
from src.visualization.o3d_utils import (
    create_camera_frustum, 
    save_camera_parameters, 
    # load_camera_parameters_from_json,
    create_dashed_line
    )

# Not using WebRTC server in this script — use local Open3D visualizer instead


def argument_parsing() -> argparse.Namespace:
    """parse arguments

    Returns:
        args: arguments
        
    """
    parser = argparse.ArgumentParser(
            description="Arguments to visualize trajectory."
        )
    parser.add_argument("--mesh_file", type=str, default="", 
                        help="mesh file")
    parser.add_argument("--traj_file", type=str, default="",
                        help="trajectory pose dir")
    parser.add_argument("--anchor_traj_file", type=str, default="",
                        help="world-anchor trajectory txt file")
    parser.add_argument("--color_mode", type=str, default="auto",
                        choices=["auto", "spatial", "height", "none"],
                        help="mesh coloring mode for visualization")
    parser.add_argument("--out_dir", type=str, default=None, 
                        help="output directory to save rendered image")
    parser.add_argument("--output_name", type=str, default=None,
                        help="output PNG filename; default is inferred from scene and run")
    parser.add_argument("--with_interact", type=int, default=1,
                        help="with interaction for visualization")
    args = parser.parse_args()
    return args


def _path_parts(path: str):
    if not path:
        return []
    return [p for p in os.path.normpath(path).split(os.sep) if p]


def infer_scene_name(*paths: str, default: str = "scene") -> str:
    """Infer an MP3D scene id from common result/data path layouts."""
    for path in paths:
        parts = _path_parts(path)
        for idx, part in enumerate(parts):
            if part == "MP3D":
                for candidate in parts[idx + 1:]:
                    if candidate in {"MP3D", "v1", "tasks", "mp3d", "scans"}:
                        continue
                    if candidate.endswith((".npz", ".ply", ".obj", ".glb", ".txt")):
                        continue
                    return candidate
        for marker in ("mp3d_sim_nvs_v2", "mp3d_sim_nvs"):
            if marker in parts:
                idx = parts.index(marker)
                if idx + 1 < len(parts):
                    return parts[idx + 1]
    return default


def infer_run_name(path: str, default: str = "run") -> str:
    for part in _path_parts(path):
        if part.startswith("run_"):
            return part
    return default


def build_output_filename(args: argparse.Namespace, scene_name: str, run_name: str) -> str:
    if args.output_name:
        return args.output_name if args.output_name.endswith(".png") else f"{args.output_name}.png"
    safe_scene = scene_name or "scene"
    safe_run = run_name or "run"
    return f"vis_trajectory_{safe_scene}_{safe_run}.png"


def is_existing_file(path: str) -> bool:
    return bool(path) and os.path.isfile(path)


def resolve_anchor_traj_file(project_dir: str, scene_name: str, provided_path: str = "") -> str:
    if is_existing_file(provided_path):
        return provided_path
    candidate = os.path.join(project_dir, "data", "mp3d_sim_nvs_v2", scene_name, "traj.txt")
    if os.path.isfile(candidate):
        if provided_path:
            print(f"[WARN] Anchor trajectory not found: {provided_path}")
            print(f"[INFO] Using inferred anchor trajectory: {candidate}")
        return candidate
    raise FileNotFoundError(
        f"Anchor trajectory not found. provided={provided_path!r}, inferred={candidate!r}"
    )


def resolve_mesh_file(project_dir: str, scene_name: str, provided_path: str = "") -> str:
    if is_existing_file(provided_path):
        return provided_path
    try:
        candidate = resolve_default_mesh_file(project_dir, scene_name)
    except FileNotFoundError:
        candidate = ""
    if candidate and os.path.isfile(candidate):
        if provided_path:
            print(f"[WARN] Mesh file not found: {provided_path}")
            print(f"[INFO] Using inferred mesh file: {candidate}")
        return candidate
    raise FileNotFoundError(
        f"Mesh file not found. provided={provided_path!r}, inferred_scene={scene_name!r}"
    )


def convert_rel2world(start_c2w_rel, rel_c2w_slam):
    c2w_slam_w = start_c2w_rel @ rel_c2w_slam
    return c2w_slam_w


def normalize_channel(values: np.ndarray) -> np.ndarray:
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax - vmin < 1e-8:
        return np.full_like(values, 0.5, dtype=np.float64)
    return ((values - vmin) / (vmax - vmin)).astype(np.float64)


def spatial_colors(points: np.ndarray) -> np.ndarray:
    x = normalize_channel(points[:, 0])
    y = normalize_channel(points[:, 1])
    z = normalize_channel(points[:, 2])
    return np.stack([x, y, z], axis=1)


def height_colors(points: np.ndarray) -> np.ndarray:
    h = normalize_channel(points[:, 1])
    return np.stack([0.15 + 0.75 * h, 0.25 + 0.50 * (1.0 - np.abs(h - 0.5) * 2.0), 0.95 - 0.70 * h], axis=1)


def build_colors(points: np.ndarray, color_mode: str) -> np.ndarray:
    if color_mode == "spatial":
        return spatial_colors(points)
    if color_mode == "height":
        return np.clip(height_colors(points), 0.0, 1.0)
    raise ValueError(f"Unsupported color_mode: {color_mode}")


def has_readable_textures(mesh) -> bool:
    for texture in getattr(mesh, "textures", []):
        arr = np.asarray(texture)
        if arr.dtype != object and arr.ndim >= 2 and arr.size > 0:
            return True
    return False


def find_mp3d_textured_mesh(mesh_file: str) -> str:
    scene = infer_scene_name(mesh_file, default="")
    if not scene:
        return ""
    candidates = glob.glob(os.path.join(
        os.getcwd(),
        "data", "MP3D", "v1", "scans", scene, scene, "matterport_mesh", "*", "*.obj"
    ))
    candidates.sort()
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return ""


def resolve_default_mesh_file(project_dir: str, scene: str) -> str:
    candidates = [
        os.path.join(project_dir, "data", "MP3D", "v1", "scans", scene, "mesh.obj"),
        os.path.join(project_dir, "data", "MP3D", "v1", "tasks", "mp3d", scene, "semantic_clean.ply"),
        os.path.join(project_dir, "data", "MP3D", "v1", "tasks", "mp3d", scene, f"{scene}_semantic.ply"),
        os.path.join(project_dir, "data", "MP3D", "v1", "tasks", "mp3d", scene, f"{scene}.glb"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"No usable MP3D mesh asset found for scene `{scene}`")


def load_visual_geometry(mesh_file: str, z_threshold: float = 1.0, color_mode: str = "auto"):
    mesh_ext = os.path.splitext(mesh_file)[1].lower()

    # Triangle-mesh-first for mesh-like assets such as OBJ/GLB or mesh PLY.
    if mesh_ext in {".obj", ".glb", ".gltf", ".off", ".stl", ".fbx", ".dae", ".ply"}:
        tri_mesh = o3d.io.read_triangle_mesh(mesh_file, enable_post_processing=True)
        if not tri_mesh.is_empty() and len(tri_mesh.vertices) > 0:
            if color_mode == "auto" and not tri_mesh.has_vertex_colors() and not has_readable_textures(tri_mesh):
                textured_mesh_file = find_mp3d_textured_mesh(mesh_file)
                if textured_mesh_file and os.path.abspath(textured_mesh_file) != os.path.abspath(mesh_file):
                    textured_mesh = o3d.io.read_triangle_mesh(textured_mesh_file, enable_post_processing=True)
                    if (not textured_mesh.is_empty()) and len(textured_mesh.vertices) > 0 and has_readable_textures(textured_mesh):
                        print(f"[INFO] Using textured MP3D mesh: {textured_mesh_file}")
                        tri_mesh = textured_mesh
            if not tri_mesh.has_vertex_normals():
                tri_mesh.compute_vertex_normals()
            if color_mode in {"spatial", "height"}:
                colors = build_colors(np.asarray(tri_mesh.vertices), color_mode)
                tri_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            elif color_mode == "auto" and not tri_mesh.has_vertex_colors() and not has_readable_textures(tri_mesh):
                colors = spatial_colors(np.asarray(tri_mesh.vertices))
                tri_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            elif color_mode == "none" and not tri_mesh.has_vertex_colors():
                tri_mesh.paint_uniform_color([0.75, 0.75, 0.75])
            return tri_mesh

    point_cloud = o3d.io.read_point_cloud(mesh_file)
    if point_cloud.is_empty():
        raise ValueError(f"Failed to load mesh or point cloud from: {mesh_file}")

    points = np.asarray(point_cloud.points)
    if color_mode in {"spatial", "height"}:
        point_cloud.colors = o3d.utility.Vector3dVector(build_colors(points, color_mode))
    elif color_mode == "auto" and not point_cloud.has_colors():
        point_cloud.colors = o3d.utility.Vector3dVector(spatial_colors(points))
    elif color_mode == "none" and not point_cloud.has_colors():
        point_cloud.paint_uniform_color([0.75, 0.75, 0.75])

    if z_threshold is not None:
        mask = points[:, 2] <= z_threshold
        filtered_cloud = o3d.geometry.PointCloud()
        filtered_cloud.points = o3d.utility.Vector3dVector(points[mask])
        filtered_cloud.colors = o3d.utility.Vector3dVector(np.asarray(point_cloud.colors)[mask])
        return filtered_cloud

    return point_cloud

def load_cam_traj(scene_path):
    params = dict(np.load(scene_path, allow_pickle=True))
    cam_traj = {}
    cam_traj['w2cs'] = params.pop('gt_w2c_all_frames')
    cam_traj['intrinsic'] = params.pop('intrinsics')
    cam_traj['height'] = params.pop('org_height')
    cam_traj['width'] = params.pop('org_width')
    return cam_traj # 80,4,4

def load_Replica_pose(line: str):
    """ load Replica pose from trajectory file

    Args:
        line (str): pose data as txt line. Format: camera-to-world, RUB

    Returns:
        c2w (np.ndarry, [4,4]): pose. Format: camera-to-world, RDF
    """
    c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
    return c2w

def load_camera_parameters_from_data(cam_traj: dict) -> o3d.camera.PinholeCameraParameters:
    """ load camera parameters from json

    Args:
        json_file (str): camera parameter json file

    Returns:
        cam_param (o3d.camera.PinholeCameraParameters): camera parameters
    """
    # Load intrinsic parameters
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        cam_traj["width"],
        cam_traj["height"],
        cam_traj["intrinsic"][0, 0],  # fx
        cam_traj["intrinsic"][1, 1],  # fy
        cam_traj["intrinsic"][0, 2],  # cx
        cam_traj["intrinsic"][1, 2]  # cy
    )

    # Load extrinsic parameters
    w2c_params = cam_traj["w2cs"][0]
    w2c = np.array(w2c_params).reshape((4, 4))
    extrinsic_matrix = np.linalg.inv(w2c)
    extrinsic_matrix = extrinsic_matrix.transpose()

    # Create PinholeCameraParameters
    cam_param = o3d.camera.PinholeCameraParameters()
    cam_param.extrinsic = extrinsic_matrix
    cam_param.intrinsic = intrinsic

    return cam_param

if __name__ == '__main__':
    ### arguments ###
    args = argument_parsing()
    "/media/phudh/X9_Pro/undergraduate/ActiveMapping/src/visualization/vis_traj_heatvoxel_mp3d.py"
    # set params ##
    HOME = "/media/phudh/X9_Pro/undergraduate"
    PROJ_DIR = f"{HOME}/ActiveMapping"
    # PROJ_DIR = f"{HOME}/ActiveSGM"
    DATASET = "MP3D"
    RESULT_DIR =f'{PROJ_DIR}/results'
    GT_DATA_DIR = f"{PROJ_DIR}/data/{DATASET}/v1/scans"

    scene = "GdvgFV5R1Z5"
    seed = 0
    method = "SemanticHeat"
    # method = "ActiveSem"
    slam = "splatam"
    run_version = "_v6" # _v1, _v2, ... empty for run_0

    if not args.traj_file:
        args.traj_file =f'{RESULT_DIR}/{DATASET}/{scene}/{method}/run_0{run_version}/{slam}/final/params.npz'
    if not args.out_dir:
        args.out_dir =f'{RESULT_DIR}/{DATASET}/{scene}/{method}/run_0{run_version}/visualization/planning_path/'

    inferred_scene = infer_scene_name(args.traj_file, args.mesh_file, args.anchor_traj_file, default=scene)
    args.mesh_file = resolve_mesh_file(PROJ_DIR, inferred_scene, args.mesh_file)
    args.anchor_traj_file = resolve_anchor_traj_file(PROJ_DIR, inferred_scene, args.anchor_traj_file)

    if not os.path.isfile(args.traj_file):
        raise FileNotFoundError(
            f"Trajectory params file not found: {args.traj_file}. "
            "If you used shell variables like ${run_root}, make sure they are set first."
        )

    print(f"Scene             : {inferred_scene}")
    print(f"Mesh file         : {args.mesh_file}")
    print(f"Trajectory params : {args.traj_file}")
    print(f"Anchor trajectory : {args.anchor_traj_file}")
    print(f"Output directory  : {args.out_dir}")

    traj_txt = args.anchor_traj_file

    with open(traj_txt, 'r') as f:
        lines = f.readlines()
        poses = [load_Replica_pose(line) for line in lines]

    transform = poses[0]

    mesh_file = args.mesh_file
    window_hw = (1024, 1024)

    visual_geometry = load_visual_geometry(mesh_file, z_threshold=1.0, color_mode=args.color_mode)

    # mesh.transform(transform)
    # camera_trajectory = np.load(traj_file)
    camera_trajectory = load_cam_traj(args.traj_file)

    ip = '127.0.0.1'
    port = '5001'
    os.environ['EGL_PLATFORM'] = 'surfaceless'
    os.environ['OPEN3D_CPU_RENDERING'] = 'true'
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = 'true'
    os.environ['WEBRTC_IP'] = ip
    os.environ['WEBRTC_PORT'] = port
    # We intentionally do not enable or import the WebRTC server here.

    ### initialize window ###
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=window_hw[1], height=window_hw[0])

    # register or add mesh to the visualizer window
    # webrtc_server.register_object("mesh", filtered_mesh)
    vis.add_geometry(visual_geometry)
    vis.poll_events()
    vis.update_renderer()

    ### Add mesh ###
    # vis.draw_geometries([mesh])
    # vis.add_geometry(mesh)
    # vis.add_geometry(filtered_mesh)

    ### set a view direction ###
    if args.traj_file is not None:
        vis_cam_param = load_camera_parameters_from_data(camera_trajectory)
    # view_control = vis.get_view_control()

    ### Add trajecotry ###
    skip_step = 5
    w2c_subset = camera_trajectory['w2cs'][::skip_step]
    # cam_traj_subset = camera_trajectory[::skip_step]
    # cam_traj_subset = camera_trajectory[:10]
    c2ws = [convert_rel2world(transform,np.linalg.inv(w2c)) for w2c in w2c_subset]
    w2c_subset = [np.linalg.inv(c2w) for c2w in c2ws]

    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)

    for step, w2c in enumerate(w2c_subset):
        ##################################################
        ### Add Camera ###
        ##################################################
        pose = np.linalg.inv(w2c)
        intrinsic = np.array([[300, 0, 300],
                            [0, 300, 300],
                            [0, 0, 1]])
        ### Create camera frustum ###
        if step == 0:
            color = [1, 0, 0]
        elif step == len(w2c_subset) - 1:
            color = [0, 0, 1]
        else:
            color = [0, 1, 0]
        camera_frustum = create_camera_frustum(color=color, extrinsic=pose, intrinsic=intrinsic, scale=1)
        # webrtc_server.register_object(f"camera_{step}", camera_frustum)
        vis.add_geometry(camera_frustum)
        vis.poll_events()
        vis.update_renderer()

        ##################################################
        ### Add line
        ##################################################
        if step > 0:
            points = [np.linalg.inv(w2c)[:3, 3] for w2c in w2c_subset[step-1:step+1]]
            line_set = create_dashed_line(points, color=[0, 0, 0])
            vis.add_geometry(line_set)
            # webrtc_server.register_object(f"line_{step}", line_set)
            vis.poll_events()
            vis.update_renderer()

        ##################################################
        ### set camera view
        ##################################################
        # if args.traj_file is not None:
        #     view_control.convert_from_pinhole_camera_parameters(vis_cam_param, allow_arbitrary=True)

        ##################################################
        ### update visualizer
        ##################################################
        # vis.poll_events()
        # vis.update_renderer()

        ##################################################
        ### save visualization
        ##################################################


        time.sleep(0.01)
    if args.out_dir is not None:
        output_scene = infer_scene_name(args.traj_file, args.mesh_file, args.anchor_traj_file, default=scene)
        output_run = infer_run_name(args.traj_file, default=f"run_0{run_version}" if run_version else "run_0")
        render_filename = build_output_filename(args, output_scene, output_run)
        render_filepath = os.path.join(args.out_dir, render_filename)
        vis.capture_screen_image(render_filepath, do_render=True)
        print(f"Saved trajectory visualization: {render_filepath}")
    ### RUN ###
    # if args.with_interact:
    #     save_camera_parameters(vis)
    #     vis.run()
    #     vis.destroy_window()
    # else:
    #     vis.destroy_window()

    # Run local Open3D visualizer (no WebRTC)
    print('Running local Open3D visualizer (interactive window).')
    if args.with_interact:
        save_camera_parameters(vis)
        vis.run()
        vis.destroy_window()
    else:
        vis.destroy_window()
