#!/usr/bin/env python3
"""Interactive Habitat-Sim trajectory recorder for Replica and MP3D scenes.

The script records keyboard-controlled camera poses and saves RGB-D/semantic
observations in the ``results_habitat`` layout used by the local dataloaders.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

cv2 = None
habitat_sim = None
mmengine = None
quaternion = None
get_pinhole_intrinsic = None
make_configuration = None

DEPTH_PNG_SCALE = 6553.5
KEY_ESC = 27
KEY_SPACE = 32
KEY_UP = {65362, 2490368}
KEY_DOWN = {65364, 2621440}
KEY_LEFT = {65361, 2424832}
KEY_RIGHT = {65363, 2555904}


def load_runtime_dependencies() -> None:
    global cv2, habitat_sim, mmengine, quaternion, get_pinhole_intrinsic, make_configuration

    try:
        import cv2 as cv2_module
        import habitat_sim as habitat_sim_module
        import mmengine as mmengine_module
        import quaternion as quaternion_module
    except ImportError as exc:
        raise ImportError(
            "Missing runtime dependency. Please run this script inside the project "
            "environment that has OpenCV, Habitat-Sim, mmengine, and numpy-quaternion installed."
        ) from exc

    from src.simulator.habitat_utils import (
        get_pinhole_intrinsic as get_pinhole_intrinsic_fn,
        make_configuration as make_configuration_fn,
    )

    cv2 = cv2_module
    habitat_sim = habitat_sim_module
    mmengine = mmengine_module
    quaternion = quaternion_module
    get_pinhole_intrinsic = get_pinhole_intrinsic_fn
    make_configuration = make_configuration_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use keyboard control to record RGB-D, semantic maps, and camera "
            "poses from Habitat-Sim Replica/MP3D scenes."
        )
    )
    parser.add_argument("--dataset", choices=["Replica", "MP3D"], required=True)
    parser.add_argument("--scene", required=True, help="Scene name, e.g. office0 or GdvgFV5R1Z5.")
    parser.add_argument("--config", default=None, help="Habitat config path. Defaults to configs/<dataset>/<scene>/habitat.py.")
    parser.add_argument("--output-dir", default=None, help="Directory that will contain traj.txt and results_habitat/.")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing output directory before recording.")
    parser.add_argument("--append", action="store_true", help="Append frames to an existing output directory.")
    parser.add_argument("--start-pose", default=None, help="Path to a traj.txt or 4x4 pose txt used for the initial pose.")
    parser.add_argument("--start-index", type=int, default=0, help="Line index used when --start-pose is a trajectory file.")
    parser.add_argument("--start-pose-format", choices=["rdf", "rub"], default="rdf", help="Coordinate convention of --start-pose.")
    parser.add_argument("--traj-format", choices=["rdf", "rub", "both"], default="rdf", help="Format written to traj.txt.")
    parser.add_argument("--height", type=int, default=None, help="Override pinhole image height.")
    parser.add_argument("--width", type=int, default=None, help="Override pinhole image width.")
    parser.add_argument("--move-step", type=float, default=0.10, help="Translation step in meters.")
    parser.add_argument("--rot-step", type=float, default=5.0, help="Rotation step in degrees.")
    parser.add_argument("--preview-width", type=int, default=1280, help="Maximum OpenCV preview width.")
    parser.add_argument("--wait-ms", type=int, default=30, help="OpenCV waitKey delay.")
    parser.add_argument("--manual-save", action="store_true", help="Only save when Space is pressed.")
    parser.add_argument("--no-save-initial", action="store_true", help="Do not save the initial rendered pose.")
    parser.add_argument("--render-once", action="store_true", help="Render and save the initial pose, then exit without opening an OpenCV window.")
    parser.add_argument("--require-navigable", action="store_true", help="Reject translations to non-navigable positions when a navmesh is loaded.")
    parser.add_argument("--enable-erp", action="store_true", help="Keep equirectangular sensors enabled from the config.")
    parser.add_argument("--semantic-raw", action="store_true", help="Save raw Habitat object ids as semantic_map_*.npy.")
    parser.add_argument("--gpu-device-id", type=int, default=None, help="Habitat-Sim GPU device id. Defaults to the config/env default.")
    parser.add_argument(
        "--gl-backend",
        choices=["auto", "default", "surfaceless", "x11", "none"],
        default="auto",
        help="OpenGL/EGL setup for Habitat-Sim. Use none to skip the safety preflight.",
    )
    return parser.parse_args()


def default_output_dir(dataset: str, scene: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "trajectory_generate" / "keyboard_trajectories" / dataset / scene / stamp


def resolve_config_path(args: argparse.Namespace) -> Path:
    if args.config:
        return Path(args.config).expanduser().resolve()
    return REPO_ROOT / "configs" / args.dataset / args.scene / "habitat.py"


def update_camera_config(cfg: mmengine.Config, args: argparse.Namespace) -> None:
    if "camera" not in cfg or "pinhole" not in cfg.camera:
        raise ValueError("Habitat config must define camera.pinhole.")

    cfg.camera.pinhole.enable = True
    cam_types = list(cfg.camera.pinhole.get("cam_type", []))
    for sensor_type in ("color", "depth", "semantic"):
        if sensor_type not in cam_types:
            cam_types.append(sensor_type)
    cfg.camera.pinhole.cam_type = cam_types

    if args.height is not None or args.width is not None:
        height, width = cfg.camera.pinhole.resolution_hw
        cfg.camera.pinhole.resolution_hw = [args.height or height, args.width or width]

    if not args.enable_erp and "equirectangular" in cfg.camera:
        cfg.camera.equirectangular.enable = False


def init_simulator(cfg: mmengine.Config, gpu_device_id: Optional[int]) -> habitat_sim.Simulator:
    sim_cfg = make_configuration(cfg)
    if gpu_device_id is not None:
        sim_cfg.sim_cfg.gpu_device_id = gpu_device_id
    sim = habitat_sim.Simulator(sim_cfg)
    if "gravity" in cfg.simulator.physics:
        sim.set_gravity(cfg.simulator.physics.gravity)
    sim.step_physics(1.0)
    return sim


def gl_backend_env(backend: str) -> Dict[str, Optional[str]]:
    if backend == "default":
        return {}
    if backend == "surfaceless":
        return {"EGL_PLATFORM": "surfaceless"}
    if backend == "x11":
        return {"EGL_PLATFORM": "x11"}
    return {}


def apply_env_overrides(overrides: Dict[str, Optional[str]]) -> None:
    for key, value in overrides.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def probe_gl_context(env_overrides: Dict[str, Optional[str]]) -> Tuple[bool, str]:
    env = os.environ.copy()
    for key, value in env_overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    cmd = [
        sys.executable,
        "-c",
        "from magnum.platform.egl import WindowlessApplication; "
        "app = WindowlessApplication(); print('ok')",
    ]
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = "\n".join(part.strip() for part in (proc.stdout, proc.stderr) if part.strip())
    return proc.returncode == 0, output


def configure_gl_backend(args: argparse.Namespace) -> None:
    if args.gl_backend == "none":
        return

    candidates = ["default", "surfaceless", "x11"] if args.gl_backend == "auto" else [args.gl_backend]
    failures = []
    for backend in candidates:
        overrides = gl_backend_env(backend)
        ok, output = probe_gl_context(overrides)
        if ok:
            apply_env_overrides(overrides)
            if backend != "default":
                print(f"[INFO] Habitat GL backend: {backend}")
            return
        failures.append((backend, output))

    if any("ModuleNotFoundError" in output and "magnum" in output for _, output in failures):
        print("[ERROR] Magnum/Habitat-Sim is not importable in this Python environment.")
        print("        Activate the project environment first, e.g. conda activate ActiveSIGMA.")
        for backend, output in failures:
            print(f"\n[GL probe: {backend}]")
            print(output or "(no output)")
        raise SystemExit(2)

    print("[ERROR] Habitat-Sim cannot create an OpenGL/EGL context on this shell.")
    print("        The simulator would abort before Python can catch it, so the recorder stopped early.")
    print("        Usual fixes: run from a working desktop/X session, or repair NVIDIA EGL/driver access.")
    print("        Quick checks:")
    print("          nvidia-smi")
    print("          conda run -n activemapping python -c \"from magnum.platform.egl import WindowlessApplication; WindowlessApplication()\"")
    for backend, output in failures:
        print(f"\n[GL probe: {backend}]")
        print(output or "(no output)")
    raise SystemExit(2)


def pose_rdf_to_rub(pose: np.ndarray) -> np.ndarray:
    out = pose.astype(np.float32).copy()
    out[:3, 1] *= -1.0
    out[:3, 2] *= -1.0
    return out


def pose_rub_to_rdf(pose: np.ndarray) -> np.ndarray:
    return pose_rdf_to_rub(pose)


def pose_from_config_agent(cfg: mmengine.Config) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = np.asarray(cfg.agent.position, dtype=np.float32)
    pose[:3, :3] = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(cfg.agent.rotation)).astype(np.float32)
    return pose


def read_pose_file(path: Path, index: int) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float32)
    if data.ndim == 1:
        if data.size != 16:
            raise ValueError(f"Pose file {path} must contain 16 values or one 4x4 matrix.")
        return data.reshape(4, 4)
    if data.shape == (4, 4):
        return data.astype(np.float32)
    if data.shape[1] != 16:
        raise ValueError(f"Trajectory file {path} must have 16 values per line.")
    if index < 0 or index >= data.shape[0]:
        raise IndexError(f"--start-index {index} is outside trajectory length {data.shape[0]}.")
    return data[index].reshape(4, 4).astype(np.float32)


def candidate_start_pose_paths(dataset: str, scene: str) -> Iterable[Path]:
    if dataset == "Replica":
        yield REPO_ROOT / "data" / "Replica" / scene / "traj.txt"
        yield REPO_ROOT / "data" / "replica_sim_nvs" / scene / "traj.txt"
        yield REPO_ROOT / "data" / "replica_sim_nvs_v2" / scene / "traj.txt"
    else:
        yield REPO_ROOT / "data" / "mp3d_sim_nvs_v2" / scene / "traj.txt"
        yield REPO_ROOT / "data" / "MP3D" / "v1" / "scans" / scene / "traj.txt"


def resolve_start_pose(args: argparse.Namespace, cfg: mmengine.Config) -> Tuple[np.ndarray, str]:
    start_path = Path(args.start_pose).expanduser().resolve() if args.start_pose else None
    pose_format = args.start_pose_format
    source = "config.agent"

    if start_path is None:
        for candidate in candidate_start_pose_paths(args.dataset, args.scene):
            if candidate.exists():
                start_path = candidate
                source = str(candidate.relative_to(REPO_ROOT))
                pose_format = "rdf"
                break

    if start_path is not None:
        pose = read_pose_file(start_path, args.start_index)
        if args.start_pose or source != "config.agent":
            source = str(start_path)
        if pose_format == "rdf":
            pose = pose_rdf_to_rub(pose)
        return pose.astype(np.float32), source

    return pose_from_config_agent(cfg), source


def set_agent_pose(sim: habitat_sim.Simulator, c2w_rub: np.ndarray) -> None:
    state = habitat_sim.AgentState()
    state.position = c2w_rub[:3, 3]
    state.rotation = quaternion.from_rotation_matrix(c2w_rub[:3, :3])
    sim.agents[0].set_state(state)


def find_sensor_key(obs: Dict[str, np.ndarray], kind: str) -> Optional[str]:
    preferred = f"pinhole_{kind}_0.0"
    if preferred in obs:
        return preferred
    for key in obs.keys():
        if key.startswith(f"pinhole_{kind}"):
            return key
    for key in obs.keys():
        if kind in key:
            return key
    return None


def render(sim: habitat_sim.Simulator, pose_rub: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
    set_agent_pose(sim, pose_rub)
    obs = sim.get_sensor_observations()
    color_key = find_sensor_key(obs, "color")
    depth_key = find_sensor_key(obs, "depth")
    semantic_key = find_sensor_key(obs, "semantic")

    color = obs[color_key][:, :, :3].astype(np.uint8) if color_key is not None else None
    depth = obs[depth_key].astype(np.float32) if depth_key is not None else None
    semantic = obs[semantic_key].astype(np.int32) if semantic_key is not None else None
    return {"color": color, "depth": depth, "semantic": semantic}


def replica_scene_dir_name(scene: str) -> str:
    if "_" in scene:
        return scene
    prefix = "".join(ch for ch in scene if not ch.isdigit())
    suffix = "".join(ch for ch in scene if ch.isdigit())
    return f"{prefix}_{suffix}" if prefix and suffix else scene


def load_semantic_mapper(dataset: str, scene: str, cfg: mmengine.Config):
    if dataset == "Replica":
        scene_id = Path(cfg.simulator.scene_id)
        candidates = [
            scene_id.parent / "info_semantic.json",
            REPO_ROOT / "data" / "replica_v1" / replica_scene_dir_name(scene) / "habitat" / "info_semantic.json",
        ]
        for path in candidates:
            if path.exists():
                with open(path, "r") as f:
                    id_to_label = np.asarray(json.load(f)["id_to_label"], dtype=np.int32)

                def mapper(object_ids: np.ndarray) -> np.ndarray:
                    labels = np.zeros_like(object_ids, dtype=np.int32)
                    valid = (object_ids >= 0) & (object_ids < len(id_to_label))
                    labels[valid] = id_to_label[object_ids[valid]]
                    return labels

                return mapper, 102, str(path)
        return None, 102, None

    mapping_path = REPO_ROOT / "configs" / "MP3D" / scene / "instance_to_mpcat40.json"
    if mapping_path.exists():
        with open(mapping_path, "r") as f:
            mapping = {int(k): int(v) for k, v in json.load(f).items()}
        mapping_source = str(mapping_path)
    else:
        mapping, mapping_source = build_mp3d_instance_mapping(scene)
        if mapping is None:
            return None, 41, None

    def mapper(object_ids: np.ndarray) -> np.ndarray:
        out = np.zeros_like(object_ids, dtype=np.int32)
        for object_id in np.unique(object_ids):
            out[object_ids == object_id] = mapping.get(int(object_id), 0)
        return out

    return mapper, 41, mapping_source


def build_mp3d_instance_mapping(scene: str) -> Tuple[Optional[Dict[int, int]], Optional[str]]:
    category_mapping_path = REPO_ROOT / "configs" / "MP3D" / "category_mapping.tsv"
    semseg_path = REPO_ROOT / "data" / "MP3D" / "v1" / "scans" / scene / scene / "house_segmentations" / f"{scene}.semseg.json"
    if not category_mapping_path.exists() or not semseg_path.exists():
        return None, None

    label_to_mpcat40 = {}
    with open(category_mapping_path, "r", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            label_idx = int(row["index"])
            mpcat40_idx = int(row["mpcat40index"])
            label_to_mpcat40[label_idx] = 0 if mpcat40_idx == 41 else mpcat40_idx

    with open(semseg_path, "r") as f:
        semseg = json.load(f)

    mapping = {}
    for group in semseg.get("segGroups", []):
        instance_id = int(group["id"])
        label_idx = int(group["label_index"])
        mapping[instance_id] = int(label_to_mpcat40.get(label_idx, 0))
    return mapping, f"{semseg_path} + {category_mapping_path}"


def make_colormap(num_classes: int) -> np.ndarray:
    try:
        from imgviz import label_colormap

        return label_colormap(num_classes)[:, :3].astype(np.uint8)
    except ModuleNotFoundError:
        pass

    random.seed(42)
    colors = np.zeros((num_classes, 3), dtype=np.uint8)
    for idx in range(num_classes):
        colors[idx] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
    colors[0] = [0, 0, 0]
    return colors


def semantic_to_rgb(semantic: np.ndarray, num_classes: int) -> np.ndarray:
    semantic = semantic.copy()
    semantic[semantic < 0] = 0
    semantic[semantic >= num_classes] = 0
    return make_colormap(num_classes)[semantic.astype(np.int64)]


def prepare_output_dir(output_dir: Path, overwrite: bool, append: bool) -> int:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and not append and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} already exists. Use --overwrite, --append, or another --output-dir.")

    results_dir = output_dir / "results_habitat"
    for subdir in (
        results_dir,
        results_dir / "semantic",
        results_dir / "semantic_obj",
        results_dir / "rgb",
        output_dir / "pose_rub",
        output_dir / "pose_rdf",
    ):
        subdir.mkdir(parents=True, exist_ok=True)

    if not append:
        for traj_name in ("traj.txt", "traj_rdf.txt", "traj_habitat_rub.txt"):
            traj_path = output_dir / traj_name
            if traj_path.exists():
                traj_path.unlink()
        return 0

    existing = sorted(results_dir.glob("frame*.jpg"))
    if not existing:
        return 0
    return max(int(path.stem.replace("frame", "")) for path in existing) + 1


def append_pose(path: Path, pose: np.ndarray) -> None:
    pose = np.where(np.abs(pose) < 1e-12, 0.0, pose)
    with open(path, "a") as f:
        f.write(" ".join(f"{v:.9g}" for v in pose.reshape(-1)) + "\n")


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    config_path: Path,
    start_source: str,
    intrinsics: Optional[np.ndarray],
    semantic_source: Optional[str],
) -> None:
    metadata = {
        "dataset": args.dataset,
        "scene": args.scene,
        "config": str(config_path),
        "start_source": start_source,
        "traj_format": args.traj_format,
        "depth_png_scale": DEPTH_PNG_SCALE,
        "semantic_source": semantic_source,
        "controls": {
            "move": "W/S forward/back, A/D left/right, R/F up/down",
            "look": "Arrow keys or I/J/K/L pitch/yaw, Q/E roll",
            "save": "Space saves one frame, P toggles auto-recording",
            "speed": "[/] move step, -/= rotation step",
            "help": "H prints controls",
            "quit": "Esc",
        },
    }
    if intrinsics is not None:
        metadata["pinhole_intrinsics"] = intrinsics.tolist()
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def save_frame(
    output_dir: Path,
    frame_idx: int,
    obs: Dict[str, Optional[np.ndarray]],
    pose_rub: np.ndarray,
    traj_format: str,
    mapper,
    num_classes: int,
    save_raw_semantic: bool,
) -> None:
    results_dir = output_dir / "results_habitat"
    pose_rdf = pose_rub_to_rdf(pose_rub)

    color = obs["color"]
    if color is not None:
        color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(results_dir / f"frame{frame_idx:06d}.jpg"), color_bgr)
        cv2.imwrite(str(results_dir / "rgb" / f"color_{frame_idx:04d}.jpg"), color_bgr)

    depth = obs["depth"]
    if depth is not None:
        depth_u16 = np.clip(np.nan_to_num(depth) * DEPTH_PNG_SCALE, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(results_dir / f"depth{frame_idx:06d}.png"), depth_u16)

    semantic = obs["semantic"]
    if semantic is not None:
        semantic_out = semantic if save_raw_semantic or mapper is None else mapper(semantic)
        semantic_out = semantic_out.astype(np.int32)
        np.save(results_dir / "semantic" / f"semantic_map_{frame_idx:04d}.npy", semantic_out)
        np.save(results_dir / "semantic_obj" / f"semantic_obj_{frame_idx:04d}.npy", semantic.astype(np.int32))
        semantic_rgb = semantic_to_rgb(semantic_out, num_classes)
        cv2.imwrite(str(results_dir / "semantic" / f"semantic_rgb_{frame_idx:04d}.png"), cv2.cvtColor(semantic_rgb, cv2.COLOR_RGB2BGR))

    np.savetxt(output_dir / "pose_rub" / f"{frame_idx:06d}.txt", pose_rub, fmt="%.9g")
    np.savetxt(output_dir / "pose_rdf" / f"{frame_idx:06d}.txt", pose_rdf, fmt="%.9g")

    if traj_format in ("rdf", "both"):
        append_pose(output_dir / "traj_rdf.txt", pose_rdf)
    if traj_format in ("rub", "both"):
        append_pose(output_dir / "traj_habitat_rub.txt", pose_rub)
    if traj_format == "rub":
        append_pose(output_dir / "traj.txt", pose_rub)
    else:
        append_pose(output_dir / "traj.txt", pose_rdf)


def rotate_pose_local(pose: np.ndarray, axis: str, degrees: float) -> np.ndarray:
    rad = np.deg2rad(degrees)
    c, s = np.cos(rad), np.sin(rad)
    rot = np.eye(3, dtype=np.float32)
    if axis == "x":
        rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    elif axis == "y":
        rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    elif axis == "z":
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    out = pose.copy()
    out[:3, :3] = (out[:3, :3] @ rot).astype(np.float32)
    return out


def translate_pose_local(pose: np.ndarray, local_delta: np.ndarray) -> np.ndarray:
    out = pose.copy()
    out[:3, 3] += pose[:3, :3] @ local_delta.astype(np.float32)
    return out


def apply_key(pose: np.ndarray, key: int, move_step: float, rot_step: float) -> Tuple[np.ndarray, bool]:
    ch = chr(key & 0xFF).lower() if 0 <= (key & 0xFF) < 128 else ""
    if ch == "w":
        return translate_pose_local(pose, np.array([0.0, 0.0, -move_step])), True
    if ch == "s":
        return translate_pose_local(pose, np.array([0.0, 0.0, move_step])), True
    if ch == "a":
        return translate_pose_local(pose, np.array([-move_step, 0.0, 0.0])), True
    if ch == "d":
        return translate_pose_local(pose, np.array([move_step, 0.0, 0.0])), True
    if ch == "r":
        return translate_pose_local(pose, np.array([0.0, move_step, 0.0])), True
    if ch == "f":
        return translate_pose_local(pose, np.array([0.0, -move_step, 0.0])), True
    if ch == "k" or key in KEY_UP:
        return rotate_pose_local(pose, "x", -rot_step), True
    if ch == "i" or key in KEY_DOWN:
        return rotate_pose_local(pose, "x", rot_step), True
    if ch == "j" or key in KEY_LEFT:
        return rotate_pose_local(pose, "y", rot_step), True
    if ch == "l" or key in KEY_RIGHT:
        return rotate_pose_local(pose, "y", -rot_step), True
    if ch == "e":
        return rotate_pose_local(pose, "z", -rot_step), True
    if ch == "q":
        return rotate_pose_local(pose, "z", rot_step), True
    return pose, False


def is_candidate_valid(sim: habitat_sim.Simulator, pose: np.ndarray, require_navigable: bool) -> bool:
    if not require_navigable:
        return True
    pathfinder = getattr(sim, "pathfinder", None)
    if pathfinder is None or not pathfinder.is_loaded:
        return True
    return bool(pathfinder.is_navigable(pose[:3, 3]))


def depth_preview(depth: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if depth is None:
        return None
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        scaled = np.zeros_like(depth, dtype=np.uint8)
    else:
        hi = max(float(np.percentile(valid, 95)), 1e-6)
        scaled = np.clip(depth / hi * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)


def compose_preview(
    obs: Dict[str, Optional[np.ndarray]],
    pose: np.ndarray,
    frame_idx: int,
    recording: bool,
    move_step: float,
    rot_step: float,
    preview_width: int,
    mapper,
    num_classes: int,
) -> np.ndarray:
    panels = []
    color = obs["color"]
    if color is not None:
        panels.append(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    depth_img = depth_preview(obs["depth"])
    if depth_img is not None:
        panels.append(depth_img)
    semantic = obs["semantic"]
    if semantic is not None:
        semantic_for_preview = mapper(semantic) if mapper is not None else semantic
        sem_rgb = semantic_to_rgb(semantic_for_preview, num_classes)
        panels.append(cv2.cvtColor(sem_rgb, cv2.COLOR_RGB2BGR))
    if not panels:
        panels.append(np.zeros((480, 640, 3), dtype=np.uint8))

    height = panels[0].shape[0]
    normalized = []
    for panel in panels:
        if panel.shape[0] != height:
            scale = height / panel.shape[0]
            panel = cv2.resize(panel, (int(panel.shape[1] * scale), height), interpolation=cv2.INTER_NEAREST)
        normalized.append(panel)
    canvas = np.concatenate(normalized, axis=1)

    xyz = pose[:3, 3]
    status = (
        f"next #{frame_idx:06d} | rec {'on' if recording else 'off'} | "
        f"xyz [{xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}] | "
        f"move {move_step:.2f}m rot {rot_step:.1f}deg"
    )
    cv2.rectangle(canvas, (8, 8), (min(canvas.shape[1] - 8, 920), 42), (0, 0, 0), -1)
    cv2.putText(canvas, status, (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

    if canvas.shape[1] > preview_width:
        scale = preview_width / canvas.shape[1]
        canvas = cv2.resize(canvas, (preview_width, int(canvas.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return canvas


def print_controls() -> None:
    print(
        "\nControls\n"
        "  W/S: forward/back, A/D: left/right, R/F: up/down\n"
        "  Arrow keys or I/J/K/L: pitch/yaw, Q/E: roll\n"
        "  Space: save current frame, P: toggle auto-recording\n"
        "  [ / ]: decrease/increase move step, - / =: decrease/increase rotation step\n"
        "  H: print this help\n"
        "  Esc: quit\n"
    )


def main() -> None:
    args = parse_args()
    configure_gl_backend(args)
    load_runtime_dependencies()
    config_path = resolve_config_path(args)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if args.overwrite and args.append:
        raise ValueError("--overwrite and --append cannot be used together.")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir(args.dataset, args.scene)

    cfg = mmengine.Config.fromfile(str(config_path))
    update_camera_config(cfg, args)

    frame_idx = prepare_output_dir(output_dir, args.overwrite, args.append)
    mapper, num_classes, semantic_source = load_semantic_mapper(args.dataset, args.scene, cfg)
    if mapper is None and not args.semantic_raw:
        print("[WARN] Semantic mapping file was not found; semantic_map_*.npy will contain raw Habitat ids.")

    print(f"[INFO] Initializing Habitat-Sim from {config_path}")
    sim = init_simulator(cfg, args.gpu_device_id)
    print("[INFO] Simulator initialized.")
    start_pose, start_source = resolve_start_pose(args, cfg)
    intrinsics = get_pinhole_intrinsic(sim)
    np.savetxt(output_dir / "pinhole_intrinsics.txt", intrinsics, fmt="%.9g")
    write_metadata(output_dir, args, config_path, start_source, intrinsics, semantic_source)

    pose = start_pose
    obs = render(sim, pose)
    recording = not args.manual_save
    move_step = args.move_step
    rot_step = args.rot_step
    print_controls()
    print(f"[INFO] Output: {output_dir}")
    print(f"[INFO] Start pose source: {start_source}")

    if args.render_once:
        save_frame(output_dir, frame_idx, obs, pose, args.traj_format, mapper, num_classes, args.semantic_raw)
        print(f"[SAVE] frame {frame_idx:06d}")
        sim.close()
        print(f"[DONE] Saved 1 frame under {output_dir}")
        return

    if not args.no_save_initial and recording:
        save_frame(output_dir, frame_idx, obs, pose, args.traj_format, mapper, num_classes, args.semantic_raw)
        print(f"[SAVE] frame {frame_idx:06d}")
        frame_idx += 1

    cv2.namedWindow("sim_trajectory", cv2.WINDOW_NORMAL)
    try:
        while True:
            preview = compose_preview(obs, pose, frame_idx, recording, move_step, rot_step, args.preview_width, mapper, num_classes)
            cv2.imshow("sim_trajectory", preview)
            key = cv2.waitKeyEx(args.wait_ms)
            if key < 0:
                continue
            ch = chr(key & 0xFF).lower() if 0 <= (key & 0xFF) < 128 else ""
            if key == KEY_ESC:
                break
            if ch == "h":
                print_controls()
                continue
            if ch == "p":
                recording = not recording
                print(f"[INFO] auto-recording {'enabled' if recording else 'disabled'}")
                continue
            if ch == "[":
                move_step = max(move_step * 0.5, 0.005)
                print(f"[INFO] move step: {move_step:.3f} m")
                continue
            if ch == "]":
                move_step = min(move_step * 2.0, 5.0)
                print(f"[INFO] move step: {move_step:.3f} m")
                continue
            if ch == "-":
                rot_step = max(rot_step * 0.5, 0.25)
                print(f"[INFO] rot step: {rot_step:.3f} deg")
                continue
            if ch == "=":
                rot_step = min(rot_step * 2.0, 90.0)
                print(f"[INFO] rot step: {rot_step:.3f} deg")
                continue
            if key == KEY_SPACE:
                save_frame(output_dir, frame_idx, obs, pose, args.traj_format, mapper, num_classes, args.semantic_raw)
                print(f"[SAVE] frame {frame_idx:06d}")
                frame_idx += 1
                continue

            candidate_pose, changed = apply_key(pose, key, move_step, rot_step)
            if not changed:
                continue
            if not is_candidate_valid(sim, candidate_pose, args.require_navigable):
                print("[INFO] rejected non-navigable translation")
                continue

            pose = candidate_pose
            obs = render(sim, pose)
            if recording:
                save_frame(output_dir, frame_idx, obs, pose, args.traj_format, mapper, num_classes, args.semantic_raw)
                print(f"[SAVE] frame {frame_idx:06d}")
                frame_idx += 1
    finally:
        cv2.destroyAllWindows()
        sim.close()

    print(f"[DONE] Saved {frame_idx} frames under {output_dir}")


if __name__ == "__main__":
    main()
