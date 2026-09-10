"""
Render ceiling-to-floor bird's-eye-view maps from reconstructed SplaTAM outputs.

This script reads `params.npz` directly and produces three aligned BEV images:
- RGB
- Semantic
- Entropy

The default renderer uses orthographic Gaussian splatting on the visible BEV surface.
"""

import argparse
import csv
import json
import os
import struct
import typing

import matplotlib.pyplot as plt
import numpy as np


Bounds = typing.Tuple[float, float, float, float, int, int]
C0 = 0.28209479177387814


MP3D_HIGH_CONTRAST_COLORS = {
    0: [255, 255, 255],  # void
    1: [174, 199, 232],  # wall
    2: [152, 223, 138],  # floor
    3: [255, 187, 120],  # chair
    4: [214, 39, 40],    # door
    5: [148, 103, 189],  # table
    6: [140, 86, 75],    # picture
    7: [31, 119, 180],   # cabinet
    8: [188, 189, 34],   # cushion
    9: [44, 160, 44],    # window
    10: [255, 127, 14],  # sofa
    11: [227, 119, 194], # bed
    12: [127, 127, 127], # curtain
    13: [23, 190, 207],  # chest_of_drawers
    14: [219, 219, 141], # plant
    15: [197, 176, 213], # sink
    16: [196, 156, 148], # stairs
    17: [247, 182, 210], # ceiling
    18: [199, 199, 199], # toilet
    19: [158, 218, 229], # stool
    20: [57, 59, 121],   # towel
    21: [82, 84, 163],   # mirror
    22: [107, 110, 207], # tv_monitor
    23: [156, 158, 222], # shower
    24: [99, 121, 57],   # column
    25: [140, 162, 82],  # bathtub
    26: [181, 207, 107], # counter
    27: [206, 219, 156], # fireplace
    28: [140, 109, 49],  # lighting
    29: [189, 158, 57],  # beam
    30: [231, 186, 82],  # railing
    31: [231, 203, 148], # shelving
    32: [132, 60, 57],   # blinds
    33: [173, 73, 74],   # gym_equipment
    34: [214, 97, 107],  # seating
    35: [231, 150, 156], # board_panel
    36: [123, 65, 115],  # furniture
    37: [165, 81, 148],  # appliances
    38: [206, 109, 189], # clothes
    39: [222, 158, 214], # objects
    40: [49, 130, 189],  # misc
}


def eval_label_colormap(n_label: int = 256) -> np.ndarray:
    cmap = np.zeros((n_label, 3), dtype=np.uint8)
    for i in range(n_label):
        r = g = b = 0
        cid = i
        for j in range(8):
            r |= ((cid >> 0) & 1) << (7 - j)
            g |= ((cid >> 1) & 1) << (7 - j)
            b |= ((cid >> 2) & 1) << (7 - j)
            cid >>= 3
        cmap[i] = np.array([r, g, b], dtype=np.uint8)
    return cmap


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))


def decode_rgb_values(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors)
    if colors.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if np.issubdtype(colors.dtype, np.floating):
        colors = np.clip(colors, 0.0, 1.0)
        colors = (colors * 255.0).astype(np.uint8)
    else:
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    return colors


def decode_rgb_from_fdc(f_dc: np.ndarray) -> np.ndarray:
    f_dc = np.asarray(f_dc, dtype=np.float32)
    return decode_rgb_values(f_dc[:, :3] * C0 + 0.5)


def softmax_entropy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    prob = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + 1e-12)
    return (-np.sum(prob * np.log(np.clip(prob, 1e-12, 1.0)), axis=1)).astype(np.float32)


def normalize_label_map(raw_map: typing.Optional[dict]) -> typing.Optional[dict]:
    if raw_map is None:
        return None

    mappings = raw_map.get("mappings", raw_map) if isinstance(raw_map, dict) else None
    if not isinstance(mappings, dict):
        return None

    out = {}
    for key, value in mappings.items():
        if isinstance(value, dict):
            rgb = value.get("rgb", value.get("color"))
        else:
            rgb = value
        if rgb is None:
            continue
        arr = np.asarray(rgb)
        if arr.size != 3:
            continue
        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.01:
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        out[str(key)] = [int(arr[0]), int(arr[1]), int(arr[2])]
    return out


def load_npz_fields(npz_path: str) -> dict:
    data = np.load(npz_path, allow_pickle=True)

    if "means3D" in data:
        points = np.asarray(data["means3D"], dtype=np.float32)
    elif "points" in data:
        points = np.asarray(data["points"], dtype=np.float32)
    else:
        raise ValueError("No `means3D` or `points` found in npz")

    colors = None
    if all(key in data for key in ("f_dc_0", "f_dc_1", "f_dc_2")):
        f_dc = np.stack([
            np.asarray(data["f_dc_0"], dtype=np.float32),
            np.asarray(data["f_dc_1"], dtype=np.float32),
            np.asarray(data["f_dc_2"], dtype=np.float32),
        ], axis=1)
        colors = decode_rgb_from_fdc(f_dc)
    elif "features_dc" in data:
        f_dc = np.asarray(data["features_dc"], dtype=np.float32).reshape(points.shape[0], -1)
        if f_dc.shape[1] >= 3:
            colors = decode_rgb_from_fdc(f_dc)
    elif "rgb_colors" in data:
        colors = decode_rgb_values(np.asarray(data["rgb_colors"], dtype=np.float32))

    semantic_logits = None
    if "semantic_logits" in data:
        semantic_logits = np.asarray(data["semantic_logits"], dtype=np.float32)

    labels = None
    if "seman_cls_ids" in data:
        raw = np.asarray(data["seman_cls_ids"])
        if raw.ndim == 2 and raw.shape[1] > 1:
            if semantic_logits is not None and semantic_logits.shape == raw.shape:
                topk_idx = np.argmax(semantic_logits, axis=1)
                labels = raw[np.arange(raw.shape[0]), topk_idx].astype(np.int32)
            else:
                labels = raw[:, 0].astype(np.int32)
        else:
            labels = raw.reshape(-1).astype(np.int32)
    elif semantic_logits is not None:
        labels = np.argmax(semantic_logits, axis=1).astype(np.int32)

    entropy = None
    if "entropy" in data:
        entropy = np.asarray(data["entropy"], dtype=np.float32).reshape(-1)
    elif semantic_logits is not None:
        entropy = softmax_entropy(semantic_logits)

    gt_w2c = None
    if "gt_w2c_all_frames" in data:
        gt_w2c = np.asarray(data["gt_w2c_all_frames"], dtype=np.float32)
    elif "w2c" in data:
        gt_w2c = np.asarray(data["w2c"], dtype=np.float32)[None]

    scales = None
    if "log_scales" in data:
        log_scales = np.asarray(data["log_scales"], dtype=np.float32)
        if log_scales.ndim == 1:
            log_scales = log_scales[:, None]
        scales = np.exp(log_scales)

    opacities = None
    if "logit_opacities" in data:
        opacities = sigmoid(np.asarray(data["logit_opacities"], dtype=np.float32)).reshape(-1)

    return {
        "points": points,
        "colors": colors,
        "labels": labels,
        "entropy": entropy,
        "semantic_logits": semantic_logits,
        "scales": scales,
        "opacities": opacities,
        "intrinsics": np.asarray(data["intrinsics"], dtype=np.float32) if "intrinsics" in data else None,
        "w2c": np.asarray(data["w2c"], dtype=np.float32) if "w2c" in data else None,
        "gt_w2c_all_frames": gt_w2c,
    }


def load_first_traj_pose(traj_path: str) -> np.ndarray:
    with open(traj_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values = np.fromstring(line, sep=" ", dtype=np.float32)
            if values.size != 16:
                raise ValueError(f"Expected 16 values in first trajectory pose, got {values.size}")
            return values.reshape(4, 4)
    raise ValueError(f"Empty trajectory file: {traj_path}")


def load_replica_first_traj_pose(traj_path: str) -> np.ndarray:
    return load_first_traj_pose(traj_path)


def load_mp3d_first_traj_pose(traj_path: str) -> np.ndarray:
    return load_first_traj_pose(traj_path)


def resolve_replica_gt_semantic_paths(scene: str) -> typing.Tuple[str, str]:
    scene_dir = f"{scene[:-1]}_{scene[-1]}"
    base = os.path.join("data", "replica_v1", scene_dir, "habitat")
    mesh_path = os.path.join(base, "mesh_semantic.ply")
    info_path = os.path.join(base, "info_semantic.json")
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"GT semantic mesh not found: {mesh_path}")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"GT semantic info not found: {info_path}")
    return mesh_path, info_path


def load_replica_gt_semantic(scene: str) -> typing.Tuple[np.ndarray, np.ndarray]:
    mesh_path, info_path = resolve_replica_gt_semantic_paths(scene)

    with open(info_path, "r", encoding="utf-8") as f:
        info_semantic = json.load(f)
    id_to_label = np.asarray(info_semantic["id_to_label"], dtype=np.int32)

    vertex_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])

    with open(mesh_path, "rb") as f:
        header = []
        vertex_count = None
        face_count = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading PLY header: {mesh_path}")
            header.append(line)
            line_str = line.decode("latin1").strip()
            if line_str.startswith("element vertex"):
                vertex_count = int(line_str.split()[-1])
            elif line_str.startswith("element face"):
                face_count = int(line_str.split()[-1])
            elif line_str == "end_header":
                break

        if vertex_count is None or face_count is None:
            raise ValueError(f"Missing vertex/face count in {mesh_path}")

        vertex_data = np.fromfile(f, dtype=vertex_dtype, count=vertex_count)
        vertices = np.stack([vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=-1).astype(np.float32)
        face_blob = f.read()

    tri_centroids = []
    tri_labels = []
    offset = 0
    blob_len = len(face_blob)
    for _ in range(face_count):
        if offset >= blob_len:
            break
        nverts = face_blob[offset]
        offset += 1
        verts = struct.unpack_from("<" + "I" * nverts, face_blob, offset)
        offset += 4 * nverts
        object_id = struct.unpack_from("<H", face_blob, offset)[0]
        offset += 2

        if object_id < 0 or object_id >= id_to_label.shape[0]:
            continue
        label = int(id_to_label[object_id])
        if nverts == 3:
            triangles = [verts]
        elif nverts == 4:
            i0, i1, i2, i3 = verts
            triangles = [(i0, i1, i2), (i0, i2, i3)]
        else:
            continue
        for tri in triangles:
            tri_pts = vertices[np.asarray(tri, dtype=np.int32)]
            tri_centroids.append(tri_pts.mean(axis=0))
            tri_labels.append(label)

    if not tri_centroids:
        raise ValueError(f"No GT semantic triangles loaded from {mesh_path}")
    return np.asarray(tri_centroids, dtype=np.float32), np.asarray(tri_labels, dtype=np.int32)


PLY_NUMPY_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}

PLY_STRUCT_CODES = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def _ply_scalar_dtype(type_name: str) -> str:
    if type_name not in PLY_NUMPY_DTYPES:
        raise ValueError(f"Unsupported PLY scalar type: {type_name}")
    return PLY_NUMPY_DTYPES[type_name]


def _ply_struct_code(type_name: str) -> str:
    if type_name not in PLY_STRUCT_CODES:
        raise ValueError(f"Unsupported PLY scalar type: {type_name}")
    return PLY_STRUCT_CODES[type_name]


def load_mpcat40_label_map(tsv_path: str = os.path.join("configs", "MP3D", "mpcat40.tsv")) -> typing.Optional[dict]:
    """Load MP-CAT40 label colors from the repository TSV as label -> RGB JSON-style map."""
    if not os.path.exists(tsv_path):
        return None

    label_map = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                label_id = int(row["mpcat40index"])
            except Exception:
                continue
            hex_color = row.get("hex", "").strip()
            if len(hex_color) != 7 or not hex_color.startswith("#"):
                continue
            try:
                rgb = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
            except ValueError:
                continue
            label_map[str(label_id)] = rgb
    return label_map


def load_mp3d_label_map(palette: str) -> typing.Optional[dict]:
    if palette == "high-contrast":
        return {str(label): rgb for label, rgb in MP3D_HIGH_CONTRAST_COLORS.items()}
    if palette == "mpcat40":
        return load_mpcat40_label_map()
    if palette == "eval":
        return None
    raise ValueError(f"Unsupported MP3D palette: {palette}")


def resolve_mp3d_gt_semantic_paths(scene: str) -> typing.Tuple[str, str, str, str]:
    task_root = os.path.join("data", "MP3D", "v1", "tasks", "mp3d", scene)
    mesh_candidates = [
        os.path.join(task_root, "semantic_clean.ply"),
        os.path.join(task_root, f"{scene}_semantic.ply"),
    ]
    mesh_path = next((p for p in mesh_candidates if os.path.exists(p)), None)
    if mesh_path is None:
        raise FileNotFoundError(f"MP3D GT semantic mesh not found. Searched: {mesh_candidates}")

    semseg_path = os.path.join(
        "data", "MP3D", "v1", "scans", scene, scene, "house_segmentations", f"{scene}.semseg.json"
    )
    category_mapping_path = os.path.join("configs", "MP3D", "category_mapping.tsv")
    instance_map_path = os.path.join("configs", "MP3D", scene, "instance_to_mpcat40.json")
    return mesh_path, semseg_path, category_mapping_path, instance_map_path


def load_mp3d_instance_to_mpcat40(
    scene: str,
    semseg_path: str,
    category_mapping_path: str,
    instance_map_path: str,
) -> dict:
    """Return MP3D object instance id -> MP-CAT40 id.

    Prefer a pre-generated configs/MP3D/<scene>/instance_to_mpcat40.json when present,
    otherwise derive it from Matterport semseg metadata and category_mapping.tsv.
    """
    if os.path.exists(instance_map_path):
        with open(instance_map_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): int(v) for k, v in raw.items()}

    if not os.path.exists(semseg_path):
        raise FileNotFoundError(f"MP3D semseg JSON not found: {semseg_path}")
    if not os.path.exists(category_mapping_path):
        raise FileNotFoundError(f"MP3D category mapping TSV not found: {category_mapping_path}")

    label_to_mpcat40 = {}
    with open(category_mapping_path, "r", encoding="utf-8") as f:
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

    with open(semseg_path, "r", encoding="utf-8") as f:
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


def _read_binary_ply_vertices_and_faces(
    ply_path: str,
) -> typing.Tuple[np.ndarray, bytes, int, typing.List[typing.Tuple]]:
    """Read binary little-endian PLY vertices and raw face blob.

    The MP3D semantic mesh stores semantic ids on faces, so the face records are
    parsed separately by load_mp3d_gt_semantic.
    """
    with open(ply_path, "rb") as f:
        vertex_count = None
        face_count = None
        current_element = None
        vertex_props = []
        face_props = []

        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading PLY header: {ply_path}")
            line_str = line.decode("latin1").strip()
            parts = line_str.split()
            if len(parts) >= 3 and parts[0] == "format" and parts[1] != "binary_little_endian":
                raise ValueError(f"Only binary_little_endian PLY is supported: {ply_path}")
            if len(parts) >= 3 and parts[0] == "element":
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
                elif current_element == "face":
                    face_count = int(parts[2])
            elif parts and parts[0] == "property":
                if current_element == "vertex" and len(parts) == 3:
                    vertex_props.append((parts[2], parts[1]))
                elif current_element == "face":
                    if len(parts) == 5 and parts[1] == "list":
                        face_props.append(("list", parts[4], parts[2], parts[3]))
                    elif len(parts) == 3:
                        face_props.append(("scalar", parts[2], parts[1]))
            elif line_str == "end_header":
                break

        if vertex_count is None or face_count is None:
            raise ValueError(f"Missing vertex/face count in {ply_path}")
        if not vertex_props:
            raise ValueError(f"No vertex properties found in {ply_path}")
        if not face_props:
            raise ValueError(f"No face properties found in {ply_path}")

        vertex_dtype = np.dtype([(name, _ply_scalar_dtype(type_name)) for name, type_name in vertex_props])
        vertex_data = np.fromfile(f, dtype=vertex_dtype, count=vertex_count)
        missing_xyz = [name for name in ("x", "y", "z") if name not in vertex_data.dtype.names]
        if missing_xyz:
            raise ValueError(f"Missing vertex properties {missing_xyz} in {ply_path}")
        vertices = np.stack([vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=-1).astype(np.float32)
        face_blob = f.read()
    return vertices, face_blob, face_count, face_props


def load_mp3d_gt_semantic(scene: str) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Load MP3D semantic GT as triangle centroids and MP-CAT40 labels."""
    mesh_path, semseg_path, category_mapping_path, instance_map_path = resolve_mp3d_gt_semantic_paths(scene)
    instance_to_mpcat40 = load_mp3d_instance_to_mpcat40(
        scene=scene,
        semseg_path=semseg_path,
        category_mapping_path=category_mapping_path,
        instance_map_path=instance_map_path,
    )
    vertices, face_blob, face_count, face_props = _read_binary_ply_vertices_and_faces(mesh_path)

    tri_centroids = []
    tri_labels = []
    offset = 0
    blob_len = len(face_blob)
    for _ in range(face_count):
        if offset >= blob_len:
            break
        verts = None
        object_id = None
        for prop in face_props:
            if prop[0] == "list":
                _, name, count_type, value_type = prop
                count_code = _ply_struct_code(count_type)
                value_code = _ply_struct_code(value_type)
                count_size = struct.calcsize("<" + count_code)
                nvalues = struct.unpack_from("<" + count_code, face_blob, offset)[0]
                offset += count_size
                values_size = struct.calcsize("<" + value_code) * nvalues
                values = struct.unpack_from("<" + value_code * nvalues, face_blob, offset)
                offset += values_size
                if name == "vertex_indices":
                    verts = values
            else:
                _, name, type_name = prop
                code = _ply_struct_code(type_name)
                size = struct.calcsize("<" + code)
                value = struct.unpack_from("<" + code, face_blob, offset)[0]
                offset += size
                if name == "object_id":
                    object_id = int(value)

        if verts is None or object_id is None or len(verts) < 3:
            continue
        label = int(instance_to_mpcat40.get(int(object_id), 0))
        for i in range(1, len(verts) - 1):
            tri = np.asarray((verts[0], verts[i], verts[i + 1]), dtype=np.int64)
            if np.any(tri < 0) or np.any(tri >= vertices.shape[0]):
                continue
            tri_pts = vertices[tri]
            tri_centroids.append(tri_pts.mean(axis=0))
            tri_labels.append(label)

    if not tri_centroids:
        raise ValueError(f"No MP3D GT semantic triangles loaded from {mesh_path}")
    return np.asarray(tri_centroids, dtype=np.float32), np.asarray(tri_labels, dtype=np.int32)


def load_gt_semantic(dataset: str, scene: str) -> typing.Tuple[np.ndarray, np.ndarray]:
    dataset_lower = dataset.lower()
    if dataset_lower == "replica":
        return load_replica_gt_semantic(scene)
    if dataset_lower == "mp3d":
        return load_mp3d_gt_semantic(scene)
    raise ValueError(f"GT semantic BEV is not implemented for dataset `{dataset}`")


def render_gt_semantic_bev(
    gt_points: np.ndarray,
    gt_labels: np.ndarray,
    bounds: Bounds,
    res: float,
    flip: bool,
    label_map: typing.Optional[dict],
    interpolation_sigma_px: float = 1.0,
    front_cutoff: typing.Optional[float] = None,
    min_height: typing.Optional[float] = None,
    max_height: typing.Optional[float] = None,
) -> np.ndarray:
    keep = np.ones(gt_points.shape[0], dtype=bool)
    view_nearness = gt_points[:, 2] if not flip else -gt_points[:, 2]
    if front_cutoff is not None:
        keep &= view_nearness <= float(front_cutoff)
    if min_height is not None:
        keep &= gt_points[:, 2] >= float(min_height)
    if max_height is not None:
        keep &= gt_points[:, 2] <= float(max_height)

    gt_points = gt_points[keep]
    gt_labels = gt_labels[keep]
    if gt_points.shape[0] == 0:
        raise ValueError("No GT semantic points remain after view-based filtering")

    px, py, ix, iy, linear = project_points_to_bev(gt_points, bounds=bounds, res=res)
    chosen_idx = top_surface_indices(linear, gt_points[:, 2], flip=flip)
    semantic_lut = label_colors(gt_labels, label_map)
    semantic_features = np.stack([semantic_lut[int(v)] for v in gt_labels[chosen_idx]], axis=0).astype(np.float32) / 255.0
    sigma_px = np.full(chosen_idx.shape[0], interpolation_sigma_px, dtype=np.float32)
    alpha = np.ones(chosen_idx.shape[0], dtype=np.float32)
    image = gaussian_splat_features(
        semantic_features,
        px[chosen_idx],
        py[chosen_idx],
        sigma_px,
        alpha,
        bounds,
        max_kernel_radius=max(1, int(np.ceil(3.0 * interpolation_sigma_px))),
        background_value=1.0,
    )
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def replica_slam_to_sim_transform(traj_path: typing.Optional[str] = None) -> np.ndarray:
    if traj_path is not None:
        return load_replica_first_traj_pose(traj_path)
    transform = np.eye(4, dtype=np.float32)
    transform[1, 1] = -1.0
    transform[2, 2] = -1.0
    return transform


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    points_h = np.concatenate([points, ones], axis=1)
    transformed = (transform @ points_h.T).T
    return transformed[:, :3].astype(np.float32)


def mp3d_slam_to_sim_transform(traj_path: str) -> np.ndarray:
    return load_mp3d_first_traj_pose(traj_path)


def convert_points_to_sim(
    points: np.ndarray,
    dataset: str,
    coord_system: str,
    traj_path: typing.Optional[str] = None,
) -> np.ndarray:
    if coord_system == "slam":
        return points.astype(np.float32)
    if coord_system != "sim":
        raise ValueError(f"Unsupported coord_system: {coord_system}")
    dataset_lower = dataset.lower()
    if dataset_lower == "replica":
        return apply_transform(points, replica_slam_to_sim_transform(traj_path))
    if dataset_lower == "mp3d":
        if traj_path is None:
            raise ValueError("MP3D --coord-system sim requires --traj or --scene to resolve data/mp3d_sim_nvs_v2/<scene>/traj.txt")
        return apply_transform(points, mp3d_slam_to_sim_transform(traj_path))
    raise ValueError(
        f"No built-in SLAM->sim transform for dataset `{dataset}`. "
        "Use --coord-system slam or extend the transform logic."
    )


def compute_bounds(points_xy: np.ndarray, res: float, pad: float) -> Bounds:
    xmin = float(points_xy[:, 0].min())
    xmax = float(points_xy[:, 0].max())
    ymin = float(points_xy[:, 1].min())
    ymax = float(points_xy[:, 1].max())
    dx = xmax - xmin
    dy = ymax - ymin
    pad_x = max(dx * pad, res)
    pad_y = max(dy * pad, res)
    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y
    width = max(1, int(np.ceil((xmax - xmin) / res)))
    height = max(1, int(np.ceil((ymax - ymin) / res)))
    return xmin, xmax, ymin, ymax, width, height


def project_points_to_bev(points: np.ndarray, bounds: Bounds, res: float) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xmin, _, ymin, _, width, height = bounds
    px = (points[:, 0] - xmin) / res
    py = (points[:, 1] - ymin) / res
    ix = np.clip(np.floor(px).astype(np.int64), 0, width - 1)
    iy = np.clip(np.floor(py).astype(np.int64), 0, height - 1)
    linear = iy * width + ix
    return px, py, ix, iy, linear


def top_surface_indices(linear: np.ndarray, depth: np.ndarray, flip: bool) -> np.ndarray:
    order = np.argsort(depth if flip else -depth)
    linear_sorted = linear[order]
    _, first = np.unique(linear_sorted, return_index=True)
    return order[first]


def label_colors(labels: np.ndarray, label_map: typing.Optional[dict]) -> dict:
    unique_labels = np.unique(labels)
    color_map = {}
    max_label = int(unique_labels.max()) if unique_labels.size else 0
    eval_colormap = eval_label_colormap(max(max_label + 1, 256))
    for label in unique_labels:
        key = str(int(label))
        if label_map is not None:
            lookup_key = key if key in label_map else str(int(label) + 1)
            if lookup_key in label_map:
                color_map[int(label)] = np.asarray(label_map[lookup_key], dtype=np.uint8)
                continue
        color_map[int(label)] = eval_colormap[int(label)]
    return color_map


def visible_scales_px(scales: typing.Optional[np.ndarray], chosen_idx: np.ndarray, res: float, min_sigma_px: float, max_sigma_px: float) -> np.ndarray:
    if scales is None:
        sigma = np.full(chosen_idx.shape[0], min_sigma_px, dtype=np.float32)
    else:
        chosen_scales = np.asarray(scales[chosen_idx], dtype=np.float32)
        if chosen_scales.ndim == 1:
            scale_xy = chosen_scales
        elif chosen_scales.shape[1] == 1:
            scale_xy = chosen_scales[:, 0]
        else:
            scale_xy = np.mean(chosen_scales[:, :2], axis=1)
        sigma = scale_xy / max(res, 1e-8)
        sigma = np.asarray(sigma, dtype=np.float32)
        sigma = np.clip(sigma, min_sigma_px, max_sigma_px)
    return sigma


def visible_opacity(opacities: typing.Optional[np.ndarray], chosen_idx: np.ndarray) -> np.ndarray:
    if opacities is None:
        return np.full(chosen_idx.shape[0], 1.0, dtype=np.float32)
    alpha = np.asarray(opacities[chosen_idx], dtype=np.float32)
    return np.clip(alpha, 1e-3, 0.999)


def gaussian_splat_features(
    feature_values: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    sigma_px: np.ndarray,
    alpha: np.ndarray,
    bounds: Bounds,
    max_kernel_radius: int,
    background_value: float,
) -> np.ndarray:
    _, _, _, _, width, height = bounds
    num_pixels = width * height
    channels = 1 if feature_values.ndim == 1 else feature_values.shape[1]
    values = feature_values.reshape(feature_values.shape[0], channels).astype(np.float32)

    numer = np.zeros((channels, num_pixels), dtype=np.float32)
    denom = np.zeros(num_pixels, dtype=np.float32)

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

            denom += np.bincount(target, weights=weight, minlength=num_pixels).astype(np.float32)
            for c in range(channels):
                numer[c] += np.bincount(target, weights=weight * values[valid, c], minlength=num_pixels).astype(np.float32)

    out = np.full((channels, num_pixels), background_value, dtype=np.float32)
    mask = denom > 1e-8
    out[:, mask] = numer[:, mask] / denom[mask]
    out = out.reshape(channels, height, width)
    out = np.transpose(out, (1, 2, 0))
    if channels == 1:
        out = out[:, :, 0]
    return np.flipud(out)


def render_nearest_rgb(colors: np.ndarray, chosen_idx: np.ndarray, ix: np.ndarray, iy: np.ndarray, bounds: Bounds) -> np.ndarray:
    _, _, _, _, width, height = bounds
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    image[iy[chosen_idx], ix[chosen_idx]] = colors[chosen_idx]
    return np.flipud(image)


def render_nearest_semantic(labels: np.ndarray, chosen_idx: np.ndarray, ix: np.ndarray, iy: np.ndarray, bounds: Bounds, label_map: typing.Optional[dict]) -> np.ndarray:
    _, _, _, _, width, height = bounds
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    lut = label_colors(labels, label_map)
    for idx in chosen_idx:
        image[iy[idx], ix[idx]] = lut[int(labels[idx])]
    return np.flipud(image)


def render_nearest_entropy(entropy: np.ndarray, chosen_idx: np.ndarray, ix: np.ndarray, iy: np.ndarray, bounds: Bounds) -> np.ndarray:
    _, _, _, _, width, height = bounds
    image = np.zeros((height, width), dtype=np.float32)
    values = entropy[chosen_idx]
    if values.size:
        vmin = float(values.min())
        vmax = float(values.max())
        if vmax > vmin + 1e-8:
            values = (values - vmin) / (vmax - vmin)
        else:
            values = np.zeros_like(values)
        image[iy[chosen_idx], ix[chosen_idx]] = values
    return np.flipud(image)


def render_gaussian_panels(
    points: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
    entropy: np.ndarray,
    scales: typing.Optional[np.ndarray],
    opacities: typing.Optional[np.ndarray],
    bounds: Bounds,
    res: float,
    flip: bool,
    label_map: typing.Optional[dict],
    min_sigma_px: float,
    max_sigma_px: float,
    max_kernel_radius: int,
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    px, py, ix, iy, linear = project_points_to_bev(points, bounds=bounds, res=res)
    chosen_idx = top_surface_indices(linear, points[:, 2], flip=flip)

    chosen_px = px[chosen_idx]
    chosen_py = py[chosen_idx]
    sigma_px = visible_scales_px(scales, chosen_idx, res=res, min_sigma_px=min_sigma_px, max_sigma_px=max_sigma_px)
    alpha = visible_opacity(opacities, chosen_idx)

    rgb_features = colors[chosen_idx].astype(np.float32) / 255.0
    rgb_img = gaussian_splat_features(
        rgb_features,
        chosen_px,
        chosen_py,
        sigma_px,
        alpha,
        bounds,
        max_kernel_radius=max_kernel_radius,
        background_value=1.0,
    )
    rgb_img = np.clip(rgb_img * 255.0, 0, 255).astype(np.uint8)

    semantic_lut = label_colors(labels, label_map)
    semantic_features = np.stack([semantic_lut[int(v)] for v in labels[chosen_idx]], axis=0).astype(np.float32) / 255.0
    semantic_img = gaussian_splat_features(
        semantic_features,
        chosen_px,
        chosen_py,
        sigma_px,
        alpha,
        bounds,
        max_kernel_radius=max_kernel_radius,
        background_value=1.0,
    )
    semantic_img = np.clip(semantic_img * 255.0, 0, 255).astype(np.uint8)

    entropy_values = entropy[chosen_idx].astype(np.float32)
    if entropy_values.size:
        emin = float(entropy_values.min())
        emax = float(entropy_values.max())
        if emax > emin + 1e-8:
            entropy_values = (entropy_values - emin) / (emax - emin)
        else:
            entropy_values = np.zeros_like(entropy_values)
    entropy_img = gaussian_splat_features(
        entropy_values,
        chosen_px,
        chosen_py,
        sigma_px,
        alpha,
        bounds,
        max_kernel_radius=max_kernel_radius,
        background_value=0.0,
    )
    entropy_img = np.clip(entropy_img, 0.0, 1.0)

    return rgb_img, semantic_img, entropy_img, int(chosen_idx.shape[0])


def save_rgb(path: str, image: np.ndarray, dpi: int, interpolation: str) -> None:
    plt.figure(figsize=(8, 8))
    plt.imshow(image, interpolation=interpolation)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_entropy(path: str, image: np.ndarray, dpi: int, interpolation: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(image, cmap="viridis", interpolation=interpolation)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def derive_output_paths(out_path: str) -> typing.Tuple[str, str, str, str]:
    root, ext = os.path.splitext(out_path)
    if not ext:
        ext = ".png"
    return root + "_rgb" + ext, root + "_semantic" + ext, root + "_entropy" + ext, root + "_gt_semantic" + ext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="path to params.npz")
    parser.add_argument("--dataset", default="Replica", help="dataset name")
    parser.add_argument("--scene", default=None, help="scene name, used to auto-resolve trajectory pose")
    parser.add_argument("--traj", default=None, help="optional trajectory txt used by exploration-map coordinates")
    parser.add_argument("--out", required=True, help="output path prefix, e.g. xxx/bev.png")
    parser.add_argument("--coord-system", choices=["sim", "slam"], default="sim", help="render points in sim or raw slam coordinates")
    parser.add_argument("--render-mode", choices=["gaussian", "nearest"], default="gaussian", help="BEV renderer: visible Gaussian splatting or nearest-cell assignment")
    parser.add_argument("--res", type=float, default=0.01, help="meters per pixel")
    parser.add_argument("--pad", type=float, default=0.05, help="XY bound padding ratio")
    parser.add_argument("--ceiling-percentile", type=float, default=None, help="legacy option: keep points with z <= this percentile after coordinate conversion")
    parser.add_argument("--remove-front-percent", type=float, default=0.5, help="remove the nearest N percent of points along the current viewing direction after coordinate conversion")
    parser.add_argument("--min-height", type=float, default=None, help="optional lower z bound after coord conversion")
    parser.add_argument("--max-height", type=float, default=None, help="optional upper z bound after coord conversion")
    parser.add_argument("--label-map", default=None, help="JSON label->RGB mapping")
    parser.add_argument("--flip", action="store_true", help="flip the viewing direction to use the lowest visible surface instead of the highest")
    parser.add_argument("--min-sigma-px", type=float, default=0.75, help="minimum projected Gaussian sigma in pixels")
    parser.add_argument("--max-sigma-px", type=float, default=3.0, help="maximum projected Gaussian sigma in pixels")
    parser.add_argument("--max-kernel-radius", type=int, default=4, help="maximum Gaussian kernel radius in pixels")
    parser.add_argument("--interpolation", choices=["nearest", "bilinear", "bicubic", "lanczos"], default="lanczos", help="image interpolation used when saving the BEV panels")
    parser.add_argument("--render-gt-semantic", action="store_true", help="also render GT semantic from the dataset semantic mesh in the same BEV view")
    parser.add_argument("--mp3d-palette", choices=["high-contrast", "mpcat40", "eval"], default="high-contrast", help="semantic color palette used for MP3D when --label-map is not provided")
    parser.add_argument("--dpi", type=int, default=500, help="output DPI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_parent(args.out)

    fields = load_npz_fields(args.npz)

    traj_path = args.traj
    dataset_lower = args.dataset.lower()
    if traj_path is None and args.coord_system == "sim" and args.scene:
        if dataset_lower == "replica":
            candidate = os.path.join("data", "Replica", args.scene, "traj.txt")
        elif dataset_lower == "mp3d":
            candidate = os.path.join("data", "mp3d_sim_nvs_v2", args.scene, "traj.txt")
        else:
            candidate = None
        if candidate is not None and os.path.exists(candidate):
            traj_path = candidate

    points_sim = convert_points_to_sim(
        fields["points"],
        dataset=args.dataset,
        coord_system=args.coord_system,
        traj_path=traj_path,
    )

    keep = np.ones(points_sim.shape[0], dtype=bool)
    ceiling_z = None
    removed_front_percent = None
    front_cutoff = None
    view_nearness = points_sim[:, 2] if not args.flip else -points_sim[:, 2]
    if args.remove_front_percent is not None:
        if not (0.0 <= args.remove_front_percent < 100.0):
            raise ValueError("--remove-front-percent must be in [0, 100)")
        removed_front_percent = float(args.remove_front_percent)
        front_cutoff = float(np.percentile(view_nearness, 100.0 - removed_front_percent))
        keep &= view_nearness <= front_cutoff
    elif args.ceiling_percentile is not None:
        ceiling_z = float(np.percentile(points_sim[:, 2], args.ceiling_percentile))
        keep &= points_sim[:, 2] <= ceiling_z

    if args.min_height is not None:
        keep &= points_sim[:, 2] >= float(args.min_height)
    if args.max_height is not None:
        keep &= points_sim[:, 2] <= float(args.max_height)

    points_sim = points_sim[keep]
    if points_sim.shape[0] == 0:
        raise ValueError("No points remain after filtering")

    colors = fields["colors"]
    labels = fields["labels"]
    entropy = fields["entropy"]
    if colors is None:
        raise ValueError("RGB colors are missing from params.npz")
    if labels is None:
        raise ValueError("Semantic labels are missing; expected `seman_cls_ids` or `semantic_logits`")
    if entropy is None:
        raise ValueError("Entropy is missing and could not be derived from `semantic_logits`")

    colors = colors[keep]
    labels = labels[keep]
    entropy = entropy[keep]
    scales = fields["scales"]
    if scales is not None:
        scales = scales[keep]
    opacities = fields["opacities"]
    if opacities is not None:
        opacities = opacities[keep]

    bounds = compute_bounds(points_sim[:, :2], res=args.res, pad=args.pad)

    label_map = None
    if args.label_map is not None:
        with open(args.label_map, "r", encoding="utf-8") as f:
            label_map = normalize_label_map(json.load(f))
    elif dataset_lower == "mp3d":
        label_map = load_mp3d_label_map(args.mp3d_palette)

    if args.render_mode == "gaussian":
        rgb_img, semantic_img, entropy_img, visible_count = render_gaussian_panels(
            points=points_sim,
            colors=colors,
            labels=labels,
            entropy=entropy,
            scales=scales,
            opacities=opacities,
            bounds=bounds,
            res=args.res,
            flip=args.flip,
            label_map=label_map,
            min_sigma_px=args.min_sigma_px,
            max_sigma_px=args.max_sigma_px,
            max_kernel_radius=args.max_kernel_radius,
        )
    else:
        px, py, ix, iy, linear = project_points_to_bev(points_sim, bounds=bounds, res=args.res)
        chosen_idx = top_surface_indices(linear, points_sim[:, 2], flip=args.flip)
        visible_count = int(chosen_idx.shape[0])
        rgb_img = render_nearest_rgb(colors, chosen_idx, ix, iy, bounds)
        semantic_img = render_nearest_semantic(labels, chosen_idx, ix, iy, bounds, label_map)
        entropy_img = render_nearest_entropy(entropy, chosen_idx, ix, iy, bounds)

    rgb_path, semantic_path, entropy_path, gt_semantic_path = derive_output_paths(args.out)
    save_rgb(rgb_path, rgb_img, dpi=args.dpi, interpolation=args.interpolation)
    save_rgb(semantic_path, semantic_img, dpi=args.dpi, interpolation=args.interpolation)
    save_entropy(entropy_path, entropy_img, dpi=args.dpi, interpolation=args.interpolation)

    gt_semantic_saved = False
    if args.render_gt_semantic:
        if not args.scene:
            raise ValueError("--render-gt-semantic requires --scene")
        if dataset_lower == "mp3d" and args.coord_system != "sim":
            raise ValueError("MP3D GT semantic mesh is in sim/world coordinates; use --coord-system sim")
        gt_points, gt_labels = load_gt_semantic(args.dataset, args.scene)
        gt_semantic_img = render_gt_semantic_bev(
            gt_points=gt_points,
            gt_labels=gt_labels,
            bounds=bounds,
            res=args.res,
            flip=args.flip,
            label_map=label_map,
            interpolation_sigma_px=max(1.0, args.min_sigma_px),
            front_cutoff=front_cutoff,
            min_height=args.min_height,
            max_height=args.max_height,
        )
        save_rgb(gt_semantic_path, gt_semantic_img, dpi=args.dpi, interpolation=args.interpolation)
        gt_semantic_saved = True

    print(f"Input npz        : {args.npz}")
    print(f"Coord system     : {args.coord_system} ({args.dataset})")
    print(f"Render mode      : {args.render_mode}")
    if traj_path is not None:
        print(f"Trajectory pose  : {traj_path}")
    elif args.coord_system == "sim" and args.dataset.lower() == "replica":
        print("Trajectory pose  : not provided, fallback to axis-flip only")
    elif args.coord_system == "sim" and args.dataset.lower() == "mp3d":
        print("Trajectory pose  : not provided")
    if ceiling_z is not None:
        print(f"Ceiling z cutoff : {ceiling_z:.4f}")
    if front_cutoff is not None:
        print(f"Front cutoff     : {front_cutoff:.4f} (near-side threshold)")
    if removed_front_percent is not None:
        print(f"Removed front %  : {removed_front_percent:.2f}")
    print(f"View direction   : {'bottom-up (flip)' if args.flip else 'top-down'}")
    print(f"Points kept      : {points_sim.shape[0]}")
    print(f"Visible splats   : {visible_count}")
    print(f"BEV size         : {bounds[5]} x {bounds[4]}")
    print(f"Saved RGB        : {rgb_path}")
    print(f"Saved Semantic   : {semantic_path}")
    print(f"Saved Entropy    : {entropy_path}")
    if gt_semantic_saved:
        print(f"Saved GT Semantic: {gt_semantic_path}")


"""
python src/visualization/render_rgb_sem_entro_bev.py   --npz results/Replica/room2/SemanticHeat/run_0/splatam/exploration_stage_1/params.npz   --dataset Replica   --scene room2   --out results/Replica/room2/SemanticHeat/run_0/visualization/bev.png   --coord-system sim   --render-mode gaussian   --remove-front-percent 25   --render-gt-semantic
"""

if __name__ == "__main__":
    main()
