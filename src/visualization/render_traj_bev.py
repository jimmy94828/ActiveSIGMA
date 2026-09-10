"""
Trajectory BEV renderer.

Renders a top-down RGB or semantic BEV background from reconstructed SplaTAM
outputs, then overlays the camera trajectory in the same coordinate frame.
"""

import argparse
import glob
import json
import os
import re
import typing

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from plyfile import PlyData

C0 = 0.28209479177387814


def decode_rgb_values(colors: np.ndarray) -> np.ndarray:
	colors = np.asarray(colors)
	if colors.size == 0:
		return np.zeros((0, 3), dtype=np.uint8)
	if np.issubdtype(colors.dtype, np.floating):
		colors = np.clip(colors, 0.0, 1.0)
		return (colors * 255.0).astype(np.uint8)
	return np.clip(colors, 0, 255).astype(np.uint8)


def decode_rgb_from_fdc(f_dc: np.ndarray) -> np.ndarray:
	f_dc = np.asarray(f_dc, dtype=np.float32)
	return decode_rgb_values(f_dc[:, :3] * C0 + 0.5)


def _extract_step_id(path: str, prefix: str) -> int:
	"""Extract numeric step id from files like <prefix>_0900.ply; returns -1 if not found."""
	name = os.path.basename(path)
	m = re.match(rf"^{re.escape(prefix)}_(\d+)\.ply$", name)
	if m is None:
		return -1
	return int(m.group(1))


def _pick_latest_ply(splatam_dir: str, prefix: str) -> str:
	candidates = glob.glob(os.path.join(splatam_dir, f"{prefix}_*.ply"))
	if not candidates:
		raise FileNotFoundError(f"No {prefix}_*.ply found in {splatam_dir}")
	candidates.sort(key=lambda p: _extract_step_id(p, prefix))
	return candidates[-1]


def resolve_io_paths(
	dataset: str,
	scene: str,
	method: str,
	run_version: str,
	out_override: typing.Optional[str],
	results_root: str = "auto",
):
	"""Resolve npz/semantic/output paths from high-level run identifiers."""
	if results_root and results_root != "auto":
		root_candidates = [results_root]
	else:
		root_candidates = ["results", f"results_{dataset}", f"results_{dataset.upper()}"]

	run_root = None
	splatam_dir = None
	npz_path = None
	for root in dict.fromkeys(root_candidates):
		candidate_run_root = os.path.join(root, dataset, scene, method, run_version)
		candidate_splatam_dir = os.path.join(candidate_run_root, "splatam")
		candidate_npz = os.path.join(candidate_splatam_dir, "exploration_stage_1", "params.npz")
		if os.path.exists(candidate_npz):
			run_root = candidate_run_root
			splatam_dir = candidate_splatam_dir
			npz_path = candidate_npz
			break

	if run_root is None or splatam_dir is None or npz_path is None:
		searched = [
			os.path.join(root, dataset, scene, method, run_version, "splatam", "exploration_stage_1", "params.npz")
			for root in dict.fromkeys(root_candidates)
		]
		raise FileNotFoundError(f"NPZ not found. Searched: {searched}")

	semantic_ply = _pick_latest_ply(splatam_dir, "semantic_GS")

	if out_override is None:
		out_path = os.path.join(run_root, "visualization", "traj_bev.png")
	else:
		out_path = out_override
	out_dir = os.path.dirname(out_path)
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)

	traj_dir = os.path.join(run_root, "visualization", "pose")

	return npz_path, semantic_ply, out_path, traj_dir, run_root


def _sorted_pose_files(traj_dir: str) -> typing.List[str]:
	if not traj_dir or not os.path.isdir(traj_dir):
		return []
	files = glob.glob(os.path.join(traj_dir, "*.npy"))
	if not files:
		return []

	def _nat_key(p: str):
		name = os.path.basename(p)
		m = re.search(r"(\d+)", name)
		if m:
			return int(m.group(1))
		return name

	files.sort(key=_nat_key)
	return files


def _bbox_xy(points: np.ndarray):
	xmin = float(np.min(points[:, 0]))
	xmax = float(np.max(points[:, 0]))
	ymin = float(np.min(points[:, 1]))
	ymax = float(np.max(points[:, 1]))
	return xmin, xmax, ymin, ymax


def _outside_ratio_xy(traj_pts: np.ndarray, bbox_xy) -> float:
	xmin, xmax, ymin, ymax = bbox_xy
	outside = (
		(traj_pts[:, 0] < xmin)
		| (traj_pts[:, 0] > xmax)
		| (traj_pts[:, 1] < ymin)
		| (traj_pts[:, 1] > ymax)
	)
	return float(np.mean(outside))


def load_trajectory_points(
	traj_dir: str,
	ref_points: typing.Optional[np.ndarray] = None,
	pose_format: str = "auto",
) -> typing.Optional[np.ndarray]:
	"""Load trajectory positions (N,3) from pose .npy files.

	pose_format:
	- auto: choose Twc/Tcw interpretation using scene XY bbox consistency
	- twc: use pose[:3, 3]
	- tcw: use -R^T t
	"""
	pose_files = _sorted_pose_files(traj_dir)
	if not pose_files:
		return None

	traj_twc = []
	traj_tcw = []
	traj_fallback = []
	for fp in pose_files:
		try:
			pose = np.load(fp)
		except Exception:
			continue

		pose = np.array(pose)
		if pose.shape == (4, 4):
			R = pose[:3, :3].astype(np.float32)
			t = pose[:3, 3].astype(np.float32)
			traj_twc.append(t)
			traj_tcw.append((-R.T @ t).astype(np.float32))
		elif pose.ndim == 2 and pose.shape[1] >= 3:
			traj_fallback.append(pose[0, :3].astype(np.float32))

	if pose_format not in ("auto", "twc", "tcw"):
		pose_format = "auto"

	if traj_twc:
		twc_pts = np.vstack(traj_twc)
		tcw_pts = np.vstack(traj_tcw)
		if pose_format == "twc":
			traj_pts = twc_pts
		elif pose_format == "tcw":
			traj_pts = tcw_pts
		else:
			if ref_points is None or ref_points.shape[0] == 0:
				traj_pts = twc_pts
			else:
				bbox = _bbox_xy(ref_points)
				score_twc = _outside_ratio_xy(twc_pts, bbox)
				score_tcw = _outside_ratio_xy(tcw_pts, bbox)
				traj_pts = twc_pts if score_twc <= score_tcw else tcw_pts
	elif traj_fallback:
		traj_pts = np.vstack(traj_fallback)
	else:
		traj_pts = None

	if traj_pts is None or traj_pts.shape[0] == 0:
		return None
	return np.array(traj_pts, dtype=np.float32)


def load_trajectory_points_from_npz(npz_path: str) -> typing.Optional[np.ndarray]:
	"""Load camera centers from params.npz w2c trajectory when pose .npy files are absent."""
	with np.load(npz_path, allow_pickle=True) as d:
		if "gt_w2c_all_frames" in d:
			w2cs = np.array(d["gt_w2c_all_frames"], dtype=np.float32)
		elif "w2c" in d:
			w2cs = np.array(d["w2c"], dtype=np.float32)
		else:
			return None

	if w2cs.shape == (4, 4):
		w2cs = w2cs[None, ...]
	if w2cs.ndim != 3 or w2cs.shape[1:] != (4, 4):
		return None

	c2ws = np.linalg.inv(w2cs)
	return c2ws[:, :3, 3].astype(np.float32)


def load_first_pose_transform(traj_path: str) -> np.ndarray:
	"""Load the first 4x4 pose from a Habitat trajectory text file."""
	with open(traj_path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			values = np.fromstring(line, sep=" ", dtype=np.float32)
			if values.size != 16:
				raise ValueError(f"Expected 16 values in first trajectory pose, got {values.size}: {traj_path}")
			return values.reshape(4, 4)
	raise ValueError(f"Empty trajectory file: {traj_path}")


def resolve_mp3d_anchor_traj(scene: str, override: typing.Optional[str]) -> typing.Optional[str]:
	if override:
		return override
	candidate = os.path.join("data", "mp3d_sim_nvs_v2", scene, "traj.txt")
	return candidate if os.path.exists(candidate) else None


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
	if points.shape[0] == 0:
		return points.astype(np.float32)
	ones = np.ones((points.shape[0], 1), dtype=np.float32)
	points_h = np.concatenate([points.astype(np.float32), ones], axis=1)
	return (transform @ points_h.T).T[:, :3].astype(np.float32)


def apply_rigid_transform_to_data(data: dict, transform: np.ndarray) -> dict:
	"""Apply a rigid 4x4 transform to point fields that share the point-cloud length."""
	pts = np.array(data["points"], dtype=np.float32)
	data["points"] = transform_points(pts, transform)

	for nk, nv in list(data.get("numeric", {}).items()):
		try:
			arr = np.array(nv, dtype=np.float32)
			if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] == pts.shape[0]:
				data["numeric"][nk] = transform_points(arr, transform)
		except Exception:
			pass
	return data


def trajectory_outside_ratio_xy(traj_pts: typing.Optional[np.ndarray], points: np.ndarray) -> typing.Tuple[int, int, float]:
	if traj_pts is None or traj_pts.shape[0] == 0 or points.shape[0] == 0:
		return 0, 0, 0.0
	xmin, xmax, ymin, ymax = _bbox_xy(points)
	outside = (
		(traj_pts[:, 0] < xmin)
		| (traj_pts[:, 0] > xmax)
		| (traj_pts[:, 1] < ymin)
		| (traj_pts[:, 1] > ymax)
	)
	return int(np.count_nonzero(outside)), int(traj_pts.shape[0]), float(np.mean(outside))


def apply_align_transform_to_traj(traj_pts: np.ndarray, transform: dict) -> np.ndarray:
	"""Apply align_floor transform to trajectory points."""
	centroid = np.array(transform["centroid"], dtype=np.float32)
	R = np.array(transform["R"], dtype=np.float32)
	floor_median_z = float(transform["floor_median_z"])
	return (R.dot((traj_pts - centroid).T)).T - np.array([0.0, 0.0, floor_median_z], dtype=np.float32)


def figure_canvas_to_rgb(fig) -> np.ndarray:
	fig.canvas.draw()
	if hasattr(fig.canvas, "tostring_rgb"):
		w, h = fig.canvas.get_width_height()
		return np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
	return np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3].copy()


def draw_trajectory_on_bev(
	sem_img: np.ndarray,
	traj_pts: typing.Optional[np.ndarray],
	bounds,
	res: float,
	line_color: str,
	line_width: float,
	start_color: str,
	end_color: str,
	marker_size: float,
):
	"""Overlay trajectory polyline and start/end markers on a semantic BEV image."""
	if sem_img is None or traj_pts is None or traj_pts.shape[0] < 2 or bounds is None:
		return sem_img

	xmin, xmax, ymin, ymax, W, H = bounds
	xs = traj_pts[:, 0]
	ys = traj_pts[:, 1]

	outside = (
		(xs < xmin)
		| (xs > xmax)
		| (ys < ymin)
		| (ys > ymax)
	)
	if np.any(outside):
		print(
			f"[WARN] Trajectory outside BEV bounds: "
			f"{int(np.count_nonzero(outside))}/{int(traj_pts.shape[0])} "
			f"({float(np.mean(outside)) * 100.0:.2f}%). "
			"Projected points are clipped to the image border."
		)

	ix = ((xs - xmin) / res).astype(np.float32)
	iy = ((ys - ymin) / res).astype(np.float32)
	ix = np.clip(ix, 0, W - 1)
	iy = np.clip(iy, 0, H - 1)

	# Match np.flipud in BEV image generation.
	iy_img = (H - 1) - iy

	fig, ax = plt.subplots(1, 1, figsize=(6, 6))
	ax.imshow(sem_img, interpolation="nearest")
	ax.plot(ix, iy_img, color=line_color, linewidth=line_width, alpha=0.95)
	ax.scatter(ix[0], iy_img[0], c=start_color, s=marker_size, edgecolors="black", linewidths=0.8, zorder=3)
	ax.scatter(ix[-1], iy_img[-1], c=end_color, s=marker_size, edgecolors="black", linewidths=0.8, zorder=3)
	ax.axis("off")
	fig.tight_layout(pad=0)

	rgb = figure_canvas_to_rgb(fig)
	plt.close(fig)
	return rgb


def load_ply_fields(ply_path):
	"""Load PLY and return dict with points, labels (optional), entropy (optional)."""
	ply = PlyData.read(ply_path)
	v = ply["vertex"].data
	names = v.dtype.names

	pts = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)

	rgb = None
	if all(k in names for k in ("red", "green", "blue")):
		rgb = np.vstack([v["red"], v["green"], v["blue"]]).T.astype(np.uint8)

	label = None
	for k in ("semantic", "label", "class", "object_id"):
		if k in names:
			label = np.array(v[k]).astype(np.int32)
			break

	f_dc_keys = [k for k in names if k.startswith("f_dc_")]
	f_dc = None
	if len(f_dc_keys) > 0:
		f_dc = np.vstack([np.array(v[k]).astype(np.float32) for k in f_dc_keys]).T

	numeric = {}
	for k in names:
		try:
			arr = np.array(v[k])
		except Exception:
			continue
		if np.issubdtype(arr.dtype, np.number) and k not in ("x", "y", "z"):
			numeric[k] = arr.astype(np.float32)

	entropy = None
	for k in ("entropy", "uncertainty", "score"):
		if k in names:
			entropy = np.array(v[k]).astype(np.float32)
			break

	return dict(points=pts, rgb=rgb, label=label, entropy=entropy, f_dc=f_dc, f_dc_keys=f_dc_keys, numeric=numeric)


def _dedupe_existing_paths(paths: typing.Iterable[str]) -> typing.List[str]:
	seen = set()
	out = []
	for path in paths:
		if not path:
			continue
		key = os.path.abspath(path)
		if key in seen or not os.path.exists(path):
			continue
		seen.add(key)
		out.append(path)
	return out


def mp3d_scene_rgb_candidates(scene: str, override: typing.Optional[str]) -> typing.List[str]:
	"""Return MP3D assets that can provide real scene RGB for BEV rendering."""
	candidates = []
	if override:
		candidates.append(override)
	if scene:
		scan_root = os.path.join("data", "MP3D", "v1", "scans", scene)
		candidates.extend(glob.glob(os.path.join(scan_root, scene, "matterport_mesh", "*", "*.obj")))
		candidates.append(os.path.join(scan_root, "mesh.obj"))
		task_root = os.path.join("data", "MP3D", "v1", "tasks", "mp3d", scene)
		candidates.append(os.path.join(task_root, "semantic_clean.ply"))
		candidates.append(os.path.join(task_root, f"{scene}_semantic.ply"))
		candidates.append(os.path.join(task_root, f"{scene}.glb"))
	return _dedupe_existing_paths(candidates)


def _texture_image_to_rgb_array(texture) -> typing.Optional[np.ndarray]:
	arr = np.asarray(texture)
	if arr.dtype == object or arr.ndim < 2 or arr.size == 0:
		return None
	if arr.ndim == 2:
		arr = arr[:, :, None]
	if arr.shape[2] == 1:
		arr = np.repeat(arr, 3, axis=2)
	elif arr.shape[2] > 3:
		arr = arr[:, :, :3]
	return decode_rgb_values(arr.reshape(-1, 3)).reshape(arr.shape[0], arr.shape[1], 3)


def _sample_texture_colors(
	texture_arrays: typing.List[typing.Optional[np.ndarray]],
	uvs: np.ndarray,
	material_ids: np.ndarray,
) -> np.ndarray:
	valid_indices = [idx for idx, tex in enumerate(texture_arrays) if tex is not None]
	if not valid_indices:
		raise ValueError("mesh has UVs but no readable texture images")
	fallback_idx = valid_indices[0]
	colors = np.zeros((uvs.shape[0], 3), dtype=np.uint8)
	for mat_id in np.unique(material_ids):
		tex_idx = int(mat_id)
		if tex_idx < 0 or tex_idx >= len(texture_arrays) or texture_arrays[tex_idx] is None:
			tex_idx = fallback_idx
		tex = texture_arrays[tex_idx]
		mask = material_ids == mat_id
		uv = np.clip(uvs[mask], 0.0, 1.0)
		h, w = tex.shape[:2]
		x = np.rint(uv[:, 0] * (w - 1)).astype(np.int64)
		y = np.rint((1.0 - uv[:, 1]) * (h - 1)).astype(np.int64)
		colors[mask] = tex[y, x, :3]
	return colors


def _textured_mesh_to_rgb_data(mesh) -> typing.Optional[dict]:
	vertices = np.asarray(mesh.vertices, dtype=np.float32)
	triangles = np.asarray(mesh.triangles, dtype=np.int64)
	if vertices.size == 0 or triangles.size == 0:
		return None

	n_tri = triangles.shape[0]
	tri_pts = vertices[triangles]
	centers = tri_pts.mean(axis=1)

	if mesh.has_vertex_colors():
		vertex_colors = decode_rgb_values(np.asarray(mesh.vertex_colors, dtype=np.float32))
		tri_cols = vertex_colors[triangles]
		points = np.concatenate([centers, tri_pts.reshape(-1, 3)], axis=0).astype(np.float32)
		rgb = np.concatenate([tri_cols.mean(axis=1), tri_cols.reshape(-1, 3)], axis=0)
		return dict(points=points, rgb=decode_rgb_values(rgb), label=None, entropy=None, f_dc=None, f_dc_keys=[], numeric={})

	texture_arrays = [_texture_image_to_rgb_array(tex) for tex in getattr(mesh, "textures", [])]
	if mesh.has_triangle_uvs() and any(tex is not None for tex in texture_arrays):
		tri_uvs = np.asarray(mesh.triangle_uvs, dtype=np.float32).reshape(-1, 3, 2)
		n = min(n_tri, tri_uvs.shape[0])
		tri_pts = tri_pts[:n]
		centers = centers[:n]
		tri_uvs = tri_uvs[:n]
		center_uvs = tri_uvs.mean(axis=1)
		points = np.concatenate([centers, tri_pts.reshape(-1, 3)], axis=0).astype(np.float32)
		uvs = np.concatenate([center_uvs, tri_uvs.reshape(-1, 2)], axis=0).astype(np.float32)

		mat_ids = np.zeros(n, dtype=np.int64)
		if len(mesh.triangle_material_ids) >= n:
			mat_ids = np.asarray(mesh.triangle_material_ids, dtype=np.int64)[:n]
		material_ids = np.concatenate([mat_ids, np.repeat(mat_ids, 3)], axis=0)
		rgb = _sample_texture_colors(texture_arrays, uvs, material_ids)
		return dict(points=points, rgb=rgb, label=None, entropy=None, f_dc=None, f_dc_keys=[], numeric={})

	return None


def load_scene_rgb_fields(scene: str, mesh_override: typing.Optional[str]) -> typing.Tuple[dict, str]:
	"""Load true MP3D scene RGB samples from textured mesh or RGB PLY assets."""
	candidates = mp3d_scene_rgb_candidates(scene, mesh_override)
	if not candidates:
		raise FileNotFoundError(f"No MP3D scene RGB asset candidates found for scene {scene!r}")

	last_error = None
	for candidate in candidates:
		ext = os.path.splitext(candidate)[1].lower()
		try:
			if ext == ".ply":
				data = load_ply_fields(candidate)
				if data.get("rgb") is not None and data["points"].shape[0] > 0:
					return data, candidate
				last_error = ValueError("PLY has no RGB vertex colors")
				continue

			import open3d as o3d
			mesh = o3d.io.read_triangle_mesh(candidate, enable_post_processing=True)
			data = _textured_mesh_to_rgb_data(mesh)
			if data is not None and data["points"].shape[0] > 0:
				return data, candidate
			last_error = ValueError("mesh has no readable vertex colors or texture RGB")
		except Exception as exc:
			last_error = exc
			continue

	raise ValueError(f"Failed to load true RGB scene asset for {scene!r}; last error: {last_error}")


def detect_floor_and_ceiling_by_semantic(points: np.ndarray, labels: np.ndarray):
	"""Return (floor_id, ceiling_id) by selecting labels with lowest/highest mean z."""
	if labels is None:
		return None, None
	unique = np.unique(labels)
	means = {}
	for u in unique:
		mask = labels == u
		if np.count_nonzero(mask) == 0:
			continue
		means[u] = float(np.mean(points[mask, 2]))
	if not means:
		return None, None
	floor_id = int(min(means, key=means.get))
	ceiling_id = int(max(means, key=means.get))
	return floor_id, ceiling_id


def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
	"""Compute rotation matrix that rotates vector a to vector b."""
	a = a / np.linalg.norm(a)
	b = b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	if np.allclose(v, 0) and c > 0.999999:
		return np.eye(3)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	R = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-12))
	return R


def align_floor(points: np.ndarray, labels: np.ndarray, floor_id: int):
	"""Rotate points so floor plane aligns with Z axis and translate floor to z=0."""
	if floor_id is None:
		raise ValueError("floor_id is required")
	mask = labels == floor_id
	if np.count_nonzero(mask) < 3:
		raise ValueError("Not enough floor points to estimate plane")
	floor_pts = points[mask]
	centroid = np.mean(floor_pts, axis=0)
	centered = floor_pts - centroid
	_, _, vh = np.linalg.svd(centered, full_matrices=False)
	normal = vh[-1, :]
	if normal[2] < 0:
		normal = -normal
	R = rotation_matrix_from_vectors(normal, np.array([0.0, 0.0, 1.0]))
	pts_centered = points - centroid
	pts_rot = (R.dot(pts_centered.T)).T
	floor_rot_z = pts_rot[mask, 2]
	med_z = float(np.median(floor_rot_z))
	pts_rot[:, 2] -= med_z
	transform = dict(R=R, centroid=centroid.tolist(), floor_median_z=med_z)
	return pts_rot, transform


def apply_mask_to_data(data: dict, mask: np.ndarray) -> dict:
	"""Return a new data dict with entries filtered by mask (1D boolean)."""
	new = dict(data)
	pts = data["points"]
	new["points"] = pts[mask]
	for k in ("rgb", "label", "entropy", "f_dc"):
		if k in data and data[k] is not None:
			arr = data[k]
			if hasattr(arr, "shape") and arr.shape[0] == pts.shape[0]:
				new[k] = arr[mask]
	if "numeric" in data:
		new_numeric = {}
		for nk, nv in data["numeric"].items():
			try:
				if nv.shape[0] == pts.shape[0]:
					new_numeric[nk] = nv[mask]
				else:
					new_numeric[nk] = nv
			except Exception:
				new_numeric[nk] = nv
		new["numeric"] = new_numeric
	return new


def apply_align_transform_to_data(data: dict, transform: dict) -> dict:
	"""Apply align_floor transform to a data dict in-place and return it."""
	pts = np.array(data["points"]).astype(np.float32)
	centroid = np.array(transform["centroid"], dtype=np.float32)
	R = np.array(transform["R"], dtype=np.float32)
	floor_median_z = float(transform["floor_median_z"])

	data["points"] = (R.dot((pts - centroid).T)).T - np.array([0.0, 0.0, floor_median_z], dtype=np.float32)

	for nk, nv in list(data.get("numeric", {}).items()):
		try:
			arr = np.array(nv)
			if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] == pts.shape[0]:
				data["numeric"][nk] = (R.dot((arr - centroid).T)).T - np.array([0.0, 0.0, floor_median_z], dtype=np.float32)
		except Exception:
			pass
	return data


def normalize_label_map(raw_map: typing.Optional[dict]) -> typing.Optional[dict]:
	"""Normalize label-map JSON shapes to dict(str(label) -> [r,g,b])."""
	if raw_map is None:
		return None
	out = {}
	if isinstance(raw_map, dict) and "mappings" in raw_map:
		for k, v in raw_map["mappings"].items():
			if isinstance(v, dict):
				if "rgb" in v:
					rgb = v["rgb"]
				elif "color" in v:
					rgb = v["color"]
				else:
					continue
			else:
				rgb = v
			try:
				arr = np.array(rgb)
				if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.01:
					arr = (arr * 255.0).astype(np.uint8)
				else:
					arr = arr.astype(np.uint8)
				if arr.size == 3:
					out[str(k)] = [int(arr[0]), int(arr[1]), int(arr[2])]
			except Exception:
				continue
		return out

	if isinstance(raw_map, dict):
		for k, v in raw_map.items():
			try:
				arr = np.array(v)
				if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.01:
					arr = (arr * 255.0).astype(np.uint8)
				else:
					arr = arr.astype(np.uint8)
				if arr.size == 3:
					out[str(k)] = [int(arr[0]), int(arr[1]), int(arr[2])]
			except Exception:
				continue
		return out

	return None


def load_npz_fields(npz_path):
	"""Load .npz fields needed for RGB/semantic ops and viewpoint bounds."""
	d = np.load(npz_path, allow_pickle=True)

	if "means3D" in d:
		pts = np.array(d["means3D"]).astype(np.float32)
	elif "points" in d:
		pts = np.array(d["points"]).astype(np.float32)
	else:
		raise ValueError("No `means3D` or `points` found in npz")

	rgb = None
	if all(k in d for k in ("f_dc_0", "f_dc_1", "f_dc_2")):
		f_dc_rgb = np.stack([
			np.array(d["f_dc_0"], dtype=np.float32).reshape(-1),
			np.array(d["f_dc_1"], dtype=np.float32).reshape(-1),
			np.array(d["f_dc_2"], dtype=np.float32).reshape(-1),
		], axis=1)
		rgb = decode_rgb_from_fdc(f_dc_rgb)
	elif "features_dc" in d:
		f_dc_rgb = np.array(d["features_dc"], dtype=np.float32).reshape(pts.shape[0], -1)
		if f_dc_rgb.shape[1] >= 3:
			rgb = decode_rgb_from_fdc(f_dc_rgb)
	elif "rgb_colors" in d:
		rgb = decode_rgb_values(np.array(d["rgb_colors"], dtype=np.float32))

	label = None
	if "seman_cls_ids" in d:
		lab = np.array(d["seman_cls_ids"])
		if lab.ndim == 2 and lab.shape[1] > 1:
			label = np.argmax(lab, axis=1).astype(np.int32)
		else:
			label = lab.ravel().astype(np.int32)
	elif "semantic_logits" in d:
		logits = np.array(d["semantic_logits"]).astype(np.float32)
		label = np.argmax(logits, axis=1).astype(np.int32)

	entropy = None
	if "entropy" in d:
		entropy = np.array(d["entropy"]).astype(np.float32)
	elif "semantic_logits" in d:
		logits = np.array(d["semantic_logits"]).astype(np.float32)
		with np.errstate(over="ignore"):
			ex = np.exp(logits - np.max(logits, axis=1, keepdims=True))
			p = ex / (np.sum(ex, axis=1, keepdims=True) + 1e-12)
		ent = -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1)
		entropy = ent.astype(np.float32)

	f_dc = None
	f_dc_keys = []
	if "semantic_logits" in d:
		f_dc = np.array(d["semantic_logits"]).astype(np.float32)
		f_dc_keys = ["semantic_logits"]

	numeric = {}
	for k in d.files:
		if k in ("means3D", "seman_cls_ids", "semantic_logits", "entropy", "points"):
			continue
		try:
			arr = np.array(d[k])
		except Exception:
			continue
		if np.issubdtype(arr.dtype, np.number):
			numeric[k] = arr.astype(np.float32)

	return dict(points=pts, rgb=rgb, label=label, entropy=entropy, f_dc=f_dc, f_dc_keys=f_dc_keys, numeric=numeric)


def bev_from_ply(
	data,
	res=0.05,
	pad=0.1,
	label_map=None,
	remove_above: float = None,
	remove_below: float = None,
	remove_top_percent: float = None,
	bounds_override=None,
	return_bounds: bool = False,
	render_semantic: bool = True,
	render_entropy: bool = True,
):
	"""Create semantic image choosing top-most point (largest z) per grid cell."""
	pts = data["points"]
	label = data.get("label")
	entropy = data.get("entropy")
	f_dc = data.get("f_dc")
	numeric = data.get("numeric", {})

	orig_n = pts.shape[0]
	if remove_top_percent is not None:
		if not (0.0 <= remove_top_percent <= 100.0):
			raise ValueError("remove_top_percent must be in [0,100]")
		z_thresh = np.percentile(pts[:, 2], remove_top_percent)
	elif remove_above is not None:
		z_thresh = float(remove_above)
	else:
		z_thresh = None

	mask = np.ones(orig_n, dtype=bool)
	if remove_below is not None:
		mask &= pts[:, 2] >= float(remove_below)
	if z_thresh is not None:
		mask &= pts[:, 2] <= z_thresh
	if not np.all(mask):
		pts = pts[mask]
		if label is not None and len(label) == orig_n:
			label = label[mask]
		if entropy is not None and len(entropy) == orig_n:
			entropy = entropy[mask]
		if f_dc is not None and len(f_dc) == orig_n:
			f_dc = f_dc[mask]
		for k, v in list(numeric.items()):
			try:
				if hasattr(v, "shape") and len(v) > 0 and v.shape[0] == orig_n:
					numeric[k] = v[mask]
			except Exception:
				pass

	if pts.shape[0] == 0:
		if return_bounds:
			return None, None, None
		return None, None

	xs = pts[:, 0]
	ys = pts[:, 1]
	zs = pts[:, 2]

	if bounds_override is None:
		xmin, xmax = xs.min(), xs.max()
		ymin, ymax = ys.min(), ys.max()
		dx = xmax - xmin
		dy = ymax - ymin
		xmin -= dx * pad
		xmax += dx * pad
		ymin -= dy * pad
		ymax += dy * pad
		W = int(np.ceil((xmax - xmin) / res))
		H = int(np.ceil((ymax - ymin) / res))
	else:
		xmin, xmax, ymin, ymax, W, H = bounds_override

	ix = ((xs - xmin) / res).astype(np.int64)
	iy = ((ys - ymin) / res).astype(np.int64)
	ix = np.clip(ix, 0, W - 1)
	iy = np.clip(iy, 0, H - 1)

	idx_1d = iy * W + ix
	order = np.argsort(-zs)
	idx_1d_sorted = idx_1d[order]
	_, first_indices = np.unique(idx_1d_sorted, return_index=True)
	chosen_idx = order[first_indices]

	semantic_img = None
	entropy_img = None

	if render_semantic and label is not None:
		unique_labels, _ = np.unique(label, return_counts=True)
		label_to_color = {}
		if label_map is not None:
			for lab in unique_labels:
				key = str(lab)
				if key not in label_map and str(int(lab) + 1) in label_map:
					key = str(int(lab) + 1)
				if key in label_map:
					col = np.array(label_map[key])
					col = np.asarray(col)
					if np.issubdtype(col.dtype, np.floating) and col.max() <= 1.01:
						col = (col * 255.0).astype(np.uint8)
					else:
						col = col.astype(np.uint8)
					label_to_color[lab] = col
		base_cmap = cm.get_cmap("tab20")
		for lab in unique_labels:
			if lab in label_to_color:
				continue
			ci = int(lab) % base_cmap.N
			colf = np.array(base_cmap(ci)[:3])
			label_to_color[lab] = (colf * 255.0).astype(np.uint8)

		semantic_img = np.zeros((H, W, 3), dtype=np.uint8)
		for i_pt in chosen_idx:
			semantic_img[iy[i_pt], ix[i_pt]] = label_to_color[int(label[i_pt])]

	elif render_semantic and f_dc is not None and f_dc.ndim == 2 and f_dc.shape[1] >= 3:
		fdc_rgb = np.clip((f_dc[:, :3] * C0 + 0.5) * 255.0, 0, 255).astype(np.uint8)
		semantic_img = np.zeros((H, W, 3), dtype=np.uint8)
		semantic_img.fill(255)
		semantic_img[iy[chosen_idx], ix[chosen_idx]] = fdc_rgb[chosen_idx]

	if render_entropy and entropy is not None:
		entropy_img = np.zeros((H, W), dtype=np.float32)
		e_vals = entropy[chosen_idx]
		if e_vals.size:
			emin, emax = e_vals.min(), e_vals.max()
			if emax - emin > 1e-6:
				e_norm = (e_vals - emin) / (emax - emin)
			else:
				e_norm = np.zeros_like(e_vals)
			entropy_img[iy[chosen_idx], ix[chosen_idx]] = e_norm

	if semantic_img is not None:
		semantic_img = np.flipud(semantic_img)
	if entropy_img is not None:
		entropy_img = np.flipud(entropy_img)

	bounds = (float(xmin), float(xmax), float(ymin), float(ymax), int(W), int(H))
	if return_bounds:
		return semantic_img, entropy_img, bounds
	return semantic_img, entropy_img


def bev_rgb_from_data(
	data,
	res=0.05,
	pad=0.1,
	remove_above: float = None,
	remove_below: float = None,
	remove_top_percent: float = None,
	bounds_override=None,
	return_bounds: bool = False,
):
	"""Create an RGB BEV image using the same top-surface projection as semantic BEV."""
	pts = data["points"]
	rgb = data.get("rgb")
	if rgb is None:
		raise ValueError("RGB colors are missing; expected `rgb_colors`, `features_dc`, or red/green/blue fields")

	rgb = decode_rgb_values(rgb)
	orig_n = pts.shape[0]
	if rgb.shape[0] != orig_n:
		raise ValueError(f"RGB length {rgb.shape[0]} does not match points length {orig_n}")

	if remove_top_percent is not None:
		if not (0.0 <= remove_top_percent <= 100.0):
			raise ValueError("remove_top_percent must be in [0,100]")
		z_thresh = np.percentile(pts[:, 2], remove_top_percent)
	elif remove_above is not None:
		z_thresh = float(remove_above)
	else:
		z_thresh = None

	mask = np.ones(orig_n, dtype=bool)
	if remove_below is not None:
		mask &= pts[:, 2] >= float(remove_below)
	if z_thresh is not None:
		mask &= pts[:, 2] <= z_thresh
	if not np.all(mask):
		pts = pts[mask]
		rgb = rgb[mask]

	if pts.shape[0] == 0:
		if return_bounds:
			return None, None
		return None

	xs = pts[:, 0]
	ys = pts[:, 1]
	zs = pts[:, 2]

	if bounds_override is None:
		xmin, xmax = xs.min(), xs.max()
		ymin, ymax = ys.min(), ys.max()
		dx = xmax - xmin
		dy = ymax - ymin
		xmin -= dx * pad
		xmax += dx * pad
		ymin -= dy * pad
		ymax += dy * pad
		W = int(np.ceil((xmax - xmin) / res))
		H = int(np.ceil((ymax - ymin) / res))
	else:
		xmin, xmax, ymin, ymax, W, H = bounds_override

	ix = ((xs - xmin) / res).astype(np.int64)
	iy = ((ys - ymin) / res).astype(np.int64)
	ix = np.clip(ix, 0, W - 1)
	iy = np.clip(iy, 0, H - 1)

	idx_1d = iy * W + ix
	order = np.argsort(-zs)
	idx_1d_sorted = idx_1d[order]
	_, first_indices = np.unique(idx_1d_sorted, return_index=True)
	chosen_idx = order[first_indices]

	rgb_img = np.full((H, W, 3), 255, dtype=np.uint8)
	rgb_img[iy[chosen_idx], ix[chosen_idx]] = rgb[chosen_idx]
	rgb_img = np.flipud(rgb_img)

	bounds = (float(xmin), float(xmax), float(ymin), float(ymax), int(W), int(H))
	if return_bounds:
		return rgb_img, bounds
	return rgb_img


def save_bev_figure(img, out_path, dpi=300, interpolation="bilinear", title: typing.Optional[str] = None):
	fig, ax = plt.subplots(1, 1, figsize=(6, 6))
	if img is not None:
		ax.imshow(img, interpolation=interpolation)
		if title:
			ax.set_title(title)
	else:
		ax.text(0.5, 0.5, "No BEV image", ha="center")
	ax.axis("off")
	plt.tight_layout()
	plt.savefig(out_path, dpi=dpi)
	plt.close(fig)


def add_rb_legend(
	img: np.ndarray,
	r_text: str,
	b_text: str,
	r_color: str = "red",
	b_color: str = "blue",
) -> np.ndarray:
	"""Add top-left R/B textual legend on final image."""
	if img is None:
		return img
	fig, ax = plt.subplots(1, 1, figsize=(6, 6))
	ax.imshow(img, interpolation="nearest")
	ax.axis("off")

	text_box = dict(boxstyle="round,pad=0.25", facecolor=(0, 0, 0, 0.35), edgecolor="white", linewidth=0.8)
	ax.text(0.02, 0.98, f"R: {r_text}", transform=ax.transAxes, ha="left", va="top", color=r_color, fontsize=11, bbox=text_box)
	ax.text(0.02, 0.91, f"B: {b_text}", transform=ax.transAxes, ha="left", va="top", color=b_color, fontsize=11, bbox=text_box)

	rgb = figure_canvas_to_rgb(fig)
	plt.close(fig)
	return rgb

"""
python src/visualization/render_semantic_bev.py --dataset Replica --scene office2 --method SemanticHeat --run-version run_0_v1 --ceiling-id 3318 --align-floor --remove-semantic-ceiling --remove-above 1.5 --flip-z
"""


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", default="Replica", help="dataset name, e.g. Replica")
	parser.add_argument("--scene", required=True, help="scene name, e.g. office2")
	parser.add_argument("--method", default="SemanticHeat", help="method name, e.g. SemanticHeat")
	parser.add_argument("--run-version", required=True, help="run version, e.g. run_0_v1")
	parser.add_argument("--results-root", default="auto", help="results root; auto tries results, results_<dataset>, and results_<DATASET>")
	parser.add_argument("--traj-file", default=None, help="optional params.npz used only for the overlaid trajectory; useful for splatam/final/params.npz")
	parser.add_argument("--out", default=None, help="optional output image path; default is auto-generated in run visualization dir")
	parser.add_argument("--res", type=float, default=0.015, help="meters per pixel (smaller = finer, less blocky)")
	parser.add_argument("--pad", type=float, default=0.05, help="bbox pad fraction")
	parser.add_argument("--dpi", type=int, default=400, help="output figure DPI")
	parser.add_argument("--interpolation", choices=["nearest", "bilinear", "bicubic", "lanczos"], default="lanczos", help="imshow interpolation for smoother rendering")
	parser.add_argument("--background", choices=["semantic", "rgb", "scene-rgb"], default="semantic", help="BEV background to overlay trajectory on")
	parser.add_argument("--scene-rgb-mesh", default=None, help="true RGB MP3D scene asset for --background scene-rgb; top-level mesh.obj is OK, nested textured OBJ is auto-detected")
	parser.add_argument("--min-height", type=float, default=None, help="optional lower z bound for BEV height slicing")
	parser.add_argument("--max-height", type=float, default=None, help="optional upper z bound for BEV height slicing; overrides --remove-above")
	parser.add_argument("--label-map", help="optional JSON file mapping label->[r,g,b] or the class_colormap style")
	parser.add_argument("--floor-id", type=int, default=None, help="semantic id for floor (optional)")
	parser.add_argument("--ceiling-id", type=int, default=None, help="semantic id for ceiling (optional)")
	parser.add_argument("--align-floor", action="store_true", help="estimate floor plane and align it to z=0")
	parser.add_argument("--remove-semantic-ceiling", action="store_true", help="remove points whose semantic label is ceiling id")
	parser.add_argument("--auto-semantic-detect", action="store_true", help="auto-detect floor/ceiling ids from semantic means")
	parser.add_argument("--remove-above", type=float, default=None, help="remove points with z > VALUE (meters)")
	parser.add_argument("--remove-top-percent", type=float, default=None, help="remove points with z above this percentile (0-100)")
	parser.add_argument("--flip-z", action="store_true", help="multiply z by -1 (flip vertical)")
	parser.add_argument("--traj-dir", default=None, help="trajectory pose dir containing .npy files")
	parser.add_argument("--traj-source", choices=["auto", "pose", "npz"], default="auto", help="trajectory source; auto uses pose .npy then falls back to params.npz")
	parser.add_argument("--coord-system", choices=["auto", "slam", "sim"], default="auto", help="BEV coordinate system; MP3D auto uses sim/world via traj.txt anchor")
	parser.add_argument("--mp3d-anchor-traj", default=None, help="MP3D trajectory txt whose first pose maps SLAM coordinates into sim/world coordinates")
	parser.add_argument("--traj-color", default="yellow", help="trajectory line color")
	parser.add_argument("--traj-linewidth", type=float, default=2.0, help="trajectory line width")
	parser.add_argument("--traj-start-color", default="blue", help="start marker color")
	parser.add_argument("--traj-end-color", default="red", help="end marker color")
	parser.add_argument("--traj-marker-size", type=float, default=36.0, help="start/end marker size")
	parser.add_argument("--traj-pose-format", choices=["auto", "twc", "tcw"], default="auto", help="pose matrix convention for trajectory points from pose .npy files")
	parser.add_argument("--legend-r-text", default="End", help="legend text for red marker")
	parser.add_argument("--legend-b-text", default="Start", help="legend text for blue marker")
	args = parser.parse_args()

	npz_path, sem_ply_path, out_path, default_traj_dir, run_root = resolve_io_paths(
		dataset=args.dataset,
		scene=args.scene,
		method=args.method,
		run_version=args.run_version,
		out_override=args.out,
		results_root=args.results_root,
	)
	traj_dir = args.traj_dir if args.traj_dir is not None else default_traj_dir
	print("Resolved run root  :", run_root)
	traj_npz_path = args.traj_file if args.traj_file is not None else npz_path
	if not os.path.isfile(traj_npz_path):
		raise FileNotFoundError(f"Trajectory params file not found: {traj_npz_path}")

	print("Resolved npz       :", npz_path)
	print("Resolved traj npz  :", traj_npz_path)
	print("Resolved semantic  :", sem_ply_path)
	print("Resolved output    :", out_path)
	print("Resolved traj dir  :", traj_dir)

	data = load_npz_fields(npz_path)
	sem_data = None if args.background == "scene-rgb" else load_ply_fields(sem_ply_path)

	traj_pts = None
	traj_source_used = "none"
	if args.traj_source in ("auto", "pose"):
		traj_pts = load_trajectory_points(
			traj_dir,
			ref_points=data.get("points"),
			pose_format=args.traj_pose_format,
		)
		if traj_pts is not None:
			traj_source_used = "pose"
	if traj_pts is None and args.traj_source in ("auto", "npz"):
		traj_pts = load_trajectory_points_from_npz(traj_npz_path)
		if traj_pts is not None:
			traj_source_used = "npz"

	coord_system = args.coord_system
	if coord_system == "auto":
		coord_system = "sim" if args.dataset.lower() == "mp3d" else "slam"

	anchor_path = None
	if coord_system == "sim":
		if args.dataset.lower() != "mp3d":
			raise ValueError("--coord-system sim is currently implemented for MP3D only")
		anchor_path = resolve_mp3d_anchor_traj(args.scene, args.mp3d_anchor_traj)
		if anchor_path is None:
			raise FileNotFoundError(
				f"MP3D sim/world conversion requested, but no anchor traj was found for scene {args.scene}. "
				"Use --mp3d-anchor-traj to provide one."
			)
		anchor_transform = load_first_pose_transform(anchor_path)
		data = apply_rigid_transform_to_data(data, anchor_transform)
		if sem_data is not None:
			sem_data = apply_rigid_transform_to_data(sem_data, anchor_transform)
		if traj_pts is not None:
			traj_pts = transform_points(traj_pts, anchor_transform)

	print("BEV coord system  :", coord_system)
	if anchor_path is not None:
		print("MP3D anchor traj  :", anchor_path)
	scene_rgb_data = None
	scene_rgb_source = None
	if args.background == "scene-rgb":
		if coord_system != "sim" and args.dataset.lower() == "mp3d":
			raise ValueError("--background scene-rgb for MP3D expects --coord-system sim/world coordinates")
		scene_rgb_data, scene_rgb_source = load_scene_rgb_fields(args.scene, args.scene_rgb_mesh)
		print("Scene RGB source  :", scene_rgb_source)

	print("Trajectory source :", traj_source_used)
	bbox_points = scene_rgb_data["points"] if scene_rgb_data is not None else data["points"]
	outside_n, total_n, outside_ratio = trajectory_outside_ratio_xy(traj_pts, bbox_points)
	if total_n > 0:
		print(f"Trajectory outside scene XY bbox: {outside_n}/{total_n} ({outside_ratio * 100.0:.2f}%)")
	else:
		print("Trajectory outside scene XY bbox: no trajectory points")

	if args.flip_z:
		for dsrc in [data, sem_data, scene_rgb_data]:
			if dsrc is None:
				continue
			pts = dsrc["points"]
			pts[:, 2] = -pts[:, 2]
			dsrc["points"] = pts
			for nk, nv in list(dsrc.get("numeric", {}).items()):
				try:
					arr = np.array(nv)
					if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] == pts.shape[0]:
						arr[:, 2] = -arr[:, 2]
						dsrc["numeric"][nk] = arr
				except Exception:
					pass
		if traj_pts is not None:
			traj_pts[:, 2] = -traj_pts[:, 2]

	labels = data.get("label")
	if args.auto_semantic_detect and labels is not None:
		auto_floor, auto_ceiling = detect_floor_and_ceiling_by_semantic(data["points"], labels)
		if args.floor_id is None:
			args.floor_id = auto_floor
		if args.ceiling_id is None:
			args.ceiling_id = auto_ceiling

	if args.remove_semantic_ceiling and data.get("label") is not None:
		if args.ceiling_id is None:
			_, auto_ceiling = detect_floor_and_ceiling_by_semantic(data["points"], data["label"])
			args.ceiling_id = auto_ceiling
		if args.ceiling_id is not None:
			mask_keep = data["label"] != args.ceiling_id
			data = apply_mask_to_data(data, mask_keep)

	if args.align_floor and data.get("label") is not None:
		if args.floor_id is None:
			auto_floor, _ = detect_floor_and_ceiling_by_semantic(data["points"], data["label"])
			args.floor_id = auto_floor
		if args.floor_id is not None:
			try:
				pts_rot, transform = align_floor(data["points"], data["label"], args.floor_id)
				data["points"] = pts_rot
				if sem_data is not None:
					sem_data = apply_align_transform_to_data(sem_data, transform)
				if traj_pts is not None:
					traj_pts = apply_align_transform_to_traj(traj_pts, transform)
			except Exception as e:
				print("align-floor failed:", e)

	label_map = None
	if args.label_map:
		with open(args.label_map, "r") as f:
			raw = json.load(f)
		label_map = normalize_label_map(raw)

	remove_above = args.max_height if args.max_height is not None else args.remove_above
	if args.background == "scene-rgb":
		bev_img, view_bounds = bev_rgb_from_data(
			scene_rgb_data,
			res=args.res,
			pad=args.pad,
			remove_above=remove_above,
			remove_below=args.min_height,
			remove_top_percent=args.remove_top_percent,
			return_bounds=True,
		)
	elif args.background == "rgb":
		bev_img, view_bounds = bev_rgb_from_data(
			data,
			res=args.res,
			pad=args.pad,
			remove_above=remove_above,
			remove_below=args.min_height,
			remove_top_percent=args.remove_top_percent,
			return_bounds=True,
		)
	else:
		# Keep the same viewpoint logic as render_rgb_sem_entro_bev.py:
		# compute bounds from primary data projection, then render semantic source with those bounds.
		_, _, view_bounds = bev_from_ply(
			data,
			res=args.res,
			pad=args.pad,
			label_map=label_map,
			remove_above=remove_above,
			remove_below=args.min_height,
			remove_top_percent=args.remove_top_percent,
			return_bounds=True,
			render_semantic=False,
			render_entropy=True,
		)

		sem_source = sem_data if sem_data is not None else data
		bev_img, _ = bev_from_ply(
			sem_source,
			res=args.res,
			pad=args.pad,
			label_map=label_map,
			remove_above=remove_above,
			remove_below=args.min_height,
			remove_top_percent=args.remove_top_percent,
			bounds_override=view_bounds,
			render_semantic=True,
			render_entropy=False,
		)

	bev_img = draw_trajectory_on_bev(
		sem_img=bev_img,
		traj_pts=traj_pts,
		bounds=view_bounds,
		res=args.res,
		line_color=args.traj_color,
		line_width=args.traj_linewidth,
		start_color=args.traj_start_color,
		end_color=args.traj_end_color,
		marker_size=args.traj_marker_size,
	)
	bev_img = add_rb_legend(
		img=bev_img,
		r_text=args.legend_r_text,
		b_text=args.legend_b_text,
		r_color=args.traj_end_color,
		b_color=args.traj_start_color,
	)

	save_bev_figure(
		bev_img,
		out_path,
		dpi=args.dpi,
		interpolation=args.interpolation,
		title=f"{args.background.upper()} + Trajectory",
	)
	print("Background       :", args.background)
	print("Saved", out_path)


if __name__ == "__main__":
	main()
