"""
Semantic-only BEV renderer.

Implementation is strictly based on semantic-related logic in
render_rgb_sem_entro_bev.py, while removing RGB/entropy panel output.
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
):
	"""Resolve npz/semantic/output paths from high-level run identifiers."""
	run_root = os.path.join("results", dataset, scene, method, run_version)
	splatam_dir = os.path.join(run_root, "splatam")
	npz_path = os.path.join(splatam_dir, "exploration_stage_1", "params.npz")
	if not os.path.exists(npz_path):
		raise FileNotFoundError(f"NPZ not found: {npz_path}")

	semantic_ply = _pick_latest_ply(splatam_dir, "semantic_GS")

	if out_override is None:
		out_path = os.path.join(run_root, "visualization", "semantic_bev.png")
	else:
		out_path = out_override
	out_dir = os.path.dirname(out_path)
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)

	return npz_path, semantic_ply, out_path


def load_ply_fields(ply_path):
	"""Load PLY and return dict with points, labels (optional), entropy (optional)."""
	ply = PlyData.read(ply_path)
	v = ply["vertex"].data
	names = v.dtype.names

	pts = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float32)

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

	return dict(points=pts, label=label, entropy=entropy, f_dc=f_dc, f_dc_keys=f_dc_keys, numeric=numeric)


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
	for k in ("label", "entropy", "f_dc"):
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
	"""Load .npz fields needed for semantic ops and viewpoint bounds."""
	d = np.load(npz_path, allow_pickle=True)

	if "means3D" in d:
		pts = np.array(d["means3D"]).astype(np.float32)
	elif "points" in d:
		pts = np.array(d["points"]).astype(np.float32)
	else:
		raise ValueError("No `means3D` or `points` found in npz")

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

	return dict(points=pts, label=label, entropy=entropy, f_dc=f_dc, f_dc_keys=f_dc_keys, numeric=numeric)


def bev_from_ply(
	data,
	res=0.05,
	pad=0.1,
	label_map=None,
	remove_above: float = None,
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

	if z_thresh is not None:
		mask = pts[:, 2] <= z_thresh
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


def save_semantic_figure(sem, out_path, dpi=300, interpolation="bilinear"):
	fig, ax = plt.subplots(1, 1, figsize=(6, 6))
	if sem is not None:
		ax.imshow(sem, interpolation=interpolation)
		ax.set_title("Semantic")
	else:
		ax.text(0.5, 0.5, "No Semantic", ha="center")
	ax.axis("off")
	plt.tight_layout()
	plt.savefig(out_path, dpi=dpi)
	plt.close(fig)

"""
python src/visualization/render_semantic_bev.py --dataset Replica --scene office2 --method SemanticHeat --run-version run_0_v1 --ceiling-id 3318 --align-floor --remove-semantic-ceiling --remove-above 1.5 --flip-z
"""


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", default="Replica", help="dataset name, e.g. Replica")
	parser.add_argument("--scene", required=True, help="scene name, e.g. office2")
	parser.add_argument("--method", default="SemanticHeat", help="method name, e.g. SemanticHeat")
	parser.add_argument("--run-version", required=True, help="run version, e.g. run_0_v1")
	parser.add_argument("--out", default=None, help="optional output image path; default is auto-generated in run visualization dir")
	parser.add_argument("--res", type=float, default=0.015, help="meters per pixel (smaller = finer, less blocky)")
	parser.add_argument("--pad", type=float, default=0.05, help="bbox pad fraction")
	parser.add_argument("--dpi", type=int, default=400, help="output figure DPI")
	parser.add_argument("--interpolation", choices=["nearest", "bilinear", "bicubic", "lanczos"], default="lanczos", help="imshow interpolation for smoother rendering")
	parser.add_argument("--label-map", help="optional JSON file mapping label->[r,g,b] or the class_colormap style")
	parser.add_argument("--floor-id", type=int, default=None, help="semantic id for floor (optional)")
	parser.add_argument("--ceiling-id", type=int, default=None, help="semantic id for ceiling (optional)")
	parser.add_argument("--align-floor", action="store_true", help="estimate floor plane and align it to z=0")
	parser.add_argument("--remove-semantic-ceiling", action="store_true", help="remove points whose semantic label is ceiling id")
	parser.add_argument("--auto-semantic-detect", action="store_true", help="auto-detect floor/ceiling ids from semantic means")
	parser.add_argument("--remove-above", type=float, default=None, help="remove points with z > VALUE (meters)")
	parser.add_argument("--remove-top-percent", type=float, default=None, help="remove points with z above this percentile (0-100)")
	parser.add_argument("--flip-z", action="store_true", help="multiply z by -1 (flip vertical)")
	args = parser.parse_args()

	npz_path, sem_ply_path, out_path = resolve_io_paths(
		dataset=args.dataset,
		scene=args.scene,
		method=args.method,
		run_version=args.run_version,
		out_override=args.out,
	)
	print("Resolved npz       :", npz_path)
	print("Resolved semantic  :", sem_ply_path)
	print("Resolved output    :", out_path)

	data = load_npz_fields(npz_path)
	sem_data = load_ply_fields(sem_ply_path)

	if args.flip_z:
		for dsrc in [data, sem_data]:
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
			except Exception as e:
				print("align-floor failed:", e)

	label_map = None
	if args.label_map:
		with open(args.label_map, "r") as f:
			raw = json.load(f)
		label_map = normalize_label_map(raw)

	# Keep the same viewpoint logic as render_rgb_sem_entro_bev.py:
	# compute bounds from primary data projection, then render semantic source with those bounds.
	_, _, view_bounds = bev_from_ply(
		data,
		res=args.res,
		pad=args.pad,
		label_map=label_map,
		remove_above=args.remove_above,
		remove_top_percent=args.remove_top_percent,
		return_bounds=True,
		render_semantic=False,
		render_entropy=True,
	)

	sem_source = sem_data if sem_data is not None else data
	sem, _ = bev_from_ply(
		sem_source,
		res=args.res,
		pad=args.pad,
		label_map=label_map,
		remove_above=args.remove_above,
		remove_top_percent=args.remove_top_percent,
		bounds_override=view_bounds,
		render_semantic=True,
		render_entropy=False,
	)

	save_semantic_figure(sem, out_path, dpi=args.dpi, interpolation=args.interpolation)
	print("Saved", out_path)


if __name__ == "__main__":
	main()
