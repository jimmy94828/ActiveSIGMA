"""
GT evaluation trajectory BEV renderer.

Renders a top-down semantic BEV directly from the dataset GT semantic mesh, then
overlays an evaluation trajectory stored as 4x4 text poses such as pose_rdf/*.txt.
"""

import argparse
import glob
import json
import os
import re
import sys
import typing

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np

if __package__ is None or __package__ == "":
	REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
	if REPO_ROOT not in sys.path:
		sys.path.insert(0, REPO_ROOT)

from src.visualization.render_rgb_sem_entro_bev import (  # noqa: E402
	compute_bounds,
	label_colors,
	load_gt_semantic,
	normalize_label_map,
	project_points_to_bev,
	render_gt_semantic_bev,
	top_surface_indices,
)


DEFAULT_EVAL_TRAJ_DIR = (
	"/media/phudh/HDD/trajectory_generate/keyboard_trajectories/"
	"Replica/office0/test/pose_rdf"
)


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


def load_replica_gt_semantic_fast(scene: str) -> typing.Tuple[np.ndarray, np.ndarray]:
	"""Load Replica GT semantic face centroids with a vectorized binary PLY parser."""
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
		vertex_count = None
		face_count = None
		while True:
			line = f.readline()
			if not line:
				raise ValueError(f"Unexpected EOF while reading PLY header: {mesh_path}")
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

	tri_dtype = np.dtype([("nverts", "u1"), ("verts", "<u4", (3,)), ("object_id", "<u2")])
	quad_dtype = np.dtype([("nverts", "u1"), ("verts", "<u4", (4,)), ("object_id", "<u2")])
	if len(face_blob) == face_count * tri_dtype.itemsize:
		faces = np.frombuffer(face_blob, dtype=tri_dtype, count=face_count)
		if not np.all(faces["nverts"] == 3):
			raise ValueError("Replica GT triangle face layout has non-triangle records")
		face_kind = "tri"
	elif len(face_blob) == face_count * quad_dtype.itemsize:
		faces = np.frombuffer(face_blob, dtype=quad_dtype, count=face_count)
		if not np.all(faces["nverts"] == 4):
			raise ValueError("Replica GT quad face layout has non-quad records")
		face_kind = "quad"
	else:
		raise ValueError("Replica GT face layout is not fixed triangle/quad+object_id; fallback required")

	object_ids = faces["object_id"].astype(np.int64)
	valid_object = object_ids < id_to_label.shape[0]
	labels = np.full(face_count, -1, dtype=np.int32)
	labels[valid_object] = id_to_label[object_ids[valid_object]]
	keep = valid_object
	if not np.any(keep):
		raise ValueError(f"No valid GT semantic labels loaded from {mesh_path}")

	if face_kind == "tri":
		tri_vertices = vertices[faces["verts"][keep]]
		tri_centroids = tri_vertices.mean(axis=1).astype(np.float32)
		return tri_centroids, labels[keep].astype(np.int32)

	quad_vertices = vertices[faces["verts"][keep]]
	tri_a = (quad_vertices[:, 0] + quad_vertices[:, 1] + quad_vertices[:, 2]) / 3.0
	tri_b = (quad_vertices[:, 0] + quad_vertices[:, 2] + quad_vertices[:, 3]) / 3.0
	tri_centroids = np.concatenate([tri_a, tri_b], axis=0).astype(np.float32)
	tri_labels = np.concatenate([labels[keep], labels[keep]], axis=0).astype(np.int32)
	return tri_centroids, tri_labels


def load_gt_semantic_points(dataset: str, scene: str) -> typing.Tuple[np.ndarray, np.ndarray]:
	if dataset.lower() == "replica":
		try:
			return load_replica_gt_semantic_fast(scene)
		except Exception as exc:
			print(f"[WARN] Fast Replica GT loader failed, falling back to generic loader: {exc}")
	return load_gt_semantic(dataset, scene)


def _natural_key(path: str) -> typing.List[typing.Union[int, str]]:
	key: typing.List[typing.Union[int, str]] = []
	for part in re.split(r"(\d+)", os.path.basename(path)):
		if part.isdigit():
			key.append(int(part))
		elif part:
			key.append(part)
	return key


def resolve_pose_text_files(path: str) -> typing.List[str]:
	"""Resolve an eval pose path to pose text files."""
	if os.path.isfile(path):
		return [path]
	if not os.path.isdir(path):
		raise FileNotFoundError(f"Evaluation trajectory path not found: {path}")

	pose_rdf_dir = os.path.join(path, "pose_rdf")
	if os.path.isdir(pose_rdf_dir):
		path = pose_rdf_dir
	else:
		traj_txt = os.path.join(path, "traj.txt")
		if os.path.isfile(traj_txt):
			return [traj_txt]

	files = glob.glob(os.path.join(path, "*.txt"))
	files.sort(key=_natural_key)
	if not files:
		raise FileNotFoundError(f"No .txt pose files found in: {path}")
	return files


def load_pose_matrices(path: str) -> np.ndarray:
	"""Load one or more 4x4 pose matrices from a text file or pose folder."""
	poses = []
	for fp in resolve_pose_text_files(path):
		with open(fp, "r", encoding="utf-8") as f:
			values = np.fromstring(f.read(), sep=" ", dtype=np.float32)
		if values.size == 0:
			continue
		if values.size % 16 != 0:
			raise ValueError(f"Expected a multiple of 16 pose values in {fp}, got {values.size}")
		poses.extend(values.reshape(-1, 4, 4))

	if not poses:
		raise ValueError(f"No valid poses loaded from: {path}")
	return np.stack(poses, axis=0).astype(np.float32)


def load_eval_trajectory_points(eval_traj_path: str, pose_format: str) -> np.ndarray:
	"""Load evaluation camera centers (N,3) from c2w or w2c text poses."""
	poses = load_pose_matrices(eval_traj_path)
	if pose_format == "w2c":
		poses = np.linalg.inv(poses)
	return poses[:, :3, 3].astype(np.float32)


def default_mapping_traj_file(dataset: str, scene: str) -> typing.Optional[str]:
	candidate = os.path.join("data", dataset, scene, "traj.txt")
	return candidate if os.path.isfile(candidate) else None


def resolve_mapping_traj_file(
	dataset: str,
	scene: str,
	override: typing.Optional[str],
) -> typing.Optional[str]:
	if override:
		if not os.path.isfile(override) and not os.path.isdir(override):
			raise FileNotFoundError(f"Mapping trajectory path not found: {override}")
		return override
	return default_mapping_traj_file(dataset, scene)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
	if points.shape[0] == 0:
		return points.astype(np.float32)
	ones = np.ones((points.shape[0], 1), dtype=np.float32)
	points_h = np.concatenate([points.astype(np.float32), ones], axis=1)
	return (transform @ points_h.T).T[:, :3].astype(np.float32)


def outside_ratio_xy(traj_pts: np.ndarray, ref_points: np.ndarray) -> typing.Tuple[int, int, float]:
	if traj_pts.shape[0] == 0 or ref_points.shape[0] == 0:
		return 0, int(traj_pts.shape[0]), 0.0
	xmin = float(np.min(ref_points[:, 0]))
	xmax = float(np.max(ref_points[:, 0]))
	ymin = float(np.min(ref_points[:, 1]))
	ymax = float(np.max(ref_points[:, 1]))
	outside = (
		(traj_pts[:, 0] < xmin)
		| (traj_pts[:, 0] > xmax)
		| (traj_pts[:, 1] < ymin)
		| (traj_pts[:, 1] > ymax)
	)
	return int(np.count_nonzero(outside)), int(traj_pts.shape[0]), float(np.mean(outside))


def resolve_eval_points_in_gt_frame(
	raw_eval_pts: np.ndarray,
	gt_points: np.ndarray,
	args: argparse.Namespace,
) -> typing.Tuple[np.ndarray, str, typing.Optional[str]]:
	"""Resolve eval trajectory points into GT sim/world coordinates."""
	if args.eval_pose_frame == "sim":
		return raw_eval_pts, "sim", None

	mapping_file = resolve_mapping_traj_file(args.dataset, args.scene, args.mapping_traj_file)
	if mapping_file is None:
		if args.eval_pose_frame == "slam":
			raise FileNotFoundError(
				f"Could not infer mapping trajectory for {args.dataset}/{args.scene}. "
				"Pass --mapping-traj-file or use --eval-pose-frame sim."
			)
		return raw_eval_pts, "sim(auto fallback)", None

	slam_to_sim = load_pose_matrices(mapping_file)[0]
	transformed_pts = transform_points(raw_eval_pts, slam_to_sim)

	if args.eval_pose_frame == "slam":
		return transformed_pts, "slam->sim", mapping_file

	raw_outside = outside_ratio_xy(raw_eval_pts, gt_points)[2]
	transformed_outside = outside_ratio_xy(transformed_pts, gt_points)[2]
	if raw_outside <= transformed_outside:
		return raw_eval_pts, "sim(auto)", None
	return transformed_pts, "slam->sim(auto)", mapping_file


def default_label_map_path(dataset: str, scene: str) -> typing.Optional[str]:
	candidate = os.path.join("configs", dataset, scene, "class_color_map.json")
	return candidate if os.path.isfile(candidate) else None


def load_label_map(path: typing.Optional[str], dataset: str, scene: str) -> typing.Optional[dict]:
	if path is None or path.lower() == "none":
		return None
	if path == "auto":
		path = default_label_map_path(dataset, scene)
		if path is None:
			return None
	with open(path, "r", encoding="utf-8") as f:
		return normalize_label_map(json.load(f))


def render_gt_semantic_nearest(
	gt_points: np.ndarray,
	gt_labels: np.ndarray,
	bounds,
	res: float,
	flip: bool,
	label_map: typing.Optional[dict],
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
		raise ValueError("No GT semantic points remain after filtering")

	_, _, ix, iy, linear = project_points_to_bev(gt_points, bounds=bounds, res=res)
	chosen_idx = top_surface_indices(linear, gt_points[:, 2], flip=flip)
	_, _, _, _, width, height = bounds
	image = np.full((height, width, 3), 255, dtype=np.uint8)
	lut = label_colors(gt_labels, label_map)
	for idx in chosen_idx:
		image[iy[idx], ix[idx]] = lut[int(gt_labels[idx])]
	return np.flipud(image)


def figure_canvas_to_rgb(fig) -> np.ndarray:
	fig.canvas.draw()
	if hasattr(fig.canvas, "tostring_rgb"):
		w, h = fig.canvas.get_width_height()
		return np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
	return np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3].copy()


def draw_trajectory_on_bev(
	img: np.ndarray,
	traj_pts: np.ndarray,
	bounds,
	res: float,
	line_color: str,
	line_width: float,
	start_color: str,
	end_color: str,
	marker_size: float,
) -> np.ndarray:
	if img is None or traj_pts is None or traj_pts.shape[0] < 2 or bounds is None:
		return img

	xmin, xmax, ymin, ymax, width, height = bounds
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

	ix = np.clip((xs - xmin) / res, 0, width - 1)
	iy = np.clip((ys - ymin) / res, 0, height - 1)
	iy_img = (height - 1) - iy

	fig, ax = plt.subplots(1, 1, figsize=(6, 6))
	ax.imshow(img, interpolation="nearest")
	ax.plot(ix, iy_img, color=line_color, linewidth=line_width, alpha=0.95)
	ax.scatter(ix[0], iy_img[0], c=start_color, s=marker_size, edgecolors="black", linewidths=0.8, zorder=3)
	ax.scatter(ix[-1], iy_img[-1], c=end_color, s=marker_size, edgecolors="black", linewidths=0.8, zorder=3)
	ax.axis("off")
	fig.tight_layout(pad=0)
	rgb = figure_canvas_to_rgb(fig)
	plt.close(fig)
	return rgb


def add_rb_legend(
	img: np.ndarray,
	r_text: str,
	b_text: str,
	r_color: str,
	b_color: str,
) -> np.ndarray:
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


def save_bev_figure(img: np.ndarray, out_path: str, dpi: int, interpolation: str, title: typing.Optional[str]) -> None:
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


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Render GT semantic BEV and overlay an evaluation trajectory."
	)
	parser.add_argument("--dataset", default="Replica", help="dataset name")
	parser.add_argument("--scene", default="office0", help="scene name")
	parser.add_argument(
		"--eval-traj-dir",
		"--eval-traj",
		dest="eval_traj_dir",
		default=DEFAULT_EVAL_TRAJ_DIR,
		help="evaluation pose_rdf dir, eval data dir, or traj.txt file",
	)
	parser.add_argument(
		"--mapping-traj-file",
		default=None,
		help="mapping trajectory used only when --eval-pose-frame slam/auto; defaults to data/<dataset>/<scene>/traj.txt",
	)
	parser.add_argument("--out", default=None, help="optional output image path")
	parser.add_argument("--res", type=float, default=0.015, help="meters per pixel")
	parser.add_argument("--pad", type=float, default=0.05, help="XY bound padding ratio")
	parser.add_argument("--dpi", type=int, default=400, help="output figure DPI")
	parser.add_argument(
		"--interpolation",
		choices=["nearest", "bilinear", "bicubic", "lanczos"],
		default="lanczos",
		help="imshow interpolation for final saved figure",
	)
	parser.add_argument("--min-height", type=float, default=None, help="optional lower GT z bound")
	parser.add_argument("--max-height", type=float, default=None, help="optional upper GT z bound")
	parser.add_argument("--front-cutoff", type=float, default=None, help="optional top-surface cutoff used by GT renderer")
	parser.add_argument("--render-mode", choices=["nearest", "gaussian"], default="nearest", help="GT BEV renderer")
	parser.add_argument("--gt-sigma-px", type=float, default=1.0, help="GT semantic Gaussian interpolation sigma in pixels")
	parser.add_argument("--label-map", default="auto", help="JSON label map; auto uses configs/<dataset>/<scene>/class_color_map.json")
	parser.add_argument("--flip", action="store_true", help="use the lowest visible surface instead of top-down highest surface")
	parser.add_argument(
		"--eval-pose-format",
		choices=["c2w", "w2c"],
		default="c2w",
		help="pose matrix convention in evaluation trajectory text files",
	)
	parser.add_argument(
		"--eval-pose-frame",
		choices=["sim", "slam", "auto"],
		default="sim",
		help="coordinate frame of eval poses; pose_rdf GT should use sim",
	)
	parser.add_argument("--traj-color", default="yellow", help="trajectory line color")
	parser.add_argument("--traj-linewidth", type=float, default=2.0, help="trajectory line width")
	parser.add_argument("--traj-start-color", default="blue", help="start marker color")
	parser.add_argument("--traj-end-color", default="red", help="end marker color")
	parser.add_argument("--traj-marker-size", type=float, default=36.0, help="start/end marker size")
	parser.add_argument("--legend-r-text", default="End", help="legend text for red marker")
	parser.add_argument("--legend-b-text", default="Start", help="legend text for blue marker")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	out_path = args.out
	if out_path is None:
		out_path = os.path.join("results", args.dataset, args.scene, "eval_gt_traj_bev.png")
	out_dir = os.path.dirname(out_path)
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)

	gt_points, gt_labels = load_gt_semantic_points(args.dataset, args.scene)
	raw_eval_pts = load_eval_trajectory_points(args.eval_traj_dir, args.eval_pose_format)
	traj_pts, eval_frame_used, mapping_file = resolve_eval_points_in_gt_frame(raw_eval_pts, gt_points, args)
	label_map = load_label_map(args.label_map, args.dataset, args.scene)

	bounds_points = np.concatenate([gt_points[:, :2], traj_pts[:, :2]], axis=0)
	bounds = compute_bounds(bounds_points, res=args.res, pad=args.pad)
	if args.render_mode == "gaussian":
		bev_img = render_gt_semantic_bev(
			gt_points=gt_points,
			gt_labels=gt_labels,
			bounds=bounds,
			res=args.res,
			flip=args.flip,
			label_map=label_map,
			interpolation_sigma_px=args.gt_sigma_px,
			front_cutoff=args.front_cutoff,
			min_height=args.min_height,
			max_height=args.max_height,
		)
	else:
		bev_img = render_gt_semantic_nearest(
			gt_points=gt_points,
			gt_labels=gt_labels,
			bounds=bounds,
			res=args.res,
			flip=args.flip,
			label_map=label_map,
			front_cutoff=args.front_cutoff,
			min_height=args.min_height,
			max_height=args.max_height,
		)
	bev_img = draw_trajectory_on_bev(
		img=bev_img,
		traj_pts=traj_pts,
		bounds=bounds,
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
		title="GT Semantic + Eval Trajectory",
	)

	outside_n, total_n, outside_ratio = outside_ratio_xy(traj_pts, gt_points)
	print("GT dataset       :", args.dataset)
	print("GT scene         :", args.scene)
	print("Resolved eval traj:", args.eval_traj_dir)
	print("Eval poses       :", int(traj_pts.shape[0]))
	print("Eval pose format :", args.eval_pose_format)
	print("Eval pose frame  :", eval_frame_used)
	print("Render mode      :", args.render_mode)
	if mapping_file is not None:
		print("Mapping traj     :", mapping_file)
	print(f"Trajectory outside GT XY bbox: {outside_n}/{total_n} ({outside_ratio * 100.0:.2f}%)")
	print(f"BEV size         : {bounds[5]} x {bounds[4]}")
	print("Saved", out_path)


if __name__ == "__main__":
	main()
