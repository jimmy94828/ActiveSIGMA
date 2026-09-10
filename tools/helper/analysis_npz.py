"""analysis_npz.py

Small utility to inspect and visualise contents of .npz archives.

Usage:
  python analysis_npz.py file1.npz [file2.npz ...] 

Features:
- lists keys, shapes and dtypes
- computes simple statistics (min/max/mean/std)
- for small arrays shows unique counts
- optionally saves image previews for array-like data
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import numpy as np


def summarize_array(a: np.ndarray) -> Dict[str, Any]:
	out: Dict[str, Any] = {
		"shape": a.shape,
		"dtype": str(a.dtype),
		"size": int(a.size),
	}
	try:
		if a.size > 0 and np.issubdtype(a.dtype, np.number):
			out.update({
				"min": float(np.min(a)),
				"max": float(np.max(a)),
				"mean": float(np.mean(a)),
				"std": float(np.std(a)),
			})
	except Exception:
		pass

	if a.size <= 10000:
		# show some sample values and number of unique entries
		try:
			out["n_unique"] = int(np.unique(a).size)
			out["sample_values"] = np.asarray(a).ravel()[:20].tolist()
		except Exception:
			pass

	return out


def is_image_like(a: np.ndarray) -> bool:
	"""Return True when array looks like an image (H,W) or (H,W,1/3).
	"""
	if a.ndim == 2:
		return True
	if a.ndim == 3 and a.shape[2] in (1, 3, 4):
		return True
	return False


def save_image_preview(arr: np.ndarray, out_path: str) -> None:
	# Lazy import to keep dependencies optional
	try:
		import matplotlib.pyplot as plt
	except Exception:
		raise RuntimeError("matplotlib is required to save image previews")

	a = np.array(arr)
	fig = None
	if a.ndim == 2:
		fig = plt.figure(frameon=False)
		plt.axis("off")
		plt.imshow(a, cmap="gray")
	elif a.ndim == 3:
		if a.shape[2] == 1:
			fig = plt.figure(frameon=False)
			plt.axis("off")
			plt.imshow(a[:, :, 0], cmap="gray")
		else:
			# try to normalize if float
			im = a
			if np.issubdtype(a.dtype, np.floating):
				amin, amax = np.nanmin(im), np.nanmax(im)
				if amax > amin:
					im = (im - amin) / (amax - amin)
			fig = plt.figure(frameon=False)
			plt.axis("off")
			plt.imshow(im)

	if fig is not None:
		fig.tight_layout(pad=0)
		fig.savefig(out_path, dpi=150)
		plt.close(fig)


def analyze_file(path: str, args: argparse.Namespace) -> Dict[str, Any]:
	res: Dict[str, Any] = {"file": path, "arrays": {}}
	if not os.path.exists(path):
		raise FileNotFoundError(path)

	with np.load(path, allow_pickle=True) as data:
		keys = list(data.files)
		res["keys"] = keys
		for k in keys:
			try:
				a = data[k]
			except Exception as e:
				res["arrays"][k] = {"error": str(e)}
				continue

			summary = summarize_array(a)
			res["arrays"][k] = summary

			if args.save_images and is_image_like(a):
				# create output filename
				basename = os.path.splitext(os.path.basename(path))[0]
				outdir = args.save_images
				os.makedirs(outdir, exist_ok=True)
				out_path = os.path.join(outdir, f"{basename}__{k}.png")
				try:
					save_image_preview(a, out_path)
					summary["preview"] = out_path
				except Exception as e:
					summary["preview_error"] = str(e)

	return res

"""
Usage example:
  python analysis_npz.py results/Replica/office0/SemanticHeat/run_0_v2/splatam/final/params.npz --save-json data.json
"""
def main() -> None:
	p = argparse.ArgumentParser(description="Inspect and optionally visualise .npz archives")
	p.add_argument("files", nargs="+", help="One or more .npz files to analyze")
	p.add_argument("--save-images", default=None, help="Directory to save image previews")
	p.add_argument("--save-json", default=None, help="Save analysis summary to JSON file")
	p.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
	args = p.parse_args()

	all_res = []
	for f in args.files:
		try:
			r = analyze_file(f, args)
			all_res.append(r)
			if args.save_json is None:
				if args.pretty:
					print(json.dumps(r, indent=2, ensure_ascii=False))
				else:
					print(json.dumps(r, ensure_ascii=False))
		except Exception as e:
			print(f"Error analyzing {f}: {e}")

	if args.save_json:
		with open(args.save_json, "w", encoding="utf-8") as fh:
			if args.pretty:
				json.dump(all_res, fh, indent=2, ensure_ascii=False)
			else:
				json.dump(all_res, fh, ensure_ascii=False)


if __name__ == "__main__":
	main()

