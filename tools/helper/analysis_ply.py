"""Small utility to inspect PLY files: header, properties, basic stats.

Usage:
	python tools/helper/analysis_ply.py --ply path/to/file.ply

Outputs a summary to stdout and optional JSON if `--json out.json`.
"""

import argparse
import json
import numpy as np
from plyfile import PlyData


def analyze_ply(path, sample=5):
	ply = PlyData.read(path)
	out = {}

	# elements and counts
	elems = {}
	for el in ply.elements:
		elems[el.name] = int(el.count)
	out['elements'] = elems

	# vertex properties
	if 'vertex' in ply:
		v = ply['vertex'].data
		props = v.dtype.names
		out['vertex_properties'] = list(props)
		# compute simple stats for numeric properties
		stats = {}
		arr = v
		for p in props:
			try:
				col = np.array(arr[p])
			except Exception:
				continue
			if np.issubdtype(col.dtype, np.number):
				stats[p] = {
					'min': float(np.min(col)),
					'max': float(np.max(col)),
					'mean': float(np.mean(col)),
					'std': float(np.std(col)),
				}
				# sample values
				stats[p]['sample'] = [float(x) for x in col[:sample]]
			else:
				stats[p] = {'dtype': str(col.dtype), 'sample': [str(x) for x in col[:sample]]}
		out['vertex_stats'] = stats

	# list other elements
	for el in ply.elements:
		if el.name == 'vertex':
			continue
		props = el.properties
		out.setdefault('other_elements', {})[el.name] = {
			'count': int(el.count),
			'properties': [p.name for p in props]
		}

	return out

"""
python tools/helper/analysis_ply.py --ply data/Replica/office4_mesh.ply --json results/Replica/office4/SemanticHeat/run_0_v6/splatam/office4_mesh_info.json
"""
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--ply', required=True)
	parser.add_argument('--json', help='optional json output path')
	args = parser.parse_args()

	info = analyze_ply(args.ply)
	print(json.dumps(info, indent=2))
	if args.json:
		with open(args.json, 'w') as f:
			json.dump(info, f, indent=2)


if __name__ == '__main__':
	main()

