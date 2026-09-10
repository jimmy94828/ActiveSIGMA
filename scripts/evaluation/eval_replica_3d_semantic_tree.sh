#!/bin/bash
set -euo pipefail

##################################################
### Generic Replica 3D semantic evaluation for a result tree.
###
### It scans all Replica scenes and run folders under result_root.
### Optional environment variable:
###   RUN_PATTERN=run_1_eval        evaluate only run paths containing this string
###   RUN_PATTERN=sgsslam/run_0     evaluate only SG-SLAM run_0
###
### Usage:
###   bash scripts/evaluation/eval_replica_3d_semantic_tree.sh [result_root] [latest|final|steps|all] [num_samples] [distance_threshold] [opacity_threshold]
###
### Example:
###   conda run -n activemapping bash scripts/evaluation/eval_replica_3d_semantic_tree.sh /media/phudh/X9_Pro/undergraduate/ActiveSGM/results/Replica latest
##################################################

result_root=${1:-/media/phudh/X9_Pro/undergraduate/ActiveSGM/results/Replica}
mode=${2:-latest}
num_samples=${3:-200000}
distance_threshold=${4:-0.05}
opacity_threshold=${5:-0.05}
run_pattern=${RUN_PATTERN:-}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
proj_dir="$(cd "${script_dir}/../.." && pwd)"

gt_root="${proj_dir}/data/replica_v1"
traj_root="${proj_dir}/data/Replica"
eval_script="${proj_dir}/src/evaluation/eval_3d_semantic.py"

if [[ "${mode}" != "latest" && "${mode}" != "final" && "${mode}" != "steps" && "${mode}" != "all" ]]; then
    echo "mode must be one of: latest, final, steps, all" >&2
    exit 1
fi

replica_scene_name() {
    local scene=$1
    if [[ "${scene}" == office* ]]; then
        echo "office_${scene#office}"
    elif [[ "${scene}" == room* ]]; then
        echo "room_${scene#room}"
    else
        echo "${scene}"
    fi
}

run_dir_from_params() {
    local params_path=$1
    local dir
    dir=$(dirname "${params_path}")
    if [[ "$(basename "${dir}")" == "final" ]]; then
        dir=$(dirname "${dir}")
    fi
    dirname "${dir}"
}

output_tag_from_params() {
    local params_path=$1
    local dir
    dir=$(dirname "${params_path}")
    if [[ "$(basename "${dir}")" == "final" ]]; then
        echo "final"
        return
    fi
    local base_name
    base_name=$(basename "${params_path}" .npz)
    if [[ "${base_name}" == params* ]]; then
        echo "step_${base_name#params}"
    else
        echo "${base_name}"
    fi
}

eval_one_npz() {
    local params_path=$1
    local output_tag_override=${2:-}
    local run_dir
    run_dir=$(run_dir_from_params "${params_path}")

    local rel_run="${run_dir#${result_root}/}"
    if [[ -n "${run_pattern}" && "${rel_run}" != *"${run_pattern}"* ]]; then
        return
    fi

    local scene="${rel_run%%/*}"
    local replica_scene
    replica_scene=$(replica_scene_name "${scene}")

    local gt_mesh="${gt_root}/${replica_scene}/habitat/mesh_semantic.ply"
    local semantic_info="${gt_root}/${replica_scene}/habitat/info_semantic.json"
    local transform_traj="${traj_root}/${scene}/traj.txt"
    local output_tag
    if [[ -n "${output_tag_override}" ]]; then
        output_tag="${output_tag_override}"
    else
        output_tag=$(output_tag_from_params "${params_path}")
    fi
    local output_dir="${run_dir}/eval_3d_semantic/${output_tag}"

    if [[ ! -f "${params_path}" ]]; then
        echo "[skip] missing params ${params_path}"
        return
    fi
    if [[ ! -f "${gt_mesh}" || ! -f "${semantic_info}" || ! -f "${transform_traj}" ]]; then
        echo "[skip] ${rel_run}: missing GT or trajectory files"
        return
    fi

    echo "[eval] scene=${scene} run=${rel_run} tag=${output_tag}"
    python "${eval_script}" \
        --dataset Replica \
        --scene "${scene}" \
        --params "${params_path}" \
        --gt_mesh "${gt_mesh}" \
        --semantic_info "${semantic_info}" \
        --transform_traj "${transform_traj}" \
        --output_dir "${output_dir}" \
        --num_samples "${num_samples}" \
        --distance_threshold "${distance_threshold}" \
        --opacity_threshold "${opacity_threshold}"
}


if [[ "${mode}" == "latest" ]]; then
    while IFS= read -r splatam_dir; do
        params_path="${splatam_dir}/final/params.npz"
        if [[ ! -f "${params_path}" ]]; then
            params_path=$(find "${splatam_dir}" -maxdepth 1 -type f -name 'params*.npz' | sort -V | tail -n 1)
        fi
        if [[ -n "${params_path}" ]]; then
            eval_one_npz "${params_path}" "latest"
        fi
    done < <(find "${result_root}" -type d -name splatam | sort -V)
fi

if [[ "${mode}" == "final" || "${mode}" == "all" ]]; then
    while IFS= read -r params_path; do
        eval_one_npz "${params_path}"
    done < <(find "${result_root}" -type f -path '*/splatam/final/params.npz' | sort -V)
fi

if [[ "${mode}" == "steps" || "${mode}" == "all" ]]; then
    while IFS= read -r params_path; do
        eval_one_npz "${params_path}"
    done < <(find "${result_root}" -type f -path '*/splatam/params*.npz' | sort -V)
fi

suffix="${mode}"
if [[ -n "${run_pattern}" ]]; then
    safe_pattern=${run_pattern//\//_}
    safe_pattern=${safe_pattern// /_}
    suffix="${safe_pattern}_${mode}"
fi
summary_csv="${result_root}/eval_3d_semantic_summary_${suffix}.csv"
python - "${result_root}" "${summary_csv}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
rows = []

for metrics_path in sorted(root.rglob("metrics.json")):
    parts = metrics_path.parts
    if "eval_3d_semantic" not in parts:
        continue
    tag = metrics_path.parent.name
    run_dir = metrics_path.parents[2]
    rel = run_dir.relative_to(root)
    scene = rel.parts[0]
    with metrics_path.open("r") as f:
        metrics = json.load(f)
    rows.append({
        "scene": scene,
        "run_rel": str(rel),
        "tag": tag,
        "miou": metrics.get("gt_domain_miou"),
        "macc": metrics.get("gt_domain_macc"),
        "overall_acc": metrics.get("gt_domain_overall_acc"),
        "semantic_fscore": metrics.get("semantic_fscore_at_threshold"),
        "semantic_precision": metrics.get("pred_semantic_precision_at_threshold"),
        "semantic_recall": metrics.get("gt_semantic_recall_at_threshold"),
        "geometry_precision": metrics.get("pred_geometry_precision_at_threshold"),
        "geometry_recall": metrics.get("gt_geometry_recall_at_threshold"),
        "num_pred_points": metrics.get("num_pred_points_after_filter"),
    })

summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "scene",
        "run_rel",
        "tag",
        "miou",
        "macc",
        "overall_acc",
        "semantic_fscore",
        "semantic_precision",
        "semantic_recall",
        "geometry_precision",
        "geometry_recall",
        "num_pred_points",
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote summary CSV: {summary_csv}")
PY
