#!/usr/bin/env python3
"""Extract MP3D render and semantic metrics from optimize52 logs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "results_all/MP3D"
SCENES = ["GdvgFV5R1Z5", "gZ6f7yhEvPG", "HxpKQynjfin", "pLe4wQe7qrG"]
METRIC_ORDER = [
    "psnr",
    "miou_g_curr",
    "miou_g",
    "ssim",
    "lpips",
    "l1(cm)",
    "rmse",
    "top1_g",
    "top3_g",
    "top5_g",
    "mAP_g",
    "f1_g",
    "miou_p",
    "f1_p",
    "mAP_p",
    "miou_p_curr",
    "miou_rp",
    "top1_p",
    "top3_p",
    "top5_p",
]

PATTERNS = {
    "psnr": r"^Average PSNR:\s*([-+0-9.eE]+)$",
    "depth_rmse_cm": r"^Average Depth RMSE:\s*([-+0-9.eE]+)\s*cm$",
    "ssim": r"^Average MS-SSIM:\s*([-+0-9.eE]+)$",
    "lpips": r"^Average LPIPS:\s*([-+0-9.eE]+)$",
    "miou_g": r"^Average MIOU with GT:\s*([-+0-9.eE]+)$",
    "miou_g_curr": r"^Average MIOU with GT \(current\):\s*([-+0-9.eE]+)$",
    "miou_rp": r"^Average MIOU with Render vs Pseudo:\s*([-+0-9.eE]+)$",
    "top1_g": r"^Average top1 acc with GT:\s*([-+0-9.eE]+)$",
    "top3_g": r"^Average top3 acc with GT:\s*([-+0-9.eE]+)$",
    "top5_g": r"^Average top5 acc with GT:\s*([-+0-9.eE]+)$",
    "mAP_g": r"^Average mAP with GT:\s*([-+0-9.eE]+)$",
    "f1_g": r"^Average F1 with GT:\s*([-+0-9.eE]+)$",
    "miou_p": r"^Average MIOU with Pseudo:\s*([-+0-9.eE]+)$",
    "miou_p_curr": r"^Average MIOU with Pseudo \(current\):\s*([-+0-9.eE]+)$",
    "top1_p": r"^Average top1 acc with Pseudo:\s*([-+0-9.eE]+)$",
    "top3_p": r"^Average top3 acc with Pseudo:\s*([-+0-9.eE]+)$",
    "top5_p": r"^Average top5 acc with Pseudo:\s*([-+0-9.eE]+)$",
    "mAP_p": r"^Average mAP with Pseudo:\s*([-+0-9.eE]+)$",
    "f1_p": r"^Average F1 with Pseudo:\s*([-+0-9.eE]+)$",
}


def summary_block(log_path: Path, eval_dir: str) -> List[str]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    checkpoint_name = eval_dir[5:] if eval_dir.startswith("eval_") else eval_dir
    checkpoint_indices = [
        index
        for index, line in enumerate(lines)
        if line.startswith("Saving parameters to:") and line.rstrip().endswith(f"/{checkpoint_name}")
    ]

    if checkpoint_indices:
        end = checkpoint_indices[-1]
        starts = [index for index in range(end) if lines[index].startswith("Final Average ATE RMSE:")]
    elif eval_dir == "eval_exploration_stage_1":
        # Older GdvgFV5R1Z5 logs saved stage 1 to splatam/ without a stage suffix.
        starts = [index for index, line in enumerate(lines) if line.startswith("Final Average ATE RMSE:")]
        if len(starts) < 2:
            raise ValueError(f"No penultimate evaluation summary found in {log_path}")
        start = starts[-2]
        end = next(
            index
            for index in range(start + 1, len(lines))
            if lines[index].startswith("Saving parameters to:")
        )
        return lines[start : end + 1]
    else:
        raise ValueError(f"No {checkpoint_name} checkpoint marker found in {log_path}")

    if not starts:
        raise ValueError(f"No evaluation summary found before {checkpoint_name} marker in {log_path}")
    return lines[starts[-1] : end + 1]


def parse_scene(log_path: Path, eval_dir: str) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {}
    for line in summary_block(log_path, eval_dir):
        for key, pattern in PATTERNS.items():
            match = re.match(pattern, line.strip())
            if match:
                metrics[key] = float(match.group(1))

    required = set(PATTERNS) - {"miou_rp"}
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError(f"{log_path} is missing final metrics: {missing}")

    depth_rmse_cm = metrics.pop("depth_rmse_cm")
    assert depth_rmse_cm is not None
    metrics["l1(cm)"] = depth_rmse_cm
    metrics["rmse"] = depth_rmse_cm / 100.0
    metrics.setdefault("miou_rp", None)
    return {key: metrics[key] for key in METRIC_ORDER}


def average_results(results: Dict[str, Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    average: Dict[str, Optional[float]] = {}
    for key in METRIC_ORDER:
        values = [scene_metrics[key] for scene_metrics in results.values() if scene_metrics[key] is not None]
        average[key] = round(sum(values) / len(values), 4) if values else None
    return average


def write_output(eval_dir: str) -> Path:
    results = {
        scene: parse_scene(LOG_DIR / f"{scene}_optimize52.txt", eval_dir)
        for scene in SCENES
    }
    output = {
        "dataset": "MP3D",
        "method": "SemanticHeat",
        "runversion": "run_0",
        "tracker": "splatam",
        "eval_dir": eval_dir,
        "results_root": str(LOG_DIR),
        "eval_dir_template": "{scene}_optimize52.txt",
        "results": results,
        "average": average_results(results),
        "missing_scenes": [],
    }
    output_path = OUTPUT_DIR / f"SemanticHeat_run_0_{eval_dir}_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
        file.write("\n")
    print(f"Saved JSON to: {output_path}")
    return output_path


def main() -> None:
    for eval_dir in ("eval_exploration_stage_1", "eval_final"):
        write_output(eval_dir)


if __name__ == "__main__":
    main()
