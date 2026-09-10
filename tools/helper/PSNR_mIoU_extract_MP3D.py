#!/usr/bin/env python3
"""Batch extract PSNR and miou_g_curr for Replica SemanticHeat scenes.

Usage:
    python tools/helper/PSNR_mIoU_extract.py run_0_v6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Union


DATASET = "MP3D"
METHOD = "SemanticHeat"
TRACKER = "splatam"
EVAL_DIR = "eval_exploration_stage_1"     # for activeSGM
# EVAL_DIR = "eval_final"         # for active mapping test
DECIMAL_PLACES = 4
SCENES = [
    "GdvgFV5R1Z5",
    "gZ6f7yhEvPG",
    "HxpKQynjfin",
    "pLe4wQe7qrG",
]


class FixedPrecisionFloatEncoder(json.JSONEncoder):
    """JSON encoder that keeps fixed decimal places for float values."""

    def iterencode(self, obj, _one_shot: bool = False):
        if self.check_circular:
            markers = {}
        else:
            markers = None

        _encoder = (
            json.encoder.encode_basestring_ascii
            if self.ensure_ascii
            else json.encoder.encode_basestring
        )

        def floatstr(
            value: float,
            allow_nan: bool = self.allow_nan,
            _inf: float = float("inf"),
            _neginf: float = -float("inf"),
        ) -> str:
            if value != value:
                text = "NaN"
            elif value == _inf:
                text = "Infinity"
            elif value == _neginf:
                text = "-Infinity"
            else:
                return f"{value:.{DECIMAL_PLACES}f}"

            if not allow_nan:
                raise ValueError(
                    f"Out of range float values are not JSON compliant: {value!r}"
                )
            return text

        _iterencode = json.encoder._make_iterencode(
            markers,
            self.default,
            _encoder,
            self.indent,
            floatstr,
            self.key_separator,
            self.item_separator,
            self.sort_keys,
            self.skipkeys,
            _one_shot,
        )
        return _iterencode(obj, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract PSNR and miou_g_curr for Replica SemanticHeat scenes."
    )
    parser.add_argument(
        "runversion",
        help="Run version name, e.g. run_0_v6",
    )
    return parser.parse_args()


def parse_metric_file(file_path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    with file_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue

            key, value = line.split(":", 1)
            metrics[key.strip()] = float(value.strip())

    return metrics


def collect_scene_result(results_root: Path, scene: str, runversion: str) -> Union[str, Dict[str, float]]:
    eval_dir = (
        results_root
        / DATASET
        / scene
        / METHOD
        / runversion
        / TRACKER
        / EVAL_DIR
    )
    render_result_path = eval_dir / "render_result.txt"
    semantic_result_path = eval_dir / "semantic_result.txt"

    if not render_result_path.is_file() or not semantic_result_path.is_file():
        return "No version"

    render_metrics = parse_metric_file(render_result_path)
    semantic_metrics = parse_metric_file(semantic_result_path)

    return {
        "psnr": render_metrics["psnr"],
        "miou_g_curr": semantic_metrics["miou_g_curr"],
    }


def main() -> None:
    args = parse_args()

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    results_root = project_root / "results_MP3D"
    output_path = script_path.with_name(f"MP3D_{args.runversion}_PSNR_mIoU.json")

    output = {
        "dataset": DATASET,
        "method": METHOD,
        "runversion": args.runversion,
        "tracker": TRACKER,
        "eval_dir": EVAL_DIR,
        "results": {},
    }

    for scene in SCENES:
        output["results"][scene] = collect_scene_result(results_root, scene, args.runversion)

    # Average PSNR and miou_g_curr across all scenes (excluding "No version")
    psnr_sum = 0.0
    miou_sum = 0.0
    count = 0
    for scene in SCENES:
        for scene_result in output["results"].values():
            if isinstance(scene_result, dict):
                psnr_sum += scene_result["psnr"]
                miou_sum += scene_result["miou_g_curr"]
                count += 1
                
    if count > 0:
        output["results"]["average"] = {
            "psnr": round(psnr_sum / count, DECIMAL_PLACES),
            "miou_g_curr": round(miou_sum / count, DECIMAL_PLACES),
        }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, cls=FixedPrecisionFloatEncoder)

    print(f"Saved JSON to: {output_path}")


if __name__ == "__main__":
    main()
