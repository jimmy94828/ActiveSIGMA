#!/usr/bin/env python3
"""Aggregate per-scene MP3D Table S.5 metrics into CSV, JSON, and Markdown."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


FIELDS = ["avg", "ceiling", "appliances", "sink", "plant", "counter", "table", "mpcat40"]
PAPER_REFERENCE_PERCENT = {
    "avg": 65.58,
    "ceiling": 70.31,
    "appliances": 76.95,
    "sink": 69.36,
    "plant": 73.60,
    "counter": 14.03,
    "table": 69.89,
    "mpcat40": 55.77,
}


def finite_mean(values: List[float]) -> float:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(values) / len(values) if values else float("nan")


def format_percent(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{100.0 * value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for path in sorted(args.input_root.glob("*/*/*/metrics.json")):
        with path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        records.append(metrics)
    if not records:
        raise ValueError(f"No metrics.json files found under {args.input_root}")

    per_scene_path = args.input_root / "table_s5_per_scene_all_stages.csv"
    with per_scene_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["experiment", "scene", "stage", "num_evaluated_frames", "num_gaussians"] + FIELDS,
        )
        writer.writeheader()
        for metrics in records:
            writer.writerow(
                {
                    "experiment": metrics["experiment"],
                    "scene": metrics["scene"],
                    "stage": metrics["stage"],
                    "num_evaluated_frames": metrics["num_evaluated_frames"],
                    "num_gaussians": metrics["num_gaussians"],
                    **metrics["table_s5"],
                }
            )

    grouped: Dict[tuple, List[dict]] = defaultdict(list)
    for metrics in records:
        grouped[(metrics["experiment"], metrics["stage"])].append(metrics)

    summary_rows = []
    for (experiment, stage), group in sorted(grouped.items()):
        values = {
            field: finite_mean([item["table_s5"].get(field) for item in group])
            for field in FIELDS
        }
        summary_rows.append(
            {
                "experiment": experiment,
                "stage": stage,
                "num_scenes": len(group),
                **values,
            }
        )

    summary_csv = args.input_root / "table_s5_macro_all_stages.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["experiment", "stage", "num_scenes"] + FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = args.input_root / "table_s5_macro_all_stages.json"
    with summary_json.open("w", encoding="utf-8") as file:
        json.dump(summary_rows, file, indent=2, allow_nan=True)

    paper_reference = args.input_root / "paper_reference_table_s5.csv"
    with paper_reference.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["source", "num_scenes"] + FIELDS)
        writer.writeheader()
        writer.writerow({"source": "Understanding While Exploring (Ours)", "num_scenes": 5, **PAPER_REFERENCE_PERCENT})

    comparison_rows = []
    for row in summary_rows:
        result = {
            "experiment": row["experiment"],
            "stage": row["stage"],
            "num_scenes": row["num_scenes"],
        }
        for field in FIELDS:
            measured_percent = 100.0 * row[field]
            result[field] = measured_percent
            result[f"delta_{field}_pp"] = measured_percent - PAPER_REFERENCE_PERCENT[field]
        comparison_rows.append(result)

    comparison = args.input_root / "comparison_with_paper_table_s5.csv"
    comparison_fields = ["experiment", "stage", "num_scenes"]
    for field in FIELDS:
        comparison_fields.extend([field, f"delta_{field}_pp"])
    with comparison.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(comparison_rows)

    markdown = args.input_root / "table_s5_macro_all_stages.md"
    with markdown.open("w", encoding="utf-8") as file:
        file.write("# MP3D Semantic Segmentation (Table S.5 format)\n\n")
        file.write("| Experiment | Stage | Scenes | Avg. | Ceiling | Appliances | Sink | Plant | Counter | Table | MPCAT40 |\n")
        file.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            file.write(
                "| {experiment} | {stage} | {num_scenes} | {avg} | {ceiling} | "
                "{appliances} | {sink} | {plant} | {counter} | {table} | {mpcat40} |\n".format(
                    experiment=row["experiment"],
                    stage=row["stage"],
                    num_scenes=row["num_scenes"],
                    **{field: format_percent(row[field]) for field in FIELDS},
                )
            )

    print(per_scene_path)
    print(summary_csv)
    print(summary_json)
    print(markdown)
    print(paper_reference)
    print(comparison)


if __name__ == "__main__":
    main()
