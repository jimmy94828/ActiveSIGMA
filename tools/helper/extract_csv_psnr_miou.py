#!/usr/bin/env python3
"""Extract PSNR and mIoU values from evaluation comparison CSV files."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_PATTERN = "results/eval_compare_*.csv"
DECIMAL_PLACES = 4


class FixedPrecisionFloatEncoder(json.JSONEncoder):
    """JSON encoder that keeps fixed decimal places for float values."""

    def iterencode(self, obj: Any, _one_shot: bool = False):
        markers = {} if self.check_circular else None
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
        description="Extract PSNR and mIoU values from eval_compare CSV files."
    )
    parser.add_argument(
        "csv_paths",
        nargs="*",
        help=f"CSV paths or glob patterns. Defaults to {DEFAULT_PATTERN}.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON path. Defaults to tools/helper/<runversion>_csv_PSNR_mIoU.json.",
    )
    parser.add_argument(
        "--method-column",
        default="run",
        help="Column containing method names. Defaults to run.",
    )
    parser.add_argument(
        "--scene-column",
        default=None,
        help="Optional column containing scene names. If omitted, scene is parsed from filename.",
    )
    parser.add_argument(
        "--psnr-column",
        default="psnr",
        help="Column containing PSNR values. Defaults to psnr.",
    )
    parser.add_argument(
        "--miou-column",
        default="miou_g_curr",
        help="Column containing mIoU values. Defaults to miou_g_curr.",
    )
    return parser.parse_args()


def expand_paths(patterns: Iterable[str], project_root: Path) -> List[Path]:
    expanded: List[Path] = []
    seen = set()

    for pattern in patterns:
        pattern_path = Path(pattern)
        search_pattern = str(pattern_path if pattern_path.is_absolute() else project_root / pattern)
        matches = [Path(path) for path in glob.glob(search_pattern)]
        if not matches and any(char in pattern for char in "*?[]"):
            continue
        if not matches:
            matches = [pattern_path if pattern_path.is_absolute() else project_root / pattern]

        for path in matches:
            resolved = path.resolve()
            if resolved not in seen:
                expanded.append(resolved)
                seen.add(resolved)

    return sorted(expanded)


def parse_csv_metadata(path: Path) -> Tuple[str | None, str, str | None]:
    stem = path.stem
    prefix = "eval_compare_"
    if not stem.startswith(prefix):
        return None, stem, None

    rest = stem[len(prefix) :]
    if "_" not in rest:
        return None, rest, None

    dataset, scene_and_run = rest.split("_", 1)
    run_marker = "_run_"
    if run_marker not in scene_and_run:
        return dataset, scene_and_run, None

    scene, run_suffix = scene_and_run.split(run_marker, 1)
    return dataset, scene, f"run_{run_suffix}"


def require_columns(path: Path, row: Dict[str, str], columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in row]
    if missing:
        raise KeyError(f"{path} missing columns: {', '.join(missing)}")


def parse_metric(value: str, path: Path, column: str) -> float:
    try:
        return round(float(value), DECIMAL_PLACES)
    except ValueError as exc:
        raise ValueError(f"{path} has non-numeric {column}: {value!r}") from exc


def default_output_path(project_root: Path, runversions: Iterable[str | None]) -> Path:
    valid_runversions = sorted({runversion for runversion in runversions if runversion})
    if len(valid_runversions) == 1:
        filename = f"{valid_runversions[0]}_csv_PSNR_mIoU.json"
    else:
        filename = "csv_PSNR_mIoU.json"
    return project_root / "tools" / "helper" / filename


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    input_patterns = args.csv_paths or [DEFAULT_PATTERN]
    csv_paths = expand_paths(input_patterns, project_root)
    csv_paths = [path for path in csv_paths if path.is_file()]

    if not csv_paths:
        raise FileNotFoundError(f"No CSV files matched: {', '.join(input_patterns)}")

    grouped_results: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    datasets = set()
    runversions = set()

    for csv_path in csv_paths:
        dataset, filename_scene, runversion = parse_csv_metadata(csv_path)
        if dataset:
            datasets.add(dataset)
        if runversion:
            runversions.add(runversion)

        with csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                required_columns = [
                    args.method_column,
                    args.psnr_column,
                    args.miou_column,
                ]
                if args.scene_column:
                    required_columns.append(args.scene_column)
                require_columns(csv_path, row, required_columns)

                method = row[args.method_column].strip()
                scene = (
                    row[args.scene_column].strip()
                    if args.scene_column
                    else filename_scene
                )
                grouped_results[method][scene] = {
                    "psnr": parse_metric(row[args.psnr_column], csv_path, args.psnr_column),
                    "miou_g_curr": parse_metric(
                        row[args.miou_column], csv_path, args.miou_column
                    ),
                }

    output: Dict[str, Any] = {
        "dataset": next(iter(datasets)) if len(datasets) == 1 else sorted(datasets),
        "runversion": next(iter(runversions)) if len(runversions) == 1 else sorted(runversions),
        "source_csv": [
            str(path.relative_to(project_root) if path.is_relative_to(project_root) else path)
            for path in csv_paths
        ],
        "metrics": ["psnr", "miou_g_curr"],
        "methods": {},
    }

    for method in sorted(grouped_results):
        scene_results = {
            scene: grouped_results[method][scene]
            for scene in sorted(grouped_results[method])
        }
        psnr_values = [metrics["psnr"] for metrics in scene_results.values()]
        miou_values = [metrics["miou_g_curr"] for metrics in scene_results.values()]
        scene_results["average"] = {
            "psnr": round(sum(psnr_values) / len(psnr_values), DECIMAL_PLACES),
            "miou_g_curr": round(sum(miou_values) / len(miou_values), DECIMAL_PLACES),
        }
        output["methods"][method] = {"results": scene_results}

    output_path = (
        Path(args.output)
        if args.output
        else default_output_path(project_root, runversions)
    )
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, cls=FixedPrecisionFloatEncoder)
        file.write("\n")

    print(f"Saved JSON to: {output_path}")


if __name__ == "__main__":
    main()
