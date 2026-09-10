#!/usr/bin/env python3
"""Extract render and semantic evaluation metrics into JSON summaries.

Examples:
    # Original single-run style, kept for compatibility.
    python tools/helper/PSNR_mIoU_extract.py run_0_v6

    # Extract the three experiment groups used in the current report.
    python tools/helper/PSNR_mIoU_extract.py --all-presets

    # Extract one configured group.
    python tools/helper/PSNR_mIoU_extract.py --preset activesgm_sgsslam_0609
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union


DATASET = "Replica"
METHOD = "SemanticHeat"
TRACKER = "splatam"
EVAL_DIR = "eval_exploration_stage_1"
DECIMAL_PLACES = 4
SCENES = [
    "office0",
    "office1",
    "office2",
    "office3",
    "office4",
    "room0",
    "room1",
    "room2",
]
DEFAULT_METRIC_ORDER = [
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
]


@dataclass(frozen=True)
class ExtractConfig:
    name: str
    dataset: str
    method: str
    runversion: str
    tracker: str
    eval_dir: str
    results_root: Path
    eval_dir_template: str
    output_path: Path


PRESETS: Dict[str, ExtractConfig] = {
    # "activesgm": ExtractConfig(
    #     name="activesgm",
    #     dataset="Replica",
    #     method="Activesem",
    #     runversion="run_0",
    #     tracker="splatam",
    #     eval_dir="eval_exploration_stage_1",
    #     results_root=Path("/media/phudh/HDD/ActiveSGM_result/results/Replica"),
    #     eval_dir_template="{scene}/Activesem/run_0/splatam/eval_exploration_stage_1",
    #     output_path=Path("/media/phudh/HDD/ActiveSGM_result/results/Replica/Activesem_eval_exploration_stage_1_metrics.json"),
    # ),
        "activesgm": ExtractConfig(
        name="activesgm",
        dataset="Replica",
        method="Activesem",
        runversion="run_0",
        tracker="splatam",
        eval_dir="eval_final",
        results_root=Path("/media/phudh/HDD/ActiveSGM_result/results/Replica"),
        eval_dir_template="{scene}/Activesem/run_0/splatam/eval_final",
        output_path=Path("/media/phudh/HDD/ActiveSGM_result/results/Replica/Activesem_eval_final_metrics.json"),
    ),
    "activemapping_replica_52keyframes": ExtractConfig(
        name="activemapping_replica_52keyframes",
        dataset="Replica",
        method="SemanticHeat",
        runversion="run_0",
        tracker="splatam",
        eval_dir="eval_exploration_stage_1",
        results_root=Path("/media/phudh/HDD/ActiveMapping_result_local_global/Replica_52keyframes/Replica"),
        eval_dir_template="{scene}/SemanticHeat/run_0/splatam/eval_exploration_stage_1",
        output_path=Path(
            "/media/phudh/HDD/ActiveMapping_result_local_global/Replica_52keyframes/Replica/semanticheat_run_0_eval_final_metrics.json"
        ),
    ),
    "activegamer_replica": ExtractConfig(
        name="activegamer_replica",
        dataset="Replica",
        method="ActiveGAMER",
        runversion="run_0",
        tracker="splatam",
        eval_dir="eval_final",
        results_root=Path("/media/phudh/X9_Pro/undergraduate/ActiveGAMER/results/Replica"),
        eval_dir_template="{scene}/ActiveGAMER/run_0/splatam/eval_final",
        output_path=Path("/media/phudh/X9_Pro/undergraduate/ActiveGAMER/results/Replica/activegamer_run_0_eval_final_metrics.json"),
    ),
}


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
        description="Extract PSNR/mIoU and related eval metrics for Replica scenes."
    )
    parser.add_argument(
        "runversion",
        nargs="?",
        help="Legacy run version name, e.g. run_0_v6. Ignored when --preset/--all-presets is used.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        action="append",
        default=[],
        help="Named extraction preset. Can be specified multiple times.",
    )
    parser.add_argument(
        "--all-presets",
        action="store_true",
        help="Run all configured presets for the current experiment groups.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Custom results root for a one-off extraction.",
    )
    parser.add_argument(
        "--eval-dir-template",
        default=None,
        help="Custom eval dir template relative to results root, e.g. '{scene}/Method/run/splatam/eval_final'.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Custom output JSON path.")
    parser.add_argument("--method", default=None)
    parser.add_argument("--tracker", default=TRACKER)
    parser.add_argument("--eval-dir", default="eval_final")
    return parser.parse_args()


def parse_metric_file(file_path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not file_path.is_file():
        return metrics
    with file_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue

            key, value = line.split(":", 1)
            try:
                metrics[key.strip()] = float(value.strip())
            except ValueError:
                continue

    return metrics


def ordered_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    ordered: Dict[str, float] = {}
    for key in DEFAULT_METRIC_ORDER:
        if key in metrics:
            ordered[key] = metrics[key]
    for key in sorted(metrics):
        if key not in ordered:
            ordered[key] = metrics[key]
    return ordered


def collect_scene_result(config: ExtractConfig, scene: str) -> Union[str, Dict[str, float]]:
    eval_dir = config.results_root / config.eval_dir_template.format(scene=scene)
    render_metrics = parse_metric_file(eval_dir / "render_result.txt")
    semantic_metrics = parse_metric_file(eval_dir / "semantic_result.txt")

    if not render_metrics and not semantic_metrics:
        return "No version"

    merged = {}
    merged.update(render_metrics)
    merged.update(semantic_metrics)
    return ordered_metrics(merged)


def average_results(results: Dict[str, Union[str, Dict[str, float]]]) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for scene_result in results.values():
        if not isinstance(scene_result, dict):
            continue
        for key, value in scene_result.items():
            sums[key] = sums.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
    return {
        key: round(sums[key] / counts[key], DECIMAL_PLACES)
        for key in DEFAULT_METRIC_ORDER + sorted(set(sums.keys()) - set(DEFAULT_METRIC_ORDER))
        if key in sums and counts[key] > 0
    }


def write_output(config: ExtractConfig) -> Path:
    results: Dict[str, Union[str, Dict[str, float]]] = {}
    missing: List[str] = []
    for scene in SCENES:
        scene_result = collect_scene_result(config, scene)
        results[scene] = scene_result
        if isinstance(scene_result, str):
            missing.append(scene)

    output = {
        "dataset": config.dataset,
        "method": config.method,
        "runversion": config.runversion,
        "tracker": config.tracker,
        "eval_dir": config.eval_dir,
        "results_root": str(config.results_root),
        "eval_dir_template": config.eval_dir_template,
        "results": results,
        "average": average_results(results),
        "missing_scenes": missing,
    }

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, cls=FixedPrecisionFloatEncoder)

    print(f"Saved JSON to: {config.output_path}")
    return config.output_path


def legacy_config(runversion: str) -> ExtractConfig:
    script_path = Path(__file__).resolve()
    results_root = Path("/media/phudh/HDD/ActiveMapping_result_local_random")
    return ExtractConfig(
        name=f"legacy_{runversion}",
        dataset=DATASET,
        method=METHOD,
        runversion=runversion,
        tracker=TRACKER,
        eval_dir=EVAL_DIR,
        results_root=results_root,
        eval_dir_template=f"{{scene}}/{METHOD}/{runversion}/{TRACKER}/{EVAL_DIR}",
        output_path=script_path.with_name(f"{runversion}_result.json"),
    )


def custom_config(args: argparse.Namespace) -> Optional[ExtractConfig]:
    if args.results_root is None and args.eval_dir_template is None and args.output is None:
        return None
    if args.results_root is None or args.eval_dir_template is None or args.output is None:
        raise SystemExit("--results-root, --eval-dir-template, and --output must be provided together.")
    method = args.method or "custom"
    runversion = args.runversion or "custom"
    return ExtractConfig(
        name="custom",
        dataset=DATASET,
        method=method,
        runversion=runversion,
        tracker=args.tracker,
        eval_dir=args.eval_dir,
        results_root=args.results_root,
        eval_dir_template=args.eval_dir_template,
        output_path=args.output,
    )


def selected_configs(args: argparse.Namespace) -> Iterable[ExtractConfig]:
    if args.all_presets:
        return [PRESETS[name] for name in sorted(PRESETS.keys())]
    if args.preset:
        return [PRESETS[name] for name in args.preset]
    custom = custom_config(args)
    if custom is not None:
        return [custom]
    if args.runversion:
        return [legacy_config(args.runversion)]
    raise SystemExit("Provide --all-presets, --preset, custom extraction args, or a legacy runversion.")


def main() -> None:
    args = parse_args()
    for config in selected_configs(args):
        write_output(config)


if __name__ == "__main__":
    main()
