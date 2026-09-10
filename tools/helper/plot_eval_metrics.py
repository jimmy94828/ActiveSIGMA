#!/usr/bin/env python3
"""Plot per-step evaluation metrics for a run directory.

Usage:
    python tools/helper/plot_eval_metrics.py \
        --run_dir /media/phudh/X9_Pro/undergraduate/ActiveMapping/results/Replica/office2/SemanticHeat/run_0_keyframe

Compare two runs in one figure:
    python tools/helper/plot_eval_metrics.py \
        --run_dir /media/phudh/X9_Pro/undergraduate/ActiveMapping/results/Replica/office2/SemanticHeat/run_0_keyframe \
        --compare_dir /media/phudh/X9_Pro/undergraduate/ActiveSGM/results/Replica/office2/ActiveSem/run_1_origin \
        --run_label ActiveMapping \
        --compare_label ActiveSGM

Outputs:
    <run_dir>/splatam/eval_plots/render_metrics.png
    <run_dir>/splatam/eval_plots/semantic_metrics.png
    <run_dir>/splatam/eval_plots/render_metrics_compare.png
    <run_dir>/splatam/eval_plots/semantic_metrics_compare.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


EXCLUDE_METRICS = {
    "f1_p",
    "mAP_p",
    "miou_p",
    "miou_p_curr",
    "top1_p",
    "top3_p",
    "top5_p",
}


def parse_kv_file(path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not path.is_file():
        return metrics
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def normalize_key(key: str) -> str:
    return key.replace("(", "_").replace(")", "").replace(" ", "").replace("/", "_")


def collect_metrics(run_dir: Path) -> Tuple[List[int], Dict[str, List[float]], Dict[str, List[float]]]:
    splatam_dir = run_dir / "splatam"
    step_items: List[Tuple[int, Path]] = []

    for step_dir in splatam_dir.glob("eval_step_*"):
        match = re.search(r"eval_step_(\d+)", step_dir.name)
        if not match:
            continue
        step_items.append((int(match.group(1)), step_dir))

    step_items.sort(key=lambda item: item[0])
    steps = [step for step, _ in step_items]

    render_by_step: Dict[str, Dict[int, float]] = {}
    semantic_by_step: Dict[str, Dict[int, float]] = {}

    for step, step_dir in step_items:
        render_file = step_dir / "render_result.txt"
        semantic_file = step_dir / "semantic_result.txt"

        render_kv = parse_kv_file(render_file)
        semantic_kv = parse_kv_file(semantic_file)

        for k, v in render_kv.items():
            nk = normalize_key(k)
            render_by_step.setdefault(nk, {})[step] = v
        for k, v in semantic_kv.items():
            nk = normalize_key(k)
            semantic_by_step.setdefault(nk, {})[step] = v

    render_metrics: Dict[str, List[float]] = {
        key: [render_by_step[key].get(step, float("nan")) for step in steps]
        for key in render_by_step
    }
    semantic_metrics: Dict[str, List[float]] = {
        key: [semantic_by_step[key].get(step, float("nan")) for step in steps]
        for key in semantic_by_step
    }

    return steps, render_metrics, semantic_metrics


def plot_metrics(
    datasets: List[Tuple[List[int], Dict[str, List[float]], str]],
    title: str,
    out_path: Path,
) -> None:
    if not datasets:
        return

    exclude_lower = {key.lower() for key in EXCLUDE_METRICS}
    keys = sorted(
        {
            key
            for _, metrics, _ in datasets
            for key in metrics.keys()
            if key.lower() not in exclude_lower
        }
    )
    if not keys:
        return
    num = len(keys)
    cols = 3
    rows = (num + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.2 * rows), squeeze=False)
    markers = ["o", "s", "^", "D", "v", "x"]
    for idx, key in enumerate(keys):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        for d_idx, (steps, metrics, label) in enumerate(datasets):
            if key not in metrics:
                continue
            ax.plot(
                steps,
                metrics[key],
                marker=markers[d_idx % len(markers)],
                linewidth=1.5,
                label=label,
            )
        ax.set_title(key)
        ax.set_xlabel("step")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        if len(datasets) > 1:
            ax.legend(fontsize=8)

    # Hide unused axes
    for idx in range(num, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot evaluation metrics over steps.")
    parser.add_argument("--run_dir", required=True, help="Path to run directory.")
    parser.add_argument("--compare_dir", default=None, help="Optional run directory to overlay.")
    parser.add_argument("--run_label", default=None, help="Legend label for run_dir.")
    parser.add_argument("--compare_label", default=None, help="Legend label for compare_dir.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    steps, render_metrics, semantic_metrics = collect_metrics(run_dir)

    if not steps:
        raise SystemExit("No eval_step_* directories found under run_dir/splatam.")

    run_label = args.run_label or run_dir.name
    render_datasets = [(steps, render_metrics, run_label)]
    semantic_datasets = [(steps, semantic_metrics, run_label)]

    compare_suffix = ""
    if args.compare_dir:
        compare_dir = Path(args.compare_dir)
        c_steps, c_render_metrics, c_semantic_metrics = collect_metrics(compare_dir)
        if not c_steps:
            raise SystemExit("No eval_step_* directories found under compare_dir/splatam.")
        compare_label = args.compare_label or compare_dir.name
        render_datasets.append((c_steps, c_render_metrics, compare_label))
        semantic_datasets.append((c_steps, c_semantic_metrics, compare_label))
        compare_suffix = "_compare"

    plot_root = run_dir / "splatam" / "eval_plots"
    plot_metrics(
        render_datasets,
        "Render Metrics",
        plot_root / f"render_metrics{compare_suffix}.png",
    )
    plot_metrics(
        semantic_datasets,
        "Semantic Metrics",
        plot_root / f"semantic_metrics{compare_suffix}.png",
    )

    print(f"Saved plots under: {plot_root}")


if __name__ == "__main__":
    main()
