import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


METRICS = [
    ("miou", "mIoU"),
    ("macc", "mAcc"),
    ("semantic_fscore", "Semantic F-score"),
]


def parse_float(value: str) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def step_from_tag(tag: str) -> Optional[int]:
    match = re.fullmatch(r"step_(\d+)", tag)
    if not match:
        return None
    return int(match.group(1))


def read_rows(summary_csv: Path) -> List[Dict[str, object]]:
    rows = []
    with summary_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item: Dict[str, object] = dict(row)
            for key, _ in METRICS:
                item[key] = parse_float(row.get(key, ""))
            item["step"] = step_from_tag(str(row.get("tag", "")))
            rows.append(item)
    return rows


def finite_values(rows: List[Dict[str, object]], key: str) -> List[float]:
    vals = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, float) and math.isfinite(value):
            vals.append(value)
    return vals


def metric_ylim(rows: List[Dict[str, object]], key: str) -> tuple:
    values = finite_values(rows, key)
    if not values:
        return (0.0, 1.0)
    upper = max(values)
    if upper <= 1.05:
        return (0.0, 1.0)
    return (0.0, upper * 1.08)


def plot_final_bars(rows: List[Dict[str, object]], output_path: Path, title: str) -> bool:
    final_rows = [row for row in rows if row.get("tag") == "final"]
    if not final_rows:
        return False
    final_rows.sort(key=lambda row: str(row.get("scene", "")))
    scenes = [str(row.get("scene", "")) for row in final_rows]
    x = list(range(len(scenes)))
    width = 0.24

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(scenes)), 4.8))
    colors = ["#2b6cb0", "#2f855a", "#b7791f"]
    for idx, (key, label) in enumerate(METRICS):
        vals = [float(row.get(key, float("nan"))) for row in final_rows]
        offsets = [v + (idx - 1) * width for v in x]
        ax.bar(offsets, vals, width=width, label=label, color=colors[idx])

    all_values = []
    for key, _ in METRICS:
        all_values.extend(finite_values(final_rows, key))
    ax.set_ylim(0.0, 1.0 if all_values and max(all_values) <= 1.05 else max(all_values, default=1.0) * 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, rotation=25, ha="right")
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def plot_step_curves(rows: List[Dict[str, object]], output_path: Path, title: str) -> bool:
    step_rows = [row for row in rows if row.get("step") is not None]
    if not step_rows:
        return False

    scenes = sorted({str(row.get("scene", "")) for row in step_rows})
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5.2 * len(METRICS), 4.6), squeeze=False)
    markers = ["o", "s", "^", "D", "v", "x", "P", "*"]

    for col, (key, label) in enumerate(METRICS):
        ax = axes[0][col]
        for idx, scene in enumerate(scenes):
            scene_rows = [row for row in step_rows if row.get("scene") == scene]
            scene_rows.sort(key=lambda row: int(row["step"]))
            xs = [int(row["step"]) for row in scene_rows]
            ys = [float(row.get(key, float("nan"))) for row in scene_rows]
            ax.plot(xs, ys, marker=markers[idx % len(markers)], linewidth=1.5, markersize=4, label=scene)
        ax.set_title(label)
        ax.set_xlabel("step")
        ax.set_ylabel("score")
        ax.set_ylim(*metric_ylim(step_rows, key))
        ax.grid(True, alpha=0.25)

    axes[0][-1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.92, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 3D semantic evaluation summary CSV.")
    parser.add_argument("--summary_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--title", default="3D Semantic Evaluation")
    args = parser.parse_args()

    rows = read_rows(args.summary_csv)
    if not rows:
        raise SystemExit(f"No rows found in {args.summary_csv}")

    output_dir = args.output_dir or (args.summary_csv.parent / "eval_3d_semantic_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    wrote = []
    final_path = output_dir / "final_metrics.png"
    if plot_final_bars(rows, final_path, args.title + " - final"):
        wrote.append(final_path)

    step_path = output_dir / "step_curves.png"
    if plot_step_curves(rows, step_path, args.title + " - steps"):
        wrote.append(step_path)

    if not wrote:
        raise SystemExit("No final or step rows were available to plot")
    for path in wrote:
        print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
