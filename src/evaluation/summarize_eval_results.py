import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PREFERRED_COLUMNS = [
    "psnr",
    "ssim",
    "lpips",
    "l1(cm)",
    "rmse",
    "miou_g_curr",
    "miou_g",
    "top1_g",
    "top3_g",
    "top5_g",
    "mAP_g",
    "f1_g",
    "acc",
    "comp",
    "comp%",
    "MAD",
    "traj_len(m)",
    "render_num_frames",
    "num_frames",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize trajectory evaluation metrics for multiple runs."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Run label and result/eval directory. Can be repeated.",
    )
    parser.add_argument(
        "--eval_suffix",
        default="final",
        help="Suffix used by eval directories, e.g. final_traj -> splatam/eval_final_traj.",
    )
    parser.add_argument(
        "--out_csv",
        default=None,
        help="Optional CSV path to save the summary.",
    )
    return parser.parse_args()


def parse_run(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--run must be LABEL=PATH, got: {value}")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Empty run label in: {value}")
    return label, Path(path).expanduser()


def candidate_eval_dirs(path: Path, eval_suffix: str) -> Iterable[Path]:
    yield path
    yield path / f"eval_{eval_suffix}"
    yield path / "splatam" / f"eval_{eval_suffix}"
    yield path / "coslam" / f"eval_{eval_suffix}"


def find_eval_dir(path: Path, eval_suffix: str) -> Path:
    for candidate in candidate_eval_dirs(path, eval_suffix):
        if (candidate / "render_result.txt").exists() or (candidate / "semantic_result.txt").exists():
            return candidate
    return path / "splatam" / f"eval_{eval_suffix}"


def parse_metric_file(path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not path.exists():
        return metrics
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if ":" in line:
                key, value = line.split(":", 1)
            elif "," in line:
                key, value = line.split(",", 1)
            else:
                continue
            key = key.strip()
            value = value.strip()
            try:
                metrics[key] = float(value)
            except ValueError:
                continue
    return metrics


def collect_metrics(result_path: Path, eval_suffix: str) -> Dict[str, object]:
    eval_dir = find_eval_dir(result_path, eval_suffix)
    metrics: Dict[str, object] = {
        "path": str(result_path),
        "eval_dir": str(eval_dir),
    }
    metrics.update(parse_metric_file(result_path / "eval_result.txt"))
    metrics.update(parse_metric_file(eval_dir / "render_result.txt"))
    metrics.update(parse_metric_file(eval_dir / "semantic_result.txt"))

    eval_3d = result_path / "eval_3d" / eval_suffix / "eval_3d_result.txt"
    if eval_3d.exists():
        metrics.update(parse_metric_file(eval_3d))
    return metrics


def format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_table(rows: List[Dict[str, object]], columns: List[str]) -> None:
    widths = {
        col: max(len(col), *(len(format_value(row.get(col))) for row in rows))
        for col in columns
    }
    print(" | ".join(col.ljust(widths[col]) for col in columns))
    print(" | ".join("-" * widths[col] for col in columns))
    for row in rows:
        print(" | ".join(format_value(row.get(col)).ljust(widths[col]) for col in columns))


def write_csv(path: Path, rows: List[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def main() -> None:
    args = parse_args()
    rows = []
    for run in args.run:
        label, path = parse_run(run)
        row = {"run": label}
        row.update(collect_metrics(path, args.eval_suffix))
        rows.append(row)

    discovered = sorted({key for row in rows for key in row.keys()})
    metric_columns = [col for col in PREFERRED_COLUMNS if col in discovered]
    metric_columns.extend(
        col for col in discovered
        if col not in metric_columns and col not in {"run", "path", "eval_dir"}
    )
    columns = ["run", *metric_columns, "eval_dir"]
    print_table(rows, columns)

    if args.out_csv:
        write_csv(Path(args.out_csv), rows, columns)


if __name__ == "__main__":
    main()
