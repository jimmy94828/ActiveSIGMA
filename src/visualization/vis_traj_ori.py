import argparse
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def extract_translation(pose: np.ndarray) -> np.ndarray:
    """
    支援常見 pose 格式：
    - (4, 4): 平移在 pose[:3, 3]
    - (3, 4): 平移在 pose[:3, 3]
    - (3,), (>=3,): 前三個值視為 xyz
    """
    pose = np.asarray(pose)

    if pose.shape == (4, 4):
        return pose[:3, 3]
    if pose.shape == (3, 4):
        return pose[:3, 3]
    if pose.ndim == 1 and pose.shape[0] >= 3:
        return pose[:3]

    raise ValueError(f"Unsupported pose shape: {pose.shape}")


def load_positions(pose_dir: str) -> np.ndarray:
    pose_dir = Path(pose_dir)
    if not pose_dir.exists() or not pose_dir.is_dir():
        raise FileNotFoundError(f"Pose directory not found: {pose_dir}")

    npy_files: List[Path] = sorted(pose_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in: {pose_dir}")

    positions = []
    for f in npy_files:
        pose = np.load(str(f), allow_pickle=False)
        t = extract_translation(pose)
        positions.append(t)

    return np.stack(positions, axis=0)


def visualize_trajectory(positions: np.ndarray, out_path: str, title: str = "Trajectory") -> None:
    xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(xs, ys, zs, linewidth=2, label="trajectory")
    ax.scatter(xs[0], ys[0], zs[0], s=60, marker="o", label="start")
    ax.scatter(xs[-1], ys[-1], zs[-1], s=60, marker="^", label="end")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    # 讓座標軸比例比較一致
    max_range = np.array([xs.max() - xs.min(), ys.max() - ys.min(), zs.max() - zs.min()]).max()
    mid_x = (xs.max() + xs.min()) * 0.5
    mid_y = (ys.max() + ys.min()) * 0.5
    mid_z = (zs.max() + zs.min()) * 0.5
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200)
    plt.close(fig)

"""
python src/visualization/vis_traj.py \
	--pose_dir results/Replica/office2/SemanticHeat/run_0_v2/visualization/pose \
	--out results/Replica/office2/SemanticHeat/run_0_v2/visualization/trajectory.png \
	--title "Trajectory"
"""

def main():
    parser = argparse.ArgumentParser(description="Visualize trajectory from a folder of pose .npy files")
    parser.add_argument("--pose_dir", type=str, required=True, help="資料夾路徑（內含多個 .npy pose 檔）")
    parser.add_argument("--out", type=str, default="trajectory.png", help="輸出圖片路徑")
    parser.add_argument("--title", type=str, default="Trajectory", help="圖標題")
    args = parser.parse_args()

    positions = load_positions(args.pose_dir)
    visualize_trajectory(positions, args.out, args.title)
    print(f"[OK] Saved trajectory image to: {args.out}")


if __name__ == "__main__":
    main()