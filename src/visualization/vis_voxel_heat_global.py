from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import torch


@dataclass
class VoxelHeatGlobalRenderResult:
    image_bgr: np.ndarray


class VoxelHeatGlobalVisualizer:
    """Render global voxel heat projections."""

    def __init__(
        self,
        panel_size: int = 520,
        margin: int = 28,
        bg_color: Tuple[int, int, int] = (16, 18, 22),
        bbox_color: Tuple[int, int, int] = (245, 245, 245),
        text_color: Tuple[int, int, int] = (240, 240, 240),
        colormap: int = cv2.COLORMAP_TURBO,
        heat_min: float = 0.0,
        heat_max: float = 1.0,
    ) -> None:
        self.panel_size = int(panel_size)
        self.margin = int(margin)
        self.bg_color = tuple(int(v) for v in bg_color)
        self.bbox_color = tuple(int(v) for v in bbox_color)
        self.text_color = tuple(int(v) for v in text_color)
        self.colormap = int(colormap)
        self.heat_min = float(heat_min)
        self.heat_max = float(heat_max)

    @torch.no_grad()
    def render(self, slam) -> VoxelHeatGlobalRenderResult:
        points, heats, bbox_min, bbox_max, voxel_size = self._collect_hot_voxels(slam)
        image = self._compose_canvas(points, heats, bbox_min, bbox_max, voxel_size)
        return VoxelHeatGlobalRenderResult(image_bgr=image)

    def _collect_hot_voxels(self, slam):
        semantic_voxel_map = getattr(slam, "semantic_voxel_map", None)
        if semantic_voxel_map is None:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32), None, None, None

        bbox_min = semantic_voxel_map.bbox_min.detach().cpu().numpy().astype(np.float32)
        bbox_max = semantic_voxel_map.bbox_max.detach().cpu().numpy().astype(np.float32)
        active_mask = getattr(semantic_voxel_map, "alpha_v_active", None)
        if active_mask is not None:
            active_voxels = torch.where(active_mask)[0].tolist()
        else:
            active_voxels = sorted(getattr(semantic_voxel_map, "directional_active_voxels", []))

        pts: List[np.ndarray] = []
        heats: List[float] = []
        for voxel_idx in active_voxels:
            heat_scalar = float(semantic_voxel_map.compute_voxel_value(int(voxel_idx)))
            if (not np.isfinite(heat_scalar)) or heat_scalar <= 0.0:
                continue
            center = semantic_voxel_map.get_voxel_center(int(voxel_idx)).detach().cpu().numpy().astype(np.float32)
            pts.append(center)
            heats.append(heat_scalar)

        if not pts:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32), bbox_min, bbox_max, float(semantic_voxel_map.voxel_size)

        return (
            np.stack(pts, axis=0),
            np.asarray(heats, dtype=np.float32),
            bbox_min,
            bbox_max,
            float(semantic_voxel_map.voxel_size),
        )

    def _compose_canvas(self, points, heats, bbox_min, bbox_max, voxel_size):
        title_h = 56
        colorbar_h = 72
        total_w = self.margin * 4 + self.panel_size * 3
        total_h = self.margin * 2 + title_h + self.panel_size + colorbar_h
        canvas = np.full((total_h, total_w, 3), self.bg_color, dtype=np.uint8)

        cv2.putText(
            canvas,
            "Global Voxel Heat",
            (self.margin, self.margin + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            self.text_color,
            2,
            cv2.LINE_AA,
        )

        if bbox_min is None or bbox_max is None:
            return canvas

        panel_top = self.margin + title_h
        panel_lefts = [self.margin, self.margin * 2 + self.panel_size, self.margin * 3 + self.panel_size * 2]
        panel_specs = [
            ((0, 1), "XY Top"),
            ((0, 2), "XZ Front"),
            ((1, 2), "YZ Side"),
        ]

        for left, (axes, label) in zip(panel_lefts, panel_specs):
            panel = canvas[panel_top:panel_top + self.panel_size, left:left + self.panel_size]
            panel[:] = (24, 27, 33)
            self._draw_projection(panel, points, heats, bbox_min, bbox_max, voxel_size, axes, label)

        self._draw_colorbar(canvas, heats, panel_top + self.panel_size + 20)
        return canvas

    def _draw_projection(self, panel, points, heats, bbox_min, bbox_max, voxel_size, axes, label):
        axis_a, axis_b = axes
        bbox_pts = np.array([
            [bbox_min[axis_a], bbox_min[axis_b]],
            [bbox_min[axis_a], bbox_max[axis_b]],
            [bbox_max[axis_a], bbox_min[axis_b]],
            [bbox_max[axis_a], bbox_max[axis_b]],
        ], dtype=np.float32)

        if points.shape[0] > 0:
            point_2d = points[:, [axis_a, axis_b]]
            all_pts = np.concatenate([bbox_pts, point_2d], axis=0)
        else:
            point_2d = np.zeros((0, 2), dtype=np.float32)
            all_pts = bbox_pts

        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
        span = np.maximum(maxs - mins, voxel_size if voxel_size is not None else 0.1)
        pad = np.maximum(span * 0.08, 2.0 * (voxel_size if voxel_size is not None else 0.1))
        mins = mins - pad
        maxs = maxs + pad
        span = np.maximum(maxs - mins, 1e-5)

        def project(pt2):
            x = (pt2[0] - mins[0]) / span[0]
            y = (pt2[1] - mins[1]) / span[1]
            px = int(round(x * (self.panel_size - 1)))
            py = int(round((1.0 - y) * (self.panel_size - 1)))
            return px, py

        cv2.putText(panel, label, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, self.text_color, 2, cv2.LINE_AA)

        if point_2d.shape[0] == 0:
            return

        norm = self._normalize_heat_scores(heats)

        order = np.argsort(norm)
        dot_radius = max(2, int(round((voxel_size / max(span.max(), voxel_size)) * self.panel_size * 0.9))) if voxel_size is not None else 3

        lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), self.colormap)[:, 0, :]
        for idx in order:
            px, py = project(point_2d[idx])
            color = tuple(int(v) for v in lut[int(round(norm[idx] * 255.0))])
            cv2.circle(panel, (px, py), dot_radius + 1, (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(panel, (px, py), dot_radius, color, thickness=-1, lineType=cv2.LINE_AA)

    def _normalize_heat_scores(self, heats: np.ndarray) -> np.ndarray:
        heats = np.nan_to_num(heats, nan=self.heat_min, posinf=self.heat_max, neginf=self.heat_min)
        heat_span = self.heat_max - self.heat_min
        if (not np.isfinite(heat_span)) or heat_span <= 0.0:
            return np.zeros_like(heats, dtype=np.float32)
        return np.clip((heats - self.heat_min) / heat_span, 0.0, 1.0).astype(np.float32)

    def _draw_colorbar(self, canvas, heats, y_top):
        bar_left = self.margin
        bar_width = self.panel_size * 3 + self.margin * 2
        bar_height = 20
        gradient = np.linspace(0, 255, bar_width, dtype=np.uint8).reshape(1, -1)
        gradient = np.repeat(gradient, bar_height, axis=0)
        colorbar = cv2.applyColorMap(gradient, self.colormap)
        canvas[y_top:y_top + bar_height, bar_left:bar_left + bar_width] = colorbar
        cv2.rectangle(canvas, (bar_left, y_top), (bar_left + bar_width, y_top + bar_height), self.bbox_color, 1)

        min_txt = f"{self.heat_min:.3f}"
        max_txt = f"{self.heat_max:.3f}"

        cv2.putText(canvas, f"Heat min {min_txt}", (bar_left, y_top + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, self.text_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Heat max {max_txt}", (bar_left + bar_width - 150, y_top + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.58, self.text_color, 1, cv2.LINE_AA)
