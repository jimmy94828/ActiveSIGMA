from dataclasses import dataclass
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch


C0 = 0.28209479177387814


@dataclass
class BevDiagnosticRenderResult:
    image_bgr: np.ndarray


class BevDiagnosticVisualizer:
    """Render a per-iteration BEV diagnostic image for ActiveMapping."""

    def __init__(
        self,
        canvas_size: int = 900,
        margin: int = 28,
        bg_color: Tuple[int, int, int] = (255, 255, 255),
        map_bg_color: Tuple[int, int, int] = (255, 255, 255),
        bbox_color: Tuple[int, int, int] = (210, 210, 210),
        text_color: Tuple[int, int, int] = (42, 42, 42),
        jsd_top_percent: float = 0.10,
        show_candidate_pool: bool = True,
        bev_pad_ratio: float = 0.05,
        remove_front_percent: Optional[float] = 25.0,
        flip_view: bool = False,
        min_sigma_px: float = 0.75,
        max_sigma_px: float = 3.0,
        max_kernel_radius: int = 4,
    ) -> None:
        self.canvas_size = int(canvas_size)
        self.margin = int(margin)
        self.bg_color = tuple(int(v) for v in bg_color)
        self.map_bg_color = tuple(int(v) for v in map_bg_color)
        self.bbox_color = tuple(int(v) for v in bbox_color)
        self.text_color = tuple(int(v) for v in text_color)
        self.jsd_top_percent = float(np.clip(jsd_top_percent, 1e-4, 1.0))
        self.show_candidate_pool = bool(show_candidate_pool)
        self.bev_pad_ratio = max(float(bev_pad_ratio), 0.0)
        self.remove_front_percent = None if remove_front_percent is None else float(np.clip(remove_front_percent, 0.0, 99.999))
        self.flip_view = bool(flip_view)
        self.min_sigma_px = max(float(min_sigma_px), 1e-3)
        self.max_sigma_px = max(float(max_sigma_px), self.min_sigma_px)
        self.max_kernel_radius = max(1, int(max_kernel_radius))

    @torch.no_grad()
    def render(self, slam, planner, current_pose: Optional[torch.Tensor]) -> BevDiagnosticRenderResult:
        canvas = np.full((self.canvas_size, self.canvas_size, 3), self.bg_color, dtype=np.uint8)
        slam2sim = self._resolve_slam2sim_transform(slam, planner)
        points, colors, scales, opacities = self._collect_live_rgb_point_cloud(slam)
        if points.shape[0] == 0:
            cv2.putText(
                canvas,
                "BEV Diagnostic: current point cloud missing",
                (self.margin, self.margin + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                self.text_color,
                2,
                cv2.LINE_AA,
            )
            return BevDiagnosticRenderResult(image_bgr=canvas)

        points = self._transform_points(points, slam2sim)
        points, colors, scales, opacities, front_cutoff = self._filter_render_points(points, colors, scales, opacities)
        if points.shape[0] == 0:
            cv2.putText(
                canvas,
                "BEV Diagnostic: no visible points after filtering",
                (self.margin, self.margin + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                self.text_color,
                2,
                cv2.LINE_AA,
            )
            return BevDiagnosticRenderResult(image_bgr=canvas)

        projector = _XYProjector.from_points(
            points_xy=points[:, :2],
            canvas_size=self.canvas_size,
            margin=self.margin,
            pad_ratio=self.bev_pad_ratio,
        )

        map_rect = projector.map_rect
        canvas[map_rect[1]:map_rect[3] + 1, map_rect[0]:map_rect[2] + 1] = self.map_bg_color
        cv2.rectangle(canvas, (map_rect[0], map_rect[1]), (map_rect[2], map_rect[3]), self.bbox_color, 1)

        visible_count = self._draw_rgb_background(canvas, points, colors, scales, opacities, projector)

        heat_pts, heat_scores = self._collect_top_heat_voxels(slam)
        if heat_pts.shape[0] > 0:
            heat_pts = self._transform_points(heat_pts, slam2sim)
            self._draw_heat_voxels(canvas, heat_pts, heat_scores, projector)

        if self.show_candidate_pool:
            cand_pts = self._collect_candidate_pool(planner)
            if cand_pts.shape[0] > 0:
                cand_pts = self._transform_points(cand_pts, slam2sim)
                self._draw_candidate_pool(canvas, cand_pts, projector)

        if current_pose is not None:
            self._draw_pose_triangle(
                canvas,
                pose=self._transform_pose(current_pose, slam2sim),
                projector=projector,
                color=(205, 110, 24),
                fill_color=(255, 185, 110),
                arrow_color=(255, 120, 28),
                label=None,
            )

        goal_pose = getattr(planner, "goal_pose", None)
        if goal_pose is not None:
            self._draw_goal_marker(canvas, self._transform_pose(goal_pose, slam2sim), projector)

        self._draw_annotations(
            canvas,
            planner=planner,
            heat_count=int(heat_pts.shape[0]),
            visible_count=visible_count,
            front_cutoff=front_cutoff,
        )
        return BevDiagnosticRenderResult(image_bgr=canvas)

    def _draw_rgb_background(
        self,
        canvas: np.ndarray,
        points: np.ndarray,
        colors: np.ndarray,
        scales: Optional[np.ndarray],
        opacities: Optional[np.ndarray],
        projector: "_XYProjector",
    ) -> int:
        if points.shape[0] == 0:
            return 0

        px, py, linear = projector.project_points_float(points[:, :2])
        chosen = _top_surface_indices(linear, points[:, 2], flip=self.flip_view)
        if chosen.shape[0] == 0:
            return 0

        sigma_px = _visible_scales_px(
            scales=scales,
            chosen_idx=chosen,
            res=projector.metric_res,
            min_sigma_px=self.min_sigma_px,
            max_sigma_px=self.max_sigma_px,
        )
        alpha = _visible_opacity(opacities, chosen)
        bgr_features = colors[chosen][:, ::-1].astype(np.float32) / 255.0
        bev_img = _gaussian_splat_features(
            feature_values=bgr_features,
            px=px[chosen],
            py=py[chosen],
            sigma_px=sigma_px,
            alpha=alpha,
            width=projector.draw_size,
            height=projector.draw_size,
            max_kernel_radius=self.max_kernel_radius,
            background_value=np.asarray(self.map_bg_color, dtype=np.float32) / 255.0,
        )
        bev_img = np.clip(bev_img * 255.0, 0.0, 255.0).astype(np.uint8)

        map_rect = projector.map_rect
        canvas[map_rect[1]:map_rect[3] + 1, map_rect[0]:map_rect[2] + 1] = bev_img
        cv2.rectangle(canvas, (map_rect[0], map_rect[1]), (map_rect[2], map_rect[3]), self.bbox_color, 1)
        return int(chosen.shape[0])

    def _collect_live_rgb_point_cloud(
        self,
        slam,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        params = getattr(slam, "params", None)
        if params is None:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint8),
                None,
                None,
            )

        means = None
        for key in ("means3D", "means3d", "means", "mu", "centers"):
            if key in params:
                means = _to_numpy(params[key], dtype=np.float32)
                break
        if means is None or means.size == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint8),
                None,
                None,
            )
        means = means[:, :3]

        if all(k in params for k in ("f_dc_0", "f_dc_1", "f_dc_2")):
            f_dc = np.stack(
                [
                    _to_numpy(params["f_dc_0"], dtype=np.float32).reshape(-1),
                    _to_numpy(params["f_dc_1"], dtype=np.float32).reshape(-1),
                    _to_numpy(params["f_dc_2"], dtype=np.float32).reshape(-1),
                ],
                axis=1,
            )
            rgb = _decode_rgb_from_fdc(f_dc)
        elif "features_dc" in params:
            f_dc = _to_numpy(params["features_dc"], dtype=np.float32).reshape(means.shape[0], -1)
            if f_dc.shape[1] >= 3:
                rgb = _decode_rgb_from_fdc(f_dc)
            else:
                rgb = np.full((means.shape[0], 3), 188, dtype=np.uint8)
        elif "rgb_colors" in params:
            rgb = _decode_rgb_values(_to_numpy(params["rgb_colors"], dtype=np.float32))
        else:
            rgb = np.full((means.shape[0], 3), 188, dtype=np.uint8)

        scales = None
        if "log_scales" in params:
            log_scales = _to_numpy(params["log_scales"], dtype=np.float32)
            if log_scales.ndim == 1:
                log_scales = log_scales[:, None]
            scales = np.exp(log_scales)

        opacities = None
        if "logit_opacities" in params:
            opacities = 1.0 / (1.0 + np.exp(-_to_numpy(params["logit_opacities"], dtype=np.float32).reshape(-1)))

        keep = np.all(np.isfinite(means), axis=1) & np.all(np.isfinite(rgb.astype(np.float32)), axis=1)
        if scales is not None:
            keep &= np.all(np.isfinite(scales), axis=1)
        if opacities is not None:
            keep &= np.isfinite(opacities)

        means = means[keep]
        rgb = rgb[keep]
        if scales is not None:
            scales = scales[keep]
        if opacities is not None:
            opacities = opacities[keep]

        return means, rgb, scales, opacities

    def _filter_render_points(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        scales: Optional[np.ndarray],
        opacities: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        if points.shape[0] == 0:
            return points, colors, scales, opacities, None

        keep = np.ones(points.shape[0], dtype=bool)
        view_nearness = points[:, 2] if not self.flip_view else -points[:, 2]
        front_cutoff = None
        if self.remove_front_percent is not None:
            front_cutoff = float(np.percentile(view_nearness, 100.0 - self.remove_front_percent))
            keep &= view_nearness <= front_cutoff

        points = points[keep]
        colors = colors[keep]
        if scales is not None:
            scales = scales[keep]
        if opacities is not None:
            opacities = opacities[keep]
        return points, colors, scales, opacities, front_cutoff

    def _resolve_slam2sim_transform(self, slam, planner) -> np.ndarray:
        explr_map = getattr(slam, "explr_map", None)
        if explr_map is not None:
            slam2sim = getattr(explr_map, "slam2sim", None)
            if slam2sim is not None:
                arr = _to_numpy(slam2sim, dtype=np.float32)
                if arr.shape == (4, 4):
                    return arr

        sim2slam = getattr(planner, "sim2slam", None)
        if sim2slam is not None:
            arr = _to_numpy(sim2slam, dtype=np.float32)
            if arr.shape == (4, 4):
                return np.linalg.inv(arr).astype(np.float32)

        return np.eye(4, dtype=np.float32)

    def _transform_points(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        if points.shape[0] == 0:
            return points
        if transform.shape != (4, 4):
            return points
        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        points_h = np.concatenate([points.astype(np.float32), ones], axis=1)
        return (transform @ points_h.T).T[:, :3].astype(np.float32)

    def _transform_pose(self, pose, transform: np.ndarray):
        pose_np = _to_numpy(pose, dtype=np.float32)
        if pose_np.shape != (4, 4) or transform.shape != (4, 4):
            return pose
        return (transform @ pose_np).astype(np.float32)

    def _collect_top_heat_voxels(self, slam) -> Tuple[np.ndarray, np.ndarray]:
        semantic_voxel_map = getattr(slam, "semantic_voxel_map", None)
        if semantic_voxel_map is None:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        active_mask = getattr(semantic_voxel_map, "alpha_v_active", None)
        voxel_ids: List[int]
        if active_mask is not None:
            voxel_ids = torch.where(active_mask)[0].detach().cpu().tolist()
        else:
            voxel_ids = sorted(getattr(semantic_voxel_map, "directional_active_voxels", []))
        if len(voxel_ids) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        top_n = max(1, int(math.ceil(len(voxel_ids) * self.jsd_top_percent)))
        top_voxels = semantic_voxel_map.topk_hot_voxels(
            k=top_n,
            active_only=True,
            fallback_to_alpha_active=True,
        )
        if len(top_voxels) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        pts: List[np.ndarray] = []
        scores: List[float] = []
        for v_idx in top_voxels:
            score = float(semantic_voxel_map.compute_voxel_value(int(v_idx)))
            if not np.isfinite(score) or score <= 0.0:
                continue
            center = semantic_voxel_map.get_voxel_center(int(v_idx)).detach().cpu().numpy().astype(np.float32)
            pts.append(center)
            scores.append(score)

        if len(pts) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        pts_np = np.stack(pts, axis=0)
        scores_np = np.asarray(scores, dtype=np.float32)
        return pts_np, scores_np

    def _collect_candidate_pool(self, planner) -> np.ndarray:
        explore_pool = getattr(planner, "explore_pool", None)
        if not explore_pool:
            return np.zeros((0, 3), dtype=np.float32)

        pts: List[np.ndarray] = []
        for item in explore_pool.values():
            pose = item.get("pose")
            if pose is None:
                continue
            pose_np = _to_numpy(pose, dtype=np.float32)
            if pose_np.shape != (4, 4):
                continue
            pts.append(pose_np[:3, 3])

        if len(pts) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack(pts, axis=0)

    def _draw_heat_voxels(self, canvas: np.ndarray, points: np.ndarray, scores: np.ndarray, projector: "_XYProjector") -> None:
        px, py, inside = projector.project_points(points[:, :2])
        if not bool(np.any(inside)):
            return

        px = px[inside]
        py = py[inside]
        scores = scores[inside]
        if scores.shape[0] == 0:
            return

        lo = float(np.min(scores))
        hi = float(np.max(scores))
        span = max(hi - lo, 1e-8)
        order = np.argsort(scores)
        for idx in order:
            alpha = 0.20 + 0.65 * ((float(scores[idx]) - lo) / span)
            radius = 3 if alpha < 0.45 else 4
            overlay = canvas.copy()
            cv2.circle(overlay, (int(px[idx]), int(py[idx])), radius, (34, 139, 230), thickness=-1, lineType=cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0.0, dst=canvas)

    def _draw_candidate_pool(self, canvas: np.ndarray, points: np.ndarray, projector: "_XYProjector") -> None:
        px, py, inside = projector.project_points(points[:, :2])
        if not bool(np.any(inside)):
            return
        for x, y in zip(px[inside], py[inside]):
            cv2.circle(canvas, (int(x), int(y)), 2, (165, 165, 165), thickness=-1, lineType=cv2.LINE_AA)

    def _draw_pose_triangle(
        self,
        canvas: np.ndarray,
        pose: torch.Tensor,
        projector: "_XYProjector",
        color: Tuple[int, int, int],
        fill_color: Tuple[int, int, int],
        arrow_color: Tuple[int, int, int],
        label: Optional[str],
    ) -> None:
        pose_np = _to_numpy(pose, dtype=np.float32)
        if pose_np.shape != (4, 4):
            return
        center = pose_np[:3, 3]
        heading_xy = _pose_heading_xy(pose_np)
        px, py, inside = projector.project_points(center[:2][None, :])
        if not bool(inside[0]):
            return
        c = np.array([float(px[0]), float(py[0])], dtype=np.float32)

        # Match image y-down coordinates.
        heading_img = np.array([heading_xy[0], -heading_xy[1]], dtype=np.float32)
        if float(np.linalg.norm(heading_img)) < 1e-6:
            heading_img = np.array([1.0, 0.0], dtype=np.float32)
        heading_img = heading_img / (np.linalg.norm(heading_img) + 1e-8)
        right_img = np.array([heading_img[1], -heading_img[0]], dtype=np.float32)

        size = 10.0
        tip = c + heading_img * (size * 1.5)
        base_l = c - heading_img * size + right_img * (size * 0.85)
        base_r = c - heading_img * size - right_img * (size * 0.85)
        pts = np.round(np.stack([tip, base_l, base_r], axis=0)).astype(np.int32)

        cv2.fillConvexPoly(canvas, pts, fill_color, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        arrow_end = np.round(c + heading_img * (size * 3.2)).astype(np.int32)
        cv2.arrowedLine(
            canvas,
            tuple(np.round(c).astype(np.int32)),
            tuple(arrow_end),
            arrow_color,
            2,
            cv2.LINE_AA,
            tipLength=0.32,
        )

        if label:
            cv2.putText(
                canvas,
                label,
                (int(c[0]) + 8, int(c[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    def _draw_goal_marker(self, canvas: np.ndarray, pose: torch.Tensor, projector: "_XYProjector") -> None:
        pose_np = _to_numpy(pose, dtype=np.float32)
        if pose_np.shape != (4, 4):
            return
        center = pose_np[:3, 3]
        px, py, inside = projector.project_points(center[:2][None, :])
        if not bool(inside[0]):
            return
        pt = (int(px[0]), int(py[0]))
        cv2.drawMarker(canvas, pt, (34, 139, 34), markerType=cv2.MARKER_STAR, markerSize=18, thickness=2, line_type=cv2.LINE_AA)
        cv2.putText(
            canvas,
            "NBV",
            (pt[0] + 10, pt[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (34, 139, 34),
            1,
            cv2.LINE_AA,
        )

    def _draw_annotations(
        self,
        canvas: np.ndarray,
        planner,
        heat_count: int,
        visible_count: int,
        front_cutoff: Optional[float],
    ) -> None:
        lines = [
            "BEV Diagnostic",
            f"state={getattr(planner, 'state', 'na')}",
            f"planning_state={getattr(planner, 'planning_state', 'na')}",
            f"visible_surface={visible_count}",
            f"voxel_heat_top={heat_count}",
        ]
        if front_cutoff is not None:
            lines.append(f"front_cutoff={front_cutoff:.3f}")
        panel_h = 24 + 20 * len(lines)
        panel_w = 260
        x0, y0 = self.margin, self.margin
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (250, 250, 250), thickness=-1)
        cv2.addWeighted(overlay, 0.86, canvas, 0.14, 0.0, dst=canvas)
        cv2.rectangle(canvas, (x0, y0), (x0 + panel_w, y0 + panel_h), (180, 180, 180), thickness=1)
        for idx, line in enumerate(lines):
            cv2.putText(
                canvas,
                line,
                (x0 + 12, y0 + 22 + idx * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                self.text_color,
                1,
                cv2.LINE_AA,
            )


class _XYProjector:
    def __init__(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        canvas_size: int,
        margin: int,
    ) -> None:
        self.canvas_size = int(canvas_size)
        self.margin = int(margin)
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.ymin = float(ymin)
        self.ymax = float(ymax)
        self.draw_size = max(1, self.canvas_size - 2 * self.margin)
        self.metric_res = max(self.xmax - self.xmin, self.ymax - self.ymin, 1e-8) / max(self.draw_size - 1, 1)
        self.map_rect = (self.margin, self.margin, self.margin + self.draw_size - 1, self.margin + self.draw_size - 1)

    @classmethod
    def from_points(
        cls,
        points_xy: np.ndarray,
        canvas_size: int,
        margin: int,
        pad_ratio: float,
    ) -> "_XYProjector":
        points_xy = np.asarray(points_xy, dtype=np.float32)
        xmin = float(points_xy[:, 0].min())
        xmax = float(points_xy[:, 0].max())
        ymin = float(points_xy[:, 1].min())
        ymax = float(points_xy[:, 1].max())
        dx = xmax - xmin
        dy = ymax - ymin
        max_span = max(dx, dy, 1e-5)
        pad = max(max_span * float(pad_ratio), 1e-3)
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        half = 0.5 * max_span + pad
        return cls(
            xmin=cx - half,
            xmax=cx + half,
            ymin=cy - half,
            ymax=cy + half,
            canvas_size=canvas_size,
            margin=margin,
        )

    def project_points_float(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xy = np.asarray(xy, dtype=np.float32)
        inside = (
            (xy[:, 0] >= self.xmin)
            & (xy[:, 0] <= self.xmax)
            & (xy[:, 1] >= self.ymin)
            & (xy[:, 1] <= self.ymax)
        )
        nx = (xy[:, 0] - self.xmin) / max(self.xmax - self.xmin, 1e-8)
        ny = (xy[:, 1] - self.ymin) / max(self.ymax - self.ymin, 1e-8)
        px = np.clip(nx * (self.draw_size - 1), 0.0, float(self.draw_size - 1))
        py = np.clip(ny * (self.draw_size - 1), 0.0, float(self.draw_size - 1))
        linear = np.floor(py).astype(np.int64) * self.draw_size + np.floor(px).astype(np.int64)
        return px.astype(np.float32), py.astype(np.float32), linear

    def project_points(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        xy = np.asarray(xy, dtype=np.float32)
        inside = (
            (xy[:, 0] >= self.xmin)
            & (xy[:, 0] <= self.xmax)
            & (xy[:, 1] >= self.ymin)
            & (xy[:, 1] <= self.ymax)
        )
        nx = (xy[:, 0] - self.xmin) / max(self.xmax - self.xmin, 1e-8)
        ny = (xy[:, 1] - self.ymin) / max(self.ymax - self.ymin, 1e-8)
        px = np.round(nx * (self.draw_size - 1) + self.margin).astype(np.int32)
        py = np.round((1.0 - ny) * (self.draw_size - 1) + self.margin).astype(np.int32)
        px = np.clip(px, 0, self.canvas_size - 1)
        py = np.clip(py, 0, self.canvas_size - 1)
        return px, py, inside


def _top_surface_indices(linear: np.ndarray, depth: np.ndarray, flip: bool) -> np.ndarray:
    order = np.argsort(depth if flip else -depth)
    linear_sorted = linear[order]
    _, first = np.unique(linear_sorted, return_index=True)
    return order[first]


def _visible_scales_px(
    scales: Optional[np.ndarray],
    chosen_idx: np.ndarray,
    res: float,
    min_sigma_px: float,
    max_sigma_px: float,
) -> np.ndarray:
    if scales is None:
        return np.full(chosen_idx.shape[0], min_sigma_px, dtype=np.float32)

    chosen_scales = np.asarray(scales[chosen_idx], dtype=np.float32)
    if chosen_scales.ndim == 1:
        scale_xy = chosen_scales
    elif chosen_scales.shape[1] == 1:
        scale_xy = chosen_scales[:, 0]
    else:
        scale_xy = np.mean(chosen_scales[:, :2], axis=1)
    sigma = scale_xy / max(res, 1e-8)
    sigma = np.asarray(sigma, dtype=np.float32)
    return np.clip(sigma, min_sigma_px, max_sigma_px)


def _visible_opacity(opacities: Optional[np.ndarray], chosen_idx: np.ndarray) -> np.ndarray:
    if opacities is None:
        return np.full(chosen_idx.shape[0], 1.0, dtype=np.float32)
    alpha = np.asarray(opacities[chosen_idx], dtype=np.float32)
    return np.clip(alpha, 1e-3, 0.999)


def _gaussian_splat_features(
    feature_values: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    sigma_px: np.ndarray,
    alpha: np.ndarray,
    width: int,
    height: int,
    max_kernel_radius: int,
    background_value,
) -> np.ndarray:
    num_pixels = int(width) * int(height)
    channels = 1 if feature_values.ndim == 1 else feature_values.shape[1]
    values = feature_values.reshape(feature_values.shape[0], channels).astype(np.float32)

    numer = np.zeros((channels, num_pixels), dtype=np.float32)
    denom = np.zeros(num_pixels, dtype=np.float32)

    base_x = np.floor(px).astype(np.int32)
    base_y = np.floor(py).astype(np.int32)
    radius_px = np.clip(np.ceil(3.0 * sigma_px).astype(np.int32), 1, max_kernel_radius)
    inv_two_sigma_sq = 0.5 / np.maximum(sigma_px * sigma_px, 1e-8)

    for dy in range(-max_kernel_radius, max_kernel_radius + 1):
        gy = base_y + dy
        valid_y = (gy >= 0) & (gy < height) & (np.abs(dy) <= radius_px)
        if not np.any(valid_y):
            continue
        dy_center = (gy.astype(np.float32) + 0.5) - py
        for dx in range(-max_kernel_radius, max_kernel_radius + 1):
            gx = base_x + dx
            valid = valid_y & (gx >= 0) & (gx < width) & (np.abs(dx) <= radius_px)
            if not np.any(valid):
                continue

            delta_x = (gx[valid].astype(np.float32) + 0.5) - px[valid]
            delta_y = dy_center[valid]
            d2 = delta_x * delta_x + delta_y * delta_y
            weight = alpha[valid] * np.exp(-d2 * inv_two_sigma_sq[valid])
            target = gy[valid].astype(np.int64) * width + gx[valid].astype(np.int64)

            denom += np.bincount(target, weights=weight, minlength=num_pixels).astype(np.float32)
            for c in range(channels):
                numer[c] += np.bincount(target, weights=weight * values[valid, c], minlength=num_pixels).astype(np.float32)

    bg = np.asarray(background_value, dtype=np.float32)
    if bg.ndim == 0:
        bg = np.full((channels, 1), float(bg), dtype=np.float32)
    else:
        bg = bg.reshape(channels, 1)
    out = np.broadcast_to(bg, (channels, num_pixels)).copy()
    mask = denom > 1e-8
    out[:, mask] = numer[:, mask] / denom[mask]
    out = out.reshape(channels, height, width)
    out = np.transpose(out, (1, 2, 0))
    if channels == 1:
        out = out[:, :, 0]
    return np.flipud(out)


def _to_numpy(value, dtype=None) -> np.ndarray:
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _decode_rgb_values(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors)
    if colors.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if np.issubdtype(colors.dtype, np.floating):
        colors = np.clip(colors, 0.0, 1.0)
        colors = (colors * 255.0).astype(np.uint8)
    else:
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    return colors


def _decode_rgb_from_fdc(f_dc: np.ndarray) -> np.ndarray:
    f_dc = np.asarray(f_dc, dtype=np.float32)
    return _decode_rgb_values(f_dc[:, :3] * C0 + 0.5)


def _pose_heading_xy(pose: np.ndarray) -> np.ndarray:
    candidates = [
        pose[:2, 2],   # OpenCV/RDF forward
        -pose[:2, 2],  # fallback if convention flipped
        pose[:2, 0],   # right axis as last resort
    ]
    for vec in candidates:
        vec = np.asarray(vec, dtype=np.float32)
        if float(np.linalg.norm(vec)) > 1e-6:
            return vec / (np.linalg.norm(vec) + 1e-8)
    return np.array([1.0, 0.0], dtype=np.float32)
