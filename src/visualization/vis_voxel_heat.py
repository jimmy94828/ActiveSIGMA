import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from src.utils.healpix_utils import dir_to_bin_batch


@dataclass
class VoxelHeatRenderResult:
    overlay_bgr: np.ndarray
    heat_bgr: np.ndarray
    heat_values: np.ndarray


@dataclass
class VoxelHeatRenderBundle:
    depth_aligned: VoxelHeatRenderResult
    all_active: VoxelHeatRenderResult


@dataclass
class VoxelScalarRenderTriplet:
    directional_heat: VoxelHeatRenderResult
    entropy: VoxelHeatRenderResult
    voxel_heat: VoxelHeatRenderResult


@dataclass
class VoxelScalarRenderBundle:
    depth_aligned: VoxelScalarRenderTriplet
    all_active: VoxelScalarRenderTriplet


@dataclass
class ProjectedVoxelData:
    color_bgr: np.ndarray
    x: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor
    voxel_ids: torch.Tensor
    view_bins: torch.Tensor
    fx: float
    voxel_size: float
    depth_valid: torch.Tensor


class VoxelHeatVisualizer:
    """Project the semantic voxel heat map into the current GT image."""

    def __init__(
        self,
        alpha: float = 0.55,
        min_radius: int = 2,
        max_radius: int = 6,
        heat_min: float = 0.0,
        heat_max: float = 0.5,
        direction_heat_min: float = None,
        direction_heat_max: float = None,
        entropy_min: float = 0.0,
        entropy_max: float = 1.0,
        voxel_value_min: float = None,
        voxel_value_max: float = None,
        color_low_bgr=(255, 0, 0),
        color_mid_bgr=(0, 255, 255),
        color_high_bgr=(0, 0, 255),
    ) -> None:
        self.alpha = float(alpha)
        self.min_radius = int(min_radius)
        self.max_radius = int(max_radius)
        self.heat_min = float(heat_min)
        self.heat_max = float(heat_max)
        self.direction_heat_min = float(heat_min if direction_heat_min is None else direction_heat_min)
        self.direction_heat_max = float(heat_max if direction_heat_max is None else direction_heat_max)
        self.entropy_min = float(entropy_min)
        self.entropy_max = float(entropy_max)
        self.voxel_value_min = float(heat_min if voxel_value_min is None else voxel_value_min)
        self.voxel_value_max = float(heat_max if voxel_value_max is None else voxel_value_max)
        self.color_low_bgr = np.asarray(color_low_bgr, dtype=np.float32).reshape(1, 1, 3)
        self.color_mid_bgr = np.asarray(color_mid_bgr, dtype=np.float32).reshape(1, 1, 3)
        self.color_high_bgr = np.asarray(color_high_bgr, dtype=np.float32).reshape(1, 1, 3)

    @torch.no_grad()
    def render(
        self,
        slam,
        color: torch.Tensor,
        depth: torch.Tensor,
        c2w_slam: torch.Tensor,
    ) -> VoxelHeatRenderBundle:
        scalar_bundle = self.render_scalar_maps(slam, color, depth, c2w_slam)
        return VoxelHeatRenderBundle(
            depth_aligned=scalar_bundle.depth_aligned.voxel_heat,
            all_active=scalar_bundle.all_active.voxel_heat,
        )

    @torch.no_grad()
    def render_scalar_maps(
        self,
        slam,
        color: torch.Tensor,
        depth: torch.Tensor,
        c2w_slam: torch.Tensor,
    ) -> VoxelScalarRenderBundle:
        projected = self._project_active_voxels(slam, color, depth, c2w_slam)
        if projected is None:
            empty_triplet = self._empty_triplet(self._to_bgr_uint8(color))
            return VoxelScalarRenderBundle(depth_aligned=empty_triplet, all_active=empty_triplet)

        semantic_voxel_map = slam.semantic_voxel_map
        device = semantic_voxel_map.device
        voxel_ids = projected.voxel_ids
        log_n_classes = float(np.log(max(2, int(getattr(semantic_voxel_map, "n_classes", 2)))))

        directional_heat_scores = torch.as_tensor(
            [semantic_voxel_map.query_h_bar(int(v.item())) for v in voxel_ids],
            device=device,
            dtype=torch.float32,
        )
        if hasattr(semantic_voxel_map, "query_value_entropy"):
            entropy_scores = torch.as_tensor(
                [semantic_voxel_map.query_value_entropy(int(v.item())) for v in voxel_ids],
                device=device,
                dtype=torch.float32,
            )
        else:
            entropy_scores = torch.as_tensor(
                [semantic_voxel_map.query_entropy(int(v.item())) for v in voxel_ids],
                device=device,
                dtype=torch.float32,
            )
            if log_n_classes > 1e-8:
                entropy_scores = entropy_scores / log_n_classes
        voxel_heat_scores = torch.as_tensor(
            [semantic_voxel_map.compute_voxel_value(int(v.item())) for v in voxel_ids],
            device=device,
            dtype=torch.float32,
        )

        all_active_triplet = VoxelScalarRenderTriplet(
            directional_heat=self._render_projected_points(
                color_bgr=projected.color_bgr,
                x=projected.x,
                y=projected.y,
                z=projected.z,
                scalar_scores=directional_heat_scores,
                fx=projected.fx,
                voxel_size=projected.voxel_size,
                score_min=self.direction_heat_min,
                score_max=self.direction_heat_max,
            ),
            entropy=self._render_projected_points(
                color_bgr=projected.color_bgr,
                x=projected.x,
                y=projected.y,
                z=projected.z,
                scalar_scores=entropy_scores,
                fx=projected.fx,
                voxel_size=projected.voxel_size,
                score_min=self.entropy_min,
                score_max=self.entropy_max,
            ),
            voxel_heat=self._render_projected_points(
                color_bgr=projected.color_bgr,
                x=projected.x,
                y=projected.y,
                z=projected.z,
                scalar_scores=voxel_heat_scores,
                fx=projected.fx,
                voxel_size=projected.voxel_size,
                score_min=self.voxel_value_min,
                score_max=self.voxel_value_max,
            ),
        )

        depth_mask = projected.depth_valid
        depth_aligned_triplet = VoxelScalarRenderTriplet(
            directional_heat=self._render_projected_points(
                color_bgr=projected.color_bgr,
                x=projected.x[depth_mask],
                y=projected.y[depth_mask],
                z=projected.z[depth_mask],
                scalar_scores=directional_heat_scores[depth_mask],
                fx=projected.fx,
                voxel_size=projected.voxel_size,
                score_min=self.direction_heat_min,
                score_max=self.direction_heat_max,
            ),
            entropy=self._render_projected_points(
                color_bgr=projected.color_bgr,
                x=projected.x[depth_mask],
                y=projected.y[depth_mask],
                z=projected.z[depth_mask],
                scalar_scores=entropy_scores[depth_mask],
                fx=projected.fx,
                voxel_size=projected.voxel_size,
                score_min=self.entropy_min,
                score_max=self.entropy_max,
            ),
            voxel_heat=self._render_projected_points(
                color_bgr=projected.color_bgr,
                x=projected.x[depth_mask],
                y=projected.y[depth_mask],
                z=projected.z[depth_mask],
                scalar_scores=voxel_heat_scores[depth_mask],
                fx=projected.fx,
                voxel_size=projected.voxel_size,
                score_min=self.voxel_value_min,
                score_max=self.voxel_value_max,
            ),
        )

        return VoxelScalarRenderBundle(
            depth_aligned=depth_aligned_triplet,
            all_active=all_active_triplet,
        )

    def _project_active_voxels(
        self,
        slam,
        color: torch.Tensor,
        depth: torch.Tensor,
        c2w_slam: torch.Tensor,
    ) -> ProjectedVoxelData:
        color_bgr = self._to_bgr_uint8(color)
        height, width = color_bgr.shape[:2]

        semantic_voxel_map = getattr(slam, "semantic_voxel_map", None)
        if semantic_voxel_map is None:
            return None

        active_mask = getattr(semantic_voxel_map, "alpha_v_active", None)
        if active_mask is not None:
            active_voxels = torch.where(active_mask)[0].tolist()
        else:
            active_voxels = sorted(getattr(semantic_voxel_map, "directional_active_voxels", []))
        if not active_voxels:
            return None

        device = semantic_voxel_map.device
        c2w_slam = c2w_slam.to(device=device, dtype=torch.float32)
        w2c = torch.linalg.inv(c2w_slam)
        cam_pos = c2w_slam[:3, 3]

        voxel_ids = torch.as_tensor(active_voxels, device=device, dtype=torch.long)
        voxel_centers = semantic_voxel_map.voxel_center_world(voxel_ids)

        ones = torch.ones((voxel_centers.shape[0], 1), device=device, dtype=torch.float32)
        voxel_centers_h = torch.cat([voxel_centers, ones], dim=-1)
        cam_pts = (w2c @ voxel_centers_h.T).T[:, :3]

        z = cam_pts[:, 2]
        valid = z > 1e-6
        if not bool(valid.any()):
            return None

        voxel_ids = voxel_ids[valid]
        voxel_centers = voxel_centers[valid]
        cam_pts = cam_pts[valid]
        z = z[valid]

        intrinsics = torch.as_tensor(slam.intrinsics, device=device, dtype=torch.float32)
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        u = fx * cam_pts[:, 0] / z + cx
        v = fy * cam_pts[:, 1] / z + cy
        x = torch.round(u).long()
        y = torch.round(v).long()

        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not bool(valid.any()):
            return None

        voxel_ids = voxel_ids[valid]
        voxel_centers = voxel_centers[valid]
        x = x[valid]
        y = y[valid]
        z = z[valid]

        view_dirs = cam_pos.unsqueeze(0) - voxel_centers
        view_bins = dir_to_bin_batch(view_dirs, nside=semantic_voxel_map.nside)

        gt_depth = depth.to(device=device, dtype=torch.float32)
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.squeeze(0)
        depth_at_pixel = gt_depth[y, x]
        depth_margin = max(0.05, 0.5 * math.sqrt(3.0) * float(semantic_voxel_map.voxel_size))
        depth_valid = (depth_at_pixel > 0) & (torch.abs(z - depth_at_pixel) <= depth_margin)

        return ProjectedVoxelData(
            color_bgr=color_bgr,
            x=x,
            y=y,
            z=z,
            voxel_ids=voxel_ids,
            view_bins=view_bins,
            fx=float(fx.item()),
            voxel_size=float(semantic_voxel_map.voxel_size),
            depth_valid=depth_valid,
        )

    def _render_projected_points(
        self,
        color_bgr: np.ndarray,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        scalar_scores: torch.Tensor,
        fx: float,
        voxel_size: float,
        score_min: float,
        score_max: float,
    ) -> VoxelHeatRenderResult:
        heat_values = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
        active_mask = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
        if scalar_scores.numel() == 0:
            return self._compose_result(color_bgr, heat_values)

        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy()

        norm_scores = self._normalize_scalar_scores(scalar_scores, score_min=score_min, score_max=score_max)

        order = torch.argsort(norm_scores)
        norm_np = norm_scores[order].detach().cpu().numpy()
        order_np = order.detach().cpu().numpy()
        x_np = x_np[order_np]
        y_np = y_np[order_np]
        z_np = z_np[order_np]

        for px, py, pz, score in zip(x_np, y_np, z_np, norm_np):
            intensity = int(round(float(score) * 255.0))
            radius = int(np.clip(round(fx * voxel_size / max(float(pz), 1e-3)), self.min_radius, self.max_radius))
            center = (int(px), int(py))
            cv2.circle(heat_values, center, radius, intensity, thickness=-1)
            cv2.circle(active_mask, center, radius, 255, thickness=-1)

        return self._compose_result(color_bgr, heat_values, active_mask=active_mask)

    def _normalize_scalar_scores(self, scalar_scores: torch.Tensor, score_min: float, score_max: float) -> torch.Tensor:
        heat_span = float(score_max) - float(score_min)
        if heat_span <= 0.0:
            return torch.zeros_like(scalar_scores)
        return ((scalar_scores - float(score_min)) / heat_span).clamp(0.0, 1.0)

    def _empty_triplet(self, color_bgr: np.ndarray) -> VoxelScalarRenderTriplet:
        empty = np.zeros((color_bgr.shape[0], color_bgr.shape[1]), dtype=np.uint8)
        empty_result = self._compose_result(color_bgr, empty)
        return VoxelScalarRenderTriplet(
            directional_heat=empty_result,
            entropy=empty_result,
            voxel_heat=empty_result,
        )

    def _heat_values_to_bgr(self, heat_values: np.ndarray, active_mask: np.ndarray = None) -> np.ndarray:
        heat_01 = heat_values.astype(np.float32) / 255.0
        heat_bgr = np.empty((*heat_values.shape, 3), dtype=np.float32)
        low_mask = heat_01 <= 0.5
        high_mask = ~low_mask

        if np.any(low_mask):
            low_t = (heat_01[low_mask] / 0.5)[:, None]
            heat_bgr[low_mask] = (1.0 - low_t) * self.color_low_bgr.reshape(3) + low_t * self.color_mid_bgr.reshape(3)

        if np.any(high_mask):
            high_t = ((heat_01[high_mask] - 0.5) / 0.5)[:, None]
            heat_bgr[high_mask] = (1.0 - high_t) * self.color_mid_bgr.reshape(3) + high_t * self.color_high_bgr.reshape(3)

        heat_bgr = np.clip(np.rint(heat_bgr), 0.0, 255.0).astype(np.uint8)
        if active_mask is not None:
            heat_bgr[active_mask <= 0] = 0
        else:
            heat_bgr[heat_values <= 0] = 0
        return heat_bgr

    def _append_colorbar(self, heat_bgr: np.ndarray) -> np.ndarray:
        height = int(heat_bgr.shape[0])
        if height <= 0:
            return heat_bgr

        pad = 12
        bar_width = 20
        label_width = 44
        total_width = heat_bgr.shape[1] + pad + bar_width + label_width
        canvas = np.zeros((height, total_width, 3), dtype=np.uint8)
        canvas[:, : heat_bgr.shape[1]] = heat_bgr

        bar_x0 = heat_bgr.shape[1] + pad
        bar_x1 = bar_x0 + bar_width
        top = 10
        bottom = max(top + 1, height - 10)
        bar_heat = np.linspace(255.0, 0.0, bottom - top, dtype=np.float32)[:, None]
        bar_colors = self._heat_values_to_bgr(np.repeat(bar_heat, bar_width, axis=1))
        canvas[top:bottom, bar_x0:bar_x1] = bar_colors
        cv2.rectangle(canvas, (bar_x0 - 1, top - 1), (bar_x1, bottom), (180, 180, 180), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        text_color = (255, 255, 255)
        label_x = bar_x1 + 6
        cv2.putText(
            canvas,
            f"{self.heat_max:.1f}",
            (label_x, top + 4),
            font,
            font_scale,
            text_color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"{self.heat_min:.1f}",
            (label_x, bottom),
            font,
            font_scale,
            text_color,
            thickness,
            lineType=cv2.LINE_AA,
        )
        return canvas

    def _compose_result(
        self,
        color_bgr: np.ndarray,
        heat_values: np.ndarray,
        active_mask: np.ndarray = None,
    ) -> VoxelHeatRenderResult:
        heat_bgr_raw = self._heat_values_to_bgr(heat_values, active_mask=active_mask)
        overlay_bgr = color_bgr.copy()
        if active_mask is not None:
            mask = active_mask > 0
        else:
            mask = heat_values > 0
        if np.any(mask):
            blended = cv2.addWeighted(color_bgr, 1.0 - self.alpha, heat_bgr_raw, self.alpha, 0.0)
            overlay_bgr[mask] = blended[mask]
        heat_bgr = self._append_colorbar(heat_bgr_raw)
        return VoxelHeatRenderResult(
            overlay_bgr=overlay_bgr,
            heat_bgr=heat_bgr,
            heat_values=heat_values,
        )

    @staticmethod
    def _to_bgr_uint8(color: torch.Tensor) -> np.ndarray:
        color_np = color.detach().cpu().numpy()
        color_np = np.clip(color_np * 255.0, 0.0, 255.0).astype(np.uint8)
        return cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)
