"""
MIT License

Copyright (c) 2024 OPPO

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import cv2
import matplotlib.pyplot as plt
import mmengine
import numpy as np
import os
import torch
from typing import List

from src.planner.active_lang_planner import ActiveLangPlanner
from src.slam.splatam.splatam import SplatamOurs as Splatam

from src.visualization.visualizer import Visualizer
from src.utils.general_utils import InfoPrinter

from typing import Union
from third_parties.coslam.utils import colormap_image
from src.visualization.vis_bev_diagnostic import BevDiagnosticVisualizer
from src.visualization.vis_voxel_heat import VoxelHeatVisualizer
from src.visualization.vis_voxel_heat_global import VoxelHeatGlobalVisualizer


class ActiveLangVisualizer(Visualizer):
    def __init__(self, 
                 main_cfg    : mmengine.Config,
                 info_printer: InfoPrinter
                 ) -> None:
        """
        Args:
            main_cfg (mmengine.Config): Configuration
            info_printer (InfoPrinter): information printer
    
        Attributes:
            main_cfg (mmengine.Config): configurations
            vis_cfg (mmengine.Config) : visualizer model configurations
            info_printer (InfoPrinter): information printer
            
        """
        super(ActiveLangVisualizer, self).__init__(main_cfg, info_printer)
        self.voxel_heat_hist_every = int(self.vis_cfg.get("voxel_heat_hist_every", 50))
        if self.voxel_heat_hist_every <= 0:
            self.voxel_heat_hist_every = 50

        keys = ['rgbd', 'pose', 'planning_path', 'lookat_tgts', 'state', 'information_gain', 'rendered_rgbd']
        # Add rendered_semantic to saved keys for semantic renderings
        keys.append('rendered_semantic')
        ### create directory ###
        for key in keys:
            if self.vis_cfg.get(f"save_{key}", False):
                vis_dir = os.path.join(self.main_cfg.dirs.result_dir, "visualization", key)
                os.makedirs(vis_dir, exist_ok=True)

        if self.vis_cfg.get('save_voxel_heat', False):
            os.makedirs(os.path.join(self.main_cfg.dirs.result_dir, "visualization", "vis_heat"), exist_ok=True)
            os.makedirs(os.path.join(self.main_cfg.dirs.result_dir, "visualization", "voxel_heat_global"), exist_ok=True)
            os.makedirs(os.path.join(self.main_cfg.dirs.result_dir, "visualization", "voxel_heat_hist"), exist_ok=True)
            os.makedirs(os.path.join(self.main_cfg.dirs.result_dir, "visualization", "render_all_info"), exist_ok=True)
        if self.vis_cfg.get('save_bev_diagnostic', False):
            os.makedirs(os.path.join(self.main_cfg.dirs.result_dir, "visualization", "bev_diagnostic"), exist_ok=True)

        ### write remark ###
        with open(os.path.join(self.main_cfg.dirs.result_dir, "visualization", "README.md"), 'w') as f:
            f.writelines("rgbd: GT RGB-D visualization\n")
            f.writelines("render_rgbd: render RGB-D visualization\n")
            f.writelines("rendered_semantic: rendered semantic visualization (color-coded classes)\n")
            f.writelines("poses: (np.ndarray, [4,4]). Camera-to-world. RUB system. \n")
            f.writelines("planning_path: (np.ndarray, [N,4,4]), each element is a planning pose \n")
            f.writelines("lookat_tgts: (np.ndarray, [N,3]), uncertaint target observation locations to lookat.  \n")
            f.writelines("state: (str), planner state \n")
            f.writelines("vis_heat: GT RGB with projected semantic voxel heat overlay\n")
            f.writelines("vis_heat/*_depth.png: depth-aligned overlay, only voxels consistent with GT depth\n")
            f.writelines("vis_heat/*_all.png: overlay without depth filtering, shows all active voxels projected to the frame\n")
            f.writelines("voxel_heat_global: global orthographic voxel heat projections\n")
            f.writelines("voxel_heat_hist: voxel heat value histogram (range 0~1, bin width 0.01, every 50 steps)\n")
            f.writelines("bev_diagnostic: per-iteration XY BEV diagnostic map with projected RGB, current pose, selected NBV pose, candidate pool, and top voxel-heat-value voxels\n")
            f.writelines("render_all_info: 2x3 grid with titles: [RGB | Depth | Semantic] / [Voxel Value | Directional (h_bar) | Entropy(active-dir)] (bottom row is depth-aligned projection)\n")

        self.voxel_heat_visualizer = VoxelHeatVisualizer()
        self.voxel_heat_global_visualizer = VoxelHeatGlobalVisualizer()
        self.bev_diagnostic_visualizer = BevDiagnosticVisualizer(
            canvas_size=int(self.vis_cfg.get("bev_diagnostic_size", 900)),
            jsd_top_percent=float(self.vis_cfg.get("bev_diag_jsd_top_percent", 0.10)),
            show_candidate_pool=bool(self.vis_cfg.get("bev_diag_show_candidate_pool", True)),
            bev_pad_ratio=float(self.vis_cfg.get("bev_diag_pad_ratio", 0.05)),
            remove_front_percent=self.vis_cfg.get("bev_diag_remove_front_percent", 25.0),
            flip_view=bool(self.vis_cfg.get("bev_diag_flip_view", False)),
            min_sigma_px=float(self.vis_cfg.get("bev_diag_min_sigma_px", 0.75)),
            max_sigma_px=float(self.vis_cfg.get("bev_diag_max_sigma_px", 3.0)),
            max_kernel_radius=int(self.vis_cfg.get("bev_diag_max_kernel_radius", 4)),
        )

    def main(self,
             slam           : Splatam,
             planner        : ActiveLangPlanner,
             color          : torch.Tensor,
             depth          : torch.Tensor,
             im              :torch.Tensor,
             rastered_depth  : torch.Tensor,
             pose            : torch.Tensor,
             voxel_heat_pose : torch.Tensor = None,
             rendered_semantic: torch.Tensor = None,
             ) -> None:
        """ save data for visualization purpose

        Args:
            slam           : SLAM module
            planner        : Planner module
            color          : [H,W,3], color image. Range  : 0-1
            depth          : [H,W,3], depth image.
            im           : [H,W,3], rendered color image. Range  : 0-1
            rastered_depth         : [H,W,3], rastered depth image.
            pose           : [4,4],   current pose.
                             The render path uses `voxel_heat_pose` when provided, which is the
                             relative SplaTAM render pose (`c2w_slam_rel`, OpenCV/RDF-style pinhole projection).

        Returns:


        Attributes:

        """
        # Rendered RGB/semantic/depth panels are generated from the relative SplaTAM pose
        # (`c2w_slam_rel`).
        heat_pose = voxel_heat_pose if voxel_heat_pose is not None else pose

        self.save_render_rgb_depth_semantic(
            slam,
            im,
            rastered_depth,
            rendered_semantic,
            heat_pose,
        )
        if self.step % 10 != 0:
            return
        
        self.save_voxel_heat_histogram(slam)
        
        
        if self.vis_cfg.get('save_voxel_heat', False):
            if self.step % 5 != 0:
                return
            self.info_printer("Saving voxel heat overlay for visualization", self.step, self.__class__.__name__)
            heat_pose = voxel_heat_pose if voxel_heat_pose is not None else pose
            self.save_vis_heat(slam, color, depth, heat_pose)
            self.save_voxel_heat_global(slam)
            self.save_voxel_heat_histogram(slam)


        return
        ### GT RGB-D ###
        if self.vis_cfg.get('save_rgbd', False):
            if self.step % 5 != 0:
                return
            self.info_printer("Saving RGBD for visualization", self.step, self.__class__.__name__)
            self.save_rgbd(color, depth)

        ### render RGB-D ###
        if self.vis_cfg.get('save_rendered_rgbd', False):
            if self.step % 5 != 0:
                return
            self.info_printer("Saving rendered RGBD for visualization", self.step, self.__class__.__name__)
            self.save_render_rgb_depth_semantic(
                slam,
                im,
                rastered_depth,
                rendered_semantic,
                heat_pose,
            )

        ### pose ###
        if self.vis_cfg.get('save_pose', False):
            self.info_printer("Saving pose for visualization", self.step, self.__class__.__name__)
            pose_np = pose.detach().cpu().numpy()
            self.save_pose(pose_np)

        ### state ###
        if self.vis_cfg.get('save_state', False):
            self.info_printer("Saving state for visualization", self.step, self.__class__.__name__)
            self.save_state(planner)

        
        # if self.step > 0:
        #     ### planning_path ###
        #     if self.vis_cfg.save_planning_path:
        #         self.info_printer("Saving planning_path for visualization", self.step, self.__class__.__name__)
        #         self.save_planning_path(planner)
        #
        #     ### lookat_tgt ###
        #     if self.vis_cfg.save_lookat_tgts:
        #         self.info_printer("Saving lookat_tgt for visualization", self.step, self.__class__.__name__)
        #         self.save_lookat_tgt(planner)

           #### information gain ##########
            # if self.vis_cfg.save_information_gain:
            #     self.info_printer("Saving state for visualization", self.step, self.__class__.__name__)
            #     self.save_igs(planner)

    def save_rgbd(self, 
                  color: torch.Tensor, 
                  depth: torch.Tensor
                  ) -> None:
        """save RGB-D visualization
    
        Args:
            rgb (torch.Tensor, [H,W,3]): color map. Range: 0-1
            depth (torch.Tensor, [H,W]): depth map.
    
        """
        rgbd_vis = self.visualize_rgbd(color, depth, return_vis=True)
        filepath = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "rgbd", f"{self.step:04}.png")
        rgbd_vis = (rgbd_vis * 255).astype(np.uint8)
        cv2.imwrite(filepath, rgbd_vis)

    @staticmethod
    def _eval_label_colormap(n_label: int = 256) -> np.ndarray:
        """Match render_rgb_sem_entro_bev semantic color assignment."""
        cmap = np.zeros((n_label, 3), dtype=np.uint8)
        for i in range(n_label):
            r = g = b = 0
            cid = i
            for j in range(8):
                r |= ((cid >> 0) & 1) << (7 - j)
                g |= ((cid >> 1) & 1) << (7 - j)
                b |= ((cid >> 2) & 1) << (7 - j)
                cid >>= 3
            cmap[i] = np.array([r, g, b], dtype=np.uint8)
        return cmap

    def _semantic_to_bgr(self, semantic: torch.Tensor) -> np.ndarray:
        """Convert a rendered semantic tensor into a uint8 BGR image."""
        sem = semantic.detach().cpu().numpy()

        if sem.ndim == 3 and sem.shape[0] in (1, 3, 4):
            sem = np.transpose(sem, (1, 2, 0))

        if sem.ndim == 2:
            s = sem.astype(np.int32)
            if s.size == 0:
                img = np.zeros((0, 0, 3), dtype=np.uint8)
            else:
                max_cls = int(np.max(s))
                cmap_size = max(max_cls + 1, 256) if max_cls >= 0 else 256
                cmap = self._eval_label_colormap(cmap_size)
                mapped = np.clip(s, 0, cmap.shape[0] - 1)
                img = cmap[mapped]
        else:
            img = sem
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = np.clip(img, 0.0, 1.0)
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        if img.ndim == 3 and img.shape[2] >= 3:
            try:
                return cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
            except Exception:
                return img[:, :, :3]
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def _render_current_view_voxel_maps_bgr(
        self,
        slam: Splatam,
        color: torch.Tensor,
        depth: torch.Tensor,
        pose: torch.Tensor,
    ) -> Union[dict, None]:
        semantic_voxel_map = getattr(slam, "semantic_voxel_map", None)
        if semantic_voxel_map is None or pose is None:
            return None

        result = self.voxel_heat_visualizer.render_scalar_maps(slam, color, depth, pose)
        return {
            "directional_heat": result.depth_aligned.directional_heat.heat_bgr,
            "voxel_entropy": result.depth_aligned.entropy.heat_bgr,
            "voxel_heat": result.depth_aligned.voxel_heat.heat_bgr,
        }

    def save_render_rgb_depth_semantic(self,
                  slam: Splatam,
                  im: torch.Tensor,
                  rastered_depth: torch.Tensor,
                  semantic: torch.Tensor = None,
                  pose: torch.Tensor = None):
        vis_size = 320
        rgb_float = cv2.cvtColor(im.detach().cpu().numpy(), cv2.COLOR_RGB2BGR)
        if rgb_float.shape[0] != vis_size or rgb_float.shape[1] != vis_size:
            rgb_float = cv2.resize(rgb_float, (vis_size, vis_size))
        rgb_uint8 = np.clip(rgb_float * 255.0, 0.0, 255.0).astype(np.uint8)

        depth = rastered_depth.unsqueeze(0)
        mask = (depth < 100.0) * 1.0
        depth_vis = colormap_image(depth, mask)
        depth_vis = depth_vis.permute(1, 2, 0).cpu().numpy()
        depth_vis = cv2.resize(depth_vis, (vis_size, vis_size))
        depth_uint8 = np.clip(depth_vis * 255.0, 0.0, 255.0).astype(np.uint8)

        semantic_uint8 = None
        if semantic is not None:
            semantic_bgr = self._semantic_to_bgr(semantic)
            if semantic_bgr.size > 0:
                semantic_bgr = cv2.resize(semantic_bgr, (vis_size, vis_size), interpolation=cv2.INTER_NEAREST)
                semantic_uint8 = semantic_bgr

        voxel_maps_bgr = self._render_current_view_voxel_maps_bgr(slam, im, rastered_depth, pose)
        if voxel_maps_bgr is not None:
            panel_specs = [
                ("voxel_heat", "voxel value"),
                ("directional_heat", "Directional (h_bar)"),
                ("voxel_entropy", "entropy (active-dir)"),
            ]
            bottom_row_panels = []
            for panel_key, panel_label in panel_specs:
                panel_bgr = voxel_maps_bgr.get(panel_key)
                if panel_bgr is None or panel_bgr.size == 0:
                    continue
                panel_bgr = cv2.resize(panel_bgr, (vis_size, vis_size), interpolation=cv2.INTER_NEAREST)
                bottom_row_panels.append((panel_bgr, panel_label))
            if len(bottom_row_panels) == 3:
                top_row_panels = [
                    (rgb_uint8, "RGB"),
                    (depth_uint8, "Depth"),
                    (semantic_uint8 if semantic_uint8 is not None else np.zeros_like(rgb_uint8), "Semantic"),
                ]
                grid_image = self._compose_titled_panel_grid(
                    [top_row_panels, bottom_row_panels],
                    panel_size=vis_size,
                )
                grid_dir = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "render_all_info")
                os.makedirs(grid_dir, exist_ok=True)
                cv2.imwrite(os.path.join(grid_dir, f"{self.step:04}.png"), grid_image)

    def _compose_titled_panel_grid(self, rows_with_panels, panel_size: int) -> np.ndarray:
        if len(rows_with_panels) == 0:
            return np.zeros((0, 0, 3), dtype=np.uint8)

        title_h = 40
        num_rows = len(rows_with_panels)
        num_cols = max(len(row) for row in rows_with_panels)
        canvas = np.full(
            ((panel_size + title_h) * num_rows, panel_size * num_cols, 3),
            255,
            dtype=np.uint8,
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        text_color = (30, 30, 30)

        for row_idx, row_panels in enumerate(rows_with_panels):
            y0 = row_idx * (panel_size + title_h)
            for col_idx, (panel_bgr, label) in enumerate(row_panels):
                x0 = col_idx * panel_size
                canvas[y0 + title_h:y0 + title_h + panel_size, x0:x0 + panel_size] = panel_bgr

                text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
                text_x = x0 + max(0, (panel_size - text_size[0]) // 2)
                text_y = y0 + max(text_size[1] + 4, (title_h + text_size[1]) // 2) - baseline
                cv2.putText(
                    canvas,
                    label,
                    (text_x, text_y),
                    font,
                    font_scale,
                    text_color,
                    thickness,
                    lineType=cv2.LINE_AA,
                )

        return canvas

    def save_vis_heat(self, slam: Splatam, color: torch.Tensor, depth: torch.Tensor, pose: torch.Tensor) -> None:
        """Save GT RGB with projected semantic voxel heat overlay."""
        vis_dir = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "vis_heat")
        os.makedirs(vis_dir, exist_ok=True)
        result = self.voxel_heat_visualizer.render(slam, color, depth, pose)
        cv2.imwrite(os.path.join(vis_dir, f"{self.step:04}_depth.png"), result.depth_aligned.overlay_bgr)
        cv2.imwrite(os.path.join(vis_dir, f"{self.step:04}_depth_heat.png"), result.depth_aligned.heat_bgr)
        cv2.imwrite(os.path.join(vis_dir, f"{self.step:04}_all.png"), result.all_active.overlay_bgr)
        cv2.imwrite(os.path.join(vis_dir, f"{self.step:04}_all_heat.png"), result.all_active.heat_bgr)

    def save_voxel_heat_global(self, slam: Splatam) -> None:
        """Save global voxel heat projections."""
        vis_dir = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "voxel_heat_global")
        os.makedirs(vis_dir, exist_ok=True)
        result = self.voxel_heat_global_visualizer.render(slam)
        cv2.imwrite(os.path.join(vis_dir, f"{self.step:04}.png"), result.image_bgr)

    def save_bev_diagnostic(self, slam: Splatam, planner: ActiveLangPlanner, pose: torch.Tensor) -> None:
        """Save a per-iteration XY BEV diagnostic image after planning."""
        if not self.vis_cfg.get("save_bev_diagnostic", False):
            return
        every = max(1, int(self.vis_cfg.get("bev_diagnostic_every", 1)))
        if self.step % every != 0:
            return

        vis_dir = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "bev_diagnostic")
        os.makedirs(vis_dir, exist_ok=True)
        result = self.bev_diagnostic_visualizer.render(slam, planner, pose)
        cv2.imwrite(os.path.join(vis_dir, f"{self.step:04}.png"), result.image_bgr)

    def _collect_voxel_heat_values(self, slam: Splatam) -> np.ndarray:
        semantic_voxel_map = getattr(slam, "semantic_voxel_map", None)
        if semantic_voxel_map is None:
            return np.zeros((0,), dtype=np.float32)

        active_mask = getattr(semantic_voxel_map, "alpha_v_active", None)
        if active_mask is not None:
            active_voxels = torch.where(active_mask)[0].tolist()
        else:
            active_voxels = sorted(getattr(semantic_voxel_map, "directional_active_voxels", []))

        if len(active_voxels) == 0:
            return np.zeros((0,), dtype=np.float32)

        heat_values = [float(semantic_voxel_map.compute_voxel_value(int(v))) for v in active_voxels]
        return np.asarray(heat_values, dtype=np.float32)

    def save_voxel_heat_histogram(self, slam: Splatam) -> None:
        self.info_printer("Saving voxel heat value histogram for visualization", self.step, self.__class__.__name__)
        vis_dir = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "voxel_heat_hist")
        os.makedirs(vis_dir, exist_ok=True)

        heat_values = self._collect_voxel_heat_values(slam)
        heat_values = heat_values[np.isfinite(heat_values)]

        bin_edges = np.linspace(0.0, 1.0, 101, dtype=np.float32)
        clipped_values = np.clip(heat_values, 0.0, 1.0)
        hist, _ = np.histogram(clipped_values, bins=bin_edges)

        y_cap = 2000
        y_tick_step = 200

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(
            bin_edges[:-1],
            hist,
            width=0.01,
            align='edge',
            color='#1f77b4',
            edgecolor='#0e223f',
            linewidth=0.2,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0, y_cap)
        ax.set_xticks(np.arange(0.0, 1.01, 0.1))
        ax.set_yticks(np.arange(0, y_cap + y_tick_step, y_tick_step))
        ax.set_xlabel("Voxel heat value interval")
        ax.set_ylabel("Number of voxels")
        ax.set_title(f"Voxel Heat Value Histogram (step {self.step:04d})")
        ax.grid(axis='y', linestyle='--', alpha=0.35)

        overflow_indices = np.where(hist > y_cap)[0]
        for idx in overflow_indices:
            x_center = float(bin_edges[idx] + 0.005)
            ax.text(
                x_center,
                y_cap - 20,
                f"{int(hist[idx])}",
                color='#b22222',
                fontsize=7,
                rotation=90,
                ha='center',
                va='top',
            )
            ax.plot(x_center, y_cap, marker='^', markersize=3, color='#b22222')

        fig.tight_layout()
        fig.savefig(os.path.join(vis_dir, f"{self.step:04}.png"), dpi=180)
        plt.close(fig)

    def save_pose(self, pose: np.ndarray) -> None:
        """ save pose
    
        Args:
            pose: [4,4], current pose. Format: camera-to-world, RUB system
    
        """
        filepath = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "pose", f"{self.step:04}.npy")
        np.save(filepath, pose)
    
    def save_planning_path(self, planner: ActiveLangPlanner) -> None:
        """ save planning path as np.ndarray (Nx3)
    
        Args:
            planner: Planner module
    
        """
        filepath = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "planning_path", f"{self.step:04}.npy")
        if planner.path is not None:
            ### path (List) : each element is a pose [GoalNode, ..., CurrentNode] ###
            # path_locs = [planner.vox2loc(node._xyz_arr) for node in planner.path]
            # path_locs = np.asarray(path_locs)
            if isinstance(planner.path, torch.Tensor):
                path_locs = [pose.cpu() for pose in planner.path]
            else:
                path_locs = planner.path
            path_locs = np.asarray(path_locs)
            np.save(filepath, path_locs)
        else:
            np.save(filepath, None)
    
    def save_lookat_tgt(self, planner: ActiveLangPlanner) -> None:
        """ save lookat targets (uncertain target observations) as np.ndarray (Nx3)
    
        Args:
            planner: planner module
    
        """
        filepath = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "lookat_tgts", f"{self.step:04}.npy")
        if planner.lookat_tgts is not None:
            ### lookat_tgts (List)      : uncertaint target observation locations to lookat. each element is (np.ndarray, [3]) ###
            lookat_tgt_locs = np.asarray(planner.lookat_tgts)
            np.save(filepath, lookat_tgt_locs)
        else:
            np.save(filepath, None)

    def save_state(self, planner: ActiveLangPlanner) -> None:
        """ save planner state
    
        Args:
            planner: planner module
    
        """
        filepath = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "state", f"{self.step:04}.txt")
        with open(filepath, 'w') as f:
            f.writelines(f"{planner.state}")

    def visualize_rgbd_w_render(self,
                       rgb       : torch.Tensor,
                       depth     : torch.Tensor,
                       im        : torch.Tensor,
                       rastered_depth: torch.Tensor,
                       max_depth : float = 100.,
                       vis_size  : int = 320,
                       return_vis: bool = False
                       ) -> Union[None, np.ndarray]:
        """ visualiz RGB-D
        Args:
            rgb (torch.Tensor, [H,W,3]): color map. Range: 0-1
            depth (torch.Tensor, [H,W]): depth map.
            max_depth (float)          : maximum depth value
            vis_size (int)             : image size used for visualization
            return_vis (bool)          : return visualization (OpenCV format) if True

        Returns:
            Union:
                - image (np.ndarray, [H,W,3]): RGB-D visualization if return_vis
        """
        ## process RGB ##
        rgb = cv2.cvtColor(rgb.cpu().numpy(), cv2.COLOR_RGB2BGR)
        rgb = cv2.resize(rgb, (vis_size, vis_size))

        im = cv2.cvtColor(im.cpu().numpy(), cv2.COLOR_RGB2BGR)
        im = cv2.resize(im, (vis_size, vis_size))


        ### process Depth map ###
        depth = depth.unsqueeze(0)
        mask = (depth < max_depth) * 1.0
        depth_colormap = colormap_image(depth, mask)
        depth_colormap = depth_colormap.permute(1, 2, 0).cpu().numpy()
        depth_colormap = cv2.resize(depth_colormap, (vis_size, vis_size))

        rastered_depth = rastered_depth.unsqueeze(0)
        mask = (rastered_depth < max_depth) * 1.0
        rastered_depth_colormap = colormap_image(rastered_depth, mask)
        rastered_depth_colormap = rastered_depth_colormap.permute(1, 2, 0).cpu().numpy()
        rastered_depth_colormap = cv2.resize(rastered_depth_colormap, (vis_size, vis_size))


        ### display RGB-D ###
        image_gt = np.hstack((rgb, depth_colormap))
        image_render = np.hstack((im, rastered_depth_colormap))
        image = np.vstack((image_gt,image_render))


        ### return visualization ###
        if return_vis:
            return image
        else:
            cv2.namedWindow('RGB-D', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RGB-D', image)
            key = cv2.waitKey(1)

    # def save_igs(self, planner:ActiveLangPlanner):
    #     if planner.state == "planning" and "exploration" in planner.planning_state:
    #         igs = []
    #         breakpoint()
    #         for key, val in planner.explore_pool.items():
    #             igs.append(val['ig'].detach().cpu().numpy())
    #         igs = np.asarray(igs)
    #         filepath = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "information_gain", f"{self.step:04}.npy")
    #         np.save(filepath, igs)
        
    # def save_color_mesh(self, slam: CoSLAM) -> None:
    #     """ save colored mesh
    
    #     Args:
    #         slam: SLAM module
    #     """
    #     if self.step % self.vis_cfg.save_mesh_freq == 0:
    #         mesh_dir = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "color_mesh")
    #         slam.save_mesh(self.step, voxel_size=self.vis_cfg.save_mesh_voxel_size, suffix='', mesh_savedir=mesh_dir)
    #     else:
    #         return
    
    # def save_uncert_mesh(self, slam: CoSLAM) -> None:
    #     """ save uncertainty mesh
    
    #     Args:
    #         slam: SLAM module
    #     """
    #     if self.step % self.vis_cfg.save_mesh_freq == 0:
    #         mesh_dir = os.path.join(self.main_cfg.dirs.result_dir, "visualization", "uncert_mesh")
    #         slam.save_uncert_mesh(self.step, voxel_size=self.vis_cfg.save_mesh_voxel_size, suffix='', mesh_savedir=mesh_dir)
    #     else:
    #         return
