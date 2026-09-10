"""
SemanticHeatPlanner
Extends ActiveGSPlannerv2 with:
  1. GI / GD semantic-heat scoring – replaces the plain Shannon-entropy
     seman_entropies in the exploration scoring loop with directional heat
     (GI) and voxel-level uncertainty (GD) from SemanticVoxelMap.

  2. Exploitation candidate branch – samples look-at poses near the
     topk_hot_voxels returned by SemanticVoxelMap, and injects them
     into the exploration pool so they compete with standard free-space
     candidates under the same λ-weighted NBV score.

NBV score
---------
  S(T) = λ_g · G_g(T) + λ_I · GI(T) + λ_D · GD(T) - λ_C · Cost(T)

  - G_g : geometric information gain  (unexplored pixel fraction, same as
           the explore_ig)
  - GI  : opacity-weighted sum of directional heat of visible voxels
  - GD  : opacity-weighted sum of semantic entropy of visible voxels
  - Cost: travel distance from current pose

When gs_slam.semantic_voxel_map is not initialised (e.g. during the
very first frames before any update), GI and GD fall back to 0 and the
formula degrades gracefully to the geometry-only scoring.
"""
from __future__ import annotations

from collections import OrderedDict
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import mmengine

from src.planner.active_gs_planner_v2 import ActiveGSPlannerv2
from src.slam.semsplatam.modified_ver.splatam.splatam import calc_shannon_entropy
from src.utils.general_utils import InfoPrinter


class SemanticHeatPlanner(ActiveGSPlannerv2):
    """
    Extends ActiveGSPlannerv2 with semantic-heat candidate generation and scoring.
    Refinement, post-refinement, and done states reuse parent behavior.
    -----------------------------------------------------
    lambda_I              : float = 0.3   # GI weight
    lambda_D              : float = 0.25   # GD weight
    lambda_g              : float = 0.3   # geometric IG weight
    lambda_C              : float = 0.15   # distance cost weight
    hot_voxel_topk        : int   = 20    # number of hot voxels to exploit
    hot_samples_per_voxel : int   = 3     # camera positions per hot voxel
    hot_sample_radius     : float = 1.0   # sampling radius around voxel (m)
    """

    def __init__(
        self,
        main_cfg: mmengine.Config,
        info_printer: InfoPrinter,
    ) -> None:
        super().__init__(main_cfg, info_printer)

        # λ weights for the NBV score
        self.lambda_I = float(self.planner_cfg.get("lambda_I", 0.3))
        self.lambda_D = float(self.planner_cfg.get("lambda_D", 0.25))
        self.lambda_g = float(self.planner_cfg.get("lambda_g", 0.3))
        self.lambda_C = float(self.planner_cfg.get("lambda_C", 0.15))

        # Exploitation candidate settings.
        self.hot_voxel_topk = int(self.planner_cfg.get("hot_voxel_topk", 20))
        self.hot_samples_per_voxel = int(self.planner_cfg.get("hot_samples_per_voxel", 3))
        self.hot_sample_radius = float(self.planner_cfg.get("hot_sample_radius", 1.0))
        self.hot_voxel_active_only = bool(self.planner_cfg.get("hot_voxel_active_only", True))
        self.hot_voxel_min_value = float(self.planner_cfg.get("hot_voxel_min_value", 0.0))
        self.hot_voxel_fallback_to_alpha_active = bool(
            self.planner_cfg.get("hot_voxel_fallback_to_alpha_active", True)
        )
        # Relaxation controls for hot-voxel candidate feasibility.
        self.hot_sample_strict_free_only = bool(
            self.planner_cfg.get("hot_sample_strict_free_only", True)
        )
        self.hot_sample_occ_max = float(self.planner_cfg.get("hot_sample_occ_max", -1e-6))
        self.hot_sample_clearance_vox = max(
            0,
            int(self.planner_cfg.get("hot_sample_clearance_vox", 0)),
        )
        self.hot_sample_require_free_segment = bool(
            self.planner_cfg.get("hot_sample_require_free_segment", True)
        )
        self.hot_sample_radius_min = float(self.planner_cfg.get("hot_sample_radius_min", 0.5))
        self.hot_voxel_candidate_factor = float(
            self.planner_cfg.get("hot_voxel_candidate_factor", 1.0)
        )
        self.hot_dir_random_extra = int(self.planner_cfg.get("hot_dir_random_extra", 0))
        self.hot_selection_revalidate = bool(
            self.planner_cfg.get("hot_selection_revalidate", True)
        )
        # Stage-1 directional sweep: check fixed radii along hot directions.
        self.stage1_hot_radial_search = bool(self.planner_cfg.get("stage1_hot_radial_search", True))
        self.stage1_hot_radius_start = float(self.planner_cfg.get("stage1_hot_radius_start", 0.1))
        self.stage1_hot_radius_end = float(self.planner_cfg.get("stage1_hot_radius_end", 1.0))
        self.stage1_hot_radius_step = float(self.planner_cfg.get("stage1_hot_radius_step", 0.1))
        self.stage1_hot_collect_all_free = bool(self.planner_cfg.get("stage1_hot_collect_all_free", True))
        # Stage-1 only use exploitation candidates (skip standard free-space sampling).
        self.stage1_exploitation_only = bool(self.planner_cfg.get("stage1_exploitation_only", False))
        # Debug: print details when selected NBV comes from hot-voxel exploitation candidates.
        self.log_selected_hot_nbv = bool(self.planner_cfg.get("log_selected_hot_nbv", True))
        # Guard to avoid repeatedly selecting a zero-motion NBV.
        self.min_nbv_move_dist = float(self.planner_cfg.get("min_nbv_move_dist", 0.0))
        self.hot_min_nbv_move_dist = float(
            self.planner_cfg.get("hot_min_nbv_move_dist", self.min_nbv_move_dist)
        )
        # Quantization used to build stable HOT keys across planning steps.
        self.hot_key_pos_resolution = float(self.planner_cfg.get("hot_key_pos_resolution", 0.0))

        # NBV scoring acceleration knobs.
        # Keep simulator-valid-mask correction by default, but cache repeated
        # candidate masks so scoring does not re-simulate identical poses.
        self.nbv_use_sim_mask_correction = bool(
            self.planner_cfg.get("nbv_use_sim_mask_correction", True)
        )
        self.nbv_enable_sim_mask_cache = bool(
            self.planner_cfg.get("nbv_enable_sim_mask_cache", True)
        )
        self.nbv_sim_mask_cache_size = max(
            0,
            int(self.planner_cfg.get("nbv_sim_mask_cache_size", 64)),
        )
        self._nbv_valid_sim_mask_cache: "OrderedDict[Tuple[int, ...], object]" = OrderedDict()
        # Debug counters for why exploitation candidate generation returned empty.
        self._last_exploitation_reject_stats = {}

    def _build_exploitation_keys(self, expl_poses: torch.Tensor, gs_slam) -> torch.Tensor:
        """Build stable HOT candidate keys from quantized camera position.

        Previous key format used current step id, which made the same physical
        HOT pose look like a brand-new candidate every planning step.
        """
        if expl_poses.numel() == 0:
            return torch.empty((0, 5), dtype=torch.long, device=self.device)

        key_res = self.hot_key_pos_resolution
        if key_res <= 0.0:
            explr_map = getattr(gs_slam, "explr_map", None)
            key_res = float(getattr(explr_map, "voxel_size", 0.1))
        key_res = max(key_res, 1e-4)

        q_xyz = torch.round(expl_poses[:, :3, 3] / key_res).to(dtype=torch.long)
        sentinel = torch.full((expl_poses.shape[0], 2), -2, dtype=torch.long, device=expl_poses.device)
        return torch.cat([sentinel, q_xyz], dim=1)

    # ------------------------------------------------------------------
    # Exploitation: topk hot voxels to candidate poses
    # ------------------------------------------------------------------

    def generate_exploitation_candidates(self, gs_slam) -> Optional[torch.Tensor]:
        """
        Sample look-at camera poses near the top-K hot voxels.

        For each hot voxel, candidate camera positions are placed along the
        **top-``hot_samples_per_voxel`` HEALPix directions** by heat score
        H_v[p], at distance ``hot_sample_radius`` from the voxel centre.

        Direction convention (matches SemanticVoxelMap):
          d_p = camera to voxel direction
          cam_pos = vox_center - d_p * r   (camera is on the -d_p side)

        Only positions that fall in free space (occupancy_grid <= 0) are kept.

        Returns
        -------
        Tensor of shape  (N, 4, 4)  c2w matrices in SplaTAM world coords,
        or ``None`` when no hot voxels are available.
        """
        stats = {
            "mode": "stage1_radial" if (self.exploration_stage == 1 and self.stage1_hot_radial_search) else "one_shot",
            "semantic_map_missing": False,
            "hot_voxels": 0,
            "hot_voxels_empty": False,
            "dirs_considered": 0,
            "radius_samples": 0,
            "reject_not_free": 0,
            "reject_segment": 0,
            "reject_lookat": 0,
            "accepted": 0,
        }
        self._last_exploitation_reject_stats = stats

        semantic_voxel_map = getattr(gs_slam, "semantic_voxel_map", None)
        if semantic_voxel_map is None:
            stats["semantic_map_missing"] = True
            return None

        raw_k = max(self.hot_voxel_topk, int(round(self.hot_voxel_topk * self.hot_voxel_candidate_factor)))
        hot_voxels: List[int] = semantic_voxel_map.topk_hot_voxels(
            raw_k,
            active_only=self.hot_voxel_active_only,
            min_value=self.hot_voxel_min_value,
            fallback_to_alpha_active=self.hot_voxel_fallback_to_alpha_active,
        )
        # Cap the ranked hot-voxel list to the configured candidate budget.
        if len(hot_voxels) > self.hot_voxel_topk:
            hot_voxels = hot_voxels[: self.hot_voxel_topk]
        stats["hot_voxels"] = len(hot_voxels)
        if not hot_voxels:
            stats["hot_voxels_empty"] = True
            return None

        # Candidate feasibility checks are evaluated in exploration-map grid space.
        explr_map = getattr(gs_slam, "explr_map", None)

        def _slam_to_sim(slam_pos: torch.Tensor) -> Optional[torch.Tensor]:
            if explr_map is None:
                return None
            try:
                return self.coord_conversion_slam2sim(slam_pos.unsqueeze(0)).squeeze(0)
            except Exception:
                return None

        def _sim_pos_to_grid_idx(sim_pos: torch.Tensor) -> Optional[Tuple[int, int, int]]:
            if explr_map is None:
                return None
            try:
                vxl = explr_map.transform_xyz_to_vxl(sim_pos.unsqueeze(0))
                xi = int(torch.floor(vxl[0, 0]).item())
                yi = int(torch.floor(vxl[0, 1]).item())
                zi = int(torch.floor(vxl[0, 2]).item())
                return xi, yi, zi
            except Exception:
                return None

        def _is_feasible_idx(xi: int, yi: int, zi: int) -> bool:
            """Occupancy feasibility check with configurable threshold.

            occ convention: 1=occupied, 0=unknown, -1=free.
            A cell is feasible when occ <= hot_sample_occ_max.
            """
            if explr_map is None:
                return False
            grid = explr_map.occupancy_grid
            Nx, Ny, Nz = grid.shape
            margin = self.hot_sample_clearance_vox
            if (
                xi < margin
                or yi < margin
                or zi < margin
                or xi >= Nx - margin
                or yi >= Ny - margin
                or zi >= Nz - margin
            ):
                return False
            occ_th = -1e-6 if self.hot_sample_strict_free_only else self.hot_sample_occ_max
            if margin <= 0:
                return float(grid[xi, yi, zi].item()) <= occ_th

            neigh = grid[
                xi - margin : xi + margin + 1,
                yi - margin : yi + margin + 1,
                zi - margin : zi + margin + 1,
            ]
            return float(neigh.max().item()) <= occ_th

        def _is_free(slam_pos: torch.Tensor) -> bool:
            """Return True if slam_pos falls in a strictly free voxel."""
            sim_pos = _slam_to_sim(slam_pos)
            if sim_pos is None:
                return False
            idx = _sim_pos_to_grid_idx(sim_pos)
            if idx is None:
                return False
            return _is_feasible_idx(*idx)

        def _is_segment_strict_free(start_slam: torch.Tensor, end_slam: torch.Tensor) -> bool:
            """Return True iff every traversed voxel between start and end is strictly free."""
            if explr_map is None:
                return False
            start_sim = _slam_to_sim(start_slam)
            end_sim = _slam_to_sim(end_slam)
            if start_sim is None or end_sim is None:
                return False

            direction = end_sim - start_sim
            dist = float(direction.norm().item())
            step = max(float(explr_map.voxel_size) * 0.5, 1e-4)
            n_steps = max(2, int(math.ceil(dist / step)) + 1)

            # Check open segment (exclude t=0 at voxel center), include endpoint (t=1).
            t_vals = torch.linspace(0.0, 1.0, n_steps, device=start_sim.device)[1:]
            for t in t_vals:
                p_sim = start_sim + t * direction
                idx = _sim_pos_to_grid_idx(p_sim)
                if idx is None:
                    return False
                if not _is_feasible_idx(*idx):
                    return False
            return True

        # d_p is the camera->voxel direction in SLAM world coordinates.
        bin_dirs: torch.Tensor = semantic_voxel_map._bin_dirs.to(device=self.device, dtype=torch.float32)

        poses: List[torch.Tensor] = []
        for v in hot_voxels:
            vox_center = semantic_voxel_map.get_voxel_center(v)

            # Select top-k bins by directional heat.
            H_v: torch.Tensor = semantic_voxel_map.compute_direction_heat(v)
            k = min(self.hot_samples_per_voxel + max(0, self.hot_dir_random_extra), int(H_v.numel()))
            topk_bins = torch.topk(H_v, k=k, largest=True).indices
            if self.hot_dir_random_extra > 0 and topk_bins.numel() > self.hot_samples_per_voxel:
                perm = torch.randperm(topk_bins.numel(), device=topk_bins.device)
                topk_bins = topk_bins[perm[: self.hot_samples_per_voxel]]

            for p_idx in topk_bins.tolist():
                stats["dirs_considered"] += 1
                d_p = bin_dirs[p_idx]
                # Stage-1 sweep mode: search outward at fixed radii along hot direction.
                # Rule:
                #   - retain each feasible free-space candidate;
                #   - skip blocked samples until the first free point is found;
                #   - stop expansion when a free region becomes blocked, or after the first accepted point
                #     when collect_all_free=False.
                if self.exploration_stage == 1 and self.stage1_hot_radial_search:
                    r0 = max(0.0, self.stage1_hot_radius_start)
                    r1 = max(r0, self.stage1_hot_radius_end)
                    dr = max(1e-4, self.stage1_hot_radius_step)

                    found_free = False
                    radius = r0
                    while radius <= r1 + 1e-8:
                        stats["radius_samples"] += 1
                        cam_pos = vox_center - d_p * radius

                        if not _is_free(cam_pos):
                            stats["reject_not_free"] += 1
                            # Keep searching farther radii. Nearby samples may lie inside
                            # obstacles/unknown cells while farther points are still feasible.
                            if not found_free:
                                radius += dr
                                continue
                            # Already found free points and now blocked -> stop expansion.
                            break

                        if self.hot_sample_require_free_segment and not _is_segment_strict_free(vox_center, cam_pos):
                            stats["reject_segment"] += 1
                            if not found_free:
                                radius += dr
                                continue
                            break

                        c2w = self._lookat_pose(cam_pos, vox_center)
                        if c2w is not None:
                            poses.append(c2w)
                            stats["accepted"] += 1
                            found_free = True
                        else:
                            stats["reject_lookat"] += 1

                        if found_free and (not self.stage1_hot_collect_all_free):       # if found a candidate and not need to collect all free points, stop searching further along this direction
                            break

                        radius += dr

                else:
                    # Default mode: randomized one-shot radius sampling.
                    stats["radius_samples"] += 1
                    random_radius = self.hot_sample_radius_min + self.hot_sample_radius * torch.rand(1).item()
                    cam_pos = vox_center - d_p * random_radius

                    if not _is_free(cam_pos):
                        stats["reject_not_free"] += 1
                        continue
                    # Optionally enforce strict LoS occupancy feasibility.
                    if self.hot_sample_require_free_segment and not _is_segment_strict_free(vox_center, cam_pos):
                        stats["reject_segment"] += 1
                        continue

                    c2w = self._lookat_pose(cam_pos, vox_center)
                    if c2w is not None:
                        poses.append(c2w)
                        stats["accepted"] += 1
                    else:
                        stats["reject_lookat"] += 1


        if not poses:
            return None
        return torch.stack(poses, dim=0)  # (N, 4, 4)

    def _lookat_pose(
        self,
        cam_pos: torch.Tensor,
        target: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Build a 4×4 c2w matrix with the camera at cam_pos looking toward
        target, using the SplaTAM RUB convention (right-up-backward).

        Returns None when cam_pos ≈ target.
        """
        forward = target - cam_pos           # camera looks in −Z (SplaTAM)
        if float(forward.norm()) < 1e-6:
            return None
        forward = forward / forward.norm()      # normalize

        # World-up heuristic: prefer Y-up (SplaTAM default); fall back to Z
        world_up = torch.tensor([0., 1., 0.], device=self.device, dtype=cam_pos.dtype)
        if abs(float(torch.dot(forward, world_up))) > 0.99:
            world_up = torch.tensor([0., 0., 1.], device=self.device, dtype=cam_pos.dtype)

        right   = torch.cross(forward, world_up, dim=0)
        right   = right / (right.norm() + 1e-8)             # normalize and avoid NaN
        up      = torch.cross(right, forward, dim=0)
        up      = up / (up.norm() + 1e-8)                   # re-orthogonalize and normalize

        # SplaTAM c2w: columns = [right, up, −forward, pos]     (RUB convention camera to world)
        c2w = torch.eye(4, device=self.device, dtype=cam_pos.dtype)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward
        c2w[:3, 3] = cam_pos
        return c2w

    def _get_cached_valid_sim_mask(
        self,
        cand_key: Tuple,
        cand_pose: torch.Tensor,
    ):
        """Get valid simulator depth mask for a candidate pose with LRU cache."""
        use_cache = self.nbv_enable_sim_mask_cache and self.nbv_sim_mask_cache_size > 0
        cache_key = tuple(int(x) for x in cand_key)

        if use_cache and cache_key in self._nbv_valid_sim_mask_cache:
            mask_np = self._nbv_valid_sim_mask_cache.pop(cache_key)
            self._nbv_valid_sim_mask_cache[cache_key] = mask_np
            return mask_np

        depth_gt = self.sim.simulate(
            self.pose_conversion_slam2sim(cand_pose).detach().cpu().numpy(),
            no_print=True,
        )["depth"]
        # Habitat wrappers may return either numpy arrays or torch tensors.
        if torch.is_tensor(depth_gt):
            mask_np = (depth_gt > 0.2).detach().cpu().numpy()
        else:
            mask_np = np.asarray(depth_gt > 0.2)

        if use_cache:
            if len(self._nbv_valid_sim_mask_cache) >= self.nbv_sim_mask_cache_size:
                self._nbv_valid_sim_mask_cache.popitem(last=False)
            self._nbv_valid_sim_mask_cache[cache_key] = mask_np

        return mask_np

    # ------------------------------------------------------------------
    # Override: rendering_based_planning
    # ------------------------------------------------------------------

    def rendering_based_planning(self, cur_pose, gs_slam):
        """
        Override of the ActiveSGM's planning loop.

        Exploration branch is completely rewritten with:
          - Exploitation candidates injected into explore_pool
          - GI / GD scoring replacing seman_entropies
          - λ-weighted NBV score

        All other planning states (``refinement``, ``post_refinement``,
        ``done``) delegate to the ActiveSGM implementation unchanged.
        """
        new_pose = cur_pose
        self._last_selected_explore_key = None
        self._last_ranked_explore_candidates = []

        # ----------------------------------------------------------------
        # Exploration state
        # ----------------------------------------------------------------
        if (
            self.planning_state == "exploration"
            and self.exploration_stage < self.num_exploration_stage
        ):
            self.info_printer(
                f"Current state: {self.state} | {self.planning_state}: "
                f"Getting New Exploration Map",
                self.step, self.__class__.__name__,
            )

            # Get exploration map (free space in simulator coordinates).
            gs_z_levels      = self.gs_z_levels[self.exploration_stage]
            xy_sampling_step = self.planner_cfg.xy_sampling_step[self.exploration_stage]
            new_free_voxels  = gs_slam.explr_map.get_new_free_voxels(
                use_xyz_filter=True,
                xy_sampling_step=xy_sampling_step,
                gs_z_levels=gs_z_levels,
            )
            new_free_locs_sim = (
                gs_slam.explr_map.origin + new_free_voxels * gs_slam.explr_map.voxel_size
            )

            # Debug: report number of new free voxels / sampled locations
            self.info_printer(
                f"[SemanticHeatPlanner] new_free_voxels: {new_free_voxels.shape[0]}, new_free_locs_sim: {new_free_locs_sim.shape[0]}",
                self.step,
                self.__class__.__name__,
            )

            # Update previous-free cache for the next planning step.
            gs_slam.explr_map.update_prev_free_voxels(
                use_xyz_filter=True,
                xy_sampling_step=xy_sampling_step,
                gs_z_levels=gs_z_levels,
            )

            use_standard_exploration = not (
                self.stage1_exploitation_only and self.exploration_stage == 1
            )

            # Sample standard exploration candidates from newly free voxels.
            if use_standard_exploration and new_free_locs_sim.shape[0] != 0:
                new_cand_poses = self.generate_candidate_poses(
                    new_free_locs_sim,
                    gs_slam.explr_map.sim2slam,
                ).reshape(-1, 4, 4)

                free_vxl_idx_exp = new_free_voxels.unsqueeze(1).repeat(
                    1, self.num_dir_samples[self.exploration_stage], 1
                )
                view_rot_idx_exp = self.view_rot_idx[self.exploration_stage].repeat(
                    new_free_voxels.shape[0], 1, 1
                )
                new_cand_pose_key = torch.cat(
                    [free_vxl_idx_exp, view_rot_idx_exp], dim=-1
                ).reshape(-1, 4)
                self.add_explore_pool_cand(new_cand_poses, new_cand_pose_key)
                self.info_printer(
                    f"[SemanticHeatPlanner] Added {new_cand_poses.shape[0]} new exploration candidates "
                    f"from {new_free_voxels.shape[0]} new free voxels into explore_pool.",
                    self.step, self.__class__.__name__,
                )
            elif not use_standard_exploration:
                self.info_printer(
                    "[SemanticHeatPlanner] Stage-1 exploitation-only mode: skip standard exploration sampling.",
                    self.step,
                    self.__class__.__name__,
                )

            # ---- Inject exploitation candidates from hot voxels ----
            # Stage 0: geometry-only exploration (match original ActiveSGM behaviour).
            # Semantic heat candidates are only injected from stage 1 onward.
            if self.exploration_stage >= 1:
                expl_poses = self.generate_exploitation_candidates(gs_slam)
                if expl_poses is None or expl_poses.shape[0] == 0:
                    self.info_printer(
                        f"[SemanticHeatPlanner] No exploitation poses generated.",
                        self.step,
                        self.__class__.__name__,
                    )
                    stats = getattr(self, "_last_exploitation_reject_stats", None)
                    if isinstance(stats, dict) and len(stats) > 0:
                        if stats.get("semantic_map_missing", False):
                            reason = "semantic_voxel_map_missing"
                        elif stats.get("hot_voxels_empty", False):
                            reason = "hot_voxels_empty"
                        else:
                            reason = "all_candidates_rejected"

                        self.info_printer(
                            "[SemanticHeatPlanner] HOT reject breakdown | "
                            f"reason={reason} | "
                            f"hot_voxels={int(stats.get('hot_voxels', 0))} | "
                            f"dirs={int(stats.get('dirs_considered', 0))} | "
                            f"samples={int(stats.get('radius_samples', 0))} | "
                            f"reject_not_free={int(stats.get('reject_not_free', 0))} | "
                            f"reject_segment={int(stats.get('reject_segment', 0))} | "
                            f"reject_lookat={int(stats.get('reject_lookat', 0))} | "
                            f"accepted={int(stats.get('accepted', 0))} | "
                            f"mode={stats.get('mode', 'na')}",
                            self.step,
                            self.__class__.__name__,
                        )
                if expl_poses is not None and expl_poses.shape[0] > 0:
                    n_expl = expl_poses.shape[0]
                    # Key format: (-2, -2, qx, qy, qz), where q* are quantized
                    # camera coordinates in SLAM world; this keeps identical HOT
                    # poses stable across steps.
                    expl_keys = self._build_exploitation_keys(expl_poses, gs_slam)
                    self.add_explore_pool_cand(expl_poses, expl_keys)
                    self.info_printer(
                        f"[SemanticHeatPlanner] Injected {n_expl} exploitation candidates "
                        f"from {min(self.hot_voxel_topk, n_expl // max(1, self.hot_samples_per_voxel))} "
                        f"hot voxels into explore_pool.",
                        self.step, self.__class__.__name__,
                    )

            # ---- Check if exploration is done ----
            is_explore_done = (len(self.explore_pool) == 0)                     # Exploration is done when the explore pool is empty.
            if self.step >= self.max_exploration_steps:                          # Also force completion when max exploration steps are reached.
                is_explore_done = True
                self.info_printer(
                    f"Current state: {self.state} | {self.planning_state} {self.exploration_stage}: "
                    f"Run out maximum exploration steps - {self.exploration_stage}, "
                    f"starting evaluation...",
                    self.step, self.__class__.__name__,
                )

            # ---- Transition when exploration is done ----
            if is_explore_done:
                if self.exploration_stage < self.num_exploration_stage:
                    self.info_printer(
                        f"Current state: {self.state} | {self.planning_state} {self.exploration_stage}: "
                        f"Done Exploration Stage - {self.exploration_stage}, "
                        f"starting evaluation...",
                        self.step, self.__class__.__name__,
                    )
                    eval_dir_suffix = f"exploration_stage_{self.exploration_stage}"         # Evaluation subdirectory named by current exploration stage.
                    self.gs_slam.print_and_save_result(                                     # Save results for this exploration stage, including semantic gains and global keyframe info.
                        eval_dir_suffix, is_prune_gaussians=False, ignore_first_frame=True
                    )
                    eval_dir = self.gs_slam.eval_dir + "_" + eval_dir_suffix
                    os.makedirs(eval_dir, exist_ok=True)
                    with open(os.path.join(eval_dir, "exploration_info.txt"), "w") as f:
                        f.write(f"exploration_stage_{self.exploration_stage}_step: {self.step}\n")
                        f.write("global_keyframe: ")
                        if self.main_cfg.slam.use_global_keyframe:
                            f.write(f"{sorted(self.gs_slam.global_keyframe_indices)}\n")

                    self.exploration_stage += 1
                    self.planning_state = "exploration"
                # When all exploration stages are finished, transition to post_refinement (if enabled) or done.
                if self.exploration_stage == self.num_exploration_stage:
                    self.info_printer(
                        f"Current state: {self.state} | {self.planning_state} {self.exploration_stage}: "
                        f"Done All Exploration.",
                        self.step, self.__class__.__name__,
                    )
                    # Check if post_refinement should be skipped
                    skip_post_refinement = bool(getattr(self.main_cfg.slam, 'skip_post_refinement_stage', False))
                    if self.main_cfg.slam.use_global_keyframe and not skip_post_refinement:
                        # Use global keyframes for the subsequent refinement stage.
                        self.planning_state = "post_refinement"
                        self.post_refine_counter = 0
                    else:
                        self.planning_state = "done"
                else:
                    # Reset exploration map for the next stage
                    gs_z_levels      = self.gs_z_levels[self.exploration_stage]
                    xy_sampling_step = self.planner_cfg.xy_sampling_step[self.exploration_stage]
                    gs_slam.explr_map.prev_free_voxels = torch.empty(0, 3).to(self.device)

            # ---- Score candidates (exploration not done yet) ----
            else:
                self.info_printer(
                    f"Current state: {self.state} | {self.planning_state} {self.exploration_stage}: "
                    f"Evaluate Exploration Candidate I.G.",
                    self.step, self.__class__.__name__,
                )
                self.planning_state = "exploration"
                cand_poses, cand_keys = self.get_explore_pool_poses()

                dists = torch.norm(cand_poses[:, :3, 3] - cur_pose[:3, 3], dim=1) + 1e-6

                explore_igs: List[torch.Tensor] = []
                gi_scores:   List[float] = []
                gd_scores:   List[float] = []
                
                has_voxel_map = (
                    hasattr(gs_slam, "semantic_voxel_map")
                    and gs_slam.semantic_voxel_map is not None
                    and hasattr(gs_slam, "eval_candidate_semantic_gains")
                    and getattr(gs_slam, "gaussian_voxel_idx", None) is not None
                )
                for i, cand_pose in enumerate(cand_poses):
                    img, depth, valid_mask, seen = gs_slam.render(cand_pose)

                    # Optional simulator-valid-mask correction for incomplete regions.
                    if self.nbv_use_sim_mask_correction:
                        valid_sim_mask_np = self._get_cached_valid_sim_mask(cand_keys[i], cand_pose)
                        if torch.is_tensor(valid_sim_mask_np):
                            valid_sim_mask = valid_sim_mask_np.to(
                                device=valid_mask.device,
                                dtype=torch.bool,
                            )
                        else:
                            valid_sim_mask = torch.from_numpy(valid_sim_mask_np).to(
                                device=valid_mask.device,
                                dtype=torch.bool,
                            )
                        if valid_sim_mask.shape != valid_mask.shape[-2:]:
                            valid_sim_mask = F.interpolate(
                                valid_sim_mask.unsqueeze(0).unsqueeze(0).float(),
                                size=valid_mask.shape[-2:],
                                mode="nearest",
                            )[0, 0].to(dtype=torch.bool)
                        valid_mask[0][~valid_sim_mask] = True

                    _, self.img_h, self.img_w = img.shape
                    explore_igs.append((valid_mask == 0).sum())
                    if has_voxel_map:
                        GI, GD = gs_slam.eval_candidate_semantic_gains(cand_pose, seen=seen)
                    else:
                        GI, GD= 0.0, 0.0
                    gi_scores.append(GI)
                    gd_scores.append(GD)

                explore_igs_t = torch.stack(explore_igs).float()
                explore_igs_norm = self.minmax_normalize(explore_igs_t)
                gi_t = torch.tensor(gi_scores, device=self.device).float()
                gd_t = torch.tensor(gd_scores, device=self.device).float()
                gi_norm = self.minmax_normalize(gi_t)
                gd_norm = self.minmax_normalize(gd_t)
                cost_norm = self.minmax_normalize(dists)
                
                weighted_score = (
                    self.lambda_g * explore_igs_norm
                    + self.lambda_I * gi_norm
                    + self.lambda_D * gd_norm
                    - self.lambda_C * cost_norm
                )

                # Guard against zero-motion loops: if we have farther options,
                # suppress candidates that are too close to the current pose.
                if self.min_nbv_move_dist > 0.0 or self.hot_min_nbv_move_dist > 0.0:
                    is_hot = torch.tensor(
                        [len(key) >= 1 and int(key[0]) == -2 for key in cand_keys],
                        dtype=torch.bool,
                        device=dists.device,
                    )
                    base_th = torch.full_like(dists, self.min_nbv_move_dist)
                    hot_th = torch.full_like(dists, self.hot_min_nbv_move_dist)
                    move_th = torch.where(is_hot, hot_th, base_th)
                    near_mask = (move_th > 0.0) & (dists < move_th)

                    if bool(near_mask.any()) and bool((~near_mask).any()):
                        weighted_score = weighted_score.clone()
                        weighted_score[near_mask] = -1e9
                        self.info_printer(
                            "[NBV Guard] Masked "
                            f"{int(near_mask.sum().item())} near-current candidates "
                            f"(min_nbv_move_dist={self.min_nbv_move_dist:.3f}, "
                            f"hot_min_nbv_move_dist={self.hot_min_nbv_move_dist:.3f}).",
                            self.step,
                            self.__class__.__name__,
                        )
                    elif bool(near_mask.any()):
                        self.info_printer(
                            "[NBV Guard] All candidates are within min-move threshold; "
                            "keeping original scores.",
                            self.step,
                            self.__class__.__name__,
                        )

                # Re-validate HOT candidates and keep a ranked list for RRT fallback.
                ranked_idx_for_rrt = []
                if self.hot_selection_revalidate:
                    explr_map = getattr(gs_slam, "explr_map", None)

                    def _hot_pose_safe(start_slam: torch.Tensor, end_slam: torch.Tensor) -> bool:
                        if explr_map is None:
                            return True

                        occ = explr_map.occupancy_grid
                        Nx, Ny, Nz = occ.shape
                        margin = self.hot_sample_clearance_vox

                        def _slam_to_sim(slam_pos: torch.Tensor) -> Optional[torch.Tensor]:
                            try:
                                return self.coord_conversion_slam2sim(slam_pos.unsqueeze(0)).squeeze(0)
                            except Exception:
                                return None

                        def _sim_to_idx(sim_pos: torch.Tensor) -> Optional[Tuple[int, int, int]]:
                            try:
                                vxl = explr_map.transform_xyz_to_vxl(sim_pos.unsqueeze(0))
                                xi = int(torch.floor(vxl[0, 0]).item())
                                yi = int(torch.floor(vxl[0, 1]).item())
                                zi = int(torch.floor(vxl[0, 2]).item())
                                return xi, yi, zi
                            except Exception:
                                return None

                        def _idx_ok(xi: int, yi: int, zi: int) -> bool:
                            if (
                                xi < margin
                                or yi < margin
                                or zi < margin
                                or xi >= Nx - margin
                                or yi >= Ny - margin
                                or zi >= Nz - margin
                            ):
                                return False
                            if margin <= 0:
                                return float(occ[xi, yi, zi].item()) == -1.0
                            neigh = occ[
                                xi - margin : xi + margin + 1,
                                yi - margin : yi + margin + 1,
                                zi - margin : zi + margin + 1,
                            ]
                            return bool((neigh == -1.0).all().item())

                        s_sim = _slam_to_sim(start_slam)
                        e_sim = _slam_to_sim(end_slam)
                        if s_sim is None or e_sim is None:
                            return False

                        e_idx = _sim_to_idx(e_sim)
                        if e_idx is None or not _idx_ok(*e_idx):
                            return False

                        direction = e_sim - s_sim
                        dist = float(direction.norm().item())
                        step = max(float(explr_map.voxel_size) * 0.5, 1e-4)
                        n_steps = max(2, int(math.ceil(dist / step)) + 1)
                        for t in torch.linspace(0.0, 1.0, n_steps, device=s_sim.device)[1:]:
                            p_sim = s_sim + t * direction
                            idx = _sim_to_idx(p_sim)
                            if idx is None or not _idx_ok(*idx):
                                return False
                        return True

                    sorted_idx = torch.argsort(weighted_score, descending=True)
                    masked_hot = 0
                    for idx_t in sorted_idx:
                        idx = int(idx_t.item())
                        key = cand_keys[idx]
                        is_hot = (
                            len(key) >= 2
                            and int(key[0]) == -2
                            and int(key[1]) == -2
                        )
                        if is_hot and not _hot_pose_safe(cur_pose[:3, 3], cand_poses[idx, :3, 3]):
                            masked_hot += 1
                            continue
                        ranked_idx_for_rrt.append(idx)

                    if masked_hot > 0:
                        self.info_printer(
                            f"[HOT Guard] Rejected {masked_hot} unsafe HOT top-ranked candidates before NBV selection.",
                            self.step,
                            self.__class__.__name__,
                        )
                else:
                    sorted_idx = torch.argsort(weighted_score, descending=True)
                    ranked_idx_for_rrt = [int(idx_t.item()) for idx_t in sorted_idx]

                if len(ranked_idx_for_rrt) == 0:
                    sorted_idx = torch.argsort(weighted_score, descending=True)
                    ranked_idx_for_rrt = [int(idx_t.item()) for idx_t in sorted_idx]

                next_visit = ranked_idx_for_rrt[0]
                new_pose = cand_poses[next_visit]
                self._last_selected_explore_key = cand_keys[next_visit]
                self._last_ranked_explore_candidates = [
                    (cand_keys[idx], cand_poses[idx].detach().clone())
                    for idx in ranked_idx_for_rrt
                ]
                self.info_printer(
                    f"Current state: {self.state} [Exploration Pool: {len(self.explore_pool)}]",
                    self.step, self.__class__.__name__,
                )
                self.info_printer(f"Exploration I.G.   : {explore_igs_norm}", self.step, self.__class__.__name__)
                self.info_printer(f"GI scores          : {gi_norm}", self.step, self.__class__.__name__)
                self.info_printer(f"GD scores          : {gd_norm}", self.step, self.__class__.__name__)
                self.info_printer(f"Distance costs     : {cost_norm}", self.step, self.__class__.__name__)
                self.info_printer(f"NBV scores         : {weighted_score}", self.step, self.__class__.__name__)

                selected_key = cand_keys[next_visit]
                # Exploitation candidates use sentinel key prefix (-2, -2, step, i).
                is_hot_exploitation = (
                    len(selected_key) >= 2
                    and int(selected_key[0]) == -2
                    and int(selected_key[1]) == -2
                )
                if self.log_selected_hot_nbv and is_hot_exploitation:
                    pose_xyz_slam = new_pose[:3, 3].detach().cpu().tolist()
                    pose_xyz_sim = None
                    try:
                        sim_pose = self.pose_conversion_slam2sim(new_pose.detach())
                        pose_xyz_sim = sim_pose[:3, 3].detach().cpu().tolist()
                    except Exception:
                        try:
                            sim_xyz = self.coord_conversion_slam2sim(new_pose[:3, 3].unsqueeze(0))
                            pose_xyz_sim = sim_xyz[0].detach().cpu().tolist()
                        except Exception:
                            pose_xyz_sim = None

                    pose_xyz_msg = f"pose_xyz_slam={[round(v, 4) for v in pose_xyz_slam]}"
                    if pose_xyz_sim is not None:
                        pose_xyz_msg += f" | pose_xyz_sim={[round(v, 4) for v in pose_xyz_sim]}"

                    self.info_printer(
                        "[SemanticHeatPlanner] Selected NBV from HOT-VOXEL exploitation | "
                        f"key={selected_key} | "
                        f"score={float(weighted_score[next_visit].item()):.6f} | "
                        f"Gg={float(explore_igs_t[next_visit].item()):.1f} "
                        f"(norm={float(explore_igs_norm[next_visit].item()):.4f}) | "
                        f"GI={float(gi_t[next_visit].item()):.6f} "
                        f"(norm={float(gi_norm[next_visit].item()):.4f}) | "
                        f"GD={float(gd_t[next_visit].item()):.6f} "
                        f"(norm={float(gd_norm[next_visit].item()):.4f}) | "
                        f"dist={float(dists[next_visit].item()):.4f} "
                        f"(norm={float(cost_norm[next_visit].item()):.4f}) | "
                        f"{pose_xyz_msg}",
                        self.step,
                        self.__class__.__name__,
                    )

                self.update_explore_pool_cand(explore_igs_t, cand_keys, next_visit)
                self.del_explore_pool_cand(
                    explore_igs_t, cand_keys,
                    self.planner_cfg.explore_thre,
                    self.planner_cfg.recognize_thre,
                )
                nbv_score_thre = float(self.planner_cfg.get('nbv_score_thre', 0.01))
                nbv_pool_age_thre = int(self.planner_cfg.get('nbv_pool_age_thre', 3))
                self.del_low_nbv_pool_cand(
                    weighted_score,
                    cand_keys,
                    nbv_score_thre,
                    pool_age_thre=nbv_pool_age_thre,
                )
                self.info_printer(
                    f"Current state: {self.state} [Exploration Pool: {len(self.explore_pool)}]",
                    self.step, self.__class__.__name__,
                )

        # ----------------------------------------------------------------
        # Refinement / Post-Refinement / Done  (delegate to parent)
        # ----------------------------------------------------------------
        if self.planning_state in ("refinement", "post_refinement", "done"):
            # Call parent for all non-exploration states.
            # We temporarily force planning_state so the parent's exploration
            # guard does NOT fire (it checks `planning_state == "exploration"`).
            _saved_stage = self.exploration_stage
            _saved_state = self.planning_state
            # Set exploration_stage to max so the exploration `if` in the
            # parent is always False (even if planning_state were "exploration")
            self.exploration_stage = self.num_exploration_stage
            parent_out = super().rendering_based_planning(cur_pose, gs_slam)
            # Restore stage (parent may have changed planning_state internally)
            self.exploration_stage = _saved_stage
            return parent_out

        out = dict(
            new_pose=new_pose,
            ranked_explore_candidates=self._last_ranked_explore_candidates,
        )
        return out