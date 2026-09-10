"""Semantic voxel map with directional heat.

Core states:
- Voxel-level Dirichlet: ``alpha_v`` with shape ``[V, C]``.
- Directional Dirichlet (sparse): ``dir_alpha[v_idx]`` with shape ``[P, K+1]``
  where ``K+1 = top-k classes + residual``.

Current update design:
- Aggregate per-frame distributions by voxel / (voxel, direction-bin).
- Apply additive Dirichlet updates (no decay term in the current variant).

Current directional heat:
    H_v[p] = m_v[p] * (mean_{q in N(p)} JSD(pbar_{v,p}, pbar_{v,q})) * 1/(S_{v,p}+1)

Implementation notes:
- Directional state is sparse and created only for observed voxels.
- Dirty tracking is precise: only touched voxels invalidate heat/value caches.
- Direction bins are HEALPix pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import healpy as hp
import numpy as np
import torch
from torch import Tensor

from src.utils.healpix_utils import build_neighbor_table, bin_to_dir, dir_to_bin_batch  # HEALPix utilities
from src.utils.divergence import normalize, entropy, jsd


@dataclass
class VoxelUpdateStats:
    num_points_in_bounds: int
    num_voxels_updated: int
    num_voxel_dir_pairs_updated: int
    mean_rho: float


class SemanticVoxelMap:
    def __init__(                # Initialize configuration parameters
        self,
        bbox_bound: np.ndarray,  # [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
        voxel_size: float,       # Keep consistent with explr_map voxel size
        n_classes: int,          # num semantic classes (C)
        nside: int = 1,          # HEALPix nside(n_bins = 12 * nside^2)
        eta: float = 0.9,
        w_i: float = 0.5,
        w_n: float = 0.5,
        lambda_h: float = 0.6,   # Weight of directional-heat term in voxel value
        lambda_u: float = 0.4,   # Weight of semantic-uncertainty term in voxel value
        heat_s_min: Optional[float] = 0.3,  # Minimum evidence threshold for neighbor-JSD evaluation
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        alpha_init: float = 1e-2,           # Dirichlet prior (small positive value to avoid divide-by-zero)
        topk: int = 16,                     # Top-k in directional compressed space (stored as top-k + residual)
        feasible_beta: float = 0.85,        # EMA coefficient for feasible confidence f_v[p]
        feasible_tau_low: float = 0.4,      # f_v[p] <= tau_low is considered infeasible
        feasible_tau_high: float = 0.7,     # f_v[p] >= tau_high is considered feasible
        value_log_enable: bool = False,     # Whether to print the two terms of V(v)
        value_log_every: int = 20,          # Print once every N topk_hot_voxels calls
        value_log_topk: int = 5,            # Maximum number of top voxels to print each time
        active_dir_log_enable: bool = False,  # Whether to print active-direction count statistics
        active_dir_log_every: int = 20,       # Print once every N topk_hot_voxels calls
        active_dir_log_topk: int = 5,         # Maximum number of selected voxels to print each time
        heat_jsd_use_all_neighbors: bool = False,  # If True, JSD uses all geometric neighbors regardless observed status
        pixel_alpha: float = 1.0,
        pixel_beta: float = 1.0,
        pixel_gamma: float = 1.0,
        pixel_delta: float = 1.0,
        pixel_w_min: float = 0.3,
        pixel_reliability_use_render_entropy: bool = True,
        pixel_weight_log_enable: bool = False,
        pixel_weight_log_every: int = 20,
        pixel_conflict_enable: bool = True,
    ):
        bbox_bound = np.array(bbox_bound)
        assert bbox_bound.shape == (3, 2), "bbox_bound must be shape (3,2) = [[xmin,xmax],[ymin,ymax],[zmin,zmax]]"
        assert voxel_size > 0
        assert n_classes > 1
        assert nside >= 1
        assert 0.0 <= eta <= 1.0

        self.device = torch.device(device)
        self.dtype = dtype

        # Bounding box
        bbox_min = torch.as_tensor(bbox_bound[:, 0], device=self.device, dtype=self.dtype)
        bbox_max = torch.as_tensor(bbox_bound[:, 1], device=self.device, dtype=self.dtype)
        self.bbox_min: Tensor = bbox_min
        self.bbox_max: Tensor = bbox_max
        self.voxel_size: float = float(voxel_size)

        # Grid shape (Nx, Ny, Nz)
        grid_shape = torch.ceil((bbox_max - bbox_min) / self.voxel_size).to(torch.int64)
        grid_shape = torch.clamp(grid_shape, min=1)
        self.grid_shape: Tuple[int, int, int] = (int(grid_shape[0].item()), int(grid_shape[1].item()), int(grid_shape[2].item()))
        self.num_voxels: int = self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2]

        # Semantic + directional config (HEALPix)
        self.n_classes: int = int(n_classes)
        self.nside: int = int(nside)
        self.n_bins: int = hp.nside2npix(self.nside)    # 12 * nside^2 equal-area pixels
        # Kept for compatibility with previous configs (not used now)
        self.eta: float = float(eta)
        self.w_i: float = float(w_i)
        self.w_n: float = float(w_n)
        self.alpha_init: float = float(alpha_init)      # Dirichlet prior
        # directional state: top-k + 1 residual bucket
        self.dir_topk: int = max(1, min(int(topk), int(n_classes) - 1))
        self.dir_dim: int = self.dir_topk + 1
        self.lambda_h: float = float(lambda_h)
        self.lambda_u: float = float(lambda_u)
        self.feasible_beta: float = float(feasible_beta)
        self.feasible_tau_low: float = float(feasible_tau_low)
        self.feasible_tau_high: float = float(feasible_tau_high)
        self.value_log_enable: bool = bool(value_log_enable)
        self.value_log_every: int = max(1, int(value_log_every))
        self.value_log_topk: int = max(1, int(value_log_topk))
        self._value_log_counter: int = 0
        self.active_dir_log_enable: bool = bool(active_dir_log_enable)
        self.active_dir_log_every: int = max(1, int(active_dir_log_every))
        self.active_dir_log_topk: int = max(1, int(active_dir_log_topk))
        self._active_dir_log_counter: int = 0
        self.heat_jsd_use_all_neighbors: bool = bool(heat_jsd_use_all_neighbors)
        self.pixel_alpha: float = max(1e-6, float(pixel_alpha))
        self.pixel_beta: float = max(1e-6, float(pixel_beta))
        self.pixel_gamma: float = max(1e-6, float(pixel_gamma))
        self.pixel_delta: float = max(1e-6, float(pixel_delta))
        self.pixel_w_min: float = float(np.clip(float(pixel_w_min), 0.0, 1.0))
        self.pixel_reliability_use_render_entropy: bool = bool(pixel_reliability_use_render_entropy)
        self.pixel_weight_log_enable: bool = bool(pixel_weight_log_enable)
        self.pixel_weight_log_every: int = max(1, int(pixel_weight_log_every))
        self.pixel_conflict_enable: bool = bool(pixel_conflict_enable)
        self._pixel_weight_log_counter: int = 0
        if heat_s_min is None:
            self.heat_s_min: float = float(self.alpha_init * self.dir_dim + 1e-8)
        else:
            self.heat_s_min = float(heat_s_min)

        print(
            f"[SemanticVoxelMap][Config] active_dir_log={self.active_dir_log_enable} "
            f"every={self.active_dir_log_every} topk={self.active_dir_log_topk} | "
            f"heat_jsd_use_all_neighbors={self.heat_jsd_use_all_neighbors}"
        )

        # Precompute HEALPix neighbor table for adjacent pixels (P, 8)
        self._nb_table: Tensor = build_neighbor_table(
            nside=self.nside, max_nb=8, device=self.device
        )

        # Precompute direction vector of each bin (SLAM world coords, P, 3)
        # Definition: d_p is the camera-to-voxel direction, so camera lies on the -d_p side.
        bin_dirs_list = [bin_to_dir(p, self.nside) for p in range(self.n_bins)]
        self._bin_dirs: Tensor = torch.stack(bin_dirs_list, dim=0).to(  # (P, 3)
            device=self.device, dtype=self.dtype
        )

        # Voxel-level Dirichlet storing full C-class space.
        self.alpha_v: Tensor = torch.full(
            (self.num_voxels, self.n_classes),
            fill_value=self.alpha_init,
            device=self.device,
            dtype=self.dtype,
        )

        # Per-voxel directional top-k class indices for K+1 compressed-space projection.
        base_cls = torch.arange(self.dir_topk, device=self.device, dtype=torch.int16).view(1, -1)
        self.dir_cls: Tensor = base_cls.repeat(self.num_voxels, 1)

        # Mark voxels that have been initialized (observed at least once).
        self.alpha_v_active: Tensor = torch.zeros(
            self.num_voxels, device=self.device, dtype=torch.bool
        )

        # Build directional Dirichlet only for surface / exploration-boundary voxels.
        # dir_alpha[v_idx]: Tensor [P, K+1] (K explicit classes + residual)
        self.dir_alpha: Dict[int, Tensor] = {}
        # Direction observation counts: dir_obs_count[v_idx][p] is update count for that bin.
        self.dir_obs_count: Dict[int, Tensor] = {}
        # Directional conflict score z_{v,p} (optional exploration signal).
        self.dir_conflict: Dict[int, Tensor] = {}
        # Set of voxel indices with initialized directional state.
        self.directional_active_voxels: Set[int] = set()

        # Feasibility mask m_v[p], maintained only for voxels that have dir_alpha.
        self.feasible_mask: Dict[int, Tensor] = {}
        # Feasibility confidence f_v[p] in [0,1] (EMA), then mapped to m_v[p].
        self.feasible_confidence: Dict[int, Tensor] = {}

        # Track only voxels impacted by updates and recompute lazily on query.
        self._dirty_voxels: Set[int] = set()           # Voxels requiring heat/value refresh
        self._jsd_cache: Dict[int, Tensor] = {}
        self._heat_cache: Dict[int, Tensor] = {}
        self._entropy_cache: Dict[int, float] = {}
        self._value_cache: Dict[int, float] = {}
        # value term cache: v_idx -> (V, lambda_h*Hbar, lambda_u*Ehat, Hbar, Ehat)
        self._value_terms_cache: Dict[int, Tuple[float, float, float, float, float]] = {}

        # Event-driven dirty trackers.
        self._dirty_semantic_voxels: Set[int] = set()
        self._dirty_occupancy_voxels: Set[int] = set()
        # voxel -> dirty bin set; used for local heat / feasible updates.
        self._dirty_direction_bins: Dict[int, Set[int]] = {}

        # Precompute neighbors once to avoid repeatedly materializing Python lists.
        self._neighbor_bins: List[List[int]] = [
            [int(q) for q in self._nb_table[p].tolist() if q >= 0]
            for p in range(self.n_bins)
        ]

    def _project_prob_to_dir_space(self, q_full: Tensor, cls_idx: Tensor) -> Tensor:
        """Project full-class probability q_full[C] to directional compressed space [K+1]."""
        q_sel = q_full[cls_idx]                                     # select top-k class probabilities
        q_res = (q_full.sum() - q_sel.sum()).clamp(min=0.0)         # residual for non-top-k classes
        q_dir = torch.cat([q_sel, q_res.unsqueeze(0)], dim=0)
        return q_dir / (q_dir.sum() + 1e-8)

    def _remap_dir_alpha(self, alpha_old: Tensor, old_cls: Tensor, new_cls: Tensor) -> Tensor:          # not used now
        """Remap directional alpha from old top-k classes to new top-k classes while preserving total mass."""
        # Remap old directional alpha to new top-k class slots; unmatched mass goes to residual.
        alpha_new = torch.zeros_like(alpha_old)
        selected_sum = torch.zeros(alpha_old.shape[0], device=self.device, dtype=self.dtype)

        for j_new, c in enumerate(new_cls.tolist()):        # Iterate over new top-k classes
            match = torch.where(old_cls == c)[0]
            if match.numel() == 0:
                continue
            j_old = int(match[0].item())
            alpha_new[:, j_new] = alpha_old[:, j_old]
            selected_sum += alpha_old[:, j_old]

        total = alpha_old.sum(dim=-1)                       # Keep total mass unchanged; remaining mass goes to residual bucket
        alpha_new[:, -1] = (total - selected_sum).clamp(min=0.0)
        return alpha_new

    def _refresh_dir_classes(self, voxel_indices: Tensor) -> None:                                      # not used now
        """Update per-voxel directional top-k class bank using current full-class alpha_v."""
        # Refresh directional top-k class bank for selected voxels using current alpha_v values.
        if voxel_indices.numel() == 0:
            return

        alpha_sel = self.alpha_v[voxel_indices]  # (M, C)
        _, top_idx = torch.topk(alpha_sel, k=self.dir_topk, dim=-1)
        top_idx = torch.sort(top_idx, dim=-1).values

        old_idx = self.dir_cls[voxel_indices].long()
        changed = (top_idx != old_idx).any(dim=-1)
        if not bool(changed.any()):
            return

        changed_vox = voxel_indices[changed]
        new_idx_changed = top_idx[changed]
        old_idx_changed = old_idx[changed]

        for i in range(int(changed_vox.numel())):
            v = int(changed_vox[i].item())
            old_cls = old_idx_changed[i]
            new_cls = new_idx_changed[i]

            if v in self.dir_alpha:
                self.dir_alpha[v] = self._remap_dir_alpha(self.dir_alpha[v], old_cls, new_cls)
                self._mark_dirty(v, mark_heat=True)

            self.dir_cls[v] = new_cls.to(torch.int16)

    def _compute_dir_entropy_norm(self, v_idx: int) -> float:
        """Compute normalized mean entropy over active directional bins in [0,1]."""
        # entropy / log(P) to normalize to [0,1], where P is the number of bins (directions).
        log_dim = float(np.log(max(2, self.dir_dim)))
        if log_dim <= 0:
            return 0.0

        if v_idx in self.dir_alpha:
            alpha_vp = self.dir_alpha[v_idx]  # (P, K+1)        dirichlet distribution of each directional bin of voxel v_idx
            pbar_vp = self.dirichlet_mean(alpha_vp)             # expected distribution
            ent_vp = entropy(pbar_vp) / log_dim  # (P,)         # normalized

            obs = self.dir_obs_count.get(v_idx)
            if obs is not None:
                active = obs > 0
                if bool(active.any()):
                    return float(ent_vp[active].mean().item())

            return float(ent_vp.mean().item())

        cls_idx = self.dir_cls[v_idx].long()
        alpha_full = self.alpha_v[v_idx]
        alpha_sel = alpha_full[cls_idx]                         # top-k selected alpha values for directional projection
        alpha_res = (alpha_full.sum() - alpha_sel.sum()).clamp(min=0.0)
        alpha_dir = torch.cat([alpha_sel, alpha_res.unsqueeze(0)], dim=0)
        pbar_dir = self.dirichlet_mean(alpha_dir)               # projected distribution in directional compressed space
        ent = float(entropy(pbar_dir.unsqueeze(0)).item())      # distribution entropy in directional compressed space
        return float(np.clip(ent / log_dim, 0.0, 1.0))

    def _compute_pixel_weight_terms(
        self,
        probs_obs: Tensor,
        probs_rend: Tensor,
        silhouette: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute per-pixel confidence/agreement/reliability/weight/conflict terms."""
        probs_obs = normalize(probs_obs)                        # semantic distribution from observation OneFormer
        probs_rend = normalize(probs_rend)                      # semantic distribution from rendering

        log_c = float(np.log(max(2, self.n_classes)))
        c_obs = (1.0 - entropy(probs_obs) / log_c).clamp(0.0, 1.0)      # observation confidence

        jsd_u = jsd(probs_obs, probs_rend).clamp(0.0, float(np.log(2.0)))       # distribution agreement (JSD) between observation and rendering
        a_u = (1.0 - jsd_u / float(np.log(2.0))).clamp(0.0, 1.0)                # similarity

        s_u = silhouette.to(device=self.device, dtype=self.dtype).clamp(0.0, 1.0)       # silhouette
        r_u = s_u.pow(self.pixel_gamma)
        if self.pixel_reliability_use_render_entropy:                           # rendered entropy as reliability signal: low entropy (high confidence) should increase reliability weight, while high entropy (low confidence) should decrease reliability weight
            c_rend = (1.0 - entropy(probs_rend) / log_c).clamp(0.0, 1.0)
            r_u = r_u * c_rend.pow(self.pixel_delta)
        r_u = r_u.clamp(0.0, 1.0)

        c_term = c_obs.pow(self.pixel_alpha)
        a_term = a_u.pow(self.pixel_beta)
        omega_u = c_term * ((1.0 - r_u) + r_u * (self.pixel_w_min + (1.0 - self.pixel_w_min) * a_term))     # pixel weight
        omega_u = omega_u.clamp(min=0.0)

        conflict_u = c_term * r_u * (1.0 - a_u)                                 # conflict score for exploration: high confidence + high reliability + low agreement should yield high conflict score, indicating potential for informative observation if explored
        return c_obs, a_u, r_u, omega_u, conflict_u                             # omega_u is pixel weight conflict not used now

    # -------------------------
    # Public API
    # -------------------------

    @torch.no_grad()
    def update_from_frame(
        self,
        points_world: Tensor,  # (N,3)
        probs: Tensor,         # (N,C) semantic distribution per point (should sum to 1)
        cam_pos_world: Tensor, # (3,)
        probs_render: Optional[Tensor] = None,  # (N,C) rendered semantic distribution per point
        silhouette: Optional[Tensor] = None,    # (N,) rendered silhouette / alpha per point
        update_directional: bool = True,
        invalidate_cache: bool = True,
    ) -> VoxelUpdateStats:
        """Update voxel and directional Dirichlet states from one frame.

        Current variant:
        - Uses confidence-weighted aggregation inside each group.
        - Applies additive updates to ``alpha_v`` and ``dir_alpha``.
        """
        assert points_world.ndim == 2 and points_world.shape[-1] == 3
        assert probs.ndim == 2 and probs.shape[0] == points_world.shape[0]
        assert probs.shape[1] == self.n_classes
        assert cam_pos_world.shape[-1] == 3

        points_world  = points_world.to(device=self.device, dtype=self.dtype)
        probs         = probs.to(device=self.device, dtype=self.dtype)
        cam_pos_world = cam_pos_world.to(device=self.device, dtype=self.dtype)

        probs = normalize(probs)

        v_idx, inb = self.voxelize_points(points_world, return_mask=True)   # return voxel index and in-bounds mask
        if inb.sum().item() == 0:
            return VoxelUpdateStats(0, 0, 0, 0.0)

        v_idx    = v_idx[inb]
        probs_in = probs[inb]
        pts_in   = points_world[inb]
        use_pixel_weight = (probs_render is not None) and (silhouette is not None)          # check if pixel-level weighting can be applied
        probs_rend_in = None
        sil_in = None
        if use_pixel_weight:
            probs_rend = probs_render.to(device=self.device, dtype=self.dtype)              # rendered probabilities for pixel-level weight computation
            sil = silhouette.to(device=self.device, dtype=self.dtype)                       # rendered silhouette for pixel-level weight computation
            # check if rendered inputs are valid for pixel weight computation; if not, fall back to no pixel weight
            if probs_rend.ndim != 2 or probs_rend.shape[0] != points_world.shape[0] or probs_rend.shape[1] != self.n_classes:
                use_pixel_weight = False
            elif sil.ndim != 1 or sil.shape[0] != points_world.shape[0]:
                use_pixel_weight = False
            else:
                probs_rend_in = probs_rend[inb]
                sil_in = sil[inb]

        # Point confidence from semantic entropy.
        entropy_u = entropy(probs_in)
        rho_u = (1.0 - entropy_u / np.log(self.n_classes)).clamp(0.0, 1.0)              # confidence of observation
        omega_u = rho_u
        a_u = None
        r_u = None
        c_obs = None
        conflict_u = None           # not used now
        if use_pixel_weight:
            c_obs, a_u, r_u, omega_u, conflict_u = self._compute_pixel_weight_terms(        # currently only omega_u is used for weighting, while conflict_u can be used as an exploration signal in future extensions
                probs_obs=probs_in,
                probs_rend=probs_rend_in,
                silhouette=sil_in,
            )

        # ---- voxel-level aggregation: q_t(v) = weighted mean probs in voxel ----
        q_v, rho_sum_v, cnt_v, unique_v = self._scatter_weighted_mean_by_index(
            v_idx, probs_in, omega_u, self.num_voxels
        )
        # Voxel-level update in full C-class space not used for downstream voxel value calculation
        self.alpha_v[unique_v] = self.alpha_v[unique_v] + q_v
        self.alpha_v_active[unique_v] = True

        # Refreshing top-k class bank is intentionally disabled in this variant.

        num_voxels_updated = int(unique_v.numel())
        # mean_rho is the average weight
        mean_rho = float(omega_u.mean().item()) if omega_u.numel() > 0 else 0.0

        if invalidate_cache:
            updated_v_list = unique_v.tolist()
            for v in updated_v_list:                        # Mark updated voxels as dirty for lazy heat/value refresh on query; directional heat will also be marked dirty if voxel is newly activated (i.e., not in directional_active_voxels).
                vi = int(v)
                self._dirty_semantic_voxels.add(vi)
                self._mark_dirty(vi, mark_heat=(vi not in self.directional_active_voxels))

        num_pairs_updated = 0
        if update_directional:
            p_idx = self.direction_bin_from_points(cam_pos_world, pts_in)           # find out directional bin
            pair_flat = v_idx * self.n_bins + p_idx
            # calculate weighted mean of probs for each (voxel, bin) pair; this will be used for directional Dirichlet update if that pair is selected.
            q_vp, omega_sum_pair, cnt_pair, unique_pair = self._scatter_weighted_mean_by_index(
                pair_flat, probs_in, omega_u, self.num_voxels * self.n_bins
            )

            v_unique = unique_pair // self.n_bins           # voxel index of each unique (voxel, bin) pair
            p_unique = unique_pair % self.n_bins            # bin index of each unique (voxel, bin) pair
            if use_pixel_weight:
                w_vp = (omega_sum_pair / cnt_pair.clamp(min=1.0)).clamp(0.0, 1.0)       # mean weight
            else:
                w_vp = torch.ones_like(omega_sum_pair)

            conflict_pair_lookup: Dict[int, float] = {}         # not used now; if pixel conflict is enabled, this dict will store the sum of conflict scores for each (voxel, bin) pair, which will be added to the directional conflict score of that pair's voxel and bin during update
            if self.pixel_conflict_enable and use_pixel_weight and (conflict_u is not None):
                conflict_sum_pair, conflict_pair_unique = self._scatter_sum_by_index(
                    pair_flat, conflict_u, self.num_voxels * self.n_bins
                )
                conflict_pair_lookup = {
                    int(conflict_pair_unique[i].item()): float(conflict_sum_pair[i].item())
                    for i in range(int(conflict_pair_unique.numel()))
                }

            # Update sparse directional entries.
            for i in range(int(v_unique.numel())):
                vi = int(v_unique[i].item())
                pi = int(p_unique[i].item())
                q_i = q_vp[i]
                w_i = w_vp[i]

                if vi not in self.directional_active_voxels:            # check if directional state needs to be initialized for this voxel; if not, initialize it with current top-k class bank and Dirichlet prior
                    self.directional_active_voxels.add(vi)
                    self.dir_alpha[vi] = torch.full(
                        (self.n_bins, self.dir_dim),
                        fill_value=self.alpha_init,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    self.dir_obs_count[vi] = torch.zeros(
                        (self.n_bins,),
                        device=self.device,
                        dtype=torch.int32,
                    )
                    self.dir_conflict[vi] = torch.zeros(
                        (self.n_bins,),
                        device=self.device,
                        dtype=self.dtype,
                    )
                elif vi not in self.dir_conflict:
                    self.dir_conflict[vi] = torch.zeros(
                        (self.n_bins,),
                        device=self.device,
                        dtype=self.dtype,
                    )

                cls_idx_vi = self.dir_cls[vi].long()
                q_proj_i = self._project_prob_to_dir_space(q_i, cls_idx_vi)
                # p_old_i = self.dirichlet_mean(self.dir_alpha[vi][pi])               # previous mean distribution of that bin; this is used as a reference for weighted aggregation with current projected distribution; ablation will directly use current projected distribution without reference to previous distribution, which is equivalent to treating each update as a new observation and ignoring historical directional distribution in the updated bin
                # q_tilde_i = w_i * q_proj_i + (1.0 - w_i) * p_old_i              # reference previous probability distribution
                # q_tilde_i = q_proj_i                         # ablation: directly use current projected distribution without reference to previous distribution
                # test for directional dirichlet update with weight (evidence increase less than 1)
                update_weight = 0.3 + 0.7 * w_i
                q_tilde_i = update_weight * q_proj_i

                self.dir_alpha[vi][pi] = self.dir_alpha[vi][pi] + q_tilde_i         # update directional dirichlet
                self.dir_obs_count[vi][pi] += 1
                if self.pixel_conflict_enable and len(conflict_pair_lookup) > 0:
                    key = int(unique_pair[i].item())
                    c_add = conflict_pair_lookup.get(key, 0.0)
                    if c_add > 0.0:
                        self.dir_conflict[vi][pi] = self.dir_conflict[vi][pi] + c_add

                if invalidate_cache:
                    self._dirty_semantic_voxels.add(vi)
                    self._mark_dirty_bins(
                        vi,
                        torch.as_tensor([pi], device=self.device, dtype=torch.long),
                        include_neighbors=True,
                    )

            num_pairs_updated = int(unique_pair.numel())

            if use_pixel_weight and self.pixel_weight_log_enable:
                self._pixel_weight_log_counter += 1
                if self._pixel_weight_log_counter % self.pixel_weight_log_every == 0:
                    s_vp_vals: List[float] = []
                    h_samples: List[float] = []
                    sample_cap = min(128, int(v_unique.numel()))
                    for i in range(int(v_unique.numel())):
                        vi = int(v_unique[i].item())
                        pi = int(p_unique[i].item())
                        s_vp_vals.append(float(self.dir_alpha[vi][pi].sum().item()))
                        if i < sample_cap:
                            h_samples.append(float(self.compute_direction_heat(vi)[pi].item()))

                    if len(s_vp_vals) > 0:
                        s_np = np.asarray(s_vp_vals, dtype=np.float32)
                        s_mean = float(s_np.mean())
                        s_p50 = float(np.percentile(s_np, 50.0))
                        s_p90 = float(np.percentile(s_np, 90.0))
                    else:
                        s_mean, s_p50, s_p90 = 0.0, 0.0, 0.0

                    if len(h_samples) > 0:
                        h_np = np.asarray(h_samples, dtype=np.float32)
                        h_zero_ratio = float((np.abs(h_np) <= 1e-8).mean())
                    else:
                        h_zero_ratio = 1.0

                    print(
                        f"[SemanticVoxelMap][PixelWeight] call={self._pixel_weight_log_counter} "
                        f"mean_omega={float(omega_u.mean().item()):.4f} "
                        f"mean_a={float(a_u.mean().item()):.4f} "
                        f"mean_s={float(sil_in.mean().item()):.4f} "
                        f"mean_w={float(w_vp.mean().item()):.4f} "
                        f"S_vp(mean/p50/p90)=({s_mean:.3f}/{s_p50:.3f}/{s_p90:.3f}) "
                        f"H_zero_ratio(sample)={h_zero_ratio:.3f}"
                    )

        return VoxelUpdateStats(
            num_points_in_bounds=int(inb.sum().item()),
            num_voxels_updated=num_voxels_updated,
            num_voxel_dir_pairs_updated=num_pairs_updated,
            mean_rho=mean_rho,
        )

    def _mark_dirty(self, v_idx: int, mark_heat: bool = False) -> None:
        """Mark a voxel as dirty for cache invalidation.

        Args:
            v_idx: voxel index.
            mark_heat: if True, mark all bins dirty for local directional heat refresh.
        """
        self._dirty_voxels.add(v_idx)
        self._jsd_cache.pop(v_idx, None)
        self._entropy_cache.pop(v_idx, None)
        self._value_cache.pop(v_idx, None)
        self._value_terms_cache.pop(v_idx, None)
        if mark_heat:
            self._dirty_direction_bins[v_idx] = set(range(self.n_bins))

    def _mark_dirty_bins(self, v_idx: int, bins: Tensor, include_neighbors: bool = True) -> None:
        """Mark specific direction bins (and optionally neighbors) as dirty."""
        if bins.numel() == 0:
            return

        dirty_set = self._dirty_direction_bins.setdefault(v_idx, set())
        for p in bins.detach().cpu().long().tolist():
            if p < 0 or p >= self.n_bins:
                continue
            dirty_set.add(int(p))
            if include_neighbors:
                for q in self._neighbor_bins[int(p)]:
                    dirty_set.add(int(q))

        # Entropy/value must refresh when directional terms change.
        self._mark_dirty(v_idx, mark_heat=False)

    @torch.no_grad()
    def mark_occupancy_dirty(
        self,
        occ_voxels: Tensor,
        occ_origin: Tensor,
        occ_voxel_size: float,
        sim2slam: Tensor,
        radius: int = 1,
    ) -> int:
        """Mark semantic voxels affected by occupancy changes as dirty.

        Args:
            occ_voxels: (N, 3) occupancy-grid indices in simulator voxel coordinates.
            occ_origin: (3,) occupancy-grid origin in simulator world coordinates.
            occ_voxel_size: occupancy voxel size.
            sim2slam: (4,4) transform from simulator world to slam world.
            radius: index-space neighborhood expansion radius.

        Returns:
            Number of semantic voxels marked dirty.
        """
        if occ_voxels is None or occ_voxels.numel() == 0:
            return 0

        occ_idx = occ_voxels.to(device=self.device, dtype=torch.long)
        if occ_idx.ndim != 2 or occ_idx.shape[1] != 3:
            return 0

        if radius > 0:
            offsets = torch.arange(-radius, radius + 1, device=self.device, dtype=torch.long)
            ox, oy, oz = torch.meshgrid(offsets, offsets, offsets, indexing='ij')
            delta = torch.stack([ox.reshape(-1), oy.reshape(-1), oz.reshape(-1)], dim=1)
            expanded = occ_idx.unsqueeze(1) + delta.unsqueeze(0)
            occ_idx = expanded.reshape(-1, 3)

        occ_origin = occ_origin.to(device=self.device, dtype=self.dtype)
        sim2slam = sim2slam.to(device=self.device, dtype=self.dtype)

        # Convert occupancy index -> simulator world coordinate.
        pts_sim = occ_origin.view(1, 3) + occ_idx.to(self.dtype) * float(occ_voxel_size)
        n_pts = pts_sim.shape[0]
        pts_h = torch.cat([pts_sim, torch.ones((n_pts, 1), device=self.device, dtype=self.dtype)], dim=1)
        pts_slam = (sim2slam @ pts_h.T).T[:, :3]

        v_idx, inb = self.voxelize_points(pts_slam, return_mask=True)
        if inb is None or not bool(inb.any()):
            return 0

        unique_v = torch.unique(v_idx[inb])
        for v in unique_v.tolist():
            vi = int(v)
            self._dirty_occupancy_voxels.add(vi)
            # Occupancy change potentially affects all bins via feasible-mask rays.
            self._dirty_direction_bins[vi] = set(range(self.n_bins))
            self._mark_dirty(vi, mark_heat=False)

        return int(unique_v.numel())

    @torch.no_grad()
    def query_entropy(self, v_idx: int) -> float:
        """Return voxel-level entropy of pbar_v, where pbar_v = alpha_v / S_v in full C-class space."""
        if v_idx in self._entropy_cache:
            return self._entropy_cache[v_idx]
        alpha = self.alpha_v[v_idx]   # (C,)
        pbar = self.dirichlet_mean(alpha)  # (C,)
        ent = float(entropy(pbar.unsqueeze(0)).item())
        self._entropy_cache[v_idx] = ent
        return ent

    @torch.no_grad()
    def update_feasible_mask(
        self,
        occ_grid: Tensor,        # explr_map.occupancy_grid [Dx, Dy, Dz], simulator coords
        occ_origin: Tensor,      # explr_map.origin (3,), simulator coords
        occ_voxel_size: float,   # voxel size of explr_map (meters)
        slam2sim: Tensor,        # (4,4) transform from SLAM world to simulator world
        check_depth: int = 3,    # probe steps along -d_p (in SemanticVoxelMap voxel units)
        gamma: float = 0.8,      # kept for compatibility; soft mask now comes from f_v[p]
        dirty_voxels: Optional[Tensor] = None,
        dirty_bins: Optional[Dict[int, Tensor]] = None,
        full_refresh: bool = False,
        clear_semantic_dirty: bool = True,
    ) -> None:
        """Update feasibility confidence f_v[p] and mask m_v[p] for directional-active voxels.

        Per voxel v and direction bin p:
            - Take bin-center direction d_p in SLAM world and rotate it to simulator coords.
            - Probe from voxel center along -d_p for check_depth steps.
            - If any probe point is in-bounds and occ_grid <= 0 (free/unknown), set r_t=1; else r_t=0.
            - Update EMA confidence: f_t = beta*f_{t-1} + (1-beta)*r_t.
            - Convert confidence to soft mask with two thresholds:
                f >= tau_high -> 1
                f <= tau_low  -> 0
                otherwise     -> (f - tau_low) / (tau_high - tau_low)

        Args:
            occ_grid: occupancy values with convention 1=occupied, 0=unknown, -1=free.
            occ_origin: occupancy-grid origin in simulator coordinates.
            occ_voxel_size: occupancy-grid voxel size in meters.
            slam2sim: SLAM->simulator rigid transform.
            check_depth: number of reverse-direction probe steps.
            gamma: compatibility parameter (currently unused).
        """
        if not self.directional_active_voxels:
            return

        # Build target voxel set.
        if full_refresh:
            target_voxels = sorted(self.directional_active_voxels)
        else:
            target_set: Set[int] = set(self._dirty_semantic_voxels)
            target_set.update(self._dirty_occupancy_voxels)

            if dirty_voxels is not None and dirty_voxels.numel() > 0:
                target_set.update(int(v) for v in dirty_voxels.detach().cpu().long().tolist())

            target_voxels = sorted(v for v in target_set if v in self.directional_active_voxels)
            if len(target_voxels) == 0:
                return

        ext_dirty_bins: Dict[int, Set[int]] = {}
        if dirty_bins is not None:
            for key, val in dirty_bins.items():
                if val is None or val.numel() == 0:
                    continue
                ext_dirty_bins[int(key)] = set(
                    int(p) for p in val.detach().cpu().long().tolist()
                    if 0 <= int(p) < self.n_bins
                )

        occ_grid       = occ_grid.to(device=self.device)
        occ_origin     = occ_origin.to(device=self.device, dtype=self.dtype)
        slam2sim       = slam2sim.to(device=self.device, dtype=self.dtype)
        occ_vs         = float(occ_voxel_size)
        Dx, Dy, Dz     = occ_grid.shape

        # Rotate all bin direction vectors to simulator coordinates (rotation only).
        slam2sim_R = slam2sim[:3, :3]
        # _bin_dirs: (P, 3) in SLAM world; slam2sim_R @ d yields simulator direction.
        bin_dirs_sim: Tensor = (slam2sim_R @ self._bin_dirs.T).T  # (P, 3)

        # Probe-step offsets (P, check_depth, 3), along the -d_p_sim direction.
        steps = torch.arange(1, check_depth + 1, device=self.device, dtype=self.dtype)  # (K,)
        # offsets[p, k] = -(k+1) * occ_vs * bin_dirs_sim[p]  (sim coords)
        offsets = -steps.view(1, -1, 1) * occ_vs * bin_dirs_sim.unsqueeze(1)  # (P, K, 3)

        for vi in target_voxels:
            # Decide which bins to update for this voxel.
            if full_refresh:
                bins = torch.arange(self.n_bins, device=self.device, dtype=torch.long)
            else:
                bins_set: Set[int] = set()
                bins_set.update(ext_dirty_bins.get(vi, set()))
                bins_set.update(self._dirty_direction_bins.get(vi, set()))

                prev_conf = self.feasible_confidence.get(vi)
                prev_mask = self.feasible_mask.get(vi)
                if prev_conf is None or prev_mask is None or len(bins_set) == 0:
                    bins = torch.arange(self.n_bins, device=self.device, dtype=torch.long)
                else:
                    bins = torch.as_tensor(sorted(bins_set), device=self.device, dtype=torch.long)

            if bins.numel() == 0:
                continue

            # Voxel center (SLAM world -> simulator world)
            c_slam = self.get_voxel_center(vi)                        # (3,) SLAM
            c_sim = (slam2sim[:3, :3] @ c_slam) + slam2sim[:3, 3]     # (3,) sim

            offsets_sel = offsets.index_select(0, bins)

            # Query points = c_sim + offsets[p, k] -> (B, K, 3)
            query_pts = c_sim.view(1, 1, 3) + offsets_sel

            # Convert to occupancy-grid voxel indices.
            vxl_f = (query_pts - occ_origin.view(1, 1, 3)) / occ_vs  # (P, K, 3)
            vxl_i = vxl_f.floor().long()                              # (P, K, 3)

            xi = vxl_i[..., 0]  # (P, K)
            yi = vxl_i[..., 1]
            zi = vxl_i[..., 2]

            # out of bounds mask
            in_bounds = (
                (xi >= 0) & (xi < Dx) &
                (yi >= 0) & (yi < Dy) &
                (zi >= 0) & (zi < Dz)
            )  # (P, K)

            xi_safe = xi.clamp(0, Dx - 1)
            yi_safe = yi.clamp(0, Dy - 1)
            zi_safe = zi.clamp(0, Dz - 1)

            occ_vals = occ_grid[xi_safe, yi_safe, zi_safe]  # (P, K) values
            # Feasible when in-bounds AND occ <= 0 (free / unknown)
            free_hit = in_bounds & (occ_vals <= 0)           # (P, K)
            feasible = free_hit.any(dim=-1)                  # (P,) bool

            rt = feasible.to(dtype=self.dtype)
            prev_conf = self.feasible_confidence.get(vi)
            if prev_conf is None:
                conf = torch.zeros(self.n_bins, device=self.device, dtype=self.dtype)
                conf[bins] = rt
            else:
                conf = prev_conf.clone()
                conf[bins] = self.feasible_beta * prev_conf[bins] + (1.0 - self.feasible_beta) * rt

            self.feasible_confidence[vi] = conf

            # Two-threshold soft mask m_v[p]
            tau_low = self.feasible_tau_low
            tau_high = self.feasible_tau_high
            if tau_high <= tau_low:
                tau_high = tau_low + 1e-6

            new_mask_vals = torch.where(
                conf[bins] >= tau_high,
                torch.ones_like(conf[bins]),
                torch.where(
                    conf[bins] <= tau_low,
                    torch.zeros_like(conf[bins]),
                    (conf[bins] - tau_low) / (tau_high - tau_low),
                ),
            )

            prev = self.feasible_mask.get(vi)
            if prev is None:
                mask = torch.ones(self.n_bins, device=self.device, dtype=self.dtype)
                old_vals = mask[bins].clone()
            else:
                mask = prev.clone()
                old_vals = mask[bins].clone()

            mask[bins] = new_mask_vals
            self.feasible_mask[vi] = mask

            changed_bins_mask = (mask[bins] - old_vals).abs() > 1e-6
            if prev is None:
                changed_bins = bins
            elif bool(changed_bins_mask.any()):
                changed_bins = bins[changed_bins_mask]
            else:
                changed_bins = torch.empty(0, device=self.device, dtype=torch.long)

            # Consume processed feasible-bin dirty marks first.
            if vi in self._dirty_direction_bins:
                processed = set(int(p) for p in bins.detach().cpu().long().tolist())
                self._dirty_direction_bins[vi].difference_update(processed)
                if len(self._dirty_direction_bins[vi]) == 0:
                    self._dirty_direction_bins.pop(vi, None)

            # Mask changes should trigger local heat recompute for affected bins.
            if changed_bins.numel() > 0:
                self._mark_dirty_bins(vi, changed_bins, include_neighbors=True)

        if clear_semantic_dirty:
            for vi in target_voxels:
                self._dirty_semantic_voxels.discard(vi)
                self._dirty_occupancy_voxels.discard(vi)

    @torch.no_grad()
    def compute_direction_heat(self, v_idx: int) -> Tensor:
        """Compute directional heat ``H_v[p]`` with neighborhood disagreement and sparsity.

        Current formula:
            H_v[p] = m_v[p] * Dbar_vp * 1/(sqrt(S_vp + 1))
            test: H_v[p] = m_v[p] * Dbar_vp * 1/(S_vp + 1)
        """
        jsd_vals = self.compute_direction_jsd(v_idx)

        if v_idx in self.directional_active_voxels:
            alpha_vp = self.dir_alpha[v_idx]  # (P, K+1) - top-k + residual class space
            obs_count = self.dir_obs_count.get(v_idx)
        else:
            # voxel never observed from any direction -> use voxel-level full-C alpha projected to K+1 space
            cls_idx = self.dir_cls[v_idx].long()                 # (K,)
            alpha_full = self.alpha_v[v_idx]                     # (C,)
            alpha_sel = alpha_full[cls_idx]                      # (K,)
            alpha_res = (alpha_full.sum() - alpha_sel.sum()).clamp(min=0.0)  # (1,)
            alpha_dir = torch.cat([alpha_sel, alpha_res.unsqueeze(0)], dim=0)  # (K+1,)
            alpha_vp = alpha_dir.unsqueeze(0).expand(self.n_bins, -1)         # (P, K+1)
            obs_count = None

        # Determine bins to recompute.
        if v_idx not in self._heat_cache:
            H = torch.zeros((self.n_bins,), device=self.device, dtype=self.dtype)
            bins_to_update = torch.arange(self.n_bins, device=self.device, dtype=torch.long)
        else:
            H = self._heat_cache[v_idx].clone()
            dirty_set = self._dirty_direction_bins.get(v_idx)
            if dirty_set is None or len(dirty_set) == 0:
                return H
            bins_to_update = torch.as_tensor(sorted(dirty_set), device=self.device, dtype=torch.long)

        Svp = alpha_vp.sum(dim=-1)             # (P,) Dirichlet strength for each (voxel, bin)

        # feasibility mask (sparse dict, default=all feasible)
        if v_idx in self.feasible_mask:
            m = self.feasible_mask[v_idx].to(dtype=self.dtype)
        else:
            m = torch.ones(self.n_bins, device=self.device, dtype=self.dtype)

        # Recompute only affected bins.
        for p in bins_to_update.tolist():
            dbar = jsd_vals[int(p)]
            H[int(p)] = m[int(p)] * (dbar * (1.0 / (Svp[int(p)] + 1.0)))                # testing the directional heat decrease
        self._heat_cache[v_idx] = H.detach().clone()

        if v_idx in self._dirty_direction_bins:
            processed = set(int(p) for p in bins_to_update.detach().cpu().long().tolist())
            self._dirty_direction_bins[v_idx].difference_update(processed)
            if len(self._dirty_direction_bins[v_idx]) == 0:
                self._dirty_direction_bins.pop(v_idx, None)

        if v_idx not in self._dirty_direction_bins:
            self._dirty_voxels.discard(v_idx)

        return H

    @torch.no_grad()
    def compute_direction_jsd(self, v_idx: int) -> Tensor:
        """Compute raw per-bin directional disagreement before feasibility/strength attenuation.

        Returns:
            Tensor[P] where each entry is the mean JSD between bin ``p`` and its
            eligible neighbor bins. This is the raw disagreement term used by
            ``compute_direction_heat`` before multiplying by the feasible mask
            and inverse-strength factor.
        """
        if v_idx in self.directional_active_voxels:
            alpha_vp = self.dir_alpha[v_idx]
            obs_count = self.dir_obs_count.get(v_idx)
        else:
            cls_idx = self.dir_cls[v_idx].long()
            alpha_full = self.alpha_v[v_idx]
            alpha_sel = alpha_full[cls_idx]
            alpha_res = (alpha_full.sum() - alpha_sel.sum()).clamp(min=0.0)
            alpha_dir = torch.cat([alpha_sel, alpha_res.unsqueeze(0)], dim=0)
            alpha_vp = alpha_dir.unsqueeze(0).expand(self.n_bins, -1)
            obs_count = None

        dirty_set = self._dirty_direction_bins.get(v_idx)
        if v_idx not in self._jsd_cache:
            jsd_tensor = torch.zeros((self.n_bins,), device=self.device, dtype=self.dtype)
            bins_to_update = torch.arange(self.n_bins, device=self.device, dtype=torch.long)
        else:
            jsd_tensor = self._jsd_cache[v_idx].clone()
            if dirty_set is None or len(dirty_set) == 0:
                return jsd_tensor
            bins_to_update = torch.as_tensor(sorted(dirty_set), device=self.device, dtype=torch.long)

        Svp = alpha_vp.sum(dim=-1)
        pbar_vp = self.dirichlet_mean(alpha_vp)

        valid_dir = Svp >= self.heat_s_min
        if (obs_count is not None) and (not self.heat_jsd_use_all_neighbors):
            valid_dir = valid_dir & (obs_count > 0)

        for p in bins_to_update.tolist():
            if not bool(valid_dir[int(p)]):
                jsd_tensor[int(p)] = 0.0
                continue

            nbs = self._neighbor_bins[int(p)]
            if len(nbs) == 0:
                jsd_tensor[int(p)] = 0.0
                continue
            nb_idx = torch.as_tensor(nbs, device=self.device, dtype=torch.long)
            if self.heat_jsd_use_all_neighbors:
                nb_eval = nb_idx
            else:
                nb_valid = valid_dir[nb_idx]
                if not bool(nb_valid.any()):
                    jsd_tensor[int(p)] = 0.0
                    continue
                nb_eval = nb_idx[nb_valid]

            p_distribution = pbar_vp[int(p)]
            neighbor_dists = pbar_vp[nb_eval]
            jsds = jsd(p_distribution.unsqueeze(0).expand_as(neighbor_dists), neighbor_dists)
            jsd_tensor[int(p)] = jsds.mean()

        self._jsd_cache[v_idx] = jsd_tensor.detach().clone()
        return jsd_tensor

    @torch.no_grad()
    def query_max_directional_jsd(self, v_idx: int) -> float:
        """Return the maximum raw directional JSD for one voxel."""
        jsd_tensor = self.compute_direction_jsd(v_idx)
        obs = self.dir_obs_count.get(v_idx)
        if obs is not None:
            active = obs > 0
            if bool(active.any()):
                return float(jsd_tensor[active].max().item())
        return float(jsd_tensor.max().item()) if jsd_tensor.numel() > 0 else 0.0

    @torch.no_grad()
    def query_heat(self, v_idx: int, p: int) -> float:      # Query H_v[p]; return cache if available, otherwise compute first.
        H = self.compute_direction_heat(v_idx)
        return float(H[int(p)].item())

    @torch.no_grad()
    def query_h_bar(self, v_idx: int) -> float:
        """Query the aggregated directional term Hbar_v used inside voxel value."""
        if v_idx not in self._value_terms_cache:
            _ = self.compute_voxel_value(v_idx)
        return float(self._value_terms_cache.get(v_idx, (0.0, 0.0, 0.0, 0.0, 0.0))[3])

    @torch.no_grad()
    def query_value_entropy(self, v_idx: int) -> float:
        """Query the normalized entropy term E(P_v) used inside voxel value V(v)."""
        if v_idx not in self._value_terms_cache:
            _ = self.compute_voxel_value(v_idx)
        return float(self._value_terms_cache.get(v_idx, (0.0, 0.0, 0.0, 0.0, 0.0))[4])

    @torch.no_grad()
    def compute_voxel_value(self, v_idx: int) -> float:     # Compute voxel value V(v) from semantic uncertainty and directional heat.
        """
        V(v) = lambda_h * Hbar_v + lambda_u * E(P_v)
        Hbar_v: mean heat over active directions
        E(P_v): normalized weighted mean entropy across directional bins
        """
        if v_idx in self._value_cache:                      # Return directly if cached.
            return self._value_cache[v_idx]

        alpha = self.alpha_v[v_idx]           # (C,)
        Sv = float(alpha.sum().item())
        if Sv <= 0:
            self._value_cache[v_idx] = 0.0
            self._value_terms_cache[v_idx] = (0.0, 0.0, 0.0, 0.0, 0.0)
            return 0.0

        ent_norm = self._compute_dir_entropy_norm(v_idx)

        H = self.compute_direction_heat(v_idx)  # (P,) directional heat (feasibility-aware)

        if v_idx in self.dir_obs_count:                     # Use only observed directions when available.
            active_mask = self.dir_obs_count[v_idx] > 0
            if bool(active_mask.any()):
                h_bar = float(H[active_mask].mean().item())
            else:
                h_bar = float(H.mean().item())
        else:
            h_bar = float(H.mean().item())

        h_term = self.lambda_h * h_bar
        u_term = self.lambda_u * ent_norm
        voxel_value = h_term + u_term
        self._value_cache[v_idx] = voxel_value
        self._value_terms_cache[v_idx] = (voxel_value, h_term, u_term, h_bar, ent_norm)
        return voxel_value

    @torch.no_grad()
    def _maybe_log_voxel_value_terms(self, voxel_indices: List[int]) -> None:
        """Print V(v) term breakdown for selected voxels with throttling."""
        if (not self.value_log_enable) or len(voxel_indices) == 0:
            return

        self._value_log_counter += 1
        if self._value_log_counter % self.value_log_every != 0:
            return

        show_n = min(self.value_log_topk, len(voxel_indices))
        print(
            f"[SemanticVoxelMap][VoxelValue] topk_call={self._value_log_counter} "
            f"show={show_n}/{len(voxel_indices)}"
        )
        for rank, v in enumerate(voxel_indices[:show_n], start=1):
            terms = self._value_terms_cache.get(v)
            if terms is None:
                _ = self.compute_voxel_value(v)
                terms = self._value_terms_cache.get(v, (0.0, 0.0, 0.0, 0.0, 0.0))
            voxel_value, h_term, u_term, h_bar, ent_norm = terms
            # Additional diagnostics: total voxel strength Sv, per-bin strength Svp, dir obs counts
            alpha = self.alpha_v[v]  # (C,)
            Sv = float(alpha.sum().item())

            mean_Svp = 0.0
            max_Svp = 0.0
            if v in self.dir_alpha:
                alpha_vp = self.dir_alpha[v]  # (P, K+1)
                Svp = alpha_vp.sum(dim=-1)
                mean_Svp = float(Svp.mean().item())
                max_Svp = float(Svp.max().item())

            active_bins = 0
            mean_obs_per_bin = 0.0
            obs = self.dir_obs_count.get(v)
            if obs is not None:
                active_bins = int((obs > 0).sum().item())
                mean_obs_per_bin = float(obs.float().mean().item())

            print(
                f"  [#{rank}] v={v} V={voxel_value:.6f} | "
                f"lambda_h*Hbar={self.lambda_h:.3f}*{h_bar:.6f}={h_term:.6f} | "
                f"lambda_u*Ehat={self.lambda_u:.3f}*{ent_norm:.6f}={u_term:.6f} | "
                f"Sv={Sv:.1f} meanSvp={mean_Svp:.2f} maxSvp={max_Svp:.2f} "
                f"active_bins={active_bins}/{self.n_bins} mean_obs_per_bin={mean_obs_per_bin:.2f}"
            )

    @torch.no_grad()
    def _maybe_log_active_direction_stats(self, voxel_indices: List[int]) -> None:
        """Print active-direction count statistics for debugging directional coverage."""
        if (not self.active_dir_log_enable) or len(voxel_indices) == 0:
            return

        self._active_dir_log_counter += 1
        if self._active_dir_log_counter % self.active_dir_log_every != 0:
            return

        alpha_active_n = int(self.alpha_v_active.sum().item())
        dir_active_voxels = sorted(self.directional_active_voxels)
        dir_active_n = len(dir_active_voxels)
        if dir_active_n == 0:
            print(
                f"[SemanticVoxelMap][ActiveDir] topk_call={self._active_dir_log_counter} "
                f"alpha_active={alpha_active_n} directional_active=0"
            )
            return

        counts_np = np.asarray(
            [
                int((self.dir_obs_count[v] > 0).sum().item())
                for v in dir_active_voxels
            ],
            dtype=np.float32,
        )
        ratio = float(dir_active_n / max(1, alpha_active_n))
        mean_cnt = float(counts_np.mean())
        med_cnt = float(np.median(counts_np))
        p90_cnt = float(np.percentile(counts_np, 90.0))
        min_cnt = int(counts_np.min())
        max_cnt = int(counts_np.max())
        zero_cnt = int((counts_np <= 0).sum())

        print(
            f"[SemanticVoxelMap][ActiveDir] topk_call={self._active_dir_log_counter} "
            f"alpha_active={alpha_active_n} directional_active={dir_active_n} ({ratio:.3f}) | "
            f"active_dirs/bin mean={mean_cnt:.2f}/{self.n_bins} median={med_cnt:.1f} "
            f"p90={p90_cnt:.1f} min={min_cnt} max={max_cnt} zero={zero_cnt}"
        )

        show_n = min(self.active_dir_log_topk, len(voxel_indices))
        for rank, v in enumerate(voxel_indices[:show_n], start=1):
            obs = self.dir_obs_count.get(v)
            active_dirs = int((obs > 0).sum().item()) if obs is not None else 0
            coverage = float(active_dirs / max(1, self.n_bins))
            print(
                f"  [#{rank}] v={v} active_dirs={active_dirs}/{self.n_bins} "
                f"coverage={coverage:.3f}"
            )

    @torch.no_grad()
    def topk_hot_voxels(
        self,
        k: int,
        active_only: bool = True,
        min_value: float = 0.0,
        fallback_to_alpha_active: bool = True,
        voxel_sample_count_dict: Optional[dict] = None,
        max_samples_per_voxel: int = 0,
    ) -> List[int]:
        """
        Return top-k voxel indices ranked by value (used for exploitation).
        Args:
            k: number of returned top-k voxels
            active_only: if True, search only directional-active voxels.
            min_value: keep voxels with value >= min_value.
            fallback_to_alpha_active: fallback to broader alpha-active set when candidates are not enough.
            voxel_sample_count_dict: optional dict mapping voxel_idx -> sample_count. If provided with
                max_samples_per_voxel > 0, voxels exceeding the limit are filtered out.
            max_samples_per_voxel: max samples per voxel; if >0, filter voxels with count >= this limit.
        """
        k = int(k)
        if k <= 0:
            return []

        # Primary candidate set.
        if active_only:
            cand_list = sorted(self.directional_active_voxels)
        else:
            cand_list = torch.where(self.alpha_v_active)[0].tolist()

        if len(cand_list) == 0:
            return []

        vals = torch.as_tensor(
            [self.compute_voxel_value(v) for v in cand_list],
            device=self.device,
            dtype=self.dtype,
        )

        # Filter by min_value and sample count limit
        keep = vals >= float(min_value)
        num_candidates_before = int(len(cand_list))
        if voxel_sample_count_dict is not None and max_samples_per_voxel > 0:
            for idx, v in enumerate(cand_list):
                if keep[idx]:
                    sample_count = int(voxel_sample_count_dict.get(v, 0))
                    if sample_count >= max_samples_per_voxel:
                        keep[idx] = False
            try:
                num_kept = int(torch.sum(keep).item())
            except Exception:
                num_kept = int(keep.sum().cpu().item())
            num_filtered = num_candidates_before - num_kept
            print(f"[SemanticVoxelMap][topk_filter] raw={num_candidates_before} filtered_by_sample={num_filtered} max_samples={max_samples_per_voxel}")
        
        cand_list_filtered = [cand_list[i] for i in torch.where(keep)[0].tolist()]
        vals_filtered = vals[keep]

        # Fallback to broader active set if primary set is not enough.
        if fallback_to_alpha_active and len(cand_list_filtered) < k:
            broader = torch.where(self.alpha_v_active)[0].tolist()
            chosen = set(cand_list_filtered)
            broader = [v for v in broader if v not in chosen]
            if len(broader) > 0:
                broader_vals = torch.as_tensor(
                    [self.compute_voxel_value(v) for v in broader],
                    device=self.device,
                    dtype=self.dtype,
                )
                broader_keep = broader_vals >= float(min_value)
                # Also filter by sample count in fallback set
                if voxel_sample_count_dict is not None and max_samples_per_voxel > 0:
                    for idx, v in enumerate(broader):
                        if broader_keep[idx]:
                            sample_count = int(voxel_sample_count_dict.get(v, 0))
                            if sample_count >= max_samples_per_voxel:
                                broader_keep[idx] = False
                broader_filtered = [broader[i] for i in torch.where(broader_keep)[0].tolist()]
                if len(broader_filtered) > 0:
                    cand_list_filtered.extend(broader_filtered)
                    vals_filtered = torch.cat([vals_filtered, broader_vals[broader_keep]], dim=0)

        if len(cand_list_filtered) == 0:
            return []

        topk_k = min(k, len(cand_list_filtered))
        topk_res = torch.topk(vals_filtered, k=topk_k, largest=True)
        top_voxels = [cand_list_filtered[int(i.item())] for i in topk_res.indices]
        self._maybe_log_voxel_value_terms(top_voxels)
        self._maybe_log_active_direction_stats(top_voxels)
        return top_voxels

    @torch.no_grad()
    def topk_hot_directions(
        self,
        k: int,
        active_only: bool = True,
        min_value: float = 0.0,
        fallback_to_alpha_active: bool = True,
        voxel_sample_count_dict: Optional[dict] = None,
        max_samples_per_voxel: int = 0,
        per_voxel_cap: int = 0,
        dir_weight: float = 1.0,
        voxel_weight: float = 1.0,
    ) -> List[Tuple[int, int, float]]:
        """Return global top-k ``(voxel, direction-bin, score)`` candidates.

        Ranking score:
            score(v, p) = dir_weight * H_v[p] + voxel_weight * V(v)

        Args:
            k: number of returned global direction candidates.
            active_only: if True, search only directional-active voxels.
            min_value: keep voxels with value >= min_value before expanding directions.
            fallback_to_alpha_active: fallback to broader alpha-active set when candidates are not enough.
            voxel_sample_count_dict: optional dict mapping voxel_idx -> sample_count.
            max_samples_per_voxel: skip voxels whose sample count already reaches this limit.
            per_voxel_cap: maximum number of direction bins kept per voxel before global top-k.
            dir_weight: weight of directional heat ``H_v[p]``.
            voxel_weight: weight of voxel value ``V(v)``.
        """
        k = int(k)
        if k <= 0:
            return []

        if active_only:
            cand_list = sorted(self.directional_active_voxels)
        else:
            cand_list = torch.where(self.alpha_v_active)[0].tolist()

        if len(cand_list) == 0:
            return []

        num_candidates_before = int(len(cand_list))
        num_filtered_by_sample = 0
        dir_weight = float(dir_weight)
        voxel_weight = float(voxel_weight)

        def _collect_direction_candidates(voxel_indices: List[int]) -> List[Tuple[int, int, float]]:
            nonlocal num_filtered_by_sample
            direction_candidates: List[Tuple[int, int, float]] = []

            for v in voxel_indices:
                voxel_value = float(self.compute_voxel_value(int(v)))
                if voxel_value < float(min_value):
                    continue

                if voxel_sample_count_dict is not None and max_samples_per_voxel > 0:
                    sample_count = int(voxel_sample_count_dict.get(int(v), 0))
                    if sample_count >= max_samples_per_voxel:
                        num_filtered_by_sample += 1
                        continue

                heat_scores = self.compute_direction_heat(int(v))
                dir_indices = torch.arange(self.n_bins, device=self.device, dtype=torch.long)

                obs = self.dir_obs_count.get(int(v))
                if obs is not None:
                    active_mask = obs > 0
                    if bool(active_mask.any()):
                        dir_indices = dir_indices[active_mask]
                        heat_scores = heat_scores[active_mask]

                positive_mask = heat_scores > 0
                if not bool(positive_mask.any()):
                    continue

                dir_indices = dir_indices[positive_mask]
                heat_scores = heat_scores[positive_mask]
                combined_scores = dir_weight * heat_scores + voxel_weight * voxel_value

                local_cap = int(dir_indices.numel())
                if per_voxel_cap > 0:
                    local_cap = min(local_cap, int(per_voxel_cap))
                if local_cap <= 0:
                    continue

                top_local = torch.topk(combined_scores, k=local_cap, largest=True)
                for local_i in top_local.indices.tolist():
                    p_idx = int(dir_indices[local_i].item())
                    score = float(combined_scores[local_i].item())
                    direction_candidates.append((int(v), p_idx, score))

            return direction_candidates

        dir_candidates = _collect_direction_candidates(cand_list)

        if fallback_to_alpha_active and len(dir_candidates) < k:
            broader = torch.where(self.alpha_v_active)[0].tolist()
            chosen = set(cand_list)
            broader = [int(v) for v in broader if int(v) not in chosen]
            if len(broader) > 0:
                dir_candidates.extend(_collect_direction_candidates(broader))

        if voxel_sample_count_dict is not None and max_samples_per_voxel > 0:
            print(
                f"[SemanticVoxelMap][topk_dir_filter] raw={num_candidates_before} "
                f"filtered_by_sample={num_filtered_by_sample} max_samples={max_samples_per_voxel}"
            )

        if len(dir_candidates) == 0:
            return []

        dir_candidates.sort(key=lambda item: item[2], reverse=True)
        top_dirs = dir_candidates[: min(k, len(dir_candidates))]

        selected_voxels: List[int] = []
        seen_voxels: Set[int] = set()
        for v_idx, _, _ in top_dirs:
            if v_idx in seen_voxels:
                continue
            seen_voxels.add(v_idx)
            selected_voxels.append(v_idx)

        self._maybe_log_voxel_value_terms(selected_voxels)
        self._maybe_log_active_direction_stats(selected_voxels)
        return top_dirs

    # -------------------------
    # Voxelization / geometry helpers
    # -------------------------

    @torch.no_grad()
    def voxelize_points(self, points_world: Tensor, return_mask: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        # Convert coordinates to voxel indices and perform bounds checking.
        """
        points_world: (N,3) in world coordinates
        return:
          v_idx: (N,) flattened voxel index
          in_bounds: (N,) bool mask if return_mask
        """
        xyz = points_world.to(device=self.device, dtype=self.dtype)     # xyz: (N,3)
        rel = (xyz - self.bbox_min) / self.voxel_size  # (N,3) position relative to bbox_min in voxel units
        ijk = torch.floor(rel).to(torch.int64)         # (N,3) voxel-grid ijk indices

        v_idx = self._flatten_ijk(ijk)  # voxel index flattened from ijk
        if return_mask:
            inb = (
                (ijk[:, 0] >= 0) & (ijk[:, 0] < self.grid_shape[0]) &
                (ijk[:, 1] >= 0) & (ijk[:, 1] < self.grid_shape[1]) &
                (ijk[:, 2] >= 0) & (ijk[:, 2] < self.grid_shape[2])
            )
            return v_idx, inb
        return v_idx, None

    @torch.no_grad()
    def voxel_center_world(self, v_idx: Tensor) -> Tensor:
        # Return voxel-center world coordinates for the given voxel indices.
        """
        v_idx: (...,) flattened int Tensor
        return: (...,3) voxel center in world
        """
        ijk = self._unflatten_v(v_idx)
        center = (ijk.to(self.dtype) + 0.5) * self.voxel_size + self.bbox_min
        return center

    @torch.no_grad()
    def get_voxel_center(self, v_idx: int) -> Tensor:
        """
        Alias accepting a plain int (used by per-voxel query paths such as eval_candidate_semantic_gains).
        return: (3,) voxel center in world coordinates
        """
        v_t = torch.as_tensor([v_idx], device=self.device, dtype=torch.int64)
        return self.voxel_center_world(v_t)[0]

    @torch.no_grad()
    def direction_bin_from_points(self, cam_pos_world: Tensor, points_world: Tensor) -> Tensor:
        """
        HEALPix binning using direction vectors from voxel/point toward the camera.
        Direction definition: cam_pos - point (consistent with cam_pos - vox_center in eval_candidate_semantic_gains).
        """
        d = cam_pos_world.view(1, 3) - points_world  # (N,3), voxel -> camera direction
        d = self._safe_normalize(d)
        return self.dir_to_bin(d)

    @torch.no_grad()
    def dir_to_bin(self, d_unit: Tensor) -> Tensor:
        """
        d_unit: (N, 3) unit vectors -> HEALPix pixel index (N,) int64
        Implemented via healpy using RING ordering.
        """
        return dir_to_bin_batch(d_unit, nside=self.nside)

    def neighbors(self, p: int) -> List[int]:
        """
        Return 8-way neighbors of HEALPix pixel p (healpy RING).
        """
        return self._neighbor_bins[int(p)]

    # -------------------------
    # Prob / divergence helpers
    # -------------------------

    @torch.no_grad()
    def dirichlet_mean(self, alpha: Tensor, eps: float = 1e-8) -> Tensor:       # Dirichlet mean semantic distribution from observation counts
        """
        alpha: (...,C)
        return pbar: (...,C) = alpha / sum(alpha)
        """
        S = alpha.sum(dim=-1, keepdim=True).clamp(min=eps)
        return alpha / S

    # -------------------------
    # Internal scatter helpers
    # -------------------------

    @torch.no_grad()
    def _scatter_mean_by_index(self, index: Tensor, values: Tensor, n_total: int) -> Tuple[Tensor, Tensor, Tensor]:
        # Compute group means by index: for each unique index value, average its corresponding values.
        # Used for voxel-level and voxel-bin-level observation aggregation.
        """
        index: (N,) in [0, n_total)
        values: (N,C)
        Return:
          mean: (K,C)
          count: (K,)
          unique_index: (K,)
        """
        assert index.ndim == 1
        assert values.ndim == 2 and values.shape[0] == index.shape[0]

        # unique indices
        unique_index, inv = torch.unique(index, sorted=False, return_inverse=True)
        K = unique_index.numel()
        C = values.shape[1]

        sum_vals = torch.zeros((K, C), device=self.device, dtype=self.dtype)
        cnt = torch.zeros((K,), device=self.device, dtype=self.dtype)

        sum_vals.index_add_(0, inv, values)
        ones = torch.ones((values.shape[0],), device=self.device, dtype=self.dtype)
        cnt.index_add_(0, inv, ones)

        mean = sum_vals / cnt.unsqueeze(-1).clamp(min=1.0)
        return mean, cnt, unique_index

    @torch.no_grad()
    def _scatter_weighted_mean_by_index(
        self,
        index: Tensor,
        values: Tensor,
        weights: Tensor,
        n_total: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute weighted means for groups specified by index.

        Args:
            index: (N,) in [0, n_total)
            values: (N, C)
            weights: (N,) non-negative weights (rho_t(u) in this context)

        Returns:
            weighted_mean: (K, C)
            weight_sum: (K,)
            count: (K,)                 # number of weights (for debugging / analysis)
            unique_index: (K,)          # number of unique groups
        """
        assert index.ndim == 1
        assert values.ndim == 2 and values.shape[0] == index.shape[0]
        assert weights.ndim == 1 and weights.shape[0] == index.shape[0]

        unique_index, inv = torch.unique(index, sorted=False, return_inverse=True)
        K = unique_index.numel()
        C = values.shape[1]

        w = weights.to(device=self.device, dtype=self.dtype).clamp(min=0.0)
        weighted_sum = torch.zeros((K, C), device=self.device, dtype=self.dtype)
        weight_sum = torch.zeros((K,), device=self.device, dtype=self.dtype)
        count = torch.zeros((K,), device=self.device, dtype=self.dtype)

        weighted_sum.index_add_(0, inv, values * w.unsqueeze(-1))
        weight_sum.index_add_(0, inv, w)
        count.index_add_(0, inv, torch.ones_like(w))            # number of weights

        weighted_mean = weighted_sum / (weight_sum.unsqueeze(-1) + 1e-8)        # weighted mean
        return weighted_mean, weight_sum, count, unique_index

    @torch.no_grad()
    def _scatter_sum_by_index(
        self,
        index: Tensor,
        values: Tensor,
        n_total: int,
    ) -> Tuple[Tensor, Tensor]:
        """Compute per-group sums for scalar values indexed by `index`."""
        assert index.ndim == 1
        assert values.ndim == 1 and values.shape[0] == index.shape[0]

        unique_index, inv = torch.unique(index, sorted=False, return_inverse=True)
        K = unique_index.numel()
        sums = torch.zeros((K,), device=self.device, dtype=self.dtype)
        sums.index_add_(0, inv, values.to(device=self.device, dtype=self.dtype))
        return sums, unique_index

    def _flatten_ijk(self, ijk: Tensor) -> Tensor:
        """
        ijk: (N,3) int64
        v = i*(Ny*Nz) + j*Nz + k
        """
        # Convert 3D voxel index (i,j,k) to flattened 1D index v for alpha_v / dir_alpha access.
        Ny, Nz = self.grid_shape[1], self.grid_shape[2]
        return ijk[:, 0] * (Ny * Nz) + ijk[:, 1] * Nz + ijk[:, 2]

    def _unflatten_v(self, v_idx: Tensor) -> Tensor:
        """
        v_idx: (...,) int64
        return ijk: (...,3)
        """
        # Convert flattened 1D voxel index v back to 3D (i,j,k).
        Ny, Nz = self.grid_shape[1], self.grid_shape[2]
        v = v_idx.to(torch.int64)
        i = v // (Ny * Nz)
        rem = v % (Ny * Nz)
        j = rem // Nz
        k = rem % Nz
        return torch.stack([i, j, k], dim=-1)

    @staticmethod
    def _safe_normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
        # For near-zero vectors, keep normalization numerically stable.
        n = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=eps)
        return x / n
