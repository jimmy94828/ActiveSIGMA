"""
維護 semantic voxel map 的類別，包含：
- voxel-level Dirichlet:  α_v  (NumVoxels, C)          [dense]
- directional Dirichlet:  dir_alpha[v_idx] → Tensor[P, C]  [sparse dict，只對 surface/boundary voxel]
- 更新：α ← η α + ρ q
- 熱度：H_v[p] = m_v[p] * ( w_i * mean_{q∈N(p)} JSD(p̄_{v,p}, p̄_{v,q}) + w_n * 1/(S_{v,p}+1) )

優化：
1. 稀疏 directional state：只有接收到點雲投影的 voxel（物體表面 / 探索邊界）才建立 dir_alpha 條目，
   避免對所有 voxel 分配 [NumVoxels, P, C] 的 dense tensor。
2. 精確 dirty tracking：更新後只標記受影響 voxel 為 dirty，查詢時 lazy 重算，
   不做全圖 cache 清空。

方向離散化：使用 HEALPix將觀測方向離散化
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import healpy as hp
import numpy as np
import torch
from torch import Tensor

from src.utils.healpix_utils import build_neighbor_table, bin_to_dir, dir_to_bin_batch  # HEALPix 工具
from src.utils.divergence import normalize, entropy, jsd


@dataclass
class VoxelUpdateStats:
    num_points_in_bounds: int
    num_voxels_updated: int
    num_voxel_dir_pairs_updated: int
    mean_rho: float


class SemanticVoxelMap:
    def __init__(                # 初始化設定參數
        self,
        bbox_bound: np.ndarray,  # [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
        voxel_size: float,       # 與 explr_map 相同
        n_classes: int,          # num semantic classes (C)
        nside: int = 1,          # HEALPix nside（n_bins = 12 * nside^2）
        eta: float = 0.9,        # 遺忘係數
        w_i: float = 0.5,        # JSD 熱度權重
        w_n: float = 0.5,        # 觀測稀疏懲罰權重
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        alpha_init: float = 1e-2,  # Dirichlet 初始先驗（小正數避免除0）
        topk: int = 16,          # 每個 voxel 只儲存 top-k 語意類別的 Dirichlet，節省 GPU 記憶體
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

        # Semantic + directional config（HEALPix）
        self.n_classes: int = int(n_classes)
        self.nside: int = int(nside)
        self.n_bins: int = hp.nside2npix(self.nside)    # 12 * nside^2 equal-area pixels
        self.eta: float = float(eta)                    # 遺忘係數
        self.w_i: float = float(w_i)                    # JSD 熱度權重
        self.w_n: float = float(w_n)                    # 觀測稀疏懲罰權重
        self.alpha_init: float = float(alpha_init)      # Dirichlet 初始先驗(prior)
        self.topk_classes: int = min(int(topk), int(n_classes))  # k 個追蹤類別

        # 預建 HEALPix 鄰居表包含相鄰pixels (P, 8)
        self._nb_table: Tensor = build_neighbor_table(
            nside=self.nside, max_nb=8, device=self.device
        )

        # 預建各 bin 的方向向量（SLAM world coords，P, 3）
        # 定義：d_p 是camera → voxel方向，因此 camera 位於 voxel 的 -d_p 方向
        bin_dirs_list = [bin_to_dir(p, self.nside) for p in range(self.n_bins)]
        self._bin_dirs: Tensor = torch.stack(bin_dirs_list, dim=0).to(  # (P, 3)
            device=self.device, dtype=self.dtype
        )

        # Dirichlet tensors — top-k sparse representation (節省 GPU 記憶體)
        # 每個 voxel 只儲存 k 個語意類別的 Dirichlet 參數，而非全部 C 個
        self.alpha_v_vals: Tensor = torch.full(         # voxel level Dirichlet 值 [V, k]
            (self.num_voxels, self.topk_classes),
            fill_value=self.alpha_init,
            device=self.device,
            dtype=self.dtype,
        )
        # 每個 voxel 追蹤的 k 個類別 class index（int16 節省記憶體）
        self.alpha_v_cls: Tensor = torch.zeros(
            (self.num_voxels, self.topk_classes),
            device=self.device,
            dtype=torch.int16,
        )
        # 標記哪些 voxel 已被初始化（有實際觀測 top-k class index）
        self.alpha_v_active: Tensor = torch.zeros(
            self.num_voxels, device=self.device, dtype=torch.bool
        )

        # 只對物體表面 / 探索邊界 voxel 建立方向性 Dirichlet
        # dir_alpha[v_idx]: Tensor [P, k]（與 alpha_v_cls[v_idx] 共用 class index）
        self.dir_alpha: Dict[int, Tensor] = {}
        # 已建立方向性狀態的 voxel index 集合
        self.directional_active_voxels: Set[int] = set()

        # 可行性遮罩 m_v[p]：只對有 dir_alpha 的 voxel 維護（dict[v_idx] → Tensor[P] bool）
        # 尚未建立方向性狀態的 voxel 全視為可行（heat 計算才需要）
        self.feasible_mask: Dict[int, Tensor] = {}

        # 只標記受更新影響的 voxel，查詢時重算，避免全圖 cache 清空。
        self._dirty_voxels: Set[int] = set()           # 需要重算 heat/value 的 voxel
        self._heat_cache: Dict[int, Tensor] = {}
        self._entropy_cache: Dict[int, float] = {}
        self._value_cache: Dict[int, float] = {}

    # -------------------------
    # Public API
    # -------------------------

    @torch.no_grad()
    def update_from_frame(                  # 接收一個 frame 的觀測更新 Dirichlet
        self,
        points_world: Tensor,  # (N,3)
        probs: Tensor,         # (N,C) semantic distribution per point (should sum to 1)
        cam_pos_world: Tensor, # (3,)
        update_directional: bool = True,
        invalidate_cache: bool = True,
    ) -> VoxelUpdateStats:
        """
        由一個 frame 的觀測更新 Dirichlet：
        - 先把 points -> voxel index（只有進入 bbox 的點才算 surface active voxel）
        - voxel 內聚合 probs => q_t(v)
        - rho = 1 - H(q)/log(C): entropy代表觀測信心程度
        - alpha_v[v]  = eta*alpha_v[v]  + rho*q（dense，所有 voxel）
        - dir_alpha[v][p] 同理，只對接收到點的 voxel 建立稀疏條目
        - 精確 dirty 標記: 只把被更新的 voxel 加入 _dirty_voxels
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

        # ---- voxel-level aggregation: q_t(v) = mean probs in voxel ----
        q_v, count_v, unique_v = self._scatter_mean_by_index(v_idx, probs_in, self.num_voxels)

        ent_v = entropy(q_v)  # (K,)
        rho_v = (1.0 - ent_v / np.log(self.n_classes)).clamp(0.0, 1.0)  # (K,)

        # update alpha_v — top-k sparse representation
        # 先快照哪些 voxel 在此次更新前已被初始化，再分兩批（新 / 已存在）向量化更新
        was_active = self.alpha_v_active[unique_v]   # (K,) bool

        # ---- 新 voxel（首次觀測）：用 top-k 初始化 ----
        inactive_local = torch.where(~was_active)[0]
        if inactive_local.numel() > 0:
            v_inactive   = unique_v[inactive_local]         # 實際 voxel index
            q_inactive   = q_v[inactive_local]              # (K_new, C)
            topk_vals, topk_idx = torch.topk(q_inactive, k=self.topk_classes, dim=-1)  # (K_new, k)
            sort_order   = topk_idx.argsort(dim=-1)         # 按 class index 排序，保持穩定性
            topk_idx_s   = topk_idx.gather(-1, sort_order)  # (K_new, k)
            topk_vals_s  = topk_vals.gather(-1, sort_order) # (K_new, k)
            rho_inact    = rho_v[inactive_local].unsqueeze(-1)  # (K_new, 1)
            self.alpha_v_cls[v_inactive]  = topk_idx_s.to(torch.int16)
            self.alpha_v_vals[v_inactive] = self.alpha_init + rho_inact * topk_vals_s
            self.alpha_v_active[v_inactive] = True

        # ---- 已存在 voxel：投影到已追蹤的 k 個類別後更新 ----
        active_local = torch.where(was_active)[0]
        if active_local.numel() > 0:
            v_active   = unique_v[active_local]             # 實際 voxel index
            q_active   = q_v[active_local]                  # (K_act, C)
            cls_idx    = self.alpha_v_cls[v_active].long()  # (K_act, k)
            q_proj     = q_active.gather(-1, cls_idx)       # (K_act, k) 投影到已追蹤類別
            q_proj     = q_proj / (q_proj.sum(dim=-1, keepdim=True) + 1e-8)  # 在 k-class 空間歸一化
            rho_act    = rho_v[active_local].unsqueeze(-1)  # (K_act, 1)
            self.alpha_v_vals[v_active] = (
                self.alpha_v_vals[v_active] * self.eta + rho_act * q_proj
            )

        num_voxels_updated = int(unique_v.numel())
        mean_rho = float(rho_v.mean().item()) if num_voxels_updated > 0 else 0.0

        if invalidate_cache:                # 標記觀測紀錄有改變的voxel為dirty，更新時 heat/value 需要重算
            updated_v_list = unique_v.tolist()
            for v in updated_v_list:
                self._mark_dirty(v)

        num_pairs_updated = 0
        if update_directional:              # 更新方向性 Dirichlet（只對有點投影的 voxel）
            p_idx    = self.direction_bin_from_points(cam_pos_world, pts_in)  # (N_in,) return bin index for each point
            pair_flat = v_idx * self.n_bins + p_idx  # (N_in,)  flattened index for (voxel, bin) pairs
            q_vp, count_vp, unique_pair = self._scatter_mean_by_index(
                pair_flat, probs_in, self.num_voxels * self.n_bins
            )

            v_unique = unique_pair // self.n_bins   # voxel indices for each (v,p) pair
            p_unique = unique_pair % self.n_bins    # bin  indices for each (v,p) pair

            rho_lookup = torch.zeros((self.num_voxels,), device=self.device, dtype=self.dtype)
            rho_lookup[unique_v] = rho_v            # 建立 voxel index → rho 的查詢表，方便後續更新 dir_alpha
            rho_pair = rho_lookup[v_unique].clamp(0.0, 1.0)

            # Update sparse dir_alpha entries — allocate (P, k) instead of (P, C)
            for i in range(int(v_unique.numel())):
                vi = int(v_unique[i].item())        # voxel index
                pi = int(p_unique[i].item())        # HEALPix bin index
                q_i   = q_vp[i]             # (C,) full-class distribution for this (voxel, bin) pair
                rho_i = float(rho_pair[i].item())

                # Promote voxel to directional active if not yet (allocate on first encounter)
                if vi not in self.directional_active_voxels:
                    self.directional_active_voxels.add(vi)
                    self.dir_alpha[vi] = torch.full(
                        (self.n_bins, self.topk_classes),  # (P, k) — 只存 top-k 類別
                        fill_value=self.alpha_init,
                        device=self.device,
                        dtype=self.dtype,
                    )

                # 投影 q_i 到該 voxel 的已追蹤 k 類別空間
                cls_idx_vi = self.alpha_v_cls[vi].long()   # (k,)
                q_proj_i   = q_i[cls_idx_vi]               # (k,) 投影到追蹤類別
                q_proj_i   = q_proj_i / (q_proj_i.sum() + 1e-8)

                self.dir_alpha[vi][pi] = (              # update directional Dirichlet for this (voxel, bin) pair
                    self.dir_alpha[vi][pi] * self.eta
                    + rho_i * q_proj_i
                )

            num_pairs_updated = int(unique_pair.numel())

        return VoxelUpdateStats(
            num_points_in_bounds=int(inb.sum().item()),
            num_voxels_updated=num_voxels_updated,
            num_voxel_dir_pairs_updated=num_pairs_updated,
            mean_rho=mean_rho,
        )

    def _mark_dirty(self, v_idx: int) -> None:
        """Mark a voxel as needing heat/value recomputation (precise dirty tracking)."""
        self._dirty_voxels.add(v_idx)
        self._heat_cache.pop(v_idx, None)
        self._entropy_cache.pop(v_idx, None)
        self._value_cache.pop(v_idx, None)

    @torch.no_grad()
    def query_entropy(self, v_idx: int) -> float:
        """回傳 voxel-level entropy( p̄_v )，p̄_v = alpha_v_vals / S_v（在 k-class 空間計算）"""
        if v_idx in self._entropy_cache:
            return self._entropy_cache[v_idx]
        alpha = self.alpha_v_vals[v_idx]   # (k,)
        pbar = self.dirichlet_mean(alpha)  # (k,)
        ent = float(entropy(pbar.unsqueeze(0)).item())
        self._entropy_cache[v_idx] = ent
        return ent

    @torch.no_grad()
    def update_feasible_mask(
        self,
        occ_grid: Tensor,        # explr_map.occupancy_grid [Dx, Dy, Dz]，sim coords
        occ_origin: Tensor,      # explr_map.origin (3,)，sim coords
        occ_voxel_size: float,   # explr_map 的 voxel size（公尺）
        slam2sim_R: Tensor,      # (3,3) 旋轉部分，SLAM world 方向 → sim 方向
        check_depth: int = 3,    # 沿 -d_p 方向探查的步數（單位：SemanticVoxelMap voxel）
        gamma: float = 0.8,      # 不可行方向的遮罩值（0=完全屏蔽，<1=降權）
    ) -> None:
        """
        對所有 directional_active_voxels 計算可行性遮罩 m_v[p]

        判斷邏輯（每個 voxel v，每個方向 bin p）：
          - 取 bin p 的中心方向向量 d_p（SLAM world），轉換到 sim 座標系得 d_p_sim
          - 從 voxel 中心 c_v（SLAM world → sim）沿 -d_p_sim 方向，
            以 step = occ_voxel_size 為步長，走 check_depth 步
          - 若任一步落在 occ_grid 範圍內且 occ_grid ≤ 0（free/unknown），則方向可行 → m=1
          - 全程均為 occupied（occ_grid = 1）或 out of bounds，則不可行 m=gamma

        Args:
            occ_grid      : explr_map.occupancy_grid，值：1=occupied, 0=unknown, -1=free
            occ_origin    : explr_map.origin，sim 座標中 occupancy grid 的原點 (3,)
            occ_voxel_size: explr_map voxel 大小（公尺）
            slam2sim_R    : (3,3) 方向旋轉矩陣（SLAM → sim，只用旋轉部分換方向向量用）
            check_depth   : 沿觀察方向反向探查的步數
            gamma         : 不可行方向的遮罩值
        """
        if not self.directional_active_voxels:
            return

        occ_grid       = occ_grid.to(device=self.device)
        occ_origin     = occ_origin.to(device=self.device, dtype=self.dtype)
        slam2sim_R     = slam2sim_R.to(device=self.device, dtype=self.dtype)
        occ_vs         = float(occ_voxel_size)
        Dx, Dy, Dz     = occ_grid.shape

        # 把所有 bin 方向向量轉到 sim 座標系（只旋轉，不平移）
        # _bin_dirs: (P, 3) in SLAM world；slam2sim_R @ d 得 sim 方向
        bin_dirs_sim: Tensor = (slam2sim_R @ self._bin_dirs.T).T  # (P, 3)

        # 探查步長向量 (P, check_depth, 3)，沿 -d_p_sim 方向
        steps = torch.arange(1, check_depth + 1, device=self.device, dtype=self.dtype)  # (K,)
        # offsets[p, k] = -(k+1) * occ_vs * bin_dirs_sim[p]  (sim coords)
        offsets = -steps.view(1, -1, 1) * occ_vs * bin_dirs_sim.unsqueeze(1)  # (P, K, 3)

        active_list = list(self.directional_active_voxels)
        for vi in active_list:
            # voxel 中心（SLAM world → sim）
            c_slam = self.get_voxel_center(vi)                        # (3,) SLAM
            c_sim = slam2sim_R @ c_slam                               # (3,) sim

            # 探查點 = c_sim + offsets[p, k]  → (P, K, 3)
            query_pts = c_sim.view(1, 1, 3) + offsets                 # (P, K, 3)

            # 轉換成 occ_grid 的 voxel 索引
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
            # 可行 = in_bounds AND occ <= 0（free / unknown）
            free_hit = in_bounds & (occ_vals <= 0)           # (P, K)
            feasible = free_hit.any(dim=-1)                  # (P,) bool

            mask = torch.where(feasible,
                               torch.ones(self.n_bins, device=self.device, dtype=self.dtype),
                               torch.full((self.n_bins,), gamma, device=self.device, dtype=self.dtype))

            prev = self.feasible_mask.get(vi)                # 只有當遮罩值有變化時才標記 dirty，避免不必要的 heat 重算
            if prev is None or not torch.equal(prev, mask):
                self.feasible_mask[vi] = mask
                self._mark_dirty(vi)   # heat cache 需要重算

    @torch.no_grad()
    def compute_direction_heat(self, v_idx: int) -> Tensor:
        """
        計算 H_v[p]（shape: [P]）方向性熱度，結合不一致性和稀疏性：
        H_v[p]=m_v[p] * ( w_i * mean_{q∈N(p)} JSD(p̄_{v,p}, p̄_{v,q}) + w_n * 1/(S_{v,p}+1) )
        """
        if v_idx in self._heat_cache:
            return self._heat_cache[v_idx].clone()

        if v_idx in self.directional_active_voxels:
            alpha_vp = self.dir_alpha[v_idx]  # (P, k)  — top-k class space
        else:
            # voxel never observed from any direction → use voxel-level prior for all bins
            alpha_vp = self.alpha_v_vals[v_idx].unsqueeze(0).expand(self.n_bins, -1)  # (P, k)

        Svp    = alpha_vp.sum(dim=-1)             # (P,) Dirichlet strength for each (voxel, bin)
        pbar_vp = self.dirichlet_mean(alpha_vp)   # (P,C) mean semantic distribution for each (voxel, bin)

        # inconsistency term: mean neighbor JSD
        jsd_term = torch.zeros((self.n_bins,), device=self.device, dtype=self.dtype)
        for p in range(self.n_bins):
            nbs = self.neighbors(p)
            if len(nbs) == 0:
                continue
            p_distribution  = pbar_vp[p]   # (C,)
            neighbor_dists = pbar_vp[torch.as_tensor(nbs, device=self.device, dtype=torch.long)]
            jsds = jsd(p_distribution.unsqueeze(0).expand_as(neighbor_dists), neighbor_dists)
            jsd_term[p] = jsds.mean()

        sparsity_term = 1.0 / (Svp + 1.0)

        H = self.w_i * jsd_term + self.w_n * sparsity_term

        # feasibility mask (sparse dict, default=all feasible)
        if v_idx in self.feasible_mask:
            m = self.feasible_mask[v_idx].to(dtype=self.dtype)
        else:
            m = torch.ones(self.n_bins, device=self.device, dtype=self.dtype)
        H = H * m           # mask out infeasible directions of the voxel

        self._heat_cache[v_idx] = H.detach().clone()        # cache the computed heat for this voxel index
        self._dirty_voxels.discard(v_idx)                   # heat 已更新，移除 dirty 標記
        return H

    @torch.no_grad()
    def query_heat(self, v_idx: int, p: int) -> float:      # 查詢特定 voxel v_idx 和 bin p 的熱度值 H_v[p]，如果已 cache 就直接回傳，否則計算後回傳
        H = self.compute_direction_heat(v_idx)
        return float(H[int(p)].item())

    @torch.no_grad()
    def compute_voxel_value(self, v_idx: int) -> float:     # 計算voxel的整體價值 V(v)，結合語意不確定性和方向熱度：
        """
        V(v) = (Entropy(p̄_v)/S_v + 1.0) * Σ_p H_v[p]
        """
        if v_idx in self._value_cache:                      # 計算過且 cache 中有值，直接回傳
            return self._value_cache[v_idx]

        alpha = self.alpha_v_vals[v_idx]           # (k,)
        Sv = float(alpha.sum().item())
        if Sv <= 0:
            self._value_cache[v_idx] = 0.0
            return 0.0

        pbar = self.dirichlet_mean(alpha)     # (k,)        # voxel平均語意分布（k-class空間）
        ent = float(entropy(pbar.unsqueeze(0)).item())

        H = self.compute_direction_heat(v_idx)  # (P,)      # 方向熱度（已考慮可行性遮罩）
        voxel_value = (ent / Sv + 1.0) * float(H.sum().item())
        self._value_cache[v_idx] = voxel_value
        return voxel_value

    @torch.no_grad()
    def topk_hot_voxels(self, k: int, active_only: bool = True) -> List[int]:
        """
        回傳 value 最高的 k 個 voxel index（exploitation 用）。
        向量化批次計算 V(v) = (H_ent / S_v + 1.0) * ΣH_v[p]。

        Args:
            k: 返回的 top-k 數量
            active_only: 若 True，只掃 directional_active_voxels。
        """
        k = int(k)
        if k <= 0:
            return []
        # --- 找要掃描的 voxel 集合 ---
        if active_only:
            cand_list = sorted(self.directional_active_voxels)
        else:
            # 所有觀測過語義的 voxel（用 alpha_v_active 判斷，比 sum 快）
            cand_list = torch.where(self.alpha_v_active)[0].tolist()

        if not cand_list:
            return []

        cand_t = torch.tensor(cand_list, dtype=torch.long, device=self.device)  # (M,)  # M = number of candidate voxels to evaluate
        M = len(cand_list)

        # --- voxel-level entropy term（在 k-class 空間計算）---
        alpha_cand = self.alpha_v_vals[cand_t]              # (M, k)
        Sv_cand = alpha_cand.sum(dim=-1).clamp(min=1e-8)   # (M,)
        pbar_cand = alpha_cand / Sv_cand.unsqueeze(-1)      # (M, k)    candidate voxels 的平均語意分布
        ent_cand = -(torch.xlogy(pbar_cand, pbar_cand.clamp(min=1e-9))).sum(dim=-1)  # (M,)

        # --- 批次組裝 alpha_vp_cand (M, P, k)，從稀疏 dir_alpha dict ---
        # 對 active_only 路徑，所有 cand_list 均在 directional_active_voxels 中
        prior_row = self.alpha_init * torch.ones(
            self.n_bins, self.topk_classes, device=self.device, dtype=self.dtype)
        alpha_vp_cand = torch.stack(
            [self.dir_alpha[v] if v in self.dir_alpha else prior_row for v in cand_list],
            dim=0)  # (M, P, k)

        Svp_cand = alpha_vp_cand.sum(dim=-1)                # (M, P)
        pbar_vp_cand = alpha_vp_cand / Svp_cand.unsqueeze(-1).clamp(min=1e-8)  # (M, P, C)

        # sparsity term
        sparsity = self.w_n / (Svp_cand + 1.0)             # (M, P)

        # JSD inconsistency term — compute per-neighbor to reduce peak memory
        P = self.n_bins
        nb_table = self._nb_table                            # (P, 8)
        valid_nb = (nb_table >= 0)                          # (P, 8)
        nb_safe = nb_table.clamp(min=0)                     # (P, 8)

        # Accumulate per-neighbor JSD without building (M, P, 8, C) tensors
        jsd_sum = torch.zeros((M, P), device=self.device, dtype=self.dtype)
        nb_count = torch.zeros((M, P), device=self.device, dtype=self.dtype)

        # Iterate over neighbor slots (typically 8)
        for nb_k in range(nb_safe.shape[1]):
            nb_idx = nb_safe[:, nb_k]                       # (P,)
            # gather neighbor distributions -> (M, P, C)  calculate JSD between pbar_vp_cand and its neighbors
            pbar_nb_k = pbar_vp_cand[:, nb_idx, :]
            # mixture distribution
            m_k = 0.5 * (pbar_vp_cand + pbar_nb_k)
            # KL terms per-class then sum over classes -> (M, P)
            kl_p = (torch.xlogy(pbar_vp_cand, pbar_vp_cand) -
                torch.xlogy(pbar_vp_cand, m_k.clamp(min=1e-9))).sum(dim=-1)
            kl_nb = (torch.xlogy(pbar_nb_k, pbar_nb_k) -
                 torch.xlogy(pbar_nb_k, m_k.clamp(min=1e-9))).sum(dim=-1)
            jsd_k = (0.5 * (kl_p + kl_nb)).clamp(min=0.0)   # (M, P)

            # apply valid neighbor mask for this neighbor index
            mask_k = valid_nb[:, nb_k].unsqueeze(0).expand(M, -1)  # (M, P)
            jsd_sum += jsd_k * mask_k.float()
            nb_count += mask_k.float()

        nb_count = nb_count.clamp(min=1.0)
        jsd_mean = jsd_sum / nb_count                       # (M, P)

        # feasibility mask
        ones_p = torch.ones(P, device=self.device, dtype=self.dtype)
        m_mask = torch.stack(
            [self.feasible_mask[v].to(dtype=self.dtype) if v in self.feasible_mask
             else ones_p for v in cand_list],
            dim=0)  # (M, P)

        H_sum = (self.w_i * jsd_mean * m_mask + sparsity * m_mask).sum(dim=-1)              # (M,)

        vals = (ent_cand / Sv_cand + 1.0) * H_sum                 # (M,)

        topk_k = min(k, M)
        topk_res = torch.topk(vals, k=topk_k, largest=True)
        return [cand_list[int(i.item())] for i in topk_res.indices]

    # -------------------------
    # Voxelization / geometry helpers
    # -------------------------

    @torch.no_grad()
    def voxelize_points(self, points_world: Tensor, return_mask: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        # 把座標值轉換成voxel的index並進行邊界檢查
        """
        points_world: (N,3) in world coordinates
        return:
          v_idx: (N,) flattened voxel index
          in_bounds: (N,) bool mask if return_mask
        """
        xyz = points_world.to(device=self.device, dtype=self.dtype)     # xyz: (N,3)
        rel = (xyz - self.bbox_min) / self.voxel_size  # (N,3)  相對於 bbox_min 的相對位置，單位為 voxel size
        ijk = torch.floor(rel).to(torch.int64)         # (N,3)  轉換成 voxel grid 的 ijk index

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
        # 給定 voxel index，回傳 voxel 中心的世界座標
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
        Alias accepting a plain int（供 eval_candidate_semantic_gains 等逐 voxel 查詢用）。
        return: (3,) voxel center in world coordinates
        """
        v_t = torch.as_tensor([v_idx], device=self.device, dtype=torch.int64)
        return self.voxel_center_world(v_t)[0]

    @torch.no_grad()
    def direction_bin_from_points(self, cam_pos_world: Tensor, points_world: Tensor) -> Tensor:
        """
        用「從 voxel/point 看向 camera」的方向向量做 HEALPix binning，回傳 p_idx: (N,)
        方向定義：cam_pos - point（與 eval_candidate_semantic_gains 中的 cam_pos - vox_center 一致）
        """
        d = cam_pos_world.view(1, 3) - points_world  # (N,3)，voxel → camera 方向
        d = self._safe_normalize(d)
        return self.dir_to_bin(d)

    @torch.no_grad()
    def dir_to_bin(self, d_unit: Tensor) -> Tensor:
        """
        d_unit: (N, 3) 單位向量 → HEALPix pixel index (N,) int64
        透過 healpy 實現（RING 排序）。
        """
        return dir_to_bin_batch(d_unit, nside=self.nside)

    def neighbors(self, p: int) -> List[int]:
        """
        回傳 HEALPix pixel p 的 8-way 鄰居（healpy RING）
        """
        row = self._nb_table[int(p)].tolist()
        return [int(q) for q in row if q >= 0]

    # -------------------------
    # Prob / divergence helpers
    # -------------------------

    @torch.no_grad()
    def dirichlet_mean(self, alpha: Tensor, eps: float = 1e-8) -> Tensor:       # Dirichlet 觀測紀錄語意分布
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
        # 計算 index 指定的 group mean：對於 index 中的每個 unique value，計算對應 values 的平均
        # 用於 voxel-level 和 voxel-bin level 的觀測聚合（update_from_frame 中的 q_v 和 q_vp 計算）
        # q_v: 由觀測到的點雲在 voxel 中的平均語意分布；q_vp: 由觀測到的點雲在 voxel-bin 中的平均語意分布
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

    def _flatten_ijk(self, ijk: Tensor) -> Tensor:
        """
        ijk: (N,3) int64
        v = i*(Ny*Nz) + j*Nz + k
        """
        # 把三維 voxel index (i,j,k) 轉換成一維的 flattened index v，方便存取 alpha_v 和 dir_alpha
        Ny, Nz = self.grid_shape[1], self.grid_shape[2]
        return ijk[:, 0] * (Ny * Nz) + ijk[:, 1] * Nz + ijk[:, 2]

    def _unflatten_v(self, v_idx: Tensor) -> Tensor:
        """
        v_idx: (...,) int64
        return ijk: (...,3)
        """
        # 把一維的 voxel index v 轉換回三維的 (i,j,k) index
        Ny, Nz = self.grid_shape[1], self.grid_shape[2]
        v = v_idx.to(torch.int64)
        i = v // (Ny * Nz)
        rem = v % (Ny * Nz)
        j = rem // Nz
        k = rem % Nz
        return torch.stack([i, j, k], dim=-1)

    @staticmethod
    def _safe_normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
        # 對於接近零的向量，直接回傳零向量，避免數值不穩定
        n = torch.linalg.norm(x, dim=-1, keepdim=True).clamp(min=eps)
        return x / n