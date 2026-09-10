"""
方向-bin 工具函式（真正 HEALPix 後端，主套件：healpy）

HEALPix（Hierarchical Equal Area isoLatitude Pixelization）
- 每個 pixel 的立體角相等：每區匹配相同的仰器立體角
- 總pixel數 N_pix = 12 * nside^2
  nside=1 →  12 pixels
  nside=2 →  48 pixels
- 主要 API：
    hp.nside2npix(nside)         → 總 pixel 數
    hp.vec2pix(nside, x, y, z)  → 單位向量 → pixel index
    hp.pix2vec(nside, ipix)     → pixel 中心 → (x,y,z)
    hp.get_all_neighbours(nside, ipix) → 8-way 鄰居（缺角用 -1 填充）

操作座標系
-----------
- healpy 使用 GALACTIC 或任意球座標系，函式本身
  無座標系假設，僅處理單位向量
- 所有入口均接受 SplaTAM-world (RUB) 座標系下的向量

公開 API
----------------------------------
dir_to_bin(d, nside)             -> int
dir_to_bin_batch(d, nside)       -> Tensor[N]
bin_to_dir(p, nside)             -> Tensor[3]
bin_4neighbors(p, nside)         -> List[int]   (healpy 8-nb 中取 4)
bin_8neighbors(p, nside)         -> List[int]
build_neighbor_table(nside, ...) -> Tensor[P, 8]
npix(nside)                      -> int
"""
from __future__ import annotations

from typing import List

import healpy as hp
import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def npix(nside: int) -> int:
    """回傳 HEALPix 總pixel數 = 12 * nside^2"""
    return hp.nside2npix(int(nside))


# ---------------------------------------------------------------------------
# 向量 → bin
# ---------------------------------------------------------------------------

def dir_to_bin(
    d: Tensor,
    nside: int = 1,
) -> int:
    """
    將單一單位方向向量映射到 HEALPix pixel index（RING 排序）。

    Args:
        d     : (3,) 張量（會內部正規化）
        nside : HEALPix nside 參數（必須為 2 的冪次）

    Returns:
        pixel index  p  in  [0, 12 * nside^2)
    """
    x = float(d[0])
    y = float(d[1])
    z = float(d[2])
    norm = (x**2 + y**2 + z**2) ** 0.5
    if norm < 1e-8:
        return 0
    x, y, z = x / norm, y / norm, z / norm
    return int(hp.vec2pix(int(nside), x, y, z, nest=False))


def dir_to_bin_batch(
    d: Tensor,
    nside: int = 1,
) -> Tensor:
    """
    Batch 版本：將 N 個具向量映射到 HEALPix pixel index。
    目的: 方便在 SemanticVoxelMap 中對每個 voxel-bin 的觀測進行 bin 索引，從而聚合到 voxel-bin level。
    Args:
        d     : (N, 3) 張量（會內部正規化）
        nside : HEALPix nside

    Returns:
        (N,) int64 pixel indices，與輸入張量在同一 device
    """
    dev = d.device
    d_np = d.cpu().double().numpy()   # (N, 3)
    norm = np.linalg.norm(d_np, axis=1, keepdims=True).clip(min=1e-8)
    d_np = d_np / norm
    pix = hp.vec2pix(int(nside), d_np[:, 0], d_np[:, 1], d_np[:, 2], nest=False)
    return torch.as_tensor(pix, dtype=torch.int64, device=dev)


# ---------------------------------------------------------------------------
# bin → 方向
# ---------------------------------------------------------------------------

def bin_to_dir(
    p: int,
    nside: int = 1,
) -> Tensor:
    """
    回傳 HEALPix pixel p 的中心方向向量。
    用於 look-at pose 生成（指向熱區 bin 中心）
    """
    x, y, z = hp.pix2vec(int(nside), int(p), nest=False)
    return torch.tensor([float(x), float(y), float(z)], dtype=torch.float32)


# ---------------------------------------------------------------------------
# 鄰居查詢
# ---------------------------------------------------------------------------

def bin_8neighbors(         # 找HEALPix pixel p 的周邊八個鄰居
    p: int,
    nside: int = 1,
) -> List[int]:
    """
    回傳 HEALPix pixel p 的 8-way 鄰居（RING 排序）。
    healpy.get_all_neighbours 回傳的 -1 表示缺鄰（極區pixel），已自動濾除。
    """
    neighbors = hp.get_all_neighbours(int(nside), int(p), nest=False)  # shape (8,)
    return [int(q) for q in neighbors if q >= 0]


def bin_4neighbors(         # 從 8-way 鄰居中取主要的 4 個（N, S, E, W 方向），不足部分用剩餘鄰居補充
    p: int,
    nside: int = 1,
) -> List[int]:
    """
    回傳 HEALPix pixel p 的 4 個主要鄰居
    （從 8-way 鄰居中取奇數位 → N, S, E, W 方向）。
    """
    all_nbs = bin_8neighbors(p, nside)
    # 奇數位 1,3,5,7：N, S, E, W（RING 排序下的 4 個最鄰鄰居）
    preferred = [nb for i, nb in enumerate(all_nbs) if i % 2 == 1]
    rest      = [nb for i, nb in enumerate(all_nbs) if i % 2 == 0]
    result = preferred[:4]
    for nb in rest:
        if len(result) >= 4:
            break
        result.append(nb)
    return result


# ---------------------------------------------------------------------------
# 預建鄰居表（SemanticVoxelMap.topk_hot_voxels 批次計算用）
# ---------------------------------------------------------------------------

def build_neighbor_table(       # 紀錄每個 pixel 的鄰居索引，方便批次查詢
    nside: int = 1,
    max_nb: int = 8,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """
    Args:
        nside  : HEALPix nside
        max_nb : 鄰居表寬度（8 = 全設，4 = 最基本）
        device : torch device

    Returns:
        Tensor of shape (P, max_nb)
    """
    P = hp.nside2npix(int(nside))
    table = torch.full((P, max_nb), -1, dtype=torch.long, device=device)
    nb_fn = bin_8neighbors if max_nb == 8 else bin_4neighbors
    for p in range(P):
        nbs = nb_fn(p, nside=nside)[:max_nb]
        for ni, nb in enumerate(nbs):
            table[p, ni] = nb
    return table
