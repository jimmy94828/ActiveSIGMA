"""
Numerically stable divergence and entropy utilities.

Used by SemanticVoxelMap and other components.
All batch functions operate on torch.Tensor of arbitrary shape (..., C),
agreeing with the convention that class dimension is the LAST dimension.

Functions
---------
normalize(p)      : clamp + L1-normalize last dim
entropy(p)        : Shannon entropy (nats)
jsd(p, q)         : Jensen-Shannon Divergence (nats, in [0, ln2])
calc_shannon_entropy : alias kept for backward-compat with semsplatam.py usage
"""
from __future__ import annotations

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def normalize(p: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Clamp to eps then L1-normalise the last dimension.

    Args:
        p   : (..., C) non-negative tensor
        eps : minimum value before normalisation

    Returns:
        (..., C) tensor that sums to 1 along the last dimension
    """
    p = torch.clamp(p, min=eps)
    return p / p.sum(dim=-1, keepdim=True).clamp(min=eps)


def entropy(p: Tensor, eps: float = 1e-6, dim: int = -1) -> Tensor:
    """
    Shannon entropy in **nats**.

    Args:
        p   : (..., C) tensor (probabilities or logits-compatible non-negatives)
        eps : numerical floor before log
        dim : class dimension

    Returns:
        (...,) entropy values  (same shape as p with `dim` removed)
    """
    p = torch.clamp(p, min=eps)
    p = p / p.sum(dim=dim, keepdim=True).clamp(min=eps)
    return -(torch.xlogy(p, p)).sum(dim=dim)


def jsd(p: Tensor, q: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Jensen-Shannon Divergence (nats, symmetric, bounded in [0, ln 2]).

    Uses ``torch.xlogy`` for numerically safe 0 * log(0) = 0 handling,
    so neither NaN nor inf is produced.

    Args:
        p : (..., C) distribution (will be normalised internally)
        q : (..., C) distribution (will be normalised internally)
        eps : numerical floor

    Returns:
        (...,) JSD values
    """
    p = normalize(p, eps=eps)          # (..., C)
    q = normalize(q, eps=eps)          # (..., C)
    m = 0.5 * (p + q)                  # mixture distribution
    m_c = m.clamp(min=eps)
    # KL(p || m) = sum p * (log p  - log m)
    # Using xlogy: xlogy(a, b) = a * log(b), with xlogy(0, anything) = 0
    kl_pm = (torch.xlogy(p, p) - torch.xlogy(p, m_c)).sum(dim=-1)
    kl_qm = (torch.xlogy(q, q) - torch.xlogy(q, m_c)).sum(dim=-1)
    return (0.5 * (kl_pm + kl_qm)).clamp(min=0.0)


# ---------------------------------------------------------------------------
# Conveniences / back-compat aliases
# ---------------------------------------------------------------------------

def calc_shannon_entropy(p: Tensor, dim: int = -1) -> Tensor:
    """
    Shannon entropy alias compatible with the usage in semsplatam.py
    (``calc_shannon_entropy(topk_prob, dim=0).mean()``).

    Unlike the internal ``entropy()`` above this does NOT re-normalise,
    so the caller is responsible for passing a valid distribution.

    Returns entropy in nats along `dim`.
    """
    p_safe = torch.clamp(p, min=1e-9)
    return -(torch.xlogy(p_safe, p_safe)).sum(dim=dim)


def kl_divergence(p: Tensor, q: Tensor, eps: float = 1e-6) -> Tensor:
    """
    KL(p || q) in nats.  Both inputs are normalised internally.

    Args:
        p : (..., C)
        q : (..., C)

    Returns:
        (...,) KL values (>= 0)
    """
    p = normalize(p, eps=eps)
    q = normalize(q, eps=eps)
    return (torch.xlogy(p, p) - torch.xlogy(p, q.clamp(min=eps))).sum(dim=-1).clamp(min=0.0)
