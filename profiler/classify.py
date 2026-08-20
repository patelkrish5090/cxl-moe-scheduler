"""Hot/cold expert classification from an activation count table (docs.md 4.2).

Input is the ``[n_sites, n_experts]`` integer dispatch-count matrix produced by
``profiler.router_hooks.RouterProfiler``. Output is a per-(layer, expert)
assignment plus the skew statistics that validation checkpoint 1 in docs.md
depends on.

All quantities here are dimensionless (counts, fractions). No energy, no time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import ClassifyConfig


def gini(counts: np.ndarray) -> float:
    """Gini coefficient of a non-negative count vector.

    0.0 means perfectly uniform routing (every expert used equally), 1.0 means
    all traffic goes to one expert. A near-zero value on a *trained* MoE model
    means the hooking is wrong (docs.md section 6, checkpoint 1); on a
    randomly-initialised model near-zero is the expected, correct result.

    Returns:
        Gini coefficient in [0, 1]; 0.0 for an all-zero vector.
    """
    values = np.sort(np.asarray(counts, dtype=np.float64).ravel())
    total = values.sum()
    if total <= 0 or values.size == 0:
        return 0.0
    n = values.size
    index = np.arange(1, n + 1)
    return float((2.0 * (index * values).sum()) / (n * total) - (n + 1.0) / n)


def normalized_entropy(counts: np.ndarray) -> float:
    """Shannon entropy of the dispatch distribution, normalised to [0, 1].

    1.0 = uniform routing across experts, 0.0 = one expert takes everything.
    Complements :func:`gini`; reported together because they fail differently.
    """
    values = np.asarray(counts, dtype=np.float64).ravel()
    total = values.sum()
    if total <= 0 or values.size <= 1:
        return 0.0
    probs = values[values > 0] / total
    entropy = float(-(probs * np.log(probs)).sum())
    return entropy / float(np.log(values.size))


def coverage_curve(counts: np.ndarray) -> np.ndarray:
    """Cumulative share of dispatches covered by the top-n experts.

    Returns:
        Array of length ``n_experts`` where element ``i`` is the fraction of all
        dispatches served by the ``i + 1`` most-used experts.
    """
    values = np.sort(np.asarray(counts, dtype=np.float64).ravel())[::-1]
    total = values.sum()
    if total <= 0:
        return np.zeros_like(values)
    return np.cumsum(values) / total


def _hot_mask_for_vector(counts: np.ndarray, cfg: ClassifyConfig) -> np.ndarray:
    """Boolean hot mask for one count vector under ``cfg``.

    Ties are broken by expert id (lower id wins) so the split is deterministic.
    Experts with zero dispatches are never marked hot.
    """
    n = counts.size
    order = np.argsort(-counts, kind="stable")  # descending, ties -> lower index first

    if cfg.method == "top_fraction":
        n_hot = int(np.ceil(cfg.value * n))
    elif cfg.method == "count":
        n_hot = int(cfg.value)
    elif cfg.method == "coverage":
        total = counts.sum()
        if total <= 0:
            n_hot = 0
        else:
            cumulative = np.cumsum(counts[order]) / total
            n_hot = int(np.searchsorted(cumulative, cfg.value) + 1)
    else:  # pragma: no cover - ClassifyConfig validates this
        raise ValueError(f"unknown method {cfg.method!r}")

    n_hot = max(0, min(n_hot, n))
    mask = np.zeros(n, dtype=bool)
    if n_hot:
        mask[order[:n_hot]] = True
    mask &= counts > 0  # an unused expert is never "hot"
    return mask


@dataclass
class ClassificationResult:
    """Per-expert hot/cold assignment plus the skew evidence behind it.

    Attributes:
        table: One row per (site_idx, layer_idx, expert_id) with dispatch count,
            within-layer share, rank, and the ``is_hot`` flag.
        per_layer_stats: One row per MoE layer with gini, normalised entropy, and
            the dispatch share captured by the hot set.
        overall: Run-level summary numbers.
        config: The ``ClassifyConfig`` that produced this result.
    """

    table: pd.DataFrame
    per_layer_stats: pd.DataFrame
    overall: dict[str, Any]
    config: ClassifyConfig


def classify(
    counts: np.ndarray,
    cfg: ClassifyConfig,
    layer_ids: Sequence[int] | None = None,
    experts_per_site: Sequence[int] | None = None,
    expert_weight_bytes: Sequence[int] | None = None,
) -> ClassificationResult:
    """Split experts into hot and cold.

    Args:
        counts: ``[n_sites, max_experts]`` dispatch counts.
        cfg: Threshold rule to apply.
        layer_ids: Transformer layer index per site; defaults to ``range(n_sites)``.
        experts_per_site: Real expert count per site, for models whose layers do
            not all have the same number of experts. Columns beyond a site's
            count are dropped rather than counted as zero-use experts.
        expert_weight_bytes: Bytes per single expert per site, carried into the
            table so stage 3 can size fetches without reloading the model.

    Returns:
        A :class:`ClassificationResult`.
    """
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 2:
        raise ValueError(f"counts must be 2-D [n_sites, n_experts], got shape {counts.shape}")
    n_sites, max_experts = counts.shape

    layer_ids = list(layer_ids) if layer_ids is not None else list(range(n_sites))
    experts_per_site = (
        list(experts_per_site) if experts_per_site is not None else [max_experts] * n_sites
    )
    expert_weight_bytes = (
        list(expert_weight_bytes) if expert_weight_bytes is not None else [0] * n_sites
    )
    for name, seq in (
        ("layer_ids", layer_ids),
        ("experts_per_site", experts_per_site),
        ("expert_weight_bytes", expert_weight_bytes),
    ):
        if len(seq) != n_sites:
            raise ValueError(f"{name} has {len(seq)} entries but counts has {n_sites} sites")

    if cfg.per_layer:
        masks = [
            _hot_mask_for_vector(counts[s, : experts_per_site[s]], cfg) for s in range(n_sites)
        ]
    else:
        flat = np.concatenate([counts[s, : experts_per_site[s]] for s in range(n_sites)])
        flat_mask = _hot_mask_for_vector(flat, cfg)
        masks, offset = [], 0
        for s in range(n_sites):
            width = experts_per_site[s]
            masks.append(flat_mask[offset : offset + width])
            offset += width

    rows = []
    layer_stats = []
    for s in range(n_sites):
        width = experts_per_site[s]
        site_counts = counts[s, :width]
        total = int(site_counts.sum())
        mask = masks[s]
        rank = np.empty(width, dtype=np.int32)
        rank[np.argsort(-site_counts, kind="stable")] = np.arange(width, dtype=np.int32)

        for expert_id in range(width):
            rows.append(
                {
                    "site_idx": s,
                    "layer_idx": int(layer_ids[s]),
                    "expert_id": expert_id,
                    "dispatch_count": int(site_counts[expert_id]),
                    "layer_share": float(site_counts[expert_id] / total) if total else 0.0,
                    "rank_in_layer": int(rank[expert_id]),
                    "is_hot": bool(mask[expert_id]),
                    "expert_weight_bytes": int(expert_weight_bytes[s]),
                }
            )

        n_hot = int(mask.sum())
        layer_stats.append(
            {
                "site_idx": s,
                "layer_idx": int(layer_ids[s]),
                "n_experts": width,
                "n_hot": n_hot,
                "n_cold": width - n_hot,
                "dispatches": total,
                "hot_dispatch_share": float(site_counts[mask].sum() / total) if total else 0.0,
                "gini": gini(site_counts),
                "normalized_entropy": normalized_entropy(site_counts),
                "max_expert_share": float(site_counts.max() / total) if total else 0.0,
                "unused_experts": int((site_counts == 0).sum()),
            }
        )

    table = pd.DataFrame(rows)
    stats = pd.DataFrame(layer_stats)
    all_counts = np.concatenate([counts[s, : experts_per_site[s]] for s in range(n_sites)])
    total_dispatches = int(all_counts.sum())
    hot_bytes = int(table.loc[table["is_hot"], "expert_weight_bytes"].sum())
    cold_bytes = int(table.loc[~table["is_hot"], "expert_weight_bytes"].sum())

    overall = {
        "method": cfg.method,
        "value": cfg.value,
        "per_layer": cfg.per_layer,
        "n_sites": n_sites,
        "n_experts_total": int(sum(experts_per_site)),
        "n_hot_total": int(table["is_hot"].sum()),
        "total_dispatches": total_dispatches,
        "hot_dispatch_share": (
            float(table.loc[table["is_hot"], "dispatch_count"].sum() / total_dispatches)
            if total_dispatches
            else 0.0
        ),
        "gini_overall": gini(all_counts),
        "normalized_entropy_overall": normalized_entropy(all_counts),
        "unused_experts_total": int((all_counts == 0).sum()),
        # Sizes in bytes -- these drive stage 3's HBM residency budget.
        "hot_weight_bytes": hot_bytes,
        "cold_weight_bytes": cold_bytes,
    }
    return ClassificationResult(
        table=table, per_layer_stats=stats, overall=overall, config=cfg
    )
