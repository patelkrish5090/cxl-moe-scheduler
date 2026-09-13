"""CXL memory POOLING analysis -- the "CXL memory expansion & pooling study"
deliverable this project's problem statement asks for, distinct from the
"expansion" half stages 2/3 already cover.

WHAT THIS MEASURES: given two or more real, independent per-GPU stage-1 runs
of the SAME model (e.g. Mixtral-8x7B on cuda:0 over one corpus slice, and on
cuda:1 over a different slice -- see configs/mixtral_8x7b_decode_gpu0.json /
_gpu1.json), each GPU classifies its own experts hot/cold from its own real
traffic (docs.md 4.2, unchanged). Pooling does not change that -- it changes
WHERE a GPU's cold experts are physically stored:

  dedicated -- each GPU has its own private CXL allocation. If both GPUs
      mark (layer=3, expert=5) cold, that expert's weights are stored TWICE.
  pooled    -- one shared CXL pool stores each unique cold (layer, expert)
      pair ONCE, regardless of how many GPUs need it. Since both GPUs are
      running the SAME model, a cold expert at (layer=3, expert=5) is
      byte-for-byte the same weights on either GPU -- the dedup is real, not
      an approximation.

WHAT THIS DELIBERATELY DOES NOT MEASURE: shared-link BANDWIDTH CONTENTION
(two GPUs' cold fetches competing for one physical link's throughput at the
same simulated instant). An earlier draft of this module attempted a
"round-robin merge the two GPUs' per-dispatch latencies onto one shared
clock" estimate -- this is mathematically vacuous: summing two independent
GPUs' own per-dispatch latencies in ANY interleaved order still sums to the
same total (addition is order-independent), so it produced a "pooled
latency" number identically equal to the dedicated sum, which is not a
contention effect at all, just relabelled arithmetic. Modelling contention
for real requires a genuine concurrent discrete-event simulation (each GPU's
own clock advancing independently, splitting a fixed shared bandwidth budget
only when both need the link at the same simulated instant) -- a real scope
increase, not attempted here. State this limitation plainly rather than
ship a number that looks like a contention estimate but isn't one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PooledMemoryReport:
    """Dedicated vs pooled cold-expert storage across N real per-GPU runs.

    Attributes:
        gpu_names: e.g. ("gpu0", "gpu1"), in the order passed in.
        per_gpu_cold_bytes: each GPU's own cold-expert byte total (what it
            would need in a dedicated, non-pooled CXL allocation).
        dedicated_total_cold_bytes: sum of per_gpu_cold_bytes -- total bytes
            needed if every GPU has its own private CXL allocation.
        pooled_total_cold_bytes: bytes needed if one shared CXL pool stores
            each unique cold (layer, expert) pair exactly once.
        shared_cold_pairs: how many unique (layer, expert) pairs are cold on
            2 or more GPUs -- the actual source of the dedup savings; 0 means
            pooling saves nothing on this data (the GPUs' cold sets don't
            overlap), which is itself a real, reportable finding.
        total_unique_cold_pairs: total distinct (layer, expert) pairs cold on
            at least one GPU.
    """

    gpu_names: tuple[str, ...]
    per_gpu_cold_bytes: dict[str, int]
    dedicated_total_cold_bytes: int
    pooled_total_cold_bytes: int
    shared_cold_pairs: int
    total_unique_cold_pairs: int

    @property
    def savings_bytes(self) -> int:
        return self.dedicated_total_cold_bytes - self.pooled_total_cold_bytes

    @property
    def savings_pct(self) -> float:
        if self.dedicated_total_cold_bytes == 0:
            return 0.0
        return 100.0 * self.savings_bytes / self.dedicated_total_cold_bytes

    def summary(self) -> str:
        lines = ["CXL POOLING -- cold-expert memory footprint", "=" * 60]
        for gpu in self.gpu_names:
            cold_gb = self.per_gpu_cold_bytes[gpu] / 1e9
            lines.append(f"  {gpu}: {cold_gb:.3f} GB cold experts (dedicated)")
        lines.append(
            f"  dedicated total (each GPU stores its own cold set): "
            f"{self.dedicated_total_cold_bytes / 1e9:.3f} GB"
        )
        lines.append(
            f"  pooled total (one shared copy per unique cold expert): "
            f"{self.pooled_total_cold_bytes / 1e9:.3f} GB"
        )
        lines.append(
            f"  savings: {self.savings_bytes / 1e9:.3f} GB ({self.savings_pct:.1f}%), from "
            f"{self.shared_cold_pairs}/{self.total_unique_cold_pairs} unique cold "
            "(layer, expert) pairs needed by 2+ GPUs"
        )
        if self.shared_cold_pairs == 0 and self.total_unique_cold_pairs > 0:
            lines.append(
                "  NOTE: zero overlap between GPUs' cold sets on this data -- pooling saves "
                "nothing here, which is itself a real result, not a broken calculation. This "
                "would change with more GPUs, more similar workloads across them, or a "
                "coarser hot/cold threshold."
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "gpu_names": list(self.gpu_names),
            "per_gpu_cold_bytes": self.per_gpu_cold_bytes,
            "dedicated_total_cold_bytes": self.dedicated_total_cold_bytes,
            "pooled_total_cold_bytes": self.pooled_total_cold_bytes,
            "savings_bytes": self.savings_bytes,
            "savings_pct": self.savings_pct,
            "shared_cold_pairs": self.shared_cold_pairs,
            "total_unique_cold_pairs": self.total_unique_cold_pairs,
            "units": {"bytes": "bytes (also reported as GB in summary())"},
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def analyze_pooled_memory(hot_cold_tables: dict[str, pd.DataFrame]) -> PooledMemoryReport:
    """Compare dedicated vs pooled cold-expert storage across N real
    per-GPU stage-1 runs.

    Args:
        hot_cold_tables: ``{gpu_name: hot_cold.csv DataFrame}``, one per real
            per-GPU profiling run. Each table must have the standard
            profiler.classify schema (``is_hot``, ``layer_idx``,
            ``expert_id``, ``expert_weight_bytes``).

    Raises:
        ValueError: if fewer than 2 GPUs are given -- pooling is meaningless
            for a single GPU.
    """
    if len(hot_cold_tables) < 2:
        raise ValueError(
            f"pooling needs at least 2 GPUs' worth of real data, got {len(hot_cold_tables)}"
        )

    gpu_names = tuple(hot_cold_tables.keys())
    per_gpu_cold_bytes: dict[str, int] = {}
    cold_pair_to_bytes: dict[tuple[int, int], int] = {}
    cold_pair_gpu_count: dict[tuple[int, int], int] = {}

    for gpu, table in hot_cold_tables.items():
        cold = table[~table["is_hot"]]
        per_gpu_cold_bytes[gpu] = int(cold["expert_weight_bytes"].sum())
        for row in cold.itertuples():
            key = (int(row.layer_idx), int(row.expert_id))
            cold_pair_to_bytes[key] = int(row.expert_weight_bytes)
            cold_pair_gpu_count[key] = cold_pair_gpu_count.get(key, 0) + 1

    dedicated_total = sum(per_gpu_cold_bytes.values())
    pooled_total = sum(cold_pair_to_bytes.values())
    shared_pairs = sum(1 for c in cold_pair_gpu_count.values() if c >= 2)

    return PooledMemoryReport(
        gpu_names=gpu_names,
        per_gpu_cold_bytes=per_gpu_cold_bytes,
        dedicated_total_cold_bytes=dedicated_total,
        pooled_total_cold_bytes=pooled_total,
        shared_cold_pairs=shared_pairs,
        total_unique_cold_pairs=len(cold_pair_to_bytes),
    )
