"""Locality analysis of an activation trace (stage 1 -> stage 3 hand-off).

The activation histogram answers "which experts are used most often". That is
NOT the question a memory tier cares about. A cache's fetch traffic is set by
*temporal locality*: whether the experts a token needs are the ones the previous
tokens needed. A perfectly uniform-by-frequency workload can still cache well if
its accesses are clustered in time, and a skewed one can cache badly if they are
not.

This module measures three things on the per-token trace:

1. **Coverage** -- how many experts per layer must be resident to serve a given
   share of dispatches, under *static* pinning by global frequency. This is the
   ceiling on the hot/cold scheme as docs.md 4.2 defines it.
2. **LRU hit rate** -- what a *dynamic* cache of the same size achieves. The gap
   between this and (1) is the value of adapting placement at runtime, which is
   what stage 3's scheduler would exploit.
3. **Belady optimal** -- the unachievable upper bound for any cache of that size,
   which bounds how much any smarter policy could possibly win.

All outputs are dimensionless rates and counts. No energy, no time.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def uniform_null_stats(n_dispatches: int, n_experts: int, seed: int = 0) -> dict[str, float]:
    """Gini/entropy a *perfectly uniform* router would show at this sample size.

    Finite sampling alone produces non-zero gini, so a raw gini means nothing
    without this baseline. Comparing the measured gini against it is what
    separates "genuinely skewed" from "uniform plus noise".

    Args:
        n_dispatches: Total dispatches observed in the layer.
        n_experts: Experts available in that layer.
        seed: RNG seed for the multinomial draw.

    Returns:
        Dict with ``gini`` and ``normalized_entropy`` of the null distribution.
    """
    from .classify import gini, normalized_entropy

    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_dispatches, [1.0 / n_experts] * n_experts)
    return {"gini": gini(counts), "normalized_entropy": normalized_entropy(counts)}


def coverage_table(
    counts: np.ndarray, experts_per_site: Sequence[int], levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95, 0.99)
) -> pd.DataFrame:
    """Experts per layer needed to cover each dispatch share, by static frequency.

    Returns:
        DataFrame with one row per level: ``level``, ``mean_experts``,
        ``max_experts``, and ``mean_fraction`` of the layer's experts required.
    """
    from .classify import coverage_curve

    rows = []
    for level in levels:
        needed = []
        for site_idx, width in enumerate(experts_per_site):
            curve = coverage_curve(counts[site_idx, :width])
            needed.append(int(np.searchsorted(curve, level) + 1) if curve.size else 0)
        needed_arr = np.array(needed, dtype=float)
        widths = np.array(list(experts_per_site), dtype=float)
        rows.append(
            {
                "level": level,
                "mean_experts": float(needed_arr.mean()),
                "max_experts": int(needed_arr.max()),
                "mean_fraction": float((needed_arr / widths).mean()),
            }
        )
    return pd.DataFrame(rows)


def _token_expert_sets(
    trace: pd.DataFrame, site_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-token expert requests for one layer, in token order.

    Returns:
        ``(offsets, experts)`` where token ``i``'s experts are
        ``experts[offsets[i]:offsets[i + 1]]``.
    """
    layer = trace[trace["site_idx"] == site_idx]
    layer = layer.sort_values(["token_uid", "slot_k"], kind="stable")
    token_uid = layer["token_uid"].to_numpy()
    experts = layer["expert_id"].to_numpy().astype(np.int32)
    # Token boundaries: token_uid is non-decreasing after the sort.
    boundaries = np.flatnonzero(np.diff(token_uid)) + 1
    offsets = np.concatenate(([0], boundaries, [len(token_uid)]))
    return offsets, experts


def simulate_lru(offsets: np.ndarray, experts: np.ndarray, capacity: int) -> float:
    """Hit rate of an LRU expert cache of ``capacity`` entries for one layer.

    A "hit" is one dispatch whose expert is already resident. All experts a token
    needs are inserted after the token is served, so the cache warms naturally.

    Returns:
        Hits divided by total dispatches, in [0, 1].
    """
    if capacity <= 0:
        return 0.0
    cache: OrderedDict[int, None] = OrderedDict()
    hits = 0
    total = 0
    for i in range(len(offsets) - 1):
        wanted = experts[offsets[i] : offsets[i + 1]]
        for expert in wanted:
            key = int(expert)
            total += 1
            if key in cache:
                hits += 1
                cache.move_to_end(key)
            else:
                cache[key] = None
                if len(cache) > capacity:
                    cache.popitem(last=False)
    return hits / total if total else 0.0


def simulate_static(
    offsets: np.ndarray, experts: np.ndarray, counts_row: np.ndarray, capacity: int
) -> float:
    """Hit rate when the ``capacity`` globally most-used experts are pinned.

    This is the docs.md 4.2 hot/cold scheme. It needs no runtime decisions, which
    is exactly why it cannot react to phase changes in the workload.

    Note the modelling asymmetry against :func:`simulate_lru` and
    :func:`simulate_belady`: pinned experts are treated as **already resident**
    (they are loaded once at init and never evicted), so static pays no
    compulsory misses, while the dynamic policies start with a cold cache. On a
    short trace this can put static marginally *above* even the Belady bound. On
    a real trace -- millions of accesses against tens of experts -- the
    compulsory misses are negligible and the comparison is fair.
    """
    if capacity <= 0:
        return 0.0
    pinned = set(np.argsort(-counts_row, kind="stable")[:capacity].tolist())
    hits = int(np.isin(experts, list(pinned)).sum())
    return hits / len(experts) if len(experts) else 0.0


def simulate_belady(offsets: np.ndarray, experts: np.ndarray, capacity: int) -> float:
    """Hit rate of the optimal (Belady MIN) policy -- an upper bound, not a policy.

    Evicts whichever resident expert is next needed furthest in the future. No
    online scheduler can beat this, so the gap between LRU and this number bounds
    what stage 3 could gain from a cleverer policy.
    """
    if capacity <= 0:
        return 0.0
    n = len(experts)
    # next_use[i] = index of the next occurrence of experts[i], or n if none.
    next_use = np.full(n, n, dtype=np.int64)
    last_seen: dict[int, int] = {}
    for i in range(n - 1, -1, -1):
        key = int(experts[i])
        next_use[i] = last_seen.get(key, n)
        last_seen[key] = i

    cache: dict[int, int] = {}  # expert -> its next-use index
    hits = 0
    for i in range(n):
        key = int(experts[i])
        if key in cache:
            hits += 1
            cache[key] = next_use[i]
            continue
        if len(cache) >= capacity:
            victim = max(cache, key=lambda k: cache[k])
            if cache[victim] <= next_use[i]:
                # Everything resident is needed sooner than this one: skip caching.
                continue
            del cache[victim]
        cache[key] = next_use[i]
    return hits / n if n else 0.0


@dataclass
class LocalityResult:
    """Per-capacity comparison of static pinning, LRU, and Belady."""

    table: pd.DataFrame
    n_experts: int
    top_k: int

    def format(self) -> str:
        lines = [
            f"{'cache':>6} {'%experts':>9} {'static':>8} {'LRU':>8} {'Belady':>8} "
            f"{'LRU-static':>11} {'headroom':>9}",
            "-" * 66,
        ]
        for _, row in self.table.iterrows():
            lines.append(
                f"{int(row['capacity']):>6} {row['capacity'] / self.n_experts:>8.0%} "
                f"{row['static']:>7.1%} {row['lru']:>7.1%} {row['belady']:>7.1%} "
                f"{row['lru'] - row['static']:>+10.1%} {row['belady'] - row['lru']:>+8.1%}"
            )
        return "\n".join(lines)


def analyze_locality(
    trace: pd.DataFrame,
    counts: np.ndarray,
    experts_per_site: Sequence[int],
    capacities: Sequence[int] | None = None,
    sites: Sequence[int] | None = None,
    include_belady: bool = True,
) -> LocalityResult:
    """Compare static pinning against LRU and Belady across cache sizes.

    Args:
        trace: The per-token dispatch trace.
        counts: ``[n_sites, max_experts]`` dispatch counts.
        experts_per_site: Expert count per layer.
        capacities: Cache sizes in experts. Defaults to a spread over the layer's
            expert count.
        sites: Which layers to simulate. Defaults to all. Restricting to a few
            layers makes this much faster on a large trace.
        include_belady: Belady is the slowest of the three; disable to skip it.

    Returns:
        A :class:`LocalityResult` averaged over the simulated layers.
    """
    n_experts = int(max(experts_per_site))
    if capacities is None:
        fractions = (0.05, 0.10, 0.20, 0.30, 0.50, 0.75)
        capacities = sorted({max(1, int(round(f * n_experts))) for f in fractions})
    site_list = list(sites) if sites is not None else list(range(len(experts_per_site)))

    per_site: dict[int, tuple[np.ndarray, np.ndarray]] = {
        s: _token_expert_sets(trace, s) for s in site_list
    }
    top_k = 0
    if site_list:
        offsets, _ = per_site[site_list[0]]
        top_k = int(np.diff(offsets).max()) if len(offsets) > 1 else 0

    rows = []
    for capacity in capacities:
        static_rates, lru_rates, belady_rates = [], [], []
        for s in site_list:
            offsets, experts = per_site[s]
            static_rates.append(simulate_static(offsets, experts, counts[s, : experts_per_site[s]], capacity))
            lru_rates.append(simulate_lru(offsets, experts, capacity))
            if include_belady:
                belady_rates.append(simulate_belady(offsets, experts, capacity))
        rows.append(
            {
                "capacity": capacity,
                "static": float(np.mean(static_rates)),
                "lru": float(np.mean(lru_rates)),
                "belady": float(np.mean(belady_rates)) if include_belady else float("nan"),
            }
        )
    return LocalityResult(table=pd.DataFrame(rows), n_experts=n_experts, top_k=top_k)


def report(run_dir: str | Path, max_sites: int | None = 4, include_belady: bool = True) -> dict[str, Any]:
    """Print the full locality report for a finished run.

    Args:
        run_dir: A run directory written by :mod:`profiler.runner`.
        max_sites: Simulate only this many layers (evenly spaced) to keep the
            runtime reasonable. None simulates every layer.
        include_belady: Whether to compute the optimal-policy bound.

    Returns:
        Dict with the computed tables, for programmatic use.
    """
    from .activation_log import counts_matrix_from_frame, load_run

    loaded = load_run(run_dir)
    metadata = loaded["metadata"]
    counts_frame = loaded["counts"]
    counts = counts_matrix_from_frame(counts_frame)
    experts_per_site = (
        counts_frame.groupby("site_idx")["expert_id"].max().add(1).sort_index().tolist()
    )

    print("=" * 70)
    print(f"LOCALITY REPORT  --  {metadata['run_name']}")
    print(f"model: {metadata['config']['model']['name_or_path']}")
    print("=" * 70)

    # --- 1. Is the measured skew real, or just sampling noise? --------------
    n_experts = int(max(experts_per_site))
    per_layer_dispatches = int(counts[0, : experts_per_site[0]].sum())
    null = uniform_null_stats(per_layer_dispatches, n_experts)
    measured_gini = metadata["classification"]["gini_overall"]
    measured_entropy = metadata["classification"]["normalized_entropy_overall"]

    print("\n[1] SKEW vs UNIFORM NULL")
    print(f"    A perfectly uniform router, sampled {per_layer_dispatches:,} times over")
    print(f"    {n_experts} experts, would itself show:")
    print(f"        gini = {null['gini']:.4f}   normalised entropy = {null['normalized_entropy']:.4f}")
    print(f"    Measured:")
    print(f"        gini = {measured_gini:.4f}   normalised entropy = {measured_entropy:.4f}")
    ratio = measured_gini / null["gini"] if null["gini"] > 0 else float("inf")
    print(f"    -> measured gini is {ratio:.0f}x the uniform null.")
    if ratio < 3:
        print("    -> NOT meaningfully skewed. Suspect the hooks before the model.")
    else:
        print("    -> Routing is genuinely non-uniform (hooks are reading real structure).")

    # --- 2. How big must a statically pinned hot set be? --------------------
    print("\n[2] STATIC COVERAGE (docs.md 4.2 hot/cold scheme)")
    cov = coverage_table(counts, experts_per_site)
    print(f"    {'coverage':>9} {'experts/layer':>14} {'% of layer':>11}")
    print("    " + "-" * 36)
    for _, row in cov.iterrows():
        print(f"    {row['level']:>8.0%} {row['mean_experts']:>14.1f} {row['mean_fraction']:>10.0%}")

    # --- 3. Does a dynamic cache beat static pinning? -----------------------
    trace_path = loaded["trace_path"]
    locality = None
    if trace_path is None:
        print("\n[3] LOCALITY: no trace file in this run; rerun with profiler.record_trace=true")
    else:
        print(f"\n[3] TEMPORAL LOCALITY  (reading {trace_path.name})")
        trace = (
            pd.read_parquet(trace_path)
            if trace_path.suffix == ".parquet"
            else pd.read_csv(trace_path)
        )
        all_sites = sorted(trace["site_idx"].unique().tolist())
        if max_sites is not None and len(all_sites) > max_sites:
            picks = np.linspace(0, len(all_sites) - 1, max_sites).round().astype(int)
            sim_sites = [all_sites[i] for i in dict.fromkeys(picks)]
        else:
            sim_sites = all_sites
        print(f"    simulating layers {sim_sites} "
              f"({len(sim_sites)} of {len(all_sites)}), averaging hit rates")
        locality = analyze_locality(
            trace, counts, experts_per_site, sites=sim_sites, include_belady=include_belady
        )
        print()
        print("    " + locality.format().replace("\n", "\n    "))
        print()
        print("    static     = pin the globally most-used experts, never change")
        print("    LRU        = evict least-recently-used at runtime")
        print("    Belady     = optimal clairvoyant policy; nothing can beat it")
        print("    LRU-static = what runtime adaptation is worth")
        print("    headroom   = what a smarter policy than LRU could still gain")

    return {"coverage": cov, "null": null, "locality": locality}
