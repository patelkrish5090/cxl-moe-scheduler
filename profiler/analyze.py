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

Layers are not interchangeable in this measurement. Some route every token to
the same handful of experts; their working set fits in the smallest cache worth
simulating, so all three policies score ~100% on them at every capacity.
Averaging those layers in pulls every policy toward 100% and shrinks the
measured LRU-minus-static gap without saying anything about placement. The
report therefore splits layers into *cache-trivial* and *diverse* (see
:func:`working_set_table`) and reports the locality comparison for both.

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
    counts: np.ndarray,
    experts_per_site: Sequence[int],
    levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95, 0.99),
    sites: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Experts per layer needed to cover each dispatch share, by static frequency.

    Args:
        counts: ``[n_sites, max_experts]`` dispatch counts.
        experts_per_site: Expert count per layer.
        levels: Dispatch shares to report.
        sites: Restrict the average to these layers. Defaults to all of them.
            Pass the diverse subset to keep cache-trivial layers from dragging
            the mean down.

    Returns:
        DataFrame with one row per level: ``level``, ``mean_experts``,
        ``max_experts``, and ``mean_fraction`` of the layer's experts required.
    """
    from .classify import coverage_curve

    site_list = list(sites) if sites is not None else list(range(len(experts_per_site)))
    rows = []
    for level in levels:
        needed = []
        for site_idx in site_list:
            width = experts_per_site[site_idx]
            curve = coverage_curve(counts[site_idx, :width])
            needed.append(int(np.searchsorted(curve, level) + 1) if curve.size else 0)
        needed_arr = np.array(needed, dtype=float)
        widths = np.array([experts_per_site[s] for s in site_list], dtype=float)
        rows.append(
            {
                "level": level,
                "mean_experts": float(needed_arr.mean()) if needed_arr.size else float("nan"),
                "max_experts": int(needed_arr.max()) if needed_arr.size else 0,
                "mean_fraction": float((needed_arr / widths).mean()) if needed_arr.size else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def counts_from_trace(
    trace: pd.DataFrame, n_sites: int, n_experts: int
) -> np.ndarray:
    """Rebuild the ``[n_sites, n_experts]`` count matrix from trace rows.

    ``expert_counts.csv`` is written over the whole run. When the report is
    restricted to one phase, every number in it must describe the same token
    population, so the counts are recomputed from the filtered trace instead of
    reused from the CSV.
    """
    matrix = np.zeros((n_sites, n_experts), dtype=np.int64)
    np.add.at(
        matrix,
        (trace["site_idx"].to_numpy().astype(np.int64), trace["expert_id"].to_numpy().astype(np.int64)),
        1,
    )
    return matrix


def working_set_table(
    trace: pd.DataFrame,
    top_k: int,
    experts_per_site: Sequence[int] | None = None,
    coverage: float = 0.99,
) -> pd.DataFrame:
    """Per-layer working-set size, and whether a cache has any decision to make.

    A layer's *working set* here is the fewest experts whose dispatches together
    cover ``coverage`` of that layer's traffic, measured over whatever token
    population ``trace`` holds -- so filter by phase first if you care about one.

    A layer whose working set is no larger than ``top_k`` is **cache-trivial**.
    ``top_k`` is the smallest cache worth simulating, since a token needs that
    many experts resident to be served at all; a layer at or below it is fully
    resident at every capacity, and static pinning, LRU and Belady therefore all
    score ~100% on it. Such a layer is real routing behaviour, not an error, but
    it carries no information about *placement policy*, and averaging it into a
    policy comparison biases every policy toward 100%.

    Args:
        trace: Per-dispatch rows, already filtered to the phase of interest.
        top_k: Experts each token is dispatched to, from
            ``run_metadata.json['model_topology']['top_k']``.
        experts_per_site: Experts available per layer. Defaults to the highest
            expert id observed in the trace for that layer, plus one.
        coverage: Dispatch share the working set must cover. Held just below 1.0
            so a single stray dispatch does not inflate the count.

    Returns:
        One row per site with ``site_idx``, ``layer_idx``, ``dispatches``,
        ``n_experts``, ``used_experts``, ``working_set``, ``top_k_share`` (the
        share taken by the ``top_k`` busiest experts) and ``cache_trivial``.
    """
    from .classify import coverage_curve

    rows = []
    for site_idx, group in trace.groupby("site_idx", sort=True):
        site_idx = int(site_idx)
        expert_ids = group["expert_id"].to_numpy().astype(np.int64)
        if experts_per_site is not None and site_idx < len(experts_per_site):
            width = int(experts_per_site[site_idx])
        else:
            width = int(expert_ids.max()) + 1 if expert_ids.size else 0
        counts_row = np.bincount(expert_ids, minlength=width)
        total = int(counts_row.sum())
        curve = coverage_curve(counts_row)
        working_set = int(np.searchsorted(curve, coverage) + 1) if total else 0
        descending = np.sort(counts_row)[::-1]
        top_share = float(descending[:top_k].sum() / total) if total else 0.0
        layer_idx = (
            int(group["layer_idx"].iloc[0]) if "layer_idx" in group.columns else site_idx
        )
        rows.append(
            {
                "site_idx": site_idx,
                "layer_idx": layer_idx,
                "dispatches": total,
                "n_experts": width,
                "used_experts": int((counts_row > 0).sum()),
                "working_set": working_set,
                "top_k_share": top_share,
                "cache_trivial": bool(0 < working_set <= top_k),
            }
        )
    return pd.DataFrame(rows)


def _pick_sites(sites: Sequence[int], max_sites: int | None) -> list[int]:
    """Evenly-spaced subsample of ``sites``, preserving order and uniqueness."""
    site_list = list(sites)
    if max_sites is None or max_sites <= 0 or len(site_list) <= max_sites:
        return site_list
    picks = np.linspace(0, len(site_list) - 1, max_sites).round().astype(int)
    return [site_list[i] for i in dict.fromkeys(picks.tolist())]


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
    cache: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
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
        cache: Optional dict reused across calls to memoise the per-layer
            extraction of the trace, which is the expensive part when the same
            trace is analysed for several layer groups. Mutated in place.

    Returns:
        A :class:`LocalityResult` averaged over the simulated layers.
    """
    n_experts = int(max(experts_per_site))
    if capacities is None:
        fractions = (0.05, 0.10, 0.20, 0.30, 0.50, 0.75)
        capacities = sorted({max(1, int(round(f * n_experts))) for f in fractions})
    site_list = list(sites) if sites is not None else list(range(len(experts_per_site)))

    if cache is None:
        cache = {}
    for s in site_list:
        if s not in cache:
            cache[s] = _token_expert_sets(trace, s)
    per_site = cache
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


def report(
    run_dir: str | Path,
    max_sites: int | None = 4,
    include_belady: bool = True,
    phase: str = "all",
    layer_set: str = "both",
    working_set_coverage: float = 0.99,
) -> dict[str, Any]:
    """Print the full locality report for a finished run.

    Args:
        run_dir: A run directory written by :mod:`profiler.runner`.
        max_sites: Simulate only this many layers (evenly spaced) to keep the
            runtime reasonable. None simulates every layer.
        include_belady: Whether to compute the optimal-policy bound.
        phase: "all", "prefill", or "decode". Decode is the regime a serving
            scheduler actually operates in -- one token at a time, touching only
            top_k experts per layer. Prefill routes thousands of tokens through a
            layer at once and therefore requests nearly every expert, so prefill
            locality numbers say little about tiering. Filtering by phase also
            recomputes the counts from the trace, so every number in the report
            describes the same token population.
        layer_set: Which layer groups get a locality table -- "both" (default),
            "all", "diverse", or "trivial". See :func:`working_set_table` for
            what makes a layer cache-trivial and why averaging it in is
            misleading.
        working_set_coverage: Dispatch share defining a layer's working set.

    Returns:
        Dict with the computed tables, for programmatic use.
    """
    from .activation_log import counts_matrix_from_frame, load_run
    from .classify import gini, normalized_entropy

    valid_layer_sets = {"both", "all", "diverse", "trivial"}
    if layer_set not in valid_layer_sets:
        raise ValueError(f"layer_set must be one of {sorted(valid_layer_sets)}, got {layer_set!r}")

    loaded = load_run(run_dir)
    metadata = loaded["metadata"]
    counts_frame = loaded["counts"]
    counts = counts_matrix_from_frame(counts_frame)
    experts_per_site = (
        counts_frame.groupby("site_idx")["expert_id"].max().add(1).sort_index().tolist()
    )
    top_k = int(metadata.get("model_topology", {}).get("top_k", 1) or 1)

    print("=" * 70)
    print(f"LOCALITY REPORT  --  {metadata['run_name']}")
    print(f"model: {metadata['config']['model']['name_or_path']}")
    print("=" * 70)

    # --- 0. Load the trace up front: sections 1-2 are phase-dependent too. ---
    trace_path = loaded["trace_path"]
    trace: pd.DataFrame | None = None
    counts_source = "whole run (expert_counts.csv)"
    if trace_path is not None:
        trace = (
            pd.read_parquet(trace_path)
            if trace_path.suffix == ".parquet"
            else pd.read_csv(trace_path)
        )
        if phase != "all":
            want_decode = phase == "decode"
            before = len(trace)
            trace = trace[trace["is_decode"] == want_decode]
            print(f"\nphase filter {phase!r}: {len(trace):,} of {before:,} trace rows")
            if trace.empty:
                print(f"no {phase} rows in this run "
                      "(set profiler.max_new_tokens > 0 to record decode steps)")
                return {"coverage": None, "null": None, "locality": None,
                        "working_sets": None}
            # Recompute counts so skew, coverage and locality all describe the
            # same tokens rather than mixing phase-local traffic with run-wide
            # frequencies.
            counts = counts_from_trace(trace, counts.shape[0], counts.shape[1])
            counts_source = f"{phase} tokens only (recomputed from the trace)"

    # --- 1. Is the measured skew real, or just sampling noise? --------------
    n_experts = int(max(experts_per_site))
    per_layer_dispatches = int(counts[0, : experts_per_site[0]].sum())
    null = uniform_null_stats(per_layer_dispatches, n_experts)
    flat_counts = np.concatenate(
        [counts[s, : experts_per_site[s]] for s in range(counts.shape[0])]
    )
    if phase == "all":
        measured_gini = metadata["classification"]["gini_overall"]
        measured_entropy = metadata["classification"]["normalized_entropy_overall"]
    else:
        measured_gini = gini(flat_counts)
        measured_entropy = normalized_entropy(flat_counts)

    print("\n[1] SKEW vs UNIFORM NULL")
    print(f"    counts: {counts_source}")
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

    # --- 2. Which layers can a cache decide anything about? -----------------
    working_sets: pd.DataFrame | None = None
    diverse_sites: list[int] = list(range(counts.shape[0]))
    trivial_sites: list[int] = []
    if trace is not None:
        working_sets = working_set_table(
            trace, top_k, experts_per_site, coverage=working_set_coverage
        )
        trivial_sites = working_sets.loc[working_sets["cache_trivial"], "site_idx"].tolist()
        diverse_sites = working_sets.loc[~working_sets["cache_trivial"], "site_idx"].tolist()

        print("\n[2] LAYER WORKING SETS")
        print(f"    working set = fewest experts covering {working_set_coverage:.0%} of a layer's "
              "dispatches.")
        print(f"    top_k is {top_k}, so a layer whose working set is <= {top_k} is fully resident")
        print("    in the smallest cache worth simulating, and static, LRU and Belady all")
        print("    score ~100% on it at every capacity. Such layers are real routing")
        print("    behaviour, but they carry no signal about placement policy, so they are")
        print("    reported separately rather than averaged into the comparison.")
        print()
        if len(working_sets) <= 48:
            print(f"    {'layer':>5} {'dispatches':>11} {'used':>5} {'workset':>8} "
                  f"{'top' + str(top_k) + '_share':>11}  class")
            print("    " + "-" * 57)
            for _, row in working_sets.iterrows():
                label = "cache-trivial" if row["cache_trivial"] else "diverse"
                print(f"    {int(row['layer_idx']):5d} {int(row['dispatches']):11,} "
                      f"{int(row['used_experts']):5d} {int(row['working_set']):8d} "
                      f"{row['top_k_share']:10.1%}  {label}")
            print()
        n_total = len(working_sets)
        print(f"    {len(trivial_sites):2d} of {n_total} layers are cache-trivial "
              f"(working set <= {top_k})")
        if trivial_sites:
            print(f"       layers {working_sets.loc[working_sets['cache_trivial'], 'layer_idx'].tolist()}")
        print(f"    {len(diverse_sites):2d} of {n_total} layers are diverse")
        if diverse_sites:
            print(f"       layers {working_sets.loc[~working_sets['cache_trivial'], 'layer_idx'].tolist()}")
        if not diverse_sites:
            print("    -> Every layer is cache-trivial. There is nothing here for a")
            print("       placement policy to decide; tiering would be a pure capacity")
            print("       question, not a scheduling one.")
    else:
        print("\n[2] LAYER WORKING SETS: needs the per-token trace; rerun with "
              "profiler.record_trace=true")

    # --- 3. How big must a statically pinned hot set be? --------------------
    print("\n[3] STATIC COVERAGE (docs.md 4.2 hot/cold scheme)")
    cov = coverage_table(counts, experts_per_site)
    cov_diverse = (
        coverage_table(counts, experts_per_site, sites=diverse_sites)
        if trivial_sites and diverse_sites
        else None
    )
    if cov_diverse is None:
        print(f"    {'coverage':>9} {'experts/layer':>14} {'% of layer':>11}")
        print("    " + "-" * 36)
        for _, row in cov.iterrows():
            print(f"    {row['level']:>8.0%} {row['mean_experts']:>14.1f} {row['mean_fraction']:>10.0%}")
    else:
        print("    Split out because the cache-trivial layers need very few experts by")
        print("    construction and drag the all-layer mean down.")
        print()
        print(f"    {'coverage':>9} | {'all layers':>21} | {'diverse layers only':>21}")
        print(f"    {'':>9} | {'experts/layer':>13} {'% layer':>7} | "
              f"{'experts/layer':>13} {'% layer':>7}")
        print("    " + "-" * 57)
        for (_, row), (_, drow) in zip(cov.iterrows(), cov_diverse.iterrows()):
            print(f"    {row['level']:>9.0%} | {row['mean_experts']:>13.1f} "
                  f"{row['mean_fraction']:>7.0%} | {drow['mean_experts']:>13.1f} "
                  f"{drow['mean_fraction']:>7.0%}")

    # --- 4. Does a dynamic cache beat static pinning? -----------------------
    locality = None
    locality_by_group: dict[str, LocalityResult] = {}
    if trace is None:
        print("\n[4] LOCALITY: no trace file in this run; rerun with profiler.record_trace=true")
    else:
        print(f"\n[4] TEMPORAL LOCALITY  (reading {trace_path.name})")
        all_sites = sorted(trace["site_idx"].unique().tolist())

        groups: list[tuple[str, str, list[int]]] = []
        if layer_set in {"both", "all"}:
            groups.append(("all", "ALL LAYERS", all_sites))
        if layer_set in {"both", "diverse"} and diverse_sites:
            # With no trivial layers the diverse group is the all group; skip it
            # rather than print the identical table twice.
            if layer_set != "both" or trivial_sites:
                groups.append(("diverse", "DIVERSE LAYERS ONLY", diverse_sites))
        if layer_set == "trivial" and trivial_sites:
            groups.append(("trivial", "CACHE-TRIVIAL LAYERS ONLY", trivial_sites))
        if not groups:
            print(f"    no layers in group {layer_set!r}")

        extraction_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for key, label, group_sites in groups:
            sim_sites = _pick_sites(group_sites, max_sites)
            layer_ids = [
                int(working_sets.loc[working_sets["site_idx"] == s, "layer_idx"].iloc[0])
                if working_sets is not None and (working_sets["site_idx"] == s).any()
                else s
                for s in sim_sites
            ]
            print(f"\n    {label}  --  simulating layers {layer_ids} "
                  f"({len(sim_sites)} of {len(group_sites)}), averaging hit rates")
            result = analyze_locality(
                trace, counts, experts_per_site, sites=sim_sites,
                include_belady=include_belady, cache=extraction_cache,
            )
            locality_by_group[key] = result
            print()
            print("    " + result.format().replace("\n", "\n    "))

        locality = locality_by_group.get("diverse") or locality_by_group.get(layer_set) \
            or next(iter(locality_by_group.values()), None)

        print()
        print("    static     = pin the globally most-used experts, never change")
        print("    LRU        = evict least-recently-used at runtime")
        print("    Belady     = optimal clairvoyant policy; nothing can beat it")
        print("    LRU-static = what runtime adaptation is worth")
        print("    headroom   = what a smarter policy than LRU could still gain")
        if "all" in locality_by_group and "diverse" in locality_by_group:
            print()
            print("    The two tables are averages over different layer samples (each group")
            print("    is subsampled separately), so compare them as populations, not row")
            print("    by row. LRU-static on the diverse layers is the number stage 3's")
            print("    scheduler has to beat; the all-layer figure is diluted by layers")
            print("    where every policy is already at 100%.")

    return {
        "coverage": cov,
        "coverage_diverse": cov_diverse,
        "null": null,
        "locality": locality,
        "locality_by_group": locality_by_group,
        "working_sets": working_sets,
        "diverse_sites": diverse_sites,
        "trivial_sites": trivial_sites,
        "top_k": top_k,
    }
