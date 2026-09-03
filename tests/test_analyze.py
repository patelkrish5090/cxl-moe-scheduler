"""Correctness checks for the locality simulators in profiler/analyze.py.

Every case here has a hand-computable answer, because a cache simulator that is
subtly wrong would produce a plausible-looking hit rate and quietly misdirect
the stage-3 design.

    python tests/test_analyze.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profiler.analyze import (  # noqa: E402
    _pick_sites, analyze_locality, counts_from_trace, coverage_table, simulate_belady,
    simulate_lru, simulate_static, uniform_null_stats, working_set_table,
)

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        failures.append(name)
        print(f"  FAIL  {name}" + (f"\n        {detail}" if detail else ""))


def flat(seq: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """One expert per token: offsets are 0,1,2,..., experts is the sequence."""
    return np.arange(len(seq) + 1, dtype=np.int64), np.array(seq, dtype=np.int32)


print("\n[LRU]")
# Cyclic access over 3 experts with capacity 2: classic LRU worst case, every
# access misses after the warm-up.
offsets, experts = flat([0, 1, 2] * 10)
check("cyclic pattern defeats LRU (hit rate ~0)", simulate_lru(offsets, experts, 2) == 0.0,
      f"got {simulate_lru(offsets, experts, 2)}")
check("capacity >= working set gives near-perfect hits",
      abs(simulate_lru(offsets, experts, 3) - 27 / 30) < 1e-12,
      f"got {simulate_lru(offsets, experts, 3)}, expected {27/30}")

# Same expert repeatedly: 1 compulsory miss, rest hits.
offsets, experts = flat([5] * 100)
check("repeated single expert = 99/100 hits", abs(simulate_lru(offsets, experts, 1) - 0.99) < 1e-12,
      f"got {simulate_lru(offsets, experts, 1)}")
check("capacity 0 gives zero hit rate", simulate_lru(offsets, experts, 0) == 0.0)

# Every access distinct: nothing can ever hit.
offsets, experts = flat(list(range(50)))
check("all-distinct accesses never hit", simulate_lru(offsets, experts, 10) == 0.0)

print("\n[static pinning]")
# Expert 0 used 80x, expert 1 20x, experts 2-3 unused.
counts_row = np.array([80, 20, 0, 0], dtype=np.int64)
offsets, experts = flat([0] * 80 + [1] * 20)
check("pinning top-1 captures its exact share",
      abs(simulate_static(offsets, experts, counts_row, 1) - 0.80) < 1e-12,
      f"got {simulate_static(offsets, experts, counts_row, 1)}")
check("pinning top-2 captures everything",
      abs(simulate_static(offsets, experts, counts_row, 2) - 1.0) < 1e-12)
check("static ignores order (unlike LRU)",
      simulate_static(*flat([0, 1] * 50), counts_row, 1)
      == simulate_static(*flat([0] * 50 + [1] * 50), counts_row, 1))

print("\n[Belady optimal]")
offsets, experts = flat([0, 1, 2] * 10)
lru_rate = simulate_lru(offsets, experts, 2)
belady_rate = simulate_belady(offsets, experts, 2)
check("Belady beats LRU on the cyclic pattern", belady_rate > lru_rate,
      f"belady={belady_rate}, lru={lru_rate}")
check("Belady never exceeds 1.0 and is non-negative", 0.0 <= belady_rate <= 1.0)

offsets, experts = flat([5] * 100)
check("Belady matches LRU when there is nothing to choose",
      abs(simulate_belady(offsets, experts, 1) - 0.99) < 1e-12,
      f"got {simulate_belady(offsets, experts, 1)}")

offsets, experts = flat(list(range(50)))
check("Belady also cannot hit on all-distinct accesses",
      simulate_belady(offsets, experts, 10) == 0.0)

# Belady must be an upper bound on LRU for random workloads.
rng = np.random.default_rng(0)
seq = rng.integers(0, 12, size=3000).tolist()
offsets, experts = flat(seq)
worse = [c for c in (2, 4, 6, 8) if simulate_belady(offsets, experts, c) < simulate_lru(offsets, experts, c) - 1e-12]
check("Belady >= LRU at every capacity (it is the optimum)", not worse,
      f"violated at capacities {worse}")

print("\n[uniform null]")
null_small = uniform_null_stats(1_000, 64)
null_large = uniform_null_stats(1_000_000, 64)
check("null gini shrinks as sample size grows", null_large["gini"] < null_small["gini"],
      f"{null_large['gini']:.5f} vs {null_small['gini']:.5f}")
check("null gini is small at large n (so real skew stands out)", null_large["gini"] < 0.02,
      f"got {null_large['gini']:.5f}")
check("null entropy is ~1.0", abs(null_large["normalized_entropy"] - 1.0) < 1e-3,
      f"got {null_large['normalized_entropy']:.5f}")

print("\n[coverage table]")
# Layer where 2 of 8 experts carry 90% of traffic.
counts = np.array([[90, 90, 5, 5, 4, 3, 2, 1]], dtype=np.int64)
cov = coverage_table(counts, [8], levels=(0.9,))
check("coverage finds the minimal expert count",
      int(cov.iloc[0]["mean_experts"]) == 2, f"got {cov.iloc[0]['mean_experts']}")
check("coverage reports the right fraction of the layer",
      abs(cov.iloc[0]["mean_fraction"] - 2 / 8) < 1e-12)

def build_trace(pick_fn, n_tokens: int = 400, sites=(0, 1)) -> tuple[pd.DataFrame, np.ndarray]:
    """Build a trace where token t at any layer requests pick_fn(t)."""
    rows = []
    for token in range(n_tokens):
        for site in sites:
            for k, expert in enumerate(pick_fn(token)):
                rows.append({"token_uid": token, "slot_k": k, "site_idx": site,
                             "expert_id": expert, "layer_idx": site})
    frame = pd.DataFrame(rows)
    counts = np.zeros((len(sites), 8), dtype=np.int64)
    np.add.at(counts, (frame["site_idx"].to_numpy(), frame["expert_id"].to_numpy()), 1)
    return frame, counts


print("\n[analyze_locality: PHASED workload -- LRU should win]")
# First half uses experts (0,1), second half uses (2,3). Global frequency is a
# 4-way tie, so static pinning of 2 serves only half the trace; LRU tracks the
# phase and misses only at the boundary.
trace, counts = build_trace(lambda t: (0, 1) if t < 200 else (2, 3))
res = analyze_locality(trace, counts, [8, 8], capacities=[2, 4], include_belady=True)
at2 = res.table.set_index("capacity").loc[2]
check("locality table has one row per capacity", len(res.table) == 2)
check("top_k inferred from the trace", res.top_k == 2, f"got {res.top_k}")
check("capacity 4 holds the whole working set -> near-perfect LRU",
      res.table.set_index("capacity").loc[4, "lru"] > 0.99,
      f"got {res.table.set_index('capacity').loc[4, 'lru']}")
check("phased workload: LRU clearly beats static pinning",
      at2["lru"] > at2["static"] + 0.4, f"lru={at2['lru']:.3f} static={at2['static']:.3f}")
check("phased workload: static pinning stuck near 50%",
      abs(at2["static"] - 0.5) < 0.01, f"got {at2['static']:.3f}")

print("\n[analyze_locality: INTERLEAVED workload -- static should win]")
# A cold pair intrudes every 4th token, evicting the hot pair each time. This is
# the opposite regime, and the simulators must show it rather than always
# flattering the dynamic policy.
trace, counts = build_trace(lambda t: (0, 1) if t % 4 else (2, 3))
res_i = analyze_locality(trace, counts, [8, 8], capacities=[2], include_belady=True)
at2i = res_i.table.set_index("capacity").loc[2]
check("interleaved workload: static beats LRU (thrashing)",
      at2i["static"] > at2i["lru"], f"static={at2i['static']:.3f} lru={at2i['lru']:.3f}")
check("interleaved workload: static captures its exact 75% share",
      abs(at2i["static"] - 0.75) < 1e-9, f"got {at2i['static']:.4f}")
# Belady bounds LRU (both start cold). It does NOT bound static, which is
# modelled as pre-loaded and so pays no compulsory misses -- see the note in
# simulate_static. Here that gap is 3 accesses out of 1600.
check("interleaved workload: Belady bounds LRU",
      at2i["belady"] >= at2i["lru"] - 1e-9,
      f"belady={at2i['belady']:.4f} lru={at2i['lru']:.4f}")
check("static's edge over Belady is only the compulsory-miss warm-up",
      at2i["static"] - at2i["belady"] < 0.01,
      f"static={at2i['static']:.4f} belady={at2i['belady']:.4f}")
res = res  # keep the phased result for the checks below
check("all rates within [0, 1]",
      bool(((res.table[["static", "lru", "belady"]] >= 0).all().all())
           and ((res.table[["static", "lru", "belady"]] <= 1).all().all())))
check("formatted table renders", "LRU" in res.format() and "Belady" in res.format())

def build_mixed_trace(pick_by_site: dict[int, object], n_tokens: int = 400):
    """Trace where each site has its own pick function. Returns (frame, counts)."""
    rows = []
    for token in range(n_tokens):
        for site, pick_fn in pick_by_site.items():
            for k, expert in enumerate(pick_fn(token)):
                rows.append({"token_uid": token, "slot_k": k, "site_idx": site,
                             "expert_id": expert, "layer_idx": site})
    frame = pd.DataFrame(rows)
    counts = np.zeros((len(pick_by_site), 8), dtype=np.int64)
    np.add.at(counts, (frame["site_idx"].to_numpy(), frame["expert_id"].to_numpy()), 1)
    return frame, counts


print("\n[counts_from_trace]")
trace, counts = build_trace(lambda t: (0, 1) if t < 200 else (2, 3))
rebuilt = counts_from_trace(trace, counts.shape[0], counts.shape[1])
check("counts rebuilt from the trace match the direct tally",
      bool((rebuilt == counts).all()), f"max diff {np.abs(rebuilt - counts).max()}")
check("rebuilt counts total equals the trace row count",
      int(rebuilt.sum()) == len(trace), f"{rebuilt.sum()} vs {len(trace)}")

print("\n[_pick_sites]")
check("no subsampling when the group already fits", _pick_sites([3, 7, 9], 4) == [3, 7, 9])
check("subsampling keeps the endpoints", _pick_sites(list(range(32)), 4)[0] == 0
      and _pick_sites(list(range(32)), 4)[-1] == 31,
      f"got {_pick_sites(list(range(32)), 4)}")
check("subsampling returns at most max_sites", len(_pick_sites(list(range(32)), 4)) <= 4)
check("max_sites None means every layer", _pick_sites([1, 2, 3], None) == [1, 2, 3])

print("\n[working_set_table]")
# Site 0 always picks the same 2 of 8 experts -> trivial at top_k=2.
# Site 1 cycles over 6 experts -> diverse.
trace, _ = build_mixed_trace({
    0: lambda t: (0, 1),
    1: lambda t: (t % 6, (t + 1) % 6),
})
ws = working_set_table(trace, top_k=2, experts_per_site=[8, 8]).set_index("site_idx")
check("fixed-pair layer has working set 2", int(ws.loc[0, "working_set"]) == 2,
      f"got {ws.loc[0, 'working_set']}")
check("fixed-pair layer is flagged cache-trivial", bool(ws.loc[0, "cache_trivial"]))
check("fixed-pair layer uses only 2 experts", int(ws.loc[0, "used_experts"]) == 2)
check("fixed-pair layer's top2 share is 100%", abs(ws.loc[0, "top_k_share"] - 1.0) < 1e-12)
check("cycling layer is not cache-trivial", not bool(ws.loc[1, "cache_trivial"]))
check("cycling layer's working set exceeds top_k", int(ws.loc[1, "working_set"]) > 2,
      f"got {ws.loc[1, 'working_set']}")
check("n_experts comes from experts_per_site, not the trace",
      int(ws.loc[0, "n_experts"]) == 8, f"got {ws.loc[0, 'n_experts']}")

# A single stray dispatch must not promote a trivial layer to diverse: that is
# what the 99% coverage threshold is for.
trace_stray, _ = build_mixed_trace({0: lambda t: (0, 1) if t else (4, 5)}, n_tokens=400)
ws_stray = working_set_table(trace_stray, top_k=2, experts_per_site=[8]).set_index("site_idx")
check("one stray dispatch in 400 tokens does not un-trivialise a layer",
      bool(ws_stray.loc[0, "cache_trivial"]),
      f"working_set={ws_stray.loc[0, 'working_set']}, used={ws_stray.loc[0, 'used_experts']}")
check("but the stray expert is still counted as used",
      int(ws_stray.loc[0, "used_experts"]) == 4, f"got {ws_stray.loc[0, 'used_experts']}")
# At coverage 1.0 nothing is allowed to be dropped, so the same layer is diverse.
ws_strict = working_set_table(
    trace_stray, top_k=2, experts_per_site=[8], coverage=1.0
).set_index("site_idx")
check("coverage=1.0 counts the stray expert into the working set",
      not bool(ws_strict.loc[0, "cache_trivial"]),
      f"working_set={ws_strict.loc[0, 'working_set']}")

print("\n[layer split changes the policy comparison]")
# Two trivial layers (fixed pair) and two phased layers. The trivial ones are at
# 100% under every policy, so averaging them in must halve the LRU-static gap.
trace, counts4 = build_mixed_trace({
    0: lambda t: (0, 1),
    1: lambda t: (0, 1),
    2: lambda t: (0, 1) if t < 200 else (2, 3),
    3: lambda t: (0, 1) if t < 200 else (2, 3),
})
ws4 = working_set_table(trace, top_k=2, experts_per_site=[8] * 4)
trivial = ws4.loc[ws4["cache_trivial"], "site_idx"].tolist()
diverse = ws4.loc[~ws4["cache_trivial"], "site_idx"].tolist()
check("the split finds exactly the fixed-pair layers", trivial == [0, 1],
      f"got trivial={trivial} diverse={diverse}")
check("the split finds exactly the phased layers", diverse == [2, 3])

experts4 = [8] * 4
res_all = analyze_locality(trace, counts4, experts4, capacities=[2], sites=[0, 1, 2, 3])
res_div = analyze_locality(trace, counts4, experts4, capacities=[2], sites=diverse)
res_triv = analyze_locality(trace, counts4, experts4, capacities=[2], sites=trivial)
gap_all = float(res_all.table.iloc[0]["lru"] - res_all.table.iloc[0]["static"])
gap_div = float(res_div.table.iloc[0]["lru"] - res_div.table.iloc[0]["static"])
gap_triv = float(res_triv.table.iloc[0]["lru"] - res_triv.table.iloc[0]["static"])
# The only gap a cache-trivial layer can show is the warm-up asymmetry noted in
# simulate_static: static is modelled as pre-loaded, LRU starts cold and misses
# its first top_k accesses. Here that is exactly 2 misses out of 400*2 = 800.
check("cache-trivial layers show only the compulsory-miss warm-up gap",
      abs(gap_triv + 2 / 800) < 1e-9, f"got {gap_triv:.6f}, expected {-2/800:.6f}")
check("cache-trivial layers are ~100% under LRU",
      res_triv.table.iloc[0]["lru"] > 0.99, f"got {res_triv.table.iloc[0]['lru']:.4f}")
check("diverse-only gap is larger than the all-layer gap", gap_div > gap_all + 0.1,
      f"diverse={gap_div:.4f} all={gap_all:.4f}")
check("the all-layer gap is the mean of the two groups (dilution, not signal)",
      abs(gap_all - (gap_div + gap_triv) / 2) < 1e-9,
      f"all={gap_all:.6f} mean={(gap_div + gap_triv) / 2:.6f}")

print("\n[extraction cache]")
shared: dict = {}
res_cached = analyze_locality(trace, counts4, experts4, capacities=[2],
                              sites=diverse, cache=shared)
check("cache is populated with the simulated layers", sorted(shared) == diverse,
      f"got {sorted(shared)}")
check("cached run matches the uncached run",
      abs(float(res_cached.table.iloc[0]["lru"]) - float(res_div.table.iloc[0]["lru"])) < 1e-12)
res_reuse = analyze_locality(trace, counts4, experts4, capacities=[2],
                             sites=[0, 1, 2, 3], cache=shared)
check("reusing a partially-filled cache fills in the rest", sorted(shared) == [0, 1, 2, 3],
      f"got {sorted(shared)}")
check("reused cache gives the same all-layer result",
      abs(float(res_reuse.table.iloc[0]["lru"]) - float(res_all.table.iloc[0]["lru"])) < 1e-12)

print("\n[coverage_table site restriction]")
# Layer 0 needs 1 expert for 90%, layer 1 needs 4. Restricting to layer 1 must
# report 4, not the 2.5 average.
counts_cov = np.array([
    [100, 0, 0, 0, 0, 0, 0, 0],
    [25, 25, 25, 25, 0, 0, 0, 0],
], dtype=np.int64)
cov_all = coverage_table(counts_cov, [8, 8], levels=(0.9,))
cov_one = coverage_table(counts_cov, [8, 8], levels=(0.9,), sites=[1])
check("unrestricted coverage averages both layers",
      abs(cov_all.iloc[0]["mean_experts"] - 2.5) < 1e-12,
      f"got {cov_all.iloc[0]['mean_experts']}")
check("restricted coverage reports only the requested layer",
      abs(cov_one.iloc[0]["mean_experts"] - 4.0) < 1e-12,
      f"got {cov_one.iloc[0]['mean_experts']}")

print("\n" + "=" * 60)
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("analyze tests passed")
