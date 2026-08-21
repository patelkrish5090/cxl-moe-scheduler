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
    analyze_locality, coverage_table, simulate_belady, simulate_lru, simulate_static,
    uniform_null_stats,
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

print("\n" + "=" * 60)
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    raise SystemExit(1)
print("analyze tests passed")
