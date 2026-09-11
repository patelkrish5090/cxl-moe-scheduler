"""Offline correctness checks for stage 4's harness, no real trace or gem5
run needed. This module orchestrates stage 3's already-tested simulator, so
these checks focus on what's NEW here: the throughput/latency metrics
ConfigResult derives, the hbm_only config's cache-capacity trick, and the
JSON round-trip -- not re-deriving stage 3's own cache/energy correctness
(see scheduler/selftest.py for that).
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pandas as pd

from scheduler.simulate import Decision, SimulationResult

from .harness import ComparisonResult, ConfigResult, run_comparison

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        failures.append(name)
        print(f"  FAIL  {name}" + (f"\n        {detail}" if detail else ""))


def _fake_result(latencies_ns: list[float], energies_pj: list[float]) -> SimulationResult:
    """A SimulationResult with hand-chosen per-dispatch latency/energy, sim_time
    accumulated exactly like scheduler.simulate._run does (running total).
    """
    result = SimulationResult(policy="naive")
    running = 0.0
    for i, (lat, en) in enumerate(zip(latencies_ns, energies_pj)):
        running += lat
        result.decisions.append(Decision(
            token_uid=i, site_idx=0, expert_id=0, hit=False, deferred=False,
            energy_pj=en, wait_ns=0.0, sim_time_ns=running,
        ))
    return result


def main() -> int:
    print("\n[ConfigResult: independent recompute]")
    # 4 dispatches, 1 dispatch per token (4 tokens), hand-chosen latencies and
    # energies so throughput/avg-latency/energy can be recomputed by hand.
    sim = _fake_result(latencies_ns=[100.0, 200.0, 300.0, 400.0], energies_pj=[10.0, 20.0, 30.0, 40.0])
    cfg = ConfigResult.from_simulation("test-config", sim, n_tokens=4)

    check("total_latency_ns is the running sum of per-dispatch latencies",
          cfg.total_latency_ns == 1000.0, f"got {cfg.total_latency_ns}")
    check("avg_latency_ns_per_token = total_latency_ns / n_tokens",
          cfg.avg_latency_ns_per_token == 250.0, f"got {cfg.avg_latency_ns_per_token}")
    # throughput = n_tokens / (total_latency_ns * 1e-9 seconds)
    expected_throughput = 4 / (1000.0 * 1e-9)
    check("throughput_tokens_per_sec matches independent recompute",
          math.isclose(cfg.throughput_tokens_per_sec, expected_throughput),
          f"got {cfg.throughput_tokens_per_sec}, expected {expected_throughput}")
    check("total_energy_pj is the sum of per-dispatch energies",
          cfg.total_energy_pj == 100.0, f"got {cfg.total_energy_pj}")
    check("total_energy_mj = total_energy_pj * 1e-9 (pJ -> mJ)",
          math.isclose(cfg.total_energy_mj, 100.0 * 1e-9), f"got {cfg.total_energy_mj}")

    print("\n[ConfigResult: degenerate cases]")
    empty = SimulationResult(policy="naive")
    empty_cfg = ConfigResult.from_simulation("empty", empty, n_tokens=0)
    check("zero tokens gives NaN throughput/latency, not a ZeroDivisionError",
          math.isnan(empty_cfg.throughput_tokens_per_sec) and math.isnan(empty_cfg.avg_latency_ns_per_token))
    check("zero tokens gives zero energy, not NaN (no decisions were made)",
          empty_cfg.total_energy_pj == 0.0)

    print("\n[run_comparison: end-to-end with synthetic fixtures]")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # 2 sites, 4 experts each. Site 0: expert 0 is hit 3x as often as
        # expert 1-3 combined (skewed); site 1 uniform. capacity=2 hot experts
        # per site (matches is_hot below), full capacity=4 (all experts).
        hot_cold = pd.DataFrame({
            "site_idx":            [0, 0, 0, 0, 1, 1, 1, 1],
            "layer_idx":           [0, 0, 0, 0, 1, 1, 1, 1],
            "expert_id":           [0, 1, 2, 3, 0, 1, 2, 3],
            "dispatch_count":      [30, 5, 3, 2, 10, 10, 10, 10],
            "layer_share":         [0.75, 0.125, 0.075, 0.05, 0.25, 0.25, 0.25, 0.25],
            "rank_in_layer":       [0, 1, 2, 3, 0, 1, 2, 3],
            "is_hot":              [True, True, False, False, True, True, False, False],
            "expert_weight_bytes": [1024] * 8,
        })
        hot_cold.to_csv(root / "hot_cold.csv", index=False)

        # 20 tokens, each dispatching to site 0 and site 1 (top_k=1 for
        # simplicity), expert choice weighted toward the skew above.
        import numpy as np
        rng = np.random.default_rng(0)
        rows = []
        for t in range(20):
            e0 = rng.choice([0, 1, 2, 3], p=[0.75, 0.125, 0.075, 0.05])
            e1 = rng.choice([0, 1, 2, 3])
            rows.append((t, 0, int(e0), 0))
            rows.append((t, 1, int(e1), 0))
        trace = pd.DataFrame(rows, columns=["token_uid", "site_idx", "expert_id", "slot_k"])
        trace.to_parquet(root / "trace.parquet")

        tier_model = {"models": {
            "hbm": {"tier": "hbm", "unloaded_latency_ns": 40.0, "peak_bandwidth_gbps": 20.0,
                    "device_energy_pj_per_bit": 5.0, "link_energy_pj_per_bit": 0.0,
                    "total_energy_pj_per_bit": 5.0},
            "cxl": {"tier": "cxl", "unloaded_latency_ns": 500.0, "peak_bandwidth_gbps": 2.0,
                    "device_energy_pj_per_bit": 30.0, "link_energy_pj_per_bit": 2.0,
                    "total_energy_pj_per_bit": 32.0},
        }}
        (root / "tier_model.json").write_text(json.dumps(tier_model), encoding="utf-8")

        comparison = run_comparison(root, root / "tier_model.json")

        check("hbm_only has 100% hit rate (cache capacity = every expert)",
              comparison.hbm_only.hit_rate == 1.0, f"got {comparison.hbm_only.hit_rate}")
        check("hbm_only dispatched the expected number of dispatches (20 tokens x 2 sites)",
              comparison.hbm_only.n_dispatches == 40, f"got {comparison.hbm_only.n_dispatches}")
        check("hbm_cxl_naive has a lower hit rate than hbm_only on a capacity-constrained cache",
              comparison.hbm_cxl_naive.hit_rate <= comparison.hbm_only.hit_rate,
              f"naive={comparison.hbm_cxl_naive.hit_rate}, hbm_only={comparison.hbm_only.hit_rate}")
        check("hbm_only's total energy is lower than the CXL configs' (never pays link/cxl-device energy)",
              comparison.hbm_only.total_energy_pj < comparison.hbm_cxl_naive.total_energy_pj,
              f"hbm_only={comparison.hbm_only.total_energy_pj}, naive={comparison.hbm_cxl_naive.total_energy_pj}")
        check("all three configs saw the same 20 tokens",
              comparison.hbm_only.n_tokens == comparison.hbm_cxl_naive.n_tokens ==
              comparison.hbm_cxl_energy_aware.n_tokens == 20)

        print("\n[JSON round-trip]")
        out_path = comparison.write(root / "result.json")
        check("result file was written", out_path.is_file())
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        check("round-tripped run_name matches", loaded["run_name"] == comparison.run_name)
        check("round-tripped hbm_only hit_rate matches",
              loaded["configs"]["hbm_only"]["hit_rate"] == comparison.hbm_only.hit_rate)
        check("all three config keys present in the written file",
              set(loaded["configs"]) == {"hbm_only", "hbm_cxl_naive", "hbm_cxl_energy_aware"})

    print("\n" + "=" * 62)
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        print("=" * 62)
        return 1
    print("experiments selftest passed")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
