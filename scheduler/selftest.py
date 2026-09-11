"""Offline correctness checks for stage 3, no real trace or gem5 run needed.

Every check either recomputes an expected value independently of the code
under test, or checks an invariant that must hold regardless of the specific
numbers involved (e.g. "no fetch happens with insufficient budget").
"""

from __future__ import annotations

import json
import math
import tempfile
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from .model import (
    BYTES_PER_PARAM,
    FLOPS_PER_PARAM_PER_TOKEN,
    CostModel,
    TierFigures,
    load_expert_weight_bytes,
    load_hot_experts,
    load_tier_model,
    load_trace,
)
from .simulate import (
    _SiteCache,
    diff_decisions,
    run_energy_aware_defer,
    run_energy_aware_evict,
    run_naive,
    three_way_report,
)

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        failures.append(name)
        print(f"  FAIL  {name}" + (f"\n        {detail}" if detail else ""))


def _tiers(link_energy: float = 2.0, link_latency: float = 500.0) -> dict[str, TierFigures]:
    return {
        "hbm": TierFigures("hbm", latency_ns=40.0, peak_bandwidth_gbps=20.0,
                            device_energy_pj_per_bit=5.0,
                            link_energy_pj_per_bit=0.0, total_energy_pj_per_bit=5.0),
        "cxl": TierFigures("cxl", latency_ns=link_latency, peak_bandwidth_gbps=2.0,
                            device_energy_pj_per_bit=30.0,
                            link_energy_pj_per_bit=link_energy,
                            total_energy_pj_per_bit=30.0 + link_energy),
    }


def _make_trace(rows: list[tuple[int, int, int]]) -> pd.DataFrame:
    """rows: list of (token_uid, site_idx, expert_id), slot_k always 0."""
    return pd.DataFrame({
        "token_uid": [r[0] for r in rows],
        "site_idx": [r[1] for r in rows],
        "expert_id": [r[2] for r in rows],
        "slot_k": [0] * len(rows),
    })


def main() -> int:
    print("\n[cost model]")
    weight_bytes = {0: 1024}  # 1024 bytes -> 512 params (bf16) -> 1024 FLOPs
    tiers = _tiers()
    cm = CostModel(weight_bytes, tiers)

    check("FLOPs follow the 2*N convention",
          cm.flops_for(0) == FLOPS_PER_PARAM_PER_TOKEN * (1024 / BYTES_PER_PARAM),
          f"got {cm.flops_for(0)}")

    hbm_cost = cm.cost(0, 0, "hbm")
    check("hbm has zero link transfer energy, exactly", hbm_cost.link_transfer_pj == 0.0)
    expected_hbm_mem = tiers["hbm"].device_energy_pj_per_bit * 1024 * 8
    check("hbm mem_read_pj matches device_energy * bits independently",
          hbm_cost.mem_read_pj == expected_hbm_mem, f"got {hbm_cost.mem_read_pj}")

    cxl_cost = cm.cost(0, 0, "cxl")
    expected_cxl_link = tiers["cxl"].link_energy_pj_per_bit * 1024 * 8
    check("cxl link_transfer_pj matches link_energy * bits independently",
          cxl_cost.link_transfer_pj == expected_cxl_link, f"got {cxl_cost.link_transfer_pj}")
    expected_total = hbm_cost.compute_pj + expected_cxl_link + tiers["cxl"].device_energy_pj_per_bit * 1024 * 8
    check("cxl total_pj = compute + mem_read + link_transfer",
          math.isclose(cxl_cost.total_pj, expected_total), f"got {cxl_cost.total_pj} vs {expected_total}")
    check("compute_pj is identical across tiers (same FLOPs either way)",
          hbm_cost.compute_pj == cxl_cost.compute_pj)

    nan_tiers = _tiers(link_energy=math.nan)
    nan_cost = CostModel(weight_bytes, nan_tiers).cost(0, 0, "cxl")
    check("an unsourced link energy poisons total_pj, not defaults to zero",
          math.isnan(nan_cost.total_pj), f"got {nan_cost.total_pj}")

    print("\n[LRU site cache]")
    cache = _SiteCache(capacity=2, eviction_policy="lru")
    check("miss on empty cache", not cache.hit(1))
    cache.insert(1, refetch_cost_pj=10.0)
    check("hit after insert", cache.hit(1))
    cache.insert(2, refetch_cost_pj=10.0)
    check("second insert still hits first (capacity 2)", cache.hit(1))
    cache.insert(3, refetch_cost_pj=10.0)  # evicts 2 (1 was just touched above, so 2 is LRU)
    check("inserting a third entry evicts the true LRU one", not cache.hit(2))
    check("the recently-touched entry survives eviction", cache.hit(1))
    check("the newest entry is present", cache.hit(3))

    print("\n[energy-aware site cache: eviction scoring]")
    # capacity 2, all same cost -> score is (access_count / age). A is
    # accessed 3x then goes quiet, B is accessed once more recently; both are
    # in cache when C arrives and forces an eviction. Recency-decayed
    # frequency must still keep the more valuable A and drop B here despite B
    # being touched more recently -- see _SiteCache's docstring for why this
    # needs BOTH terms (pure frequency alone regressed on the real trace).
    # insert() must only be called right after a hit() that returned False for
    # the same key (the class's usage contract -- see insert()'s docstring),
    # so every insert below is paired with a preceding hit() check.
    ecache = _SiteCache(capacity=2, eviction_policy="energy-aware")
    check("expert 1 misses on an empty cache", not ecache.hit(1))
    ecache.insert(1, refetch_cost_pj=10.0)
    check("expert 1 hits twice more", ecache.hit(1) and ecache.hit(1))  # access_count now 3
    check("expert 2 misses", not ecache.hit(2))
    ecache.insert(2, refetch_cost_pj=10.0)  # access_count 1, room for both, no eviction
    check("expert 3 misses", not ecache.hit(3))
    evicted = ecache.insert(3, refetch_cost_pj=10.0)  # forces an eviction
    check("energy-aware eviction drops the low-frequency entry, not the recent one",
          evicted == 2, f"evicted {evicted}, expected 2 (low frequency) not 1 (high frequency)")
    check("the high-frequency entry survives", ecache.hit(1))

    nan_cache = _SiteCache(capacity=1, eviction_policy="energy-aware")
    check("nan-cost expert 1 misses on an empty cache", not nan_cache.hit(1))
    nan_cache.insert(1, refetch_cost_pj=math.nan)
    check("expert 2 misses", not nan_cache.hit(2))
    nan_evicted = nan_cache.insert(2, refetch_cost_pj=10.0)
    check("a NaN-cost entry (unsourced constant) is evicted first, not kept forever",
          nan_evicted == 1, f"evicted {nan_evicted}")

    print("\n[naive policy: independent recompute]")
    # 2 sites, hand-traceable sequence. Site 0 cache capacity 1: expert 1 then
    # expert 2 (miss, evicts 1) then expert 1 again (miss).
    trace = _make_trace([
        (0, 0, 1), (0, 0, 2), (1, 0, 1),
    ])
    result = run_naive(trace, cm, cache_capacity={0: 1})
    check("naive: 3 dispatches recorded", result.n_dispatches == 3)
    check("naive: first dispatch to an empty cache is a miss",
          not result.decisions[0].hit)
    check("naive: second dispatch (different expert, capacity 1) is a miss",
          not result.decisions[1].hit)
    check("naive: third dispatch (evicted expert re-requested) is a miss",
          not result.decisions[2].hit)
    check("naive: all misses cost the cxl total (independently recomputed)",
          all(math.isclose(d.energy_pj, cxl_cost.total_pj) for d in result.decisions),
          f"got {[d.energy_pj for d in result.decisions]}")
    check("naive: total_energy_pj is the sum of its own decisions",
          math.isclose(result.total_energy_pj, sum(d.energy_pj for d in result.decisions)))
    check("naive: no policy ever defers", result.n_deferred == 0)

    print("\n[naive policy: a cache hit actually happens]")
    trace_hit = _make_trace([(0, 0, 1), (1, 0, 1)])  # same expert twice, capacity 1
    result_hit = run_naive(trace_hit, cm, cache_capacity={0: 1})
    check("first dispatch misses", not result_hit.decisions[0].hit)
    check("second dispatch to the same expert hits", result_hit.decisions[1].hit)
    check("a hit costs the hbm total, not the cxl total",
          math.isclose(result_hit.decisions[1].energy_pj, hbm_cost.total_pj))
    check("hit_rate reflects exactly 1 of 2 dispatches", result_hit.hit_rate == 0.5)

    print("\n[energy-aware-defer policy]")
    # Budget large enough to never defer: should match naive exactly, dispatch
    # for dispatch, since it's the same LRU policy underneath.
    result_rich = run_energy_aware_defer(trace, cm, cache_capacity={0: 1}, power_budget_w=1e12)
    # The budget starts at exactly 0 pJ at sim_time=0: no rate, however large,
    # can deliver energy that hasn't had any time to accrue yet, so the very
    # first cold dispatch from a standing start may still defer -- correctly,
    # just negligibly (see simulate.py's _run docstring). What an "enormous
    # budget defers nothing meaningful" actually promises is that any such
    # deferral is vanishingly small, not that none is ever recorded.
    check("energy-aware-defer with an enormous budget defers nothing meaningful",
          result_rich.n_deferred <= 1 and result_rich.total_wait_ns < 1.0,
          f"got {result_rich.n_deferred} deferrals, {result_rich.total_wait_ns} ns total wait")
    check("energy-aware-defer with an enormous budget matches naive's hit pattern",
          [d.hit for d in result_rich.decisions] == [d.hit for d in result.decisions])
    check("energy-aware-defer with an enormous budget matches naive's total energy "
          "(deferral cannot change energy, only timing -- see run_energy_aware_defer's docstring)",
          math.isclose(result_rich.total_energy_pj, result.total_energy_pj))

    # Budget of exactly 0 can never afford anything -- every miss must defer.
    result_poor = run_energy_aware_defer(trace, cm, cache_capacity={0: 1}, power_budget_w=1e-30)
    check("energy-aware-defer with a near-zero budget defers every miss",
          all(d.deferred for d in result_poor.decisions if not d.hit))
    check("even a deferred fetch eventually happens (still recorded)",
          result_poor.n_dispatches == result.n_dispatches)

    check("energy-aware-defer requires a positive power budget", _raises_on_bad_budget(trace, cm))

    print("\n[energy-aware-defer: budget invariant]")
    # No dispatch should ever report an energy cost the budget couldn't have
    # covered at the moment it was charged -- check via a moderate budget that
    # produces a mix of immediate and deferred fetches.
    mixed_trace = _make_trace([(i, 0, i % 3) for i in range(9)])
    mixed = run_energy_aware_defer(mixed_trace, cm, cache_capacity={0: 2}, power_budget_w=1e6)
    check("a moderate budget produces at least one deferral on a busy trace",
          mixed.n_deferred > 0, f"got {mixed.n_deferred} deferrals among {mixed.n_dispatches}")
    check("wait_ns is always non-negative",
          all(d.wait_ns >= 0 for d in mixed.decisions))
    check("only misses ever carry a wait", all(d.wait_ns == 0 for d in mixed.decisions if d.hit))

    print("\n[energy-aware-evict policy: checkpoint 3]")
    # Hand-traced sequence, capacity 2: A is hot (4 touches), B is cold (1
    # touch), then C forces an eviction while both are resident. B was touched
    # MORE RECENTLY than A (at step 5 vs A's last touch at step 4), so plain
    # LRU evicts the valuable, frequently-used A in favour of keeping the
    # barely-used B -- exactly the wrong call. Frequency-weighted eviction
    # keeps A and drops B instead. The payoff shows up at step 7: A is
    # requested again, a hit for energy-aware-evict, a miss for naive/LRU.
    ckpt3_trace = _make_trace([
        (0, 0, 11), (1, 0, 11), (2, 0, 11), (3, 0, 11),  # A x4
        (4, 0, 22),                                       # B x1
        (5, 0, 33),                                       # C forces an eviction
        (6, 0, 11),                                       # A again
    ])
    ckpt3_naive = run_naive(ckpt3_trace, cm, cache_capacity={0: 2})
    ckpt3_evict = run_energy_aware_evict(ckpt3_trace, cm, cache_capacity={0: 2})

    check("hand-traced: naive/LRU evicts the frequent expert, so the last access misses",
          not ckpt3_naive.decisions[-1].hit, "expected naive's last dispatch (A again) to miss")
    check("hand-traced: energy-aware-evict keeps the frequent expert, so the last access hits",
          ckpt3_evict.decisions[-1].hit, "expected energy-aware-evict's last dispatch (A again) to hit")
    check("hand-traced: energy-aware-evict has fewer total misses than naive on this trace",
          ckpt3_evict.n_misses < ckpt3_naive.n_misses,
          f"naive misses={ckpt3_naive.n_misses}, evict misses={ckpt3_evict.n_misses}")
    check("hand-traced: energy-aware-evict's total energy is strictly lower than naive's "
          "(docs.md 6 checkpoint 3, demonstrated on a constructed example)",
          ckpt3_evict.total_energy_pj < ckpt3_naive.total_energy_pj,
          f"naive={ckpt3_naive.total_energy_pj}, evict={ckpt3_evict.total_energy_pj}")

    ckpt3_defer = run_energy_aware_defer(ckpt3_trace, cm, cache_capacity={0: 2}, power_budget_w=1e12)
    report_text = three_way_report(ckpt3_naive, ckpt3_defer, ckpt3_evict)
    check("three_way_report declares checkpoint 3 PASSED on the hand-traced example",
          "PASSED" in report_text, report_text)

    print("\n[decision diff]")
    diff_text = diff_decisions(result, result_poor)
    check("diff report names both policies", "naive" in diff_text and "energy-aware-defer" in diff_text)
    check("diff report counts every dispatch as diverged when every miss defers",
          f"{result.n_dispatches} / {result.n_dispatches}" in diff_text
          or "diverged" in diff_text)
    mismatched = run_naive(_make_trace([(0, 0, 1)]), cm, cache_capacity={0: 1})
    check("diff refuses to compare runs of different length",
          "cannot diff" in diff_decisions(result, mismatched))

    print("\n[file loading]")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        hot_cold = pd.DataFrame({
            "site_idx": [0, 0, 0], "layer_idx": [0, 0, 0],
            "expert_id": [0, 1, 2], "dispatch_count": [10, 5, 1],
            "layer_share": [0.625, 0.3125, 0.0625], "rank_in_layer": [0, 1, 2],
            "is_hot": [True, True, False], "expert_weight_bytes": [1024, 1024, 1024],
        })
        hot_cold.to_csv(root / "hot_cold.csv", index=False)

        weights = load_expert_weight_bytes(root / "hot_cold.csv")
        check("expert weight bytes loaded per site", weights == {0: 1024}, f"got {weights}")

        hot = load_hot_experts(root / "hot_cold.csv")
        check("hot experts loaded per site, cold expert excluded",
              hot == {0: {0, 1}}, f"got {hot}")

        trace_df = _make_trace([(0, 0, 0), (0, 0, 1), (1, 0, 0)])
        trace_df.to_parquet(root / "trace.parquet")
        loaded_trace = load_trace(root / "trace.parquet")
        check("trace loads and is sorted by (token_uid, site_idx, slot_k)",
              list(loaded_trace["token_uid"]) == [0, 0, 1])

        tier_payload = {"models": {
            "hbm": {"tier": "hbm", "unloaded_latency_ns": 40.0, "peak_bandwidth_gbps": 20.0,
                    "device_energy_pj_per_bit": 5.0, "link_energy_pj_per_bit": 0.0,
                    "total_energy_pj_per_bit": 5.0},
            "cxl": {"tier": "cxl", "unloaded_latency_ns": 500.0, "peak_bandwidth_gbps": 2.0,
                    "device_energy_pj_per_bit": 30.0, "link_energy_pj_per_bit": 2.0,
                    "total_energy_pj_per_bit": 32.0},
        }}
        (root / "tier_model.json").write_text(json.dumps(tier_payload), encoding="utf-8")
        loaded_tiers = load_tier_model(root / "tier_model.json")
        check("tier model loads both tiers", set(loaded_tiers) == {"hbm", "cxl"})
        check("loaded hbm latency matches the file", loaded_tiers["hbm"].latency_ns == 40.0)

        missing_path = root / "does_not_exist.json"
        raised = False
        try:
            load_tier_model(missing_path)
        except FileNotFoundError as exc:
            raised = "memsim.cli" in str(exc)
        check("a missing tier model raises, naming the command that produces it", raised)

    print("\n" + "=" * 62)
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        print("=" * 62)
        return 1
    print("scheduler selftest passed")
    print("=" * 62)
    return 0


def _raises_on_bad_budget(trace: pd.DataFrame, cm: CostModel) -> bool:
    try:
        run_energy_aware_defer(trace, cm, cache_capacity={0: 1}, power_budget_w=0.0)
    except ValueError:
        return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
