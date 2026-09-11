"""Runs the three docs.md 4.6 configs over one stage-1 trace and computes
throughput, average latency, and total energy for each.

THE THREE CONFIGS, AND WHY EACH IS JUST A CACHE-CAPACITY CHOICE:

  hbm_only          -- cache capacity = every expert at every site, so no
                        dispatch is ever a miss and no CXL fetch ever
                        happens. This is memsim/README.md's own definition:
                        "everything resident in GPU HBM (or fails once it
                        doesn't fit, which is the motivating case)" -- the
                        idealised, usually-infeasible baseline the whole
                        project exists to offer an alternative to.
  hbm_cxl_naive     -- cache capacity = stage 1's real hot-expert count per
                        site. Cold misses fetch over CXL immediately, no
                        energy weighting -- scheduler.simulate.run_naive.
  hbm_cxl_energy_aware -- same capacity, but eviction is stage 3's validated
                        energy-aware policy (scheduler.simulate.
                        run_energy_aware_evict), which passed docs.md 6
                        checkpoint 3 on the real Mixtral decode trace (see
                        scheduler/README.md).

No new simulation logic lives here -- every config is exactly one call into
the already-tested stage-3 simulator with a specific cache_capacity dict.
This module's own job is the three NEW metrics docs.md 4.6 asks for
(throughput, average per-token latency) that stage 3's SimulationResult
does not already expose, plus running all three and packaging the result.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from scheduler.model import (
    CostModel,
    load_expert_dispatch_counts,
    load_expert_weight_bytes,
    load_hot_experts,
    load_tier_model,
    load_trace,
)
from scheduler.simulate import SimulationResult, run_energy_aware_evict, run_hbm_only, run_naive

#: pJ -> mJ. 1 mJ = 1e9 pJ (1e-3 J / 1e-12 J-per-pJ). Kept as an explicit,
#: named conversion rather than a bare 1e-9 at the call site -- see
#: memsim/constants.py's own rationale for why unit mixups are the recurring
#: silent bug in energy code.
_PJ_TO_MJ = 1e-9


@dataclass(frozen=True)
class ConfigResult:
    """One config's docs.md 4.6 figures, derived from a stage-3 SimulationResult."""

    config: str
    n_tokens: int
    n_dispatches: int
    hit_rate: float
    total_energy_pj: float
    total_energy_mj: float
    total_latency_ns: float
    avg_latency_ns_per_token: float
    throughput_tokens_per_sec: float

    @classmethod
    def from_simulation(cls, config: str, result: SimulationResult, n_tokens: int) -> "ConfigResult":
        total_latency_ns = result.total_latency_ns
        avg_latency = total_latency_ns / n_tokens if n_tokens else math.nan
        # total_latency_ns is nanoseconds; seconds = ns * 1e-9; tokens/sec =
        # n_tokens / seconds. Written as one expression so the 1e9 doesn't
        # separately need to be gotten right at two call sites.
        throughput = n_tokens / (total_latency_ns * 1e-9) if total_latency_ns > 0 else math.nan
        return cls(
            config=config,
            n_tokens=n_tokens,
            n_dispatches=result.n_dispatches,
            hit_rate=result.hit_rate,
            total_energy_pj=result.total_energy_pj,
            total_energy_mj=result.total_energy_pj * _PJ_TO_MJ,
            total_latency_ns=total_latency_ns,
            avg_latency_ns_per_token=avg_latency,
            throughput_tokens_per_sec=throughput,
        )


@dataclass(frozen=True)
class ComparisonResult:
    """All three configs' figures for one stage-1 run, plus enough provenance
    to know exactly what produced them without re-deriving it.
    """

    run_name: str
    hbm_only: ConfigResult
    hbm_cxl_naive: ConfigResult
    hbm_cxl_energy_aware: ConfigResult

    def to_dict(self) -> dict:
        return {
            "run_name": self.run_name,
            "configs": {
                "hbm_only": asdict(self.hbm_only),
                "hbm_cxl_naive": asdict(self.hbm_cxl_naive),
                "hbm_cxl_energy_aware": asdict(self.hbm_cxl_energy_aware),
            },
            "units": {
                "energy": "pJ (also reported as mJ for readability)",
                "latency": "ns",
                "throughput": "tokens/sec",
            },
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def run_comparison(run_dir: str | Path, tier_model_path: str | Path) -> ComparisonResult:
    """Run all three docs.md 4.6 configs over one stage-1 run directory.

    Args:
        run_dir: stage-1 run directory (trace.parquet + hot_cold.csv).
        tier_model_path: memsim/tier_model.json (stage 2 output).
    """
    run_dir = Path(run_dir)
    trace = load_trace(run_dir / "trace.parquet")
    weight_bytes = load_expert_weight_bytes(run_dir / "hot_cold.csv")
    hot = load_hot_experts(run_dir / "hot_cold.csv")
    dispatch_counts = load_expert_dispatch_counts(run_dir / "hot_cold.csv")
    tiers = load_tier_model(tier_model_path)
    cost_model = CostModel(weight_bytes, tiers)

    n_tokens = int(trace["token_uid"].nunique())
    hot_capacity = {site: len(experts) for site, experts in hot.items()}

    # hbm_only is NOT run_naive() with a full-size cache -- that still starts
    # empty and pays a genuine cold miss the first time each expert is seen.
    # run_hbm_only() skips the cache entirely: every dispatch is served from
    # HBM by construction, matching a real HBM-only deployment that pre-loads
    # the whole model before serving starts. See its docstring in
    # scheduler/simulate.py for the bug this replaced (caught by
    # experiments/selftest.py).
    hbm_only_sim = run_hbm_only(trace, cost_model)
    naive_sim = run_naive(trace, cost_model, hot_capacity)
    evict_sim = run_energy_aware_evict(trace, cost_model, hot_capacity, dispatch_counts)

    return ComparisonResult(
        run_name=run_dir.name,
        hbm_only=ConfigResult.from_simulation("hbm_only", hbm_only_sim, n_tokens),
        hbm_cxl_naive=ConfigResult.from_simulation("hbm_cxl_naive", naive_sim, n_tokens),
        hbm_cxl_energy_aware=ConfigResult.from_simulation("hbm_cxl_energy_aware", evict_sim, n_tokens),
    )
