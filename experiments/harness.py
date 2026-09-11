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
from scheduler.simulate import (
    EvictionDivergenceReport,
    SimulationResult,
    eviction_divergence_report,
    run_energy_aware_evict,
    run_hbm_only,
    run_naive,
)

#: pJ -> mJ. 1 mJ = 1e9 pJ (1e-3 J / 1e-12 J-per-pJ). Kept as an explicit,
#: named conversion rather than a bare 1e-9 at the call site -- see
#: memsim/constants.py's own rationale for why unit mixups are the recurring
#: silent bug in energy code.
_PJ_TO_MJ = 1e-9

#: ns -> ms. 1 ms = 1e6 ns. Same rationale as _PJ_TO_MJ, same class of bug
#: CLAUDE.md flags (ns/us/ms mixups in latency code, not just energy code).
_NS_TO_MS = 1e-6

#: Above this, avg_latency_ms_per_token is flagged implausible rather than
#: silently trusted. Not a claim that latencies above this are impossible --
#: this simulator's own documented "no overlap, one dispatch at a time"
#: model (scheduler/README.md's "What this does not model") combined with a
#: real, slow measured CXL bandwidth CAN legitimately produce numbers this
#: large (a real run: ~6.8 s/token on the Mixtral decode trace, walked back
#: to 2.36 GB/s CXL bandwidth x 179,090 sequential 352 MB fetches with zero
#: concurrency -- see experiments/README.md). The point of this ceiling is to
#: force that explanation to be checked and stated explicitly every time,
#: not to assert the number is wrong.
LATENCY_SANITY_CEILING_MS = 1000.0


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
    avg_latency_ms_per_token: float
    throughput_tokens_per_sec: float
    latency_plausible: bool
    latency_warning: str | None

    @classmethod
    def from_simulation(cls, config: str, result: SimulationResult, n_tokens: int) -> "ConfigResult":
        total_latency_ns = result.total_latency_ns
        avg_latency_ns = total_latency_ns / n_tokens if n_tokens else math.nan
        avg_latency_ms = avg_latency_ns * _NS_TO_MS
        # total_latency_ns is nanoseconds; seconds = ns * 1e-9; tokens/sec =
        # n_tokens / seconds. Written as one expression so the 1e9 doesn't
        # separately need to be gotten right at two call sites.
        throughput = n_tokens / (total_latency_ns * 1e-9) if total_latency_ns > 0 else math.nan

        plausible = not (avg_latency_ms > LATENCY_SANITY_CEILING_MS)  # NaN-safe: `not (nan > x)` is True
        warning = None
        if not plausible:
            warning = (
                f"avg latency {avg_latency_ms:,.1f} ms/token exceeds the "
                f"{LATENCY_SANITY_CEILING_MS:,.0f} ms/token sanity ceiling. Before trusting this "
                "number: check units first (ns vs ms vs s), then check whether this simulator's "
                "documented no-overlap/no-concurrency model (scheduler/README.md) plus a slow "
                "measured tier bandwidth explains it -- see LATENCY_SANITY_CEILING_MS's docstring "
                "in experiments/harness.py for a real example this happened on."
            )

        return cls(
            config=config,
            n_tokens=n_tokens,
            n_dispatches=result.n_dispatches,
            hit_rate=result.hit_rate,
            total_energy_pj=result.total_energy_pj,
            total_energy_mj=result.total_energy_pj * _PJ_TO_MJ,
            total_latency_ns=total_latency_ns,
            avg_latency_ns_per_token=avg_latency_ns,
            avg_latency_ms_per_token=avg_latency_ms,
            throughput_tokens_per_sec=throughput,
            latency_plausible=plausible,
            latency_warning=warning,
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
    eviction_divergence: EvictionDivergenceReport

    #: Below this, docs.md 6 checkpoint 3's energy-aware-vs-naive gap is
    #: flagged as potentially noise rather than a clean win -- the mechanism
    #: might still be real (this project's own real Mixtral result passed at
    #: 0.23%, right at this edge -- see scheduler/README.md's "Checkpoint 3
    #: result"), but a marginal gap needs the divergence-rate evidence
    #: (eviction_divergence_report) to back it up, not just the percentage
    #: alone.
    MARGINAL_ENERGY_GAP_PCT = 1.0

    @property
    def energy_gap_pct(self) -> float:
        """(naive - energy_aware) / naive * 100. Positive = energy-aware used
        less energy (a win); negative = it used more (checkpoint 3 fails).
        """
        naive_energy = self.hbm_cxl_naive.total_energy_pj
        if naive_energy == 0 or math.isnan(naive_energy):
            return math.nan
        return 100.0 * (naive_energy - self.hbm_cxl_energy_aware.total_energy_pj) / naive_energy

    @property
    def energy_gap_is_marginal(self) -> bool:
        """True if the energy gap is small enough that it could plausibly be
        noise rather than the eviction mechanism doing real work -- check
        eviction_divergence_report for the actual evidence either way.
        """
        gap = self.energy_gap_pct
        return not math.isnan(gap) and abs(gap) < self.MARGINAL_ENERGY_GAP_PCT

    def to_dict(self) -> dict:
        return {
            "run_name": self.run_name,
            "configs": {
                "hbm_only": asdict(self.hbm_only),
                "hbm_cxl_naive": asdict(self.hbm_cxl_naive),
                "hbm_cxl_energy_aware": asdict(self.hbm_cxl_energy_aware),
            },
            "checkpoint3": {
                "energy_gap_pct": self.energy_gap_pct,
                "energy_gap_is_marginal": self.energy_gap_is_marginal,
                "marginal_threshold_pct": self.MARGINAL_ENERGY_GAP_PCT,
            },
            "eviction_divergence": {
                "total_eviction_events": self.eviction_divergence.total_eviction_events,
                "divergent_eviction_events": self.eviction_divergence.divergent_eviction_events,
                "divergence_rate": self.eviction_divergence.divergence_rate,
                "examples": [asdict(ex) for ex in self.eviction_divergence.examples],
            },
            "units": {
                "energy": "pJ (also reported as mJ for readability)",
                "latency": "ns (also reported as ms for readability)",
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

    # The direct evidence for whether the eviction mechanism is doing
    # anything on this trace, independent of whatever the total-energy delta
    # happens to be -- see eviction_divergence_report's docstring and
    # dashboard/README.md's diagnostics panel section.
    divergence = eviction_divergence_report(naive_sim, evict_sim, trace, dispatch_counts)

    return ComparisonResult(
        run_name=run_dir.name,
        hbm_only=ConfigResult.from_simulation("hbm_only", hbm_only_sim, n_tokens),
        hbm_cxl_naive=ConfigResult.from_simulation("hbm_cxl_naive", naive_sim, n_tokens),
        hbm_cxl_energy_aware=ConfigResult.from_simulation("hbm_cxl_energy_aware", evict_sim, n_tokens),
        eviction_divergence=divergence,
    )
