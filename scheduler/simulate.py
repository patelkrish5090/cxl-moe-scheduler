"""Replays a stage-1 trace through a per-site LRU expert cache under one of
two policies, producing a decision log and aggregate energy/latency/throughput
figures.

WHAT THIS DOES NOT MODEL (be upfront about this wherever these numbers are
quoted): dispatches are processed one at a time on a single simulated
timeline, in the trace's stored order. There is no cross-layer pipelining or
overlap between a fetch and the next token's issue, and no modelling of
multiple in-flight fetches. This is a scheduling-*decision* simulator, not a
cycle-accurate execution model -- gem5 already owns that role for the memory
tiers themselves (memsim/). Real hardware would overlap much of this, so
throughput/latency figures from here should be read as directional
(policy A vs policy B), not as an absolute performance prediction.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from .model import CostModel

Policy = Literal["naive", "energy-aware"]

#: W (watts) * ns -> pJ. 1 W = 1 J/s; over dt nanoseconds that is
#: dt * 1e-9 J = dt * 1e-9 * 1e12 pJ = dt * 1e3 pJ.
_WATT_NS_TO_PJ = 1e3


@dataclass
class Decision:
    """The outcome of one (token, site, expert) dispatch."""

    token_uid: int
    site_idx: int
    expert_id: int
    hit: bool
    deferred: bool
    energy_pj: float
    wait_ns: float
    sim_time_ns: float  # clock reading when this dispatch finished


@dataclass
class SimulationResult:
    policy: Policy
    decisions: list[Decision] = field(default_factory=list)

    @property
    def n_dispatches(self) -> int:
        return len(self.decisions)

    @property
    def n_hits(self) -> int:
        return sum(1 for d in self.decisions if d.hit)

    @property
    def n_misses(self) -> int:
        return self.n_dispatches - self.n_hits

    @property
    def n_deferred(self) -> int:
        return sum(1 for d in self.decisions if d.deferred)

    @property
    def hit_rate(self) -> float:
        return self.n_hits / self.n_dispatches if self.n_dispatches else 0.0

    @property
    def total_energy_pj(self) -> float:
        return sum(d.energy_pj for d in self.decisions)

    @property
    def total_wait_ns(self) -> float:
        return sum(d.wait_ns for d in self.decisions)

    @property
    def total_latency_ns(self) -> float:
        """Wall-clock span of the simulated run: the last dispatch's clock
        reading. NaN-safe: if energy accounting produced NaN anywhere, that
        does not affect this, since latency and energy are tracked separately.
        """
        return self.decisions[-1].sim_time_ns if self.decisions else 0.0

    @property
    def implied_avg_power_w(self) -> float:
        """Average power this run's energy would take if delivered evenly over
        its own wall-clock span: total_energy_pj / total_latency_ns is in
        pJ/ns, which is milliwatts, not watts (pJ/ns = 1e-12 J / 1e-9 s =
        1e-3 W) -- divide by _WATT_NS_TO_PJ to get watts. Getting this wrong
        silently reports a number 1000x too small, exactly the kind of
        pJ/nJ/W mixup CLAUDE.md flags as the recurring bug in energy code.

        This is the reference point for choosing --power-budget-w on a real
        trace: passing this run's own naive implied_avg_power_w as the
        energy-aware budget reproduces roughly naive's own pace (near-zero
        deferral); a materially smaller budget is what actually forces the
        scheduler to trade latency for staying under a real power ceiling.
        Picking a budget with no relation to this number (as a first guess
        might) risks either changing nothing or deferring almost everything --
        see scheduler/README.md's worked example.
        """
        if self.total_latency_ns <= 0:
            return 0.0
        return (self.total_energy_pj / self.total_latency_ns) / _WATT_NS_TO_PJ

    def summary(self) -> dict[str, float | int | str]:
        return {
            "policy": self.policy,
            "n_dispatches": self.n_dispatches,
            "n_hits": self.n_hits,
            "n_misses": self.n_misses,
            "hit_rate": self.hit_rate,
            "n_deferred": self.n_deferred,
            "total_energy_pj": self.total_energy_pj,
            "total_wait_ns": self.total_wait_ns,
            "total_latency_ns": self.total_latency_ns,
            "implied_avg_power_w": self.implied_avg_power_w,
        }


class _SiteCache:
    """One MoE layer's LRU expert cache. Starts empty and warms from the
    trace -- capacity is sized from stage 1's hot-expert count, but nothing is
    pinned; this is genuinely LRU-adaptive, matching
    profiler/analyze.py::simulate_lru's semantics.
    """

    def __init__(self, capacity: int):
        self.capacity = max(capacity, 1)
        self._entries: "OrderedDict[int, None]" = OrderedDict()

    def hit(self, expert_id: int) -> bool:
        if expert_id in self._entries:
            self._entries.move_to_end(expert_id)
            return True
        return False

    def insert(self, expert_id: int) -> None:
        self._entries[expert_id] = None
        self._entries.move_to_end(expert_id)
        if len(self._entries) > self.capacity:
            self._entries.popitem(last=False)


def _run(
    trace: pd.DataFrame,
    cost_model: CostModel,
    cache_capacity: dict[int, int],
    policy: Policy,
    power_budget_w: float | None,
) -> SimulationResult:
    if policy == "energy-aware" and (power_budget_w is None or power_budget_w <= 0):
        raise ValueError("energy-aware policy requires power_budget_w > 0")

    caches = {site: _SiteCache(cap) for site, cap in cache_capacity.items()}
    result = SimulationResult(policy=policy)

    sim_time_ns = 0.0
    # Only used by energy-aware; replenishes with sim_time_ns. Starting at
    # exactly 0 is a deliberate cold-start: no power budget rate, however
    # large, can deliver energy that has not yet had any simulated time to
    # accrue, so the very first cold dispatch can still see a (correctly)
    # negligible defer even under an enormous budget -- this is physically
    # honest, not a bug (see scheduler/selftest.py's note on this).
    energy_budget_pj = 0.0

    for row in trace.itertuples(index=False):
        site_idx = int(row.site_idx)
        expert_id = int(row.expert_id)
        cache = caches[site_idx]

        if cache.hit(expert_id):
            cost = cost_model.cost(site_idx, expert_id, "hbm")
            sim_time_ns += cost.latency_ns
            result.decisions.append(Decision(
                token_uid=int(row.token_uid), site_idx=site_idx, expert_id=expert_id,
                hit=True, deferred=False, energy_pj=cost.total_pj, wait_ns=0.0,
                sim_time_ns=sim_time_ns,
            ))
            continue

        # Miss: this expert must be fetched over the cxl link.
        cost = cost_model.cost(site_idx, expert_id, "cxl")
        wait_ns = 0.0
        deferred = False

        if policy == "energy-aware":
            # Budget replenishes continuously; NaN if link energy is
            # unsourced propagates through this comparison as False, which is
            # the safe direction -- it defers rather than silently spending an
            # un-costed fetch.
            if not (energy_budget_pj >= cost.total_pj):
                shortfall_pj = cost.total_pj - energy_budget_pj
                wait_ns = shortfall_pj / (power_budget_w * _WATT_NS_TO_PJ)
                if math.isnan(wait_ns) or wait_ns < 0:
                    wait_ns = 0.0
                sim_time_ns += wait_ns
                energy_budget_pj += wait_ns * power_budget_w * _WATT_NS_TO_PJ
                deferred = True
            energy_budget_pj -= cost.total_pj

        sim_time_ns += cost.latency_ns
        cache.insert(expert_id)
        result.decisions.append(Decision(
            token_uid=int(row.token_uid), site_idx=site_idx, expert_id=expert_id,
            hit=False, deferred=deferred, energy_pj=cost.total_pj, wait_ns=wait_ns,
            sim_time_ns=sim_time_ns,
        ))

        if policy == "energy-aware":
            # Budget keeps accruing at power_budget_w between dispatches too,
            # not just while waiting -- credit the fetch's own latency span.
            energy_budget_pj += cost.latency_ns * power_budget_w * _WATT_NS_TO_PJ

    return result


def run_naive(trace: pd.DataFrame, cost_model: CostModel, cache_capacity: dict[int, int]) -> SimulationResult:
    """Fetch every cold expert immediately. No energy weighting -- this is
    docs.md 4.5's baseline, "what prior capacity/latency-only systems
    effectively do."
    """
    return _run(trace, cost_model, cache_capacity, "naive", power_budget_w=None)


def run_energy_aware(
    trace: pd.DataFrame, cost_model: CostModel, cache_capacity: dict[int, int], power_budget_w: float
) -> SimulationResult:
    """Same LRU cache as :func:`run_naive`, but gates a cold fetch against a
    continuously-replenishing power budget: if the fetch's E_total exceeds
    what's currently available, it is deferred until enough has accrued.
    """
    return _run(trace, cost_model, cache_capacity, "energy-aware", power_budget_w=power_budget_w)


def diff_decisions(naive: SimulationResult, energy_aware: SimulationResult, max_examples: int = 10) -> str:
    """Where the two policies' outcomes diverge for the same trace.

    The two runs process the identical dispatch sequence, so decisions line up
    index-for-index. A LRU cache is order-sensitive: once the energy-aware
    policy defers one fetch, subsequent arrivals can shift what's resident by
    the time that fetch happens, so hit/miss outcomes can genuinely diverge
    downstream of a single deferral -- that divergence is the real point of
    this comparison, not a bug.
    """
    if len(naive.decisions) != len(energy_aware.decisions):
        return (f"cannot diff: naive has {len(naive.decisions)} decisions, "
                f"energy-aware has {len(energy_aware.decisions)} -- not the same trace")

    lines = ["DECISION DIFF -- naive vs energy-aware, same trace", "=" * 60]
    diverged = 0
    examples: list[str] = []
    for n, e in zip(naive.decisions, energy_aware.decisions):
        if n.hit != e.hit or e.deferred:
            diverged += 1
            if len(examples) < max_examples:
                examples.append(
                    f"  token {n.token_uid} site {n.site_idx} expert {n.expert_id}: "
                    f"naive={'hit' if n.hit else 'miss'} "
                    f"energy-aware={'hit' if e.hit else 'miss'}"
                    f"{' (deferred ' + f'{e.wait_ns:.1f}ns)' if e.deferred else ''}"
                )

    lines.append(f"{diverged} / {len(naive.decisions)} dispatches diverged "
                 f"(different hit/miss outcome, or an energy-aware deferral)")
    lines.extend(examples)
    if diverged > max_examples:
        lines.append(f"  ... and {diverged - max_examples} more")

    lines.append("")
    lines.append(f"{'metric':<20}{'naive':>18}{'energy-aware':>18}")
    for key in ("total_energy_pj", "total_latency_ns", "hit_rate", "n_deferred"):
        nv, ev = getattr(naive, key), getattr(energy_aware, key)
        lines.append(f"{key:<20}{nv:>18.4g}{ev:>18.4g}")
    return "\n".join(lines)
