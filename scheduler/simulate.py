"""Replays a stage-1 trace through a per-site expert cache under one of three
policies, producing a decision log and aggregate energy/latency/throughput
figures.

THREE POLICIES:

  naive               -- LRU eviction, fetch every cold expert immediately.
                          docs.md 4.5's baseline: "what prior capacity/
                          latency-only systems effectively do."
  energy-aware-defer   -- same LRU eviction as naive, but gates *when* a cold
                          fetch happens against a replenishing power budget,
                          deferring it under pressure. This changes fetch
                          TIMING only, not WHAT gets cached -- so, by
                          construction, it cannot change total energy (see
                          its own docstring). Kept as an intermediate
                          comparison point, not the checkpoint-3 answer.
  energy-aware-evict   -- changes eviction itself: scores cached experts by
                          (access frequency x refetch cost) / (age since last
                          touch) instead of pure recency, so a
                          rarely-used-but-still-recent expert can be evicted
                          in favour of a more valuable one touched slightly
                          longer ago. The age term is load-bearing, not a
                          nicety: a pure-frequency (no decay) version of this
                          was tried first and FAILED on the real Mixtral
                          decode trace (7.1% more energy than naive, worse hit
                          rate) from stale popularity -- see _SiteCache's
                          docstring. This is the policy that can actually
                          reduce total energy relative to naive (docs.md 6,
                          checkpoint 3), because it changes which misses
                          happen at all, not just when.

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

Policy = Literal["naive", "energy-aware-defer", "energy-aware-evict"]
EvictionPolicy = Literal["lru", "energy-aware"]

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
        energy-aware-defer budget reproduces roughly naive's own pace
        (near-zero deferral); a materially smaller budget is what actually
        forces the scheduler to trade latency for staying under a real power
        ceiling. Picking a budget with no relation to this number (as a first
        guess might) risks either changing nothing or deferring almost
        everything -- see scheduler/README.md's worked example.
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


@dataclass
class _Entry:
    refetch_cost_pj: float
    access_count: int
    last_touched: int  # this site's local dispatch-clock reading at last touch


class _SiteCache:
    """One MoE layer's expert cache. Starts empty and warms from the trace --
    capacity is sized from stage 1's hot-expert count, but nothing is pinned;
    with eviction_policy="lru" this is exactly
    profiler/analyze.py::simulate_lru's semantics.

    eviction_policy="energy-aware" scores each candidate by
    (access_count * refetch_cost_pj) / age_since_last_touch and evicts the
    minimum -- a recency-decayed frequency score (the GDSF,
    Greedy-Dual-Size-Frequency, family of cache-replacement algorithms), not
    plain frequency. WHY THE DECAY TERM IS LOAD-BEARING, NOT A NICETY: a first
    version of this scored by (access_count * refetch_cost_pj) alone, with no
    decay. On the real Mixtral decode trace that version FAILED docs.md 6
    checkpoint 3 -- 7.1% MORE total energy than naive, and a WORSE hit rate
    (25.8% vs naive's 31.7%). Diagnosis: an expert popular early in a 262K-
    dispatch trace keeps a permanently high access_count and is never evicted
    again even once the trace moves past needing it, while genuinely-current
    experts get evicted prematurely because their count hasn't caught up yet
    -- the classic "stale popularity" failure mode of pure LFU. The
    hand-traced example in selftest.py didn't catch this because it has no
    temporal drift; a 262K-dispatch real trace does. Dividing by
    time-since-last-touch fixes it: a stale entry's score decays even with a
    high historical count, while a genuinely-current one stays competitive.

    WHY NOT COST ALONE: within one site, every expert has the same
    weight_bytes (hot_cold.csv's expert_weight_bytes is constant per site),
    so refetch_cost_pj is IDENTICAL across every candidate in a single
    eviction decision -- weighting by cost alone would degenerate to
    (constant) * nothing, i.e. no signal, i.e. an arbitrary tie-break. The
    terms that actually vary per expert are access frequency (what stage 1's
    gini/entropy skew measurements are about) and recency; cost is kept in
    the formula because it is the structurally correct term docs.md 4.5 asks
    for, and it WILL differentiate candidates in a model whose experts vary
    in size (this project's traces do not).
    """

    def __init__(self, capacity: int, eviction_policy: EvictionPolicy = "lru"):
        self.capacity = max(capacity, 1)
        self.eviction_policy = eviction_policy
        self._entries: "OrderedDict[int, _Entry]" = OrderedDict()
        self._clock = 0  # this site's own dispatch counter, advances on every hit() call

    def hit(self, expert_id: int) -> bool:
        # The clock advances on every dispatch this site sees, hit or miss --
        # "time passing" makes every OTHER cached entry one step staler
        # regardless of whether this particular dispatch found its expert.
        self._clock += 1
        if expert_id in self._entries:
            entry = self._entries[expert_id]
            entry.access_count += 1
            entry.last_touched = self._clock
            self._entries.move_to_end(expert_id)
            return True
        return False

    def insert(self, expert_id: int, refetch_cost_pj: float) -> int | None:
        """Insert a freshly-fetched expert, evicting per eviction_policy if
        now over capacity. Returns the evicted expert_id, or None.

        Only ever called immediately after a hit() that returned False for
        the same dispatch, so self._clock has already been advanced for this
        dispatch -- last_touched=self._clock is correct without a second tick.
        """
        self._entries[expert_id] = _Entry(
            refetch_cost_pj=refetch_cost_pj, access_count=1, last_touched=self._clock
        )
        self._entries.move_to_end(expert_id)
        if len(self._entries) <= self.capacity:
            return None
        if self.eviction_policy == "lru":
            evicted, _ = self._entries.popitem(last=False)
            return evicted
        return self._evict_by_value()

    def _evict_by_value(self) -> int:
        """Evict the entry with the smallest
        (access_count * refetch_cost_pj) / (age_since_last_touch + 1).

        The +1 avoids division by zero for an entry touched on this exact
        dispatch (age 0) -- see insert()'s note on why the just-inserted entry
        is itself a candidate here: it participates in the same eviction pass
        that added it, which is correct (nothing exempts a fresh insert from
        immediate re-eviction if its score really is the worst).

        NaN-safe: a NaN refetch cost (an unsourced tier constant) makes that
        entry's score NaN, and NaN comparisons are always False in Python, so
        a NaN-scored entry is never selected as the minimum by `<` -- it would
        silently never be evicted, which is the wrong direction for a
        placeholder. Guard explicitly instead of relying on comparison
        semantics here.
        """
        worst_key, worst_score = None, math.inf
        for key, entry in self._entries.items():
            age = self._clock - entry.last_touched + 1
            score = (entry.access_count * entry.refetch_cost_pj) / age
            if math.isnan(score):
                worst_key, worst_score = key, math.nan
                break  # an unsourced-cost entry is the most urgent to flag, evict it first
            if score < worst_score:
                worst_key, worst_score = key, score
        del self._entries[worst_key]
        return worst_key


def _run(
    trace: pd.DataFrame,
    cost_model: CostModel,
    cache_capacity: dict[int, int],
    policy: Policy,
    eviction_policy: EvictionPolicy,
    power_budget_w: float | None,
) -> SimulationResult:
    if policy == "energy-aware-defer" and (power_budget_w is None or power_budget_w <= 0):
        raise ValueError("energy-aware-defer policy requires power_budget_w > 0")

    caches = {site: _SiteCache(cap, eviction_policy) for site, cap in cache_capacity.items()}
    result = SimulationResult(policy=policy)

    sim_time_ns = 0.0
    # Only used by energy-aware-defer; replenishes with sim_time_ns. Starting
    # at exactly 0 is a deliberate cold-start: no power budget rate, however
    # large, can deliver energy that has not yet had any simulated time to
    # accrue, so the very first cold dispatch can still see a (correctly)
    # negligible defer even under an enormous budget -- this is physically
    # honest, not a bug (see scheduler/selftest.py's note on this).
    energy_budget_pj = 0.0

    for row in trace.itertuples(index=False):
        site_idx = int(row.site_idx)
        expert_id = int(row.expert_id)
        cache = caches[site_idx]
        # The cost to refetch THIS expert if it were evicted right now -- used
        # by energy-aware-evict's scoring regardless of hit/miss, so a hit
        # still refreshes the cache's bookkeeping (a no-op numerically today,
        # since cost is deterministic per site/expert, but kept correct for a
        # future dynamic cost model).
        refetch_cost = cost_model.cost(site_idx, expert_id, "cxl").total_pj

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

        if policy == "energy-aware-defer":
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
        cache.insert(expert_id, refetch_cost)
        result.decisions.append(Decision(
            token_uid=int(row.token_uid), site_idx=site_idx, expert_id=expert_id,
            hit=False, deferred=deferred, energy_pj=cost.total_pj, wait_ns=wait_ns,
            sim_time_ns=sim_time_ns,
        ))

        if policy == "energy-aware-defer":
            # Budget keeps accruing at power_budget_w between dispatches too,
            # not just while waiting -- credit the fetch's own latency span.
            energy_budget_pj += cost.latency_ns * power_budget_w * _WATT_NS_TO_PJ

    return result


def run_naive(trace: pd.DataFrame, cost_model: CostModel, cache_capacity: dict[int, int]) -> SimulationResult:
    """LRU eviction, fetch every cold expert immediately. No energy weighting
    at all -- this is docs.md 4.5's baseline, "what prior capacity/
    latency-only systems effectively do."
    """
    return _run(trace, cost_model, cache_capacity, "naive", "lru", power_budget_w=None)


def run_energy_aware_defer(
    trace: pd.DataFrame, cost_model: CostModel, cache_capacity: dict[int, int], power_budget_w: float
) -> SimulationResult:
    """Same LRU cache and eviction as :func:`run_naive`, but gates a cold
    fetch against a continuously-replenishing power budget: if the fetch's
    E_total exceeds what's currently available, it is deferred until enough
    has accrued.

    This changes WHEN a fetch happens, never WHAT gets cached -- eviction
    decisions are identical to naive's, so the same set of experts eventually
    gets fetched at the same per-fetch cost either way. total_energy_pj is
    therefore identical to naive's BY CONSTRUCTION; this policy trades latency
    for staying under an instantaneous power ceiling, it does not reduce total
    energy (docs.md 6 checkpoint 3 needs energy-aware-evict for that, not
    this policy -- see module docstring).
    """
    return _run(trace, cost_model, cache_capacity, "energy-aware-defer", "lru", power_budget_w=power_budget_w)


def run_energy_aware_evict(
    trace: pd.DataFrame, cost_model: CostModel, cache_capacity: dict[int, int]
) -> SimulationResult:
    """Energy-aware EVICTION: fetch every cold expert immediately (no
    deferral -- isolating the eviction mechanism's effect on total energy
    from the (energy-neutral) timing effect run_energy_aware_defer has), but
    evict by (access_count * refetch_cost_pj) instead of pure recency.

    This is the policy that can actually satisfy docs.md 6 checkpoint 3
    (energy-aware total energy <= naive's): by changing which experts get
    evicted, it can change how many misses happen at all, unlike deferral
    alone. Whether it actually does so on a given trace is an empirical
    question -- run it and check total_energy_pj against naive's, do not
    assume it from the architecture (see scheduler/README.md's reported
    numbers).
    """
    return _run(trace, cost_model, cache_capacity, "energy-aware-evict", "energy-aware", power_budget_w=None)


def diff_decisions(a: SimulationResult, b: SimulationResult, max_examples: int = 10) -> str:
    """Where two policies' outcomes diverge for the same trace.

    The two runs process the identical dispatch sequence, so decisions line up
    index-for-index. Either cache policy is order-sensitive: once one run
    defers a fetch or evicts differently, subsequent arrivals can shift what's
    resident by the time a later dispatch happens, so hit/miss outcomes can
    genuinely diverge downstream of a single decision -- that divergence is
    the real point of this comparison, not a bug.
    """
    if len(a.decisions) != len(b.decisions):
        return (f"cannot diff: {a.policy} has {len(a.decisions)} decisions, "
                f"{b.policy} has {len(b.decisions)} -- not the same trace")

    lines = [f"DECISION DIFF -- {a.policy} vs {b.policy}, same trace", "=" * 60]
    diverged = 0
    examples: list[str] = []
    for da, db in zip(a.decisions, b.decisions):
        if da.hit != db.hit or db.deferred:
            diverged += 1
            if len(examples) < max_examples:
                examples.append(
                    f"  token {da.token_uid} site {da.site_idx} expert {da.expert_id}: "
                    f"{a.policy}={'hit' if da.hit else 'miss'} "
                    f"{b.policy}={'hit' if db.hit else 'miss'}"
                    f"{' (deferred ' + f'{db.wait_ns:.1f}ns)' if db.deferred else ''}"
                )

    lines.append(f"{diverged} / {len(a.decisions)} dispatches diverged "
                 f"(different hit/miss outcome, or a deferral)")
    lines.extend(examples)
    if diverged > max_examples:
        lines.append(f"  ... and {diverged - max_examples} more")

    lines.append("")
    lines.append(f"{'metric':<20}{a.policy:>22}{b.policy:>22}")
    for key in ("total_energy_pj", "total_latency_ns", "hit_rate", "n_deferred"):
        av, bv = getattr(a, key), getattr(b, key)
        lines.append(f"{key:<20}{av:>22.6g}{bv:>22.6g}")
    return "\n".join(lines)


def three_way_report(naive: SimulationResult, defer: SimulationResult, evict: SimulationResult) -> str:
    """docs.md 6 checkpoint 3's required report: total energy for all three
    policies on the same trace, with an explicit PASS/FAIL against naive's.
    """
    lines = ["THREE-WAY COMPARISON -- docs.md 6, checkpoint 3", "=" * 70]
    lines.append(f"{'policy':<22}{'total_energy_pj':>20}{'total_latency_ns':>20}{'hit_rate':>12}")
    for r in (naive, defer, evict):
        lines.append(f"{r.policy:<22}{r.total_energy_pj:>20.6g}{r.total_latency_ns:>20.6g}{r.hit_rate:>12.4f}")

    lines.append("")
    if math.isnan(evict.total_energy_pj) or math.isnan(naive.total_energy_pj):
        lines.append("CHECKPOINT 3: UNDECIDED -- a NaN energy total means an unsourced "
                      "constant is involved; see memsim.cli provenance.")
    else:
        passed = evict.total_energy_pj <= naive.total_energy_pj
        delta_pct = 100.0 * (naive.total_energy_pj - evict.total_energy_pj) / naive.total_energy_pj
        lines.append(
            f"CHECKPOINT 3: {'PASSED' if passed else 'FAILED'} -- "
            f"energy-aware-evict total energy is "
            f"{'lower' if passed else 'HIGHER'} than naive's by {abs(delta_pct):.2f}%"
        )
        if not passed:
            lines.append("  The placement logic is not reducing total energy on this trace -- "
                          "see scheduler/README.md's eviction-policy section before trusting "
                          "this scheduler's energy claims.")
    return "\n".join(lines)
