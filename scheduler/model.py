"""Loads stage 1's trace + hot/cold table and stage 2's tier model, and turns
them into a per-(site, expert) fetch cost: E_total = E_gpu_compute + E_mem_read
+ E_link_transfer (docs.md 4.5), plus the latency that fetch would take.

Nothing in this module invents a number. Every energy/latency figure traces
back to either a real gem5/DRAMSim3 run (memsim/tier_model.json) or a cited
constant (scheduler/constants.py, memsim/constants.py). If a needed constant
is still a placeholder, the result is NaN -- it must not be mistaken for zero.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import GPU_COMPUTE_ENERGY_PJ_PER_FLOP

#: 2 FLOPs per multiply-add, per parameter, for one token's forward pass
#: through one expert -- the standard convention used throughout the ML
#: systems literature for counting transformer FLOPs (see e.g. Kaplan et al.
#: 2020, "Scaling Laws for Neural Language Models", where forward-pass FLOPs
#: per token are taken as 2N for N parameters). Not itself a physical
#: constant, so it lives here rather than in constants.py.
FLOPS_PER_PARAM_PER_TOKEN = 2

#: Bytes per parameter for the dtype the profiler ran in. bf16/fp16 both store
#: 2 bytes/parameter; this project's stage-1 runs used bfloat16 throughout
#: (see profiler/README.md).
BYTES_PER_PARAM = 2


@dataclass(frozen=True)
class TierFigures:
    """One tier's latency/bandwidth/energy, read straight from tier_model.json.

    Attributes:
        tier: "hbm" or "cxl".
        latency_ns: Unloaded round-trip latency for ONE SMALL memory access on
            this tier -- what stage 2's gem5 sweep actually characterised.
            This is time-to-first-byte, not time to move a whole expert; see
            CostModel.cost() for why a fetch's latency is not this number
            alone.
        peak_bandwidth_gbps: Read bandwidth at saturation (GB = 1e9 bytes).
            1 GB/s = 1 byte/ns, which is what makes the bytes/bandwidth ->
            nanoseconds conversion in CostModel.cost() unit-free.
        device_energy_pj_per_bit: DRAM device energy, from a real DRAMSim3 run.
        link_energy_pj_per_bit: Link energy; 0 for hbm (direct-attach), a cited
            constant for cxl. NaN if that constant is still unsourced.
        total_energy_pj_per_bit: device + link. NaN whenever link energy is.
    """

    tier: str
    latency_ns: float
    peak_bandwidth_gbps: float
    device_energy_pj_per_bit: float
    link_energy_pj_per_bit: float
    total_energy_pj_per_bit: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TierFigures":
        return cls(
            tier=d["tier"],
            latency_ns=float(d["unloaded_latency_ns"]) if d["unloaded_latency_ns"] is not None else math.nan,
            peak_bandwidth_gbps=float(d["peak_bandwidth_gbps"]) if d["peak_bandwidth_gbps"] is not None else math.nan,
            device_energy_pj_per_bit=float(d["device_energy_pj_per_bit"]) if d["device_energy_pj_per_bit"] is not None else math.nan,
            link_energy_pj_per_bit=float(d["link_energy_pj_per_bit"]),
            total_energy_pj_per_bit=float(d["total_energy_pj_per_bit"]) if d["total_energy_pj_per_bit"] is not None else math.nan,
        )


def load_tier_model(path: str | Path) -> dict[str, TierFigures]:
    """Read memsim/tier_model.json, written by ``python -m memsim.cli compare``.

    Raises:
        FileNotFoundError: with a pointer to the command that produces it.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"no tier model at {path}. Produce it with:\n"
            "  python -m memsim.cli sweep --tiers hbm\n"
            "  python -m memsim.cli sweep --tiers cxl --link-latency-ns <value>\n"
            "  python -m memsim.cli compare"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {t: TierFigures.from_dict(d) for t, d in payload["models"].items()}


@dataclass(frozen=True)
class ExpertCost:
    """The cost of one cold-expert fetch-and-compute, per docs.md 4.5.

    All energy fields are in picojoules (pJ) for ONE token's dispatch to ONE
    expert. ``latency_ns`` is the wall-clock time that fetch occupies the link
    for: the tier's round-trip latency PLUS the bandwidth-limited time to
    actually move the whole expert (weight_bytes / peak_bandwidth_gbps) --
    for a multi-hundred-MB expert the bandwidth term dominates by several
    orders of magnitude, so using the round-trip latency alone (as an earlier
    version of this module did) understates fetch time by ~10^4x. The compute
    itself is assumed to overlap with the next dispatch's issue, so it is not
    added to latency here -- see README's "what this does not model".
    """

    site_idx: int
    expert_id: int
    compute_pj: float
    mem_read_pj: float
    link_transfer_pj: float
    latency_ns: float

    @property
    def total_pj(self) -> float:
        return self.compute_pj + self.mem_read_pj + self.link_transfer_pj


class CostModel:
    """Turns (site, expert) into an :class:`ExpertCost`, for either tier.

    Args:
        expert_weight_bytes: Bytes of one expert's weights, per site_idx (from
            hot_cold.csv's ``expert_weight_bytes`` column -- constant within a
            site, since all experts in a layer share the same shape).
        tiers: hbm/cxl figures from :func:`load_tier_model`.
    """

    def __init__(self, expert_weight_bytes: dict[int, int], tiers: dict[str, TierFigures]):
        self._weight_bytes = expert_weight_bytes
        self._tiers = tiers

    def flops_for(self, site_idx: int) -> float:
        """FLOPs for one token's forward pass through one expert at ``site_idx``.

        FLOPs = 2 * n_params = 2 * (weight_bytes / bytes_per_param). See
        FLOPS_PER_PARAM_PER_TOKEN's docstring for the convention this follows.
        """
        n_params = self._weight_bytes[site_idx] / BYTES_PER_PARAM
        return FLOPS_PER_PARAM_PER_TOKEN * n_params

    def cost(self, site_idx: int, expert_id: int, tier: str) -> ExpertCost:
        """Cost of fetching+computing this expert from ``tier`` ("hbm" or "cxl").

        For "hbm" there is no link transfer (direct-attach): link_transfer_pj
        is exactly 0.0, not merely small. For "cxl", mem_read_pj and
        link_transfer_pj are both driven by the same real DRAMSim3/gem5 run
        via the tier model; either can be NaN if that tier's constants are
        still unsourced, and NaN propagates into total_pj by design.

        latency_ns = round-trip latency + weight_bytes / peak_bandwidth_gbps.
        1 GB/s = 1 byte/ns, so that division is already in nanoseconds with no
        extra conversion factor -- see TierFigures.peak_bandwidth_gbps.
        """
        figures = self._tiers[tier]
        weight_bytes = self._weight_bytes[site_idx]
        weight_bits = weight_bytes * 8

        compute_pj = float(GPU_COMPUTE_ENERGY_PJ_PER_FLOP) * self.flops_for(site_idx)
        mem_read_pj = figures.device_energy_pj_per_bit * weight_bits
        link_transfer_pj = 0.0 if tier == "hbm" else figures.link_energy_pj_per_bit * weight_bits
        transfer_ns = weight_bytes / figures.peak_bandwidth_gbps
        latency_ns = figures.latency_ns + transfer_ns

        return ExpertCost(
            site_idx=site_idx,
            expert_id=expert_id,
            compute_pj=compute_pj,
            mem_read_pj=mem_read_pj,
            link_transfer_pj=link_transfer_pj,
            latency_ns=latency_ns,
        )


def load_expert_weight_bytes(hot_cold_csv: str | Path) -> dict[int, int]:
    """Bytes of one expert's weights per site_idx, from stage 1's hot_cold.csv."""
    table = pd.read_csv(hot_cold_csv)
    per_site = table.groupby("site_idx")["expert_weight_bytes"].first()
    return {int(site): int(nbytes) for site, nbytes in per_site.items()}


def load_hot_experts(hot_cold_csv: str | Path) -> dict[int, set[int]]:
    """Stage 1's hot-expert set per site_idx -- used only to size the stage 3
    cache (see scheduler/README.md), never to hard-pin residency: stage 3's
    cache is genuinely LRU-adaptive, starting empty and warming from the trace,
    exactly like profiler/analyze.py's simulate_lru.
    """
    table = pd.read_csv(hot_cold_csv)
    hot = table[table["is_hot"]]
    out: dict[int, set[int]] = {}
    for site, group in hot.groupby("site_idx"):
        out[int(site)] = set(int(e) for e in group["expert_id"])
    return out


def load_trace(trace_parquet: str | Path) -> pd.DataFrame:
    """The per-(token, layer, expert) dispatch trace, in dispatch order.

    Sorted by (token_uid, slot_k) so that a token's top-k experts for one site
    appear together, matching profiler/README.md's documented trace ordering.
    """
    trace = pd.read_parquet(trace_parquet)
    return trace.sort_values(["token_uid", "site_idx", "slot_k"], kind="stable").reset_index(drop=True)
