"""Validation checkpoint 2: is HBM actually cheaper and faster than CXL?

    python -m memsim.cli compare

docs.md 6 states the check plainly: "HBM-only latency/energy numbers should be
lower than HBM+CXL for the same access pattern; if CXL comes out faster, the
model config is wrong." This module runs that comparison against real gem5
output and says pass or fail, rather than leaving it to the reader's eye.

It also writes ``memsim/tier_model.json``, which is the stage 2 -> stage 3
hand-off: the per-tier latency, bandwidth and energy figures the scheduler
multiplies by real fetch counts from the stage-1 trace.

HONESTY RULES ENFORCED HERE
---------------------------
- A tier whose numbers are missing prints as ``--`` and never as 0.
- Any figure derived from an unsourced constant is NaN and prints as
  ``TODO_PLACEHOLDER``, with a banner naming what is missing.
- The link energy is reported as a separate column from the device energy,
  because gem5 measures one and does not measure the other. Summing them into a
  single number would hide that difference.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from . import constants
from .parse_stats import TierMeasurement, parse_run


def _fmt(value: float | None, spec: str = ".1f", width: int = 10) -> str:
    """Format a number, or a visible marker when it is missing or unsourced."""
    if value is None:
        return f"{'--':>{width}}"
    if isinstance(value, float) and math.isnan(value):
        return f"{'TODO':>{width}}"
    return f"{value:>{width}{spec}}"


def collect(out_root: str | Path) -> list[TierMeasurement]:
    """Parse every gem5 run directory under ``out_root``."""
    out_root = Path(out_root)
    if not out_root.is_dir():
        return []
    runs = []
    for path in sorted(out_root.iterdir()):
        if path.is_dir() and (path / "stats.txt").exists():
            runs.append(parse_run(path))
    return runs


def _period_of(measurement: TierMeasurement) -> int | None:
    """Recover the injection period from the run directory name."""
    name = measurement.outdir.name
    if "_p" not in name:
        return None
    try:
        return int(name.rsplit("_p", 1)[1])
    except ValueError:
        return None


@dataclass
class TierModel:
    """Per-tier figures stage 3 consumes.

    Attributes:
        tier: "hbm" or "cxl".
        unloaded_latency_ns: Read latency at the slowest injection rate, where
            the memory system is idle. This is the stall one fetch imposes when
            nothing else is in flight.
        peak_bandwidth_gbps: Read bandwidth at the fastest injection rate. Sets
            how many concurrent fetches the tier can absorb.
        device_energy_pj_per_bit: DRAM device energy from DRAMSim3. Excludes the
            link.
        link_energy_pj_per_bit: CXL link energy. NaN until sourced; zero for
            HBM, which has no link.
        total_energy_pj_per_bit: Device + link. NaN whenever the link is.
    """

    tier: str
    unloaded_latency_ns: float | None
    peak_bandwidth_gbps: float | None
    device_energy_pj_per_bit: float | None
    link_energy_pj_per_bit: float
    total_energy_pj_per_bit: float | None

    def to_dict(self) -> dict:
        return {
            "tier": self.tier,
            "unloaded_latency_ns": self.unloaded_latency_ns,
            "peak_bandwidth_gbps": self.peak_bandwidth_gbps,
            "device_energy_pj_per_bit": self.device_energy_pj_per_bit,
            "link_energy_pj_per_bit": self.link_energy_pj_per_bit,
            "total_energy_pj_per_bit": self.total_energy_pj_per_bit,
            "units": {
                "latency": "ns",
                "bandwidth": "GB/s (GB = 1e9 bytes)",
                "energy": "pJ/bit",
            },
        }


def build_tier_model(runs: list[TierMeasurement], tier: str) -> TierModel | None:
    """Reduce a tier's sweep points to the figures stage 3 needs."""
    points = [(p, r) for r in runs if r.tier == tier and (p := _period_of(r)) is not None]
    if not points:
        return None

    # Slowest injection = largest period = idle memory system = unloaded latency.
    slowest = max(points, key=lambda pr: pr[0])[1]
    # Fastest injection = smallest period = saturated = peak bandwidth.
    fastest = min(points, key=lambda pr: pr[0])[1]

    # Energy per bit should not depend on injection rate, but background and
    # refresh energy do accrue with simulated time, so a nearly-idle run charges
    # more of them per bit moved. The saturated run is the honest figure for a
    # bulk expert fetch, which is what this project actually models.
    device_energy = fastest.device_energy_pj_per_bit

    link_energy = 0.0 if tier == "hbm" else float(constants.CXL_LINK_ENERGY_PJ_PER_BIT)
    total = None
    if device_energy is not None:
        total = device_energy + link_energy  # NaN propagates if the link is unsourced

    return TierModel(
        tier=tier,
        unloaded_latency_ns=slowest.avg_read_latency_ns,
        peak_bandwidth_gbps=fastest.bandwidth_gbps,
        device_energy_pj_per_bit=device_energy,
        link_energy_pj_per_bit=link_energy,
        total_energy_pj_per_bit=total,
    )


def report(out_root: str | Path, write_model: bool = True) -> dict:
    """Print the checkpoint-2 comparison and write the stage-3 hand-off."""
    out_root = Path(out_root)
    runs = collect(out_root)

    print("=" * 78)
    print("MEMORY TIER COMPARISON  --  docs.md 6, validation checkpoint 2")
    print("=" * 78)

    if not runs:
        print(f"\nNo completed gem5 runs under {out_root}.")
        print("Run:  python -m memsim.cli sweep --link-latency-ns <ns>")
        return {"runs": [], "models": {}, "checkpoint2": None}

    print(f"\n[1] SWEEP POINTS  ({len(runs)} runs under {out_root})")
    print(f"    {'tier':<5} {'period_ps':>10} {'latency_ns':>11} {'BW_GB/s':>10} "
          f"{'dev_pJ/bit':>11} {'bytes_read':>14}")
    print("    " + "-" * 66)
    for measurement in sorted(runs, key=lambda r: (r.tier, _period_of(r) or 0)):
        period = _period_of(measurement)
        period_text = str(period) if period is not None else "--"
        bytes_text = (f"{int(measurement.bytes_read):,}"
                      if measurement.bytes_read is not None else "--")
        print(f"    {measurement.tier:<5} {period_text:>10} "
              f"{_fmt(measurement.avg_read_latency_ns, '.2f', 11)}"
              f"{_fmt(measurement.bandwidth_gbps, '.2f', 10)}"
              f"{_fmt(measurement.device_energy_pj_per_bit, '.4f', 11)}"
              f"{bytes_text:>14}")

    incomplete = [r for r in runs if not r.is_complete]
    if incomplete:
        print(f"\n    {len(incomplete)} run(s) are missing numbers:")
        for measurement in incomplete:
            print(f"      {measurement.outdir.name}:")
            for warning in measurement.warnings:
                print(f"        - {warning}")
        print("      Dump the raw keys with:")
        print(f"        python -m memsim.cli parse {incomplete[0].outdir} --dump")

    print("\n[2] PER-TIER MODEL  (the stage 2 -> stage 3 hand-off)")
    models: dict[str, TierModel] = {}
    for tier in ("hbm", "cxl"):
        model = build_tier_model(runs, tier)
        if model:
            models[tier] = model
    if not models:
        print("    no tier could be reduced; see the warnings above")
        return {"runs": runs, "models": {}, "checkpoint2": None}

    print(f"    {'tier':<5} {'unloaded_lat_ns':>16} {'peak_BW_GB/s':>13} "
          f"{'device_pJ/bit':>14} {'link_pJ/bit':>12} {'total_pJ/bit':>13}")
    print("    " + "-" * 76)
    for tier, model in models.items():
        print(f"    {tier:<5} {_fmt(model.unloaded_latency_ns, '.2f', 16)}"
              f"{_fmt(model.peak_bandwidth_gbps, '.2f', 13)}"
              f"{_fmt(model.device_energy_pj_per_bit, '.4f', 14)}"
              f"{_fmt(model.link_energy_pj_per_bit, '.4f', 12)}"
              f"{_fmt(model.total_energy_pj_per_bit, '.4f', 13)}")

    print("\n[3] CHECKPOINT 2: HBM must be faster AND cheaper than CXL")
    verdict: bool | None = None
    if "hbm" in models and "cxl" in models:
        hbm, cxl = models["hbm"], models["cxl"]
        checks: list[tuple[str, bool | None, str]] = []

        if hbm.unloaded_latency_ns is not None and cxl.unloaded_latency_ns is not None:
            passed = hbm.unloaded_latency_ns < cxl.unloaded_latency_ns
            checks.append((
                "HBM unloaded latency < CXL", passed,
                f"{hbm.unloaded_latency_ns:.2f} ns vs {cxl.unloaded_latency_ns:.2f} ns",
            ))
        else:
            checks.append(("HBM unloaded latency < CXL", None, "latency missing for a tier"))

        if hbm.peak_bandwidth_gbps is not None and cxl.peak_bandwidth_gbps is not None:
            passed = hbm.peak_bandwidth_gbps > cxl.peak_bandwidth_gbps
            checks.append((
                "HBM peak bandwidth > CXL", passed,
                f"{hbm.peak_bandwidth_gbps:.2f} vs {cxl.peak_bandwidth_gbps:.2f} GB/s",
            ))
        else:
            checks.append(("HBM peak bandwidth > CXL", None, "bandwidth missing for a tier"))

        h_total, c_total = hbm.total_energy_pj_per_bit, cxl.total_energy_pj_per_bit
        if h_total is not None and c_total is not None and not (
            math.isnan(h_total) or math.isnan(c_total)
        ):
            passed = h_total <= c_total
            checks.append((
                "HBM total energy/bit <= CXL", passed,
                f"{h_total:.4f} vs {c_total:.4f} pJ/bit",
            ))
        else:
            checks.append((
                "HBM total energy/bit <= CXL", None,
                "CXL link energy is TODO_PLACEHOLDER, so no total exists yet",
            ))

        for label, passed, detail in checks:
            mark = "n/a " if passed is None else ("PASS" if passed else "FAIL")
            print(f"    [{mark}] {label:<32} {detail}")
        decided = [c[1] for c in checks if c[1] is not None]
        verdict = all(decided) if decided else None

        if verdict is False:
            print("\n    -> Checkpoint 2 FAILED. Per docs.md 6 that means the model")
            print("       configuration is wrong, not that CXL is genuinely faster.")
            print("       Check the link delay is actually applied (memsim/gem5_configs")
            print("       /tier.py) and that the two tiers use different device configs.")
        elif verdict is None:
            print("\n    -> Checkpoint 2 UNDECIDED: not enough sourced numbers yet.")
        else:
            print("\n    -> Checkpoint 2 PASSED for the checks that could be decided.")
    else:
        print(f"    only {sorted(models)} present; need both tiers")

    unsourced = constants.unsourced()
    if unsourced:
        print("\n" + "!" * 78)
        print("PRELIMINARY: these results depend on constants that are not yet sourced.")
        print("Any column showing TODO is NaN by construction, not zero.")
        for constant in unsourced:
            print(f"  {constant.name} ({constant.unit})")
            print(f"    {constant.source}")
        print("!" * 78)

    payload = {
        "models": {t: m.to_dict() for t, m in models.items()},
        "checkpoint2_passed": verdict,
        "unsourced_constants": [c.name for c in unsourced],
        "n_runs": len(runs),
    }
    if write_model:
        model_path = Path(out_root).parent / "tier_model.json"
        model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {model_path}")

    return {"runs": runs, "models": models, "checkpoint2": verdict, "payload": payload}
