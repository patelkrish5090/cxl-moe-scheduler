"""Physical constants for the memory model, each tagged with its provenance.

CLAUDE.md forbids two things this module exists to make impossible:

  1. A bare magic number with no source.
  2. An invented number that looks plausible enough to survive into a plot.

Every constant here is a :class:`Constant` carrying its unit and its source. A
constant we have not yet sourced is declared with ``PLACEHOLDER`` as its value,
which is ``float('nan')``. That choice is deliberate: NaN propagates through
arithmetic, so any total computed from an unsourced constant comes out NaN and
cannot be mistaken for a result, printed in a table, or plotted. There is no way
to accidentally ship a fabricated figure.

Use :func:`unsourced` to list what still needs a citation, and
:func:`require_sourced` to fail loudly before presenting numbers as final.

UNITS
-----
Energy is picojoules (``_pj``), time is nanoseconds (``_ns``), bandwidth is
gigabytes per second (``_gbps``), sizes are bytes (``_bytes``). Suffixes are
mandatory on every name -- mixing pJ and nJ is the classic silent bug in energy
code and the suffix is what makes a mismatch visible at the call site.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

#: Value for a constant we have not yet sourced. NaN by design -- see module
#: docstring. Grep for TODO_PLACEHOLDER to find everything still outstanding.
PLACEHOLDER = float("nan")

Status = Literal["cited", "placeholder", "simulated", "derived"]


@dataclass(frozen=True)
class Constant:
    """One physical constant and where it came from.

    Attributes:
        name: Identifier used in reports and provenance lists.
        value: The number, in ``unit``. ``PLACEHOLDER`` (NaN) if unsourced.
        unit: Explicit unit string, e.g. "pJ/bit", "ns", "GB/s".
        source: Citation, or -- for a placeholder -- what to go and find.
        status: ``cited`` (from a named datasheet or paper), ``simulated``
            (produced by a real gem5/DRAMSim3 run, not typed in by hand),
            ``derived`` (computed from other constants here), or
            ``placeholder`` (not yet sourced; value is NaN).
    """

    name: str
    value: float
    unit: str
    source: str
    status: Status

    @property
    def is_sourced(self) -> bool:
        """False if this constant still needs a citation before publication."""
        return self.status != "placeholder" and not math.isnan(self.value)

    def __float__(self) -> float:
        return float(self.value)

    def describe(self) -> str:
        # Three states, not two: a placeholder needs a citation, whereas a
        # 'simulated' constant with no value yet just needs a run. Marking both
        # the same way would make stage 2 look further from done than it is.
        if self.status == "placeholder":
            marker, shown = "!!", "TODO_PLACEHOLDER"
        elif math.isnan(self.value):
            marker, shown = "..", "(pending a run)"
        else:
            marker, shown = "  ", f"{self.value:g}"
        return f"{marker} {self.name:<32} {shown:>18} {self.unit:<10} [{self.status}] {self.source}"


# ---------------------------------------------------------------------------
# CXL link
# ---------------------------------------------------------------------------
# No physical CXL hardware exists for this project (CLAUDE.md), so the link is
# modelled, never measured. gem5 models the *interconnect delay* we configure,
# but it does not model a CXL PHY's energy: that has to come from published
# characterisation. Both numbers below are outstanding.
#
# WHAT TO LOOK FOR, so these can be filled in with a real citation:
#
#   CXL_LINK_LATENCY_NS
#     The added round-trip latency of a read served from a CXL Type 3 memory
#     device versus one served from local DDR on the same host. Reported in
#     measurement studies of real CXL-attached memory, and in device vendors'
#     product briefs. Record the CXL revision (2.0 vs 3.0), the PHY generation
#     (PCIe 5.0 vs 6.0), and whether the figure is idle or loaded latency --
#     loaded latency under bandwidth pressure is much higher and is the honest
#     number for a memory-bound workload.
#
#   CXL_LINK_ENERGY_PJ_PER_BIT
#     SerDes plus protocol energy per bit moved across the link, excluding the
#     DRAM device itself (DRAMSim3 accounts for that separately -- do not
#     double-count it here). Usually quoted for the PCIe PHY generation the CXL
#     link runs on. Record whether the figure is per-direction or aggregate.
#
# Until both are filled in, every CXL energy total this project produces is NaN
# and every report prints an UNSOURCED banner. That is the intended behaviour.

CXL_LINK_LATENCY_NS = Constant(
    name="CXL_LINK_LATENCY_NS",
    value=PLACEHOLDER,  # TODO_PLACEHOLDER
    unit="ns",
    source="TODO_PLACEHOLDER: added read round-trip vs local DDR, from a CXL "
           "Type 3 measurement study or vendor product brief. Note CXL rev, "
           "PHY generation, and idle-vs-loaded.",
    status="placeholder",
)

CXL_LINK_ENERGY_PJ_PER_BIT = Constant(
    name="CXL_LINK_ENERGY_PJ_PER_BIT",
    value=PLACEHOLDER,  # TODO_PLACEHOLDER
    unit="pJ/bit",
    source="TODO_PLACEHOLDER: PHY + protocol energy per bit for the CXL link, "
           "excluding the DRAM device. Note per-direction vs aggregate.",
    status="placeholder",
)

# ---------------------------------------------------------------------------
# Memory devices
# ---------------------------------------------------------------------------
# These are NOT typed in. DRAMSim3 computes device energy from the JEDEC-derived
# IDD parameters in its own device .ini files, and gem5 reports the timing. The
# entries here exist so the provenance report can state where the numbers in a
# results table actually came from, and so a reader can tell simulated output
# apart from literature values at a glance.

HBM_DEVICE_ENERGY = Constant(
    name="HBM_DEVICE_ENERGY",
    value=PLACEHOLDER,  # filled from a run; see memsim.parse_stats
    unit="pJ/bit",
    source="DRAMSim3 device model, IDD parameters from the shipped HBM .ini "
           "(JEDEC-derived). Produced per run, not hardcoded.",
    status="simulated",
)

CXL_DEVICE_ENERGY = Constant(
    name="CXL_DEVICE_ENERGY",
    value=PLACEHOLDER,  # filled from a run; see memsim.parse_stats
    unit="pJ/bit",
    source="DRAMSim3 device model for the DRAM behind the CXL link, IDD "
           "parameters from the shipped DDR .ini. Produced per run.",
    status="simulated",
)

#: Every constant this module declares, for the provenance report.
ALL_CONSTANTS: tuple[Constant, ...] = (
    CXL_LINK_LATENCY_NS,
    CXL_LINK_ENERGY_PJ_PER_BIT,
    HBM_DEVICE_ENERGY,
    CXL_DEVICE_ENERGY,
)


def unsourced() -> list[Constant]:
    """Constants that still need a citation before any result can be published.

    ``simulated`` constants are excluded: their value arrives from a real gem5
    or DRAMSim3 run rather than from this file, so a NaN here means "not run
    yet", not "not sourced".
    """
    return [c for c in ALL_CONSTANTS if c.status == "placeholder"]


def require_sourced(context: str = "this result") -> None:
    """Raise if any constant needed for a publishable number is still a placeholder.

    Call this before writing a final results table or plot. Reports that are
    explicitly labelled as preliminary should call :func:`provenance_report`
    and print the banner instead of raising.

    Raises:
        RuntimeError: naming every outstanding constant.
    """
    missing = unsourced()
    if missing:
        names = "\n".join(f"    {c.name} ({c.unit}) -- {c.source}" for c in missing)
        raise RuntimeError(
            f"{context} depends on {len(missing)} unsourced constant(s):\n{names}\n"
            "Fill these in with a real citation in memsim/constants.py, or mark "
            "the output as preliminary."
        )


def provenance_report() -> str:
    """Human-readable table of every constant and its source."""
    lines = [
        "CONSTANT PROVENANCE",
        "  '!!' needs a citation before any result using it can be published.",
        "  '..' comes from a gem5/DRAMSim3 run and is simply not measured yet.",
        "",
    ]
    lines.extend(c.describe() for c in ALL_CONSTANTS)
    missing = unsourced()
    lines.append("")
    if missing:
        lines.append(f"  {len(missing)} constant(s) unsourced -- any total computed "
                     "from them is NaN by design.")
    else:
        lines.append("  All constants sourced.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------
# Trivial, but written once and used everywhere so a factor of 8 or 1000 cannot
# creep in at a call site.

def bits_from_bytes(n_bytes: float) -> float:
    """Bytes -> bits."""
    return n_bytes * 8.0


def energy_pj(pj_per_bit: float, n_bytes: float) -> float:
    """Energy in picojoules to move ``n_bytes`` at ``pj_per_bit``.

    Returns NaN if ``pj_per_bit`` came from an unsourced constant, which is how
    a placeholder makes itself visible downstream.
    """
    return pj_per_bit * bits_from_bytes(n_bytes)


def pj_to_joules(pj: float) -> float:
    """Picojoules -> joules."""
    return pj * 1e-12


def pj_to_millijoules(pj: float) -> float:
    """Picojoules -> millijoules, the readable scale for one expert fetch."""
    return pj * 1e-9
