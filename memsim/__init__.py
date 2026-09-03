"""Stage 2: memory system model (docs.md 4.3).

Characterises two memory tiers with gem5 + DRAMSim3 and hands stage 3 a per-tier
latency / bandwidth / energy model:

    HBM   direct-attach memory, the GPU baseline.
    CXL   the same class of device behind a modelled link delay.

No physical CXL hardware is involved, and none is implied (CLAUDE.md). The link
latency is a configured parameter and the link *energy* is not modelled by gem5
at all -- it is a separate, explicitly-sourced constant. See
:mod:`memsim.constants` for what is measured, what is cited, and what is still
outstanding.
"""

__all__ = ["constants", "parse_stats", "run_sweep", "compare"]
