"""Physical constants for the energy-aware scheduler (stage 3), each tagged
with its provenance. Mirrors memsim/constants.py's no-bare-magic-numbers rule
(CLAUDE.md) -- see that module's docstring for the full rationale: an
unsourced constant is PLACEHOLDER (NaN), which poisons any total computed from
it rather than silently defaulting to zero.

This file exists ahead of the rest of stage 3's code because the citation work
is independent of the scheduler logic itself -- capturing it now, while
sourced, avoids re-deriving it later.
"""

from __future__ import annotations

from memsim.constants import Constant, PLACEHOLDER  # noqa: F401  (PLACEHOLDER re-exported for future constants here)

# ---------------------------------------------------------------------------
# GPU compute energy
# ---------------------------------------------------------------------------
# docs.md 4.5: E_gpu_compute is "estimated from published per-FLOP/per-op
# energy figures for the target GPU (Blackwell RTX 6000 datasheet), scaled by
# the expert's FLOP count for that forward pass."

GPU_COMPUTE_ENERGY_PJ_PER_FLOP = Constant(
    name="GPU_COMPUTE_ENERGY_PJ_PER_FLOP",
    value=1.191,
    unit="pJ/FLOP",
    source="Derived as max power / peak throughput for the NVIDIA RTX PRO "
           "6000 Blackwell Workstation Edition: 600 W max power consumption "
           "divided by 503.8 TFLOPS dense BF16/FP16 (not the 1007.6 TFLOPS "
           "2:4-structured-sparse figure, since MoE expert GEMMs are dense), "
           "both from NVIDIA's official product page "
           "(nvidia.com/en-us/products/workstations/professional-desktop-"
           "gpus/rtx-pro-6000). This is a theoretical best-case figure -- it "
           "assumes the GPU sustains peak dense-BF16 throughput at exactly "
           "its rated max power the entire time, which real workloads never "
           "do. An expert fetch/forward-pass is comparatively low arithmetic "
           "intensity (memory-bound, not compute-saturating), so real "
           "per-FLOP energy is almost certainly higher than this. Treat as a "
           "lower bound, not a measured figure (CLAUDE.md: never present an "
           "estimate as measured) -- revisit with a profiled measurement "
           "(e.g. nvidia-smi power draw sampled during a real expert forward "
           "pass) before using this for anything beyond an "
           "order-of-magnitude sanity check.",
    status="cited",
)

#: Every constant this module declares, for the provenance report.
ALL_CONSTANTS = (GPU_COMPUTE_ENERGY_PJ_PER_FLOP,)
