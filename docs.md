# Energy-Aware Dynamic Expert Tiering for MoE Models — Technical Docs

## 1. Problem

Mixture-of-Experts (MoE) models replace a single dense feedforward block with
many smaller expert networks, and a gating function activates only a few
experts per token. This keeps compute per token low, but every expert still
has its own weights, so total parameter count is large. All expert weights
normally need to sit in GPU High Bandwidth Memory (HBM) for fast access, but
HBM capacity is limited, so once expert count grows, the full model no longer
fits on a single GPU.

The standard fix is offloading rarely-used ("cold") experts to memory outside
the GPU, increasingly via Compute Express Link (CXL), which allows memory
expansion and pooling with lower overhead than plain PCIe offload. But every
cold-expert fetch moves data across the interconnect, costing latency and
energy. Existing offloading systems (see Related Work) place experts based on
activation frequency, cache hit rate, or latency — not on an explicit energy
budget. This project treats energy (GPU compute + memory read + link
transfer, in pJ/bit) as the primary scheduling variable.

## 2. Related work (so we don't re-derive or duplicate it)

- **PIMoE** — introduced the hot/cold expert split concept.
- **Diff-MoE** — priority-driven differential expert cache hierarchy
  (globally hot / locally hot / cold), optimized for cache hit rate and
  throughput, not energy.
- **HybriMoE** — CPU-GPU scheduling and cache management for MoE inference.
- **"Context-Aware MoE Inference on CXL-Enabled GPU-NDP Systems"** (Dec 2025)
  — closest prior work. Uses prefill-stage activation stats to pin hot
  experts in HBM and map cold experts to CXL-attached *near-data processing*
  (compute happens at the memory device, only activations move). Placement
  driven by activation frequency + per-expert quantization for NDP
  throughput, not an explicit energy budget.

**This project's scope difference:** CXL as memory expansion only (data
moves, no near-data compute), which is realistic to simulate with gem5 +
DRAMSim3 + QEMU without custom NDP hardware. **This project's novelty:** an
explicit joint energy-budget scheduler (GPU execution + memory read + link
transfer energy) as the placement decision function, instead of activation
frequency or cache-hit-rate heuristics alone.

## 3. System architecture

```
                 ┌────────────────────┐
   token  ─────▶ │  Gating function     │
                 └─────────┬──────────┘
                           │ expert id
                           ▼
                 ┌────────────────────┐        hot?  ── serve from GPU HBM
   activation ──▶│  Hot/Cold Classifier│───────┐
   log (offline) └────────────────────┘        │ cold?
                                                ▼
                                      ┌────────────────────┐
                                      │ Energy-Aware        │
                                      │ Scheduler            │──▶ fetch now /
                                      └─────────┬──────────┘    defer / batch
                                                │
                          ┌─────────────────────┼─────────────────────┐
                          ▼                     ▼                     ▼
                  GPU compute energy   Memory read energy    Link transfer
                  (per-op estimate)    (DRAMSim3 output)      energy (pJ/bit,
                                                               CXL spec /
                                                               gem5 model)
```

## 4. Components

### 4.1 Profiler (`/profiler`)
Instruments a Hugging Face MoE model's gating function to log, per forward
pass: layer id, expert id(s) selected, token position. Run over WikiText-2
(or another open corpus) to get a realistic activation frequency
distribution. Produces the activation-frequency histogram used to justify
hot/cold classification (same style of evidence used in prior CXL-MoE work,
so results are directly comparable).

**Models:** start with a small custom or open MoE (fast iteration), move to
Mixtral-8x7B at full precision once the pipeline is validated — feasible
given the available 180GB combined VRAM.

### 4.2 Hot/cold classifier (`/profiler`)
Takes the activation log, applies a configurable frequency threshold (e.g.
top 20% by access count = hot), outputs a per-layer hot/cold assignment.

### 4.3 Memory system model (`/memsim`)
Two gem5 + DRAMSim3 configurations:
- **HBM-only** — baseline, everything resident in GPU HBM (or fails once
  it doesn't fit, which is the motivating case).
- **HBM + CXL** — hot experts in HBM, cold experts in a simulated CXL tier
  with added round-trip latency and per-access energy.

Output: latency (ns) and energy (pJ or nJ) per memory access, per tier, fed
into the scheduler's energy calculation.

### 4.4 CXL emulation (`/memsim`, optional strengthening layer)
QEMU with a virtual CXL Type 3 memory device attached to a Linux guest VM.
Purely software-emulated — does not require or use any physical CXL hardware
or the host's CXL-capable root ports. Used to validate that a real OS-level
CXL memory region behaves as the memory model assumes (DAX device
allocation, access patterns), not as the primary source of energy numbers.

### 4.5 Energy-aware scheduler (`/scheduler`)
For each token needing a cold expert, computes:

```
E_total = E_gpu_compute + E_mem_read + E_link_transfer
```

- `E_gpu_compute` — estimated from published per-FLOP/per-op energy figures
  for the target GPU (Blackwell RTX 6000 datasheet), scaled by the expert's
  FLOP count for that forward pass.
- `E_mem_read` — from the memory system model (4.3), tier-dependent.
- `E_link_transfer` — bits transferred (expert weight size) × pJ/bit for the
  CXL link, sourced from CXL spec/characterization literature, or gem5's
  interconnect model if available.

Scheduler compares `E_total` against a live power budget and either serves
from the hot cache, fetches the cold expert immediately, or defers/batches
the fetch with other pending cold-expert requests. A naive baseline
(fetch cold experts immediately, no energy weighting) is kept alongside it
as the comparison point — this baseline represents what prior capacity/
latency-only systems effectively do.

### 4.6 Experiment harness + dashboard (`/experiments`, `/dashboard`)
Runs the same workload trace through three configs — HBM-only, HBM+CXL
naive, HBM+CXL energy-aware — logging throughput (tokens/sec), average
latency, and total energy per run. Streamlit/Plotly Dash app visualizes the
three-way comparison plus the activation heatmap from 4.1.

## 5. Deliverables mapping

| Deliverable (from original scope)              | Produced by |
|--------------------------------------------------|-------------|
| HBM + CXL memory architecture model               | 4.3 |
| CXL-based Transformer/MoE workload analysis        | 4.1 |
| Bandwidth, latency & scalability evaluation        | 4.3, 4.6 |
| Memory tiering & expert placement strategy         | 4.2, 4.5 |
| CXL memory expansion & pooling study               | 4.3, 4.4 |
| Simulation framework (HBM-only vs HBM+CXL)         | 4.3, 4.6 |
| Interactive demo/dashboard                          | 4.6 |
| Final report with architecture recommendations     | synthesis of all above |

## 6. Validation checkpoints (what "working" means at each stage)

1. Profiler: activation histogram is non-uniform (a few experts dominate),
   matching the skew reported in prior MoE activation studies — a flat
   distribution means the hooking is wrong, not that the model is unusual.
2. Memory sim: HBM-only latency/energy numbers should be lower than
   HBM+CXL for the same access pattern; if CXL comes out faster, the model
   config is wrong.
3. Scheduler: on a fixed trace, the energy-aware scheduler's total energy
   must be ≤ the naive baseline's, or the placement logic isn't working as
   intended — this is the core result the whole project rests on.
4. End-to-end: three-way comparison table + dashboard renders from a single
   command, off real run outputs, not hardcoded numbers.

## 7. Explicit non-goals

- No real CXL hardware, no near-data processing / in-memory compute at the
  CXL tier (that's the harder NDP variant from prior work, out of scope).
- No claim that GPU compute or memory energy figures are physically
  measured unless a specific experiment says so and describes how.
- No training of new MoE models — inference-time profiling and scheduling
  only.