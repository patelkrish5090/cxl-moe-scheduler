# CLAUDE.md

Guidance for Claude Code (or any agent) working in this repository. Read this
and `docs.md` before making changes.

## Project in one line

Simulate energy-aware placement of MoE expert weights across GPU HBM and
CXL-attached (emulated) host memory, and show that an energy-budget-aware
scheduler beats capacity/latency-only offloading on total energy and
throughput.

## Ground truth constraints

- No physical CXL hardware exists. All CXL behavior is either emulated
  (QEMU virtual CXL Type 3 device) or modeled (gem5 + DRAMSim3 timing/energy
  numbers, or published CXL spec figures). Never write code or docs that
  imply a real CXL card is present.
- Hardware available: 2x Blackwell RTX 6000 (180GB VRAM combined), 200GB
  system RAM, Xeon Gold 6530. Model choices and batch sizes should be sized
  against this, not against a laptop or free-tier notebook.
- The GPU-execution and memory-read energy numbers are estimates (from
  datasheets, DRAMSim3 output, or published characterization papers), not
  hardware measurements, unless a stage explicitly says otherwise. Never
  present an estimated number as measured in docs, plots, or comments.

## Repo layout (create if missing)

```
/profiler        stage 1: HF model hooks, activation logging, hot/cold split
/memsim           stage 2: gem5 + DRAMSim3 configs and result parsers
/scheduler        stage 3: energy-aware placement/scheduling logic + baseline
/experiments      stage 4: run harness, comparison configs, result CSVs
/dashboard        stage 4: Streamlit app
/data             logs, traces, activation CSVs (gitignored if large)
docs.md
CLAUDE.md
```

## Working rules

- Build and validate one stage at a time, in the order in PROMPT. Do
  not start stage N+1 until stage N produces a working, inspectable output
  (a log file, a plot, a number) and the user has confirmed it looks right.
- Every stage needs a way to sanity-check output without trusting the code
  blindly: a plotted histogram for the profiler, a printed comparison table
  for the memory sim, a diff between baseline and energy-aware scheduler
  decisions for the same trace.
- Keep the naive baseline scheduler (fetch cold experts immediately, no
  energy term) implemented and runnable at all times — it's the comparison
  point for the whole project and must not silently rot.
- Default to small/fast configs first (small model, short trace, short sim
  run) and only scale up to full Mixtral-8x7B / full-length traces once the
  pipeline is proven correct on the small case.
- Python: type hints on public functions, docstrings explaining units
  (joules vs picojoules vs watts — get this right, it's a recurring source
  of silent bugs in energy code). Prefer explicit unit suffixes in variable
  names (`energy_pj`, `latency_ns`) over bare names.
- Any energy or latency constant pulled from a datasheet or paper must be
  cited inline as a comment with a source, not a bare magic number.
- Ask before downloading large model weights or datasets, and before
  installing packages outside pip/conda's normal indexes.
- Do not fabricate benchmark numbers, energy figures, or citations anywhere
  in code, comments, or docs. If a number isn't derived from a real source
  or a real run, mark it clearly as a placeholder (e.g. `TODO_PLACEHOLDER`)
  rather than inventing a plausible-looking value.

## Definition of done for the whole project

A single experiment run produces, for HBM-only / HBM+CXL-naive /
HBM+CXL-energy-aware: throughput, average latency, and total energy, as a
comparison table and a dashboard chart, driven off a real (or clearly
labeled simulated) activation trace and a real gem5/DRAMSim3-derived memory
timing/energy model.