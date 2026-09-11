# Stage 3 — Energy-aware expert-fetch scheduler

Implements docs.md 4.5. Consumes stage 1's real activation trace and stage
2's real gem5/DRAMSim3 tier model, and simulates two scheduling policies over
the identical dispatch sequence:

- **naive** — fetch every cold expert immediately, no energy weighting. This
  is the comparison point docs.md calls for: "what prior capacity/latency-only
  systems effectively do." Must stay runnable at all times (CLAUDE.md).
- **energy-aware** — the same LRU cache, but gates *when* a cold fetch happens
  against a continuously-replenishing power budget, deferring it under budget
  pressure instead of always fetching immediately.

## Commands

```bash
python -m scheduler.cli selftest                                     # offline, no data needed
python -m scheduler.cli run data/runs/<name> --policy naive
python -m scheduler.cli run data/runs/<name> --policy energy-aware --power-budget-w <W>
python -m scheduler.cli compare data/runs/<name> --power-budget-w <W> # both policies + a decision diff
```

`<name>` is a stage-1 run directory containing `trace.parquet` and
`hot_cold.csv` (e.g. `data/runs/mixtral_8x7b_decode`). All commands default
to reading `memsim/tier_model.json` (`--tier-model` to override), which must
already exist — produce it with `python -m memsim.cli compare` (stage 2).

## Picking `--power-budget-w`

**Do not guess this number.** Run `run --policy naive` first and read off its
`implied_avg_power_w` — the average power naive's own energy would take if
spent evenly over its own wall-clock span. That is the reference point:

- A budget near or above naive's `implied_avg_power_w` reproduces
  near-naive behaviour (little to no deferral) — there was never any pressure
  to relieve.
- A budget well below it is what actually forces trade-offs: the scheduler
  defers fetches, trading latency for staying under a real power ceiling.

Passing an arbitrary number with no relation to this figure risks either of
two unhelpful extremes: a budget picked too high changes nothing, and one
picked too low defers *everything*, since even a fetch that will eventually
be affordable can't be paid for by a rate that hasn't had time to accrue
anything yet (see "Cold start", below). A synthetic smoke test during
development showed exactly this: an arbitrary `30 W` guess against a
352&nbsp;MB-expert trace deferred every single miss, each by several
milliseconds, for a trace whose naive baseline actually implied ~273
**kilowatts** of average power — three orders of magnitude off, purely from
guessing instead of reading the reference number.

## The model

For each cold-expert dispatch, per docs.md 4.5:

```
E_total = E_gpu_compute + E_mem_read + E_link_transfer
```

- `E_gpu_compute` — `GPU_COMPUTE_ENERGY_PJ_PER_FLOP` (scheduler/constants.py,
  cited from the RTX PRO 6000 Blackwell datasheet) × FLOPs for one token's
  forward pass through one expert. FLOPs are estimated as `2 × n_params`
  (the standard convention throughout the ML-systems literature for
  multiply-add-counted forward-pass FLOPs), and `n_params` is derived from
  `hot_cold.csv`'s `expert_weight_bytes` at 2 bytes/parameter (bf16, matching
  what stage 1 profiled). Identical for a hit or a miss — the compute happens
  either way; only the memory path differs.
- `E_mem_read` — `device_energy_pj_per_bit` × bits, from the tier the expert
  is served from (real DRAMSim3 output, stage 2).
- `E_link_transfer` — `link_energy_pj_per_bit` × bits. Exactly `0.0` for hbm
  (direct-attach, no link); the cited CXL constant for cxl, which is `NaN`
  until sourced — see memsim/constants.py. NaN propagates through `total_pj`
  by design; a scheduler run against an unsourced tier will show NaN energy
  totals rather than a silently-wrong number.

## Units

Energy is picojoules (`_pj`) throughout, matching memsim/constants.py, until
the final `implied_avg_power_w` conversion to watts. **`pJ/ns` is
milliwatts, not watts** (`1e-12 J / 1e-9 s = 1e-3 W`) — this project already
caught one real bug from exactly this mixup during development (see
`_WATT_NS_TO_PJ` in simulate.py, and simulate.py's own unit-conversion
comments). Latency and wait times are nanoseconds (`_ns`), matching the tick
convention used throughout memsim.

## Cache sizing

Stage 3's cache is genuinely LRU-adaptive — it starts empty and warms from
the trace, exactly like `profiler/analyze.py`'s `simulate_lru`. Per-layer
capacity defaults to that layer's hot-expert *count* from stage 1's
`hot_cold.csv` (a real, already-computed sizing decision, not an arbitrary
new parameter) — nothing is hard-pinned; a "hot" expert can still be evicted
under sustained pressure from other experts, same as it would in a true LRU
cache.

## What this does not model

Be upfront about this wherever these numbers are quoted:

- **Cold start.** The energy budget starts at exactly 0 pJ at simulated
  time 0. No power rate, however large, can deliver energy that hasn't yet
  had simulated time to accrue, so the very first cold dispatch can still see
  a (correctly) negligible defer even under an enormous budget. This is
  physically honest — you cannot spend energy that has not accrued — not a
  bug; `scheduler/selftest.py` documents and tests for exactly this.
- **No overlap / no concurrency.** Dispatches are processed one at a time on
  a single simulated timeline, in the trace's stored order. There is no
  cross-layer pipelining, no modelling of multiple in-flight fetches, and no
  batching of concurrent cold-expert requests (docs.md mentions "batching" as
  a possible scheduler action; this version implements the "defer" half of
  that but not batching — see below). Real hardware would overlap much of
  this, so throughput/latency figures here are directional (policy A vs
  policy B on the *same* simplified timeline), not an absolute performance
  prediction.
- **Batching not implemented.** docs.md 4.5 frames the scheduler's options as
  "fetch now / defer / batch." This version implements fetch-now and defer.
  Batching was deliberately left out of v1: the current energy model is
  purely linear in bytes (`pJ/bit × bits`, no fixed per-transfer overhead
  term), so batching several cold fetches together would not reduce total
  energy under this model — there is no fixed cost to amortize. It would be
  worth adding once a fixed per-transfer link/protocol overhead is separately
  sourced (distinct from the pJ/bit figure already cited), since that is
  exactly the kind of cost batching is meant to amortize.

## Validation checkpoint

`python -m scheduler.cli compare <run_dir> --power-budget-w <W>` is this
stage's required sanity-check output (CLAUDE.md: "a diff between baseline and
energy-aware scheduler decisions for the same trace"). Both policies replay
the identical dispatch sequence index-for-index, so their `Decision` lists
line up directly; the diff reports where they diverge — a different hit/miss
outcome, or any energy-aware deferral — plus a side-by-side summary table.
Divergence downstream of a single deferral is expected and correct: once the
energy-aware policy delays one fetch, later arrivals can shift what the LRU
cache holds by the time that fetch actually happens, so hit/miss outcomes can
genuinely differ from naive's from that point on. That emergent divergence,
not a hand-designed difference in eviction logic, is what makes the two
policies worth comparing at all.

## Correctness checks

`python -m scheduler.cli selftest` (41 checks, offline, no gem5/real trace
needed) independently recomputes the cost model's energy figures, verifies
the LRU cache's hit/miss/eviction behaviour by hand-traced sequences, checks
that an unsourced tier constant poisons totals to NaN rather than defaulting
to zero, verifies the energy-budget invariant (no dispatch is ever charged
against insufficient budget), and round-trips file loading (trace.parquet,
hot_cold.csv, tier_model.json) through small fixture files.
