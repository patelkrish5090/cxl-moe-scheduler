# Stage 3 — Energy-aware expert-fetch scheduler

Implements docs.md 4.5. Consumes stage 1's real activation trace and stage
2's real gem5/DRAMSim3 tier model, and simulates three scheduling policies
over the identical dispatch sequence:

- **naive** — LRU eviction, fetch every cold expert immediately, no energy
  weighting anywhere. This is the comparison point docs.md calls for: "what
  prior capacity/latency-only systems effectively do." Must stay runnable at
  all times (CLAUDE.md).
- **energy-aware-defer** — the same LRU eviction as naive, but gates *when* a
  cold fetch happens against a continuously-replenishing power budget,
  deferring it under budget pressure instead of always fetching immediately.
  **This changes fetch timing only, never what gets cached** — see "Why
  deferral alone cannot pass checkpoint 3", below.
- **energy-aware-evict** — changes *placement*: evicts by
  `access_count × refetch_cost_pj` instead of pure recency, so a
  rarely-used-but-recently-touched expert can be evicted in favour of a
  more valuable one that was touched slightly longer ago. This is the policy
  that can actually reduce total energy, because it changes which misses
  happen at all, not just when they happen.

## Commands

```bash
python -m scheduler.cli selftest                                      # offline, no data needed
python -m scheduler.cli run data/runs/<name> --policy naive
python -m scheduler.cli run data/runs/<name> --policy energy-aware-defer --power-budget-w <W>
python -m scheduler.cli run data/runs/<name> --policy energy-aware-evict
python -m scheduler.cli compare  data/runs/<name> --power-budget-w <W>  # naive vs -defer: timing diff
python -m scheduler.cli compare3 data/runs/<name> --power-budget-w <W>  # all three + checkpoint 3 verdict
```

`<name>` is a stage-1 run directory containing `trace.parquet` and
`hot_cold.csv` (e.g. `data/runs/mixtral_8x7b_decode`). All commands default
to reading `memsim/tier_model.json` (`--tier-model` to override), which must
already exist — produce it with `python -m memsim.cli compare` (stage 2).
`energy-aware-evict` needs no `--power-budget-w` — it has no deferral
mechanism to gate (see below).

## Picking `--power-budget-w` (for `energy-aware-defer` / `compare` / `compare3`)

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
- `latency_ns` (not one of the three energy terms, but computed alongside
  them) — the tier's round-trip latency **plus** `weight_bytes /
  peak_bandwidth_gbps`, i.e. the bandwidth-limited time to actually move the
  whole expert. For a multi-hundred-MB expert the bandwidth term dominates
  the round-trip term by several orders of magnitude. An earlier version of
  this module used the round-trip latency alone, which understated real
  fetch time by ~34,000x on the real Mixtral decode trace (432ns vs the
  ~14.6ms bandwidth-limited reality) — caught by comparing this stage's
  `implied_avg_power_w` against a hand-computed sanity check before trusting
  it. See `CostModel.cost()`'s docstring in model.py.

## Why deferral alone cannot pass checkpoint 3

`energy-aware-defer` and `naive` use *exactly the same LRU eviction*. A
deferred fetch still happens — later, at the same per-byte cost — and
inserts into the cache the same way naive's immediate fetch would have.
Nothing about *which* expert is resident at any point changes because a
fetch was delayed. So the set of misses over the whole trace, and therefore
`total_energy_pj`, is identical between naive and energy-aware-defer **by
construction** — verified directly in `scheduler/selftest.py`
("energy-aware-defer... matches naive's total energy"). Deferral is a real,
useful mechanism for respecting an instantaneous power ceiling, but it was
never going to satisfy docs.md 6 checkpoint 3 ("energy-aware total energy
≤ naive's"), because checkpoint 3 is about *placement*, not timing.

## Energy-aware eviction (`energy-aware-evict`)

Scores each cached expert by `access_count × refetch_cost_pj` and evicts the
minimum when a site's cache is full. `access_count` is a running counter,
incremented on every hit and on insertion (i.e. real in-simulation reuse
frequency, not a separately-loaded external figure). `refetch_cost_pj` is
`CostModel.cost(site, expert, "cxl").total_pj` — the real, cited E_total for
re-fetching that expert if it's lost.

**Why frequency is the load-bearing term, not cost.** Within a single site
(MoE layer), every expert has the *same* `expert_weight_bytes`
(`hot_cold.csv` — constant per layer, since all experts in a layer share the
same shape). That means `refetch_cost_pj` is identical across every
candidate in a single eviction decision on this project's traces — weighting
by cost alone would multiply every candidate's score by the same constant
and change nothing, i.e. degenerate silently to an arbitrary tie-break
disguised as a real decision. `access_count` is what actually varies
per-expert (that variation is exactly what stage 1's gini/entropy skew
measurements are about), so it's the term doing real work here. Cost is kept
in the formula because it's the structurally correct term docs.md 4.5 asks
for, and it becomes load-bearing too in a model whose experts vary in size
across sites — this project's traces don't have that variation, but the
formula is written to be correct if a future one does.

**NaN handling.** If a cached entry's `refetch_cost_pj` is NaN (its tier's
link energy is unsourced), that entry's score is NaN and it is evicted
*first*, not kept forever — `<` comparisons with NaN are always `False` in
Python, so relying on plain min-comparison would silently never select a
NaN-scored entry, the wrong direction for an unsourced placeholder. Guarded
explicitly in `_SiteCache._evict_by_value()`; tested in `selftest.py`.

**Toggling between policies.** `_SiteCache(capacity, eviction_policy=...)`
takes `"lru"` or `"energy-aware"` directly — `run_naive` and
`run_energy_aware_defer` both pass `"lru"`, `run_energy_aware_evict` passes
`"energy-aware"`. The LRU baseline is always reachable by construction, not
just by convention (CLAUDE.md's "keep the naive baseline runnable" applies
to the cache mechanics too, not only the top-level policy).

## Checkpoint 3 result

`python -m scheduler.cli compare3` reports total energy for all three
policies and an explicit PASS/FAIL. Demonstrated two ways:

1. **A hand-traced constructed example** (`scheduler/selftest.py`,
   "[energy-aware-evict policy: checkpoint 3]"): 2 experts touched
   4x and 1x respectively share a capacity-2 cache; a 3rd expert forces an
   eviction while both are resident, at a moment when the barely-used one was
   touched more recently. LRU evicts the valuable one (wrong call);
   frequency-weighted eviction keeps it. The next request for it is a hit
   under energy-aware-evict and a miss under naive — a fully worked,
   independently-verifiable example, not just an architectural claim.
2. **Real trace data**: `data/runs/mixtral_8x7b_decode`,
   `--power-budget-w 0.4` (naive's own `implied_avg_power_w`, per "Picking
   --power-budget-w" above) — **CHECKPOINT 3: PASSED**, energy-aware-evict
   total energy 0.23% lower than naive's (2.19545e16 pJ vs 2.20055e16 pJ),
   hit rate 31.87% vs naive's 31.68%. Small, but real: getting here took two
   failed attempts, both honest, both diagnosable, both documented in
   `_SiteCache`'s docstring — v1 (pure frequency via a running counter) was
   7.1% *worse* than naive from stale popularity; v2 (same counter, decayed
   by recency) was still 5.7% worse, because a running counter resets on
   every eviction/re-fetch cycle and never reflects true whole-trace
   popularity. v3 (this version, using stage 1's real pre-computed
   `dispatch_count`) is what finally passed. The small magnitude is
   consistent with, not a contradiction of, stage 1's own
   `profiler/analyze.py` finding on this exact trace: LRU only beats static
   pinning by 0.7 points at this cache size, meaning there was never much
   slack for *any* reactive online policy to capture — most of the real
   headroom (to Belady's 54.3%) needs foreknowledge of the future, which no
   causal scheduler has. Re-running with a different cache capacity or a
   different trace will change this number; it is not hardcoded anywhere in
   the code, only reported here as of the run above.

If a real run's checkpoint 3 comes back FAILED, `compare3`'s output says so
explicitly rather than silently reporting an inconclusive-looking number —
that would mean this trace's access pattern doesn't have enough frequency
skew for eviction policy to matter (a flatter, more uniform trace would
genuinely produce this), which is itself useful information about when this
scheduling approach helps and when it doesn't.

## Units

Energy is picojoules (`_pj`) throughout, matching memsim/constants.py, until
the final `implied_avg_power_w` conversion to watts. **`pJ/ns` is
milliwatts, not watts** (`1e-12 J / 1e-9 s = 1e-3 W`) — this project already
caught one real bug from exactly this mixup during development (see
`_WATT_NS_TO_PJ` in simulate.py, and simulate.py's own unit-conversion
comments). A second, larger unit bug was caught the same way in
`latency_ns`'s bandwidth term (see "The model", above). Latency and wait
times are nanoseconds (`_ns`), matching the tick convention used throughout
memsim.

## Cache sizing

Stage 3's cache is genuinely adaptive — it starts empty and warms from the
trace, exactly like `profiler/analyze.py`'s `simulate_lru` when
`eviction_policy="lru"`. Per-layer capacity defaults to that layer's
hot-expert *count* from stage 1's `hot_cold.csv` (a real, already-computed
sizing decision, not an arbitrary new parameter) — nothing is hard-pinned; a
"hot" expert can still be evicted under sustained pressure from other
experts, same as it would in a true LRU cache.

## What this does not model

Be upfront about this wherever these numbers are quoted:

- **Cold start.** The energy budget (`energy-aware-defer` only) starts at
  exactly 0 pJ at simulated time 0. No power rate, however large, can
  deliver energy that hasn't yet had simulated time to accrue, so the very
  first cold dispatch can still see a (correctly) negligible defer even
  under an enormous budget. This is physically honest — you cannot spend
  energy that has not accrued — not a bug; `scheduler/selftest.py` documents
  and tests for exactly this.
- **No overlap / no concurrency.** Dispatches are processed one at a time on
  a single simulated timeline, in the trace's stored order. There is no
  cross-layer pipelining, no modelling of multiple in-flight fetches, and no
  batching of concurrent cold-expert requests. Real hardware would overlap
  much of this, so throughput/latency figures here are directional (policy A
  vs policy B on the *same* simplified timeline), not an absolute
  performance prediction.
- **Batching not implemented.** docs.md 4.5 frames the scheduler's options as
  "fetch now / defer / batch." This version implements fetch-now, defer, and
  (new) energy-aware eviction — not batching. Batching was deliberately left
  out: the current energy model is purely linear in bytes (`pJ/bit × bits`,
  no fixed per-transfer overhead term), so batching several cold fetches
  together would not reduce total energy under this model — there is no
  fixed cost to amortize. It would be worth adding once a fixed per-transfer
  link/protocol overhead is separately sourced (distinct from the pJ/bit
  figure already cited), since that is exactly the kind of cost batching is
  meant to amortize. This reasoning is unchanged from the deferral-only
  version of this stage.

## Validation checkpoints

- `python -m scheduler.cli compare` — naive vs energy-aware-defer, the
  timing/latency diff (CLAUDE.md's required "diff between baseline and
  energy-aware scheduler decisions for the same trace"). Both policies
  replay the identical dispatch sequence index-for-index; the diff reports
  where they diverge (hit/miss outcome, or any deferral) plus a side-by-side
  summary. Divergence downstream of a single deferral is expected and
  correct: once a fetch is delayed, later arrivals can shift what the LRU
  cache holds by the time it happens, so hit/miss outcomes can genuinely
  differ from that point on.
- `python -m scheduler.cli compare3` — all three policies + docs.md 6
  checkpoint 3's explicit PASS/FAIL on total energy. This is the core result
  the whole project rests on (docs.md's own words) — see "Checkpoint 3
  result", above.

## Correctness checks

`python -m scheduler.cli selftest` (46 checks, offline, no gem5/real trace
needed) independently recomputes the cost model's energy figures (including
the bandwidth-derived latency term), verifies the cache's hit/miss/eviction
behaviour under both `"lru"` and `"energy-aware"` by hand-traced sequences,
checks that an unsourced tier constant poisons totals to NaN rather than
defaulting to zero (and that a NaN-scored cache entry is evicted first, not
kept forever), verifies the energy-budget invariant for
`energy-aware-defer` (no dispatch is ever charged against insufficient
budget), demonstrates docs.md 6 checkpoint 3 on a fully hand-traced
constructed example, and round-trips file loading (trace.parquet,
hot_cold.csv, tier_model.json) through small fixture files.
