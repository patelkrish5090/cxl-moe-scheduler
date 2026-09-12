# Stage 4a — Experiment harness

Implements docs.md 4.6's harness half: runs one stage-1 trace through the
three configs the project's whole comparison rests on, and logs throughput,
average latency, and total energy for each.

## The three configs

| config | cache capacity | eviction | what it represents |
| --- | --- | --- | --- |
| `hbm_only` | unbounded (no cache at all) | n/a | the idealised, usually-infeasible baseline: the whole model resident in HBM, no CXL involved. Every dispatch is a hit by construction — see `scheduler.simulate.run_hbm_only`'s docstring for why this is NOT the same as `run_naive` with a full-size cache (that still pays real cold-start misses; a real HBM-only deployment pre-loads everything before serving starts). |
| `hbm_cxl_naive` | stage 1's real hot-expert count per site | LRU | `scheduler.simulate.run_naive` — docs.md 4.5's baseline, "what prior capacity/latency-only systems effectively do." |
| `hbm_cxl_energy_aware` | same | frequency-weighted (stage 1's real `dispatch_count`, decayed by recency) | `scheduler.simulate.run_energy_aware_evict` — the policy that passed docs.md 6 checkpoint 3 on the real Mixtral decode trace (see `scheduler/README.md`). |

No new simulation logic lives in this stage. Every config is one call into
the already-validated stage-3 simulator with a specific `cache_capacity`
dict; this module's own job is the two NEW metrics docs.md 4.6 asks for that
stage 3's `SimulationResult` doesn't already expose (throughput, average
per-token latency), plus running all three and writing the comparison out.

## Commands

```bash
python -m experiments.cli selftest                   # offline, no data needed
python -m experiments.cli run data/runs/<name>        # writes experiments/results/<name>.json
python -m experiments.cli run data/runs/<name> --tier-model memsim/tier_model.json --out <path>
```

`<name>` is a stage-1 run directory (needs `trace.parquet` + `hot_cold.csv`).
Needs `memsim/tier_model.json` to already exist (stage 2's
`python -m memsim.cli compare` output).

## Metrics and units

- `throughput_tokens_per_sec` = `n_tokens / (total_latency_ns * 1e-9)`.
  `n_tokens` is the trace's real distinct `token_uid` count, not derived from
  dispatch count (one token dispatches to multiple experts across multiple
  layers).
- `avg_latency_ns_per_token` = `total_latency_ns / n_tokens`.
- `total_energy_pj` (picojoules, matching every other stage) and
  `total_energy_mj` (`total_energy_pj * 1e-9`, kept as an explicit named
  conversion — see `experiments/harness.py`'s `_PJ_TO_MJ` — for a
  human-readable figure without redoing the pJ-vs-mJ arithmetic at every call
  site, the recurring silent bug in energy code per CLAUDE.md).
- Both are computed on the SAME simplified timeline stage 3 already documents
  as directional, not an absolute performance prediction — see
  `scheduler/README.md`'s "What this does not model" (no cross-dispatch
  overlap, one thing happens at a time on a single simulated clock).
- `avg_latency_ms_per_token` (`_NS_TO_MS`) is the same number as
  `avg_latency_ns_per_token`, just rescaled for readability — both are always
  written, never just one, so a reader never has to redo the ns/ms conversion
  by hand (the recurring silent bug class CLAUDE.md calls out).

## Latency sanity ceiling

`ConfigResult.latency_plausible` / `latency_warning`: if
`avg_latency_ms_per_token` exceeds `LATENCY_SANITY_CEILING_MS` (default 1000
ms/token), the result is flagged rather than silently trusted. This does NOT
stop at "check units" as a manual step — `ConfigResult.from_simulation` calls
`scheduler.simulate.latency_breakdown` every time the ceiling fires, which
independently recomputes the total latency from each decision's own tier
cost (a separate summation than the simulation loop's own running clock) and
compares it to the reported total:

- If they **disagree** (`latency_accounting_consistent` is `False`), the
  warning states plainly that this IS a units/accounting bug, with both
  numbers and the discrepancy percentage, and points at
  `scheduler.simulate.latency_breakdown`.
- If they **agree**, the warning states the determined, non-bug cause: this
  simulator's own documented "no overlap, one dispatch at a time" model
  (`scheduler/README.md`'s "WHAT THIS DOES NOT MODEL") combined with a real,
  slow measured CXL bandwidth, and shows the EXACT decomposition that
  reproduces the reported figure — not an approximation. An earlier version
  of this warning approximated total latency as `dispatches_per_token x
  (1 - hit_rate) x mean_miss_latency_ms`, which silently dropped the
  hit-latency term entirely (a "hit" in this model still pays an HBM read
  for the expert's weights — it is not free) and was off by ~5% on the real
  Mixtral trace as a result. The warning now quotes the exact identity
  instead: `n_hits x mean_hit_latency_ns + n_misses x mean_miss_latency_ns`,
  divided by `n_tokens`, which reconciles to the reported
  `avg_latency_ms_per_token` to the printed decimal place every time (see
  `experiments/selftest.py`'s "mixed hit/miss fixture" check, built
  specifically to catch a dropped term like this one).

On the real `mixtral_8x7b_decode` trace, `hbm_cxl_naive`'s ~6.8 s/token
figure is this second case: accounting is consistent, and the exact
decomposition (`n_hits`, `n_misses`, `mean_hit_latency_ns`,
`mean_miss_latency_ns` — all always available on `ConfigResult`, not only
when the ceiling fires) reproduces it exactly. **Still open**: whether the
~2.36 GB/s effective cold-fetch bandwidth this implies is itself a
deliberate fully-serial worst-case bound, or reflects a DRAMSim3/gem5 config
issue — see `scheduler/README.md`'s "Is a large latency figure a units bug,
or this model?" for what's confirmed and what's still pending real
gem5/DRAMSim3 config data.

## Checkpoint 3 gap and eviction divergence

`ComparisonResult.energy_gap_pct` / `energy_gap_is_marginal`: the docs.md 6
checkpoint 3 energy-aware-vs-naive percentage gap, flagged as possibly noise
when it's under `MARGINAL_ENERGY_GAP_PCT` (1.0% by default) — this project's
own real Mixtral result passed at 0.23%, right at this edge (see
`scheduler/README.md`'s "Checkpoint 3 result"). A marginal gap is not
necessarily wrong, but it needs corroborating evidence, which is exactly what
`eviction_divergence` provides.

`ComparisonResult.eviction_divergence`
(`scheduler.simulate.eviction_divergence_report`): the direct evidence for
whether the energy-aware eviction policy is doing anything different from
pure LRU on this trace, independent of the total-energy delta. Reports the
total number of LRU eviction events, how many of those the energy-aware
policy would have decided differently, the resulting divergence rate, and up
to 5 sampled diverging examples (which expert each policy evicted, each
one's real stage-1 dispatch frequency, and how many more times each was
requested later in the trace). A divergence rate under 1% is flagged the
same way a marginal energy gap is — see its `summary()` method.

## Result file

`experiments/results/<run_name>.json`:

```json
{
  "run_name": "...",
  "configs": {
    "hbm_only": {"config": "...", "n_tokens": ..., "throughput_tokens_per_sec": ...,
                 "avg_latency_ms_per_token": ..., "latency_plausible": true, "latency_warning": null, ...},
    "hbm_cxl_naive": {...},
    "hbm_cxl_energy_aware": {...}
  },
  "checkpoint3": {"energy_gap_pct": ..., "energy_gap_is_marginal": ..., "marginal_threshold_pct": 1.0},
  "eviction_divergence": {"total_eviction_events": ..., "divergent_eviction_events": ...,
                          "divergence_rate": ..., "examples": [...]},
  "units": {"energy": "pJ (also reported as mJ)", "latency": "ns (also reported as ms)", "throughput": "tokens/sec"}
}
```

This is what `dashboard/app.py` reads — see `dashboard/README.md`.

## Correctness checks

`python -m experiments.cli selftest` independently recomputes throughput and
average latency from hand-chosen per-dispatch figures, checks the zero-token
degenerate case gives NaN rather than a `ZeroDivisionError`, checks the
latency sanity ceiling fires past its threshold and not at/below it, runs the
full pipeline end-to-end against small synthetic fixture files (verifying
`hbm_only` genuinely reaches 100% hit rate — the exact bug this checked for
during development, see `scheduler.simulate.run_hbm_only`'s docstring, and
that `eviction_divergence` is actually populated by `run_comparison`, not
left as a missing field), and round-trips the written JSON file including the
new `checkpoint3` and `eviction_divergence` sections.
