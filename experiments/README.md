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
ms/token), the result is flagged rather than silently trusted. This is NOT a
claim that latencies above the ceiling are wrong — on the real
`mixtral_8x7b_decode` trace, `hbm_cxl_naive`'s ~6.8 s/token figure is real,
walked back to a real, slow measured CXL bandwidth (2.36 GB/s) times 179,090
sequential 352 MB expert fetches with zero concurrency, under this
simulator's own documented "no overlap" model. The ceiling exists to force
that explanation to be checked and stated every time a number this large
shows up, not to assert it's impossible — see `LATENCY_SANITY_CEILING_MS`'s
docstring in `experiments/harness.py`.

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
