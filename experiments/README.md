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

## Result file

`experiments/results/<run_name>.json`:

```json
{
  "run_name": "...",
  "configs": {
    "hbm_only": {"config": "...", "n_tokens": ..., "throughput_tokens_per_sec": ..., ...},
    "hbm_cxl_naive": {...},
    "hbm_cxl_energy_aware": {...}
  },
  "units": {"energy": "pJ (also reported as mJ)", "latency": "ns", "throughput": "tokens/sec"}
}
```

This is what `dashboard/app.py` reads — see `dashboard/README.md`.

## Correctness checks

`python -m experiments.cli selftest` independently recomputes throughput and
average latency from hand-chosen per-dispatch figures, checks the zero-token
degenerate case gives NaN rather than a `ZeroDivisionError`, runs the full
pipeline end-to-end against small synthetic fixture files (verifying
`hbm_only` genuinely reaches 100% hit rate — the exact bug this checked for
during development, see `scheduler.simulate.run_hbm_only`'s docstring), and
round-trips the written JSON file.
