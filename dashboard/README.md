# Stage 4b — Dashboard

Implements docs.md 4.6's visualisation half: a Streamlit app showing the
three-way comparison from `experiments/` plus the activation heatmap from
stage 1.

## Run it

```bash
streamlit run dashboard/app.py
```

Opens in a browser (Streamlit prints the local URL). Two sidebar selectors:

- **Experiment result** — any `experiments/results/*.json` written by
  `python -m experiments.cli run`.
- **Stage-1 run** — any `data/runs/<name>/` directory with a `hot_cold.csv`.
  Defaults to the run that actually produced the selected experiment result
  (matched by name — `experiments.cli run data/runs/<name>` always writes
  `<name>.json`), so the two selectors can't silently point at different
  runs. A checkbox ("use an independent stage-1 run") opts out of the link
  for the rare case of wanting to look at a different run's heatmap; if the
  matching run is missing entirely, the page falls back to the independent
  selector automatically and says so explicitly, rather than silently
  showing a heatmap from an unrelated run.

If either is missing, the page says so explicitly and prints the exact
command to produce it — it never fabricates a number or silently renders an
empty chart (CLAUDE.md).

## What it shows

1. **Three-way comparison** — grouped bar charts (throughput, average
   latency in ms/token, total energy) for `hbm_only` / `hbm_cxl_naive` /
   `hbm_cxl_energy_aware`, a figures table (both ns/tok and ms/tok), a
   one-line docs.md 6 checkpoint-3 verdict computed directly from the loaded
   result (not hardcoded — recomputed from whichever result file is
   selected), and two sanity-check banners that fire but never hide the
   underlying number: a per-config warning when `avg_latency_ms_per_token`
   exceeds `experiments.harness.LATENCY_SANITY_CEILING_MS`, and a banner when
   the checkpoint-3 energy gap is below `MARGINAL_ENERGY_GAP_PCT` (1%) and
   could plausibly be noise.
2. **Eviction diagnostics** — the direct evidence for whether the
   energy-aware eviction policy actually does anything different from pure
   LRU on this trace (`scheduler.simulate.eviction_divergence_report`):
   total LRU eviction events, how many diverged, the divergence rate (with
   its own near-zero warning), and up to 5 sampled diverging decisions
   (which expert each policy evicted, its real stage-1 dispatch frequency,
   and how many more times it was requested later in the trace).
3. **Activation heatmap** — layer x expert within-layer dispatch share for
   the selected stage-1 run. Same data and pivot as
   `profiler/plots.py::plot_activation_heatmap`'s static PNG
   (`hot_cold.csv`'s `layer_idx`/`expert_id`/`layer_share` columns), rendered
   as an interactive Plotly heatmap instead — hover a cell to read its exact
   share, rather than reading a fixed color scale off a saved image.
4. **Activation skew summary** — the heatmap alone reads as fairly flat by
   eye at Mixtral's scale (32 layers x 8 experts), so this adds a number:
   overall Gini coefficient and max/mean dispatch ratio
   (`dashboard.data.build_expert_skew_summary`, reusing
   `profiler.classify.gini`), plus the top-5 and bottom-5 individual
   (layer, expert) bins by share of all dispatches in the run. Computed at
   the SAME granularity as `profiler.classify`'s own `gini_overall` — one
   independent bin per (layer, expert) pair, never summed by expert_id
   across layers first. An earlier version of this function did exactly
   that cross-layer sum before measuring skew, and it silently erased the
   very signal this checkpoint exists to catch: Mixtral routes each layer
   somewhat independently (arXiv:2401.04088 sec. 5), so two layers that are
   each genuinely, strongly skewed toward a *different* expert average out
   toward uniform once collapsed into per-expert totals. See
   `build_expert_skew_summary`'s docstring and `dashboard/selftest.py`'s
   "cross-layer washout regression" test, which reproduces that exact bug on
   a constructed example (two maximally-skewed layers with identical
   per-expert totals, correctly still reported as skewed, not uniform).

## Architecture

`dashboard/data.py` holds every data-loading/transformation function as a
plain function returning a `pandas.DataFrame` — no Streamlit calls in that
file at all. `dashboard/app.py` is thin: it calls into `data.py` and renders
whatever comes back. This split exists specifically so the data layer is
unit-testable (`dashboard/selftest.py`) in this project's established style,
since a Streamlit script itself isn't testable that way — it needs a running
server. Keep new data logic in `data.py`, not inline in `app.py`.

`app.py` inserts the project root onto `sys.path` explicitly at the top,
since `streamlit run dashboard/app.py` (unlike `python -m`) does not do this
automatically — without it, `from dashboard.data import ...` fails depending
on the caller's working directory.

## Correctness checks

`python -m dashboard.selftest` (offline, no Streamlit server needed) checks
`list_comparison_results`/`list_stage1_runs` degrade to an empty list rather
than raising when a directory doesn't exist yet, that `load_comparison`
raises `FileNotFoundError` naming the producing command (not a silent
default) and `KeyError` on a malformed/incomplete result file rather than
quietly dropping a row, that `list_stage1_runs` correctly excludes a run
directory with no `hot_cold.csv` (an empty placeholder, not a real
completed run), that `build_heatmap_grid`'s pivot matches its source values
exactly, that `load_comparison_payload` carries the `checkpoint3` and
`eviction_divergence` sections through untouched, and that
`build_expert_skew_summary`'s per-(layer,expert)-bin Gini and max/mean ratio
match an independent recompute (including the degenerate perfectly-uniform
case, where Gini must be exactly 0 and the ratio exactly 1.0, and the
cross-layer washout regression case described above).

The app itself (`dashboard/app.py`) was verified headlessly with
Streamlit's own `streamlit.testing.v1.AppTest` during development, both
against an empty repository (no data yet — confirms the guidance messages
render correctly) and against real fixture data (confirms the charts and
heatmap render with no exceptions). That verification isn't part of the
repo's own test suite (`AppTest` needs the full `streamlit`/`plotly`
runtime, which the other stages' offline selftests deliberately avoid
depending on) — it caught two real Streamlit-version deprecation issues
before they shipped: `use_container_width` (already past its removal
deadline in the installed Streamlit version) and an incorrect `width=`
kwarg on `st.plotly_chart` (which doesn't actually accept that parameter in
this version — only `st.dataframe` does). If Streamlit is upgraded later
and the dashboard starts warning or breaking, re-running that same
`AppTest`-based check first is faster than guessing at the new API.
