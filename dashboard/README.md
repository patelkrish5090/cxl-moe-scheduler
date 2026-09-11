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

If either is missing, the page says so explicitly and prints the exact
command to produce it — it never fabricates a number or silently renders an
empty chart (CLAUDE.md).

## What it shows

1. **Three-way comparison** — grouped bar charts (throughput, average
   latency, total energy) for `hbm_only` / `hbm_cxl_naive` /
   `hbm_cxl_energy_aware`, a figures table, and a one-line docs.md 6
   checkpoint-3 verdict computed directly from the loaded result (not
   hardcoded — recomputed from whichever result file is selected).
2. **Activation heatmap** — layer x expert within-layer dispatch share for
   the selected stage-1 run. Same data and pivot as
   `profiler/plots.py::plot_activation_heatmap`'s static PNG
   (`hot_cold.csv`'s `layer_idx`/`expert_id`/`layer_share` columns), rendered
   as an interactive Plotly heatmap instead — hover a cell to read its exact
   share, rather than reading a fixed color scale off a saved image.

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
completed run), and that `build_heatmap_grid`'s pivot matches its source
values exactly.

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
