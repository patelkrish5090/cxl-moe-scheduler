"""Stage 4 dashboard (docs.md 4.6): the three-way HBM-only / HBM+CXL-naive /
HBM+CXL-energy-aware comparison, plus the stage-1 activation heatmap.

Run with:
    streamlit run dashboard/app.py

Renders real output only -- an experiments.cli run result (stage 4) or a
profiler hot_cold.csv (stage 1). If neither exists yet for what's selected,
this shows the exact command to produce it rather than fabricating a number
or plotting an empty chart silently (CLAUDE.md: no fabricated figures).
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run dashboard/app.py` does not add the project root to sys.path
# the way `python -m` would -- add it explicitly so `from dashboard.data
# import ...` resolves regardless of the caller's cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.data import (
    build_expert_skew_summary,
    build_heatmap_grid,
    list_comparison_results,
    list_stage1_runs,
    load_comparison,
    load_comparison_payload,
)

st.set_page_config(page_title="CXL-MoE Scheduler Dashboard", layout="wide")

st.title("Energy-Aware CXL-MoE Scheduler")
st.caption(
    "docs.md 4.6 -- HBM-only vs HBM+CXL-naive vs HBM+CXL-energy-aware, "
    "driven off real stage-1 traces and stage-2/3 output. Nothing on this "
    "page is a fabricated or placeholder number: every figure traces back to "
    "a real profiling run, a real gem5/DRAMSim3 simulation, and a real "
    "scheduler replay."
)

# --------------------------------------------------------------------- data
comparison_paths = list_comparison_results()
stage1_run_paths = list_stage1_runs()

with st.sidebar:
    st.header("Data source")
    if comparison_paths:
        selected_comparison = st.selectbox(
            "Experiment result (stage 4)",
            comparison_paths,
            format_func=lambda p: p.stem,
        )
    else:
        selected_comparison = None
        st.warning(
            "No experiment results yet. Produce one with:\n\n"
            "`python -m experiments.cli run data/runs/<name>`"
        )

    if stage1_run_paths:
        # experiments.cli run always writes <name>.json from a data/runs/<name>
        # directory (experiments/cli.py: out_path defaults to
        # DEFAULT_RESULTS_DIR / f"{run_dir.name}.json"), so the matching
        # stage-1 run for the selected comparison is the one with the same
        # name. Default to it so the two selectors can't silently point at
        # different runs -- only decouple them if the matching run is
        # missing, or the user explicitly asks to.
        matching_run = None
        if selected_comparison is not None:
            matching_run = next(
                (p for p in stage1_run_paths if p.name == selected_comparison.stem), None
            )

        independent = st.checkbox(
            "Use an independent stage-1 run (not the one that produced the "
            "selected experiment result)",
            value=matching_run is None and selected_comparison is not None,
        )

        if matching_run is not None and not independent:
            selected_run = matching_run
            st.caption(f"Linked to experiment result's stage-1 run: `{selected_run.name}`")
        else:
            selected_run = st.selectbox(
                "Stage-1 run (for the activation heatmap)",
                stage1_run_paths,
                format_func=lambda p: p.name,
            )
            if selected_comparison is not None and matching_run is None:
                st.warning(
                    f"No stage-1 run named `{selected_comparison.stem}` found under "
                    "data/runs/ -- this selector is independent of the experiment "
                    "result above; the heatmap below is NOT necessarily from the "
                    "same run as the comparison."
                )
    else:
        selected_run = None
        st.warning(
            "No stage-1 runs with a hot_cold.csv found under data/runs/."
        )

# ------------------------------------------------------- three-way comparison
st.header("Three-way comparison")

if selected_comparison is None:
    st.info("Select or produce an experiment result to see the comparison.")
else:
    df = load_comparison(selected_comparison)
    payload = load_comparison_payload(selected_comparison)

    col1, col2, col3 = st.columns(3)
    for col, metric, title in (
        (col1, "throughput_tokens_per_sec", "Throughput (tokens/sec)"),
        (col2, "avg_latency_ms_per_token", "Avg latency (ms/token)"),
        (col3, "total_energy_mj", "Total energy (mJ)"),
    ):
        with col:
            fig = px.bar(
                df, x="label", y=metric, color="label",
                title=title, labels={"label": "", metric: title},
            )
            fig.update_layout(showlegend=False, height=360)
            st.plotly_chart(fig)

    st.subheader("Figures")
    display_df = df[[
        "label", "throughput_tokens_per_sec", "avg_latency_ns_per_token",
        "avg_latency_ms_per_token", "total_energy_mj", "hit_rate", "n_tokens", "n_dispatches",
    ]].rename(columns={
        "label": "config",
        "throughput_tokens_per_sec": "throughput (tok/s)",
        "avg_latency_ns_per_token": "avg latency (ns/tok)",
        "avg_latency_ms_per_token": "avg latency (ms/tok)",
        "total_energy_mj": "total energy (mJ)",
        "hit_rate": "hit rate",
    })
    st.dataframe(display_df, width='stretch', hide_index=True)

    # Per-config latency sanity-check warnings (experiments/harness.py's
    # LATENCY_SANITY_CEILING_MS) -- surfaced here, not silently dropped, even
    # though a real run on this project's own trace legitimately triggers it
    # (see that constant's docstring for the walked-back explanation).
    for _, row in df.iterrows():
        if not row["latency_plausible"]:
            st.warning(f"**{row['label']}**: {row['latency_warning']}")

    naive_row = df[df["config"] == "hbm_cxl_naive"].iloc[0]
    evict_row = df[df["config"] == "hbm_cxl_energy_aware"].iloc[0]
    ck3 = payload["checkpoint3"]
    energy_delta_pct = ck3["energy_gap_pct"]
    verdict = "lower" if energy_delta_pct > 0 else "HIGHER"
    st.caption(
        f"docs.md 6 checkpoint 3: energy-aware total energy is {verdict} than "
        f"naive's by {abs(energy_delta_pct):.2f}% on this run "
        f"({evict_row['total_energy_mj']:.6g} mJ vs {naive_row['total_energy_mj']:.6g} mJ). "
        "See scheduler/README.md's 'Checkpoint 3 result' for the full story, "
        "including two earlier design attempts that failed before this one passed."
    )
    if ck3["energy_gap_is_marginal"]:
        st.warning(
            f"**Sanity check**: the energy gap ({abs(energy_delta_pct):.2f}%) is below "
            f"the {ck3['marginal_threshold_pct']:.1f}% marginal-gap threshold "
            "(`ComparisonResult.MARGINAL_ENERGY_GAP_PCT`) and could plausibly be noise "
            "rather than the eviction mechanism doing real work. This does NOT mean the "
            "number is wrong -- check the eviction diagnostics panel below for the actual "
            "divergence-rate evidence before treating this as a validated win."
        )

    # -------------------------------------------------------- eviction diagnostics
    st.header("Eviction diagnostics")
    st.caption(
        "Direct evidence for whether the energy-aware eviction policy is doing "
        "anything different from pure LRU on this trace, independent of whatever "
        "the total-energy delta above happens to be -- "
        "scheduler.simulate.eviction_divergence_report."
    )
    div = payload["eviction_divergence"]
    d_col1, d_col2, d_col3 = st.columns(3)
    d_col1.metric("Total LRU eviction events", f"{div['total_eviction_events']:,}")
    d_col2.metric("Diverged from LRU", f"{div['divergent_eviction_events']:,}")
    d_col3.metric("Divergence rate", f"{div['divergence_rate']:.2%}")

    if div["total_eviction_events"] > 0 and div["divergence_rate"] < 0.01:
        st.warning(
            "**Eviction choice diverged on fewer than 1% of eviction events.** The "
            "energy-aware policy is barely doing anything different from LRU on this "
            "trace -- any total-energy improvement above is more likely noise than a "
            "real mechanism effect. Do not report a total-energy win without this "
            "caveat (scheduler.simulate.EvictionDivergenceReport.summary)."
        )
    elif div["total_eviction_events"] == 0:
        st.info("No eviction events occurred (cache never filled) -- nothing to compare.")

    if div["examples"]:
        st.subheader("Sample diverging decisions")
        examples_df = pd.DataFrame(div["examples"]).rename(columns={
            "token_uid": "token",
            "site_idx": "site",
            "dispatch_index": "dispatch #",
            "lru_evicted_expert": "LRU evicted",
            "energy_aware_evicted_expert": "energy-aware evicted",
            "lru_evicted_global_frequency": "LRU-evicted global freq",
            "energy_aware_evicted_global_frequency": "energy-aware-evicted global freq",
            "lru_evicted_future_requests": "LRU-evicted future requests",
            "energy_aware_evicted_future_requests": "energy-aware-evicted future requests",
        })
        st.dataframe(examples_df, width='stretch', hide_index=True)

# ------------------------------------------------------------ activation heatmap
st.header("Activation heatmap (stage 1)")
st.caption(
    "Within-layer dispatch share per expert -- which experts are hot, per "
    "layer, for the selected stage-1 run. Same data as "
    "profiler/plots.py::plot_activation_heatmap's static PNG, rendered "
    "interactively here."
)

if selected_run is None:
    st.info("Select a stage-1 run to see its activation heatmap.")
else:
    grid = build_heatmap_grid(selected_run / "hot_cold.csv")
    fig = go.Figure(data=go.Heatmap(
        z=grid.to_numpy(),
        x=[str(c) for c in grid.columns],
        y=[str(i) for i in grid.index],
        colorscale="Magma",
        colorbar={"title": "share"},
        hovertemplate="layer %{y}<br>expert %{x}<br>share %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="expert id",
        yaxis_title="transformer layer index",
        height=max(400, 22 * len(grid.index) + 150),
    )
    st.plotly_chart(fig)

    st.subheader("Activation skew summary")
    st.caption(
        "The heatmap above can read as fairly flat by eye at this many experts x "
        "layers -- these numbers are the same underlying dispatch_count data, "
        "summed across all layers per expert, as an explicit skew figure."
    )
    skew = build_expert_skew_summary(selected_run / "hot_cold.csv")
    s_col1, s_col2 = st.columns(2)
    s_col1.metric(
        "Gini coefficient", f"{skew['gini']:.3f}",
        help="0 = perfectly uniform routing across experts, 1 = all traffic to one expert.",
    )
    s_col2.metric(
        "Max/mean dispatch ratio", f"{skew['max_mean_ratio']:.2f}x",
        help="The busiest expert's total dispatch count, divided by the average across all experts.",
    )

    top_col, bottom_col = st.columns(2)
    with top_col:
        st.markdown(f"**Top {len(skew['top'])} experts by share**")
        st.dataframe(pd.DataFrame(skew["top"]), width='stretch', hide_index=True)
    with bottom_col:
        st.markdown(f"**Bottom {len(skew['bottom'])} experts by share**")
        st.dataframe(pd.DataFrame(skew["bottom"]), width='stretch', hide_index=True)
