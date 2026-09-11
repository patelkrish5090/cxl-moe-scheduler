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

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.data import (
    build_heatmap_grid,
    list_comparison_results,
    list_stage1_runs,
    load_comparison,
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
        selected_run = st.selectbox(
            "Stage-1 run (for the activation heatmap)",
            stage1_run_paths,
            format_func=lambda p: p.name,
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

    col1, col2, col3 = st.columns(3)
    for col, metric, title, fmt in (
        (col1, "throughput_tokens_per_sec", "Throughput (tokens/sec)", "{:.4g}"),
        (col2, "avg_latency_ns_per_token", "Avg latency (ns/token)", "{:.4g}"),
        (col3, "total_energy_mj", "Total energy (mJ)", "{:.4g}"),
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
        "total_energy_mj", "hit_rate", "n_tokens", "n_dispatches",
    ]].rename(columns={
        "label": "config",
        "throughput_tokens_per_sec": "throughput (tok/s)",
        "avg_latency_ns_per_token": "avg latency (ns/tok)",
        "total_energy_mj": "total energy (mJ)",
        "hit_rate": "hit rate",
    })
    st.dataframe(display_df, width='stretch', hide_index=True)

    naive_row = df[df["config"] == "hbm_cxl_naive"].iloc[0]
    evict_row = df[df["config"] == "hbm_cxl_energy_aware"].iloc[0]
    energy_delta_pct = 100.0 * (naive_row["total_energy_mj"] - evict_row["total_energy_mj"]) / naive_row["total_energy_mj"]
    verdict = "lower" if energy_delta_pct > 0 else "HIGHER"
    st.caption(
        f"docs.md 6 checkpoint 3: energy-aware total energy is {verdict} than "
        f"naive's by {abs(energy_delta_pct):.2f}% on this run "
        f"({evict_row['total_energy_mj']:.6g} mJ vs {naive_row['total_energy_mj']:.6g} mJ). "
        "See scheduler/README.md's 'Checkpoint 3 result' for the full story, "
        "including two earlier design attempts that failed before this one passed."
    )

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
