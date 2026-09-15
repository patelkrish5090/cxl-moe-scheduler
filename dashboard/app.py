"""Stage 4 dashboard (docs.md 4.6): the three-way HBM-only / HBM+CXL-naive /
HBM+CXL-energy-aware comparison, plus the stage-1 activation heatmap.

Run with:
    streamlit run dashboard/app.py

Shows real output only: an experiments.cli run result (stage 4) or a
profiler hot_cold.csv (stage 1). If something is not there yet, this shows
the exact command to produce it instead of making up a number or drawing an
empty chart quietly (CLAUDE.md: no fabricated figures).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.data import (
    build_expert_skew_summary,
    build_heatmap_grid,
    build_model_comparison,
    list_comparison_results,
    list_pooling_results,
    list_stage1_runs,
    load_comparison,
    load_comparison_payload,
    load_pooling_result,
)

st.set_page_config(page_title="CXL-MoE Scheduler Dashboard", layout="wide")

st.title("Energy-Aware CXL-MoE Scheduler")
st.caption(
    "This compares three setups: HBM-only, HBM+CXL-naive, and "
    "HBM+CXL-energy-aware. It is built from real stage-1 traces and real "
    "stage-2/3 output. Nothing on this page is a made-up or placeholder number. "
    "Every figure comes from a real profiling run, a real gem5/DRAMSim3 "
    "simulation, and a real scheduler replay."
)

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
                    "data/runs/. This selector works on its own, so the heatmap "
                    "below is not necessarily from the same run as the comparison."
                )
    else:
        selected_run = None
        st.warning(
            "No stage-1 runs with a hot_cold.csv found under data/runs/."
        )

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
        "label", "throughput_tokens_per_sec", "avg_latency_ms_per_token",
        "total_energy_mj", "hit_rate", "n_tokens", "n_dispatches",
    ]].rename(columns={
        "label": "config",
        "throughput_tokens_per_sec": "throughput (tok/s)",
        "avg_latency_ms_per_token": "avg latency (ms/tok)",
        "total_energy_mj": "total energy (mJ)",
        "hit_rate": "hit rate",
    })
    st.dataframe(display_df, width='stretch', hide_index=True)

    inconsistent = df[~df["latency_accounting_consistent"]]
    if not inconsistent.empty:
        st.error(
            "**Latency accounting bug found** for: " + ", ".join(inconsistent["label"]) +
            ". The total latency, when re-added by hand (scheduler.simulate.latency_breakdown), "
            "does not match the reported total. Do not trust any latency or throughput number on "
            "this page until this is fixed."
        )

    implausible = df[~df["latency_plausible"]]
    if not implausible.empty:
        st.warning(
            "**" + ", ".join(implausible["label"]) + "**: the average latency is higher than "
            "our sanity limit. Cause: this simulator treats every step as fully serial with no "
            "overlap, using real measured tier numbers. It is not a units bug (checked below). "
            "Every cache hit still pays for a real HBM read of the expert's weights, so hits are "
            "not free either."
        )
        with st.expander("Show the exact latency breakdown and the math behind it"):
            breakdown_df = df[[
                "label", "n_hits", "n_misses", "mean_hit_latency_ns", "mean_miss_latency_ns",
                "avg_latency_ns_per_token",
            ]].rename(columns={
                "label": "config", "mean_hit_latency_ns": "mean hit latency (ns)",
                "mean_miss_latency_ns": "mean cold-fetch latency (ns)",
                "avg_latency_ns_per_token": "avg latency (ns/tok)",
            })
            st.dataframe(breakdown_df, width='stretch', hide_index=True)
            st.caption(
                "The math: avg latency (ms/tok) equals (n_hits times mean hit latency, plus "
                "n_misses times mean cold-fetch latency), divided by n_tokens. The mean "
                "cold-fetch latency implies a low CXL bandwidth of about 2.3 GB/s. We checked "
                "this directly instead of assuming it: swapping in a DRAM device rated 1.7x "
                "faster (confirmed by its own clock speed) left this bandwidth unchanged, while "
                "device energy per bit nearly doubled. So the DRAM chip itself is not what is "
                "slow here. The real cause is the CXL link, which can only handle a limited "
                "number of requests at once, a fixed delay plus a default queue depth."
            )
            for _, row in implausible.iterrows():
                st.markdown(f"**{row['label']} full breakdown:**")
                st.caption(row["latency_warning"])

    naive_row = df[df["config"] == "hbm_cxl_naive"].iloc[0]
    evict_row = df[df["config"] == "hbm_cxl_energy_aware"].iloc[0]
    ck3 = payload["checkpoint3"]
    energy_delta_pct = ck3["energy_gap_pct"]
    verdict = "lower" if energy_delta_pct > 0 else "HIGHER"
    st.caption(
        f"Checkpoint 3: the energy-aware total energy is {verdict} than "
        f"naive's, by {abs(energy_delta_pct):.2f}% on this run "
        f"({evict_row['total_energy_mj']:.6g} mJ vs {naive_row['total_energy_mj']:.6g} mJ). "
        "Two earlier designs were tried and did not pass this check before this one did."
    )
    if ck3["energy_gap_is_marginal"]:
        st.warning(
            f"**Sanity check**: the energy gap ({abs(energy_delta_pct):.2f}%) is below "
            f"the {ck3['marginal_threshold_pct']:.1f}% threshold we use to flag a small gap "
            "(`ComparisonResult.MARGINAL_ENERGY_GAP_PCT`), so it could just be noise instead of "
            "the eviction rule doing real work. This does not mean the number is wrong. Check "
            "the eviction diagnostics panel below for the real evidence before calling this a "
            "confirmed win."
        )

    st.header("Eviction diagnostics")
    st.caption(
        "Direct evidence for whether the energy-aware eviction rule is actually doing "
        "something different from plain LRU on this trace, apart from whatever the "
        "total-energy gap above shows. From scheduler.simulate.eviction_divergence_report."
    )
    div = payload["eviction_divergence"]
    d_col1, d_col2, d_col3 = st.columns(3)
    d_col1.metric("Total LRU eviction events", f"{div['total_eviction_events']:,}")
    d_col2.metric("Diverged from LRU", f"{div['divergent_eviction_events']:,}")
    d_col3.metric("Divergence rate", f"{div['divergence_rate']:.2%}")

    if div["total_eviction_events"] > 0 and div["divergence_rate"] < 0.01:
        st.warning(
            "**Eviction choice was different from LRU on fewer than 1% of events.** The "
            "energy-aware policy is barely acting different from LRU on this trace, so any "
            "total-energy improvement above is more likely noise than a real effect. Do not "
            "report a total-energy win without saying this "
            "(scheduler.simulate.EvictionDivergenceReport.summary)."
        )
    elif div["total_eviction_events"] == 0:
        st.info("No eviction events happened (the cache never filled up), so there is nothing to compare.")

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

st.header("Activation heatmap (stage 1)")
st.caption(
    "How much of each layer's dispatches go to each expert, showing which experts "
    "are hot, per layer, for the chosen stage-1 run. This is the same data as "
    "profiler/plots.py::plot_activation_heatmap's saved PNG, shown here so you can "
    "interact with it."
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
        "With this many experts and layers, the heatmap above can look fairly flat "
        "just by eye. The numbers below come from the same underlying dispatch count "
        "data, per (layer, expert) pair, and are never added up across layers first "
        "(see build_expert_skew_summary's docstring for why that matters). They give "
        "a clear, exact skew number."
    )
    skew = build_expert_skew_summary(selected_run / "hot_cold.csv")
    s_col1, s_col2 = st.columns(2)
    s_col1.metric(
        "Gini coefficient", f"{skew['gini']:.3f}",
        help="Computed per (layer, expert) pair, the same level of detail as the heatmap "
             "above, not summed across layers first. 0 means routing is perfectly even, "
             "1 means one expert gets everything.",
    )
    s_col2.metric(
        "Max/mean dispatch ratio", f"{skew['max_mean_ratio']:.2f}x",
        help="The busiest single (layer, expert) pair's dispatch count, divided by the average.",
    )

    top_col, bottom_col = st.columns(2)
    with top_col:
        st.markdown(f"**Top {len(skew['top'])} (layer, expert) pairs by share**")
        st.dataframe(pd.DataFrame(skew["top"]), width='stretch', hide_index=True)
    with bottom_col:
        st.markdown(f"**Bottom {len(skew['bottom'])} (layer, expert) pairs by share**")
        st.dataframe(pd.DataFrame(skew["bottom"]), width='stretch', hide_index=True)

st.header("Model comparison: scale, and dense vs MoE")
st.caption(
    "The problem statement asks us to look at memory access patterns across "
    "different models and expert counts, and to compare a plain Transformer "
    "against MoE. Pick 2 or more real stage-1 runs (any model, any architecture) "
    "to compare them side by side. Every number here comes straight from each "
    "run's own real run_metadata.json file. Nothing here is estimated."
)

if len(stage1_run_paths) < 2:
    st.info(
        "Need at least 2 real stage-1 runs to compare. Make more with "
        "`python -m profiler.cli run <config.json>`, for example "
        "`configs/mixtral_8x7b_decode.json`, `configs/olmoe_1b7b_decode.json`, "
        "and `configs/gpt2_dense_decode.json` (a real dense-Transformer example to "
        "compare against)."
    )
else:
    selected_for_comparison = st.multiselect(
        "Stage-1 runs to compare",
        stage1_run_paths,
        default=stage1_run_paths[: min(3, len(stage1_run_paths))],
        format_func=lambda p: p.name,
    )
    missing_metadata = [p for p in selected_for_comparison if not (p / "run_metadata.json").is_file()]
    if missing_metadata:
        st.warning(
            "No run_metadata.json for: " + ", ".join(p.name for p in missing_metadata) +
            ". These are left out below (either an older run, or hot_cold.csv was made a "
            "different way)."
        )
    comparable = [p for p in selected_for_comparison if p not in missing_metadata]

    if len(comparable) < 2:
        st.info("Select at least 2 runs with a real run_metadata.json to compare.")
    else:
        model_df = build_model_comparison(comparable)

        mc_col1, mc_col2 = st.columns(2)
        with mc_col1:
            fig = px.bar(
                model_df, x="run_name", y="gini_overall", color="architecture",
                title="Activation skew (Gini) by model",
                labels={"run_name": "", "gini_overall": "Gini coefficient", "architecture": "architecture"},
                hover_data=["model", "experts_per_layer", "top_k"],
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig)
        with mc_col2:
            fig = px.bar(
                model_df, x="run_name", y="experts_per_layer", color="architecture",
                title="Experts per layer by model (a scale measure)",
                labels={"run_name": "", "experts_per_layer": "experts / layer", "architecture": "architecture"},
                hover_data=["model", "gini_overall"],
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig)

        st.caption(
            "A dense run (architecture='dense') always shows experts_per_layer=1 and "
            "Gini=0.000, by definition. That itself is the finding: a dense Transformer "
            "has no routing decision to be skewed, so there is no hot/cold split for expert "
            "tiering to use, unlike a real routed MoE model."
        )

        display_model_df = model_df.rename(columns={
            "run_name": "run", "model": "model id", "architecture": "architecture",
            "n_layers": "layers", "experts_per_layer": "experts/layer", "top_k": "top_k",
            "total_expert_weight_gb": "total expert weight (GB)",
            "total_dispatches": "total dispatches", "gini_overall": "Gini",
            "entropy_overall": "normalized entropy", "hot_dispatch_share": "hot dispatch share",
        })
        st.dataframe(display_model_df, width='stretch', hide_index=True)

st.header("CXL memory pooling")
st.caption(
    "This shows several GPUs sharing ONE CXL memory pool, compared with each GPU "
    "having its own separate CXL space. This is the 'pooling' half of the problem "
    "statement's 'CXL memory expansion and pooling' goal, separate from the "
    "expansion (single-GPU offload) shown everywhere else on this page. Made with "
    "`scheduler.cli pool <gpu0_run_dir> <gpu1_run_dir> --out <path>` over two real "
    "per-GPU stage-1 runs of the same model."
)

pooling_paths = list_pooling_results()
if not pooling_paths:
    st.info(
        "No pooling results yet. Produce one with:\n\n"
        "`python -m scheduler.cli pool data/runs/mixtral_8x7b_decode_gpu0 "
        "data/runs/mixtral_8x7b_decode_gpu1 --out experiments/results/mixtral_pooling.json`"
    )
else:
    selected_pooling = st.selectbox(
        "Pooling result", pooling_paths, format_func=lambda p: p.stem,
    )
    pooling = load_pooling_result(selected_pooling)

    p_col1, p_col2, p_col3 = st.columns(3)
    p_col1.metric("Dedicated total (GB)", f"{pooling['dedicated_total_cold_bytes'] / 1e9:.3f}")
    p_col2.metric("Pooled total (GB)", f"{pooling['pooled_total_cold_bytes'] / 1e9:.3f}")
    p_col3.metric(
        "Savings", f"{pooling['savings_pct']:.1f}%",
        help=f"{pooling['savings_bytes'] / 1e9:.3f} GB saved, from "
             f"{pooling['shared_cold_pairs']}/{pooling['total_unique_cold_pairs']} unique cold "
             "(layer, expert) pairs needed by 2 or more GPUs.",
    )

    pooling_bar_df = pd.DataFrame({
        "storage": ["dedicated (sum of each GPU's own cold set)", "pooled (one shared copy each)"],
        "GB": [pooling["dedicated_total_cold_bytes"] / 1e9, pooling["pooled_total_cold_bytes"] / 1e9],
    })
    fig = px.bar(
        pooling_bar_df, x="storage", y="GB", color="storage",
        title="Cold-expert storage: dedicated vs pooled",
        labels={"storage": "", "GB": "GB"},
    )
    fig.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig)

    if pooling["shared_cold_pairs"] == 0:
        st.info(
            "The GPUs' cold expert sets do not overlap at all on this data, so pooling "
            "saves nothing here. This is a real result, not a broken calculation (see "
            "scheduler/pooling.py's docstring). It would likely change with more GPUs, "
            "more similar workloads across them, or a wider hot/cold threshold."
        )

    st.caption(
        "This panel does not show a shared-link bandwidth-contention latency number on "
        "purpose. scheduler/pooling.py's module docstring explains why: an earlier attempt "
        "at that turned out to be meaningless math. Adding up two independent GPUs' own "
        "latencies gives the same total no matter what order you add them in, so a 'shared "
        "clock' estimate would just equal the plain no-contention sum. Doing this honestly "
        "would need a real concurrent, step-by-step simulator, which is beyond what we built "
        "here."
    )
