"""Pure data-loading/transformation functions for the stage-4 dashboard,
kept separate from dashboard/app.py's Streamlit UI code specifically so they
can be unit-tested offline (see dashboard/selftest.py) -- a Streamlit script
itself isn't unit-testable in this project's check()-based style, but the
data it renders should be held to the same standard as every other stage.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from profiler.classify import gini

#: Column order for the three-way comparison table/charts. Matches
#: experiments/harness.py's ComparisonResult field order.
CONFIG_ORDER = ["hbm_only", "hbm_cxl_naive", "hbm_cxl_energy_aware"]

#: Human-readable labels for the three configs, for chart/table display.
CONFIG_LABELS = {
    "hbm_only": "HBM-only (idealised)",
    "hbm_cxl_naive": "HBM+CXL, naive",
    "hbm_cxl_energy_aware": "HBM+CXL, energy-aware",
}


def list_comparison_results(results_dir: str | Path = "experiments/results") -> list[Path]:
    """Every experiment-comparison JSON written by experiments.cli run.

    experiments/results/ is also where scheduler.cli pool --out writes CXL
    pooling results (list_pooling_results) -- both write into the same
    directory by convention, so a pooling JSON must be excluded here (it has
    no "configs" key and would otherwise raise a KeyError the moment
    load_comparison tried to read it), the same way list_pooling_results
    excludes comparison files by checking for "gpu_names" instead.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return []
    found = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "configs" in payload:
            found.append(path)
    return found


def load_comparison_payload(path: str | Path) -> dict:
    """Raw JSON payload of an experiments.cli run result -- everything
    load_comparison's tidy per-config DataFrame doesn't carry (run_name,
    checkpoint3 verdict, eviction_divergence), for the dashboard sections
    that need those directly rather than a per-config row.

    Raises:
        FileNotFoundError: if path does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"no comparison result at {path}. Produce one with:\n"
            "  python -m experiments.cli run data/runs/<name>"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_comparison(path: str | Path) -> pd.DataFrame:
    """One experiments.cli run's three-way comparison, as a tidy DataFrame --
    one row per config, in CONFIG_ORDER.

    Raises:
        FileNotFoundError: if path does not exist.
        KeyError: if the file is missing an expected config (a malformed or
            stale result file -- surfaced rather than silently dropping rows).
    """
    payload = load_comparison_payload(path)
    rows = []
    for config_key in CONFIG_ORDER:
        cfg = payload["configs"][config_key]
        rows.append({
            "config": config_key,
            "label": CONFIG_LABELS[config_key],
            "throughput_tokens_per_sec": cfg["throughput_tokens_per_sec"],
            "avg_latency_ns_per_token": cfg["avg_latency_ns_per_token"],
            "avg_latency_ms_per_token": cfg["avg_latency_ms_per_token"],
            "total_energy_mj": cfg["total_energy_mj"],
            "hit_rate": cfg["hit_rate"],
            "n_tokens": cfg["n_tokens"],
            "n_dispatches": cfg["n_dispatches"],
            "n_hits": cfg["n_hits"],
            "n_misses": cfg["n_misses"],
            "mean_hit_latency_ns": cfg["mean_hit_latency_ns"],
            "mean_miss_latency_ns": cfg["mean_miss_latency_ns"],
            "latency_accounting_consistent": cfg["latency_accounting_consistent"],
            "latency_plausible": cfg["latency_plausible"],
            "latency_warning": cfg["latency_warning"],
        })
    return pd.DataFrame(rows)


def list_stage1_runs(data_runs_dir: str | Path = "data/runs") -> list[Path]:
    """Every stage-1 run directory that has a hot_cold.csv (i.e. a completed
    profiling run, not just an empty placeholder directory).
    """
    data_runs_dir = Path(data_runs_dir)
    if not data_runs_dir.is_dir():
        return []
    return sorted(p.parent for p in data_runs_dir.glob("*/hot_cold.csv"))


def build_heatmap_grid(hot_cold_csv: str | Path) -> pd.DataFrame:
    """Layer x expert grid of within-layer dispatch share, from stage 1's
    hot_cold.csv -- the same data and pivot profiler/plots.py's
    plot_activation_heatmap uses for its static PNG, rendered here as a
    DataFrame so the dashboard can build an interactive Plotly heatmap
    instead of embedding a static image.
    """
    hot_cold_csv = Path(hot_cold_csv)
    if not hot_cold_csv.is_file():
        raise FileNotFoundError(f"no hot_cold.csv at {hot_cold_csv}")
    table = pd.read_csv(hot_cold_csv)
    grid = table.pivot(index="layer_idx", columns="expert_id", values="layer_share")
    return grid.sort_index()


def build_expert_skew_summary(hot_cold_csv: str | Path, top_n: int = 5) -> dict:
    """Numeric summary of how skewed expert usage actually is, for the
    selected stage-1 run -- the heatmap alone reads as fairly flat by eye at
    Mixtral's scale (32 layers x 8 experts), so this gives the same
    checkpoint a number.

    Computed over the SAME bins as profiler.classify's own
    ``gini_overall`` -- one independent count per (layer_idx, expert_id)
    pair, exactly as hot_cold.csv already stores one row per pair. This is
    NOT the same as summing dispatch_count by expert_id first and measuring
    skew across that: an earlier version of this function did exactly that,
    and it is a real bug, not a stylistic choice -- Mixtral routes each
    layer somewhat independently (per its own published routing analysis,
    arXiv:2401.04088 sec. 5), so two layers that each favour a *different*
    expert average out toward uniform once collapsed into 8 per-expert
    totals, even though every individual layer is genuinely skewed. Collapsing
    away the layer axis before measuring skew silently erases the exact
    signal this checkpoint exists to catch (caught when this function's
    result, ~0.03 Gini, contradicted profiler.classify's own gini_overall for
    the same run, ~0.115 -- see dashboard/selftest.py's regression test for a
    constructed example that reproduces this directly).

    Gini (profiler.classify.gini -- 0 = uniform, 1 = one bin takes
    everything) and max/mean ratio (a second, more literal skew measure) are
    both reported, since neither alone is self-explanatory.

    Returns:
        {"gini": float, "max_mean_ratio": float, "top": [...], "bottom": [...]}
        where "top"/"bottom" are the `top_n` (layer, expert) bins by share of
        all dispatches in this run, each {"layer_idx": int, "expert_id": int,
        "dispatch_count": int, "share": float}.

    Raises:
        FileNotFoundError: if hot_cold_csv does not exist.
    """
    hot_cold_csv = Path(hot_cold_csv)
    if not hot_cold_csv.is_file():
        raise FileNotFoundError(f"no hot_cold.csv at {hot_cold_csv}")
    table = pd.read_csv(hot_cold_csv)

    counts = table["dispatch_count"].to_numpy(dtype=float)
    total = float(counts.sum())
    mean = float(counts.mean()) if counts.size else 0.0
    max_mean_ratio = (float(counts.max()) / mean) if mean > 0 else math.nan

    ranked = table.sort_values("dispatch_count", ascending=False)

    def _rows(sub: pd.DataFrame) -> list[dict]:
        return [
            {
                "layer_idx": int(row.layer_idx),
                "expert_id": int(row.expert_id),
                "dispatch_count": int(row.dispatch_count),
                "share": float(row.dispatch_count / total) if total > 0 else 0.0,
            }
            for row in sub.itertuples()
        ]

    return {
        "gini": gini(counts),
        "max_mean_ratio": max_mean_ratio,
        "top": _rows(ranked.head(top_n)),
        "bottom": _rows(ranked.tail(top_n)),
    }


def load_run_metadata(run_dir: str | Path) -> dict:
    """``run_metadata.json`` from a stage-1 run directory -- model topology
    (experts/layer, top_k, total weight bytes) and the real classification
    summary (gini_overall, normalized_entropy_overall, hot_dispatch_share),
    written by profiler.runner.run() for every run, MoE or dense alike.

    Raises:
        FileNotFoundError: if run_metadata.json does not exist.
    """
    run_dir = Path(run_dir)
    path = run_dir / "run_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no run_metadata.json at {path}. Produce it with:\n"
            "  python -m profiler.cli run <config.json>"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_model_comparison(run_dirs: list[str | Path]) -> pd.DataFrame:
    """One row per stage-1 run, for comparing architectures/scales side by
    side -- the "scalability" (expert count) and "dense vs MoE" comparisons
    the dashboard's Model Comparison panel renders. Every field comes
    straight from that run's own real run_metadata.json; nothing here is
    computed, estimated, or hardcoded per model.

    Raises:
        FileNotFoundError: if any run_dir lacks a run_metadata.json (see
            load_run_metadata).
        KeyError: if a run_metadata.json is missing an expected key (an
            older or malformed run) -- surfaced rather than silently
            producing a row of blanks.
    """
    rows = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        meta = load_run_metadata(run_dir)
        topo = meta["model_topology"]
        classification = meta["classification"]
        experts_per_layer = topo["experts_per_layer"]
        rows.append({
            "run_name": meta["run_name"],
            "model": meta["config"]["model"]["name_or_path"],
            "architecture": meta["config"]["model"].get("architecture", "moe"),
            "n_layers": topo["n_moe_layers"],
            "experts_per_layer": experts_per_layer[0] if experts_per_layer else None,
            "top_k": topo["top_k"],
            "total_expert_weight_gb": topo["total_expert_weight_bytes"] / 1e9,
            "total_dispatches": classification["total_dispatches"],
            "gini_overall": classification["gini_overall"],
            "entropy_overall": classification["normalized_entropy_overall"],
            "hot_dispatch_share": classification["hot_dispatch_share"],
        })
    return pd.DataFrame(rows)


def list_pooling_results(results_dir: str | Path = "experiments/results") -> list[Path]:
    """Every CXL-pooling JSON written by ``scheduler.cli pool --out``.

    Distinguished from a three-way comparison result by the presence of a
    top-level ``"gpu_names"`` key (comparison results have ``"configs"``
    instead) -- both are written into the same results directory by
    convention, so list_comparison_results alone would also match these.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return []
    found = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "gpu_names" in payload:
            found.append(path)
    return found


def load_pooling_result(path: str | Path) -> dict:
    """Raw JSON payload of a ``scheduler.cli pool --out`` result.

    Raises:
        FileNotFoundError: if path does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"no pooling result at {path}. Produce one with:\n"
            "  python -m scheduler.cli pool <gpu0_run_dir> <gpu1_run_dir> --out <path>"
        )
    return json.loads(path.read_text(encoding="utf-8"))
