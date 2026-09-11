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
    """Every experiment-comparison JSON written by experiments.cli run."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return []
    return sorted(results_dir.glob("*.json"))


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
    Mixtral's scale (32 experts x many layers), so this gives the same
    checkpoint a number: total dispatch share per expert (summed across all
    layers), overall Gini coefficient (profiler.classify.gini -- 0 = uniform,
    1 = one expert takes everything), and max/mean ratio (a second, more
    literal skew measure that doesn't require knowing how to read a Gini
    coefficient).

    Returns:
        {"gini": float, "max_mean_ratio": float, "top": [...], "bottom": [...]}
        where "top"/"bottom" are the `top_n` experts by total dispatch share,
        each {"expert_id": int, "dispatch_count": int, "share": float}.

    Raises:
        FileNotFoundError: if hot_cold_csv does not exist.
    """
    hot_cold_csv = Path(hot_cold_csv)
    if not hot_cold_csv.is_file():
        raise FileNotFoundError(f"no hot_cold.csv at {hot_cold_csv}")
    table = pd.read_csv(hot_cold_csv)
    by_expert = table.groupby("expert_id")["dispatch_count"].sum().sort_values(ascending=False)
    total = float(by_expert.sum())
    shares = (by_expert / total) if total > 0 else by_expert.astype(float) * 0.0

    counts = by_expert.to_numpy(dtype=float)
    mean = float(counts.mean()) if counts.size else 0.0
    max_mean_ratio = (float(counts.max()) / mean) if mean > 0 else math.nan

    def _rows(series: pd.Series) -> list[dict]:
        return [
            {"expert_id": int(expert_id), "dispatch_count": int(by_expert[expert_id]), "share": float(share)}
            for expert_id, share in series.items()
        ]

    return {
        "gini": gini(counts),
        "max_mean_ratio": max_mean_ratio,
        "top": _rows(shares.head(top_n)),
        "bottom": _rows(shares.tail(top_n)),
    }
