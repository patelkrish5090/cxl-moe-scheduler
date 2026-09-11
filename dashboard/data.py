"""Pure data-loading/transformation functions for the stage-4 dashboard,
kept separate from dashboard/app.py's Streamlit UI code specifically so they
can be unit-tested offline (see dashboard/selftest.py) -- a Streamlit script
itself isn't unit-testable in this project's check()-based style, but the
data it renders should be held to the same standard as every other stage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

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


def load_comparison(path: str | Path) -> pd.DataFrame:
    """One experiments.cli run's three-way comparison, as a tidy DataFrame --
    one row per config, in CONFIG_ORDER.

    Raises:
        FileNotFoundError: if path does not exist.
        KeyError: if the file is missing an expected config (a malformed or
            stale result file -- surfaced rather than silently dropping rows).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"no comparison result at {path}. Produce one with:\n"
            "  python -m experiments.cli run data/runs/<name>"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for config_key in CONFIG_ORDER:
        cfg = payload["configs"][config_key]
        rows.append({
            "config": config_key,
            "label": CONFIG_LABELS[config_key],
            "throughput_tokens_per_sec": cfg["throughput_tokens_per_sec"],
            "avg_latency_ns_per_token": cfg["avg_latency_ns_per_token"],
            "total_energy_mj": cfg["total_energy_mj"],
            "hit_rate": cfg["hit_rate"],
            "n_tokens": cfg["n_tokens"],
            "n_dispatches": cfg["n_dispatches"],
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
