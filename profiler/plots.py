"""Sanity plots for a profiling run (docs.md section 6, checkpoint 1).

Three figures, each answering a specific question:

* ``activation_histogram.png`` -- is routing skewed, or flat? A flat histogram on
  a *trained* model means the hooks are wrong, not that the model is unusual.
* ``activation_heatmap.png``   -- which experts are hot, per layer? Also feeds the
  stage-4 dashboard.
* ``coverage_curve.png``       -- how many experts must stay resident to serve a
  given share of dispatches? This is the number that justifies the hot/cold
  threshold, and directly sizes stage 3's HBM budget.

Uses matplotlib's default colour cycle; no seaborn, no style file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless: profiling runs on a compute node
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .classify import ClassificationResult, coverage_curve  # noqa: E402


def plot_activation_histogram(
    result: ClassificationResult, path: str | Path, title_suffix: str = ""
) -> Path:
    """Sorted per-expert dispatch counts, pooled across layers, hot set shaded.

    The x-axis is expert rank, not expert id, so the shape of the distribution is
    readable at a glance regardless of which ids happen to be popular.
    """
    table = result.table
    ordered = table.sort_values("dispatch_count", ascending=False).reset_index(drop=True)
    ranks = np.arange(len(ordered))

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, log in ((ax_lin, False), (ax_log, True)):
        colors = ["tab:red" if hot else "tab:blue" for hot in ordered["is_hot"]]
        ax.bar(ranks, ordered["dispatch_count"], color=colors, width=1.0)
        ax.set_xlabel("expert rank (all layers pooled)")
        ax.set_ylabel("dispatch count")
        if log:
            ax.set_yscale("log")
            ax.set_title("log scale")
        else:
            ax.set_title("linear scale")
        ax.margins(x=0.01)

    gini_value = result.overall["gini_overall"]
    entropy_value = result.overall["normalized_entropy_overall"]
    fig.suptitle(
        f"Expert activation distribution{title_suffix}\n"
        f"gini={gini_value:.3f}  normalised entropy={entropy_value:.3f}  "
        f"(red = hot, {result.overall['n_hot_total']}/{result.overall['n_experts_total']} experts "
        f"covering {result.overall['hot_dispatch_share']:.1%} of dispatches)"
    )
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_activation_heatmap(
    result: ClassificationResult, path: str | Path, title_suffix: str = ""
) -> Path:
    """Layer x expert heatmap of within-layer dispatch share.

    Share (not raw count) is plotted so layers are comparable even when they see
    different token counts.
    """
    table = result.table
    grid = table.pivot(index="layer_idx", columns="expert_id", values="layer_share")
    grid = grid.sort_index()

    fig, ax = plt.subplots(figsize=(max(6.0, 0.35 * grid.shape[1] + 3), max(4.0, 0.22 * grid.shape[0] + 2)))
    image = ax.imshow(grid.to_numpy(), aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("expert id")
    ax.set_ylabel("transformer layer index")
    ax.set_title(f"Within-layer dispatch share{title_suffix}")
    ax.set_yticks(range(len(grid.index)), [str(i) for i in grid.index], fontsize=7)
    fig.colorbar(image, ax=ax, label="fraction of that layer's dispatches")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_coverage_curve(
    counts: np.ndarray,
    experts_per_site: Sequence[int],
    path: str | Path,
    title_suffix: str = "",
) -> Path:
    """Cumulative dispatch coverage vs. number of resident experts per layer.

    One faint line per layer plus the mean, with reference lines at 80% / 90% /
    95% coverage. Reading off where the mean crosses 90% gives a defensible
    hot-set size for stage 3.
    """
    curves = [coverage_curve(counts[s, : experts_per_site[s]]) for s in range(counts.shape[0])]
    width = max(len(c) for c in curves)
    padded = np.array(
        [np.pad(c, (0, width - len(c)), mode="edge") if len(c) else np.zeros(width) for c in curves]
    )
    x = np.arange(1, width + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for curve in padded:
        ax.plot(x, curve, color="tab:blue", alpha=0.15, linewidth=1)
    ax.plot(x, padded.mean(axis=0), color="tab:red", linewidth=2.2, label="mean over layers")
    for level, style in ((0.80, ":"), (0.90, "--"), (0.95, "-.")):
        ax.axhline(level, color="grey", linestyle=style, linewidth=0.9)
        ax.text(width, level, f" {level:.0%}", va="center", fontsize=8, color="grey")
    ax.set_xlabel("number of most-used experts kept resident (per layer)")
    ax.set_ylabel("fraction of dispatches served")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(1, width)
    ax.set_title(f"Dispatch coverage vs. resident expert count{title_suffix}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_all(
    result: ClassificationResult,
    counts: np.ndarray,
    experts_per_site: Sequence[int],
    plot_dir: str | Path,
    title_suffix: str = "",
) -> list[Path]:
    """Render every sanity plot into ``plot_dir``; returns the written paths."""
    plot_dir = Path(plot_dir)
    return [
        plot_activation_histogram(result, plot_dir / "activation_histogram.png", title_suffix),
        plot_activation_heatmap(result, plot_dir / "activation_heatmap.png", title_suffix),
        plot_coverage_curve(counts, experts_per_site, plot_dir / "coverage_curve.png", title_suffix),
    ]


def format_summary_table(result: ClassificationResult, max_rows: int = 24) -> str:
    """Human-readable per-layer summary for the terminal.

    This is the printed artefact the working rules in CLAUDE.md ask for -- a way
    to eyeball stage-1 output without trusting the code blindly.
    """
    stats = result.per_layer_stats
    lines = [
        f"{'layer':>5} {'experts':>7} {'hot':>4} {'dispatches':>11} "
        f"{'hot_share':>9} {'gini':>6} {'entropy':>7} {'max_share':>9} {'unused':>6}",
        "-" * 76,
    ]
    shown = stats if len(stats) <= max_rows else pd.concat([stats.head(max_rows // 2), stats.tail(max_rows // 2)])
    prev_layer = None
    for _, row in shown.iterrows():
        if prev_layer is not None and row["layer_idx"] > prev_layer + 1:
            lines.append(f"{'...':>5}")
        prev_layer = row["layer_idx"]
        lines.append(
            f"{int(row['layer_idx']):>5} {int(row['n_experts']):>7} {int(row['n_hot']):>4} "
            f"{int(row['dispatches']):>11} {row['hot_dispatch_share']:>8.1%} "
            f"{row['gini']:>6.3f} {row['normalized_entropy']:>7.3f} "
            f"{row['max_expert_share']:>8.1%} {int(row['unused_experts']):>6}"
        )
    overall = result.overall
    lines += [
        "-" * 76,
        f"OVERALL  experts={overall['n_experts_total']}  hot={overall['n_hot_total']}  "
        f"dispatches={overall['total_dispatches']}  "
        f"hot_share={overall['hot_dispatch_share']:.1%}  "
        f"gini={overall['gini_overall']:.3f}  "
        f"entropy={overall['normalized_entropy_overall']:.3f}  "
        f"unused={overall['unused_experts_total']}",
        f"         hot weights={overall['hot_weight_bytes'] / 1e9:.2f} GB  "
        f"cold weights={overall['cold_weight_bytes'] / 1e9:.2f} GB",
    ]
    return "\n".join(lines)
