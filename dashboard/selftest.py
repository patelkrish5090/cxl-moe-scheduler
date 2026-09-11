"""Offline correctness checks for dashboard/data.py -- the pure data
functions dashboard/app.py's Streamlit UI renders. The UI itself isn't
unit-testable in this project's check()-based style (it needs a running
Streamlit server), but the data it's fed should be held to the same standard
as every other stage: no silently-wrong numbers reaching the page.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from .data import (
    CONFIG_ORDER,
    build_heatmap_grid,
    list_comparison_results,
    list_stage1_runs,
    load_comparison,
)

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        failures.append(name)
        print(f"  FAIL  {name}" + (f"\n        {detail}" if detail else ""))


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        print("\n[list_comparison_results / list_stage1_runs: missing directories]")
        check("an absent results dir gives an empty list, not an exception",
              list_comparison_results(root / "does_not_exist") == [])
        check("an absent data/runs dir gives an empty list, not an exception",
              list_stage1_runs(root / "does_not_exist") == [])

        print("\n[load_comparison]")
        results_dir = root / "results"
        results_dir.mkdir()
        payload = {
            "run_name": "test_run",
            "configs": {
                "hbm_only": {
                    "throughput_tokens_per_sec": 100.0, "avg_latency_ns_per_token": 10.0,
                    "total_energy_mj": 1.0, "hit_rate": 1.0, "n_tokens": 50, "n_dispatches": 400,
                },
                "hbm_cxl_naive": {
                    "throughput_tokens_per_sec": 10.0, "avg_latency_ns_per_token": 100.0,
                    "total_energy_mj": 5.0, "hit_rate": 0.3, "n_tokens": 50, "n_dispatches": 400,
                },
                "hbm_cxl_energy_aware": {
                    "throughput_tokens_per_sec": 12.0, "avg_latency_ns_per_token": 83.0,
                    "total_energy_mj": 4.5, "hit_rate": 0.35, "n_tokens": 50, "n_dispatches": 400,
                },
            },
        }
        result_path = results_dir / "test_run.json"
        result_path.write_text(json.dumps(payload), encoding="utf-8")

        check("list_comparison_results finds the written file",
              list_comparison_results(results_dir) == [result_path])

        df = load_comparison(result_path)
        check("load_comparison returns one row per config, in CONFIG_ORDER",
              list(df["config"]) == CONFIG_ORDER, f"got {list(df['config'])}")
        check("throughput values round-trip exactly",
              list(df["throughput_tokens_per_sec"]) == [100.0, 10.0, 12.0],
              f"got {list(df['throughput_tokens_per_sec'])}")
        check("every config has a non-empty human-readable label",
              all(isinstance(lbl, str) and lbl for lbl in df["label"]))

        missing = results_dir / "does_not_exist.json"
        raised = False
        try:
            load_comparison(missing)
        except FileNotFoundError as exc:
            raised = "experiments.cli run" in str(exc)
        check("a missing comparison file raises, naming the command that produces it", raised)

        malformed_path = results_dir / "malformed.json"
        malformed_path.write_text(json.dumps({"run_name": "x", "configs": {"hbm_only": payload["configs"]["hbm_only"]}}),
                                   encoding="utf-8")
        raised_key_error = False
        try:
            load_comparison(malformed_path)
        except KeyError:
            raised_key_error = True
        check("a comparison file missing a config raises, not silently drops a row",
              raised_key_error)

        print("\n[list_stage1_runs]")
        data_runs_dir = root / "data_runs"
        (data_runs_dir / "empty_placeholder").mkdir(parents=True)  # no hot_cold.csv -- must be excluded
        real_run_dir = data_runs_dir / "real_run"
        real_run_dir.mkdir()
        (real_run_dir / "hot_cold.csv").write_text("site_idx,layer_idx,expert_id,layer_share\n0,0,0,1.0\n",
                                                     encoding="utf-8")
        found_runs = list_stage1_runs(data_runs_dir)
        check("only the run directory with a real hot_cold.csv is listed",
              found_runs == [real_run_dir], f"got {found_runs}")

        print("\n[build_heatmap_grid]")
        hot_cold = pd.DataFrame({
            "layer_idx":   [0, 0, 1, 1],
            "expert_id":   [0, 1, 0, 1],
            "layer_share": [0.7, 0.3, 0.4, 0.6],
        })
        hot_cold_path = root / "hot_cold_grid.csv"
        hot_cold.to_csv(hot_cold_path, index=False)
        grid = build_heatmap_grid(hot_cold_path)
        check("heatmap grid is indexed by layer_idx, columned by expert_id",
              list(grid.index) == [0, 1] and list(grid.columns) == [0, 1])
        check("heatmap grid values match the source layer_share exactly",
              grid.loc[0, 0] == 0.7 and grid.loc[1, 1] == 0.6,
              f"got\n{grid}")

        missing_csv = root / "no_such_hot_cold.csv"
        raised_missing = False
        try:
            build_heatmap_grid(missing_csv)
        except FileNotFoundError:
            raised_missing = True
        check("a missing hot_cold.csv raises FileNotFoundError, not a confusing pandas error",
              raised_missing)

    print("\n" + "=" * 62)
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        print("=" * 62)
        return 1
    print("dashboard selftest passed")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
