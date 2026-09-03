"""Command-line entry points for stage 1.

    python -m profiler.cli run       configs/smoke_mixtral_tiny.json
    python -m profiler.cli inspect   data/runs/<name>
    python -m profiler.cli reclassify data/runs/<name> --method coverage --value 0.9
    python -m profiler.cli analyze  data/runs/<name>
    python -m profiler.cli selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .activation_log import counts_matrix_from_frame, load_run
from .classify import ClassifyConfig, classify
from .config import RunConfig
from . import plots


def _cmd_run(args: argparse.Namespace) -> int:
    from .runner import run

    cfg = RunConfig.from_json(args.config)
    if args.run_name:
        cfg.run_name = args.run_name
    if args.max_sequences is not None:
        cfg.data.max_sequences = args.max_sequences
    if args.device_map:
        cfg.model.device_map = args.device_map
    if args.no_trace:
        cfg.profiler.record_trace = False
    run(cfg, verbose=not args.quiet)
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    loaded = load_run(args.run_dir)
    metadata = loaded["metadata"]
    print(json.dumps(
        {
            "run_name": metadata["run_name"],
            "model": metadata["config"]["model"]["name_or_path"],
            "model_topology": metadata["model_topology"],
            "workload": metadata["workload"],
            "profiler_summary": metadata["profiler_summary"],
            "classification": metadata["classification"],
            "trace_rows": metadata["trace"]["rows"],
        },
        indent=2,
    ))
    if loaded["counts"] is None or loaded["hot_cold"] is None:
        print("\n(no CSV tables in this run directory)")
        return 1

    counts = counts_matrix_from_frame(loaded["counts"])
    experts_per_site = (
        loaded["counts"].groupby("site_idx")["expert_id"].max().add(1).sort_index().tolist()
    )
    cfg = ClassifyConfig(**{k: metadata["classification"][k] for k in ("method", "value", "per_layer")})
    layer_ids = loaded["counts"].groupby("site_idx")["layer_idx"].first().sort_index().tolist()
    result = classify(counts, cfg, layer_ids=layer_ids, experts_per_site=experts_per_site)
    print()
    print(plots.format_summary_table(result))
    return 0


def _cmd_reclassify(args: argparse.Namespace) -> int:
    """Re-run the hot/cold split at a new threshold without re-profiling."""
    run_dir = Path(args.run_dir)
    loaded = load_run(run_dir)
    if loaded["counts"] is None:
        print("no expert_counts.csv in that run directory", file=sys.stderr)
        return 1

    counts_frame = loaded["counts"]
    counts = counts_matrix_from_frame(counts_frame)
    experts_per_site = counts_frame.groupby("site_idx")["expert_id"].max().add(1).sort_index().tolist()
    layer_ids = counts_frame.groupby("site_idx")["layer_idx"].first().sort_index().tolist()
    weights = loaded["metadata"]["model_topology"]["expert_weight_bytes_per_layer"]

    cfg = ClassifyConfig(method=args.method, value=args.value, per_layer=not args.global_split)
    result = classify(
        counts, cfg, layer_ids=layer_ids, experts_per_site=experts_per_site,
        expert_weight_bytes=weights,
    )

    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    result.table.to_csv(out_dir / "hot_cold.csv", index=False)
    result.per_layer_stats.to_csv(out_dir / "layer_stats.csv", index=False)
    plots.plot_all(
        result, counts, experts_per_site, out_dir / "plots",
        title_suffix=f" - {loaded['metadata']['run_name']} ({cfg.method}={cfg.value})",
    )
    print(plots.format_summary_table(result))
    print(f"\nwrote hot/cold split to {out_dir.resolve()}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    from .analyze import report

    max_sites = args.max_sites if args.max_sites and args.max_sites > 0 else None
    report(args.run_dir, max_sites=max_sites, include_belady=not args.no_belady,
           phase=args.phase, layer_set=args.layers)
    return 0


def _cmd_selftest(args: argparse.Namespace) -> int:
    from .selftest import main as selftest_main

    return selftest_main(verbose=not args.quiet)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="profiler", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="profile a model and write a run directory")
    p_run.add_argument("config", help="path to a JSON run config (see configs/)")
    p_run.add_argument("--run-name", default=None, help="override run_name")
    p_run.add_argument("--max-sequences", type=int, default=None, help="override data.max_sequences")
    p_run.add_argument("--device-map", default=None, help="override model.device_map (e.g. cpu, auto)")
    p_run.add_argument("--no-trace", action="store_true", help="skip the per-token parquet trace")
    p_run.add_argument("--quiet", action="store_true")
    p_run.set_defaults(func=_cmd_run)

    p_inspect = sub.add_parser("inspect", help="print the summary of a finished run")
    p_inspect.add_argument("run_dir")
    p_inspect.set_defaults(func=_cmd_inspect)

    p_re = sub.add_parser("reclassify", help="re-split hot/cold at a new threshold")
    p_re.add_argument("run_dir")
    p_re.add_argument("--method", default="top_fraction", choices=["top_fraction", "coverage", "count"])
    p_re.add_argument("--value", type=float, default=0.2)
    p_re.add_argument("--global-split", action="store_true", help="apply the rule across all layers at once")
    p_re.add_argument("--out-dir", default=None, help="write elsewhere instead of overwriting")
    p_re.set_defaults(func=_cmd_reclassify)

    p_an = sub.add_parser("analyze", help="skew-vs-null, coverage and cache locality of a run")
    p_an.add_argument("run_dir")
    p_an.add_argument("--max-sites", type=int, default=4,
                      help="layers to simulate (default 4; 0 or negative means all)")
    p_an.add_argument("--no-belady", action="store_true", help="skip the optimal-policy bound")
    p_an.add_argument("--phase", default="all", choices=["all", "prefill", "decode"],
                      help="which tokens to simulate; decode is the serving regime")
    p_an.add_argument("--layers", default="both",
                      choices=["both", "all", "diverse", "trivial"],
                      help="which layer group gets a locality table; layers whose "
                           "working set fits in top_k experts are cache-trivial and "
                           "score ~100%% under every policy (default: both)")
    p_an.set_defaults(func=_cmd_analyze)

    p_self = sub.add_parser("selftest", help="run the offline plumbing test (no downloads)")
    p_self.add_argument("--quiet", action="store_true")
    p_self.set_defaults(func=_cmd_selftest)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
