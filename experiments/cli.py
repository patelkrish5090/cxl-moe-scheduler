"""Command-line entry point for stage 4's experiment harness.

    python -m experiments.cli selftest                          # offline, no data needed
    python -m experiments.cli run data/runs/<name>               # writes experiments/results/<name>.json
    python -m experiments.cli run data/runs/<name> --out <path>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .harness import run_comparison

DEFAULT_RESULTS_DIR = Path("experiments/results")


def _cmd_run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    out_path = Path(args.out) if args.out else DEFAULT_RESULTS_DIR / f"{run_dir.name}.json"

    comparison = run_comparison(run_dir, args.tier_model)

    print(f"\nTHREE-WAY EXPERIMENT -- {comparison.run_name}")
    print("=" * 78)
    print(f"{'config':<24}{'throughput_tok/s':>18}{'avg_lat_ms/tok':>18}{'energy_mJ':>16}")
    for cfg in (comparison.hbm_only, comparison.hbm_cxl_naive, comparison.hbm_cxl_energy_aware):
        print(f"{cfg.config:<24}{cfg.throughput_tokens_per_sec:>18.6g}"
              f"{cfg.avg_latency_ms_per_token:>18.6g}{cfg.total_energy_mj:>16.6g}")
        if not cfg.latency_accounting_consistent:
            print(f"  ERROR: latency accounting is INCONSISTENT for {cfg.config} -- the "
                  "independently-recomputed total latency disagrees with the reported total. "
                  "See scheduler.simulate.latency_breakdown; do not trust this config's latency "
                  "or throughput numbers until this is fixed.")
        if not cfg.latency_plausible:
            print(f"  WARNING: {cfg.latency_warning}")

    gap = comparison.energy_gap_pct
    verdict = "lower" if gap > 0 else "HIGHER"
    print(f"\ndocs.md 6 checkpoint 3: energy-aware total energy is {verdict} than naive's "
          f"by {abs(gap):.2f}%")
    if comparison.energy_gap_is_marginal:
        print(f"  WARNING: gap is below the {comparison.MARGINAL_ENERGY_GAP_PCT:.1f}% marginal "
              "threshold -- could be noise. Check the eviction divergence rate below before "
              "reporting this as a validated win.")

    print()
    print(comparison.eviction_divergence.summary())

    written = comparison.write(out_path)
    print(f"\nwrote {written}")
    return 0


def _cmd_selftest(_args: argparse.Namespace) -> int:
    from .selftest import main as selftest_main
    return selftest_main()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run all three configs over a stage-1 trace")
    p_run.add_argument("run_dir", help="stage-1 run directory, e.g. data/runs/mixtral_8x7b_decode")
    p_run.add_argument("--tier-model", default="memsim/tier_model.json")
    p_run.add_argument("--out", default=None, help=f"default: {DEFAULT_RESULTS_DIR}/<run_name>.json")
    p_run.set_defaults(func=_cmd_run)

    p_selftest = sub.add_parser("selftest", help="offline correctness checks, no data needed")
    p_selftest.set_defaults(func=_cmd_selftest)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
