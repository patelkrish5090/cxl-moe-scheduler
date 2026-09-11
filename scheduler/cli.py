"""Command-line entry points for stage 3 (energy-aware scheduler).

    python -m scheduler.cli selftest                         # offline, no data needed
    python -m scheduler.cli run data/runs/<name> --policy naive
    python -m scheduler.cli run data/runs/<name> --policy energy-aware-defer --power-budget-w 50
    python -m scheduler.cli run data/runs/<name> --policy energy-aware-evict
    python -m scheduler.cli compare3 data/runs/<name> --power-budget-w 50   # docs.md 6 checkpoint 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .model import (
    CostModel,
    load_expert_dispatch_counts,
    load_expert_weight_bytes,
    load_hot_experts,
    load_tier_model,
    load_trace,
)
from .simulate import (
    diff_decisions,
    eviction_divergence_report,
    run_energy_aware_defer,
    run_energy_aware_evict,
    run_naive,
    three_way_report,
)


def _load_inputs(run_dir: Path, tier_model_path: Path):
    trace = load_trace(run_dir / "trace.parquet")
    weight_bytes = load_expert_weight_bytes(run_dir / "hot_cold.csv")
    hot = load_hot_experts(run_dir / "hot_cold.csv")
    dispatch_counts = load_expert_dispatch_counts(run_dir / "hot_cold.csv")
    tiers = load_tier_model(tier_model_path)
    cost_model = CostModel(weight_bytes, tiers)
    cache_capacity = {site: len(experts) for site, experts in hot.items()}
    return trace, cost_model, cache_capacity, dispatch_counts


def _print_summary(label: str, summary: dict) -> None:
    print(f"\n[{label}]")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key:<20} {value:.4g}")
        else:
            print(f"  {key:<20} {value}")


def _cmd_run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    trace, cost_model, cache_capacity, dispatch_counts = _load_inputs(run_dir, Path(args.tier_model))

    if args.policy == "naive":
        result = run_naive(trace, cost_model, cache_capacity)
    elif args.policy == "energy-aware-defer":
        if args.power_budget_w is None:
            print("--power-budget-w is required for --policy energy-aware-defer", file=sys.stderr)
            return 2
        result = run_energy_aware_defer(trace, cost_model, cache_capacity, args.power_budget_w)
    else:  # energy-aware-evict
        result = run_energy_aware_evict(trace, cost_model, cache_capacity, dispatch_counts)

    _print_summary(f"{args.policy} policy -- {run_dir.name}", result.summary())
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    """naive vs energy-aware-defer only -- the timing/latency comparison.
    For the checkpoint-3 total-energy comparison, use `compare3`.
    """
    if args.power_budget_w is None:
        print("--power-budget-w is required", file=sys.stderr)
        return 2
    run_dir = Path(args.run_dir)
    trace, cost_model, cache_capacity, _dispatch_counts = _load_inputs(run_dir, Path(args.tier_model))

    naive = run_naive(trace, cost_model, cache_capacity)
    defer = run_energy_aware_defer(trace, cost_model, cache_capacity, args.power_budget_w)

    _print_summary("naive", naive.summary())
    _print_summary("energy-aware-defer", defer.summary())
    print()
    print(diff_decisions(naive, defer))
    return 0


def _cmd_compare3(args: argparse.Namespace) -> int:
    """naive vs energy-aware-defer vs energy-aware-evict, with the
    docs.md 6 checkpoint 3 total-energy verdict.
    """
    if args.power_budget_w is None:
        print("--power-budget-w is required", file=sys.stderr)
        return 2
    run_dir = Path(args.run_dir)
    trace, cost_model, cache_capacity, dispatch_counts = _load_inputs(run_dir, Path(args.tier_model))

    naive = run_naive(trace, cost_model, cache_capacity)
    defer = run_energy_aware_defer(trace, cost_model, cache_capacity, args.power_budget_w)
    evict = run_energy_aware_evict(trace, cost_model, cache_capacity, dispatch_counts)

    _print_summary("naive", naive.summary())
    _print_summary("energy-aware-defer", defer.summary())
    _print_summary("energy-aware-evict", evict.summary())
    print()
    print(three_way_report(naive, defer, evict))
    print()
    print(diff_decisions(naive, evict))
    print()
    print(eviction_divergence_report(naive, evict, trace, dispatch_counts).summary())
    return 0


def _cmd_selftest(_args: argparse.Namespace) -> int:
    from .selftest import main as selftest_main
    return selftest_main()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run one scheduling policy over a stage-1 trace")
    p_run.add_argument("run_dir", help="stage-1 run directory, e.g. data/runs/mixtral_8x7b_decode")
    p_run.add_argument("--policy", choices=["naive", "energy-aware-defer", "energy-aware-evict"], required=True)
    p_run.add_argument("--power-budget-w", type=float, default=None, help="watts; required for energy-aware-defer")
    p_run.add_argument("--tier-model", default="memsim/tier_model.json")
    p_run.set_defaults(func=_cmd_run)

    p_compare = sub.add_parser("compare", help="naive vs energy-aware-defer: the timing/latency diff")
    p_compare.add_argument("run_dir")
    p_compare.add_argument("--power-budget-w", type=float, default=None, required=True)
    p_compare.add_argument("--tier-model", default="memsim/tier_model.json")
    p_compare.set_defaults(func=_cmd_compare)

    p_compare3 = sub.add_parser("compare3", help="all three policies + docs.md 6 checkpoint 3 verdict")
    p_compare3.add_argument("run_dir")
    p_compare3.add_argument("--power-budget-w", type=float, default=None, required=True,
                             help="only used for the energy-aware-defer leg")
    p_compare3.add_argument("--tier-model", default="memsim/tier_model.json")
    p_compare3.set_defaults(func=_cmd_compare3)

    p_selftest = sub.add_parser("selftest", help="offline correctness checks, no data needed")
    p_selftest.set_defaults(func=_cmd_selftest)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
