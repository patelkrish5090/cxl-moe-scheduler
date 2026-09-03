"""Command-line entry points for stage 2 (memory system model).

    python -m memsim.cli provenance                    # where every constant came from
    python -m memsim.cli sweep --dry-run               # what gem5 would be asked to run
    python -m memsim.cli sweep --link-latency-ns 150   # run it
    python -m memsim.cli parse memsim/out/hbm_p1000    # one run directory
    python -m memsim.cli parse <dir> --dump            # every stat key, for fixing parsers
    python -m memsim.cli compare                       # checkpoint 2 + stage-3 hand-off
    python -m memsim.cli selftest                      # offline, needs no gem5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .run_sweep import DEFAULT_OUT, DEFAULT_PERIODS_PS, DEFAULT_THIRD_PARTY


def _cmd_provenance(args: argparse.Namespace) -> int:
    from . import constants

    print(constants.provenance_report())
    return 0 if not constants.unsourced() else 1


def _cmd_sweep(args: argparse.Namespace) -> int:
    from .run_sweep import run_sweep

    periods = tuple(args.periods_ps) if args.periods_ps else DEFAULT_PERIODS_PS
    tiers = tuple(args.tiers)
    if "cxl" in tiers and args.link_latency_ns is None and not args.dry_run:
        print("--link-latency-ns is required for the cxl tier.\n"
              "It is TODO_PLACEHOLDER in memsim/constants.py, so there is no default to\n"
              "fall back on. Supply an explicit value and label the results preliminary,\n"
              "or run --tiers hbm to characterise the HBM tier alone.")
        return 2
    run_sweep(
        link_latency_ns=args.link_latency_ns or 0.0,
        periods_ps=periods,
        third_party=Path(args.third_party),
        out_root=Path(args.out),
        transfer_bytes=args.transfer_bytes,
        dry_run=args.dry_run,
        tiers=tiers,
    )
    return 0


def _cmd_parse(args: argparse.Namespace) -> int:
    from .parse_stats import dump_keys, parse_run

    if args.dump:
        print(dump_keys(args.run_dir))
        return 0

    measurement = parse_run(args.run_dir)
    print(f"tier                     {measurement.tier}")
    print(f"bytes_read               {measurement.bytes_read}")
    print(f"sim_seconds              {measurement.sim_seconds}")
    print(f"avg_read_latency_ns      {measurement.avg_read_latency_ns}")
    print(f"bandwidth_gbps           {measurement.bandwidth_gbps}")
    print(f"device_energy_pj         {measurement.device_energy_pj}")
    print(f"device_energy_pj_per_bit {measurement.device_energy_pj_per_bit}")
    if measurement.stat_sources:
        print("\nstat keys used:")
        for quantity, key in measurement.stat_sources.items():
            print(f"  {quantity:<26} <- {key}")
    if measurement.energy_breakdown_pj:
        print("\nenergy breakdown (pJ):")
        for name, value in sorted(measurement.energy_breakdown_pj.items()):
            print(f"  {name:<40} {value:g}")
    if measurement.warnings:
        print("\nwarnings:")
        for warning in measurement.warnings:
            print(f"  - {warning}")
        print(f"\nRun with --dump to see every key in {args.run_dir}")
    return 0 if measurement.is_complete else 1


def _cmd_compare(args: argparse.Namespace) -> int:
    from .compare import report

    result = report(args.out, write_model=not args.no_write)
    return 0 if result.get("checkpoint2") is not False else 1


def _cmd_selftest(args: argparse.Namespace) -> int:
    from .selftest import main as selftest_main

    return selftest_main()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memsim", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_prov = sub.add_parser("provenance", help="print every constant and its source")
    p_prov.set_defaults(func=_cmd_provenance)

    p_sweep = sub.add_parser("sweep", help="run gem5 over the tier configurations")
    p_sweep.add_argument("--link-latency-ns", type=float, default=None,
                         help="one-way CXL link delay; required for the cxl tier")
    p_sweep.add_argument("--periods-ps", type=int, nargs="*", default=None,
                         help=f"injection periods to sweep (default {list(DEFAULT_PERIODS_PS)})")
    p_sweep.add_argument("--tiers", nargs="+", default=["hbm", "cxl"],
                         choices=["hbm", "cxl"])
    p_sweep.add_argument("--transfer-bytes", type=int, default=64 * 1024 * 1024,
                         help="bytes read per point (default 64 MiB)")
    p_sweep.add_argument("--out", default=str(DEFAULT_OUT))
    p_sweep.add_argument("--third-party", default=str(DEFAULT_THIRD_PARTY))
    p_sweep.add_argument("--dry-run", action="store_true",
                         help="print the gem5 commands without running them")
    p_sweep.set_defaults(func=_cmd_sweep)

    p_parse = sub.add_parser("parse", help="read one gem5 output directory")
    p_parse.add_argument("run_dir")
    p_parse.add_argument("--dump", action="store_true",
                         help="print every stat key found, for correcting the parsers")
    p_parse.set_defaults(func=_cmd_parse)

    p_cmp = sub.add_parser("compare", help="checkpoint 2 and the stage-3 hand-off")
    p_cmp.add_argument("--out", default=str(DEFAULT_OUT))
    p_cmp.add_argument("--no-write", action="store_true",
                       help="do not write memsim/tier_model.json")
    p_cmp.set_defaults(func=_cmd_compare)

    p_self = sub.add_parser("selftest", help="offline checks; needs no gem5 build")
    p_self.set_defaults(func=_cmd_selftest)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        # A missing gem5 build or DRAMSim3 checkout is an expected state with a
        # known fix, not a bug. Print the guidance, not a traceback.
        print(f"\n{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
