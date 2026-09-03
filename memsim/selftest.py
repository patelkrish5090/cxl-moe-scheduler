"""Offline checks for stage 2. Runs without gem5, without DRAMSim3, without a GPU.

    python -m memsim.cli selftest

gem5 takes an hour to build and minutes per simulated point, so the parsing,
unit-conversion and checkpoint logic must be verifiable without it. Every case
here uses synthetic fixture data with a hand-computable answer, because a stats
parser that silently returns the wrong key produces a plausible number rather
than an error -- exactly the failure this project cannot afford.

What this does NOT check: that the gem5 stat key names in
``parse_stats.CANDIDATES`` match the gem5 build you actually have. Only a real
run can settle that. Use ``python -m memsim.cli parse <dir> --dump`` on the first
real run and correct the candidate lists against what it prints.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

from . import constants
from .compare import _period_of, build_tier_model
from .parse_stats import (
    TICKS_PER_NS,
    _lookup,
    parse_run,
    read_dramsim_energy,
    read_gem5_stats,
)
from .run_sweep import DEVICE_PREFERENCE, build_points, command_for, pick_device_config

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        failures.append(name)
        print(f"  FAIL  {name}" + (f"\n        {detail}" if detail else ""))


def write_run(
    root: Path,
    name: str,
    sim_seconds: float = 1e-4,
    bytes_read: int = 67_108_864,
    total_reads: int = 1_048_576,
    avg_latency_ticks: float = 80_000.0,
    total_energy_pj: float | None = 1e9,
) -> Path:
    """Write a synthetic gem5 output directory."""
    outdir = root / name
    outdir.mkdir(parents=True, exist_ok=True)
    lines = [
        "---------- Begin Simulation Statistics ----------",
        f"simSeconds                        {sim_seconds:.9f}   # Number of seconds simulated (Second)",
        f"system.generator.bytesRead        {bytes_read}        # Bytes read (Byte)",
        f"system.generator.totalReads       {total_reads}       # Reads (Count)",
        f"system.generator.avgReadLatency   {avg_latency_ticks} # Avg read latency (Tick)",
        "system.some.distribution          |  3 33.3%  |  6 66.7%   # a non-numeric row",
        "---------- End Simulation Statistics   ----------",
    ]
    (outdir / "stats.txt").write_text("\n".join(lines), encoding="utf-8")
    if total_energy_pj is not None:
        (outdir / "dramsim3.json").write_text(
            json.dumps({"0": {"total_energy": total_energy_pj, "read_energy": 1.0}}),
            encoding="utf-8",
        )
    return outdir


def main() -> int:
    print("\n[constants: placeholders must poison, not default to zero]")
    check("PLACEHOLDER is NaN", math.isnan(constants.PLACEHOLDER))
    check("an unsourced constant reports is_sourced False",
          not constants.CXL_LINK_ENERGY_PJ_PER_BIT.is_sourced)
    check("unsourced() lists the CXL link constants",
          {c.name for c in constants.unsourced()}
          >= {"CXL_LINK_LATENCY_NS", "CXL_LINK_ENERGY_PJ_PER_BIT"},
          f"got {[c.name for c in constants.unsourced()]}")
    check("simulated constants are not counted as unsourced",
          "HBM_DEVICE_ENERGY" not in {c.name for c in constants.unsourced()})

    raised = False
    try:
        constants.require_sourced("a test result")
    except RuntimeError as exc:
        raised = "CXL_LINK_ENERGY_PJ_PER_BIT" in str(exc)
    check("require_sourced raises and names the missing constant", raised)

    poisoned = constants.energy_pj(float(constants.CXL_LINK_ENERGY_PJ_PER_BIT), 1024)
    check("energy computed from a placeholder is NaN, never 0",
          math.isnan(poisoned), f"got {poisoned}")
    check("provenance report marks unsourced entries",
          "TODO_PLACEHOLDER" in constants.provenance_report())

    print("\n[unit conversions]")
    check("bytes -> bits is x8", constants.bits_from_bytes(10) == 80.0)
    check("2 pJ/bit over 1 byte is 16 pJ", constants.energy_pj(2.0, 1) == 16.0)
    check("pJ -> J is 1e-12", constants.pj_to_joules(1e12) == 1.0)
    check("pJ -> mJ is 1e-9", constants.pj_to_millijoules(1e9) == 1.0)
    check("one tick is one picosecond", TICKS_PER_NS == 1000.0)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        print("\n[gem5 stats.txt parsing]")
        outdir = write_run(root, "hbm_p1000")
        stats = read_gem5_stats(outdir / "stats.txt")
        check("numeric keys are parsed", stats.get("simSeconds") == 1e-4,
              f"got {stats.get('simSeconds')}")
        check("non-numeric distribution rows are skipped",
              "system.some.distribution" not in stats)
        check("header and footer rules are skipped", "----------" not in "".join(stats))
        check("missing stats.txt yields an empty dict, not an exception",
              read_gem5_stats(root / "nope" / "stats.txt") == {})

        value, key = _lookup(stats, "bytes_read")
        check("_lookup finds the first matching candidate",
              key == "system.generator.bytesRead" and value == 67_108_864,
              f"got {key}={value}")
        check("_lookup returns (None, None) when nothing matches",
              _lookup({}, "bytes_read") == (None, None))

        print("\n[DRAMSim3 energy parsing]")
        total, breakdown, warnings = read_dramsim_energy(outdir / "dramsim3.json")
        check("total_energy is preferred over summing components", total == 1e9,
              f"got {total} (would be 1000000001.0 if components were summed too)")
        check("the breakdown keeps every energy field", len(breakdown) == 2,
              f"got {sorted(breakdown)}")

        nested = root / "nested.json"
        nested.write_text(json.dumps(
            {"channels": [{"act_energy": 10.0, "read_energy": 5.0},
                           {"act_energy": 2.0, "read_energy": 3.0}]}), encoding="utf-8")
        total_nested, fields, warn_nested = read_dramsim_energy(nested)
        check("component fields are summed when no total is present",
              total_nested == 20.0, f"got {total_nested}")
        check("summing components is flagged in the warnings",
              any("summed" in w for w in warn_nested), f"got {warn_nested}")
        check("nested channels are all walked", len(fields) == 4, f"got {sorted(fields)}")

        missing_total, _, warn_missing = read_dramsim_energy(root / "absent.json")
        check("a missing DRAMSim3 file gives None, not 0", missing_total is None)
        check("and says so", bool(warn_missing))

        malformed = root / "bad.json"
        malformed.write_text("{not json", encoding="utf-8")
        bad_total, _, warn_bad = read_dramsim_energy(malformed)
        check("malformed JSON gives None and a warning, not a crash",
              bad_total is None and any("malformed" in w for w in warn_bad))

        csv_path = root / "dramsim3.csv"
        csv_path.write_text("read_energy,12.5\nact_energy,7.5\n", encoding="utf-8")
        csv_total, csv_fields, _ = read_dramsim_energy(csv_path)
        check("csv energy fields are parsed and summed", csv_total == 20.0,
              f"got {csv_total} from {csv_fields}")

        print("\n[end-to-end run parsing]")
        measurement = parse_run(outdir)
        # 80000 ticks / 1000 = 80 ns.
        check("latency converts ticks -> ns", measurement.avg_read_latency_ns == 80.0,
              f"got {measurement.avg_read_latency_ns}")
        # 67108864 B / 1e-4 s / 1e9 = 671.08864 GB/s.
        check("bandwidth is bytes / seconds / 1e9",
              abs(measurement.bandwidth_gbps - 671.08864) < 1e-6,
              f"got {measurement.bandwidth_gbps}")
        # 1e9 pJ / (67108864 * 8 bits) = 1.86264514923...
        check("energy per bit is total pJ / bits read",
              abs(measurement.device_energy_pj_per_bit - 1e9 / (67_108_864 * 8)) < 1e-12,
              f"got {measurement.device_energy_pj_per_bit}")
        check("a complete run reports is_complete", measurement.is_complete)
        check("tier is inferred from the directory name", measurement.tier == "hbm")
        check("cxl is inferred too", parse_run(write_run(root, "cxl_p1000")).tier == "cxl")
        check("the stat key used for each quantity is recorded",
              measurement.stat_sources.get("bytes_read") == "system.generator.bytesRead",
              f"got {measurement.stat_sources}")

        print("\n[missing data must not become zero]")
        no_energy = write_run(root, "hbm_p9999", total_energy_pj=None)
        bare = parse_run(no_energy)
        check("absent DRAMSim3 output leaves energy None, not 0",
              bare.device_energy_pj is None and bare.device_energy_pj_per_bit is None,
              f"got {bare.device_energy_pj} / {bare.device_energy_pj_per_bit}")
        check("an incomplete run says so", not bare.is_complete)
        check("and explains why", bool(bare.warnings), f"got {bare.warnings}")

        empty = root / "empty_run"
        empty.mkdir()
        (empty / "stats.txt").write_text("", encoding="utf-8")
        blank = parse_run(empty)
        check("an empty stats.txt warns about an unfinished run",
              any("stats.txt" in w for w in blank.warnings), f"got {blank.warnings}")

        print("\n[tier model reduction]")
        # Slow injection (large period) = idle = unloaded latency.
        # Fast injection (small period) = saturated = peak bandwidth.
        sweep_root = root / "sweep"
        write_run(sweep_root, "hbm_p1", sim_seconds=1e-4, avg_latency_ticks=200_000)
        write_run(sweep_root, "hbm_p100000", sim_seconds=1e-2, avg_latency_ticks=80_000)
        runs = [parse_run(p) for p in sorted(sweep_root.iterdir())]
        check("_period_of recovers the injection period from the name",
              sorted(_period_of(r) for r in runs) == [1, 100_000],
              f"got {[(_period_of(r), r.outdir.name) for r in runs]}")

        model = build_tier_model(runs, "hbm")
        check("unloaded latency comes from the SLOWEST injection rate",
              model.unloaded_latency_ns == 80.0, f"got {model.unloaded_latency_ns}")
        check("peak bandwidth comes from the FASTEST injection rate",
              abs(model.peak_bandwidth_gbps - 671.08864) < 1e-6,
              f"got {model.peak_bandwidth_gbps}")
        check("hbm has no link energy", model.link_energy_pj_per_bit == 0.0)
        check("hbm total energy equals its device energy",
              model.total_energy_pj_per_bit == model.device_energy_pj_per_bit)
        check("a tier with no runs reduces to None",
              build_tier_model(runs, "cxl") is None)

        cxl_root = root / "cxlsweep"
        write_run(cxl_root, "cxl_p1")
        write_run(cxl_root, "cxl_p100000", sim_seconds=1e-2)
        cxl_runs = [parse_run(p) for p in sorted(cxl_root.iterdir())]
        cxl_model = build_tier_model(cxl_runs, "cxl")
        check("cxl link energy is the unsourced constant, so NaN",
              math.isnan(cxl_model.link_energy_pj_per_bit))
        check("cxl total energy is therefore NaN, not the device energy alone",
              math.isnan(cxl_model.total_energy_pj_per_bit),
              f"got {cxl_model.total_energy_pj_per_bit}")
        check("but cxl device energy is still a real number",
              cxl_model.device_energy_pj_per_bit is not None
              and not math.isnan(cxl_model.device_energy_pj_per_bit))

        print("\n[sweep planning]")
        configs = root / "configs"
        configs.mkdir()
        for name in ("HBM2_8Gb_x128.ini", "DDR4_8Gb_x8_3200.ini", "GDDR6_8Gb_x16.ini"):
            (configs / name).write_text("[dram_structure]\n", encoding="utf-8")
        check("hbm tier prefers an HBM device config",
              pick_device_config(configs, "hbm").name == "HBM2_8Gb_x128.ini")
        check("cxl tier falls back to DDR4 when DDR5 is absent",
              pick_device_config(configs, "cxl").name == "DDR4_8Gb_x8_3200.ini")

        (configs / "DDR5_16Gb_x8_4800.ini").write_text("[dram_structure]\n", encoding="utf-8")
        check("cxl tier prefers DDR5 once it exists",
              pick_device_config(configs, "cxl").name == "DDR5_16Gb_x8_4800.ini")

        only_gddr = root / "only_gddr"
        only_gddr.mkdir()
        (only_gddr / "GDDR6_8Gb_x16.ini").write_text("", encoding="utf-8")
        listed = False
        try:
            pick_device_config(only_gddr, "hbm")
        except FileNotFoundError as exc:
            listed = "GDDR6_8Gb_x16.ini" in str(exc)
        check("an unmatched tier lists what IS available, so the list can be fixed",
              listed)
        check("every tier has a preference list", set(DEVICE_PREFERENCE) == {"hbm", "cxl"})

    print("\n[gem5 command construction]")
    fake_third_party = Path(tempfile.mkdtemp())
    (fake_third_party / "gem5" / "ext" / "dramsim3" / "DRAMsim3" / "configs").mkdir(parents=True)
    cfg_dir = fake_third_party / "gem5" / "ext" / "dramsim3" / "DRAMsim3" / "configs"
    (cfg_dir / "HBM2_8Gb_x128.ini").write_text("", encoding="utf-8")
    (cfg_dir / "DDR5_16Gb_x8_4800.ini").write_text("", encoding="utf-8")

    points = build_points(link_latency_ns=150.0, periods_ps=(1, 1000),
                          third_party=fake_third_party, out_root=Path("/tmp/out"))
    check("the sweep covers both tiers at every period", len(points) == 4,
          f"got {[p.label for p in points]}")
    check("hbm points carry no link latency",
          all(p.link_latency_ns == 0.0 for p in points if p.tier == "hbm"))
    check("cxl points carry the requested link latency",
          all(p.link_latency_ns == 150.0 for p in points if p.tier == "cxl"))
    check("the two tiers use different device configs",
          len({p.device_config.name for p in points}) == 2,
          f"got {sorted({p.device_config.name for p in points})}")

    hbm_cmd = command_for(points[0], Path("/fake/gem5.opt"), 1024)
    cxl_point = next(p for p in points if p.tier == "cxl")
    cxl_cmd = command_for(cxl_point, Path("/fake/gem5.opt"), 1024)
    check("--link-latency-ns is passed only for the cxl tier",
          "--link-latency-ns" not in hbm_cmd and "--link-latency-ns" in cxl_cmd)
    check("the outdir is passed to gem5, not to the config script",
          any(a.startswith("--outdir=") for a in hbm_cmd)
          and hbm_cmd.index(next(a for a in hbm_cmd if a.startswith("--outdir=")))
          < hbm_cmd.index(str(points[0].device_config)),
          f"got {hbm_cmd}")
    check("transfer size is passed through", "--transfer-bytes" in hbm_cmd)

    print("\n" + "=" * 62)
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        return 1
    print("memsim selftest passed (gem5 stat key names still need a real run to confirm)")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
