"""gem5 configuration for one memory tier. Runs INSIDE gem5, not under python3.

    third_party/gem5/build/X86/gem5.opt \
        --outdir=memsim/out/hbm_p1000 \
        memsim/gem5_configs/tier.py \
        --tier hbm --dramsim-config <path to a DRAMSim3 .ini> \
        --injection-period-ps 1000

Do not import this module from ordinary Python: ``m5`` only exists inside the
gem5 binary. :mod:`memsim.run_sweep` is the thing that invokes it.

WHY CHARACTERISE, NOT REPLAY
----------------------------
The obvious design -- push the whole activation trace through gem5 and read off
the total energy -- does not work. gem5 in timing mode simulates on the order of
10^5 instructions per second, and one Mixtral expert is 352 MB, which is ~5.5
million 64-byte reads for a *single* fetch. The decode trace has tens of
thousands of dispatches. That is many years of wall clock.

So gem5's job here is to *characterise the tiers*, not to run the workload. Each
invocation measures, for one tier under one load level:

    - achieved read bandwidth (GB/s)
    - average read latency (ns)
    - DRAM device energy (pJ), from DRAMSim3's own device model

Stage 3 then multiplies those per-byte figures by the real fetch counts and
expert sizes taken from the stage-1 trace. That keeps the expensive simulation
proportional to the number of *configurations* (a handful) rather than to the
number of *accesses* (billions), and it is the standard way this kind of study
is structured.

The access pattern is a saturating sequential read, because that is what an
expert-weight fetch is: one large contiguous block moved in one direction. It is
deliberately not a random-access pattern -- modelling expert fetches as random
accesses would understate both tiers' achievable bandwidth.

TIERS
-----
hbm  Memory attached directly to the memory bus. The GPU-HBM baseline.
cxl  The same DRAMSim3 device model behind a ``Bridge`` whose delay stands in
     for the CXL link round trip. No physical CXL hardware is involved and none
     is implied (CLAUDE.md); the link latency is a modelled parameter supplied
     on the command line, and the link's *energy* is not modelled by gem5 at all
     -- see memsim/constants.py.
"""

import argparse
import sys

import m5
from m5.objects import (
    AddrRange,
    Bridge,
    IOXBar,
    PyTrafficGen,
    Root,
    SrcClockDomain,
    System,
    VoltageDomain,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=["hbm", "cxl"], required=True,
                        help="hbm = direct attach; cxl = behind a link delay")
    parser.add_argument("--dramsim-config", required=True,
                        help="path to a DRAMSim3 device .ini")
    parser.add_argument("--dramsim-path", default="",
                        help="directory DRAMSim3 prepends to relative file names")
    parser.add_argument("--link-latency-ns", type=float, default=0.0,
                        help="one-way link delay for the cxl tier; 0 for hbm")
    parser.add_argument("--mem-size", default="4GiB",
                        help="simulated memory range (must exceed --transfer-bytes)")
    parser.add_argument("--transfer-bytes", type=int, default=64 * 1024 * 1024,
                        help="bytes to read; stands in for one expert-weight fetch. "
                             "Default 64 MiB -- large enough to reach steady state, "
                             "small enough to simulate in minutes. Per-byte results "
                             "are what get used, so this need not equal a real "
                             "expert size.")
    parser.add_argument("--block-bytes", type=int, default=64,
                        help="request size, i.e. one cache line")
    parser.add_argument("--injection-period-ps", type=int, default=1000,
                        help="ticks between issued requests. Small = saturating "
                             "(measures bandwidth); large = unloaded (measures "
                             "latency). Sweep this to get both.")
    parser.add_argument("--clock", default="2GHz")
    return parser


def attach_dramsim(config_file: str, file_path: str, addr_range):
    """Instantiate the DRAMsim3 SimObject, tolerating parameter-name drift.

    gem5 has spelled these parameters differently across releases, and a wrong
    name is a silent misconfiguration rather than an error, so set whichever the
    installed version actually declares and report the choice.
    """
    try:
        from m5.objects import DRAMsim3
    except ImportError:
        sys.exit(
            "This gem5 was built WITHOUT DRAMSim3 support: the DRAMsim3 SimObject "
            "does not exist.\nRebuild with:  bash scripts/build_gem5.sh dramsim3 "
            "&& bash scripts/build_gem5.sh gem5"
        )

    ctrl = DRAMsim3()
    declared = set(type(ctrl)._params.keys())

    for name in ("config_file", "configFile"):
        if name in declared:
            setattr(ctrl, name, config_file)
            print(f"[tier.py] DRAMsim3 config parameter is '{name}'")
            break
    else:
        sys.exit(f"DRAMsim3 declares no recognised config parameter; has: {sorted(declared)}")

    if file_path:
        for name in ("filePath", "file_path"):
            if name in declared:
                setattr(ctrl, name, file_path)
                break

    ctrl.range = addr_range
    return ctrl


def main() -> None:
    args = build_parser().parse_args()

    if args.tier == "hbm" and args.link_latency_ns:
        sys.exit("--link-latency-ns is meaningless for the hbm tier; it is direct-attach")
    if args.tier == "cxl" and not args.link_latency_ns:
        # Refuse rather than silently produce a "CXL" result identical to HBM,
        # which would quietly break validation checkpoint 2 in docs.md 6.
        sys.exit(
            "--link-latency-ns is required and must be non-zero for the cxl tier.\n"
            "It is still TODO_PLACEHOLDER in memsim/constants.py; supply an explicit "
            "value to run an exploratory point, and label the result accordingly."
        )

    system = System()
    system.clk_domain = SrcClockDomain(clock=args.clock, voltage_domain=VoltageDomain())
    system.mem_mode = "timing"
    system.mem_ranges = [AddrRange(args.mem_size)]

    system.generator = PyTrafficGen()
    system.membus = IOXBar()
    system.generator.port = system.membus.cpu_side_ports

    system.mem_ctrl = attach_dramsim(
        args.dramsim_config, args.dramsim_path, system.mem_ranges[0]
    )

    if args.tier == "cxl":
        # A Bridge with a fixed delay is the simplest honest stand-in for the
        # link: it charges every request and every response the configured
        # one-way latency, so a read pays it twice, which is the round trip.
        system.link = Bridge(
            delay=f"{args.link_latency_ns}ns",
            ranges=system.mem_ranges,
        )
        system.membus.mem_side_ports = system.link.cpu_side_port
        system.linkbus = IOXBar()
        system.link.mem_side_port = system.linkbus.cpu_side_ports
        system.linkbus.mem_side_ports = system.mem_ctrl.port
        print(f"[tier.py] cxl tier: {args.link_latency_ns} ns one-way link delay "
              f"({2 * args.link_latency_ns} ns round trip)")
    else:
        system.membus.mem_side_ports = system.mem_ctrl.port
        print("[tier.py] hbm tier: direct attach, no link delay")

    end_addr = min(int(AddrRange(args.mem_size).size()), args.transfer_bytes)
    print(f"[tier.py] reading {args.transfer_bytes:,} B in {args.block_bytes} B blocks, "
          f"period {args.injection_period_ps} ps")

    def traffic():
        # 100% reads: an expert fetch is a pure read of weights into the GPU.
        yield system.generator.createLinear(
            10_000_000_000_000,          # duration cap in ticks; data_limit ends it first
            0,                            # start address
            end_addr,                     # end address
            args.block_bytes,             # request size
            args.injection_period_ps,     # min period between requests
            args.injection_period_ps,     # max period (equal => fixed rate)
            100,                          # read percentage
            args.transfer_bytes,          # data limit: stop after this many bytes
        )
        yield system.generator.createExit(0)

    root = Root(full_system=False, system=system)
    m5.instantiate()
    system.generator.start(traffic())

    event = m5.simulate()
    print(f"[tier.py] finished at tick {m5.curTick()} because: {event.getCause()}")
    # m5 dumps stats.txt into --outdir; memsim.parse_stats reads it from there,
    # alongside whatever DRAMSim3 wrote for its own device statistics.
    m5.stats.dump()


if __name__ == "__m5_main__":
    main()
