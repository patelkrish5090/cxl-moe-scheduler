"""Drive gem5 over the tier configurations that stage 2 needs.

    python -m memsim.cli sweep --link-latency-ns 150
    python -m memsim.cli sweep --dry-run          # print the commands, run nothing

One gem5 invocation measures one (tier, injection rate) point. The sweep runs
the HBM tier and the CXL tier over the same injection rates so the comparison in
:mod:`memsim.compare` holds the access pattern fixed and varies only the tier --
which is what validation checkpoint 2 in docs.md 6 actually asks for.

Injection rate is swept rather than fixed because one rate cannot answer both
questions we need. A slow rate leaves the memory system idle and measures
*unloaded latency*; a fast rate saturates it and measures *peak bandwidth*. A
tiering decision needs both: latency sets the stall a fetch imposes, bandwidth
sets how many fetches can be in flight before they interfere.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THIRD_PARTY = REPO_ROOT / "third_party"
TIER_CONFIG = REPO_ROOT / "memsim" / "gem5_configs" / "tier.py"
DEFAULT_OUT = REPO_ROOT / "memsim" / "out"

#: Injection periods in picoseconds, fastest first. 1 ps issues a request every
#: tick (saturating); 100000 ps is one request per 100 ns, far below what either
#: tier can serve, so the memory system is idle and the measured latency is the
#: unloaded one.
DEFAULT_PERIODS_PS = (1, 100, 1_000, 10_000, 100_000)

#: DRAMSim3 device configs to prefer per tier, most preferred first. Matched as
#: case-insensitive substrings against the .ini file names DRAMSim3 ships, because
#: the exact names differ across DRAMSim3 revisions.
DEVICE_PREFERENCE: dict[str, tuple[str, ...]] = {
    # The GPU tier. HBM2 is the closest device DRAMSim3 ships to the HBM3e on a
    # Blackwell RTX 6000; the substitution is a modelling limitation and must be
    # stated wherever these numbers appear.
    "hbm": ("HBM2", "HBM"),
    # The CXL tier is commodity DRAM behind the link, so DDR5 if available and
    # DDR4 otherwise.
    "cxl": ("DDR5", "DDR4", "DDR3"),
}


@dataclass
class SweepPoint:
    """One gem5 invocation."""

    tier: str
    injection_period_ps: int
    outdir: Path
    device_config: Path
    link_latency_ns: float

    @property
    def label(self) -> str:
        return f"{self.tier}_p{self.injection_period_ps}"


def find_gem5(third_party: Path, arch: str = "X86") -> Path:
    """Locate the gem5 binary built by ``scripts/build_gem5.sh``."""
    candidates = [
        third_party / "gem5" / "build" / arch / "gem5.opt",
        third_party / "gem5" / "build" / "ALL" / "gem5.opt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    found = shutil.which("gem5.opt")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "no gem5.opt found. Looked in:\n"
        + "\n".join(f"  {c}" for c in candidates)
        + "\nBuild it with:  bash scripts/build_gem5.sh"
    )


def find_dramsim_configs(third_party: Path) -> Path:
    """Locate DRAMSim3's shipped device-config directory."""
    candidate = third_party / "gem5" / "ext" / "dramsim3" / "DRAMsim3" / "configs"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f"no DRAMSim3 configs directory at {candidate}.\n"
        "Fetch it with:  bash scripts/build_gem5.sh fetch"
    )


def pick_device_config(configs_dir: Path, tier: str) -> Path:
    """Choose a DRAMSim3 device .ini for a tier, by preference with fallback.

    Raises:
        FileNotFoundError: listing what DRAMSim3 does ship, so the preference
            list can be corrected rather than guessed at again.
    """
    available = sorted(configs_dir.glob("*.ini"))
    if not available:
        raise FileNotFoundError(f"no .ini device configs in {configs_dir}")
    for wanted in DEVICE_PREFERENCE[tier]:
        for path in available:
            if wanted.lower() in path.name.lower():
                return path
    raise FileNotFoundError(
        f"no DRAMSim3 config for tier {tier!r} matching any of "
        f"{DEVICE_PREFERENCE[tier]}.\nAvailable:\n"
        + "\n".join(f"  {p.name}" for p in available)
    )


def build_points(
    link_latency_ns: float,
    periods_ps: tuple[int, ...] = DEFAULT_PERIODS_PS,
    third_party: Path = DEFAULT_THIRD_PARTY,
    out_root: Path = DEFAULT_OUT,
    tiers: tuple[str, ...] = ("hbm", "cxl"),
    require_configs: bool = True,
) -> list[SweepPoint]:
    """Enumerate every gem5 invocation the sweep will make.

    Args:
        require_configs: When False, a missing DRAMSim3 install yields
            placeholder device paths instead of an error, so the plan can be
            inspected while gem5 is still building.
    """
    configs_dir: Path | None
    try:
        configs_dir = find_dramsim_configs(third_party)
    except FileNotFoundError:
        if require_configs:
            raise
        configs_dir = None

    points: list[SweepPoint] = []
    for tier in tiers:
        if configs_dir is None:
            # Not a real path -- Path() is only so SweepPoint stays one type.
            # Slashes are avoided so it does not read as a directory.
            preference = " or ".join(DEVICE_PREFERENCE[tier])
            device = Path(f"<DRAMSim3 {preference} config, not fetched yet>")
        else:
            device = pick_device_config(configs_dir, tier)
        for period in periods_ps:
            point = SweepPoint(
                tier=tier,
                injection_period_ps=period,
                outdir=out_root / f"{tier}_p{period}",
                device_config=device,
                link_latency_ns=link_latency_ns if tier == "cxl" else 0.0,
            )
            points.append(point)
    return points


def command_for(point: SweepPoint, gem5: Path, transfer_bytes: int) -> list[str]:
    """The exact argv for one gem5 invocation."""
    argv = [
        str(gem5),
        f"--outdir={point.outdir}",
        str(TIER_CONFIG),
        "--tier", point.tier,
        "--dramsim-config", str(point.device_config),
        "--injection-period-ps", str(point.injection_period_ps),
        "--transfer-bytes", str(transfer_bytes),
    ]
    if point.tier == "cxl":
        argv += ["--link-latency-ns", str(point.link_latency_ns)]
    return argv


def run_sweep(
    link_latency_ns: float,
    periods_ps: tuple[int, ...] = DEFAULT_PERIODS_PS,
    third_party: Path = DEFAULT_THIRD_PARTY,
    out_root: Path = DEFAULT_OUT,
    transfer_bytes: int = 64 * 1024 * 1024,
    dry_run: bool = False,
    tiers: tuple[str, ...] = ("hbm", "cxl"),
) -> list[SweepPoint]:
    """Run every point, or print the commands when ``dry_run``.

    Args:
        link_latency_ns: One-way link delay for the CXL tier. Still
            TODO_PLACEHOLDER in :mod:`memsim.constants`; supply it explicitly and
            label results derived from an unsourced value as preliminary.
        periods_ps: Injection periods to sweep.
        transfer_bytes: Bytes read per point. Bigger reaches steady state more
            convincingly and costs proportionally more wall clock.
        dry_run: Print commands without running gem5.

    Returns:
        The points, so the caller can parse their output directories.
    """
    points = build_points(link_latency_ns, periods_ps, third_party, out_root, tiers,
                          require_configs=not dry_run)

    if dry_run:
        gem5_display = "<gem5.opt, not built yet>"
        notes: list[str] = []
        try:
            gem5_display = str(find_gem5(third_party))
        except FileNotFoundError as exc:
            notes.append(str(exc))
        if any("not fetched yet" in str(p.device_config) for p in points):
            notes.append("DRAMSim3 device configs are not present yet; the real sweep "
                         "picks them by preference from what it ships "
                         f"({DEVICE_PREFERENCE}).")
        for note in notes:
            print(f"NOTE: {note}\n")
        print(f"{len(points)} gem5 invocations:\n")
        for point in points:
            argv = command_for(point, Path(gem5_display), transfer_bytes)
            print("  " + " ".join(argv))
        if link_latency_ns == 0.0 and "cxl" in tiers:
            print("\nNOTE: --link-latency-ns was not given, so the cxl points above show 0,")
            print("      which the real run refuses. The constant is TODO_PLACEHOLDER in")
            print("      memsim/constants.py; supply a value explicitly.")
        return points

    gem5 = find_gem5(third_party)
    for i, point in enumerate(points, 1):
        point.outdir.mkdir(parents=True, exist_ok=True)
        argv = command_for(point, gem5, transfer_bytes)
        print(f"\n[{i}/{len(points)}] {point.label}  "
              f"({point.device_config.name}, period {point.injection_period_ps} ps)")
        print("  " + " ".join(argv))
        log_path = point.outdir / "gem5_stdout.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, text=True)
        if completed.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
            print(f"  FAILED (exit {completed.returncode}); last lines of {log_path}:",
                  file=sys.stderr)
            for line in tail:
                print(f"    {line}", file=sys.stderr)
            raise SystemExit(f"gem5 failed on {point.label}")
        print(f"  ok -> {point.outdir}")
    return points
