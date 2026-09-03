"""Turn one gem5 run directory into latency, bandwidth and energy numbers.

    python -m memsim.cli parse memsim/out/hbm_p1000
    python -m memsim.cli parse memsim/out/hbm_p1000 --dump   # every key found

gem5 and DRAMSim3 have both renamed statistics across releases, and a stat key
that silently does not match produces a *missing* number, not an error. So this
parser never assumes a key exists: for each quantity it tries a list of
candidates in priority order, records which one it actually used, and -- when
none match -- prints the keys that look related so the parser can be corrected
against the real output rather than guessed at.

That is what ``--dump`` is for. On the first real gem5 run, dump the keys and
check them against :data:`CANDIDATES` before trusting any number here.

UNITS
-----
``*_ns`` nanoseconds, ``*_pj`` picojoules, ``*_gbps`` gigabytes per second,
``*_bytes`` bytes. gem5 reports latency in ticks (1 tick = 1 ps by default) and
DRAMSim3 reports energy in picojoules; both conversions happen here, once.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: gem5's default tick rate. One tick is one picosecond unless the simulation
#: overrides it, which none of our configs do.
TICKS_PER_NS = 1000.0

#: Candidate gem5 stat keys per quantity, most specific first. Extend these
#: from a --dump of a real run rather than guessing.
CANDIDATES: dict[str, tuple[str, ...]] = {
    "sim_seconds": ("simSeconds",),
    "bytes_read": (
        "system.generator.bytesRead",
        "system.mem_ctrl.bytesRead::total",
        "system.mem_ctrl.bytesRead",
    ),
    "total_reads": (
        "system.generator.totalReads",
        "system.generator.readReqs",
        "system.mem_ctrl.readReqs",
    ),
    "avg_read_latency_ticks": (
        "system.generator.avgReadLatency",
        "system.mem_ctrl.avgMemAccLat",
    ),
    "total_read_latency_ticks": (
        "system.generator.totalReadLatency",
        "system.mem_ctrl.totMemAccLat",
    ),
}

#: Substrings used to surface "did you mean" keys when nothing matched.
RELATED_HINTS: dict[str, tuple[str, ...]] = {
    "bytes_read": ("bytesread", "readbytes"),
    "total_reads": ("readreq", "totalread", "numreads"),
    "avg_read_latency_ticks": ("latency", "lat"),
    "total_read_latency_ticks": ("latency", "lat"),
}


def read_gem5_stats(path: Path) -> dict[str, float]:
    """Parse a gem5 ``stats.txt`` into ``{key: value}``.

    Non-numeric statistics (distributions, strings) are skipped: nothing here
    needs them, and including them would make every lookup return a type the
    caller has to re-check.
    """
    stats: dict[str, float] = {}
    if not path.exists():
        return stats
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            stats[parts[0]] = float(parts[1])
        except ValueError:
            continue  # distribution rows and the like
    return stats


def _lookup(stats: dict[str, float], quantity: str) -> tuple[float | None, str | None]:
    """First matching candidate key for ``quantity``, and which key it was."""
    for key in CANDIDATES.get(quantity, ()):
        if key in stats:
            return stats[key], key
    return None, None


def _related_keys(stats: dict[str, float], quantity: str, limit: int = 8) -> list[str]:
    """Keys that look like they might be the one we wanted, for the error message."""
    hints = RELATED_HINTS.get(quantity, ())
    if not hints:
        return []
    found = [k for k in stats if any(h in k.lower() for h in hints)]
    return sorted(found)[:limit]


def find_dramsim_output(outdir: Path) -> Path | None:
    """Locate DRAMSim3's own statistics dump inside a gem5 output directory.

    DRAMSim3 has written this as .json, .csv and .txt depending on version and
    build options, so search rather than assume a name.
    """
    patterns = ("dramsim3*.json", "dramsim3*.csv", "dramsim3*.txt", "*dramsim*")
    for pattern in patterns:
        matches = sorted(outdir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _walk_energy_fields(node: Any, prefix: str = "") -> dict[str, float]:
    """Collect every numeric ``*_energy`` leaf from DRAMSim3's nested JSON."""
    found: dict[str, float] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if key.endswith("_energy") or key == "energy":
                    found[path] = float(value)
            else:
                found.update(_walk_energy_fields(value, path))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found.update(_walk_energy_fields(value, f"{prefix}[{i}]"))
    return found


def read_dramsim_energy(path: Path | None) -> tuple[float | None, dict[str, float], list[str]]:
    """Total DRAM device energy in picojoules, plus the per-field breakdown.

    Returns:
        ``(total_pj, breakdown, warnings)``. ``total_pj`` is None when the file
        is absent or holds no recognisable energy field -- never 0.0, because a
        zero would flow into a results table looking like a real measurement.
    """
    warnings: list[str] = []
    if path is None or not path.exists():
        return None, {}, ["no DRAMSim3 output file found in the run directory"]

    text = path.read_text(encoding="utf-8", errors="replace")

    if path.suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            return None, {}, [f"DRAMSim3 JSON at {path.name} is malformed: {exc}"]
        fields = _walk_energy_fields(data)
        if not fields:
            return None, {}, [f"{path.name} parsed but declares no *_energy field"]
        # Prefer an explicit total if DRAMSim3 provided one; summing component
        # fields as well as a total would double count.
        totals = {k: v for k, v in fields.items() if k.endswith("total_energy")}
        if totals:
            return sum(totals.values()), fields, warnings
        warnings.append("no total_energy field; summed the component *_energy fields")
        return sum(fields.values()), fields, warnings

    # csv / txt: key,value or key = value per line.
    fields = {}
    for line in text.splitlines():
        match = re.match(r"\s*([A-Za-z0-9_.\[\]]*energy[A-Za-z0-9_.\[\]]*)\s*[,=:]\s*([-\d.eE+]+)",
                         line, re.IGNORECASE)
        if match:
            try:
                fields[match.group(1)] = float(match.group(2))
            except ValueError:
                continue
    if not fields:
        return None, {}, [f"{path.name} holds no recognisable energy field"]
    totals = {k: v for k, v in fields.items() if "total" in k.lower()}
    if totals:
        return sum(totals.values()), fields, warnings
    warnings.append("no total energy field; summed the component energy fields")
    return sum(fields.values()), fields, warnings


@dataclass
class TierMeasurement:
    """One simulated point: one tier at one injection rate.

    Every field is either a real number from the simulator or None. Nothing is
    defaulted to zero, because a zero in an energy column is indistinguishable
    from a measurement and would survive into a plot.

    Attributes:
        tier: "hbm" or "cxl".
        outdir: The gem5 output directory this came from.
        bytes_read: Total bytes the traffic generator read.
        sim_seconds: Simulated wall time.
        avg_read_latency_ns: Mean read latency.
        bandwidth_gbps: Achieved read bandwidth.
        device_energy_pj: DRAM device energy from DRAMSim3. Excludes any link
            energy -- gem5 does not model a CXL PHY, see memsim/constants.py.
        device_energy_pj_per_bit: The figure stage 3 actually consumes.
        stat_sources: Which gem5 stat key each number was read from.
        warnings: Anything the parser could not resolve.
    """

    tier: str
    outdir: Path
    bytes_read: float | None = None
    sim_seconds: float | None = None
    avg_read_latency_ns: float | None = None
    bandwidth_gbps: float | None = None
    device_energy_pj: float | None = None
    device_energy_pj_per_bit: float | None = None
    energy_breakdown_pj: dict[str, float] = field(default_factory=dict)
    stat_sources: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """True when every number a comparison needs is present."""
        return None not in (
            self.bytes_read,
            self.avg_read_latency_ns,
            self.bandwidth_gbps,
            self.device_energy_pj_per_bit,
        )


def parse_run(outdir: str | Path, tier: str | None = None) -> TierMeasurement:
    """Read one gem5 output directory into a :class:`TierMeasurement`.

    Args:
        outdir: A directory written by ``gem5.opt --outdir=...``.
        tier: "hbm" or "cxl". Inferred from the directory name when omitted.
    """
    outdir = Path(outdir)
    if tier is None:
        name = outdir.name.lower()
        tier = "cxl" if "cxl" in name else "hbm"

    result = TierMeasurement(tier=tier, outdir=outdir)
    if not outdir.is_dir():
        result.warnings.append(f"{outdir} is not a directory")
        return result

    stats = read_gem5_stats(outdir / "stats.txt")
    if not stats:
        result.warnings.append("stats.txt missing or empty -- did the gem5 run finish?")
        return result

    for quantity in ("sim_seconds", "bytes_read", "total_reads"):
        value, key = _lookup(stats, quantity)
        if key:
            result.stat_sources[quantity] = key
        if value is None:
            related = _related_keys(stats, quantity)
            hint = f"; related keys present: {related}" if related else ""
            result.warnings.append(f"no gem5 stat matched {quantity}{hint}")

    result.sim_seconds = _lookup(stats, "sim_seconds")[0]
    result.bytes_read = _lookup(stats, "bytes_read")[0]

    # Latency: prefer a mean the simulator computed; otherwise derive it, which
    # needs both a total and a count.
    avg_ticks, key = _lookup(stats, "avg_read_latency_ticks")
    if avg_ticks is not None:
        result.stat_sources["avg_read_latency_ticks"] = key or ""
    else:
        total_ticks, total_key = _lookup(stats, "total_read_latency_ticks")
        n_reads, n_key = _lookup(stats, "total_reads")
        if total_ticks is not None and n_reads:
            avg_ticks = total_ticks / n_reads
            result.stat_sources["avg_read_latency_ticks"] = f"{total_key}/{n_key}"
        else:
            related = _related_keys(stats, "avg_read_latency_ticks")
            result.warnings.append(
                "could not determine read latency"
                + (f"; related keys present: {related}" if related else "")
            )
    if avg_ticks is not None:
        result.avg_read_latency_ns = avg_ticks / TICKS_PER_NS

    if result.bytes_read and result.sim_seconds:
        # GB/s with GB = 1e9 bytes, matching how memory bandwidth is quoted.
        result.bandwidth_gbps = result.bytes_read / result.sim_seconds / 1e9

    dramsim_path = find_dramsim_output(outdir)
    energy_pj, breakdown, energy_warnings = read_dramsim_energy(dramsim_path)
    result.device_energy_pj = energy_pj
    result.energy_breakdown_pj = breakdown
    result.warnings.extend(energy_warnings)
    if energy_pj is not None and result.bytes_read:
        result.device_energy_pj_per_bit = energy_pj / (result.bytes_read * 8.0)

    return result


def dump_keys(outdir: str | Path) -> str:
    """Every statistic key found, for correcting :data:`CANDIDATES` against reality."""
    outdir = Path(outdir)
    lines = [f"KEYS IN {outdir}", ""]

    stats = read_gem5_stats(outdir / "stats.txt")
    lines.append(f"gem5 stats.txt: {len(stats)} numeric keys")
    if stats:
        matched = {k for names in CANDIDATES.values() for k in names if k in stats}
        lines.append(f"  matched by CANDIDATES: {sorted(matched) or 'NONE'}")
        lines.append("  all keys:")
        lines.extend(f"    {k} = {v:g}" for k, v in sorted(stats.items()))

    dramsim_path = find_dramsim_output(outdir)
    lines.append("")
    lines.append(f"DRAMSim3 output: {dramsim_path if dramsim_path else 'NOT FOUND'}")
    if dramsim_path:
        _, breakdown, warnings = read_dramsim_energy(dramsim_path)
        for warning in warnings:
            lines.append(f"  WARNING: {warning}")
        lines.extend(f"    {k} = {v:g}" for k, v in sorted(breakdown.items()))
        if not breakdown:
            lines.append("  (no energy fields; first 40 lines of the file:)")
            head = dramsim_path.read_text(encoding="utf-8", errors="replace").splitlines()[:40]
            lines.extend(f"    {line}" for line in head)

    lines.append("")
    lines.append("Files in the run directory:")
    lines.extend(f"    {p.name}  ({p.stat().st_size:,} B)"
                 for p in sorted(outdir.iterdir()) if p.is_file())
    return "\n".join(lines)
