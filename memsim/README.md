# Stage 2 — Memory system model

Characterises two memory tiers with gem5 + DRAMSim3 and hands stage 3 a per-tier
latency / bandwidth / energy model (docs.md §4.3).

| tier | what it is |
| --- | --- |
| `hbm` | direct-attach memory on the memory bus — the GPU baseline |
| `cxl` | the same class of DRAM device behind a modelled link delay |

**No physical CXL hardware is involved and none is implied.** The link latency is
a parameter supplied on the command line; the link *energy* is not modelled by
gem5 at all and is a separate, explicitly-sourced constant. See
[Provenance](#provenance).

## Why characterise, not replay

The obvious design — push the stage-1 activation trace through gem5 and read off
total energy — is infeasible by several orders of magnitude. gem5 in timing mode
runs ~10⁵ simulated instructions/sec, and one Mixtral expert is 352 MB ≈ 5.5
million 64-byte reads for a **single** fetch. The decode trace has tens of
thousands of dispatches.

So gem5 characterises the *tiers*, not the workload. Each invocation measures, for
one tier at one load level: achieved read bandwidth, average read latency, and
DRAM device energy from DRAMSim3's device model. Stage 3 multiplies those
per-byte figures by real fetch counts and expert sizes from the stage-1 trace.
Simulation cost scales with the number of **configurations** (a handful), not the
number of **accesses** (billions).

The access pattern is a saturating sequential read, because that is what an
expert-weight fetch is: one large contiguous block moved one way. Modelling it as
random access would understate both tiers.

## Commands

```bash
bash scripts/build_gem5.sh check        # prerequisites, ~5 s, downloads nothing
bash scripts/build_gem5.sh              # fetch + build, 30-60 min, ~10 GB
bash scripts/build_gem5.sh verify       # does this gem5 really have DRAMSim3?

python -m memsim.cli selftest           # offline, needs no gem5
python -m memsim.cli provenance         # every constant and where it came from
python -m memsim.cli sweep --dry-run    # the exact gem5 commands, run nothing
python -m memsim.cli sweep --tiers hbm  # characterise HBM alone (needs no link value)
python -m memsim.cli parse memsim/out/hbm_p1000 --dump
python -m memsim.cli compare            # checkpoint 2 + writes tier_model.json
```

`build_gem5.sh verify` exists because gem5 will happily build **without**
DRAMSim3 and then fail at runtime with `object 'DRAMsim3' not found`, an hour
later. `verify` actually instantiates the SimObject rather than trusting the
link step.

## Injection-rate sweep

One injection rate cannot answer both questions a tiering decision needs, so the
sweep varies it:

- **Fast injection** (small `--periods-ps`) saturates the memory system →
  measures **peak bandwidth**, i.e. how many concurrent fetches a tier absorbs.
- **Slow injection** (large period) leaves it idle → measures **unloaded
  latency**, i.e. the stall one fetch imposes with nothing else in flight.

`compare` reduces the sweep to one figure of each kind per tier. Energy per bit
is taken from the *saturated* point: background and refresh energy accrue with
simulated time, so a nearly-idle run charges more of them per bit moved, and the
saturated figure is the honest one for a bulk expert fetch.

## Provenance

CLAUDE.md forbids both bare magic numbers and invented ones. `memsim/constants.py`
enforces this structurally rather than by convention: an unsourced constant has
the value `float('nan')`, so **any total computed from it is NaN** and cannot be
mistaken for a result, tabulated, or plotted. `compare` prints a PRELIMINARY
banner naming what is still missing.

| marker | meaning |
| --- | --- |
| `!!` | needs a citation before any result using it can be published |
| `..` | comes from a gem5/DRAMSim3 run; simply not measured yet |
| (blank) | sourced |

Two constants are outstanding, both about the link itself:

- `CXL_LINK_LATENCY_NS` — added read round-trip vs local DDR. Record the CXL
  revision, the PHY generation, and whether the figure is **idle or loaded**
  latency; loaded latency under bandwidth pressure is much higher and is the
  honest number for a memory-bound workload.
- `CXL_LINK_ENERGY_PJ_PER_BIT` — PHY + protocol energy per bit, **excluding** the
  DRAM device, which DRAMSim3 already accounts for. Double-counting it here would
  inflate every CXL energy figure. Record per-direction vs aggregate.

Until both are filled in, `compare` reports device energy (real, simulated) and
link energy (`TODO`) in **separate columns**. Summing them into one number would
hide the difference between what was measured and what was assumed.

## Modelling limitations to state wherever these numbers appear

- DRAMSim3 ships no HBM3e device model, so the HBM tier uses the closest device
  it does ship (HBM2). The substitution is a limitation, not a detail.
- **The CXL tier's achieved bandwidth (~2.35 GB/s) is bottlenecked by the
  link path's request concurrency, NOT by the underlying DRAM device.**
  Checked with a controlled experiment, not assumed: swapping the cxl tier's
  DRAMSim3 device from DDR4-1866 to DDR4-3200 (`tCK` 1.07 ns -> 0.63 ns, a
  genuine ~1.7x clock difference, confirmed by reading both `.ini` files)
  left achieved bandwidth **unchanged** (2.36 -> 2.35 GB/s) while device
  energy per bit nearly doubled (29.70 -> 56.68 pJ/bit) — a faster part
  burning more power for no extra throughput, the signature of a bottleneck
  that isn't the DRAM. The `hbm` tier (direct-attach, no `Bridge`) shows no
  such flatness; its bandwidth scales with its own device config normally.
  The CXL link is a fixed-delay `Bridge` (charges a constant per-request
  round-trip latency, `CXL_LINK_LATENCY_NS`) with gem5's default outstanding-
  request queue depth on that path — that combination, not the DRAM device
  behind it, is the most plausible explanation for the ceiling. In effect,
  this measures "one traffic-generator stream's throughput against a
  fixed-round-trip link at default queue depth," which is the same
  single-outstanding-request / no-overlap assumption this project already
  states explicitly at the scheduler level (`scheduler/README.md`'s "WHAT
  THIS DOES NOT MODEL") — just also present one layer down, in how the
  memory tier itself was characterised. State this explicitly wherever
  `mean_miss_latency_ns` / the cxl bandwidth figure is quoted (see
  `scheduler/README.md`'s "Is a large latency figure a units bug, or this
  model?" for the full evidence trail) — it is a deliberate-in-effect
  worst-case bound now, not an unexplained anomaly still being chased.
- **The CXL tier also characterises a SINGLE DDR4 channel**, not multiple
  channels aggregated the way a real CXL memory expander commonly is — a
  separate, additional reason absolute bandwidth reads low next to a real
  device's aggregate spec. Adding multi-channel aggregation would be a real
  scope increase (parallel DRAMSim3 instances or a wider channel config), not
  a one-line fix.
- Device energy is DRAMSim3's model from JEDEC-derived IDD parameters. It is a
  simulator output, not a hardware measurement (docs.md §7).
- `run_sweep.py::pick_device_config` selects the FASTEST speed grade DRAMSim3
  ships within a device family (e.g. DDR4 up to 3200 MT/s), not merely
  whichever filename happens to sort first (fixed 2026-09-12: this project's
  own DRAMSim3 checkout ships no DDR5 configs at all, so the cxl tier fell
  back to DDR4, and the picker's old alphabetical tie-break had silently
  chosen `DDR4_4Gb_x16_1866.ini`, one of the slowest DDR4 speed grades
  shipped, purely because "1866" sorts before "3200" as a string — never a
  deliberate choice; see `memsim/selftest.py`'s "sweep planning" regression
  test). This fix is still worth keeping (never pick the slowest available
  part by accident, and the corrected part's device-energy figure is more
  representative of a real DDR4-3200 chip) — but re-running the sweep with it
  did NOT change the achieved bandwidth (see the bullet above): the actual
  bottleneck was the link path, not the device selection.

## Validation checkpoint 2

docs.md §6: *"HBM-only latency/energy numbers should be lower than HBM+CXL for
the same access pattern; if CXL comes out faster, the model config is wrong."*

`python -m memsim.cli compare` runs that as three checks and prints PASS / FAIL /
n/a per check, exiting non-zero on failure. **Equal** latencies between tiers fail
rather than pass — that is the signature of the link delay not being applied, and
it should not slip through.

## QEMU CXL VM (optional)

docs.md 4.4 marks this an OPTIONAL strengthening layer: a real Linux guest
kernel's OS-level view of a CXL memory region (DAX device allocation,
`/sys/bus/cxl` topology), to check the tier model's assumptions look right
from inside a real (if virtual) CXL-attached memory device -- NOT a source
of any latency or energy number used anywhere in this project (that is
gem5 + DRAMSim3, above, fully validated already). If this VM never boots or
the guest never sees the device, none of this project's reported numbers
change; only this corroboration step is missed.

```bash
bash scripts/build_qemu_cxl.sh check                          # prerequisites only, changes nothing
bash scripts/build_qemu_cxl.sh launch-cmd <disk.qcow2> [size]  # print the exact command, don't run it
bash scripts/build_qemu_cxl.sh launch <disk.qcow2> [size]      # run it
```

**Prerequisites `check` verifies**: QEMU >= 8.0 (basic CXL support landed in
7.0; real fixes through 8.0 -- older versions may build and boot but are far
more likely to misbehave than to work), the `cxl-type3` device compiled into
that QEMU build (`qemu-system-x86_64 -device help`), and `/dev/kvm`
read/write access for hardware acceleration (falls back to slow software
emulation otherwise, with a warning, not a hard stop).

**Getting a guest disk image**: this script does not create one. The
fastest path is a minimal cloud image with a kernel new enough for CXL
support compiled in (mainline since 5.12; Ubuntu 22.04's HWE kernel or
Ubuntu 23.04+'s default kernel both qualify) --

```bash
mkdir -p third_party/qemu-cxl
curl -L -o third_party/qemu-cxl/ubuntu.img \
  https://cloud-images.ubuntu.com/jammy/current/jammy-server-cloudimg-amd64.img
qemu-img resize third_party/qemu-cxl/ubuntu.img +16G
```

(cloud images boot via cloud-init and expect a seed ISO for first-boot
credentials -- see Ubuntu's own cloud-image docs for `cloud-localds` if this
is your first time; that step is standard QEMU/cloud-image setup, not
specific to this project.)

**The exact device topology** `build_qemu_args` constructs (a PCIe expander
bus carrying a CXL root port with one CXL Type 3 device, backed by a
memory-backend-ram object, plus a machine-level CXL Fixed Memory Window
declaring the routing) is QEMU's own documented minimal single-device CXL
setup (`docs/system/devices/cxl.rst` in the QEMU source tree) -- not an
invented configuration.

**Verifying it worked, from INSIDE the guest** (this script cannot do this
part for you -- it only gets you to a running VM):

```bash
ssh -p 2222 <user>@localhost          # the launch command forwards guest:22 -> host:2222
ls /sys/bus/cxl/devices/               # expect a mem0 (or similar) CXL memory device
sudo apt install cxl-utils daxctl      # cxl-cli + ndctl's daxctl, if not already present
sudo cxl list -M                       # lists CXL memory devices the kernel found
sudo daxctl list                       # once the region is configured as devdax
```

If `/sys/bus/cxl/devices/` is empty: check the guest kernel actually has
`CONFIG_CXL_BUS`/`CONFIG_CXL_ACPI`/`CONFIG_CXL_PCI`/`CONFIG_CXL_MEM` enabled
(`grep CXL /boot/config-$(uname -r)`) before suspecting the QEMU side again
-- an older or minimal-config guest kernel with CXL support compiled out is
at least as likely a cause as a QEMU misconfiguration.

## Correctness checks

`python -m memsim.cli selftest` (59 checks) runs entirely offline against
synthetic fixtures with hand-computable answers. It covers unit conversions, the
NaN-poisoning of unsourced constants, gem5 `stats.txt` parsing, DRAMSim3 JSON/CSV
energy extraction, the tier reduction (unloaded latency from the slowest point,
peak bandwidth from the fastest), device-config preference, and gem5 command
construction.

**What it cannot check:** that the gem5 stat key names in
`parse_stats.CANDIDATES` match your gem5 build. Only a real run settles that. On
the first run, do:

```bash
python -m memsim.cli parse memsim/out/hbm_p1 --dump
```

and correct the candidate lists against what it prints. The parser reports which
key it used for each quantity, and when nothing matches it lists the keys that
look related — it never silently returns a wrong number.

## Reproducibility

`scripts/build_gem5.sh verify` prints the gem5 tag and DRAMSim3 commit. Record
both alongside any results. The build tree lives in `third_party/` and is
gitignored; gem5 output lives in `memsim/out/` and is gitignored too.
