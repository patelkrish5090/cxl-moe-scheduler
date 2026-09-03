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
- The CXL link is a fixed-delay `Bridge`. It charges a constant per-request
  latency and models no queuing, retries, or protocol overhead at the link.
- Device energy is DRAMSim3's model from JEDEC-derived IDD parameters. It is a
  simulator output, not a hardware measurement (docs.md §7).

## Validation checkpoint 2

docs.md §6: *"HBM-only latency/energy numbers should be lower than HBM+CXL for
the same access pattern; if CXL comes out faster, the model config is wrong."*

`python -m memsim.cli compare` runs that as three checks and prints PASS / FAIL /
n/a per check, exiting non-zero on failure. **Equal** latencies between tiers fail
rather than pass — that is the signature of the link delay not being applied, and
it should not slip through.

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
