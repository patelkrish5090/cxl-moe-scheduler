# CXL-MoE Scheduler

This project tests an idea: when a Mixture-of-Experts (MoE) model is too big to
fit in GPU memory, can we choose which experts to keep in fast memory (HBM)
and which to push to slower memory (CXL) in a way that saves real energy, not
just speed. We built four stages that work together: a profiler that watches
a real model run, a memory simulator that times real hardware tiers, a
scheduler that decides where each expert lives, and a dashboard that shows
the results.

This guide covers what to install, where to put things, and how to run each
stage in order.

---

## 1. Hardware this was built for

- 2 GPUs (this project used 2x Blackwell RTX 6000, 180GB VRAM combined)
- About 200GB of system RAM
- A modern multi-core CPU (this project used a Xeon Gold 6530)
- About 100GB of free disk space (the Mixtral-8x7B model alone is about 87GB)

You do not need this exact hardware to read the code, but the real profiling
and scheduler runs described below do need a real GPU with enough memory to
load the model you pick.

---

## 2. Where to keep this project

Put this whole folder anywhere on the machine you plan to run it on. Every
command in this guide assumes you are inside that folder (the one with
`CLAUDE.md`, `docs.md`, and this `README.md` in it).

A few subfolders are meant to stay local and are not pushed to GitHub
(see `.gitignore`):

- `models/` holds downloaded model weights. These are tens of gigabytes each,
  too big for the repo.
- `data/` holds profiler output (activation traces, hot/cold tables).
- `third_party/` holds the gem5 and DRAMSim3 source, fetched by a script.
- `memsim/out/` holds raw memory simulator output.
- `experiments/results/` holds the comparison JSON files the dashboard reads.

None of these need to exist before you start. The scripts below create them.

---

## 3. Install the Python environment

You need Python 3.12 and conda. From the project folder, run:

```bash
bash scripts/setup_env.sh
conda activate astera
```

This script creates a conda environment named `astera`, installs a CUDA
build of PyTorch that matches your GPU driver, then installs everything in
`requirements.txt` (transformers, datasets, accelerate, pandas, streamlit,
plotly, and a few others). It prints a short GPU check at the end so you can
confirm your GPUs are visible.

If you want to check your machine before installing anything, run:

```bash
python scripts/probe_env.py
```

---

## 4. Download a model to profile

The profiler needs a real Hugging Face model on disk. For the main result in
this project, that is Mixtral-8x7B:

```bash
bash scripts/download_mixtral.sh ./models/Mixtral-8x7B-v0.1
```

This downloads about 87GB of weights. It can take a while depending on your
connection. Smaller models used elsewhere in this project (GPT-2, OLMoE) are
downloaded automatically the first time you run a config that points at
them, since they are small enough to fetch on demand.

Check the download finished correctly with:

```bash
python scripts/verify_mixtral.py ./models/Mixtral-8x7B-v0.1
```

---

## 5. Run the four stages, in order

Each stage reads real output from the one before it. Run them in this order
the first time. All commands run from the project folder with the `astera`
environment active.

### Stage 1: Profiler

Watches a real model run and records which expert each token goes to, then
splits experts into hot (used a lot) and cold (rarely used).

```bash
python -m profiler.cli selftest                          # offline check, no GPU needed
python -m profiler.cli run configs/mixtral_8x7b_decode.json   # a real profiling run
```

This writes a new folder under `data/runs/` with a trace file and a
`hot_cold.csv` table. Other configs live in `configs/` for other models
(OLMoE, dense GPT-2, and the two-GPU Mixtral configs used for pooling).

### Stage 2: Memory simulator

Times two memory tiers (HBM and CXL) as real hardware, using gem5 and
DRAMSim3, so later stages have real latency, bandwidth, and energy numbers
to work with.

```bash
bash scripts/build_gem5.sh check     # checks prerequisites, downloads nothing
bash scripts/build_gem5.sh           # builds gem5 + DRAMSim3, takes 30-60 minutes
python -m memsim.cli selftest        # offline check, needs no gem5 build
python -m memsim.cli sweep           # runs the real timing sweep
python -m memsim.cli compare         # writes memsim/tier_model.json
```

`memsim/tier_model.json` is what stage 3 uses for its cost numbers. You only
need to build gem5 once.

### Stage 3: Scheduler

Replays the stage-1 trace against the stage-2 tier numbers, deciding which
experts stay in HBM and which get evicted to CXL. Runs a plain baseline
alongside the energy-aware policy so they can be compared fairly.

```bash
python -m scheduler.cli selftest
python -m scheduler.cli compare3 data/runs/mixtral_8x7b_decode --power-budget-w 50
```

This prints a three-way comparison (HBM-only, naive, energy-aware) straight
to the terminal, including the total energy for each.

### Stage 4: Experiments and dashboard

Packages the three-way comparison into a JSON file, then shows it on a
dashboard.

```bash
python -m experiments.cli selftest
python -m experiments.cli run data/runs/mixtral_8x7b_decode
```

This writes `experiments/results/mixtral_8x7b_decode.json`. Now start the
dashboard:

```bash
streamlit run dashboard/app.py
```

Open the URL it prints in a browser. Pick your result from the sidebar and
the page fills in with real charts and tables.

---

## 6. Two extra results

These are separate from the main three-way comparison, but the dashboard
shows them too if you produce them.

**Dense vs MoE comparison.** Run a plain (non-MoE) model through the same
profiler, then compare it against Mixtral on the dashboard's "Model
comparison" panel:

```bash
python -m profiler.cli run configs/gpt2_dense_decode.json
```

**CXL pooling.** If you have 2 GPUs, run the profiler once per GPU, then
compare storing each GPU's cold experts separately against pooling them into
one shared CXL space:

```bash
CUDA_VISIBLE_DEVICES=0 python -m profiler.cli run configs/mixtral_8x7b_decode_gpu0.json
CUDA_VISIBLE_DEVICES=1 python -m profiler.cli run configs/mixtral_8x7b_decode_gpu1.json
python -m scheduler.cli pool data/runs/mixtral_8x7b_decode_gpu0 data/runs/mixtral_8x7b_decode_gpu1 \
  --out experiments/results/mixtral_pooling.json
```

---

## 7. Where everything lands

| what | where |
| --- | --- |
| downloaded model weights | `models/<model name>/` |
| profiler output (one folder per run) | `data/runs/<run name>/` |
| memory tier timing and energy numbers | `memsim/tier_model.json` |
| three-way comparison results | `experiments/results/<run name>.json` |
| CXL pooling results | `experiments/results/<pooling name>.json` |

---

## 8. More detail

Each stage has its own README with the full explanation of its design,
constants used, and any real problems found and fixed along the way:

- `profiler/README.md`
- `memsim/README.md`
- `scheduler/README.md`
- `experiments/README.md`
- `dashboard/README.md`
