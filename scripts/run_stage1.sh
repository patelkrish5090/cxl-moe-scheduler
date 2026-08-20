#!/usr/bin/env bash
# Stage 1 on the HPC box, in the order things should be checked.
#
#   bash scripts/run_stage1.sh probe      # environment report (paste back)
#   bash scripts/run_stage1.sh selftest   # offline correctness checks, no downloads
#   bash scripts/run_stage1.sh smoke      # tiny untrained model, ~1 min, no big download
#   bash scripts/run_stage1.sh olmoe      # first REAL profiling run (~14 GB download)
#   bash scripts/run_stage1.sh mixtral    # full target model (~93 GB download)
#
# Environment setup is separate:  bash scripts/setup_env.sh && conda activate astera
#
# Stop and check the output after each step. `selftest` and `smoke` must be
# clean before spending a download on `olmoe`.
#
# GPU SELECTION
# -------------
# The probe showed GPU 0 holding ~88 GB of someone else's work and GPU 1 idle,
# so this script pins to physical GPU 1 by default. CUDA_VISIBLE_DEVICES
# RENUMBERS devices: with `CUDA_VISIBLE_DEVICES=1`, physical GPU 1 becomes
# `cuda:0` inside the process, which is what the configs refer to. Override with:
#
#   ASTERA_GPUS=0,1 bash scripts/run_stage1.sh olmoe    # once GPU 0 is free
#
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${ASTERA_GPUS:-1}"
CMD="${1:-help}"

gpu_status() {
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true
  echo
}

case "$CMD" in
  probe)
    python scripts/probe_env.py
    ;;

  selftest)
    # Offline, CPU-only. Builds tiny models in-process and checks the
    # profiler's counts against an independently recomputed ground truth.
    python -m profiler.cli selftest
    python tests/test_runner_integration.py
    ;;

  smoke)
    # Untrained tiny model + WikiText-2 on CPU. Proves the corpus path and the
    # full run directory. Its histogram WILL look flat -- that is correct for
    # untrained weights and is NOT evidence about real expert skew.
    python -m profiler.cli run configs/smoke_tiny.json
    ;;

  olmoe)
    # First run with real trained weights (~14 GB bf16, fits GPU 1 alone).
    # The histogram here is what answers validation checkpoint 1 in docs.md.
    gpu_status
    python -m profiler.cli run configs/olmoe_1b7b.json
    ;;

  mixtral)
    # ~93 GB in bf16 against ~98 GB of usable GPU 1. It fits only with GPU 0
    # still occupied elsewhere and batch_size 1; if it OOMs, either wait for
    # GPU 0 to free up and rerun with ASTERA_GPUS=0,1, or lower seq_len.
    gpu_status
    python -m profiler.cli run configs/mixtral_8x7b.json
    ;;

  *)
    sed -n '2,24p' "$0"
    exit 1
    ;;
esac
