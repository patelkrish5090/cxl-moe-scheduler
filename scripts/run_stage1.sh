#!/usr/bin/env bash
# Stage 1 on the HPC box, in the order things should be checked.
#
#   bash scripts/run_stage1.sh probe      # environment report (paste back)
#   bash scripts/run_stage1.sh selftest   # offline correctness checks, no downloads
#   bash scripts/run_stage1.sh smoke      # tiny untrained model, ~1 min, no big download
#   bash scripts/run_stage1.sh olmoe      # first REAL profiling run (~14 GB download)
#   bash scripts/run_stage1.sh olmoe-decode   # decode regime, no new download
#   bash scripts/run_stage1.sh check          # Mixtral access/size/fit preflight
#   bash scripts/run_stage1.sh verify [dir]   # check an already-downloaded Mixtral
#   bash scripts/run_stage1.sh mixtral        # full target model (~87 GiB download)
#   bash scripts/run_stage1.sh mixtral-decode # Mixtral decode regime
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

  check)
    # Gated-access + size + fit preflight. Downloads nothing.
    python scripts/check_model_access.py "${2:-mistralai/Mixtral-8x7B-v0.1}"
    ;;

  verify)
    # Inspect a Mixtral directory downloaded by any means: are all 19 shards
    # present and intact, and how much of the disk is the unused consolidated
    # copy. Loads no weights.
    python scripts/verify_mixtral.py "${2:-models/mixtral}" --headers
    ;;

  olmoe-decode)
    # Decode is the regime a serving scheduler actually operates in. No new
    # download; OLMoE is already cached after `olmoe`.
    gpu_status
    python -m profiler.cli run configs/olmoe_decode.json
    python -m profiler.cli analyze data/runs/olmoe_1b7b_decode --phase decode
    ;;

  mixtral-decode)
    gpu_status
    python -m profiler.cli run configs/mixtral_8x7b_decode.json
    python -m profiler.cli analyze data/runs/mixtral_8x7b_decode --phase decode
    ;;

  mixtral)
    # 87 GiB in bf16, of which 84 GiB is expert weights (97% of the model).
    # Against GPU 1 alone (91 GiB) that leaves ~4 GiB -- too tight, so the
    # config caps GPU use and spills the remainder to CPU. Much better once
    # GPU 0 is free:  ASTERA_GPUS=0,1 bash scripts/run_stage1.sh mixtral
    gpu_status
    python -m profiler.cli run configs/mixtral_8x7b.json
    ;;

  *)
    sed -n '2,24p' "$0"
    exit 1
    ;;
esac
