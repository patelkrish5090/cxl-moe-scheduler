#!/usr/bin/env bash
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
    python -m profiler.cli selftest
    python tests/test_runner_integration.py
    ;;

  smoke)
    python -m profiler.cli run configs/smoke_tiny.json
    ;;

  olmoe)
    gpu_status
    python -m profiler.cli run configs/olmoe_1b7b.json
    ;;

  check)
    python scripts/check_model_access.py "${2:-mistralai/Mixtral-8x7B-v0.1}"
    ;;

  verify)
    python scripts/verify_mixtral.py "${2:-models/mixtral}" --headers
    ;;

  olmoe-decode)
    gpu_status
    python -m profiler.cli run configs/olmoe_decode.json
    python -m profiler.cli analyze data/runs/olmoe_1b7b_decode \
      --phase decode --max-sites 0 --layers both
    ;;

  mixtral-decode)
    gpu_status
    python -m profiler.cli run configs/mixtral_8x7b_decode.json
    python -m profiler.cli analyze data/runs/mixtral_8x7b_decode \
      --phase decode --max-sites 0 --layers both
    ;;

  mixtral)
    gpu_status
    python -m profiler.cli run configs/mixtral_8x7b.json
    ;;

  *)
    sed -n '2,24p' "$0"
    exit 1
    ;;
esac
