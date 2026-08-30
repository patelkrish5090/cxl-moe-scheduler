"""Preflight for a large gated model: access, size, and whether it will fit.

    python scripts/check_model_access.py mistralai/Mixtral-8x7B-v0.1

Downloads nothing. Checks, in order:

1. Does the repo exist and is it gated? If gated and not yet approved, says so
   rather than letting a 93 GB download fail at the last file.
2. How large is the checkpoint, and how large are the expert weights alone?
3. Given the GPUs visible right now (respecting CUDA_VISIBLE_DEVICES), does it
   fit, and what max_memory should the run config use?
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

GIB = 1024 ** 3


def _api(url: str, token: str | None) -> tuple[int, dict | None]:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, None
    except Exception as exc:  # network / TLS
        print(f"  request failed: {exc}")
        return -1, None


def check_access(repo: str, token: str | None) -> dict | None:
    print(f"\n[1] REPO ACCESS  --  {repo}")
    status, data = _api(f"https://huggingface.co/api/models/{repo}", token)

    if status == 401 or status == 403:
        print(f"  status {status}: ACCESS DENIED.")
        print("  This model is gated. Do both of these:")
        print(f"    1. Open https://huggingface.co/{repo} while logged in and accept the terms.")
        print("    2. export HF_TOKEN=hf_...   (or run: hf auth login)")
        return None
    if status == 404:
        print("  status 404: repo not found. Check the spelling.")
        return None
    if status != 200 or data is None:
        print(f"  status {status}: could not read repo metadata.")
        return None

    gated = data.get("gated")
    print(f"  status 200: readable{' (gated, and you have access)' if gated else ''}")
    if gated and not token:
        print("  NOTE: this repo is marked gated but responded without a token.")
        print("        The download may still require one -- set HF_TOKEN to be safe.")
    return data


def report_size(repo: str, data: dict, token: str | None) -> float:
    """Print checkpoint size. Returns total parameter bytes in bf16, in GiB."""
    print(f"\n[2] SIZE")
    status, tree = _api(
        f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true", token
    )
    total_bytes = 0
    if status == 200 and tree:
        for entry in tree:
            if entry.get("type") == "file" and entry["path"].endswith(".safetensors"):
                size = entry.get("size") or (entry.get("lfs") or {}).get("size") or 0
                total_bytes += size
    if total_bytes:
        print(f"  checkpoint on disk : {total_bytes / GIB:.1f} GiB")
    else:
        print("  checkpoint size    : unavailable from the tree API")

    config = None
    try:
        with urllib.request.urlopen(
            f"https://huggingface.co/{repo}/raw/main/config.json", timeout=30
        ) as response:
            config = json.load(response)
    except Exception:
        pass

    if config:
        layers = config.get("num_hidden_layers")
        experts = config.get("num_local_experts") or config.get("num_experts")
        top_k = config.get("num_experts_per_tok")
        hidden = config.get("hidden_size")
        inter = config.get("intermediate_size")
        print(f"  layers={layers}  experts/layer={experts}  top_k={top_k}  "
              f"hidden={hidden}  intermediate={inter}")
        if all(isinstance(v, int) for v in (layers, experts, hidden, inter)):
            # Mixtral-style expert: w1, w2, w3 each hidden x intermediate.
            per_expert = 3 * hidden * inter * 2  # bf16 = 2 bytes
            expert_total = per_expert * experts * layers
            print(f"  expert weights     : {per_expert / GIB:.3f} GiB each, "
                  f"{expert_total / GIB:.1f} GiB total")
            print(f"  -> experts are {expert_total / total_bytes:.0%} of the checkpoint"
                  if total_bytes else "")
    return total_bytes / GIB if total_bytes else 0.0


def report_fit(size_gib: float) -> None:
    print(f"\n[3] FIT  (respecting CUDA_VISIBLE_DEVICES)")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(f"  CUDA_VISIBLE_DEVICES = {visible or '(unset -> all GPUs)'}")
    try:
        import torch
    except Exception:
        print("  torch unavailable; cannot check GPU capacity.")
        return
    if not torch.cuda.is_available():
        print("  no CUDA devices visible.")
        return

    free_total = 0.0
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        free_total += free / GIB
        print(f"  cuda:{i} ({torch.cuda.get_device_properties(i).name}): "
              f"{free / GIB:.1f} GiB free of {total / GIB:.1f} GiB")

    if not size_gib:
        print("  (checkpoint size unknown; cannot judge fit)")
        return

    headroom = free_total - size_gib
    print(f"\n  model {size_gib:.1f} GiB vs {free_total:.1f} GiB free -> "
          f"headroom {headroom:+.1f} GiB")
    if headroom < 0:
        spill = -headroom
        print(f"  DOES NOT FIT on the visible GPUs. Either free a GPU, make more visible")
        print(f"  (ASTERA_GPUS=0,1), or let ~{spill:.0f} GiB spill to CPU via max_memory.")
    elif headroom < 8:
        print("  Fits, but with under 8 GiB spare for activations and KV cache.")
        print("  Use batch_size 1, and consider capping max_memory so a few layers")
        print("  sit on CPU rather than OOMing mid-run.")
    else:
        print("  Fits comfortably.")

    # Suggested max_memory, leaving 10 GiB per GPU for activations.
    suggestion = {}
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        usable = max(0, free / GIB - 10)
        suggestion[str(i)] = f"{usable:.0f}GiB"
    suggestion["cpu"] = "150GiB"
    print(f"\n  suggested config max_memory: {json.dumps(suggestion)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", nargs="?", default="mistralai/Mixtral-8x7B-v0.1")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token

            token = get_token()
        except Exception:
            token = None
    print(f"HF token: {'found' if token else 'NOT SET'}")

    data = check_access(args.repo, token)
    if data is None:
        return 1
    size_gib = report_size(args.repo, data, token)
    report_fit(size_gib)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
