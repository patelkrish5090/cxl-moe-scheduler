"""Print the target machine's environment so profiler configs can be sized correctly.

Run on the HPC box and paste the output back:

    python scripts/probe_env.py

Read-only: downloads nothing, allocates no GPU memory beyond a tiny probe tensor.
"""

from __future__ import annotations

import importlib
import os
import platform
import shutil
import subprocess
import sys


def _section(title: str) -> None:
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def probe_platform() -> None:
    _section("PLATFORM")
    print(f"python      : {sys.version.split()[0]} ({sys.executable})")
    print(f"os          : {platform.platform()}")
    print(f"machine     : {platform.machine()}")
    print(f"cpu         : {platform.processor() or 'unknown'}")
    print(f"cpu_count   : {os.cpu_count()}")
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    print(f"cpu_model   : {line.split(':', 1)[1].strip()}")
                    break
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    print(f"system_ram  : {kb / 1024 / 1024:.1f} GiB")
                    break
    except OSError:
        pass


def probe_disk() -> None:
    _section("DISK / HF CACHE")
    cache = (
        os.environ.get("HF_HOME")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.path.expanduser("~/.cache/huggingface")
    )
    print(f"HF cache dir: {cache}")
    for label, path in (("cwd", os.getcwd()), ("hf_cache_parent", os.path.dirname(cache) or "/")):
        try:
            usage = shutil.disk_usage(path)
            print(f"{label:16s}: {usage.free / 1e9:.1f} GB free of {usage.total / 1e9:.1f} GB  ({path})")
        except OSError as exc:
            print(f"{label:16s}: unavailable ({exc})")


def probe_packages() -> None:
    _section("PACKAGES")
    for name in (
        "torch", "transformers", "datasets", "accelerate", "safetensors",
        "numpy", "pandas", "pyarrow", "matplotlib", "tqdm", "streamlit", "plotly",
    ):
        try:
            mod = importlib.import_module(name)
            print(f"{name:14s}: {getattr(mod, '__version__', 'unknown')}")
        except Exception:
            print(f"{name:14s}: MISSING")


def probe_torch() -> None:
    _section("TORCH / GPU")
    try:
        import torch
    except Exception as exc:
        print(f"torch import failed: {exc}")
        return

    print(f"torch        : {torch.__version__}")
    print(f"cuda_built   : {torch.version.cuda}")
    print(f"cuda_avail   : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("-> CPU-only build. Profiler will run, but only on a small model.")
        return

    print(f"device_count : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(
            f"  gpu{i}: {props.name} | {props.total_memory / 1e9:.1f} GB | "
            f"sm_{props.major}{props.minor} | {props.multi_processor_count} SMs"
        )
    print(f"bf16_support : {torch.cuda.is_bf16_supported()}")


def probe_nvidia_smi() -> None:
    _section("NVIDIA-SMI")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        print(out.stdout.strip() or out.stderr.strip() or "(no output)")
    except FileNotFoundError:
        print("nvidia-smi not found on PATH")
    except Exception as exc:
        print(f"nvidia-smi failed: {exc}")


def probe_hub() -> None:
    """Report whether target models are already cached locally. Downloads nothing."""
    _section("MODEL CACHE STATUS (no download attempted)")
    models = [
        "hf-internal-testing/Mixtral-tiny",
        "allenai/OLMoE-1B-7B-0924",
        "mistralai/Mixtral-8x7B-v0.1",
    ]
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception as exc:
        print(f"huggingface_hub unavailable: {exc}")
        return
    for repo in models:
        hit = try_to_load_from_cache(repo, "config.json")
        state = "CACHED" if isinstance(hit, str) else "not cached"
        print(f"{state:11s}: {repo}")

    print("\noffline flags:")
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_TOKEN", "HF_HOME"):
        val = os.environ.get(var)
        if var == "HF_TOKEN" and val:
            val = "<set>"
        print(f"  {var:20s}= {val or '(unset)'}")


def main() -> None:
    probe_platform()
    probe_disk()
    probe_packages()
    probe_torch()
    probe_nvidia_smi()
    probe_hub()
    print("\nDone. Paste everything above back into the chat.\n")


if __name__ == "__main__":
    main()
