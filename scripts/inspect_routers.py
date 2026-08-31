"""Are the router weights in the checkpoint actually being loaded?

    python scripts/inspect_routers.py models/mixtral

Reads router tensors straight out of the safetensors shards and instantiates the
model skeleton on the meta device. No GPU, no weights loaded into memory, a few
seconds.

WHY THIS EXISTS
---------------
The mixtral_8x7b_decode run showed 15 consecutive layers sending every token to
the same two experts, at exactly 50.0% each, on genuinely varied input. That is
the exact signature of tied router logits: if a gate's weights are zero or
constant, every token scores every expert identically, torch.topk breaks the tie
by index, and experts 0 and 1 win forever.

Tied logits have two plausible causes, and this script separates them:

  [1] The checkpoint itself carries degenerate gate weights (norm ~0, or rows
      identical). Would mean the download or the upstream file is at fault.
  [2] The checkpoint is fine but transformers cannot find the gate tensors under
      the names its model class expects -- a rename across transformers
      versions. Then the routers are randomly initialized or left at zero, and
      transformers reports them as "newly initialized" in a warning that is easy
      to miss in a long load log.

Section [3] is the direct check for [2]: build the model skeleton on meta (free,
allocates nothing) and diff its expected parameter names against the names in
the checkpoint index.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# 'gate_proj' and 'gate_up_proj' are expert FFN weights, not the router.
ROUTER_NAME_RE = re.compile(r"\.(gate|router)\.weight$")


def find_router_keys(weight_map: dict[str, str]) -> dict[int, list[str]]:
    """Map layer index -> checkpoint keys that look like a router projection."""
    by_layer: dict[int, list[str]] = defaultdict(list)
    for key in weight_map:
        match = LAYER_RE.search(key)
        if not match or not ROUTER_NAME_RE.search(key):
            continue
        by_layer[int(match.group(1))].append(key)
    return dict(sorted(by_layer.items()))


def inspect_checkpoint(model_dir: Path) -> dict[str, str]:
    """Print per-layer router weight statistics. Returns the checkpoint weight map."""
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"  no {index_path.name}; cannot locate shards")
        return {}
    weight_map: dict[str, str] = json.loads(index_path.read_text())["weight_map"]

    router_keys = find_router_keys(weight_map)
    if not router_keys:
        print("  NO router/gate tensors found in the checkpoint index.")
        print("  Keys sampled from the index:")
        for key in list(weight_map)[:10]:
            print(f"    {key}")
        return weight_map

    try:
        from safetensors import safe_open
    except ImportError:
        print("  safetensors not installed; cannot read tensor values.")
        return weight_map

    import numpy as np

    print(f"  found router tensors for {len(router_keys)} layers")
    print()
    print("    layer  key                                          shape       "
          "norm      std   row-spread  status")
    print("    " + "-" * 104)

    degenerate: list[int] = []
    for layer, keys in router_keys.items():
        for key in keys:
            shard = model_dir / weight_map[key]
            with safe_open(shard, framework="np") as handle:
                tensor = handle.get_tensor(key)
            values = tensor.astype(np.float32)
            norm = float(np.linalg.norm(values))
            std = float(values.std())
            # If every expert's row is the same, all logits tie regardless of
            # input, which is exactly what produces the 50/50 collapse.
            row_spread = float(np.linalg.norm(values - values.mean(axis=0, keepdims=True)))
            if norm == 0.0:
                status = "ALL ZERO -> logits tie"
                degenerate.append(layer)
            elif row_spread < 1e-6:
                status = "ROWS IDENTICAL -> logits tie"
                degenerate.append(layer)
            else:
                status = "ok"
            short = key if len(key) <= 44 else "..." + key[-41:]
            print(f"    {layer:5d}  {short:<44} {str(tuple(values.shape)):<11} "
                  f"{norm:8.3f} {std:8.5f} {row_spread:11.5f}  {status}")

    print()
    if degenerate:
        print(f"  {len(degenerate)} layers have degenerate router weights IN THE CHECKPOINT:")
        print(f"    {sorted(set(degenerate))}")
        print("  The collapse is in the downloaded files, not in the loading code.")
    else:
        print("  All router tensors in the checkpoint are non-degenerate.")
        print("  So the checkpoint is fine, and any collapse happened at LOAD time.")
        print("  Section [3] checks whether transformers can find these tensors.")
    return weight_map


def compare_expected_keys(model_dir: Path, weight_map: dict[str, str]) -> None:
    """Diff the model's expected parameter names against the checkpoint's names."""
    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:
        print(f"  transformers/torch unavailable ({exc}); skipping.")
        return

    try:
        import transformers

        print(f"  transformers {transformers.__version__}")
        config = AutoConfig.from_pretrained(model_dir)
        # Meta device: builds the module tree with no storage allocated at all.
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
    except Exception as exc:
        print(f"  could not build the model skeleton: {type(exc).__name__}: {exc}")
        return

    expected = set(model.state_dict().keys())
    present = set(weight_map.keys())

    # transformers renames legacy checkpoint keys on load via a regex mapping
    # (Mixtral moved block_sparse_moe -> mlp and fused the expert weights). A raw
    # name diff without it reports false mismatches, so apply it first.
    mapping = getattr(model, "_checkpoint_conversion_mapping", None) or {}
    if mapping:
        print(f"  applying {len(mapping)} checkpoint-conversion rules before diffing")
        remapped = set()
        for key in present:
            new_key = key
            for pattern, replacement in mapping.items():
                new_key = re.sub(pattern, replacement, new_key)
            remapped.add(new_key)
        present = remapped
    else:
        print("  NOTE: this model class declares no checkpoint-conversion mapping,")
        print("        so the diff below compares raw names.")

    # Tied embeddings and buffers legitimately differ; focus on router tensors.
    exp_routers = {k for k in expected if ROUTER_NAME_RE.search(k)}
    ckpt_routers = {k for k in present if ROUTER_NAME_RE.search(k)}

    print(f"  model expects {len(expected)} params; checkpoint index has {len(present)}")
    print(f"  router tensors -- model expects {len(exp_routers)}, "
          f"checkpoint has {len(ckpt_routers)}")

    missing_routers = sorted(exp_routers - ckpt_routers)
    if missing_routers:
        print()
        print(f"  {len(missing_routers)} ROUTER TENSORS THE MODEL EXPECTS ARE NOT IN THE")
        print("  CHECKPOINT. transformers will initialize these itself, so those")
        print("  layers route on untrained weights:")
        for key in missing_routers[:8]:
            print(f"    {key}")
        if len(missing_routers) > 8:
            print(f"    ... and {len(missing_routers) - 8} more")
    else:
        print("  Every router tensor the model expects is present in the checkpoint.")

    missing_any = sorted(expected - present)
    if missing_any:
        print()
        print(f"  {len(missing_any)} expected params absent from the checkpoint overall "
              f"(showing up to 10):")
        for key in missing_any[:10]:
            print(f"    {key}")
    unexpected = sorted(present - expected)
    if unexpected:
        print()
        print(f"  {len(unexpected)} checkpoint tensors the model does not expect "
              f"(showing up to 10):")
        for key in unexpected[:10]:
            print(f"    {key}")
        print("  Tensors here that look like routers mean a NAME MISMATCH: the weights")
        print("  exist but transformers is looking for them under a different name.")


def load_and_check(model_dir: Path, device_map: str) -> None:
    """Load the model for real and report what transformers did with the routers.

    This is the decisive test. ``output_loading_info=True`` returns the keys
    transformers could not fill from the checkpoint; anything router-shaped in
    that list was initialized by the model class itself and is untrained. The
    weight statistics that follow are read off the LOADED module, so they
    reflect what actually ran during profiling rather than what sits on disk.
    """
    import torch
    from transformers import AutoModelForCausalLM

    print(f"  loading {model_dir} (device_map={device_map}) ...")
    kwargs = {"device_map": device_map, "output_loading_info": True}
    try:
        model, info = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16, **kwargs
        )
    except TypeError:
        # transformers < 5 spells it torch_dtype.
        model, info = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, **kwargs
        )

    for label in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        keys = info.get(label) or []
        routerish = [k for k in keys if ROUTER_NAME_RE.search(k)]
        print(f"  {label}: {len(keys)}"
              + (f"  ({len(routerish)} of them router tensors)" if routerish else ""))
        for key in keys[:6]:
            print(f"      {key}")
        if len(keys) > 6:
            print(f"      ... and {len(keys) - 6} more")
        if label == "missing_keys" and routerish:
            print("      ^ THESE ROUTERS WERE NOT LOADED FROM THE CHECKPOINT.")
            print("        They hold whatever the model class initialized them to,")
            print("        which explains collapsed routing in exactly those layers.")

    print()
    print("    layer   norm      std   row-spread  status   (as loaded, not on disk)")
    print("    " + "-" * 68)
    bad: list[int] = []
    for name, param in model.named_parameters():
        if not ROUTER_NAME_RE.search(name):
            continue
        match = LAYER_RE.search(name)
        layer = int(match.group(1)) if match else -1
        values = param.detach().to(torch.float32).cpu()
        norm = float(values.norm())
        spread = float((values - values.mean(dim=0, keepdim=True)).norm())
        if norm == 0.0:
            status, ok = "ALL ZERO -> ties", False
        elif spread < 1e-6:
            status, ok = "ROWS IDENTICAL -> ties", False
        else:
            status, ok = "ok", True
        if not ok:
            bad.append(layer)
        print(f"    {layer:5d} {norm:8.3f} {float(values.std()):8.5f} {spread:11.5f}  {status}")

    print()
    if bad:
        print(f"  {len(bad)} layers have dead routers IN THE LOADED MODEL: {sorted(bad)}")
        print("  Compare against section [1]: if those same layers were fine on disk,")
        print("  the weights exist but transformers did not load them into the router.")
    else:
        print("  Every loaded router is non-degenerate. The routers are real, so the")
        print("  collapse observed in the trace is genuine routing behaviour.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", nargs="?", default="models/mixtral")
    parser.add_argument("--load", action="store_true",
                        help="also load the model and inspect the routers as loaded")
    parser.add_argument("--device-map", default="auto",
                        help="device_map for --load (default auto; use cpu to avoid GPUs)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"not a directory: {model_dir}")
        return 1

    print(f"\n[1] ROUTER TENSORS IN THE CHECKPOINT  --  {model_dir}")
    weight_map = inspect_checkpoint(model_dir)

    print(f"\n[2] MODEL SKELETON (meta device, no weights allocated)")
    if weight_map:
        compare_expected_keys(model_dir, weight_map)

    if args.load:
        print(f"\n[3] ROUTERS AS ACTUALLY LOADED  (decisive test)")
        try:
            load_and_check(model_dir, args.device_map)
        except Exception as exc:
            print(f"  load failed: {type(exc).__name__}: {exc}")
            return 1
    else:
        print("\n[3] ROUTERS AS ACTUALLY LOADED")
        print("    Skipped. This is the decisive test -- re-run with --load to")
        print("    compare the on-disk weights against what transformers loaded:")
        print(f"      python scripts/inspect_routers.py {model_dir} --load")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
