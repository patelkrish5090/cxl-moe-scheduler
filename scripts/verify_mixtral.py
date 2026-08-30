"""Inspect a downloaded Mixtral directory: is it complete, and what is junk?

    python scripts/verify_mixtral.py models/mixtral

Downloads nothing and loads nothing. It answers four questions:

1. Where does transformers actually need to be pointed? (An HF cache layout puts
   the real files under snapshots/<sha>/ with symlinks into blobs/, so the top
   directory is not always the right path.)
2. Are all 19 safetensors shards present and the right size?
3. How much of the disk usage is the duplicate `consolidated.0N.pt` reference
   weights, which transformers never reads and which can be deleted?
4. Do the shards parse as safetensors, without loading any tensor data?

Reported sizes are apparent file sizes. Symlinked files are counted once, at
their target's size, so the total matches what `du -L --exclude` would say
rather than what `ls -l` on the symlinks suggests.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

GIB = 1024 ** 3

# What transformers loads for Mixtral-8x7B-v0.1, from the repo file listing.
EXPECTED_SHARDS = 19
EXPECTED_SHARD_GIB = 87.0
REQUIRED_SMALL_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def resolve_model_dir(root: Path) -> Path:
    """Find the directory holding config.json, descending an HF cache layout."""
    if (root / "config.json").exists():
        return root
    snapshots = root / "snapshots"
    if snapshots.is_dir():
        candidates = [d for d in snapshots.iterdir() if (d / "config.json").exists()]
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            print(f"  NOTE: {len(candidates)} snapshots found; using the newest")
            return max(candidates, key=lambda d: d.stat().st_mtime)
    # A single nested directory, e.g. models/mixtral/Mixtral-8x7B-v0.1/
    nested = [d for d in root.iterdir() if d.is_dir() and (d / "config.json").exists()]
    if len(nested) == 1:
        return nested[0]
    return root


def real_size(path: Path) -> int:
    """Apparent size of a file, following symlinks (HF cache points into blobs/)."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def categorise(model_dir: Path) -> dict[str, list[tuple[Path, int]]]:
    groups: dict[str, list[tuple[Path, int]]] = {
        "safetensors": [], "consolidated": [], "small": [], "other": []
    }
    for path in sorted(model_dir.rglob("*")):
        if path.is_dir():
            continue
        size = real_size(path)
        name = path.name
        if name.endswith(".safetensors"):
            groups["safetensors"].append((path, size))
        elif name.startswith("consolidated.") or name.endswith(".pt") or name.endswith(".bin"):
            groups["consolidated"].append((path, size))
        elif name.endswith((".json", ".model", ".md", ".txt")):
            groups["small"].append((path, size))
        else:
            groups["other"].append((path, size))
    return groups


def check_safetensors_header(path: Path) -> str:
    """Read only the 8-byte length prefix and the JSON header. No tensor data."""
    try:
        with path.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) < 8:
                return "truncated (under 8 bytes)"
            header_len = struct.unpack("<Q", raw)[0]
            if header_len == 0 or header_len > 200_000_000:
                return f"implausible header length {header_len}"
            header = handle.read(header_len)
            if len(header) < header_len:
                return "truncated (header incomplete)"
            meta = json.loads(header)
            n_tensors = len([k for k in meta if k != "__metadata__"])
            # The last tensor's end offset plus the header tells us the true size.
            end = max(
                (v["data_offsets"][1] for k, v in meta.items() if k != "__metadata__"),
                default=0,
            )
            expected = 8 + header_len + end
            actual = path.stat().st_size
            if actual != expected:
                short = (expected - actual) / GIB
                return f"INCOMPLETE: {actual/GIB:.2f} GiB on disk, {expected/GIB:.2f} GiB expected ({short:+.2f} GiB)"
            return f"ok ({n_tensors} tensors)"
    except Exception as exc:
        return f"unreadable: {type(exc).__name__}: {exc}"


MIXTRAL_CONFIGS = ("configs/mixtral_8x7b.json", "configs/mixtral_8x7b_decode.json")


def repoint_configs(model_dir: Path) -> None:
    """Rewrite model.name_or_path in the mixtral run configs to a local path.

    Edits the JSON in place, preserving every other field. An absolute path is
    written so the config works regardless of the shell's working directory.
    """
    target = str(model_dir.resolve())
    repo_root = Path(__file__).resolve().parents[1]
    for rel in MIXTRAL_CONFIGS:
        path = repo_root / rel
        if not path.exists():
            print(f"  skipped (not found): {rel}")
            continue
        config = json.loads(path.read_text(encoding="utf-8"))
        before = config["model"].get("name_or_path")
        if before == target:
            print(f"  unchanged: {rel}")
            continue
        config["model"]["name_or_path"] = target
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(f"  {rel}: {before} -> {target}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="models/mixtral")
    parser.add_argument("--headers", action="store_true",
                        help="parse every shard header (a few seconds, no tensor reads)")
    parser.add_argument("--write-config", action="store_true",
                        help="rewrite the mixtral run configs to load from this directory")
    args = parser.parse_args()

    root = Path(args.path).expanduser()
    if not root.is_dir():
        print(f"not a directory: {root}")
        return 1

    print(f"\n[1] LOCATION")
    print(f"  given      : {root}")
    model_dir = resolve_model_dir(root)
    print(f"  model dir  : {model_dir}")
    if not (model_dir / "config.json").exists():
        print("  config.json NOT FOUND -- the download is incomplete or the path is wrong.")
        print("  Files found at the top level:")
        for path in sorted(root.iterdir())[:20]:
            print(f"    {path.name}")
        return 1

    groups = categorise(model_dir)
    shard_bytes = sum(s for _, s in groups["safetensors"])
    junk_bytes = sum(s for _, s in groups["consolidated"])
    small_bytes = sum(s for _, s in groups["small"]) + sum(s for _, s in groups["other"])
    total = shard_bytes + junk_bytes + small_bytes

    print(f"\n[2] WHAT IS ON DISK")
    print(f"  safetensors shards : {len(groups['safetensors']):>3} files  {shard_bytes/GIB:8.1f} GiB   <- transformers loads these")
    print(f"  consolidated/.pt   : {len(groups['consolidated']):>3} files  {junk_bytes/GIB:8.1f} GiB   <- NOT used, safe to delete")
    print(f"  config/tokenizer   : {len(groups['small']) + len(groups['other']):>3} files  {small_bytes/GIB:8.2f} GiB")
    print(f"  total (apparent)   : {total/GIB:8.1f} GiB")
    for path, size in groups["consolidated"]:
        print(f"      {path.name:<32} {size/GIB:6.1f} GiB")

    print(f"\n[3] COMPLETENESS")
    index_path = model_dir / "model.safetensors.index.json"
    missing: list[str] = []
    for name in REQUIRED_SMALL_FILES:
        if not (model_dir / name).exists():
            missing.append(name)
    if missing:
        print(f"  MISSING required files: {missing}")

    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
        absent = [s for s in shards if not (model_dir / s).exists()]
        print(f"  index lists {len(shards)} shards (expect {EXPECTED_SHARDS})")
        print(f"  present    : {len(shards) - len(absent)}/{len(shards)}")
        if absent:
            print(f"  MISSING SHARDS: {absent}")
            missing.extend(absent)
        print(f"  shard bytes: {shard_bytes/GIB:.1f} GiB (expect ~{EXPECTED_SHARD_GIB:.1f} GiB)")
        size_is_off = abs(shard_bytes / GIB - EXPECTED_SHARD_GIB) > 2.0 and not absent
        if size_is_off:
            print("  WARNING: total shard size is off by more than 2 GiB even though")
            print("           every shard file exists -- some are probably truncated.")
            if not args.headers:
                # Do not call a suspect download complete. The header check is
                # authoritative, so demand it rather than trusting the total.
                print("           Re-run with --headers to find which.")
                missing.append("size mismatch, unverified (re-run with --headers)")

        if args.headers:
            print(f"\n[3b] SHARD HEADERS")
            bad = 0
            for shard in shards:
                path = model_dir / shard
                if not path.exists():
                    continue
                status = check_safetensors_header(path)
                if not status.startswith("ok"):
                    bad += 1
                    print(f"  {shard:<40} {status}")
            print(f"  {len(shards) - bad}/{len(shards)} shards parse cleanly")
            if bad:
                missing.append(f"{bad} truncated shard(s)")
    else:
        print("  model.safetensors.index.json missing -- cannot verify shard set.")
        missing.append("model.safetensors.index.json")

    print(f"\n[4] NEXT STEPS")
    if missing:
        print(f"  Download is INCOMPLETE: {missing}")
        print(f"  Resume it (hf download skips what is already correct):")
        print(f"    bash scripts/download_mixtral.sh {model_dir} hf")
        return 1

    print("  Download is complete and usable.")
    if junk_bytes:
        print(f"\n  Reclaim {junk_bytes/GIB:.0f} GiB -- these are Mistral's reference-format")
        print("  weights, a second copy of the same parameters that transformers never")
        print("  opens. Delete them with:")
        print(f"    rm -f {model_dir}/consolidated.*.pt")
    if args.write_config:
        print("\n  Repointing run configs at the local directory:")
        repoint_configs(model_dir)
    else:
        print(f"\n  Point the run configs at the local directory (no re-download):")
        print(f"    python scripts/verify_mixtral.py {root} --write-config")
        print(f"  or edit configs/mixtral_8x7b.json and configs/mixtral_8x7b_decode.json by hand:")
        print(f'    "name_or_path": "{model_dir}"')
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
