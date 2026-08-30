#!/usr/bin/env bash
# Download Mixtral-8x7B weights, skipping the 90 GiB of duplicate files.
#
#   bash scripts/download_mixtral.sh [dest_dir] [method]
#
#   dest_dir  where to put the model      (default: ./models/Mixtral-8x7B-v0.1)
#   method    hf | aria2 | wget | urls    (default: hf)
#
# The repo carries the weights in TWO formats totalling 177 GiB:
#   * model-000NN-of-00019.safetensors  -- 87.0 GiB, what transformers loads
#   * consolidated.0N.pt                -- 90.4 GiB, Mistral's reference format
# Only the safetensors set is needed. Every method below excludes the rest.
#
# The repo is NOT gated: no HF token and no terms acceptance are required.
# A token still helps with rate limits on a shared IP.
#
# After it finishes, point the run config at the directory:
#   configs/mixtral_8x7b.json -> model.name_or_path = "<dest_dir>"

set -euo pipefail
cd "$(dirname "$0")/.."

REPO="mistralai/Mixtral-8x7B-v0.1"
DEST="${1:-./models/Mixtral-8x7B-v0.1}"
METHOD="${2:-hf}"
BASE="https://huggingface.co/${REPO}/resolve/main"

# Everything transformers needs, and nothing else.
SMALL_FILES=(
  config.json
  generation_config.json
  model.safetensors.index.json
  special_tokens_map.json
  tokenizer.json
  tokenizer.model
  tokenizer_config.json
)

build_url_list() {
  local out="$1"
  : > "$out"
  for f in "${SMALL_FILES[@]}"; do
    echo "${BASE}/${f}" >> "$out"
  done
  for i in $(seq -w 1 19); do
    echo "${BASE}/model-000${i}-of-00019.safetensors" >> "$out"
  done
  # seq -w pads to 2 digits; fix the file naming to the repo's 5-digit form.
  sed -i 's|model-000\([0-9]\{2\}\)-of-00019|model-000\1-of-00019|' "$out"
}

mkdir -p "$DEST"

case "$METHOD" in
  hf)
    # Preferred: resumable, verifies hashes, handles the CA bundle via env vars.
    if command -v hf >/dev/null 2>&1; then
      HF_CMD="hf download"
    elif command -v huggingface-cli >/dev/null 2>&1; then
      HF_CMD="huggingface-cli download"
    else
      echo "neither 'hf' nor 'huggingface-cli' found; pip install -U huggingface_hub" >&2
      exit 1
    fi
    echo "==> $HF_CMD $REPO -> $DEST  (excluding consolidated.*)"
    $HF_CMD "$REPO" \
      --exclude "consolidated.*" \
      --exclude "*.pt" \
      --local-dir "$DEST"
    ;;

  aria2)
    # Fastest over a throttled link: 16 parallel connections per file.
    command -v aria2c >/dev/null 2>&1 || { echo "aria2c not installed" >&2; exit 1; }
    LIST="$(mktemp)"
    build_url_list "$LIST"
    echo "==> aria2c downloading $(wc -l < "$LIST") files into $DEST"
    aria2c -x 16 -s 16 -k 10M -c -d "$DEST" -i "$LIST"
    rm -f "$LIST"
    ;;

  wget)
    LIST="$(mktemp)"
    build_url_list "$LIST"
    echo "==> wget downloading $(wc -l < "$LIST") files into $DEST"
    wget -c -P "$DEST" -i "$LIST"
    rm -f "$LIST"
    ;;

  urls)
    # Just print the URLs, for downloading on another machine entirely.
    LIST="$(mktemp)"
    build_url_list "$LIST"
    cat "$LIST"
    rm -f "$LIST"
    exit 0
    ;;

  *)
    echo "unknown method: $METHOD (use hf | aria2 | wget | urls)" >&2
    exit 1
    ;;
esac

echo
echo "==> verifying"
python - "$DEST" <<'PY'
import json
import sys
from pathlib import Path

dest = Path(sys.argv[1])
index = dest / "model.safetensors.index.json"
if not index.exists():
    print(f"MISSING: {index}")
    raise SystemExit(1)

shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
missing, total = [], 0
for shard in shards:
    path = dest / shard
    if not path.exists():
        missing.append(shard)
    else:
        total += path.stat().st_size

for name in ("config.json", "tokenizer_config.json"):
    if not (dest / name).exists():
        missing.append(name)

print(f"shards present : {len(shards) - len([m for m in missing if m.endswith('.safetensors')])}/{len(shards)}")
print(f"bytes on disk  : {total / 1024**3:.1f} GiB (expect ~87.0 GiB)")
if missing:
    print(f"MISSING: {missing}")
    raise SystemExit(1)
print("complete.")
print()
print("Next: point the run config at this directory, e.g.")
print(f'  jq \'.model.name_or_path = "{dest}"\' configs/mixtral_8x7b.json > /tmp/m.json \\')
print("    && mv /tmp/m.json configs/mixtral_8x7b.json")
PY
