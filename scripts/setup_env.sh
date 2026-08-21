#!/usr/bin/env bash
# Create an isolated conda env for this project.
#
#   bash scripts/setup_env.sh
#   conda activate astera
#
# Why a NEW env rather than installing into `teto`: the probe showed `teto` has
# numpy 2.5.2 / pandas 3.0.5 and is missing transformers, datasets, accelerate,
# safetensors and pyarrow. Installing those can make pip resolve numpy/pandas
# downwards, and something is currently occupying ~88 GB on GPU 0 -- breaking
# that job's env would be an expensive way to save five minutes.
#
# torch is installed for CUDA 12.8 to match the probed `torch 2.11.0+cu128`
# and the sm_120 Blackwell cards.

set -euo pipefail
cd "$(dirname "$0")/.."

ENV_NAME="${ENV_NAME:-astera}"
PY_VERSION="${PY_VERSION:-3.12}"

# --- TLS interception preflight -------------------------------------------
# This network runs a Sophos firewall that terminates and re-signs HTTPS. The
# system CA store can be taught to trust it (see profiler/README.md), but Python
# does NOT use the system store -- pip, requests and huggingface_hub all use
# certifi's own bundle, so they keep failing after `update-ca-certificates`
# alone. Setting these makes every Python HTTPS client use the system bundle.
CA_BUNDLE="${ASTERA_CA_BUNDLE:-/etc/ssl/certs/ca-certificates.crt}"
if [ -f "$CA_BUNDLE" ]; then
  export SSL_CERT_FILE="$CA_BUNDLE"
  export REQUESTS_CA_BUNDLE="$CA_BUNDLE"
  export CURL_CA_BUNDLE="$CA_BUNDLE"
  echo "using CA bundle: $CA_BUNDLE"
else
  echo "WARNING: CA bundle not found at $CA_BUNDLE; TLS verification may fail." >&2
fi

if ! curl -fsS --max-time 20 https://pypi.org/simple/ -o /dev/null 2>/dev/null; then
  cat >&2 <<'MSG'

WARNING: cannot verify a TLS connection to pypi.org.

If this is the Sophos inspection proxy, install its root certificate first:

  openssl s_client -showcerts -connect github.com:443 </dev/null 2>/dev/null |
    awk '/-----BEGIN CERTIFICATE-----/{c++} c==2{print} /-----END CERTIFICATE-----/{if(c==2) exit}' > /tmp/sophos-ca.crt
  sudo cp /tmp/sophos-ca.crt /usr/local/share/ca-certificates/sophos-ca.crt
  sudo update-ca-certificates
  pip config set global.cert /etc/ssl/certs/ca-certificates.crt

Verify the certificate's SHA-256 fingerprint with your network admin before
trusting it. Continuing anyway -- the install below will likely fail.

MSG
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "conda env '$ENV_NAME' already exists; installing into it."
else
  echo "creating conda env '$ENV_NAME' (python $PY_VERSION)"
  conda create -y -n "$ENV_NAME" "python=$PY_VERSION"
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo
echo "==> installing torch (cu128, matching the probed driver 570.211.01 / sm_120)"
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128

echo
echo "==> installing project dependencies"
python -m pip install -r requirements.txt

echo
echo "==> verifying"
python - <<'PY'
import torch, transformers, datasets, pyarrow, numpy, pandas, accelerate
print(f"torch        {torch.__version__}  cuda_avail={torch.cuda.is_available()}")
print(f"transformers {transformers.__version__}")
print(f"datasets     {datasets.__version__}")
print(f"accelerate   {accelerate.__version__}")
print(f"pyarrow      {pyarrow.__version__}")
print(f"numpy        {numpy.__version__}")
print(f"pandas       {pandas.__version__}")
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f"  gpu{i}: {torch.cuda.get_device_properties(i).name}  "
          f"{free/1e9:.1f} GB free / {total/1e9:.1f} GB")
PY

echo
echo "Done. Activate with:  conda activate $ENV_NAME"
