#!/usr/bin/env bash
# run like `source setup.sh` on any remote to ensure active shell is set up

# --- 1. System Setup ---
cp ~/.env .env
set -a; source .env; set +a

mkdir -p data

source $HOME/.local/bin/env
if [[ -n "$SSH_CONNECTION" && -d /workspace ]]; then
  echo "🐧 Running on remote runpod with storage attached - setting custom hf cache dir"
  set -a
  export HF_HOME="/workspace/.cache"
  export HF_DATASETS_CACHE="/dev/shm/.cache/datasets_cache"
  export PIP_CACHE_DIR=/workspace/pip_cache
  mkdir -p $HF_HOME
  mkdir -p $HF_DATASETS_CACHE
  mkdir -p $PIP_CACHE_DIR
  set +a
fi

echo "🔍 Detecting system package versions..."

# Define the path to the system's Python executable
PY_SYSTEM="/usr/bin/python3.11"

# Execute Python commands to get the exact version strings
TORCH_VERSION=$($PY_SYSTEM -c "import torch; print(torch.__version__)")
NUMPY_VERSION=$($PY_SYSTEM -c "import numpy; print(numpy.__version__)")

# Check that the commands succeeded
if [[ -z "$TORCH_VERSION" || -z "$NUMPY_VERSION" ]]; then
  echo "❌ Error: Failed to detect torch or numpy. Ensure they are installed in the system's Python."
  exit 1
fi

echo "  ✅ Detected Torch: $TORCH_VERSION"
echo "  ✅ Detected NumPy: $NUMPY_VERSION"

# Write the collected versions into the constraints.txt file
cat > constraints.txt << EOL
# This file is auto-generated to constrain torch and numpy to the system versions.
torch==$TORCH_VERSION
numpy==$NUMPY_VERSION
EOL

echo "🎉 Success! 'constraints.txt' has been created."
#pip install -r requirements.txt -c constraints.txt
# or ./setup_5090.sh on 5090