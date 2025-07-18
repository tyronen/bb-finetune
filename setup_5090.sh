pip uninstall flash-attn flash_attn -y

cd /workspace/flash-attention

# 2  Tell the build which GPU architecture to target
#    (compute 12.0 ➜ sm_120; change if your `torch.cuda.get_device_capability(0)`
#     reports something else)
export FORCE_CUDA_ARCH=120
# optional but nice: keep Torch’s JIT in sync
export TORCH_CUDA_ARCH_LIST="12.0"
export FLASH_ATTENTION_FORCE_BUILD=TRUE   # skip the 404 wheel fetch
export FLASH_ATTN_CUDA_ARCHS=120          # just Blackwell
export MAX_JOBS=8                         # ninja job slots  (keeps RAM down)
export NVCC_THREADS=8                     # threads per nvcc compile
export NINJAFLAGS="-j8"
export TMPDIR=/dev/shm
#fallocate -l 32G /swap && mkswap /swap && swapon /swap
# 3  Build & install against the Torch that’s already in the venv
# python -m pip install -v --no-build-isolation .

# 4  Quick import test
#python - <<'PY'
#import torch, flash_attn_cuda
#print("Flash-Attention OK on", torch.version.cuda,
#      "with", torch.cuda.get_device_name(0))
#PY