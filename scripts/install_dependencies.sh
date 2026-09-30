#!/usr/bin/env bash
# One-time installation into fresh source directories; activate the Python/CUDA
# environment first. Relative overrides are resolved from this project root.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SLIME_UPSTREAM_DIR=$(realpath -m "${SLIME_UPSTREAM_DIR:-../slime}")
MEGATRON_PATH=$(realpath -m "${MEGATRON_PATH:-$SLIME_UPSTREAM_DIR/Megatron-LM}")
SGLANG_PATH=$(realpath -m "${SGLANG_PATH:-$SLIME_UPSTREAM_DIR/sglang/python}")
SGLANG_DIR=$(dirname "$SGLANG_PATH")

# Refuse existing checkouts before cloning, patching, or installing anything.
# This installer intentionally does not resume partially completed installations.
for source_dir in "$SLIME_UPSTREAM_DIR" "$SGLANG_DIR" "$MEGATRON_PATH"; do
  if [[ -e "$source_dir" || -L "$source_dir" ]]; then
    echo "Installation requires a new checkout location: $source_dir" >&2
    exit 1
  fi
done

# Fetch the verified source baseline, including Megatron's submodules.
git clone https://github.com/THUDM/slime.git "$SLIME_UPSTREAM_DIR"
git -C "$SLIME_UPSTREAM_DIR" checkout 2418a7844abe4d490a1676dcf88fc3401adab362
git clone https://github.com/sgl-project/sglang.git "$SGLANG_DIR"
git -C "$SGLANG_DIR" checkout 24c91001cf99ba642be791e099d358f4dfe955f5
git clone https://github.com/NVIDIA/Megatron-LM.git "$MEGATRON_PATH"
git -C "$MEGATRON_PATH" checkout 3714d81d418c9f1bca4594fc35f9e8289f652862
git -C "$MEGATRON_PATH" submodule update --init --recursive

# Apply upstream slime's SGLang compatibility patch.
git -C "$SGLANG_DIR" apply < "$SLIME_UPSTREAM_DIR/docker/patch/v0.5.7/sglang.patch"
# Apply upstream slime's Megatron compatibility patch.
git -C "$MEGATRON_PATH" apply < "$SLIME_UPSTREAM_DIR/docker/patch/v0.5.7/megatron.patch"
# Evaluate after the final rollout, even outside the regular evaluation interval.
git -C "$SLIME_UPSTREAM_DIR" apply < patches/slime-final-eval.patch
# Use separate input embeddings and output weights for the Qwen2.5 student.
git -C "$SLIME_UPSTREAM_DIR" apply < patches/slime-qwen2.5-untied-embeddings.patch
# Enable temperature-scaled teacher scoring via SGLANG_TEMP_SCALED_LOGPROBS.
git -C "$SGLANG_DIR" apply < patches/sglang-temperature-scaled-logprobs.patch

# PyTorch and build tools must be installed before CUDA extensions are built.
python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r requirements-bootstrap.txt
python -m pip install -e "${SGLANG_PATH}[all]"

# These packages need installation flags that requirements.txt cannot express.
python -m pip install --no-deps \
  git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c
NVCC_APPEND_FLAGS='--threads 4' python -m pip install --no-build-isolation \
  --config-settings '--build-option=--cpp_ext --cuda_ext --parallel 8' \
  git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4
python -m pip install --no-cache-dir --force-reinstall \
  git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b

# Resolve the remaining requirements with Megatron so the listed pins apply to both.
MAX_JOBS=${MAX_JOBS:-8} python -m pip install --no-build-isolation \
  -r requirements.txt -e "$MEGATRON_PATH"
# Load this project's slime package while retaining upstream entry points/plugins.
python -m pip install --no-deps -e .
