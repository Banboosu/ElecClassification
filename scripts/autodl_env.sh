#!/usr/bin/env bash
# Source this from a shell; does not start training or enable test evaluation.
export PATH="/root/.local/bin:/root/miniconda3/bin:${PATH}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/autodl-tmp/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/root/autodl-tmp/.uv-python}"
export UV_LINK_MODE="${UV_LINK_MODE:-hardlink}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/.huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
