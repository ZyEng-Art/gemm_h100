#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${TMPDIR:-}" ]]; then
  export TMPDIR="${SCRIPT_DIR}/.tmp"
fi
mkdir -p "${TMPDIR}"

if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
  NVCC="${CUDA_HOME}/bin/nvcc"
elif [[ -x /usr/local/cuda-12.9/bin/nvcc ]]; then
  NVCC=/usr/local/cuda-12.9/bin/nvcc
elif [[ -x /usr/local/cuda-13.0/bin/nvcc ]]; then
  NVCC=/usr/local/cuda-13.0/bin/nvcc
elif command -v nvcc >/dev/null 2>&1; then
  NVCC="$(command -v nvcc)"
else
  echo "nvcc not found. Set CUDA_HOME or add nvcc to PATH." >&2
  exit 1
fi

OUT="${SCRIPT_DIR}/gemm_bench"

"${NVCC}" -O3 -std=c++17 -arch=sm_90 -lineinfo \
  "${SCRIPT_DIR}/gemm.cu" -lcublas -o "${OUT}"

"${OUT}" "$@"
