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

ARCH="${TENSOR_GEMM_ARCH:-sm_90}"
OUT="${TENSOR_GEMM_OUT:-${SCRIPT_DIR}/.tmp/tensor_gemm_bench}"

has_dtype=0
run_args=()
for arg in "$@"; do
  if [[ "${arg}" == "--dtype" ]]; then
    has_dtype=1
    run_args+=("${arg}")
  elif [[ "${arg}" == --dtype=* ]]; then
    has_dtype=1
    run_args+=(--dtype "${arg#--dtype=}")
  else
    run_args+=("${arg}")
  fi
done

if [[ "${has_dtype}" -eq 0 ]]; then
  run_args=(--dtype tf32 "${run_args[@]}")
fi

"${NVCC}" -O3 -std=c++17 -arch="${ARCH}" -lineinfo \
  --ptxas-options=-v \
  "${SCRIPT_DIR}/tensor_gemm_bench.cu" -lcublas -o "${OUT}"

"${OUT}" "${run_args[@]}"
