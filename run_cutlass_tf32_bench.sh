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
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
  NVCC=/usr/local/cuda/bin/nvcc
else
  NVCC="$(command -v nvcc)"
fi

CUTLASS_DIR="${CUTLASS_DIR:-/SharedData/models/data/.deps/cutlass}"
if [[ ! -d "${CUTLASS_DIR}/include" ]]; then
  echo "CUTLASS_DIR does not point to a CUTLASS checkout: ${CUTLASS_DIR}" >&2
  exit 1
fi

OUT="${CUTLASS_TF32_OUT:-${SCRIPT_DIR}/.tmp/cutlass_tf32_bench}"
ARCH="${CUTLASS_TF32_ARCH:-sm_90a}"

"${NVCC}" -O3 -std=c++17 -arch="${ARCH}" -lineinfo \
  --expt-relaxed-constexpr \
  -Xcudafe --diag_suppress=20012 \
  -Xcudafe --diag_suppress=20013 \
  -Xcudafe --diag_suppress=20015 \
  --ptxas-options=-v \
  -I"${CUTLASS_DIR}/include" \
  -I"${CUTLASS_DIR}/tools/util/include" \
  "${SCRIPT_DIR}/cutlass_tf32_bench.cu" -lcublas -o "${OUT}"

"${OUT}" "$@"
