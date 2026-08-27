#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BIN="${SCRIPT_DIR}/gemm_bench"

if [[ -z "${TMPDIR:-}" ]]; then
  export TMPDIR="${SCRIPT_DIR}/.tmp"
fi
mkdir -p "${TMPDIR}"

SHAPE="4096x4096x4096"
BENCH_WARMUP=3
BENCH_REPEAT=10
NSYS_WARMUP=1
NSYS_REPEAT=3
NCU_WARMUP=0
NCU_REPEAT=1
NCU_LAUNCH_COUNT=1
KERNEL_NAME="regex:.*gemm_kernel.*"
PROFILE_CUBLAS=0
PROFILE_CHECK=0
BUILD=1
RUN_BENCH=1
OUT_DIR=""
PROFILE_TMPDIR=""
FINAL_RC=0
NCU_SET=""
NCU_SECTIONS=(
  "SpeedOfLight"
  "SpeedOfLight_RooflineChart"
  "LaunchStats"
  "Occupancy"
  "MemoryWorkloadAnalysis"
  "ComputeWorkloadAnalysis"
)
COMMON_ARGS=()
NSYS_EXTRA=()
NCU_EXTRA=()

usage() {
  cat <<'EOF'
Usage: ./run_gemm_profile.sh [options] [-- extra gemm_bench args...]

Build gemm.cu, run a normal benchmark, then collect both Nsight Systems
and Nsight Compute reports for the selected GEMM shape.

Default profile target:
  ./gemm_bench --shape 4096x4096x4096 --no-check --no-cublas

Options:
  --shape MxNxK             GEMM shape to profile, default 4096x4096x4096
  --out-dir DIR             Output directory, default results/profile/YYYYmmdd_HHMMSS
  --profile-tmpdir DIR      Temporary directory for nsys/ncu, default /dev/shm/$USER_gemm_profile_$$

  --bench-warmup N          Warmup launches for the normal benchmark, default 3
  --bench-repeat N          Timed launches for the normal benchmark, default 10
  --skip-bench              Skip the normal benchmark run

  --nsys-warmup N           Warmup launches inside nsys run, default 1
  --nsys-repeat N           Timed launches inside nsys run, default 3
  --ncu-warmup N            Warmup launches inside ncu run, default 0
  --ncu-repeat N            Timed launches inside ncu run, default 1
  --ncu-launch-count N      Number of matching kernel launches ncu profiles, default 1

  --kernel-name EXPR        ncu kernel filter, default regex:.*gemm_kernel.*
  --profile-cublas          Include cuBLAS timed kernels in profile runs
  --profile-check           Include correctness check in profile runs
  --no-build                Reuse existing ./gemm_bench

  --ncu-set SET             Use an ncu set instead of default sections, for example full
  --ncu-section SECTION     Add one extra ncu section
  --ncu-extra ARG           Add one raw argument to ncu; repeat for multiple args
  --nsys-extra ARG          Add one raw argument to nsys; repeat for multiple args

Common gemm_bench options accepted directly:
  --device ID
  --peak TFLOPS
  --peak-mode fp32|tf32|fp16|bf16|fp8
  --hbm-tbps TBPS
  --atol X
  --rtol X
  --cublas-fast-tf32

Examples:
  ./run_gemm_profile.sh
  ./run_gemm_profile.sh --shape 2048x2048x2048
  ./run_gemm_profile.sh --shape 4096x4096x4096 --peak-mode tf32 --cublas-fast-tf32
  ./run_gemm_profile.sh --ncu-set full --ncu-launch-count 1
  ./run_gemm_profile.sh --ncu-extra --metrics --ncu-extra sm__warps_active.avg.pct_of_peak_sustained_active
EOF
}

find_nvcc() {
  if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    printf '%s\n' "${CUDA_HOME}/bin/nvcc"
    return 0
  fi
  local p
  for p in \
    /usr/local/cuda-12.9/bin/nvcc \
    /usr/local/cuda-13.0/bin/nvcc \
    /usr/local/cuda/bin/nvcc; do
    if [[ -x "${p}" ]]; then
      printf '%s\n' "${p}"
      return 0
    fi
  done
  command -v nvcc 2>/dev/null
}

find_nsys() {
  local p
  for p in \
    /usr/local/cuda-12.9/bin/nsys \
    /usr/local/cuda-13.0/bin/nsys \
    /usr/local/cuda/bin/nsys \
    /usr/local/cuda-12.9/nsight-systems-*/target-linux-x64/nsys \
    /usr/local/cuda-13.0/nsight-systems-*/target-linux-x64/nsys; do
    if [[ -x "${p}" ]]; then
      printf '%s\n' "${p}"
      return 0
    fi
  done
  command -v nsys 2>/dev/null
}

find_ncu() {
  local p
  for p in \
    /usr/local/cuda-12.9/bin/ncu \
    /usr/local/cuda-13.0/bin/ncu \
    /usr/local/cuda/bin/ncu \
    /usr/local/cuda-12.9/nsight-compute-*/ncu \
    /usr/local/cuda-13.0/nsight-compute-*/ncu; do
    if [[ -x "${p}" ]]; then
      printf '%s\n' "${p}"
      return 0
    fi
  done
  command -v ncu 2>/dev/null
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --shape)
      SHAPE="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --profile-tmpdir)
      PROFILE_TMPDIR="$2"
      shift 2
      ;;
    --bench-warmup)
      BENCH_WARMUP="$2"
      shift 2
      ;;
    --bench-repeat)
      BENCH_REPEAT="$2"
      shift 2
      ;;
    --skip-bench)
      RUN_BENCH=0
      shift
      ;;
    --nsys-warmup)
      NSYS_WARMUP="$2"
      shift 2
      ;;
    --nsys-repeat)
      NSYS_REPEAT="$2"
      shift 2
      ;;
    --ncu-warmup)
      NCU_WARMUP="$2"
      shift 2
      ;;
    --ncu-repeat)
      NCU_REPEAT="$2"
      shift 2
      ;;
    --ncu-launch-count)
      NCU_LAUNCH_COUNT="$2"
      shift 2
      ;;
    --kernel-name)
      KERNEL_NAME="$2"
      shift 2
      ;;
    --profile-cublas)
      PROFILE_CUBLAS=1
      shift
      ;;
    --profile-check)
      PROFILE_CHECK=1
      shift
      ;;
    --no-build)
      BUILD=0
      shift
      ;;
    --ncu-set)
      NCU_SET="$2"
      shift 2
      ;;
    --ncu-section)
      NCU_SECTIONS+=("$2")
      shift 2
      ;;
    --ncu-extra)
      NCU_EXTRA+=("$2")
      shift 2
      ;;
    --nsys-extra)
      NSYS_EXTRA+=("$2")
      shift 2
      ;;
    --device|--peak|--peak-mode|--hbm-tbps|--atol|--rtol)
      COMMON_ARGS+=("$1" "$2")
      shift 2
      ;;
    --cublas-fast-tf32)
      COMMON_ARGS+=("$1")
      shift
      ;;
    --)
      shift
      COMMON_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="${SCRIPT_DIR}/results/profile/$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${OUT_DIR}"

if [[ -z "${PROFILE_TMPDIR}" ]]; then
  if [[ -d /dev/shm && -w /dev/shm ]]; then
    PROFILE_TMPDIR="/dev/shm/${USER:-user}_gemm_profile_$$"
  else
    PROFILE_TMPDIR="${SCRIPT_DIR}/.tmp/profile_$$"
  fi
fi
mkdir -p "${PROFILE_TMPDIR}"
chmod 700 "${PROFILE_TMPDIR}" 2>/dev/null || true

NVCC="$(find_nvcc || true)"
NSYS="$(find_nsys || true)"
NCU="$(find_ncu || true)"

if [[ -z "${NSYS}" ]]; then
  echo "nsys not found. Add it to PATH or install Nsight Systems." >&2
  exit 1
fi
if [[ -z "${NCU}" ]]; then
  echo "ncu not found. Add it to PATH or install Nsight Compute." >&2
  exit 1
fi
if [[ "${BUILD}" -eq 1 && -z "${NVCC}" ]]; then
  echo "nvcc not found. Set CUDA_HOME or add nvcc to PATH." >&2
  exit 1
fi
if [[ "${BUILD}" -eq 0 && ! -x "${BIN}" ]]; then
  echo "${BIN} does not exist or is not executable. Remove --no-build." >&2
  exit 1
fi

{
  echo "script_dir=${SCRIPT_DIR}"
  echo "out_dir=${OUT_DIR}"
  echo "shape=${SHAPE}"
  echo "nvcc=${NVCC:-<skipped>}"
  echo "nsys=${NSYS}"
  echo "ncu=${NCU}"
  echo "tmpdir=${TMPDIR}"
  echo "profile_tmpdir=${PROFILE_TMPDIR}"
  date --iso-8601=seconds
  echo
  df -h /tmp "${PROFILE_TMPDIR}" "${SCRIPT_DIR}" 2>/dev/null || true
} | tee "${OUT_DIR}/run_info.txt"

if [[ "${BUILD}" -eq 1 ]]; then
  echo
  echo "== Build =="
  "${NVCC}" -O3 -std=c++17 -arch=sm_90 -lineinfo \
    "${SCRIPT_DIR}/gemm.cu" -lcublas -o "${BIN}" \
    2>&1 | tee "${OUT_DIR}/build.txt"
fi

if [[ "${RUN_BENCH}" -eq 1 ]]; then
  echo
  echo "== Benchmark =="
  bench_args=(--shape "${SHAPE}" --warmup "${BENCH_WARMUP}" --repeat "${BENCH_REPEAT}" "${COMMON_ARGS[@]}")
  {
    printf 'Command:'
    printf ' %q' "${BIN}" "${bench_args[@]}"
    echo
  } | tee "${OUT_DIR}/benchmark_command.txt"
  "${BIN}" "${bench_args[@]}" 2>&1 | tee "${OUT_DIR}/benchmark.txt"
fi

make_profile_args() {
  local warmup="$1"
  local repeat="$2"
  PROFILE_ARGS=(--shape "${SHAPE}" --warmup "${warmup}" --repeat "${repeat}" "${COMMON_ARGS[@]}")
  if [[ "${PROFILE_CHECK}" -eq 0 ]]; then
    PROFILE_ARGS+=(--no-check)
  fi
  if [[ "${PROFILE_CUBLAS}" -eq 0 ]]; then
    PROFILE_ARGS+=(--no-cublas)
  fi
}

echo
echo "== Nsight Systems =="
make_profile_args "${NSYS_WARMUP}" "${NSYS_REPEAT}"
nsys_base="${OUT_DIR}/nsys_gemm"
nsys_cmd=(
  env
  "TMPDIR=${PROFILE_TMPDIR}"
  "TMP=${PROFILE_TMPDIR}"
  "TEMP=${PROFILE_TMPDIR}"
  "NSYS_AGENT_TMP_DIR=${PROFILE_TMPDIR}"
  "${NSYS}" profile
  --force-overwrite=true
  --trace=cuda,nvtx,osrt
  --sample=none
  --stats=true
  -o "${nsys_base}"
  "${NSYS_EXTRA[@]}"
  "${BIN}"
  "${PROFILE_ARGS[@]}"
)
{
  printf 'Command:'
  printf ' %q' "${nsys_cmd[@]}"
  echo
} | tee "${OUT_DIR}/nsys_command.txt"
set +e
"${nsys_cmd[@]}" 2>&1 | tee "${OUT_DIR}/nsys_stdout.txt"
nsys_rc=${PIPESTATUS[0]}
set -e

if [[ "${nsys_rc}" -ne 0 ]]; then
  FINAL_RC=1
  echo
  echo "nsys failed with exit code ${nsys_rc}; see ${OUT_DIR}/nsys_stdout.txt" >&2
  if grep -q "Failed to write CUDA config file" "${OUT_DIR}/nsys_stdout.txt"; then
    echo "diagnosis: nsys failed while writing its CUDA injection config." >&2
    echo "diagnosis: this host currently has /tmp on a full root filesystem; nsys may still touch /tmp/injection_config_* even when TMPDIR is set." >&2
  fi
fi

if [[ -f "${nsys_base}.nsys-rep" ]]; then
  echo
  echo "== Nsight Systems Stats =="
  set +e
  "${NSYS}" stats \
    --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
    --format table \
    "${nsys_base}.nsys-rep" \
    > "${OUT_DIR}/nsys_stats.txt" 2>&1
  nsys_stats_rc=$?
  set -e
  cat "${OUT_DIR}/nsys_stats.txt"
  if [[ "${nsys_stats_rc}" -ne 0 ]]; then
    echo "nsys stats failed; the raw .nsys-rep file is still available." >&2
  fi
fi

echo
echo "== Nsight Compute =="
make_profile_args "${NCU_WARMUP}" "${NCU_REPEAT}"
ncu_base="${OUT_DIR}/ncu_gemm"
ncu_cmd=(
  env
  "TMPDIR=${PROFILE_TMPDIR}"
  "TMP=${PROFILE_TMPDIR}"
  "TEMP=${PROFILE_TMPDIR}"
  "${NCU}"
  --target-processes all
  --kernel-name-base demangled
  --launch-count "${NCU_LAUNCH_COUNT}"
  --force-overwrite
  --export "${ncu_base}"
)
if [[ "${PROFILE_CUBLAS}" -eq 0 ]]; then
  ncu_cmd+=(--kernel-name "${KERNEL_NAME}")
fi
if [[ -n "${NCU_SET}" ]]; then
  ncu_cmd+=(--set "${NCU_SET}")
else
  for section in "${NCU_SECTIONS[@]}"; do
    ncu_cmd+=(--section "${section}")
  done
fi
ncu_cmd+=("${NCU_EXTRA[@]}" "${BIN}" "${PROFILE_ARGS[@]}")

{
  printf 'Command:'
  printf ' %q' "${ncu_cmd[@]}"
  echo
} | tee "${OUT_DIR}/ncu_command.txt"
set +e
"${ncu_cmd[@]}" 2>&1 | tee "${OUT_DIR}/ncu_stdout.txt"
ncu_rc=${PIPESTATUS[0]}
set -e

if [[ "${ncu_rc}" -ne 0 ]]; then
  FINAL_RC=1
  echo
  echo "ncu failed with exit code ${ncu_rc}; see ${OUT_DIR}/ncu_stdout.txt" >&2
  if grep -q "ERR_NVGPUCTRPERM" "${OUT_DIR}/ncu_stdout.txt"; then
    echo "diagnosis: NVIDIA performance counters are restricted for this user." >&2
    echo "diagnosis: an admin must allow non-admin profiling or run ncu with sufficient privileges." >&2
  fi
  if grep -q "Failed to access the temporary directory" "${OUT_DIR}/ncu_stdout.txt"; then
    echo "diagnosis: ncu could not use its temporary directory; try --profile-tmpdir /dev/shm/${USER:-user}_ncu_tmp." >&2
  fi
fi

echo
echo "== Outputs =="
find "${OUT_DIR}" -maxdepth 1 -type f | sort | sed "s#^#  #"

exit "${FINAL_RC}"
