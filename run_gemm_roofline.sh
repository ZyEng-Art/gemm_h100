#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CSV_OUT="${CSV_OUT:-${SCRIPT_DIR}/gemm_roofline.csv}"
SVG_OUT="${SVG_OUT:-${SCRIPT_DIR}/roofline.svg}"

"${SCRIPT_DIR}/run_gemm_bench.sh" --csv "$@" > "${CSV_OUT}"
python3 "${SCRIPT_DIR}/plot_roofline.py" --csv "${CSV_OUT}" --out "${SVG_OUT}"

echo "CSV: ${CSV_OUT}"
echo "SVG: ${SVG_OUT}"
