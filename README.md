# H100 GEMM Benchmark

CUDA FP32 GEMM benchmark and roofline harness for an NVIDIA H100 80GB HBM3 GPU.

The current kernel is a correctness-first FP32 CUDA core GEMM baseline with shared-memory tiling. The benchmark reports:

- GPU hardware information
- average kernel time
- custom kernel TFLOPS
- cuBLAS FP32 reference timing
- correctness against cuBLAS
- theoretical peak percentage
- arithmetic intensity
- roofline memory/compute bound
- roofline utilization
- SVG roofline plot

## Files

```text
gemm.cu                  CUDA GEMM kernel + benchmark harness
sgemm_v2.cu              SGEMM V2 kernel-body snapshot for comparison
run_gemm_bench.sh        compile and run benchmark
plot_roofline.py         generate roofline.svg from CSV without matplotlib
run_gemm_roofline.sh     run CSV benchmark and generate roofline.svg
results/                 benchmark outputs committed from h100-gpu3
```

## Kernel Contract

The custom kernel computes:

```text
C[M, N] = A[M, K] * B[K, N]
```

All matrices are row-major `float`.

The kernel entry point is:

```cpp
__global__ void gemm_kernel(const float* A,
                            const float* B,
                            float* C,
                            int M,
                            int N,
                            int K);
```

## Current Optimization Content

The current kernel is a CUDA-core FP32 shared-memory tiled GEMM:

- `BM=128`, `BN=128`, `BK=32`
- `TM=8`, `TN=8`
- one CUDA block computes one `128 x 128` tile of C
- one thread computes an `8 x 8` register accumulator tile
- A and B tiles are staged through shared memory
- global/shared C paths use `float4` vectorized loads or stores on aligned benchmark shapes
- each thread maps its 8 output columns as two 4-column groups separated by `BN/2`

This is not Tensor Core code. It uses normal CUDA-core FP32 FMA.

Current caveat: this is an aligned-shape fast path. Default committed results use `512/1024/2048/4096`, so the `float4` accesses are aligned and correctness passes. For arbitrary non-multiple shapes, add scalar/masked tails and complete boundary checks before relying on correctness.

Latest detailed optimization report:

```text
results/optimization_reports/b_split_float4_20260904/b_split_float4_report.md
```

Expected next optimization directions:

- move B `float4` loads out of the `tm` loop so each K-step B vector is reused across all 8 rows
- investigate remaining shared-memory bank conflicts with Nsight Compute when GPU counters are available
- tune the A shared-memory layout separately from the B split
- compare register pressure and occupancy after each mapping change
- optional TF32/FP16/BF16 Tensor Core implementation if changing the datatype/precision target

## Run

On `h100-gpu3`:

```bash
cd /SharedData/dengzy/kernel
./run_gemm_bench.sh
```

Run one shape:

```bash
./run_gemm_bench.sh --shape 4096x4096x4096 --warmup 5 --repeat 50
```

Generate CSV and roofline SVG:

```bash
./run_gemm_roofline.sh
```

Outputs:

```text
gemm_roofline.csv
roofline.svg
```

The script automatically uses a local `.tmp` directory under the project when `TMPDIR` is unset. This avoids failures when `/tmp` is full.

## Theory

Measured GEMM throughput:

```text
FLOPs = 2 * M * N * K
TFLOPS = FLOPs / seconds / 1e12
```

Default compute peak is FP32 CUDA core peak:

```text
peak = SM_count * 128 FP32 cores/SM * 2 FLOPs/FMA * clock
```

For the measured H100:

```text
SM_count = 132
clock = 1980 MHz
peak = 66.908 TFLOPS
```

Roofline uses ideal minimum DRAM traffic:

```text
ideal_dram_bytes = 4 * (M*K + K*N + M*N)
AI = FLOPs / ideal_dram_bytes
mem_roof = AI * HBM_TBps
roofline = min(compute_peak, mem_roof)
roof_% = measured_TFLOPS / roofline * 100
```

Default HBM bandwidth for roofline:

```text
3.35 TB/s
```

Override with:

```bash
./run_gemm_bench.sh --hbm-tbps 3.35
```

## Current Results

Committed result files:

```text
results/optimization_reports/b_split_float4_20260904/run_info.txt
results/optimization_reports/b_split_float4_20260904/build.log
results/optimization_reports/b_split_float4_20260904/benchmark.txt
results/optimization_reports/b_split_float4_20260904/perf.csv
results/optimization_reports/b_split_float4_20260904/perf_comparison.csv
results/optimization_reports/b_split_float4_20260904/memcheck_512.txt
results/optimization_reports/b_split_float4_20260904/b_split_float4_report.md
```

Benchmark environment:

```text
GPU: NVIDIA H100 80GB HBM3
SM count: 132
SM clock: 1980.0 MHz
FP32 CUDA core peak used: 66.908 TFLOPS
HBM bandwidth used for roofline: 3.350 TB/s
Timing: warmup=5, repeat=20
cuBLAS reference: fp32_pedantic
```

Build resource usage:

```text
registers/thread: 128
shared memory/block: 32768 B
stack frame: 0 B
spill stores: 0 B
spill loads: 0 B
```

Correctness summary from `results/optimization_reports/b_split_float4_20260904/benchmark.txt`:

```text
512/1024/2048/4096: bad_count=0, ok=yes
```

Score summary from `results/optimization_reports/b_split_float4_20260904/perf.csv` and `results/optimization_reports/b_split_float4_20260904/benchmark.txt`:

| M=N=K | custom ms | custom TFLOPS | peak % | AI | roofline bound | roof % | cuBLAS TFLOPS | ok |
|---:|---:|---:|---:|---:|---|---:|---:|:--|
| 512 | 0.062888 | 4.26847 | 6.37959 | 85.333 | compute | 6.37959 | 18.310 | yes |
| 1024 | 0.124078 | 17.3075 | 25.8675 | 170.667 | compute | 25.8675 | 38.013 | yes |
| 2048 | 0.424072 | 40.5117 | 60.5482 | 341.333 | compute | 60.5482 | 50.759 | yes |
| 4096 | 3.37411 | 40.7334 | 60.8796 | 682.667 | compute | 60.8796 | 52.084 | yes |

The default shapes are all compute-bound under the ideal DRAM roofline model. The current custom kernel reaches about `60.9%` of the measured FP32 CUDA-core peak on `4096x4096x4096`.

Compared with previous commit `5d35995`, the current `4096x4096x4096` score improves from `37.5215` TFLOPS to `40.7334` TFLOPS, a `1.09x` speedup.

Shared-memory bank conflict status for current `gemm.cu` versus previous `5d35995`:

```text
results/optimization_reports/b_split_float4_20260904/shared_conflict_gemm_current_vs_prev.csv
```

The available previous-version 4096 NCU reference reports:

| Metric | Previous `5d35995` | Previous conflicts/wavefront | Current `03139ba` |
|---|---:|---:|---:|
| shared load bank conflicts | 268,451,537 | 0.40001438 | NA |
| shared store bank conflicts | 481,387 | 0.01414354 | NA |
| shared total bank conflicts | 268,932,924 | 0.38138911 | NA |

Fresh current-version NCU counter collection is currently blocked by `ERR_NVGPUCTRPERM`, so the current-vs-previous shared conflict delta for `gemm.cu` is not yet measured. The failed permission check is logged in `results/optimization_reports/b_split_float4_20260904/ncu_permission_check.txt`.

## SGEMM V2 Snapshot

`sgemm_v2.cu` is committed as a comparison snapshot. It is a kernel-body fragment, so the committed measurement wraps it with the existing benchmark harness and uses the V2 parameters:

```text
BM=128, BN=128, BK=8, TM=8, TN=8
```

Detailed report:

```text
results/optimization_reports/sgemm_v2_20260904/sgemm_v2_report.md
```

Build resource usage:

```text
registers/thread: 127
shared memory/block: 8192 B
stack frame: 0 B
spill stores: 0 B
spill loads: 0 B
```

Correctness summary from `results/optimization_reports/sgemm_v2_20260904/benchmark.txt`:

```text
512/1024/2048/4096: bad_count=0, ok=yes
```

Score summary from `results/optimization_reports/sgemm_v2_20260904/perf.csv`:

| M=N=K | SGEMM V2 ms | SGEMM V2 TFLOPS | peak % | current main TFLOPS | V2 / current main |
|---:|---:|---:|---:|---:|---:|
| 512 | 0.0735744 | 3.64849 | 5.45298 | 4.26847 | 0.855x |
| 1024 | 0.144406 | 14.8711 | 22.2262 | 17.3075 | 0.859x |
| 2048 | 0.437907 | 39.2318 | 58.6352 | 40.5117 | 0.968x |
| 4096 | 3.48404 | 39.4481 | 58.9586 | 40.7334 | 0.968x |

On the measured run, SGEMM V2 is close on large shapes but still below the current main GEMM. On `4096x4096x4096`, current main is about `1.03x` faster.

Shared-memory bank conflict snapshot for `4096x4096x4096`, using existing root-collected NCU data:

| Metric | SGEMM V1 | SGEMM V2 | Change |
|---|---:|---:|---:|
| shared load bank conflicts | 67,143,197 | 31,168 | -99.9536% |
| shared store bank conflicts | 792,520 | 4,824,453 | +508.7484% |
| shared total bank conflicts | 67,935,717 | 4,855,621 | -92.8526% |

Fresh NCU counter collection is currently blocked by `ERR_NVGPUCTRPERM`; the failed permission check is logged in `results/optimization_reports/sgemm_v2_20260904/ncu_permission_check.txt`.
