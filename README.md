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

The current kernel is a CUDA-core FP32 shared-memory tiled GEMM using the
v3-style register-prefetch software pipeline:

- `BM=128`, `BN=128`, `BK=8`
- `TM=8`, `TN=8`
- one CUDA block computes one `128 x 128` tile of C
- one thread computes an `8 x 8` register accumulator tile
- A is stored in shared memory as `s_a[2][BK][BM]`, so the compute path reads an A column as contiguous shared-memory data
- B is stored in shared memory as `s_b[2][BK][BN]`
- global loads use `float4` vectorized loads on aligned benchmark shapes
- the next K tile is prefetched into registers before computing the current shared-memory tile
- after the current tile compute, the prefetched register data is written into the alternate shared-memory buffer

This is not Tensor Core code. It uses normal CUDA-core FP32 FMA. The current
implementation intentionally does not use `cp.async`; the software pipeline is
implemented with register prefetch plus shared-memory double buffering.

Current caveat: this is an aligned-shape fast path. Default committed results
use `512/1024/2048/4096`, so the `float4` accesses are aligned and correctness
passes. For arbitrary non-multiple shapes, add scalar/masked tails and complete
boundary checks before relying on correctness.

Latest detailed optimization report:

```text
results/optimization_reports/v3style_software_pipeline_20260907/v3style_software_pipeline_report.md
```

Expected next optimization directions:

- inspect whether the current B shared-memory access pattern can be reduced further without increasing instruction count
- compare the register-prefetch software pipeline against a carefully scheduled `cp.async` pipeline with the same shared-memory layout
- keep `ptxas` register count under the 128-register occupancy cliff for this thread/block shape
- add a separate masked-tail kernel or scalar fallback if arbitrary shapes are required
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
results/optimization_reports/v3style_software_pipeline_20260907/build.log
results/optimization_reports/v3style_software_pipeline_20260907/perf_gpu7_nocheck.csv
results/optimization_reports/v3style_software_pipeline_20260907/correctness_default_sizes.csv
results/optimization_reports/v3style_software_pipeline_20260907/run_script_sh_smoke.csv
results/optimization_reports/v3style_software_pipeline_20260907/prev_cpasync_perf_gpu7.csv
results/optimization_reports/v3style_software_pipeline_20260907/v3style_perf_gpu7.csv
results/optimization_reports/v3style_software_pipeline_20260907/prev_cpasync_4096_details.txt
results/optimization_reports/v3style_software_pipeline_20260907/v3style_4096_details.txt
results/optimization_reports/v3style_software_pipeline_20260907/v3style_software_pipeline_report.md
```

Benchmark environment:

```text
GPU: NVIDIA H100 80GB HBM3
SM count: 132
SM clock: 1980.0 MHz
FP32 CUDA core peak used: 66.908 TFLOPS
HBM bandwidth used for roofline: 3.350 TB/s
Timing: warmup/repeat varies by artifact and is recorded in each CSV
cuBLAS reference: fp32_pedantic when correctness checking is enabled
```

Build resource usage:

```text
registers/thread: 127
shared memory/block: 16384 B
stack frame: 0 B
spill stores: 0 B
spill loads: 0 B
```

Correctness summary from
`results/optimization_reports/v3style_software_pipeline_20260907/correctness_default_sizes.csv`:

```text
512/1024/2048/4096: bad_count=0, passed=1
```

Score summary from
`results/optimization_reports/v3style_software_pipeline_20260907/perf_gpu7_nocheck.csv`:

| M=N=K | custom ms | custom TFLOPS | peak % | AI | roofline bound | roof % | ok |
|---:|---:|---:|---:|---:|---|---:|:--|
| 512 | 0.050878 | 5.27602 | 7.88547 | 85.333 | compute | 7.88547 | yes |
| 1024 | 0.099827 | 21.5120 | 32.1515 | 170.667 | compute | 32.1515 | yes |
| 2048 | 0.363261 | 47.2934 | 70.6841 | 341.333 | compute | 70.6841 | yes |
| 4096 | 2.83845 | 48.4204 | 72.3684 | 682.667 | compute | 72.3684 | yes |

The default shapes are all compute-bound under the ideal DRAM roofline model.
The current custom kernel reaches about `72.4%` of the measured FP32 CUDA-core
peak on `4096x4096x4096`.

Compared with the previous cp.async double-buffer implementation measured in
the same reference run, the current `4096x4096x4096` score improves from
`38.3056` TFLOPS to `48.1134` TFLOPS, a `1.256x` speedup.

NCU scheduler comparison on `4096x4096x4096`, collected inside Docker:

| Metric | Previous cp.async | Current v3-style |
|---|---:|---:|
| Compute (SM) Throughput | 69.68% | 80.19% |
| Memory Throughput | 46.01% | 72.13% |
| L1/TEX Cache Throughput | 47.49% | 74.45% |
| No Eligible | 28.09% | 17.22% |
| Issued Warp Per Scheduler | 0.72 | 0.83 |
| Eligible Warps Per Scheduler | 2.36 | 2.55 |
| Achieved Occupancy | 24.06% | 23.93% |
| Executed Instructions | 2.614B | 2.396B |

The previous cp.async implementation does generate `LDGSTS`/`DEPBAR`, but the
current software-pipelined version has fewer no-eligible scheduler cycles and
higher issue utilization.

Both profiled binaries use `128` registers/thread and `32768 B` static shared memory/block (`33792 B` allocated). The raw NCU files are `shared_conflict_prev_5d35995_4096_bank.csv` and `shared_conflict_current_03139ba_4096_bank.csv`; the earlier non-Docker permission failure remains logged in `ncu_permission_check.txt`.

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

Cross-kernel comparison against current main `gemm.cu` is recorded in `results/optimization_reports/b_split_float4_20260904/gemm_current_vs_sgemm_v2_shared_conflict.csv`. It is not a strict one-change ablation because SGEMM V2 is a snapshot harness, but it explains the store-side conflict gap:

| Metric | Current `gemm.cu` | SGEMM V2 | SGEMM V2 / current |
|---|---:|---:|---:|
| shared store bank conflicts | 786,535 | 4,824,453 | 6.13x |
| shared store conflicts/wavefront | 0.02290369 | 0.36512758 | 15.94x |
| shared total bank conflicts | 859,718 | 4,855,621 | 5.65x |
| shared total conflicts/wavefront | 0.00196701 | 0.04262775 | 21.67x |

The main store-conflict gap is consistent with SGEMM V2's transposed A shared layout. With `s_a[BK][BM]` and `BM=128`, the bank index for `s_a[k][m]` is effectively `m % 32`; SGEMM V2 maps adjacent lanes to the same `m` but different `k`, so paired lanes write different addresses in the same bank. Current `gemm.cu` mostly keeps shared stores vectorized and contiguous, avoiding that store-side penalty.

Fresh NCU counter collection is currently blocked by `ERR_NVGPUCTRPERM`; the failed permission check is logged in `results/optimization_reports/sgemm_v2_20260904/ncu_permission_check.txt`.
