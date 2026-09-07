# V3-Style Software-Pipelined GEMM Report

Date: 2026-09-07

Host: `h100-gpu3`

GPU: `NVIDIA H100 80GB HBM3`

Compared versions:

- Previous reference: cp.async double-buffer implementation saved before the v3-style rewrite
- Current: working-tree `gemm.cu` with v3-style register prefetch and shared-memory double buffering

## Optimization Content

The current kernel keeps the CUDA-core FP32 register-tiled GEMM structure, but
rewrites the K-tile pipeline to match the faster v3-style implementation.

Current tile configuration:

- CTA tile: `BM x BN = 128 x 128`
- K tile: `BK = 8`
- Per-thread accumulator tile: `TM x TN = 8 x 8`
- Threads per CTA: `16 x 16 = 256`
- Shared memory:
  - `s_a[2][BK][BM]`
  - `s_b[2][BK][BN]`
  - Total: `2 * (BK * BM + BK * BN) * sizeof(float) = 16384 B`

Main changes:

- Removed the previous `cp.async` helper path from `gemm.cu`.
- Stores A in shared memory as `[K][M]` so the compute stage loads the per-thread A vector with contiguous `LDS.128` operations.
- Keeps B in shared memory as `[K][N]`.
- Uses `float4` global loads for A and B on aligned benchmark shapes.
- Prefetches the next K tile from global memory into registers before computing the current shared-memory tile.
- Writes the prefetched register tile into the alternate shared-memory buffer after the current tile compute.
- Keeps accumulators in registers with no stack frame or spills.

This remains an FP32 CUDA-core FMA kernel, not a Tensor Core kernel.

## Current Caveat

This kernel is an aligned-shape fast path. The committed benchmark shapes
`512/1024/2048/4096` are multiples of the tile/vectorization sizes and pass
correctness. For arbitrary non-multiple shapes, add scalar/masked tails or a
separate fallback kernel.

## Build Resource Usage

Compiled with:

```bash
nvcc -O3 -std=c++17 -arch=sm_90 -lineinfo --ptxas-options=-v gemm.cu -lcublas
```

`build.log`:

| Metric | Current |
|---|---:|
| Registers/thread | 127 |
| Shared memory/block | 16384 B |
| Stack frame | 0 B |
| Spill stores | 0 B |
| Spill loads | 0 B |

Compared to the previous cp.async reference build:

| Version | Registers/thread | Shared memory/block | Stack frame | Spills |
|---|---:|---:|---:|---:|
| Previous cp.async | 127 | 32768 B | 0 B | 0 |
| Current v3-style | 127 | 16384 B | 0 B | 0 |

The occupancy-relevant register count is unchanged, while shared-memory usage
drops by 50%.

## Correctness

Correctness was checked against cuBLAS FP32 pedantic reference on the default
aligned shapes.

Source artifact:

```text
correctness_default_sizes.csv
```

| M=N=K | custom TFLOPS | max_abs | max_rel | bad_count | passed |
|---:|---:|---:|---:|---:|---:|
| 512 | 4.75227 | 0 | 0 | 0 | 1 |
| 1024 | 19.8094 | 0 | 0 | 0 | 1 |
| 2048 | 42.8882 | 0 | 0 | 0 | 1 |
| 4096 | 48.4573 | 0 | 0 | 0 | 1 |

`sh run_gemm_bench.sh` smoke also passed on shape `512x512x512`; see
`run_script_sh_smoke.csv`.

## Performance

Performance was measured with correctness and cuBLAS timing disabled:

```bash
CUDA_VISIBLE_DEVICES=7 .tmp/v3style_gemm_20260907/gemm_v3style_bench \
  --device 0 --sizes 512,1024,2048,4096 \
  --warmup 5 --repeat 20 --no-check --no-cublas --csv
```

Source artifact:

```text
perf_gpu7_nocheck.csv
```

Theoretical peak used by the benchmark harness:

```text
SM count = 132
FP32 lanes/SM = 128
FMA = 2 FLOPs
SM clock = 1980 MHz
FP32 CUDA-core peak = 66.9082 TFLOPS
HBM bandwidth for roofline = 3.35 TB/s
```

| M=N=K | custom ms | custom TFLOPS | peak % | AI FLOP/B | roofline bound | roofline % |
|---:|---:|---:|---:|---:|---|---:|
| 512 | 0.0508784 | 5.27602 | 7.88547 | 85.3333 | compute | 7.88547 |
| 1024 | 0.0998272 | 21.5120 | 32.1515 | 170.667 | compute | 32.1515 |
| 2048 | 0.363261 | 47.2934 | 70.6841 | 341.333 | compute | 70.6841 |
| 4096 | 2.83845 | 48.4204 | 72.3684 | 682.667 | compute | 72.3684 |

The 4096 score reaches about `72.37%` of the benchmark's FP32 CUDA-core peak.

## Comparison With Previous cp.async Version

For a direct scheduler/pipeline comparison, both versions were compiled and
profiled in the same run directory:

```text
prev_cpasync_perf_gpu7.csv
v3style_perf_gpu7.csv
prev_cpasync_4096_details.txt
v3style_4096_details.txt
```

Performance comparison:

| Version | Shape | custom ms | TFLOPS | peak % |
|---|---:|---:|---:|---:|
| Previous cp.async | 2048 | 0.454773 | 37.7768 | 56.4607 |
| Current v3-style | 2048 | 0.374144 | 45.9178 | 68.6280 |
| Previous cp.async | 4096 | 3.58796 | 38.3056 | 57.2510 |
| Current v3-style | 4096 | 2.85656 | 48.1134 | 71.9096 |

Relative improvement in the same reference run:

| Shape | TFLOPS speedup | Time reduction |
|---:|---:|---:|
| 2048 | 1.215x | 17.73% |
| 4096 | 1.256x | 20.38% |

NCU comparison on `4096x4096x4096`:

| Metric | Previous cp.async | Current v3-style | Direction |
|---|---:|---:|---|
| Duration | 3.56 ms | 2.86 ms | better |
| Compute (SM) Throughput | 69.68% | 80.19% | better |
| Memory Throughput | 46.01% | 72.13% | better |
| L1/TEX Cache Throughput | 47.49% | 74.45% | better |
| DRAM Throughput | 5.44% | 6.82% | not HBM-bound |
| One or More Eligible | 71.91% | 82.78% | better |
| No Eligible | 28.09% | 17.22% | better |
| Issued Warp Per Scheduler | 0.72 | 0.83 | better |
| Active Warps Per Scheduler | 3.85 | 3.83 | about equal |
| Eligible Warps Per Scheduler | 2.36 | 2.55 | better |
| Achieved Occupancy | 24.06% | 23.93% | about equal |
| Executed Instructions | 2.614B | 2.396B | better |

Interpretation:

- The previous cp.async version does generate the asynchronous-copy path:
  `LDGSTS.E.BYPASS.128`, `LDGDEPBAR`, and `DEPBAR` appear in the NCU source export.
- The previous cp.async version is therefore not a pure synchronous-copy path.
- However, it has more no-eligible scheduler cycles and a lower issued-warp rate.
- Occupancy is effectively unchanged, so the performance gap is mainly from
  instruction scheduling/dependency behavior and extra instruction overhead.
- The current software-pipelined version gives the scheduler more ready warps
  and reaches higher SM utilization.

SASS/source-export instruction counts from text grep:

| Instruction | Previous cp.async | Current v3-style |
|---|---:|---:|
| `LDGSTS` | 4 | 0 |
| `LDGDEPBAR` | 2 | 0 |
| `DEPBAR` | 4 | 0 |
| `BAR.SYNC` | 4 | 2 |
| `LDS.128` | 128 | 64 |
| `FFMA` | 2048 | 1024 |

The large `.ncu-rep` GUI files were saved under:

```text
/SharedData/dengzy/kernel/results/cpasync_pipeline_reference_20260907/
C:\Users\X0251\Documents\gemm_h100_profiles\cpasync_pipeline_reference_20260907\
```

They are intentionally not committed because the sampling reports are tens of
MB each and are better kept as local profiling artifacts. The committed text
exports preserve the key profiler metrics used in this report.

## Reproduction Commands

Build current:

```bash
cd /SharedData/dengzy/kernel
mkdir -p .tmp/v3style_gemm_20260907 results/v3style_gemm_20260907
nvcc -O3 -std=c++17 -arch=sm_90 -lineinfo --ptxas-options=-v \
  gemm.cu -lcublas -o .tmp/v3style_gemm_20260907/gemm_v3style_bench \
  2> results/v3style_gemm_20260907/build.log
```

Run current performance:

```bash
CUDA_VISIBLE_DEVICES=7 .tmp/v3style_gemm_20260907/gemm_v3style_bench \
  --device 0 --sizes 512,1024,2048,4096 \
  --warmup 5 --repeat 20 --no-check --no-cublas --csv \
  > results/v3style_gemm_20260907/perf_gpu7_nocheck.csv
```

Run current correctness:

```bash
CUDA_VISIBLE_DEVICES=7 .tmp/v3style_gemm_20260907/gemm_v3style_bench \
  --device 0 --sizes 512,1024,2048,4096 \
  --warmup 1 --repeat 3 --csv \
  > results/v3style_gemm_20260907/correctness_default_sizes.csv
```

Run NCU from Docker:

```bash
docker run --rm --privileged --cap-add=SYS_ADMIN --gpus all \
  -v /SharedData/dengzy/kernel:/work \
  -v /usr/local/cuda-12.9:/usr/local/cuda-12.9:ro \
  -w /work nvcr.io/nvidia/pytorch:26.04-py3 \
  env LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64 CUDA_VISIBLE_DEVICES=7 \
  ncu --target-processes all \
      --kernel-name regex:gemm_kernel \
      --launch-count 1 \
      --section SpeedOfLight \
      --section MemoryWorkloadAnalysis \
      --section SchedulerStats \
      --section WarpStateStats \
      --section SourceCounters \
      --section InstructionStats \
      --section Occupancy \
      --force-overwrite \
      -o results/cpasync_pipeline_reference_20260907/v3style_4096_ncu \
      .tmp/cpasync_pipeline_reference_20260907/v3style_bench \
      --device 0 --shape 4096x4096x4096 \
      --warmup 0 --repeat 1 --no-check --no-cublas
```
