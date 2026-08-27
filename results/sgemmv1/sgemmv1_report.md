# SGEMM V1 Benchmark Report

Date: 2026-08-27

Host: `h100-gpu3`

GPU: `NVIDIA H100 80GB HBM3`

Source file: `sgemmv1.cu`

This report corrects the previous mistaken `sgemmv1.cu` snapshot. The current `sgemmv1.cu` is the real SGEMM V1 kernel supplied in `/SharedData/dengzy/kernel/sgemmv1.cu`.

## Kernel Shape

- CTA tile: `BM x BN = 128 x 128`
- K tile: `BK = 8`
- Per-thread output tile: `TM x TN = 8 x 8`
- Threads per CTA: `16 x 16 = 256`
- Shared memory:
  - `s_a[128][8]`
  - `s_b[8][128]`
  - Total: `8192 bytes`
- Global A/B loads use vectorized `float4`.
- C stores use vectorized `float4`.

Important constraint: this kernel does not implement general boundary masking. The benchmark results below use shapes where `M` and `N` are multiples of `128`, and `K` is a multiple of `8`.

## Build

The benchmark harness was generated temporarily by wrapping `sgemmv1.cu` as `gemm_kernel` and reusing the existing GEMM benchmark driver.

Command:

```bash
TMPDIR=/SharedData/dengzy/kernel/.tmp \
/usr/local/cuda-12.9/bin/nvcc -O3 -std=c++17 -arch=sm_90 -lineinfo \
  --ptxas-options=-v .tmp/sgemmv1_actual/bench_sgemmv1.cu -lcublas \
  -o .tmp/sgemmv1_actual/sgemmv1_bench
```

ptxas resource usage:

| Metric | Value |
|---|---:|
| Registers/thread | 128 |
| Shared memory/block | 8192 B |
| Stack frame | 0 B |
| Spill stores | 0 B |
| Spill loads | 0 B |

## Correctness

Correctness was checked against the cuBLAS FP32 pedantic reference.

Command:

```bash
.tmp/sgemmv1_actual/sgemmv1_bench \
  --shape 512x512x512 \
  --shape 1024x1024x1024 \
  --shape 4096x4096x4096 \
  --warmup 1 --repeat 2 --no-cublas
```

Result:

| Shape | Time ms | TFLOPS | Peak % | max_abs | max_rel | bad_count | ok |
|---|---:|---:|---:|---:|---:|---:|---|
| 512x512x512 | 0.083 | 3.243 | 4.847 | 9.06e-06 | 5.84e-02 | 0 | yes |
| 1024x1024x1024 | 0.156 | 13.769 | 20.579 | 0 | 0 | 0 | yes |
| 4096x4096x4096 | 4.044 | 33.988 | 50.798 | 0 | 0 | 0 | yes |

## Memcheck

Command:

```bash
TMPDIR=/dev/shm/dengzy_compute_sanitizer_tmp \
/usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck \
  .tmp/sgemmv1_actual/sgemmv1_bench \
  --shape 512x512x512 --warmup 1 --repeat 2 --no-cublas
```

Result:

```text
ERROR SUMMARY: 0 errors
```

## Performance

Performance command:

```bash
.tmp/sgemmv1_actual/sgemmv1_bench \
  --sizes 512,1024,2048,4096 \
  --warmup 5 --repeat 20 \
  --no-check --no-cublas --csv
```

FLOPs are computed as:

```text
FLOPs = 2 * M * N * K
```

The factor 2 counts each multiply-add as two floating-point operations.

| Shape | FLOPs | Time ms | TFLOPS | Peak % | AI | Roofline TFLOPS | Bound | Roofline % |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| 512x512x512 | 268,435,456 | 0.0805488 | 3.33258 | 4.98083 | 85.3333 | 66.9082 | compute | 4.98083 |
| 1024x1024x1024 | 2,147,483,648 | 0.155350 | 13.8235 | 20.6604 | 170.667 | 66.9082 | compute | 20.6604 |
| 2048x2048x2048 | 17,179,869,184 | 0.503141 | 34.1453 | 51.0330 | 341.333 | 66.9082 | compute | 51.0330 |
| 4096x4096x4096 | 137,438,953,472 | 4.07045 | 33.7650 | 50.4647 | 682.667 | 66.9082 | compute | 50.4647 |

## Notes

This V1 is much faster than the earlier BM128/BN128/BK32 experiment because the accumulator stays in registers:

- Current SGEMM V1: `128 registers/thread`, `0 B stack frame`, `0 spill`
- Earlier experiment: `40 registers/thread`, `256 B stack frame`, `0 spill`

The current kernel still has limitations:

- No general boundary masking for arbitrary `M/N/K`.
- High register count limits occupancy.
- It is still FP32 CUDA-core SGEMM, not a Tensor Core/MMA kernel.
- Further tuning should use `ncu` to inspect occupancy, shared-memory bank conflicts, and instruction mix.

Raw artifacts:

- `build.log`
- `correctness.txt`
- `memcheck_512.txt`
- `perf.csv`
- `perf_summary.csv`
