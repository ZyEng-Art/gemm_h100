# Float4 Vectorized GEMM Report

Date: 2026-08-28

Host: `h100-gpu3`

GPU: `NVIDIA H100 80GB HBM3`

Source file: `/SharedData/dengzy/kernel/gemm.cu`

## Optimization Content

This version keeps the register-resident accumulator path from commit `054d420` and adds 16-byte vectorized memory operations:

- Added `FLOAT4(pointer)` based on `reinterpret_cast<float4*>`.
- A/B global-to-shared copies now move 4 FP32 values per vectorized operation.
- C stores now write 4 FP32 accumulator values per vectorized operation.
- The compute loop remains explicitly unrolled.
- `tmp_c[8][8]` remains register-resident.

Kernel shape:

| Parameter | Value |
|---|---:|
| `BM` | 128 |
| `BN` | 128 |
| `BK` | 32 |
| `TM` | 8 |
| `TN` | 8 |
| `TK` | 8 |
| Threads/CTA | 256 |
| Shared memory/CTA | 32768 B |

Important constraint: this version is currently an aligned-shape `float4` fast path. It is valid for the benchmarked square sizes where vectorized load/store addresses are aligned and complete. It is not yet a general boundary-safe implementation for arbitrary `M/N/K`.

## Build Resource Usage

Compiled with:

```bash
TMPDIR=/SharedData/dengzy/kernel/.tmp \
/usr/local/cuda-12.9/bin/nvcc -O3 -std=c++17 -arch=sm_90 -lineinfo \
  --ptxas-options=-v gemm.cu -lcublas \
  -o .tmp/float4_vectorized/gemm_bench
```

ptxas result:

| Version | Registers/thread | Shared memory/block | Stack frame | Spill stores | Spill loads |
|---|---:|---:|---:|---:|---:|
| `054d420` register-unrolled | 148 | 32768 B | 0 B | 0 B | 0 B |
| Current float4 vectorized | 128 | 32768 B | 0 B | 0 B | 0 B |

The current version keeps the accumulator off local memory and reduces register usage from `148` to `128` registers/thread.

## Correctness

Aligned correctness was checked against the cuBLAS FP32 pedantic reference.

Command:

```bash
.tmp/float4_vectorized/gemm_bench \
  --shape 512x512x512 \
  --shape 1024x1024x1024 \
  --shape 4096x4096x4096 \
  --warmup 1 --repeat 2 --no-cublas
```

Result:

| Shape | Time ms | TFLOPS | Peak % | max_abs | max_rel | bad_count | ok |
|---|---:|---:|---:|---:|---:|---:|---|
| 512x512x512 | 0.079 | 3.398 | 5.078 | 9.06e-06 | 5.84e-02 | 0 | yes |
| 1024x1024x1024 | 0.150 | 14.361 | 21.464 | 0 | 0 | 0 | yes |
| 4096x4096x4096 | 3.670 | 37.449 | 55.971 | 0 | 0 | 0 | yes |

Non-aligned boundary check:

```bash
.tmp/float4_vectorized/gemm_bench \
  --shape 130x130x130 \
  --warmup 1 --repeat 2 --no-cublas
```

Result:

```text
ERROR: CUDA error at gemm.cu:442: misaligned address (716)
```

This failure is expected with the current raw `FLOAT4` path because the boundary checks only validate the first element of each 4-float vector and do not guarantee 16-byte alignment or that all 4 values remain inside the row/tile.

## Memcheck

Aligned memcheck command:

```bash
TMPDIR=/dev/shm/dengzy_compute_sanitizer_tmp \
/usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck \
  .tmp/float4_vectorized/gemm_bench \
  --shape 512x512x512 --warmup 1 --repeat 2 --no-cublas
```

Result:

```text
ERROR SUMMARY: 0 errors
```

## Performance

Performance command:

```bash
.tmp/float4_vectorized/gemm_bench \
  --sizes 512,1024,2048,4096 \
  --warmup 5 --repeat 20 \
  --no-check --no-cublas --csv
```

FLOPs are computed as:

```text
FLOPs = 2 * M * N * K
```

| Shape | FLOPs | Time ms | TFLOPS | Peak % | AI | Roofline TFLOPS | Bound | Roofline % |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| 512x512x512 | 268,435,456 | 0.0753632 | 3.56189 | 5.32355 | 85.3333 | 66.9082 | compute | 5.32355 |
| 1024x1024x1024 | 2,147,483,648 | 0.146504 | 14.6582 | 21.9079 | 170.667 | 66.9082 | compute | 21.9079 |
| 2048x2048x2048 | 17,179,869,184 | 0.467397 | 36.7565 | 54.9357 | 341.333 | 66.9082 | compute | 54.9357 |
| 4096x4096x4096 | 137,438,953,472 | 3.66825 | 37.4672 | 55.9980 | 682.667 | 66.9082 | compute | 55.9980 |

## Performance By Commit

The table below records the performance impact of every repository commit up to this report. Commits that only changed scripts or metadata did not change the GEMM kernel, so they inherit the previous kernel's performance. The `1e31b73` SGEMM V1 snapshot was later corrected by `fc046a7`; it is kept here for history but should not be treated as the final `sgemmv1.cu` result.

| Commit | Change | Kernel target | 512 TFLOPS | 1024 TFLOPS | 2048 TFLOPS | 4096 TFLOPS | Notes |
|---|---|---|---:|---:|---:|---:|---|
| `b1f1322` | Add H100 GEMM benchmark and roofline results | `gemm.cu` baseline | 6.43422 | 7.17930 | 7.49115 | 7.52327 | First benchmarked baseline |
| `707d379` | Add Nsight profiling script | unchanged | 6.43422 | 7.17930 | 7.49115 | 7.52327 | Script-only commit |
| `012b678` | Initialize h100 working tree metadata | unchanged | 6.43422 | 7.17930 | 7.49115 | 7.52327 | Metadata/script cleanup |
| `1fcccfe` | Allow shell scripts to reexec bash | unchanged | 6.44039 | 7.18168 | 7.49868 | 7.56203 | Rebenchmarked before BM128 change |
| `558e70f` | Add BM128 BN128 BK32 GEMM experiment | `gemm.cu` BM128/BK32 local accumulator | 0.341147 | 1.33351 | 3.11496 | 2.91086 | `tmp_c[8][8]` used 256 B stack frame |
| `1e31b73` | Add SGEMM V1 benchmark snapshot | superseded `sgemmv1.cu` snapshot | 0.341319 | 1.33463 | 3.11575 | 2.90870 | Mistaken snapshot, later corrected |
| `fc046a7` | Replace SGEMM v1 snapshot with actual kernel | actual `sgemmv1.cu` | 3.33258 | 13.8235 | 34.1453 | 33.7650 | Actual vectorized SGEMM V1 |
| `054d420` | Keep GEMM accumulator in registers | `gemm.cu` register-unrolled BM128/BK32 | 2.57256 | 11.0861 | 18.2080 | 18.5298 | `tmp_c[8][8]` scalarized into registers |
| this commit | Add float4 vectorized load/store | `gemm.cu` float4 vectorized BM128/BK32 | 3.56189 | 14.6582 | 36.7565 | 37.4672 | Aligned-shape fast path |

Timing data for the same commit sequence:

| Commit | Kernel target | 512 ms | 1024 ms | 2048 ms | 4096 ms |
|---|---|---:|---:|---:|---:|
| `b1f1322` | `gemm.cu` baseline | 0.041720 | 0.299122 | 2.29336 | 18.2685 |
| `707d379` | unchanged | 0.041720 | 0.299122 | 2.29336 | 18.2685 |
| `012b678` | unchanged | 0.041720 | 0.299122 | 2.29336 | 18.2685 |
| `1fcccfe` | unchanged | 0.041680 | 0.299022 | 2.29105 | 18.1749 |
| `558e70f` | `gemm.cu` BM128/BK32 local accumulator | 0.786862 | 1.61040 | 5.51527 | 47.2159 |
| `1e31b73` | superseded `sgemmv1.cu` snapshot | 0.786466 | 1.60905 | 5.51387 | 47.2509 |
| `fc046a7` | actual `sgemmv1.cu` | 0.0805488 | 0.155350 | 0.503141 | 4.07045 |
| `054d420` | `gemm.cu` register-unrolled BM128/BK32 | 0.104346 | 0.193710 | 0.943533 | 7.41717 |
| this commit | `gemm.cu` float4 vectorized BM128/BK32 | 0.0753632 | 0.146504 | 0.467397 | 3.66825 |

## Comparison With Previous GEMM Version

The previous `gemm.cu` version is commit `054d420`, which kept the accumulator in registers but used scalar load/store paths.

| Shape | Previous ms | Current ms | Time speedup | Previous TFLOPS | Current TFLOPS | TFLOPS speedup |
|---|---:|---:|---:|---:|---:|---:|
| 512x512x512 | 0.104346 | 0.0753632 | 1.38x | 2.57256 | 3.56189 | 1.38x |
| 1024x1024x1024 | 0.193710 | 0.146504 | 1.32x | 11.0861 | 14.6582 | 1.32x |
| 2048x2048x2048 | 0.943533 | 0.467397 | 2.02x | 18.2080 | 36.7565 | 2.02x |
| 4096x4096x4096 | 7.41717 | 3.66825 | 2.02x | 18.5298 | 37.4672 | 2.02x |

## Comparison With Initial Baseline

The initial benchmarked `gemm.cu` baseline is commit `b1f1322`.

| Shape | Initial ms | Current ms | Time speedup | Initial TFLOPS | Current TFLOPS | TFLOPS speedup |
|---|---:|---:|---:|---:|---:|---:|
| 512x512x512 | 0.041720 | 0.0753632 | 0.55x | 6.43422 | 3.56189 | 0.55x |
| 1024x1024x1024 | 0.299122 | 0.146504 | 2.04x | 7.17930 | 14.6582 | 2.04x |
| 2048x2048x2048 | 2.29336 | 0.467397 | 4.91x | 7.49115 | 36.7565 | 4.91x |
| 4096x4096x4096 | 18.2685 | 3.66825 | 4.98x | 7.52327 | 37.4672 | 4.98x |

## Comparison With SGEMM V1

| Shape | Current TFLOPS | SGEMM V1 TFLOPS | Current / SGEMM V1 |
|---|---:|---:|---:|
| 512x512x512 | 3.56189 | 3.33258 | 106.9% |
| 1024x1024x1024 | 14.6582 | 13.8235 | 106.0% |
| 2048x2048x2048 | 36.7565 | 34.1453 | 107.6% |
| 4096x4096x4096 | 37.4672 | 33.7650 | 111.0% |

## Notes

The current aligned fast path is a strong large-matrix improvement, but it deliberately trades off general boundary safety:

- Raw `FLOAT4` loads/stores require aligned addresses.
- For FP32, the vector start should be 4-float aligned.
- The current boundary checks do not cover `col + 1`, `col + 2`, or `col + 3`.
- `move_shm_in` no longer zero-fills skipped elements, so partial K/M/N tiles need a separate scalar or masked tail path before arbitrary shapes are supported again.
- The kernel still uses CUDA-core FP32 FMA, not Tensor Core/MMA.

Raw artifacts in this directory:

- `build.log`
- `correctness_aligned.txt`
- `correctness_non_aligned_130.txt`
- `memcheck_512.txt`
- `perf.csv`
- `perf_comparison.csv`
- `performance_by_commit.csv`
