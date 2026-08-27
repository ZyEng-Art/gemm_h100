# Register-Resident Accumulator GEMM Report

Date: 2026-08-27

Host: `h100-gpu3`

GPU: `NVIDIA H100 80GB HBM3`

Source file: `/SharedData/dengzy/kernel/gemm.cu`

## Optimization Content

This version keeps the existing CTA/register tile shape:

- CTA tile: `BM x BN = 128 x 128`
- K tile: `BK = 32`
- Per-thread output tile: `TM x TN = 8 x 8`
- Threads per CTA: `16 x 16 = 256`
- Shared memory:
  - `shm_A[128 * 32]`
  - `shm_B[32 * 128]`
  - Total: `32768 bytes`

The kernel was changed from a local-memory accumulator path into a register-resident accumulator path:

- Added explicit `#pragma unroll` directives on the compute loops.
- Removed the row/column bounds check from the innermost FMA path.
- Kept boundary safety through zero-filled shared-memory loads and guarded final stores.
- The output accumulator `tmp_c[8][8]` is now scalarized into registers by ptxas.

## Build Resource Usage

Compiled with:

```bash
TMPDIR=/SharedData/dengzy/kernel/.tmp \
/usr/local/cuda-12.9/bin/nvcc -O3 -std=c++17 -arch=sm_90 -lineinfo \
  --ptxas-options=-v gemm.cu -lcublas \
  -o .tmp/register_unroll/gemm_bench
```

| Version | Registers/thread | Shared memory/block | Stack frame | Spill stores | Spill loads |
|---|---:|---:|---:|---:|---:|
| Previous BM128/BN128/BK32 | 40 | 32768 B | 256 B | 0 B | 0 B |
| Current register-unrolled | 148 | 32768 B | 0 B | 0 B | 0 B |

The important change is the stack frame dropping from `256 B` to `0 B`. The previous stack frame matched `8 * 8 * sizeof(float)`, which indicated that `tmp_c[8][8]` was stored as local memory. The current version uses more registers, but keeps the accumulator off the local-memory path.

## Correctness

Correctness was checked against the cuBLAS FP32 pedantic reference.

Command:

```bash
.tmp/register_unroll/gemm_bench \
  --shape 130x130x130 \
  --shape 512x512x512 \
  --shape 4096x4096x4096 \
  --warmup 1 --repeat 2 --no-cublas
```

Result:

| Shape | Time ms | TFLOPS | Peak % | max_abs | max_rel | bad_count | ok |
|---|---:|---:|---:|---:|---:|---:|---|
| 130x130x130 | 0.042 | 0.106 | 0.158 | 0 | 0 | 0 | yes |
| 512x512x512 | 0.106 | 2.527 | 3.776 | 9.06e-06 | 5.84e-02 | 0 | yes |
| 4096x4096x4096 | 7.416 | 18.533 | 27.700 | 0 | 0 | 0 | yes |

The `130x130x130` case verifies non-multiple boundary handling for this version.

## Memcheck

Command:

```bash
TMPDIR=/dev/shm/dengzy_compute_sanitizer_tmp \
/usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck \
  .tmp/register_unroll/gemm_bench \
  --shape 130x130x130 --warmup 1 --repeat 2 --no-cublas
```

Result:

```text
ERROR SUMMARY: 0 errors
```

## Performance

Performance command:

```bash
.tmp/register_unroll/gemm_bench \
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
| 512x512x512 | 268,435,456 | 0.104346 | 2.57256 | 3.84491 | 85.3333 | 66.9082 | compute | 3.84491 |
| 1024x1024x1024 | 2,147,483,648 | 0.193710 | 11.0861 | 16.5691 | 170.667 | 66.9082 | compute | 16.5691 |
| 2048x2048x2048 | 17,179,869,184 | 0.943533 | 18.2080 | 27.2135 | 341.333 | 66.9082 | compute | 27.2135 |
| 4096x4096x4096 | 137,438,953,472 | 7.41717 | 18.5298 | 27.6944 | 682.667 | 66.9082 | compute | 27.6944 |

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
| this commit | Register-resident accumulator GEMM | `gemm.cu` register-unrolled BM128/BK32 | 2.57256 | 11.0861 | 18.2080 | 18.5298 | `tmp_c[8][8]` scalarized into registers |

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
| this commit | `gemm.cu` register-unrolled BM128/BK32 | 0.104346 | 0.193710 | 0.943533 | 7.41717 |

## Comparison With Previous Version

The previous version is the last committed BM128/BN128/BK32 `gemm.cu` path, which had a `256 B` stack frame for `tmp_c[8][8]`.

| Shape | Previous ms | Current ms | Time speedup | Previous TFLOPS | Current TFLOPS | TFLOPS speedup |
|---|---:|---:|---:|---:|---:|---:|
| 512x512x512 | 0.786862 | 0.104346 | 7.54x | 0.341147 | 2.57256 | 7.54x |
| 1024x1024x1024 | 1.610400 | 0.193710 | 8.31x | 1.333510 | 11.0861 | 8.31x |
| 2048x2048x2048 | 5.515270 | 0.943533 | 5.85x | 3.114960 | 18.2080 | 5.85x |
| 4096x4096x4096 | 47.215900 | 7.41717 | 6.37x | 2.910860 | 18.5298 | 6.37x |

## Comparison With SGEMM V1

The standalone `sgemmv1.cu` kernel remains faster on large matrices:

| Shape | Current TFLOPS | SGEMM V1 TFLOPS | Current / SGEMM V1 |
|---|---:|---:|---:|
| 512x512x512 | 2.57256 | 3.33258 | 77.2% |
| 1024x1024x1024 | 11.0861 | 13.8235 | 80.2% |
| 2048x2048x2048 | 18.2080 | 34.1453 | 53.3% |
| 4096x4096x4096 | 18.5298 | 33.7650 | 54.9% |

Likely remaining gaps:

- Current A/B shared-memory loads and C stores are scalar, while `sgemmv1.cu` uses vectorized `float4` loads/stores.
- Current cooperative loads keep general boundary zero-fill logic, while `sgemmv1.cu` assumes aligned shapes and has a leaner fast path.
- `148 registers/thread` limits this `BK=32` version to one 256-thread CTA per SM on H100's 65536-register SM budget.
- The kernel still has no double buffering and no Tensor Core/MMA path.

Raw artifacts in this directory:

- `build.log`
- `correctness.txt`
- `memcheck_130.txt`
- `perf.csv`
- `perf_comparison.csv`
- `performance_by_commit.csv`
