# BM128/BN128/BK32 GEMM Optimization Report

Date: 2026-08-27

Host: `h100-gpu3`

GPU: `NVIDIA H100 80GB HBM3`

Compared versions:

- Previous: repository `HEAD` before this optimization, commit `1fcccfe`
- Current: working-tree `gemm.cu` with `BM=128`, `BN=128`, `BK=32`, `TM=8`, `TN=8`, `TK=8`

## Optimization Content

The current version changes the original one-output-per-thread tiled GEMM into a larger block tile design:

- C tile per CTA: `BM x BN = 128 x 128`
- K tile: `BK = 32`
- Per-thread output tile: `TM x TN = 8 x 8`
- Threads per CTA: `(BN / TN) x (BM / TM) = 16 x 16 = 256`
- Shared memory:
  - `shm_A[BM * BK]`
  - `shm_B[BK * BN]`
  - Total: `(128 * 32 + 32 * 128) * 4 = 32768 bytes`
- A/B tiles are cooperatively loaded into shared memory by all CTA threads.
- Out-of-bound global loads are zero-filled in shared memory.
- Accumulation is held in a per-thread `tmp_c[TM][TN]` tile and written to C after all K tiles.
- Grid mapping uses `grid.x = ceil(N / BN)`, `grid.y = ceil(M / BM)`.

This version is a correctness/stability step for a register-tiled shared-memory GEMM. It is not yet a performance improvement.

## Build Resource Usage

Compiled with:

```bash
nvcc -O3 -std=c++17 -arch=sm_90 -lineinfo --ptxas-options=-v gemm.cu -lcublas
```

| Version | Registers/thread | Shared memory/block | Stack frame | Spill stores | Spill loads |
|---|---:|---:|---:|---:|---:|
| Previous | 32 | 2048 B | 0 B | 0 B | 0 B |
| Current | 40 | 32768 B | 256 B | 0 B | 0 B |

The current version uses substantially more shared memory and has a 256-byte stack frame from the per-thread `tmp_c[8][8]` accumulator.

## Correctness

Correctness was checked against cuBLAS FP32 pedantic reference.

Previous version:

```bash
.tmp/compare/gemm_prev_bench \
  --shape 128x128x128 \
  --shape 4096x4096x4096 \
  --warmup 1 --repeat 2 --no-cublas
```

Result:

| Shape | ok | max_abs | max_rel | bad_count |
|---|---|---:|---:|---:|
| 128x128x128 | yes | 0 | 0 | 0 |
| 4096x4096x4096 | yes | 0 | 0 | 0 |

Current version:

```bash
.tmp/compare/gemm_current_bench \
  --shape 128x128x128 \
  --shape 130x130x130 \
  --shape 4096x4096x4096 \
  --warmup 1 --repeat 2 --no-cublas
```

Result:

| Shape | ok | max_abs | max_rel | bad_count |
|---|---|---:|---:|---:|
| 128x128x128 | yes | 0 | 0 | 0 |
| 130x130x130 | yes | 0 | 0 | 0 |
| 4096x4096x4096 | yes | 0 | 0 | 0 |

The `130x130x130` case verifies non-multiple boundary handling for the current version.

`compute-sanitizer` memcheck was also run on the current version:

```bash
TMPDIR=/dev/shm/dengzy_compute_sanitizer_tmp \
/usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck \
  .tmp/compare/gemm_current_bench \
  --shape 130x130x130 --warmup 1 --repeat 2 --no-cublas
```

Result:

```text
ERROR SUMMARY: 0 errors
```

## Performance Comparison

Performance was measured with correctness and cuBLAS timed baseline disabled:

```bash
--sizes 512,1024,2048,4096 --warmup 5 --repeat 20 --no-check --no-cublas --csv
```

FLOPs are computed as:

```text
FLOPs = 2 * M * N * K
```

The factor 2 counts each multiply-add as two floating-point operations.

| Shape | FLOPs | Previous ms | Current ms | Time ratio | Previous TFLOPS | Current TFLOPS | TFLOPS ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512x512x512 | 268,435,456 | 0.041680 | 0.786862 | 18.879x slower | 6.440390 | 0.341147 | 0.053x |
| 1024x1024x1024 | 2,147,483,648 | 0.299022 | 1.610400 | 5.386x slower | 7.181680 | 1.333510 | 0.186x |
| 2048x2048x2048 | 17,179,869,184 | 2.291050 | 5.515270 | 2.407x slower | 7.498680 | 3.114960 | 0.415x |
| 4096x4096x4096 | 137,438,953,472 | 18.174900 | 47.215900 | 2.599x slower | 7.562030 | 2.910860 | 0.385x |

## Interpretation

The current version fixes the earlier repeated-launch correctness issue by accumulating into `tmp_c` and writing C once after the K loop. It also handles non-multiple boundaries correctly, as shown by `130x130x130` and `compute-sanitizer`.

However, performance regresses relative to the previous simple tiled kernel:

- Each thread computes `8 x 8 = 64` C elements serially, which increases per-thread work and reduces available parallelism.
- `tmp_c[8][8]` creates a 256-byte stack frame, so the accumulator may not be held as efficiently as intended.
- The cooperative load helper uses runtime `row = i / bn` and `col = i % bn`; because `bn` is a runtime parameter, this may generate expensive integer division/modulo.
- The compute loop still has row/column bounds checks inside the innermost path.
- Shared memory access patterns are not yet optimized for bank conflicts or vectorized loads.
- There is no double buffering or Tensor Core/MMA path.

Next optimization targets:

- Specialize shared-memory load loops for compile-time `BM/BN/BK` constants.
- Move row/column bounds checks out of the innermost compute loop and keep them only on final stores.
- Encourage the compiler to keep accumulators in registers, or reduce `TM/TN`.
- Add vectorized global loads where alignment allows.
- Check shared-memory bank conflicts with `ncu` inside the profiling container.
- Consider a warp-tiled design or Tensor Core MMA for meaningful H100 performance.

Raw artifacts:

- `build_prev.log`
- `build_current.log`
- `correctness_prev.txt`
- `correctness_current.txt`
- `memcheck_current_130.txt`
- `perf_prev.csv`
- `perf_current.csv`
- `perf_comparison.csv`
