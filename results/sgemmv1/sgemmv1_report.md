# SGEMM V1 Benchmark Report

Date: 2026-08-27

Host: `h100-gpu3`

GPU: `NVIDIA H100 80GB HBM3`

Source file: `sgemmv1.cu`

`sgemmv1.cu` is a snapshot of the current GEMM experiment version:

- `BM = 128`
- `BN = 128`
- `BK = 32`
- `TM = 8`
- `TN = 8`
- `TK = 8`
- CTA tile: `128 x 128`
- K tile: `32`
- Per-thread output tile: `8 x 8`
- Threads per CTA: `(BN / TN) x (BM / TM) = 16 x 16 = 256`
- Shared memory per CTA: `(BM * BK + BK * BN) * sizeof(float) = 32768 bytes`

## Build

Command:

```bash
TMPDIR=/SharedData/dengzy/kernel/.tmp \
/usr/local/cuda-12.9/bin/nvcc -O3 -std=c++17 -arch=sm_90 -lineinfo \
  --ptxas-options=-v sgemmv1.cu -lcublas \
  -o .tmp/sgemmv1/sgemmv1_bench
```

ptxas resource usage:

| Metric | Value |
|---|---:|
| Registers/thread | 40 |
| Shared memory/block | 32768 B |
| Stack frame | 256 B |
| Spill stores | 0 B |
| Spill loads | 0 B |

## Correctness

Command:

```bash
.tmp/sgemmv1/sgemmv1_bench \
  --shape 128x128x128 \
  --shape 130x130x130 \
  --shape 4096x4096x4096 \
  --warmup 1 --repeat 2 --no-cublas
```

Result:

| Shape | Time ms | TFLOPS | Peak % | max_abs | max_rel | bad_count | ok |
|---|---:|---:|---:|---:|---:|---:|---|
| 128x128x128 | 0.207 | 0.020 | 0.030 | 0 | 0 | 0 | yes |
| 130x130x130 | 0.251 | 0.017 | 0.026 | 0 | 0 | 0 | yes |
| 4096x4096x4096 | 47.242 | 2.909 | 4.348 | 0 | 0 | 0 | yes |

The `130x130x130` case checks non-multiple boundary handling.

## Memcheck

Command:

```bash
TMPDIR=/dev/shm/dengzy_compute_sanitizer_tmp \
/usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck \
  .tmp/sgemmv1/sgemmv1_bench \
  --shape 130x130x130 --warmup 1 --repeat 2 --no-cublas
```

Result:

```text
ERROR SUMMARY: 0 errors
```

## Performance

Performance command:

```bash
.tmp/sgemmv1/sgemmv1_bench \
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
| 512x512x512 | 268,435,456 | 0.786466 | 0.341319 | 0.510130 | 85.3333 | 66.9082 | compute | 0.510130 |
| 1024x1024x1024 | 2,147,483,648 | 1.609050 | 1.334630 | 1.994720 | 170.667 | 66.9082 | compute | 1.994720 |
| 2048x2048x2048 | 17,179,869,184 | 5.513870 | 3.115750 | 4.656760 | 341.333 | 66.9082 | compute | 4.656760 |
| 4096x4096x4096 | 137,438,953,472 | 47.250900 | 2.908700 | 4.347310 | 682.667 | 66.9082 | compute | 4.347310 |

## Notes

This version is correct for the tested repeated-launch and boundary cases, but performance is still low for H100 FP32 CUDA-core SGEMM. The likely bottlenecks remain:

- The `tmp_c[8][8]` accumulator creates a 256-byte per-thread stack frame.
- Each thread computes 64 C elements serially.
- The shared-memory load helper uses runtime division/modulo.
- Row/column bounds checks still sit inside the compute loop.
- Shared-memory layout has not been tuned for bank conflicts.
- There is no double buffering, vectorized load path, or Tensor Core/MMA path.

Raw artifacts:

- `build.log`
- `correctness.txt`
- `memcheck_130.txt`
- `perf.csv`
- `perf_summary.csv`
