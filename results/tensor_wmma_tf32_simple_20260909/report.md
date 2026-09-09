# Simple TF32 WMMA GEMM Benchmark

Date: 2026-09-09

Host path: `/SharedData/dengzy/kernel`

Code under test: `tensor_gemm_bench.cu`

## Summary

This version replaces the TF32 Tensor Core stub with a minimal WMMA implementation.

- Enables `USER_TENSOR_GEMM_TF32_READY`.
- Uses one warp to compute one `16x16` C tile.
- Uses 4 warps per block, so one block computes a `16x64` C region.
- Uses TF32 WMMA fragments for A/B and FP32 accumulator fragments for C.
- Initializes the accumulator with `wmma::fill_fragment`.
- Validates TF32 shapes with `M/N` multiples of 16 and `K` multiple of 8.

This is a syntax-learning baseline, not an optimized Tensor Core GEMM. It does not use shared memory tiling, `cp.async`, double buffering, or tail handling for non-aligned shapes.

## Benchmark Command

The benchmark was rerun on an idle H100 in `h100-gpu5`:

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda \
  bash ./run_tensor_gemm_bench.sh --dtype tf32 --sizes 512,1024,2048,4096 --warmup 5 --repeat 20
```

`CUDA_HOME=/usr/local/cuda` selects CUDA 12.8 on `h100-gpu5`. The CUDA 13.0 headers on this host are not compatible with the current `cudaDeviceProp::clockRate` / `memoryClockRate` usage.

Original logs:

- `results/tensor_wmma_tf32_simple_20260909/benchmark_tf32.log`
- `results/tensor_wmma_tf32_simple_20260909/gpu_snapshot.txt`

## Hardware And Environment

- Host: `gpu5`
- Selected GPU: `CUDA_VISIBLE_DEVICES=0`
- GPU: NVIDIA H100 80GB HBM3
- Pre-run GPU state on `gpu5`: all 8 GPUs showed 0 MiB memory used and 0% GPU utilization.
- Compute capability: 9.0
- SM count: 132
- SM clock reported by CUDA runtime: 1980.0 MHz
- HBM bandwidth used by benchmark roofline: 3.350 TB/s
- Tensor Core dense TF32 peak used by benchmark: 494.700 TFLOPS
- Shared memory per block: 48.0 KiB
- Shared memory per block opt-in: 227.0 KiB
- Shared memory per SM: 228.0 KiB
- Registers per SM: 65536

## Compile Information

For `tensor_gemm_kernel_tf32`:

- Registers per thread: 32
- Spill stores: 0 bytes
- Spill loads: 0 bytes
- Shared memory: 0 bytes
- Barriers: 0

Current compile warnings:

- `row_num` is declared but not used.
- `col_num` is declared but not used.

## Results

| Shape | Custom Time (ms) | Custom TFLOPS | Custom % Of TF32 TC Peak | Roofline Bound | cuBLAS Time (ms) | cuBLAS TFLOPS | cuBLAS % Of TF32 TC Peak | Correctness |
|---|---:|---:|---:|---|---:|---:|---:|---|
| 512x512x512 | 0.031990 | 8.3911 | 1.6962% | memory | 0.009571 | 28.0462 | 5.6693% | PASS |
| 1024x1024x1024 | 0.139467 | 15.3978 | 3.1125% | compute | 0.016934 | 126.8119 | 25.6341% | PASS |
| 2048x2048x2048 | 1.069462 | 16.0640 | 3.2472% | compute | 0.058406 | 294.1436 | 59.4590% | PASS |
| 4096x4096x4096 | 8.386608 | 16.3879 | 3.3127% | compute | 0.384515 | 357.4344 | 72.2528% | PASS |

## Previous Loaded-Host Result

The first committed run was taken on `h100-gpu3` while GPU0 had other active processes. The idle `h100-gpu5` rerun supersedes it for performance comparison.

| Shape | Loaded Custom TFLOPS | Idle Custom TFLOPS | Change | Loaded cuBLAS TFLOPS | Idle cuBLAS TFLOPS | Change |
|---|---:|---:|---:|---:|---:|---:|
| 512x512x512 | 8.9536 | 8.3911 | -6.28% | 34.1764 | 28.0462 | -17.94% |
| 1024x1024x1024 | 8.8876 | 15.3978 | +73.25% | 142.1346 | 126.8119 | -10.78% |
| 2048x2048x2048 | 11.8891 | 16.0640 | +35.12% | 349.8784 | 294.1436 | -15.93% |
| 4096x4096x4096 | 10.6889 | 16.3879 | +53.32% | 266.2976 | 357.4344 | +34.22% |

## Notes

The custom kernel is much slower than cuBLAS because this version directly loads each warp's WMMA operands from global memory and does not reuse A/B tiles through shared memory. It is useful for confirming WMMA fragment, load, MMA, and store syntax before moving to a tiled Tensor Core kernel.
