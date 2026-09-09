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

```bash
bash ./run_tensor_gemm_bench.sh --dtype tf32 --sizes 512,1024,2048,4096 --warmup 5 --repeat 20
```

Original logs:

- `results/tensor_wmma_tf32_simple_20260909/benchmark_tf32.log`
- `results/tensor_wmma_tf32_simple_20260909/gpu_snapshot.txt`

## Hardware And Environment

- GPU: NVIDIA H100 80GB HBM3
- Compute capability: 9.0
- SM count: 132
- SM clock reported by CUDA runtime: 1980.0 MHz
- HBM bandwidth used by benchmark roofline: 3.350 TB/s
- Tensor Core dense TF32 peak used by benchmark: 494.700 TFLOPS
- Shared memory per block: 48.0 KiB
- Shared memory per block opt-in: 227.0 KiB
- Shared memory per SM: 228.0 KiB
- Registers per SM: 65536

The captured `nvidia-smi` snapshot showed other processes occupying GPU memory and using the GPUs during the run. Treat these numbers as a current-environment reference rather than an isolated peak-performance measurement.

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
| 512x512x512 | 0.029981 | 8.9536 | 1.8099% | memory | 0.007854 | 34.1764 | 6.9085% | PASS |
| 1024x1024x1024 | 0.241627 | 8.8876 | 1.7966% | compute | 0.015109 | 142.1346 | 28.7315% | PASS |
| 2048x2048x2048 | 1.445010 | 11.8891 | 2.4033% | compute | 0.049102 | 349.8784 | 70.7254% | PASS |
| 4096x4096x4096 | 12.858126 | 10.6889 | 2.1607% | compute | 0.516110 | 266.2976 | 53.8301% | PASS |

## Notes

The custom kernel is much slower than cuBLAS because this version directly loads each warp's WMMA operands from global memory and does not reuse A/B tiles through shared memory. It is useful for confirming WMMA fragment, load, MMA, and store syntax before moving to a tiled Tensor Core kernel.
