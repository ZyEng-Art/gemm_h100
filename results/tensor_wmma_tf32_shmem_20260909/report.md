# Shared-Memory TF32 WMMA GEMM Benchmark

Date: 2026-09-09

Host path: `/SharedData/dengzy/kernel`

Code under test: `tensor_gemm_bench.cu`

## Summary

This version changes the TF32 WMMA kernel from direct global-memory WMMA loads to a shared-memory staged kernel.

- Uses a `64x64` C block tile.
- Uses `4x4` warps per block, 16 warps total, 512 threads per block.
- Each warp computes one `16x16` C WMMA tile.
- Stages `A[64x32]` and `B[32x64]` in shared memory for each K block.
- Uses `float4` vectorized global-to-shared copies.
- Uses TF32 WMMA fragments for A/B and FP32 accumulator fragments for C.
- Uses an aligned fast path: current TF32 validation requires `M/N` multiples of 64 and `K` multiple of 32.

This implementation is correct on the benchmarked aligned shapes, but it is not faster than the previous direct-global WMMA baseline. The main value of this version is to show the first shared-memory tiling structure and provide a measured comparison.

## Benchmark Command

The benchmark was run on an idle H100 in `h100-gpu5`:

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda \
  bash ./run_tensor_gemm_bench.sh --dtype tf32 --sizes 512,1024,2048,4096 --warmup 5 --repeat 20
```

`CUDA_HOME=/usr/local/cuda` selects CUDA 12.8 on `h100-gpu5`.

Original logs:

- `results/tensor_wmma_tf32_shmem_20260909/benchmark_tf32.log`
- `results/tensor_wmma_tf32_shmem_20260909/gpu_snapshot.txt`
- `results/tensor_wmma_tf32_shmem_20260909/sweep/*.log`

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

## Compile Information

For `tensor_gemm_kernel_tf32` in the selected `4x4, BK=32` shared-memory configuration:

- Registers per thread: 40
- Static shared memory per block: 16384 bytes
- Spill stores: 0 bytes
- Spill loads: 0 bytes
- Barriers: 1

For comparison, the previous direct-global WMMA baseline used 32 registers per thread, 0 bytes shared memory, and 0 barriers.

## Results

| Shape | Custom Time (ms) | Custom TFLOPS | Custom % Of TF32 TC Peak | Roofline Bound | cuBLAS Time (ms) | cuBLAS TFLOPS | cuBLAS % Of TF32 TC Peak | Correctness |
|---|---:|---:|---:|---|---:|---:|---:|---|
| 512x512x512 | 0.045234 | 5.9344 | 1.1996% | memory | 0.009498 | 28.2635 | 5.7133% | PASS |
| 1024x1024x1024 | 0.153890 | 13.9547 | 2.8208% | compute | 0.016837 | 127.5470 | 25.7827% | PASS |
| 2048x2048x2048 | 1.181592 | 14.5396 | 2.9391% | compute | 0.057990 | 296.2537 | 59.8855% | PASS |
| 4096x4096x4096 | 9.337840 | 14.7185 | 2.9752% | compute | 0.384622 | 357.3348 | 72.2326% | PASS |

## Comparison With Direct-Global WMMA Baseline

Baseline source: `results/tensor_wmma_tf32_simple_20260909/report.md`.

| Shape | Direct WMMA TFLOPS | Shared WMMA TFLOPS | Change |
|---|---:|---:|---:|
| 512x512x512 | 8.3911 | 5.9344 | -29.28% |
| 1024x1024x1024 | 15.3978 | 13.9547 | -9.37% |
| 2048x2048x2048 | 16.0640 | 14.5396 | -9.49% |
| 4096x4096x4096 | 16.3879 | 14.7185 | -10.19% |

## Sweep Summary

The selected configuration is the fastest tested shared-memory variant on 4096.

| WARPS_M x WARPS_N | BLOCK_K | SMEM/block | Registers/thread | 4096 TFLOPS |
|---|---:|---:|---:|---:|
| 1x4 | 32 | 10240 B | 40 | 13.4457 |
| 2x4 | 32 | 12288 B | 40 | 14.4207 |
| 4x4 | 32 | 16384 B | 40 | 14.7205 |
| 4x2 | 32 | 12288 B | 40 | 14.4432 |
| 2x8 | 32 | 20480 B | 40 | 14.2315 |
| 8x2 | 32 | 20480 B | 40 | 14.2789 |
| 2x4 | 64 | 24576 B | 56 | 14.3280 |
| 4x4 | 64 | 32768 B | 56 | 14.1296 |
| 4x2 | 64 | 24576 B | 56 | 14.5134 |

## Analysis

The shared-memory version reduces global-memory traffic by reusing A and B tiles across warps, but that does not translate into a speedup here.

The likely reason is that the previous direct-global WMMA baseline is not strongly limited by global memory for these large aligned shapes. The shared-memory version adds extra global-to-shared copy instructions, shared-to-fragment loads, two CTA synchronizations per `BLOCK_K=32` stage, more registers, and static shared-memory usage. Because this version does not use `cp.async` or double buffering, the copy and synchronization overhead is not hidden behind Tensor Core MMA work.

Next useful optimization would be to compute multiple WMMA tiles per warp or use a pipelined shared-memory design so global-to-shared staging overlaps with MMA computation.
