# WGMMA TF32 CuTe Baseline

Date: 2026-09-09

Host: `gpu5`

CUDA device: `CUDA_VISIBLE_DEVICES=0`, NVIDIA H100 80GB HBM3, SM 9.0, 132 SMs

## Files

- `wgmma_tf32_cute_bench.cu`
- `run_wgmma_tf32_bench.sh`
- `results/wgmma_tf32_cute_20260909/benchmark_tf32.log`
- `results/wgmma_tf32_cute_20260909/gpu_snapshot.txt`

## How To Run

```bash
cd /SharedData/dengzy/kernel
CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda bash ./run_wgmma_tf32_bench.sh
```

Single shape:

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda bash ./run_wgmma_tf32_bench.sh \
  --shape 4096x4096x4096 --warmup 5 --repeat 20
```

## Kernel Contract

- Computes row-major `C[M, N] = A[M, K] * B[K, N]`.
- Host benchmark source tensors are float.
- WGMMA path uses explicit TF32 input buffers:
  - `A_tf32[M, K]`
  - `B_T_tf32[N, K]`
- `B_T_tf32` is the transpose of row-major `B[K, N]`, so WGMMA sees a logical `(N, K)` operand with contiguous K dimension.
- Accumulation is FP32.
- Fast path currently requires `M`, `N`, and `K` to be multiples of 64.

## Implementation Notes

- Uses CuTe SM90 GMMA atom:
  - `SM90_64x64x8_F32TF32TF32_SS_TN<>`
- CTA tile:
  - `64 x 64 x 64`
- Pipeline stages:
  - 2 shared-memory stages
- Shared-memory layout:
  - `GMMA::Layout_K_SW128_Atom<tfloat32_t>` for A
  - `GMMA::Layout_K_SW128_Atom<tfloat32_t>` for B
- Global-to-shared copy:
  - `SM80_CP_ASYNC_CACHEALWAYS<uint128_t>`
- PTX contains:
  - `wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32`

## Build Notes

Compile target is `sm_90a`.

ptxas reports:

```text
Used 64 registers, used 1 barriers
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
```

ptxas also reports:

```text
Potential Performance Loss: wgmma.mma_async instructions are serialized due to wgmma pipeline crossing function boundary
```

This means the file is a correct learning baseline, but the WGMMA issue/commit/wait sequence is not yet a tuned high-throughput pipeline.

## Performance

The theoretical TF32 Tensor Core dense peak used by the script is `494.7 TFLOPS`.

| Shape | Custom WGMMA ms | Custom WGMMA TFLOPS | Peak % | cuBLAS FAST_TF32 ms | cuBLAS FAST_TF32 TFLOPS | cuBLAS Peak % | Check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 512x512x512 | 0.012722 | 21.1008 | 4.2654% | 0.009763 | 27.4946 | 5.5578% | PASS |
| 1024x1024x1024 | 0.031043 | 69.1773 | 13.9837% | 0.017030 | 126.0971 | 25.4896% | PASS |
| 2048x2048x2048 | 0.209912 | 81.8432 | 16.5440% | 0.057851 | 296.9665 | 60.0296% | PASS |
| 4096x4096x4096 | 1.432790 | 95.9240 | 19.3903% | 0.384704 | 357.2590 | 72.2173% | PASS |

## Comparison With Previous Tensor-Core Baselines

Previous direct WMMA baseline, 4096:

- `16.3879 TFLOPS`

Previous shared-memory WMMA baseline, 4096:

- `14.7185 TFLOPS`

Current simple WGMMA baseline, 4096:

- `95.9240 TFLOPS`

So the simple WGMMA baseline is about:

- `6.55x` faster than the shared-memory WMMA baseline at 4096
- `5.85x` faster than the direct WMMA baseline at 4096

## Current Bottlenecks

- The CTA tile is only `64x64`, so each CTA has one warpgroup doing one WGMMA tile. This does not expose enough per-CTA work compared with CUTLASS/cuBLAS style kernels.
- Uses `cp.async`, not TMA. On Hopper, high-performance WGMMA GEMM normally uses TMA plus warp-specialized producer/consumer scheduling.
- The current loop waits with `cp_async_wait<0>()`, so copies are conservative and not deeply overlapped with WGMMA.
- ptxas warns that WGMMA instructions are serialized around a function boundary, which is a direct signal that the WGMMA pipeline is not optimal yet.
- Host-side benchmark materializes `B_T_tf32` before timing. This is intentional for isolating WGMMA syntax and layout, but a production row-major GEMM would either require a different B layout contract or a kernel/dataflow that handles row-major B directly.

## Next Optimization Directions

- Increase CTA tile, for example `128x128x64`, with `Layout<Shape<_2,_2,_1>>` or a similar tiled-MMA layout.
- Use 3 or more shared-memory stages and replace conservative `wait<0>` with a real overlapped schedule.
- Move toward TMA load for Hopper instead of per-thread `cp.async`.
- Add a real epilogue path for edge tiles instead of requiring multiples of 64.
- Profile with NCU sections for WGMMA issue rate, tensor pipe utilization, shared-memory conflicts, and stall reasons.
