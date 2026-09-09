# CUTLASS SM90 TF32 GEMM Baseline

Date: 2026-09-10

Host: `gpu5`

CUDA device: `CUDA_VISIBLE_DEVICES=0`, NVIDIA H100 80GB HBM3, SM 9.0, 132 SMs

## Files

- `cutlass_tf32_bench.cu`
- `run_cutlass_tf32_bench.sh`
- `results/cutlass_tf32_sm90_20260910/benchmark_tf32.log`
- `results/cutlass_tf32_sm90_20260910/gpu_snapshot.txt`

## How To Run

```bash
cd /SharedData/dengzy/kernel
CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda bash ./run_cutlass_tf32_bench.sh
```

Single shape:

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_HOME=/usr/local/cuda bash ./run_cutlass_tf32_bench.sh \
  --shape 4096x4096x4096 --warmup 5 --repeat 20
```

## Kernel Contract

- Computes row-major `C[M, N] = A[M, K] * B[K, N]`.
- Public benchmark tensors are float.
- CUTLASS path materializes `B_col` for the CUTLASS B operand because this SM90 collective uses a column-major B layout.
- CUTLASS execution time excludes the host-side materialization of `B_col`; the timing isolates the GEMM kernel.
- cuBLAS reference uses original row-major `A` and `B`.

## Implementation Notes

- Uses CUTLASS 3 `GemmUniversalAdapter`.
- Mainloop is built by `cutlass::gemm::collective::CollectiveBuilder`.
- Epilogue is built by `cutlass::epilogue::collective::CollectiveBuilder`.
- Data type:
  - A/B/C/D storage: `float`
  - accumulator: `float`
  - Tensor Core compute: TF32 selected by SM90 TensorOp builder for float inputs
- Layout:
  - A: row-major
  - B: column-major internal buffer `B_col`
  - C/D: row-major
- Tile:
  - `128 x 128 x 64`
- Cluster:
  - `2 x 1 x 1`
- Stage count and schedule:
  - auto, with epilogue shared-memory carveout

## PTX Check

The generated PTX contains WGMMA:

```text
wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32
```

It also contains Hopper TMA-style bulk copy:

```text
cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes
cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
```

## Build Notes

Compile target is `sm_90a`.

ptxas reports:

```text
Used 168 registers, used 1 barriers
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
```

ptxas also reports:

```text
Potential Performance Loss: wgmma.mma_async instructions are serialized due to wgmma pipeline crossing function boundary
```

The warning appears in this standalone benchmark build, but the generated code still uses the expected WGMMA/TMA path.

## Performance

The theoretical TF32 Tensor Core dense peak used by the script is `494.7 TFLOPS`.

| Shape | CUTLASS ms | CUTLASS TFLOPS | Peak % | cuBLAS FAST_TF32 ms | cuBLAS FAST_TF32 TFLOPS | cuBLAS Peak % | Check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 512x512x512 | 0.016872 | 15.9101 | 3.2161% | 0.009710 | 27.6441 | 5.5881% | PASS |
| 1024x1024x1024 | 0.025099 | 85.5598 | 17.2953% | 0.016618 | 129.2295 | 26.1228% | PASS |
| 2048x2048x2048 | 0.084282 | 203.8389 | 41.2045% | 0.057326 | 299.6851 | 60.5792% | PASS |
| 4096x4096x4096 | 0.689306 | 199.3875 | 40.3047% | 0.381006 | 360.7261 | 72.9181% | PASS |

## Comparison

4096 square GEMM:

| Version | TFLOPS | Peak % |
| --- | ---: | ---: |
| direct WMMA baseline | 16.3879 | 3.31% |
| shared-memory WMMA baseline | 14.7185 | 2.98% |
| hand-written CuTe WGMMA baseline | 95.9240 | 19.3903% |
| simple CUTLASS SM90 builder baseline | 199.3875 | 40.3047% |
| cuBLAS FAST_TF32 in this run | 360.7261 | 72.9181% |

The simple CUTLASS builder version is:

- `2.08x` faster than the hand-written CuTe WGMMA baseline at 4096.
- `13.55x` faster than the shared-memory WMMA baseline at 4096.
- `55.27%` of cuBLAS FAST_TF32 throughput at 4096 in this run.

## Takeaway

This is a simple CUTLASS implementation, but it is not a naive kernel. The builder selected a Hopper SM90 path using WGMMA plus TMA bulk copies. That explains why it is much faster than the hand-written CuTe WGMMA baseline, which uses a smaller `64x64x64` CTA tile and per-thread `cp.async` instead of TMA producer/consumer scheduling.
