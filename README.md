# H100 GEMM Benchmark

CUDA FP32 GEMM benchmark and roofline harness for an NVIDIA H100 80GB HBM3 GPU.

The current kernel is a correctness-first FP32 CUDA core GEMM baseline with shared-memory tiling. The benchmark reports:

- GPU hardware information
- average kernel time
- custom kernel TFLOPS
- cuBLAS FP32 reference timing
- correctness against cuBLAS
- theoretical peak percentage
- arithmetic intensity
- roofline memory/compute bound
- roofline utilization
- SVG roofline plot

## Files

```text
gemm.cu                  CUDA GEMM kernel + benchmark harness
run_gemm_bench.sh        compile and run benchmark
plot_roofline.py         generate roofline.svg from CSV without matplotlib
run_gemm_roofline.sh     run CSV benchmark and generate roofline.svg
results/                 benchmark outputs committed from h100-gpu3
```

## Kernel Contract

The custom kernel computes:

```text
C[M, N] = A[M, K] * B[K, N]
```

All matrices are row-major `float`.

The kernel entry point is:

```cpp
__global__ void gemm_kernel(const float* A,
                            const float* B,
                            float* C,
                            int M,
                            int N,
                            int K);
```

## Current Optimization Content

The current kernel uses a simple `GEMM_TILE=16` shared-memory tiled implementation:

- one CUDA block computes one `16 x 16` tile of C
- one thread computes one C element
- A and B tiles are staged through shared memory
- accumulation is done in FP32 using normal CUDA core FMA

This is intentionally a baseline for further optimization. It is not Tensor Core code.

Expected next optimization directions:

- larger block tiles, such as `64x64` or `128x64`
- register tiling so each thread computes multiple C elements
- improved data reuse from shared memory
- fewer synchronizations per FMA
- optional TF32/FP16/BF16 Tensor Core implementation if changing the datatype/precision target

Current caveat: the baseline is intended for benchmark sizes that are multiples of `GEMM_TILE`. Default committed results use `512/1024/2048/4096`, so correctness passes. If testing arbitrary non-multiple shapes, add full boundary zeroing and guarded C stores before relying on correctness.

## Run

On `h100-gpu3`:

```bash
cd /SharedData/dengzy/kernel
./run_gemm_bench.sh
```

Run one shape:

```bash
./run_gemm_bench.sh --shape 4096x4096x4096 --warmup 5 --repeat 50
```

Generate CSV and roofline SVG:

```bash
./run_gemm_roofline.sh
```

Outputs:

```text
gemm_roofline.csv
roofline.svg
```

The script automatically uses a local `.tmp` directory under the project when `TMPDIR` is unset. This avoids failures when `/tmp` is full.

## Theory

Measured GEMM throughput:

```text
FLOPs = 2 * M * N * K
TFLOPS = FLOPs / seconds / 1e12
```

Default compute peak is FP32 CUDA core peak:

```text
peak = SM_count * 128 FP32 cores/SM * 2 FLOPs/FMA * clock
```

For the measured H100:

```text
SM_count = 132
clock = 1980 MHz
peak = 66.908 TFLOPS
```

Roofline uses ideal minimum DRAM traffic:

```text
ideal_dram_bytes = 4 * (M*K + K*N + M*N)
AI = FLOPs / ideal_dram_bytes
mem_roof = AI * HBM_TBps
roofline = min(compute_peak, mem_roof)
roof_% = measured_TFLOPS / roofline * 100
```

Default HBM bandwidth for roofline:

```text
3.35 TB/s
```

Override with:

```bash
./run_gemm_bench.sh --hbm-tbps 3.35
```

## Current Results

Committed result files:

```text
results/default_benchmark.txt
results/gemm_roofline.csv
results/roofline.svg
results/run_info.txt
```

Benchmark environment:

```text
GPU: NVIDIA H100 80GB HBM3
SM count: 132
SM clock: 1980.0 MHz
FP32 CUDA core peak used: 66.908 TFLOPS
HBM bandwidth used for roofline: 3.350 TB/s
Timing: warmup=5, repeat=20
cuBLAS reference: fp32_pedantic
```

Summary from `results/default_benchmark.txt`:

| M=N=K | custom ms | custom TFLOPS | peak % | AI | roofline bound | roof % | cuBLAS TFLOPS | ok |
|---:|---:|---:|---:|---:|---|---:|---:|:--|
| 512 | 0.042 | 6.414 | 9.586 | 85.333 | compute | 9.586 | 18.100 | yes |
| 1024 | 0.299 | 7.179 | 10.729 | 170.667 | compute | 10.729 | 38.074 | yes |
| 2048 | 2.292 | 7.494 | 11.201 | 341.333 | compute | 11.201 | 50.773 | yes |
| 4096 | 18.265 | 7.525 | 11.246 | 682.667 | compute | 11.246 | 51.716 | yes |

The default shapes are all compute-bound under the ideal DRAM roofline model. The baseline kernel reaches roughly `9.6% - 11.2%` of the FP32 CUDA core roofline, while cuBLAS FP32 pedantic reaches roughly `27% - 77%` on these sizes.

