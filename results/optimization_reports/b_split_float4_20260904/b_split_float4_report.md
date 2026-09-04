# B-Split Float4 GEMM Report

Date: 2026-09-04

Host: `h100-gpu3`

GPU: `NVIDIA H100 80GB HBM3`

Source file: `/SharedData/dengzy/kernel/gemm.cu`

Base commit before this change: `5d35995` (`Simplify GEMM K loop`)

## Optimization Content

This version keeps the `BM=128`, `BN=128`, `BK=32`, `TM=8`, `TN=8` CUDA-core FP32 GEMM shape and changes how each thread maps its 8 output columns.

Previous mapping:

```cpp
col_start = block_col_start + threadIdx.x * TN;
```

Current mapping:

```cpp
col_start = block_col_start + threadIdx.x * TN / 2;
```

Each thread now computes two 4-column `float4` output segments:

- columns `threadIdx.x * 4 + [0, 3]`
- columns `threadIdx.x * 4 + BN / 2 + [0, 3]`

For every shared-memory K step, the kernel loads two B vectors:

```cpp
FLOAT4(r_b0[0]) = FLOAT4(shm_B[i * BN + threadIdx.x * TN / 2 + tn + BN / 2]);
FLOAT4(r_b1[0]) = FLOAT4(shm_B[i * BN + threadIdx.x * TN / 2 + tn]);
```

Then all four lanes from each vector are accumulated into the `tmp_c[8][8]` register tile.

This is an aligned-shape fast path. It is valid for the committed benchmark shapes (`512/1024/2048/4096`) where the `float4` loads and stores are aligned and complete. It is not a fully general tail-safe implementation for arbitrary `M/N/K`.

## Build Resource Usage

Build command:

```bash
TMPDIR=/SharedData/dengzy/kernel/.tmp \
/usr/local/cuda-12.9/bin/nvcc -O3 -std=c++17 -arch=sm_90 -lineinfo \
  -Xptxas=-v gemm.cu -lcublas \
  -o .tmp/b_split_float4_20260904/gemm_bench
```

ptxas result:

| Registers/thread | Shared memory/block | Stack frame | Spill stores | Spill loads |
|---:|---:|---:|---:|---:|
| 128 | 32768 B | 0 B | 0 B | 0 B |

## Correctness

Correctness was checked against cuBLAS FP32 pedantic reference with:

```bash
CUDA_VISIBLE_DEVICES=6 \
.tmp/b_split_float4_20260904/gemm_bench \
  --sizes 512,1024,2048,4096 \
  --warmup 5 --repeat 20
```

| Shape | max_abs | max_rel | bad_count | ok |
|---|---:|---:|---:|:--|
| 512x512x512 | 9.06e-06 | 5.84e-02 | 0 | yes |
| 1024x1024x1024 | 0.00e+00 | 0.00e+00 | 0 | yes |
| 2048x2048x2048 | 0.00e+00 | 0.00e+00 | 0 | yes |
| 4096x4096x4096 | 0.00e+00 | 0.00e+00 | 0 | yes |

## Score

Performance command:

```bash
CUDA_VISIBLE_DEVICES=6 \
.tmp/b_split_float4_20260904/gemm_bench \
  --sizes 512,1024,2048,4096 \
  --warmup 5 --repeat 20 \
  --no-check --no-cublas --csv
```

The benchmark uses the FP32 CUDA-core peak calculated from the measured H100 clock:

```text
peak = 132 SM * 128 FP32 cores/SM * 2 FLOPs/FMA * 1980 MHz
     = 66.9082 TFLOPS
```

| Shape | Time ms | TFLOPS | FP32 peak % | AI | Roofline bound | Roofline % |
|---|---:|---:|---:|---:|---|---:|
| 512x512x512 | 0.062888 | 4.26847 | 6.37959 | 85.3333 | compute | 6.37959 |
| 1024x1024x1024 | 0.124078 | 17.3075 | 25.8675 | 170.667 | compute | 25.8675 |
| 2048x2048x2048 | 0.424072 | 40.5117 | 60.5482 | 341.333 | compute | 60.5482 |
| 4096x4096x4096 | 3.37411 | 40.7334 | 60.8796 | 682.667 | compute | 60.8796 |

## Comparison With Previous Commit

Previous commit: `5d35995` (`Simplify GEMM K loop`)

Previous data source: `results/recheck_20260904/current_perf_no_check_gpu6.csv`

| Shape | Previous ms | Current ms | Time speedup | Previous TFLOPS | Current TFLOPS | TFLOPS speedup | Peak % delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512x512x512 | 0.0660237 | 0.062888 | 1.05x | 4.06575 | 4.26847 | 1.05x | +0.303 |
| 1024x1024x1024 | 0.129984 | 0.124078 | 1.05x | 16.5211 | 17.3075 | 1.05x | +1.175 |
| 2048x2048x2048 | 0.461197 | 0.424072 | 1.09x | 37.2506 | 40.5117 | 1.09x | +4.874 |
| 4096x4096x4096 | 3.66294 | 3.37411 | 1.09x | 37.5215 | 40.7334 | 1.09x | +4.801 |

## GEMM Shared Memory Bank Conflict

This section is for `gemm.cu` current B-split GEMM versus the previous GEMM commit. It is separate from the `sgemm_v1`/`sgemm_v2` snapshot comparison.

Current GEMM commit:

```text
03139ba Add B-split float4 GEMM experiment
```

Previous GEMM commit:

```text
5d35995 Simplify GEMM K loop
```

NCU counter collection was run inside Docker with `--cap-add=SYS_ADMIN --security-opt seccomp=unconfined`, because direct host-side NCU as the normal user hits `ERR_NVGPUCTRPERM`. The original failed non-Docker check is still kept in `ncu_permission_check.txt`.

Comparison setup:

```text
shape: 4096x4096x4096
kernel: gemm_kernel
block: (16, 16, 1)
grid: (32, 32, 1)
registers/thread: 128
static shared memory/block: 32768 B
allocated shared memory/block: 33792 B
warmup: 0
repeat: 1
correctness/cublas: disabled for profiling
previous raw: shared_conflict_prev_5d35995_4096_bank.csv
current raw: shared_conflict_current_03139ba_4096_bank.csv
summary: shared_conflict_gemm_current_vs_prev.csv
```

Raw metric names:

```text
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum
```

| Metric | Previous `5d35995` | Previous wavefronts | Previous conflicts/wavefront | Current `03139ba` | Current wavefronts | Current conflicts/wavefront | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Shared load bank conflicts | 268,449,772 | 671,102,956 | 0.40001280 | 73,183 | 402,726,367 | 0.00018172 | -99.9727% |
| Shared store bank conflicts | 823,271 | 34,377,703 | 0.02394782 | 786,535 | 34,340,967 | 0.02290369 | -4.4622% |
| Shared total bank conflicts | 269,273,043 | 705,480,659 | 0.38168735 | 859,718 | 437,067,334 | 0.00196701 | -99.6807% |

Known performance delta for the same current-vs-previous GEMM comparison is still valid and is recorded above. At `4096x4096x4096`, current B-split GEMM improves from `37.5215` TFLOPS to `40.7334` TFLOPS, a `1.09x` speedup.

### Cross Check Against SGEMM V2

This is a counter-level comparison, not a strict one-change ablation: current `gemm.cu` uses the B-split `BK=32` implementation, while `sgemm_v2.cu` is a wrapped snapshot using its own `BK=8` mapping. The comparison is still useful for identifying the store-conflict mechanism.

Summary file:

```text
gemm_current_vs_sgemm_v2_shared_conflict.csv
```

| Metric | Current `gemm.cu` | SGEMM V2 | SGEMM V2 / current |
|---|---:|---:|---:|
| Shared load bank conflicts | 73,183 | 31,168 | 0.43x |
| Shared load conflicts/wavefront | 0.00018172 | 0.00030953 | 1.70x |
| Shared store bank conflicts | 786,535 | 4,824,453 | 6.13x |
| Shared store conflicts/wavefront | 0.02290369 | 0.36512758 | 15.94x |
| Shared total bank conflicts | 859,718 | 4,855,621 | 5.65x |
| Shared total conflicts/wavefront | 0.00196701 | 0.04262775 | 21.67x |

Interpretation:

- Current `gemm.cu` has much lower store conflict than SGEMM V2: `83.70%` lower by raw store conflict count and `93.73%` lower after normalizing by store wavefronts.
- The store-conflict gap is mainly explained by SGEMM V2's transposed A shared-memory write pattern. It writes `s_a[load_a_smem_k + q][load_a_smem_m]` into a `s_a[BK][BM]` layout. With `BM=128`, `bank(s_a[k][m]) = (k * 128 + m) % 32 = m % 32`; paired lanes can have the same `m` and different `k`, so they write different addresses in the same bank.
- This means SGEMM V2 trades shared-load conflict reduction for a store-side conflict cost. Current `gemm.cu` mostly keeps shared stores as vectorized contiguous row-major stores, so it avoids most of that store-side penalty.

Docker NCU command pattern used for these raw files:

```bash
docker run --rm --device nvidia.com/gpu=7 \
  --cap-add=SYS_ADMIN --security-opt seccomp=unconfined \
  -v /SharedData/dengzy/kernel:/work \
  -v /usr/local/cuda-12.9:/usr/local/cuda-12.9:ro \
  -w /work nvcr.io/nvidia/pytorch:26.04-py3 \
  env -u BASH_ENV bash --noprofile --norc \
  /work/.tmp/gemm_conflict_20260904/run_ncu_gemm_current_4096.sh
```

## Memcheck

Aligned 512x512x512 memcheck command:

```bash
TMPDIR=/dev/shm/dengzy_compute_sanitizer_tmp \
CUDA_VISIBLE_DEVICES=6 \
/usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck \
  .tmp/b_split_float4_20260904/gemm_bench \
  --shape 512x512x512 --warmup 1 --repeat 1 --no-check --no-cublas
```

Result:

```text
ERROR SUMMARY: 0 errors
```

## Artifacts

Raw artifacts in this directory:

- `run_info.txt`
- `build.log`
- `build_current_for_conflict.log`
- `build_prev_5d35995_for_conflict.log`
- `benchmark.txt`
- `perf.csv`
- `perf_comparison.csv`
- `gemm_current_vs_sgemm_v2_shared_conflict.csv`
- `shared_conflict_gemm_current_vs_prev.csv`
- `shared_conflict_prev_5d35995_4096_bank.csv`
- `shared_conflict_prev_5d35995_4096_bank.stderr`
- `shared_conflict_current_03139ba_4096_bank.csv`
- `shared_conflict_current_03139ba_4096_bank.stderr`
- `perf_prev_5d35995_reference.csv`
- `ncu_permission_check.txt`
- `memcheck_512.txt`
