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

Fresh NCU counter collection for the current B-split GEMM is currently blocked for this user by `ERR_NVGPUCTRPERM`; see `ncu_permission_check.txt`.

Because of that, this report does not claim a measured current-vs-previous shared conflict delta for `03139ba`. The table below records the available previous-version NCU reference and marks the current counter fields as `NA`.

Available previous-version NCU reference:

```text
source: results/current_diag/head_device1_bank.csv
copied: shared_conflict_prev_5d35995_4096_bank.csv
shape: 4096x4096x4096
kernel: gemm_kernel
block: (16, 16, 1)
grid: (32, 32, 1)
registers/thread: 128
```

Raw metric names:

```text
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum
```

| Metric | Previous `5d35995` | Previous wavefronts | Previous conflicts/wavefront | Current `03139ba` | Delta |
|---|---:|---:|---:|---:|---:|
| Shared load bank conflicts | 268,451,537 | 671,104,721 | 0.40001438 | NA | NA |
| Shared store bank conflicts | 481,387 | 34,035,819 | 0.01414354 | NA | NA |
| Shared total bank conflicts | 268,932,924 | 705,140,540 | 0.38138911 | NA | NA |

Known performance delta for the same current-vs-previous GEMM comparison is still valid and is recorded above. At `4096x4096x4096`, current B-split GEMM improves from `37.5215` TFLOPS to `40.7334` TFLOPS, a `1.09x` speedup.

Command to fill the missing current NCU counters when admin profiling is enabled:

```bash
CUDA_VISIBLE_DEVICES=7 \
/usr/local/cuda-12.9/bin/ncu \
  --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum \
  --target-processes all \
  --csv --page raw \
  --kernel-name gemm_kernel \
  .tmp/b_split_float4_20260904/gemm_bench \
  --shape 4096x4096x4096 --warmup 1 --repeat 1 --no-check --no-cublas
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
- `benchmark.txt`
- `perf.csv`
- `perf_comparison.csv`
- `shared_conflict_gemm_current_vs_prev.csv`
- `shared_conflict_prev_5d35995_4096_bank.csv`
- `perf_prev_5d35995_reference.csv`
- `ncu_permission_check.txt`
- `memcheck_512.txt`
