# SGEMM V2 Snapshot Report

Date: 2026-09-04

Host: `h100-gpu3`

GPU: `NVIDIA H100 80GB HBM3`

Source fragment: `/SharedData/dengzy/kernel/sgemm_v2.cu`

Repository base commit: `03139ba` (`Add B-split float4 GEMM experiment`)

## Source Status

`sgemm_v2.cu` is a kernel-body fragment, not a standalone benchmark file. It expects the caller to provide:

- macros: `BM`, `BN`, `BK`, `TM`, `TN`, `OFFSET`, `FLOAT4`
- variables: `a`, `b`, `c`, `M`, `N`, `K`
- thread/block variables: `bx`, `by`, `tx`, `ty`, `tid`

For benchmarking, the fragment was wrapped in the existing `gemm.cu` benchmark harness without changing the fragment logic.

The benchmark wrapper uses:

| Parameter | Value |
|---|---:|
| `BM` | 128 |
| `BN` | 128 |
| `BK` | 8 |
| `TM` | 8 |
| `TN` | 8 |
| Threads/CTA | 256 |
| Shared memory/CTA | 8192 B |

`BK=8` is required by this V2 load mapping: `load_a_smem_k = (tid & 1) << 2` only covers the two A `float4` groups `k=0..3` and `k=4..7`, and `load_b_smem_k = tid >> 5` only covers B rows `k=0..7`.

## Build Resource Usage

Build command:

```bash
/usr/local/cuda-12.9/bin/nvcc -O3 -std=c++17 -arch=sm_90 -lineinfo \
  -Xptxas=-v .tmp/sgemm_v2_20260904/sgemm_v2_bench.cu -lcublas \
  -o .tmp/sgemm_v2_20260904/sgemm_v2_bench
```

ptxas result:

| Registers/thread | Shared memory/block | Stack frame | Spill stores | Spill loads |
|---:|---:|---:|---:|---:|
| 127 | 8192 B | 0 B | 0 B | 0 B |

## Correctness

Correctness was checked against cuBLAS FP32 pedantic reference with:

```bash
CUDA_VISIBLE_DEVICES=6 \
.tmp/sgemm_v2_20260904/sgemm_v2_bench \
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
.tmp/sgemm_v2_20260904/sgemm_v2_bench \
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
| 512x512x512 | 0.0735744 | 3.64849 | 5.45298 | 85.3333 | compute | 5.45298 |
| 1024x1024x1024 | 0.144406 | 14.8711 | 22.2262 | 170.667 | compute | 22.2262 |
| 2048x2048x2048 | 0.437907 | 39.2318 | 58.6352 | 341.333 | compute | 58.6352 |
| 4096x4096x4096 | 3.48404 | 39.4481 | 58.9586 | 682.667 | compute | 58.9586 |

## Comparison With Current Main GEMM

Current main GEMM commit: `03139ba`

Current main data source: `results/optimization_reports/b_split_float4_20260904/perf.csv`

| Shape | Current main TFLOPS | SGEMM V2 TFLOPS | V2 / current main | Current main faster by |
|---|---:|---:|---:|---:|
| 512x512x512 | 4.26847 | 3.64849 | 0.855x | 1.17x |
| 1024x1024x1024 | 17.3075 | 14.8711 | 0.859x | 1.16x |
| 2048x2048x2048 | 40.5117 | 39.2318 | 0.968x | 1.03x |
| 4096x4096x4096 | 40.7334 | 39.4481 | 0.968x | 1.03x |

## Shared Memory Bank Conflict

Fresh NCU counter collection is currently blocked for this user by `ERR_NVGPUCTRPERM`; see `ncu_permission_check.txt`.

The conflict table below uses the existing root-collected 4096x4096x4096 NCU CSV files from:

```text
results/profile_bankconflict/external_sgemm_compare/v1_4096_bank.csv
results/profile_bankconflict/external_sgemm_compare/v2_4096_bank.csv
```

Those source CSVs are also copied into this report directory as:

```text
shared_conflict_v1_4096_bank.csv
shared_conflict_v2_4096_bank.csv
```

Raw metric names:

```text
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum
```

Conflict change from SGEMM V1 to SGEMM V2:

| Metric | SGEMM V1 | SGEMM V2 | Delta | Change | V2 / V1 |
|---|---:|---:|---:|---:|---:|
| Shared load bank conflicts | 67,143,197 | 31,168 | -67,112,029 | -99.9536% | 0.000464x |
| Shared store bank conflicts | 792,520 | 4,824,453 | +4,031,933 | +508.7484% | 6.087484x |
| Shared total bank conflicts | 67,935,717 | 4,855,621 | -63,080,096 | -92.8526% | 0.071474x |

Normalized by shared-memory wavefront count:

| Metric | SGEMM V1 conflicts/wavefront | SGEMM V2 conflicts/wavefront | Delta | Change |
|---|---:|---:|---:|---:|
| Shared load | 0.40012276 | 0.00030953 | -0.39981323 | -99.9226% |
| Shared store | 0.08632055 | 0.36512758 | +0.27880703 | +322.9903% |
| Shared total | 0.38384446 | 0.04262775 | -0.34121671 | -88.8945% |

Interpretation:

- V2's transposed A shared-memory layout almost eliminates the shared-load bank conflict seen in V1.
- The same change makes shared stores into `s_a[BK][BM]` much less coalesced at the bank level, so store conflict increases by about `6.09x`.
- Total shared-memory bank conflicts still drop by about `92.85%`, because V1's load conflict was much larger than its store conflict.
- This conflict reduction does not fully translate into end-to-end speedup here. On the separately timed benchmark, current main GEMM is still about `1.03x` faster than SGEMM V2 at 4096.

## Comparison With Previous Commit 5d35995

Previous commit data source: `results/recheck_20260904/current_perf_no_check_gpu6.csv`

| Shape | `5d35995` TFLOPS | SGEMM V2 TFLOPS | V2 / `5d35995` |
|---|---:|---:|---:|
| 512x512x512 | 4.06575 | 3.64849 | 0.897x |
| 1024x1024x1024 | 16.5211 | 14.8711 | 0.900x |
| 2048x2048x2048 | 37.2506 | 39.2318 | 1.053x |
| 4096x4096x4096 | 37.5215 | 39.4481 | 1.051x |

## Memcheck

Aligned 512x512x512 memcheck command:

```bash
TMPDIR=/dev/shm/dengzy_compute_sanitizer_tmp \
CUDA_VISIBLE_DEVICES=6 \
/usr/local/cuda-12.9/bin/compute-sanitizer --tool memcheck \
  .tmp/sgemm_v2_20260904/sgemm_v2_bench \
  --shape 512x512x512 --warmup 1 --repeat 1 --no-check --no-cublas
```

Result:

```text
ERROR SUMMARY: 0 errors
```

## Notes

This snapshot is useful as a comparison point because it uses a transposed A shared-memory layout and preloads both A and B compute vectors into registers before the outer-product loop.

The current main GEMM remains faster on these runs, especially for 512 and 1024. For 2048 and 4096, SGEMM V2 is close but still behind current main by about `3.2%` in TFLOPS.

Raw artifacts in this directory:

- `run_info.txt`
- `benchmark_wrapper_kernel.txt`
- `build.log`
- `benchmark.txt`
- `perf.csv`
- `perf_comparison.csv`
- `shared_conflict_comparison.csv`
- `shared_conflict_v1_4096_bank.csv`
- `shared_conflict_v2_4096_bank.csv`
- `ncu_permission_check.txt`
- `memcheck_512.txt`
