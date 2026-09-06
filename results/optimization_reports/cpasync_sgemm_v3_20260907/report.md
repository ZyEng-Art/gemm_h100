# cp.async GEMM and SGEMM V3 Results - 2026-09-07

This report records the current `gemm.cu` snapshot and the `sgemm_v3.cu` kernel-body snapshot. Measurements were collected on `h100-gpu3`, GPU0, with the benchmark default FP32 CUDA-core peak model.

## Source Snapshots

- `gemm.cu`: current cp.async double-buffered FP32 CUDA-core GEMM.
- `sgemm_v3.cu`: SGEMM V3 kernel-body snapshot, kept in the same style as `sgemm_v2.cu`.
- `sgemm_v3_fixed_bench.cu`: full benchmark wrapper used for valid SGEMM V3 performance numbers. The original generated SGEMM V3 wrapper was also tested and failed correctness, so its TFLOPS are not treated as valid performance.

## Verification

Current `gemm.cu`:

```text
compute-sanitizer --tool racecheck, shape 128x128x32: 0 hazards, passed=1
512/1024/2048/4096: passed=1
```

SGEMM V3 fixed wrapper:

```text
512/1024/2048/4096: passed=1
```

Original SGEMM V3 wrapper:

```text
512/1024/2048/4096: passed=0, see sgemm_v3_original_failed.csv
```

## Build Resources

Current `gemm.cu`:

```text
registers/thread: 127
shared memory/block: 32768 B
spill stores: 0 B
spill loads: 0 B
```

SGEMM V3 fixed wrapper:

```text
registers/thread: 127
shared memory/block: 16384 B
spill stores: 0 B
spill loads: 0 B
```

## Performance Summary

Timing command shape set: `--sizes 512,1024,2048,4096 --warmup 10 --repeat 50 --csv`.
The theoretical peak used by the benchmark is 66.9082 TFLOPS FP32 CUDA-core peak.

| M=N=K | current ms | current TFLOPS | current peak % | SGEMM V3 fixed ms | SGEMM V3 fixed TFLOPS | SGEMM V3 fixed peak % | V3/current |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.064536 | 4.1595 | 6.2167 | 0.050891 | 5.2747 | 7.8835 | 1.2681x |
| 1024 | 0.125863 | 17.0621 | 25.5007 | 0.099873 | 21.5022 | 32.1369 | 1.2602x |
| 2048 | 0.451627 | 38.0399 | 56.8539 | 0.363180 | 47.3041 | 70.7000 | 1.2435x |
| 4096 | 3.561210 | 38.5934 | 57.6811 | 2.836300 | 48.4571 | 72.4233 | 1.2556x |

## Notes

- The current cp.async version is correctness-clean for the recorded default aligned shapes and has a clean racecheck smoke test.
- Its 4096 score is 38.5934 TFLOPS, 57.6811% of the FP32 CUDA-core peak.
- SGEMM V3 fixed reaches 48.4571 TFLOPS on 4096, 72.4233% of the same peak, about 1.2556x faster than the current cp.async `gemm.cu`.
- The main performance gap is not only async global-to-shared copy. SGEMM V3 stages A/B values into registers and performs a cleaner 8x8 register outer-product per K step, while the current `gemm.cu` still has heavier shared-memory load structure and more synchronization/cp.async overhead from `BK=16`.
- Both paths are aligned-shape fast paths. Additional masked tails are needed before treating arbitrary non-multiple shapes as fully supported.

## Files

```text
current_gemm.csv
current_gemm_build.log
current_gemm_racecheck_128_32.log
sgemm_v3_fixed.csv
sgemm_v3_fixed_build.log
sgemm_v3_fixed_bench.cu
sgemm_v3_original_failed.csv
summary.csv
run_info.txt
```
