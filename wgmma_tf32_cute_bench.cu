// Standalone TF32 WGMMA GEMM experiment for H100.
//
// Contract:
//   Row-major C[M, N] = A[M, K] * B[K, N]
//   Benchmark source data is float.
//   The WGMMA path receives explicit TF32 buffers: A_tf32[M, K] and
//   B_T_tf32[N, K]. This keeps the public GEMM contract row-major while giving
//   WGMMA the K-major shared-memory operands it expects.
//   Compute path is TF32 WGMMA with FP32 accumulation.
//
// This file intentionally uses CuTe's SM90 GMMA atom so the code can focus on
// the Hopper WGMMA dataflow instead of manually decoding accumulator layouts.

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cutlass/tfloat32.h>
#include <cute/tensor.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t status = (expr);                                             \
    if (status != cudaSuccess) {                                             \
      std::ostringstream oss;                                                \
      oss << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "        \
          << cudaGetErrorString(status);                                     \
      throw std::runtime_error(oss.str());                                   \
    }                                                                        \
  } while (0)

#define CUBLAS_CHECK(expr)                                                   \
  do {                                                                       \
    cublasStatus_t status = (expr);                                          \
    if (status != CUBLAS_STATUS_SUCCESS) {                                   \
      std::ostringstream oss;                                                \
      oss << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << ": "      \
          << static_cast<int>(status);                                       \
      throw std::runtime_error(oss.str());                                   \
    }                                                                        \
  } while (0)

namespace wgmma_bench {

using namespace cute;

constexpr double kH100Tf32TensorPeakTflops = 494.7;
constexpr double kH100HbmTbps = 3.350;

struct Problem {
  int M;
  int N;
  int K;
};

struct Config {
  int device = 0;
  int warmup = 5;
  int repeat = 20;
  bool check = true;
  bool run_cublas = true;
  std::vector<Problem> problems;
};

struct CheckResult {
  bool pass = false;
  double max_abs = 0.0;
  double max_rel = 0.0;
  std::size_t bad_count = 0;
};

template <class ElementA,
          class ElementB,
          class SmemLayoutA,
          class SmemLayoutB>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
};

template <class ProblemShape,
          class CtaTiler,
          class ASmemLayout,
          class TiledCopyA,
          class BSmemLayout,
          class TiledCopyB,
          class TiledMma>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value) void
wgmma_tf32_kernel(ProblemShape shape_MNK,
                  CtaTiler cta_tiler,
                  const tfloat32_t* A,
                  const tfloat32_t* B_T,
                  float* C,
                  ASmemLayout sA_layout,
                  TiledCopyA copy_a,
                  BSmemLayout sB_layout,
                  TiledCopyB copy_b,
                  TiledMma mma) {
#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  auto dA = make_stride(get<2>(shape_MNK), Int<1>{});
  auto dB = make_stride(get<2>(shape_MNK), Int<1>{});
  auto dC = make_stride(get<1>(shape_MNK), Int<1>{});

  Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA);
  Tensor mB = make_tensor(make_gmem_ptr(B_T), select<1, 2>(shape_MNK), dB);
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC);

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

  extern __shared__ char shared_memory[];
  using SmemStorage =
      SharedStorage<tfloat32_t, tfloat32_t, ASmemLayout, BSmemLayout>;
  SmemStorage& smem = *reinterpret_cast<SmemStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), ASmemLayout{});
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), BSmemLayout{});

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor sA_pi = as_position_independent_swizzle_tensor(sA);
  Tensor tAsA = thr_copy_a.partition_D(sA_pi);

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);
  Tensor sB_pi = as_position_independent_swizzle_tensor(sB);
  Tensor tBsB = thr_copy_b.partition_D(sB_pi);

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCgC = thr_mma.partition_C(gC);

  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);

  auto K_TILE_MAX = size<3>(tAgA);
  auto K_PIPE_MAX = size<3>(tAsA);

  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX - 1; ++pipe) {
    copy(copy_a, tAgA(_, _, _, pipe), tAsA(_, _, _, pipe));
    copy(copy_b, tBgB(_, _, _, pipe), tBsB(_, _, _, pipe));
    cp_async_fence();
  }

  clear(tCrC);
  __syncthreads();

  int k_pipe_read = 0;
  int k_pipe_write = K_PIPE_MAX - 1;

  CUTE_NO_UNROLL
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    int k_tile_next = k_tile + (K_PIPE_MAX - 1);
    k_tile_next = (k_tile_next >= K_TILE_MAX) ? K_TILE_MAX - 1 : k_tile_next;

    copy(copy_a, tAgA(_, _, _, k_tile_next), tAsA(_, _, _, k_pipe_write));
    copy(copy_b, tBgB(_, _, _, k_tile_next), tBsB(_, _, _, k_pipe_write));
    cp_async_fence();

    ++k_pipe_write;
    k_pipe_write = (k_pipe_write == K_PIPE_MAX) ? 0 : k_pipe_write;

    cp_async_wait<0>();

    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    gemm(mma, tCrA(_, _, _, k_pipe_read), tCrB(_, _, _, k_pipe_read), tCrC);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);

    ++k_pipe_read;
    k_pipe_read = (k_pipe_read == K_PIPE_MAX) ? 0 : k_pipe_read;
  }

  axpby(1.0f, tCrC, 0.0f, tCgC);
#endif
}

int ceil_div(int x, int y) {
  return (x + y - 1) / y;
}

template <typename Fn>
float time_cuda_ms(Fn&& fn, int warmup, int repeat) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeat; ++i) {
    fn();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / repeat;
}

void launch_wgmma_tf32(const tfloat32_t* A,
                       const tfloat32_t* B_T,
                       float* C,
                       Problem p) {
  auto shape_MNK = make_shape(p.M, p.N, p.K);

  auto bM = Int<64>{};
  auto bN = Int<64>{};
  auto bK = Int<64>{};
  auto bP = Int<2>{};
  auto cta_tiler = make_shape(bM, bN, bK);

  auto sA = tile_to_shape(GMMA::Layout_K_SW128_Atom<tfloat32_t>{},
                          make_shape(bM, bK, bP));
  auto sB = tile_to_shape(GMMA::Layout_K_SW128_Atom<tfloat32_t>{},
                          make_shape(bN, bK, bP));

  TiledCopy copyA = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, tfloat32_t>{},
      Layout<Shape<_16, _8>, Stride<_8, _1>>{},
      Layout<Shape<_1, _8>>{});
  TiledCopy copyB = make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, tfloat32_t>{},
      Layout<Shape<_16, _8>, Stride<_8, _1>>{},
      Layout<Shape<_1, _8>>{});

  TiledMMA tiled_mma =
      make_tiled_mma(SM90_64x64x8_F32TF32TF32_SS_TN<>{});

  dim3 block(size(tiled_mma));
  dim3 grid(ceil_div(p.M, 64), ceil_div(p.N, 64));
  int smem_bytes =
      int(sizeof(SharedStorage<tfloat32_t, tfloat32_t, decltype(sA), decltype(sB)>));

  auto kernel = &wgmma_tf32_kernel<decltype(shape_MNK),
                                   decltype(cta_tiler),
                                   decltype(sA),
                                   decltype(copyA),
                                   decltype(sB),
                                   decltype(copyB),
                                   decltype(tiled_mma)>;
  CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
  kernel<<<grid, block, smem_bytes>>>(
      shape_MNK, cta_tiler, A, B_T, C, sA, copyA, sB, copyB, tiled_mma);
  CUDA_CHECK(cudaGetLastError());
}

void run_cublas_tf32(cublasHandle_t handle,
                     const float* A,
                     const float* B,
                     float* C,
                     Problem p) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  // Row-major C = A * B is equivalent to column-major C^T = B^T * A^T.
  CUBLAS_CHECK(cublasGemmEx(handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            p.N,
                            p.M,
                            p.K,
                            &alpha,
                            B,
                            CUDA_R_32F,
                            p.N,
                            A,
                            CUDA_R_32F,
                            p.K,
                            &beta,
                            C,
                            CUDA_R_32F,
                            p.N,
                            CUBLAS_COMPUTE_32F_FAST_TF32,
                            CUBLAS_GEMM_DEFAULT));
}

double flops_for(Problem p) {
  return 2.0 * static_cast<double>(p.M) * p.N * p.K;
}

double tflops_for(Problem p, double ms) {
  return flops_for(p) / (ms * 1.0e-3) / 1.0e12;
}

double arithmetic_intensity_for(Problem p) {
  const double bytes =
      sizeof(float) * (static_cast<double>(p.M) * p.K +
                       static_cast<double>(p.K) * p.N +
                       static_cast<double>(p.M) * p.N);
  return flops_for(p) / bytes;
}

CheckResult compare_results(const std::vector<float>& got,
                            const std::vector<float>& ref,
                            double atol,
                            double rtol) {
  CheckResult result;
  result.pass = true;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double a = static_cast<double>(got[i]);
    const double b = static_cast<double>(ref[i]);
    const double abs_err = std::abs(a - b);
    const double rel_err = abs_err / std::max(std::abs(b), 1.0e-12);
    result.max_abs = std::max(result.max_abs, abs_err);
    result.max_rel = std::max(result.max_rel, rel_err);
    if (abs_err > atol && rel_err > rtol) {
      result.pass = false;
      ++result.bad_count;
    }
  }
  return result;
}

std::size_t checked_count(int a, int b, const char* name) {
  const auto aa = static_cast<std::uint64_t>(a);
  const auto bb = static_cast<std::uint64_t>(b);
  if (aa != 0 && bb > std::numeric_limits<std::size_t>::max() / aa) {
    std::ostringstream oss;
    oss << name << " is too large";
    throw std::runtime_error(oss.str());
  }
  return static_cast<std::size_t>(aa * bb);
}

void validate_shape(Problem p) {
  if ((p.M % 64) || (p.N % 64) || (p.K % 64)) {
    std::ostringstream oss;
    oss << "WGMMA TF32 fast path requires M/N/K multiples of 64. Got "
        << p.M << "x" << p.N << "x" << p.K;
    throw std::runtime_error(oss.str());
  }
}

Problem parse_shape(const std::string& text) {
  int m = 0;
  int n = 0;
  int k = 0;
  char x1 = 0;
  char x2 = 0;
  std::istringstream iss(text);
  if (!(iss >> m >> x1 >> n >> x2 >> k) || x1 != 'x' || x2 != 'x' ||
      m <= 0 || n <= 0 || k <= 0) {
    throw std::runtime_error("invalid shape, expected MxNxK");
  }
  return {m, n, k};
}

std::vector<Problem> parse_sizes(const std::string& text) {
  std::vector<Problem> out;
  std::size_t start = 0;
  while (start < text.size()) {
    std::size_t end = text.find(',', start);
    std::string token =
        text.substr(start, end == std::string::npos ? std::string::npos
                                                    : end - start);
    int size = std::stoi(token);
    if (size <= 0) {
      throw std::runtime_error("size must be positive");
    }
    out.push_back({size, size, size});
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return out;
}

int parse_int_arg(const char* value, const char* name) {
  char* end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (*value == '\0' || *end != '\0' || parsed < 0 ||
      parsed > std::numeric_limits<int>::max()) {
    std::ostringstream oss;
    oss << "invalid " << name << ": " << value;
    throw std::runtime_error(oss.str());
  }
  return static_cast<int>(parsed);
}

Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need_value = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::ostringstream oss;
        oss << name << " requires a value";
        throw std::runtime_error(oss.str());
      }
      return argv[++i];
    };

    if (arg == "--device") {
      cfg.device = parse_int_arg(need_value("--device"), "--device");
    } else if (arg == "--shape") {
      cfg.problems.push_back(parse_shape(need_value("--shape")));
    } else if (arg == "--sizes") {
      auto parsed = parse_sizes(need_value("--sizes"));
      cfg.problems.insert(cfg.problems.end(), parsed.begin(), parsed.end());
    } else if (arg == "--warmup") {
      cfg.warmup = parse_int_arg(need_value("--warmup"), "--warmup");
    } else if (arg == "--repeat") {
      cfg.repeat = parse_int_arg(need_value("--repeat"), "--repeat");
    } else if (arg == "--no-check") {
      cfg.check = false;
    } else if (arg == "--no-cublas") {
      cfg.run_cublas = false;
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: " << argv[0] << " [options]\n"
          << "  --device ID        CUDA device, default 0\n"
          << "  --shape MxNxK      Add one problem\n"
          << "  --sizes LIST       Square sizes, e.g. 512,1024,2048\n"
          << "  --warmup N         Warmup iterations, default 5\n"
          << "  --repeat N         Timed iterations, default 20\n"
          << "  --no-check         Skip correctness check\n"
          << "  --no-cublas        Skip timed cuBLAS baseline\n";
      std::exit(0);
    } else {
      std::ostringstream oss;
      oss << "unknown option: " << arg;
      throw std::runtime_error(oss.str());
    }
  }

  if (cfg.problems.empty()) {
    cfg.problems = {{512, 512, 512},
                    {1024, 1024, 1024},
                    {2048, 2048, 2048},
                    {4096, 4096, 4096}};
  }
  return cfg;
}

void print_hardware_info(const cudaDeviceProp& prop, int device) {
  std::cout << "Hardware:\n";
  std::cout << "  device: " << device << "\n";
  std::cout << "  name: " << prop.name << "\n";
  std::cout << "  compute capability: " << prop.major << "." << prop.minor
            << "\n";
  std::cout << "  SM count: " << prop.multiProcessorCount << "\n";
  std::cout << "  warp size: " << prop.warpSize << "\n";
  std::cout << "  shared memory per block: "
            << prop.sharedMemPerBlock / 1024.0 << " KiB\n";
  std::cout << "  shared memory per block opt-in: "
            << prop.sharedMemPerBlockOptin / 1024.0 << " KiB\n";
  std::cout << "  shared memory per SM: "
            << prop.sharedMemPerMultiprocessor / 1024.0 << " KiB\n";
  std::cout << "  registers per SM: " << prop.regsPerMultiprocessor << "\n";
  std::cout << "  theoretical TF32 Tensor Core dense peak used: "
            << std::fixed << std::setprecision(3)
            << kH100Tf32TensorPeakTflops << " TFLOPS\n";
  std::cout << "  HBM bandwidth used for roofline: " << kH100HbmTbps
            << " TB/s\n";
  std::cout << "  WGMMA tile: 64x64x8, CTA tile: 64x64x64, stages: 2\n";
}

void run_one(cublasHandle_t handle, const Config& cfg, Problem p) {
  validate_shape(p);

  const std::size_t a_count = checked_count(p.M, p.K, "A");
  const std::size_t b_count = checked_count(p.K, p.N, "B");
  const std::size_t bt_count = checked_count(p.N, p.K, "B_T");
  const std::size_t c_count = checked_count(p.M, p.N, "C");

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> h_A(a_count);
  std::vector<float> h_B(b_count);
  std::vector<tfloat32_t> h_A_tf32(a_count);
  std::vector<tfloat32_t> h_BT_tf32(bt_count);
  std::vector<float> h_C(c_count, 0.0f);
  std::vector<float> h_ref(c_count, 0.0f);
  for (float& x : h_A) {
    x = dist(rng);
  }
  for (float& x : h_B) {
    x = dist(rng);
  }
  for (std::size_t i = 0; i < a_count; ++i) {
    h_A_tf32[i] = tfloat32_t(h_A[i]);
  }
  for (int kk = 0; kk < p.K; ++kk) {
    for (int nn = 0; nn < p.N; ++nn) {
      h_BT_tf32[static_cast<std::size_t>(nn) * p.K + kk] =
          tfloat32_t(h_B[static_cast<std::size_t>(kk) * p.N + nn]);
    }
  }

  float* d_A = nullptr;
  float* d_B = nullptr;
  tfloat32_t* d_A_tf32 = nullptr;
  tfloat32_t* d_BT_tf32 = nullptr;
  float* d_C = nullptr;
  float* d_ref = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, a_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, b_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_A_tf32, a_count * sizeof(tfloat32_t)));
  CUDA_CHECK(cudaMalloc(&d_BT_tf32, bt_count * sizeof(tfloat32_t)));
  CUDA_CHECK(cudaMalloc(&d_C, c_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ref, c_count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_count * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_count * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_A_tf32, h_A_tf32.data(),
                        a_count * sizeof(tfloat32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_BT_tf32, h_BT_tf32.data(),
                        bt_count * sizeof(tfloat32_t),
                        cudaMemcpyHostToDevice));

  auto custom_launch = [&]() {
    launch_wgmma_tf32(d_A_tf32, d_BT_tf32, d_C, p);
  };
  auto cublas_launch = [&]() { run_cublas_tf32(handle, d_A, d_B, d_ref, p); };

  CUDA_CHECK(cudaMemset(d_C, 0, c_count * sizeof(float)));
  const float custom_ms = time_cuda_ms(custom_launch, cfg.warmup, cfg.repeat);
  CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, c_count * sizeof(float),
                        cudaMemcpyDeviceToHost));

  float cublas_ms = 0.0f;
  if (cfg.check || cfg.run_cublas) {
    CUDA_CHECK(cudaMemset(d_ref, 0, c_count * sizeof(float)));
    cublas_ms = time_cuda_ms(cublas_launch, cfg.warmup, cfg.repeat);
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref, c_count * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }

  CheckResult check;
  if (cfg.check) {
    check = compare_results(h_C, h_ref, 0.5, 0.05);
  }

  const double custom_tflops = tflops_for(p, custom_ms);
  const double ai = arithmetic_intensity_for(p);
  const double mem_roof = ai * kH100HbmTbps;
  const double roof = std::min(kH100Tf32TensorPeakTflops, mem_roof);

  std::cout << "\nProblem: " << p.M << "x" << p.N << "x" << p.K << "\n";
  std::cout << "  dtype: tf32\n";
  std::cout << "  custom WGMMA: " << std::fixed << std::setprecision(6)
            << custom_ms << " ms, " << std::setprecision(4) << custom_tflops
            << " TFLOPS, "
            << custom_tflops / kH100Tf32TensorPeakTflops * 100.0
            << "% peak\n";
  std::cout << "  roofline: AI=" << ai << " FLOP/B, mem_roof=" << mem_roof
            << " TFLOPS, bound="
            << (mem_roof < kH100Tf32TensorPeakTflops ? "memory" : "compute")
            << ", custom_roof=" << custom_tflops / roof * 100.0 << "%\n";
  if (cfg.run_cublas) {
    const double cublas_tflops = tflops_for(p, cublas_ms);
    std::cout << "  cuBLAS FAST_TF32: " << std::fixed << std::setprecision(6)
              << cublas_ms << " ms, " << std::setprecision(4) << cublas_tflops
              << " TFLOPS, "
              << cublas_tflops / kH100Tf32TensorPeakTflops * 100.0
              << "% peak\n";
  }
  if (cfg.check) {
    std::cout << "  check: " << (check.pass ? "PASS" : "FAIL")
              << ", max_abs=" << check.max_abs
              << ", max_rel=" << check.max_rel
              << ", bad_count=" << check.bad_count << "\n";
    if (!check.pass) {
      throw std::runtime_error("correctness check failed");
    }
  }

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_A_tf32));
  CUDA_CHECK(cudaFree(d_BT_tf32));
  CUDA_CHECK(cudaFree(d_C));
  CUDA_CHECK(cudaFree(d_ref));
}

}  // namespace wgmma_bench

int main(int argc, char** argv) {
  try {
    wgmma_bench::Config cfg = wgmma_bench::parse_args(argc, argv);
    CUDA_CHECK(cudaSetDevice(cfg.device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));
    wgmma_bench::print_hardware_info(prop, cfg.device);

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    for (wgmma_bench::Problem p : cfg.problems) {
      wgmma_bench::run_one(handle, cfg, p);
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
