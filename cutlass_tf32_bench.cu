// Simple CUTLASS SM90 TF32 GEMM benchmark for H100.
//
// Contract:
//   Row-major C[M, N] = A[M, K] * B[K, N]
//   A/B/C storage is float.
//   CUTLASS CollectiveBuilder maps float inputs to TF32 Tensor Core compute on
//   Hopper when using OpClassTensorOp.

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler.hpp>
#include <cutlass/util/packed_stride.hpp>

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

#define CUTLASS_CHECK(expr)                                                  \
  do {                                                                       \
    cutlass::Status status = (expr);                                         \
    if (status != cutlass::Status::kSuccess) {                               \
      std::ostringstream oss;                                                \
      oss << "CUTLASS error at " << __FILE__ << ":" << __LINE__ << ": "     \
          << cutlassGetStatusString(status);                                 \
      throw std::runtime_error(oss.str());                                   \
    }                                                                        \
  } while (0)

namespace cutlass_tf32_bench {

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

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using ElementA = float;
using ElementB = float;
using ElementC = float;
using ElementD = float;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

constexpr int AlignmentA = 16 / sizeof(ElementA);
constexpr int AlignmentB = 16 / sizeof(ElementB);
constexpr int AlignmentC = 16 / sizeof(ElementC);
constexpr int AlignmentD = 16 / sizeof(ElementD);

using TileShape = Shape<_128, _128, _64>;
using ClusterShape = Shape<_2, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    LayoutC,
    AlignmentC,
    ElementD,
    LayoutD,
    AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementA,
    LayoutA,
    AlignmentA,
    ElementB,
    LayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;

using CutlassGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename CutlassGemm::GemmKernel::StrideA;
using StrideB = typename CutlassGemm::GemmKernel::StrideB;
using StrideC = typename CutlassGemm::GemmKernel::StrideC;
using StrideD = typename CutlassGemm::GemmKernel::StrideD;

#endif

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
  if (p.M <= 0 || p.N <= 0 || p.K <= 0) {
    throw std::runtime_error("M/N/K must be positive");
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
  std::cout << "  CUTLASS kernel: SM90 CollectiveBuilder, tile 128x128x64,"
            << " cluster 2x1x1, auto stages/schedule\n";
  std::cout << "  CUTLASS B operand is materialized as column-major B_col for"
            << " the benchmark; public contract remains row-major C=A*B\n";
}

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

struct CutlassRunner {
  CutlassGemm gemm;
  typename CutlassGemm::Arguments arguments;
  std::uint8_t* workspace = nullptr;

  CutlassRunner(float* A, float* B, float* C, Problem p, int device) {
    StrideA stride_A =
        cutlass::make_cute_packed_stride(StrideA{}, make_shape(p.M, p.K, 1));
    StrideB stride_B =
        cutlass::make_cute_packed_stride(StrideB{}, make_shape(p.N, p.K, 1));
    StrideC stride_C =
        cutlass::make_cute_packed_stride(StrideC{}, make_shape(p.M, p.N, 1));
    StrideD stride_D =
        cutlass::make_cute_packed_stride(StrideD{}, make_shape(p.M, p.N, 1));

    auto hw_info =
        cutlass::KernelHardwareInfo::make_kernel_hardware_info<GemmKernel>(
            device);

    arguments = typename CutlassGemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {p.M, p.N, p.K},
        {A, stride_A, B, stride_B},
        {{1.0f, 0.0f}, C, stride_C, C, stride_D},
        hw_info};

    const std::size_t workspace_size = CutlassGemm::get_workspace_size(arguments);
    if (workspace_size > 0) {
      CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }
    CUTLASS_CHECK(gemm.can_implement(arguments));
    CUTLASS_CHECK(gemm.initialize(arguments, workspace));
  }

  CutlassRunner(const CutlassRunner&) = delete;
  CutlassRunner& operator=(const CutlassRunner&) = delete;

  ~CutlassRunner() {
    if (workspace != nullptr) {
      cudaFree(workspace);
    }
  }

  void run() {
    CUTLASS_CHECK(gemm.run());
  }
};

#endif

void run_one(cublasHandle_t handle,
             const Config& cfg,
             const cudaDeviceProp& prop,
             Problem p) {
  validate_shape(p);

#if !defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  throw std::runtime_error("CUTLASS_ARCH_MMA_SM90_SUPPORTED is not defined");
#else
  const std::size_t a_count = checked_count(p.M, p.K, "A");
  const std::size_t b_count = checked_count(p.K, p.N, "B");
  const std::size_t c_count = checked_count(p.M, p.N, "C");

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> h_A(a_count);
  std::vector<float> h_B(b_count);
  std::vector<float> h_B_col(b_count);
  std::vector<float> h_C(c_count, 0.0f);
  std::vector<float> h_ref(c_count, 0.0f);
  for (float& x : h_A) {
    x = dist(rng);
  }
  for (float& x : h_B) {
    x = dist(rng);
  }
  for (int kk = 0; kk < p.K; ++kk) {
    for (int nn = 0; nn < p.N; ++nn) {
      h_B_col[static_cast<std::size_t>(nn) * p.K + kk] =
          h_B[static_cast<std::size_t>(kk) * p.N + nn];
    }
  }

  float* d_A = nullptr;
  float* d_B = nullptr;
  float* d_B_col = nullptr;
  float* d_C = nullptr;
  float* d_ref = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A, a_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, b_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B_col, b_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, c_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ref, c_count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_count * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_count * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B_col, h_B_col.data(), b_count * sizeof(float),
                        cudaMemcpyHostToDevice));

  CutlassRunner cutlass_runner(d_A, d_B_col, d_C, p, cfg.device);
  auto custom_launch = [&]() { cutlass_runner.run(); };
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
  std::cout << "  custom CUTLASS: " << std::fixed << std::setprecision(6)
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
  CUDA_CHECK(cudaFree(d_B_col));
  CUDA_CHECK(cudaFree(d_C));
  CUDA_CHECK(cudaFree(d_ref));
#endif
}

}  // namespace cutlass_tf32_bench

int main(int argc, char** argv) {
  try {
    cutlass_tf32_bench::Config cfg =
        cutlass_tf32_bench::parse_args(argc, argv);
    CUDA_CHECK(cudaSetDevice(cfg.device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));
    cutlass_tf32_bench::print_hardware_info(prop, cfg.device);

    if (prop.major != 9) {
      throw std::runtime_error("this CUTLASS SM90 benchmark requires H100/SM90");
    }

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    for (cutlass_tf32_bench::Problem p : cfg.problems) {
      cutlass_tf32_bench::run_one(handle, cfg, prop, p);
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
