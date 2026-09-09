// Standalone Tensor Core GEMM benchmark skeleton for H100.
//
// Contract:
//   Row-major C[M, N] = A[M, K] * B[K, N]
//   A/B dtype is selected by --dtype: tf32, fp16, or bf16.
//   C is always float.
//
// The benchmark code handles:
//   - hardware information
//   - random input generation
//   - cuBLAS Tensor Core reference
//   - kernel timing
//   - TFLOPS and peak percentage
//   - ideal roofline fields
//   - correctness checking
//
// Implement only the kernels/launch mapping in the "User kernel interface"
// section unless you need to change the benchmark contract.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>

#include <algorithm>
#include <cctype>
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
#include <type_traits>
#include <vector>

#ifndef USER_TENSOR_GEMM_FP16_READY
#define USER_TENSOR_GEMM_FP16_READY 0
#endif

#ifndef USER_TENSOR_GEMM_BF16_READY
#define USER_TENSOR_GEMM_BF16_READY 0
#endif

#ifndef USER_TENSOR_GEMM_TF32_READY
#define USER_TENSOR_GEMM_TF32_READY 1
#endif

#ifndef USER_TENSOR_GEMM_TILE_M
#define USER_TENSOR_GEMM_TILE_M 16
#endif

#ifndef USER_TENSOR_GEMM_TILE_N
#define USER_TENSOR_GEMM_TILE_N 16
#endif

#ifndef USER_TENSOR_GEMM_TILE_K
#define USER_TENSOR_GEMM_TILE_K 32
#endif

#ifndef USER_TENSOR_GEMM_WARPS_PER_BLOCK
#define USER_TENSOR_GEMM_WARPS_PER_BLOCK 4
#endif

#ifndef USER_TENSOR_GEMM_TF32_WARPS_M
#define USER_TENSOR_GEMM_TF32_WARPS_M 4
#endif

#ifndef USER_TENSOR_GEMM_TF32_WARPS_N
#define USER_TENSOR_GEMM_TF32_WARPS_N 4
#endif

// ========================= User kernel interface =========================
//
// The current TF32 implementation uses a shared-memory block tile:
//   blockDim.x = USER_TENSOR_GEMM_TF32_WARPS_M *
//                USER_TENSOR_GEMM_TF32_WARPS_N * 32
//   grid.x = ceil(N / (16 * USER_TENSOR_GEMM_TF32_WARPS_N))
//   grid.y = ceil(M / (16 * USER_TENSOR_GEMM_TF32_WARPS_M))
//
// Within each block, one warp computes one 16x16 C tile. The block
// cooperatively stages A and B tiles into shared memory before WMMA loads.
//
// If you implement a different tiling, change the launch_custom_* wrapper
// below together with your kernel.

__device__ __forceinline__ void zero_fill_stub(float* C, int M, int N) {
  const std::int64_t block_linear =
      static_cast<std::int64_t>(blockIdx.x) +
      static_cast<std::int64_t>(gridDim.x) * blockIdx.y;
  const std::int64_t thread_linear =
      block_linear * blockDim.x + threadIdx.x;
  const std::int64_t stride =
      static_cast<std::int64_t>(gridDim.x) * gridDim.y * blockDim.x;
  const std::int64_t total =
      static_cast<std::int64_t>(M) * static_cast<std::int64_t>(N);

  for (std::int64_t i = thread_linear; i < total; i += stride) {
    C[i] = 0.0f;
  }
}

__global__ void tensor_gemm_kernel_fp16(const half* A,
                                        const half* B,
                                        float* C,
                                        int M,
                                        int N,
                                        int K) {
#if USER_TENSOR_GEMM_FP16_READY
  // TODO: implement FP16 Tensor Core GEMM here.
  // Suggested first target:
  //   using namespace nvcuda;
  //   wmma::fragment<matrix_a, 16, 16, 16, half, row_major>
  //   wmma::fragment<matrix_b, 16, 16, 16, half, row_major>
  //   wmma::fragment<accumulator, 16, 16, 16, float>
  //
  // The benchmark expects C to be fully overwritten.
  (void)A;
  (void)B;
  (void)C;
  (void)M;
  (void)N;
  (void)K;
#else
  (void)A;
  (void)B;
  (void)K;
  zero_fill_stub(C, M, N);
#endif
}

__global__ void tensor_gemm_kernel_bf16(const __nv_bfloat16* A,
                                        const __nv_bfloat16* B,
                                        float* C,
                                        int M,
                                        int N,
                                        int K) {
#if USER_TENSOR_GEMM_BF16_READY
  // TODO: implement BF16 Tensor Core GEMM here.
  (void)A;
  (void)B;
  (void)C;
  (void)M;
  (void)N;
  (void)K;
#else
  (void)A;
  (void)B;
  (void)K;
  zero_fill_stub(C, M, N);
#endif
}

__global__ void tensor_gemm_kernel_tf32(const float* A,
                                        const float* B,
                                        float* C,
                                        int M,
                                        int N,
                                        int K) {
#if USER_TENSOR_GEMM_TF32_READY
  using namespace nvcuda;

  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 8;
  constexpr int WARPS_M = USER_TENSOR_GEMM_TF32_WARPS_M;
  constexpr int WARPS_N = USER_TENSOR_GEMM_TF32_WARPS_N;
  constexpr int BLOCK_M = WMMA_M * WARPS_M;
  constexpr int BLOCK_K = USER_TENSOR_GEMM_TILE_K;
  constexpr int BLOCK_N = WMMA_N * WARPS_N;
  constexpr int VEC_FLOATS = 4;

  static_assert(BLOCK_K % WMMA_K == 0,
                "USER_TENSOR_GEMM_TILE_K must be a multiple of WMMA_K");
  static_assert(BLOCK_K % VEC_FLOATS == 0,
                "USER_TENSOR_GEMM_TILE_K must support float4 copies");
  static_assert(BLOCK_N % VEC_FLOATS == 0,
                "BLOCK_N must support float4 copies");
  static_assert(WARPS_M > 0 && WARPS_N > 0,
                "TF32 warp tile dimensions must be positive");
  static_assert(WARPS_M * WARPS_N * 32 <= 1024,
                "TF32 block must not exceed the CUDA thread-block limit");

  const int warp_id = threadIdx.x >> 5;
  const int warp_m = warp_id / WARPS_N;
  const int warp_n = warp_id % WARPS_N;
  const int block_row = blockIdx.y * BLOCK_M;
  const int row_id = block_row + warp_m * WMMA_M;
  const int block_col = blockIdx.x * BLOCK_N;
  const int col_id = block_col + warp_n * WMMA_N;

  __shared__ __align__(16) float s_a[BLOCK_M][BLOCK_K];
  __shared__ __align__(16) float s_b[BLOCK_K][BLOCK_N];

  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b_frag;

  wmma::fill_fragment(c_frag, 0.0f);

  for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
    constexpr int A_VEC_COUNT = BLOCK_M * BLOCK_K / VEC_FLOATS;
    constexpr int B_VEC_COUNT = BLOCK_K * BLOCK_N / VEC_FLOATS;

    for (int vec = threadIdx.x; vec < A_VEC_COUNT; vec += blockDim.x) {
      const int row = vec / (BLOCK_K / VEC_FLOATS);
      const int col = (vec % (BLOCK_K / VEC_FLOATS)) * VEC_FLOATS;
      const int global_row = block_row + row;
      const int global_col = k0 + col;
      *reinterpret_cast<float4*>(&s_a[row][col]) =
          *reinterpret_cast<const float4*>(
              A + static_cast<std::int64_t>(global_row) * K + global_col);
    }

    for (int vec = threadIdx.x; vec < B_VEC_COUNT; vec += blockDim.x) {
      const int row = vec / (BLOCK_N / VEC_FLOATS);
      const int col = (vec % (BLOCK_N / VEC_FLOATS)) * VEC_FLOATS;
      const int global_row = k0 + row;
      const int global_col = block_col + col;
      *reinterpret_cast<float4*>(&s_b[row][col]) =
          *reinterpret_cast<const float4*>(
              B + static_cast<std::int64_t>(global_row) * N + global_col);
    }

    __syncthreads();

    for (int kk = 0; kk < BLOCK_K; kk += WMMA_K) {
      wmma::load_matrix_sync(a_frag, &s_a[warp_m * WMMA_M][kk], BLOCK_K);
      wmma::load_matrix_sync(b_frag, &s_b[kk][warp_n * WMMA_N], BLOCK_N);
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    __syncthreads();
  }

  wmma::store_matrix_sync(C + static_cast<std::int64_t>(row_id) * N + col_id,
                          c_frag,
                          N,
                          wmma::mem_row_major);
#else
  (void)A;
  (void)B;
  (void)K;
  zero_fill_stub(C, M, N);
#endif
}

struct Problem {
  int M;
  int N;
  int K;
};

static int ceil_div(int x, int y) {
  return (x + y - 1) / y;
}

static void launch_custom_fp16(const half* d_A,
                               const half* d_B,
                               float* d_C,
                               const Problem& p) {
  dim3 block(USER_TENSOR_GEMM_WARPS_PER_BLOCK * 32);
  dim3 grid(ceil_div(p.N, USER_TENSOR_GEMM_TILE_N *
                              USER_TENSOR_GEMM_WARPS_PER_BLOCK),
            ceil_div(p.M, USER_TENSOR_GEMM_TILE_M));
  tensor_gemm_kernel_fp16<<<grid, block>>>(d_A, d_B, d_C, p.M, p.N, p.K);
}

static void launch_custom_bf16(const __nv_bfloat16* d_A,
                               const __nv_bfloat16* d_B,
                               float* d_C,
                               const Problem& p) {
  dim3 block(USER_TENSOR_GEMM_WARPS_PER_BLOCK * 32);
  dim3 grid(ceil_div(p.N, USER_TENSOR_GEMM_TILE_N *
                              USER_TENSOR_GEMM_WARPS_PER_BLOCK),
            ceil_div(p.M, USER_TENSOR_GEMM_TILE_M));
  tensor_gemm_kernel_bf16<<<grid, block>>>(d_A, d_B, d_C, p.M, p.N, p.K);
}

static void launch_custom_tf32(const float* d_A,
                               const float* d_B,
                               float* d_C,
                               const Problem& p) {
  constexpr int wmma_m = 16;
  constexpr int wmma_n = 16;
  constexpr int warps_m = USER_TENSOR_GEMM_TF32_WARPS_M;
  constexpr int warps_n = USER_TENSOR_GEMM_TF32_WARPS_N;
  constexpr int block_m = wmma_m * warps_m;
  constexpr int block_n = wmma_n * warps_n;
  constexpr int warps_per_block = warps_m * warps_n;

  dim3 block(warps_per_block * 32);
  dim3 grid(ceil_div(p.N, block_n), ceil_div(p.M, block_m));
  tensor_gemm_kernel_tf32<<<grid, block>>>(d_A, d_B, d_C, p.M, p.N, p.K);
}

// ===================== Benchmark code below this line =====================

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err__ = (call);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      std::ostringstream oss__;                                                 \
      oss__ << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "       \
            << cudaGetErrorString(err__) << " (" << static_cast<int>(err__)    \
            << ")";                                                            \
      throw std::runtime_error(oss__.str());                                    \
    }                                                                           \
  } while (0)

#define CUBLAS_CHECK(call)                                                      \
  do {                                                                          \
    cublasStatus_t st__ = (call);                                               \
    if (st__ != CUBLAS_STATUS_SUCCESS) {                                        \
      std::ostringstream oss__;                                                 \
      oss__ << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << ": "     \
            << static_cast<int>(st__);                                          \
      throw std::runtime_error(oss__.str());                                    \
    }                                                                           \
  } while (0)

enum class BenchDtype {
  kFp16,
  kBf16,
  kTf32,
};

struct Config {
  int device = 0;
  int warmup = 5;
  int repeat = 20;
  BenchDtype dtype = BenchDtype::kTf32;
  bool check = true;
  bool csv = false;
  bool run_cublas = true;
  bool allow_stub = false;
  bool allow_nonmultiple = false;
  double peak_tflops = 0.0;
  double hbm_tbps = 3.35;
  double atol = -1.0;
  double rtol = -1.0;
  std::vector<Problem> problems;
};

struct CheckResult {
  double max_abs = 0.0;
  double max_rel = 0.0;
  std::size_t bad_count = 0;
  std::size_t worst_index = 0;
  float worst_got = 0.0f;
  float worst_ref = 0.0f;
  bool passed = true;
};

struct RowResult {
  double custom_ms = 0.0;
  double custom_tflops = 0.0;
  double cublas_ms = std::numeric_limits<double>::quiet_NaN();
  double cublas_tflops = std::numeric_limits<double>::quiet_NaN();
  CheckResult check;
};

template <typename T>
struct DeviceBuffer {
  T* ptr = nullptr;
  std::size_t count = 0;

  explicit DeviceBuffer(std::size_t n) : count(n) {
    if (n > 0) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)));
    }
  }

  ~DeviceBuffer() {
    if (ptr) {
      cudaFree(ptr);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

static std::string lower_copy(std::string s) {
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

static int parse_int_value(const char* s, const char* name) {
  char* end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (!s[0] || (end && *end) || v <= 0 ||
      v > std::numeric_limits<int>::max()) {
    std::ostringstream oss;
    oss << "Invalid positive int for " << name << ": " << s;
    throw std::runtime_error(oss.str());
  }
  return static_cast<int>(v);
}

static int parse_nonnegative_int_value(const char* s, const char* name) {
  char* end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (!s[0] || (end && *end) || v < 0 ||
      v > std::numeric_limits<int>::max()) {
    std::ostringstream oss;
    oss << "Invalid nonnegative int for " << name << ": " << s;
    throw std::runtime_error(oss.str());
  }
  return static_cast<int>(v);
}

static double parse_positive_double(const char* s, const char* name) {
  char* end = nullptr;
  double v = std::strtod(s, &end);
  if (!s[0] || (end && *end) || !std::isfinite(v) || v <= 0.0) {
    std::ostringstream oss;
    oss << "Invalid positive double for " << name << ": " << s;
    throw std::runtime_error(oss.str());
  }
  return v;
}

static Problem parse_shape(const std::string& raw) {
  std::string s = lower_copy(raw);
  std::replace(s.begin(), s.end(), ',', 'x');
  const std::size_t p1 = s.find('x');
  const std::size_t p2 =
      (p1 == std::string::npos) ? std::string::npos : s.find('x', p1 + 1);
  if (p1 == std::string::npos || p2 == std::string::npos ||
      s.find('x', p2 + 1) != std::string::npos) {
    throw std::runtime_error("Shape must be MxNxK, for example 4096x4096x4096");
  }
  Problem p;
  p.M = parse_int_value(s.substr(0, p1).c_str(), "M");
  p.N = parse_int_value(s.substr(p1 + 1, p2 - p1 - 1).c_str(), "N");
  p.K = parse_int_value(s.substr(p2 + 1).c_str(), "K");
  return p;
}

static std::vector<Problem> parse_square_list(const std::string& raw) {
  std::vector<Problem> out;
  std::string token;
  std::stringstream ss(raw);
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      continue;
    }
    int n = parse_int_value(token.c_str(), "--sizes");
    out.push_back({n, n, n});
  }
  if (out.empty()) {
    throw std::runtime_error("--sizes produced no valid sizes");
  }
  return out;
}

static BenchDtype parse_dtype(const std::string& raw) {
  const std::string s = lower_copy(raw);
  if (s == "fp16" || s == "half" || s == "hgemm") {
    return BenchDtype::kFp16;
  }
  if (s == "bf16" || s == "bfloat16") {
    return BenchDtype::kBf16;
  }
  if (s == "tf32") {
    return BenchDtype::kTf32;
  }
  throw std::runtime_error("Unknown --dtype: " + raw);
}

static const char* dtype_name(BenchDtype dtype) {
  switch (dtype) {
    case BenchDtype::kFp16:
      return "fp16";
    case BenchDtype::kBf16:
      return "bf16";
    case BenchDtype::kTf32:
      return "tf32";
  }
  return "unknown";
}

static bool custom_kernel_ready(BenchDtype dtype) {
  switch (dtype) {
    case BenchDtype::kFp16:
      return USER_TENSOR_GEMM_FP16_READY != 0;
    case BenchDtype::kBf16:
      return USER_TENSOR_GEMM_BF16_READY != 0;
    case BenchDtype::kTf32:
      return USER_TENSOR_GEMM_TF32_READY != 0;
  }
  return false;
}

static double default_atol(BenchDtype dtype) {
  switch (dtype) {
    case BenchDtype::kFp16:
      return 5.0e-1;
    case BenchDtype::kBf16:
      return 1.0;
    case BenchDtype::kTf32:
      return 5.0e-1;
  }
  return 1.0;
}

static double default_rtol(BenchDtype dtype) {
  switch (dtype) {
    case BenchDtype::kFp16:
      return 5.0e-2;
    case BenchDtype::kBf16:
      return 1.0e-1;
    case BenchDtype::kTf32:
      return 5.0e-2;
  }
  return 1.0e-1;
}

static void print_help(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n"
      << "\n"
      << "Default: benchmark TF32 Tensor Core GEMM sizes 512,1024,2048,4096.\n"
      << "Matrix layout: row-major C[M,N] = A[M,K] * B[K,N].\n"
      << "Default dtype tf32 keeps A/B/C storage as float like gemm.cu.\n"
      << "A/B dtype is selected by --dtype. C and accumulator reference are float.\n"
      << "\n"
      << "Options:\n"
      << "  --device ID              CUDA device id, default 0\n"
      << "  --dtype fp16|bf16|tf32   Input/compute path, default fp16\n"
      << "  --shape MxNxK            Add one problem shape, repeatable\n"
      << "  --sizes A,B,C            Square sizes, for example 1024,2048,4096\n"
      << "  --warmup N               Warmup launches, default 5\n"
      << "  --repeat N               Timed launches, default 20\n"
      << "  --peak TFLOPS            Override theoretical Tensor Core peak\n"
      << "  --hbm-tbps TBPS          HBM bandwidth for roofline, default 3.35\n"
      << "  --atol X                 Absolute tolerance, dtype-specific default\n"
      << "  --rtol X                 Relative tolerance, dtype-specific default\n"
      << "  --allow-stub             Permit timing the zero-fill stub kernel\n"
      << "  --allow-nonmultiple      Do not require fast-path tile multiples\n"
      << "  --no-check               Skip cuBLAS correctness check\n"
      << "  --no-cublas              Skip timed cuBLAS baseline\n"
      << "  --csv                    Print CSV rows to stdout; hardware info to stderr\n"
      << "  --help                   Show this message\n"
      << "\n"
      << "Examples:\n"
      << "  " << argv0 << " --dtype fp16 --shape 4096x4096x4096\n"
      << "  " << argv0 << " --dtype bf16 --sizes 1024,2048,4096 --repeat 50\n"
      << "  " << argv0 << " --dtype tf32 --peak 494.7 --csv\n"
      << "  " << argv0 << " --dtype fp16 --allow-stub --shape 512x512x512\n";
}

static Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_value = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::ostringstream oss;
        oss << name << " requires a value";
        throw std::runtime_error(oss.str());
      }
      return argv[++i];
    };

    if (arg == "--help" || arg == "-h") {
      print_help(argv[0]);
      std::exit(0);
    } else if (arg == "--device") {
      cfg.device = parse_nonnegative_int_value(need_value("--device"),
                                               "--device");
    } else if (arg == "--dtype") {
      cfg.dtype = parse_dtype(need_value("--dtype"));
    } else if (arg == "--shape") {
      cfg.problems.push_back(parse_shape(need_value("--shape")));
    } else if (arg == "--sizes") {
      std::vector<Problem> sizes = parse_square_list(need_value("--sizes"));
      cfg.problems.insert(cfg.problems.end(), sizes.begin(), sizes.end());
    } else if (arg == "--warmup") {
      cfg.warmup = parse_nonnegative_int_value(need_value("--warmup"),
                                               "--warmup");
    } else if (arg == "--repeat") {
      cfg.repeat = parse_int_value(need_value("--repeat"), "--repeat");
    } else if (arg == "--peak") {
      cfg.peak_tflops = parse_positive_double(need_value("--peak"), "--peak");
    } else if (arg == "--hbm-tbps") {
      cfg.hbm_tbps =
          parse_positive_double(need_value("--hbm-tbps"), "--hbm-tbps");
    } else if (arg == "--atol") {
      cfg.atol = parse_positive_double(need_value("--atol"), "--atol");
    } else if (arg == "--rtol") {
      cfg.rtol = parse_positive_double(need_value("--rtol"), "--rtol");
    } else if (arg == "--allow-stub") {
      cfg.allow_stub = true;
    } else if (arg == "--allow-nonmultiple") {
      cfg.allow_nonmultiple = true;
    } else if (arg == "--no-check") {
      cfg.check = false;
    } else if (arg == "--no-cublas") {
      cfg.run_cublas = false;
    } else if (arg == "--csv") {
      cfg.csv = true;
    } else {
      std::ostringstream oss;
      oss << "Unknown option: " << arg << " (use --help)";
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

static void validate_shape(const Problem& p, const Config& cfg) {
  if (cfg.allow_nonmultiple) {
    return;
  }
  const int mn_multiple = (cfg.dtype == BenchDtype::kTf32) ? 64 : 16;
  const int k_multiple = (cfg.dtype == BenchDtype::kTf32) ? 32 : 16;
  if ((p.M % mn_multiple) || (p.N % mn_multiple) || (p.K % k_multiple)) {
    std::ostringstream oss;
    oss << "Tensor Core fast path requires M/N multiples of " << mn_multiple
        << " and K multiple of " << k_multiple
        << ". Got " << p.M << "x" << p.N << "x" << p.K
        << ". Use --allow-nonmultiple only after adding tail handling.";
    throw std::runtime_error(oss.str());
  }
}

static std::size_t checked_count(int a, int b, const char* name) {
  const auto aa = static_cast<std::uint64_t>(a);
  const auto bb = static_cast<std::uint64_t>(b);
  if (aa != 0 &&
      bb > std::numeric_limits<std::size_t>::max() / aa) {
    std::ostringstream oss;
    oss << name << " is too large";
    throw std::runtime_error(oss.str());
  }
  return static_cast<std::size_t>(aa * bb);
}

static void fill_random(std::vector<float>& v, std::uint64_t seed) {
  std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  for (float& x : v) {
    x = dist(rng);
  }
}

template <typename T>
static T convert_input(float x);

template <>
half convert_input<half>(float x) {
  return __float2half(x);
}

template <>
__nv_bfloat16 convert_input<__nv_bfloat16>(float x) {
  return __float2bfloat16(x);
}

template <>
float convert_input<float>(float x) {
  return x;
}

template <typename T>
static std::vector<T> convert_vector(const std::vector<float>& src) {
  std::vector<T> out(src.size());
  for (std::size_t i = 0; i < src.size(); ++i) {
    out[i] = convert_input<T>(src[i]);
  }
  return out;
}

static cudaDataType_t cuda_dtype(BenchDtype dtype) {
  switch (dtype) {
    case BenchDtype::kFp16:
      return CUDA_R_16F;
    case BenchDtype::kBf16:
      return CUDA_R_16BF;
    case BenchDtype::kTf32:
      return CUDA_R_32F;
  }
  return CUDA_R_32F;
}

static cublasComputeType_t cublas_compute_type(BenchDtype dtype) {
  switch (dtype) {
    case BenchDtype::kFp16:
      return CUBLAS_COMPUTE_32F;
    case BenchDtype::kBf16:
      return CUBLAS_COMPUTE_32F;
    case BenchDtype::kTf32:
      return CUBLAS_COMPUTE_32F_FAST_TF32;
  }
  return CUBLAS_COMPUTE_32F;
}

template <typename T>
static void launch_custom(const T* d_A,
                          const T* d_B,
                          float* d_C,
                          const Problem& p);

template <>
void launch_custom<half>(const half* d_A,
                         const half* d_B,
                         float* d_C,
                         const Problem& p) {
  launch_custom_fp16(d_A, d_B, d_C, p);
  CUDA_CHECK(cudaGetLastError());
}

template <>
void launch_custom<__nv_bfloat16>(const __nv_bfloat16* d_A,
                                  const __nv_bfloat16* d_B,
                                  float* d_C,
                                  const Problem& p) {
  launch_custom_bf16(d_A, d_B, d_C, p);
  CUDA_CHECK(cudaGetLastError());
}

template <>
void launch_custom<float>(const float* d_A,
                          const float* d_B,
                          float* d_C,
                          const Problem& p) {
  launch_custom_tf32(d_A, d_B, d_C, p);
  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
static void run_cublas_gemm(cublasHandle_t handle,
                            const T* d_A,
                            const T* d_B,
                            float* d_C,
                            const Problem& p,
                            BenchDtype dtype) {
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
                            d_B,
                            cuda_dtype(dtype),
                            p.N,
                            d_A,
                            cuda_dtype(dtype),
                            p.K,
                            &beta,
                            d_C,
                            CUDA_R_32F,
                            p.N,
                            cublas_compute_type(dtype),
                            CUBLAS_GEMM_DEFAULT));
}

template <typename Fn>
static float time_cuda_ms(Fn&& fn, int warmup, int repeat) {
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

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed_ms / static_cast<float>(repeat);
}

static double flops_for(const Problem& p) {
  return 2.0 * static_cast<double>(p.M) * static_cast<double>(p.N) *
         static_cast<double>(p.K);
}

static double tflops_for(const Problem& p, double ms) {
  return flops_for(p) / (ms * 1.0e-3) / 1.0e12;
}

static double input_bytes_per_element(BenchDtype dtype) {
  switch (dtype) {
    case BenchDtype::kFp16:
    case BenchDtype::kBf16:
      return 2.0;
    case BenchDtype::kTf32:
      return 4.0;
  }
  return 4.0;
}

static double ideal_dram_bytes_for(const Problem& p, BenchDtype dtype) {
  const double input_bytes = input_bytes_per_element(dtype);
  return input_bytes *
             (static_cast<double>(p.M) * p.K +
              static_cast<double>(p.K) * p.N) +
         sizeof(float) * static_cast<double>(p.M) * p.N;
}

static double arithmetic_intensity_for(const Problem& p, BenchDtype dtype) {
  return flops_for(p) / ideal_dram_bytes_for(p, dtype);
}

static double memory_roof_tflops(double ai_flop_per_byte, double hbm_tbps) {
  return ai_flop_per_byte * hbm_tbps;
}

static double roofline_tflops(double compute_peak_tflops,
                              double mem_roof_tflops) {
  return std::min(compute_peak_tflops, mem_roof_tflops);
}

static const char* roofline_bound(double compute_peak_tflops,
                                  double mem_roof_tflops) {
  return (mem_roof_tflops < compute_peak_tflops) ? "memory" : "compute";
}

static double tensor_peak_tflops(const cudaDeviceProp& prop, BenchDtype dtype) {
  const std::string name = lower_copy(prop.name);
  if (name.find("h100") == std::string::npos) {
    throw std::runtime_error(
        "Tensor Core peak is only auto-filled for H100. Use --peak.");
  }

  // Dense H100 SXM-class Tensor Core peaks. Use --peak for PCIe parts,
  // sparse GEMM, or a different clock/power configuration.
  switch (dtype) {
    case BenchDtype::kFp16:
    case BenchDtype::kBf16:
      return 989.4;
    case BenchDtype::kTf32:
      return 494.7;
  }
  return 0.0;
}

static double bytes_to_kib(std::size_t bytes) {
  return static_cast<double>(bytes) / 1024.0;
}

static double bytes_to_gib(std::size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

static void print_hardware_info(std::ostream& os,
                                const cudaDeviceProp& prop,
                                int device,
                                const Config& cfg,
                                double peak_tflops) {
  os << "Hardware:\n";
  os << "  device: " << device << "\n";
  os << "  name: " << prop.name << "\n";
  os << "  compute capability: " << prop.major << "." << prop.minor << "\n";
  os << "  SM count: " << prop.multiProcessorCount << "\n";
  os << "  warp size: " << prop.warpSize << "\n";
  os << "  max threads per SM: " << prop.maxThreadsPerMultiProcessor << "\n";
  os << "  max threads per block: " << prop.maxThreadsPerBlock << "\n";
  os << "  SM clock: " << std::fixed << std::setprecision(1)
     << prop.clockRate / 1000.0 << " MHz\n";
  os << "  memory clock: " << std::fixed << std::setprecision(1)
     << prop.memoryClockRate / 1000.0 << " MHz\n";
  os << "  global memory: " << std::fixed << std::setprecision(2)
     << bytes_to_gib(prop.totalGlobalMem) << " GiB\n";
  os << "  memory bus width: " << prop.memoryBusWidth << " bit\n";
  os << "  L2 cache: " << std::fixed << std::setprecision(1)
     << bytes_to_kib(prop.l2CacheSize) << " KiB\n";
  os << "  shared memory per block: " << std::fixed << std::setprecision(1)
     << bytes_to_kib(prop.sharedMemPerBlock) << " KiB\n";
  os << "  shared memory per block opt-in: " << std::fixed
     << std::setprecision(1) << bytes_to_kib(prop.sharedMemPerBlockOptin)
     << " KiB\n";
  os << "  shared memory per SM: " << std::fixed << std::setprecision(1)
     << bytes_to_kib(prop.sharedMemPerMultiprocessor) << " KiB\n";
  os << "  registers per block: " << prop.regsPerBlock << "\n";
  os << "  registers per SM: " << prop.regsPerMultiprocessor << "\n";
  os << "  benchmark dtype: " << dtype_name(cfg.dtype) << "\n";
  os << "  theoretical Tensor Core dense peak used: " << std::fixed
     << std::setprecision(3) << peak_tflops << " TFLOPS";
  if (cfg.peak_tflops > 0.0) {
    os << " (--peak override)";
  }
  os << "\n";
  os << "  HBM bandwidth used for roofline: " << std::fixed
     << std::setprecision(3) << cfg.hbm_tbps << " TB/s\n";
  os << "  custom kernel ready: "
     << (custom_kernel_ready(cfg.dtype) ? "yes" : "no") << "\n";
}

static CheckResult compare_results(const std::vector<float>& got,
                                   const std::vector<float>& ref,
                                   double atol,
                                   double rtol) {
  if (got.size() != ref.size()) {
    throw std::runtime_error("compare_results size mismatch");
  }

  CheckResult out;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double g = static_cast<double>(got[i]);
    const double r = static_cast<double>(ref[i]);
    const double abs_err = std::fabs(g - r);
    const double denom = std::max(std::fabs(r), 1.0e-20);
    const double rel_err = abs_err / denom;
    if (abs_err > out.max_abs) {
      out.max_abs = abs_err;
      out.worst_index = i;
      out.worst_got = got[i];
      out.worst_ref = ref[i];
    }
    out.max_rel = std::max(out.max_rel, rel_err);
    if (abs_err > atol && rel_err > rtol) {
      ++out.bad_count;
    }
  }
  out.passed = (out.bad_count == 0);
  return out;
}

template <typename T>
static RowResult run_one_typed(cublasHandle_t handle,
                               const Config& cfg,
                               const Problem& p,
                               double atol,
                               double rtol) {
  const std::size_t count_A = checked_count(p.M, p.K, "A");
  const std::size_t count_B = checked_count(p.K, p.N, "B");
  const std::size_t count_C = checked_count(p.M, p.N, "C");

  std::vector<float> h_A_float(count_A);
  std::vector<float> h_B_float(count_B);
  fill_random(h_A_float, 0xA123ull + count_A);
  fill_random(h_B_float, 0xB456ull + count_B);

  std::vector<T> h_A = convert_vector<T>(h_A_float);
  std::vector<T> h_B = convert_vector<T>(h_B_float);
  std::vector<float> h_C(count_C);
  std::vector<float> h_ref(count_C);

  DeviceBuffer<T> d_A(count_A);
  DeviceBuffer<T> d_B(count_B);
  DeviceBuffer<float> d_C(count_C);
  DeviceBuffer<float> d_ref(count_C);

  CUDA_CHECK(cudaMemcpy(d_A.ptr, h_A.data(), count_A * sizeof(T),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B.ptr, h_B.data(), count_B * sizeof(T),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_C.ptr, 0, count_C * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_ref.ptr, 0, count_C * sizeof(float)));

  RowResult result;

  result.custom_ms = time_cuda_ms(
      [&]() {
        launch_custom<T>(d_A.ptr, d_B.ptr, d_C.ptr, p);
      },
      cfg.warmup,
      cfg.repeat);
  result.custom_tflops = tflops_for(p, result.custom_ms);

  if (cfg.check) {
    run_cublas_gemm<T>(handle, d_A.ptr, d_B.ptr, d_ref.ptr, p, cfg.dtype);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C.ptr, count_C * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref.ptr, count_C * sizeof(float),
                          cudaMemcpyDeviceToHost));
    result.check = compare_results(h_C, h_ref, atol, rtol);
  }

  if (cfg.run_cublas) {
    result.cublas_ms = time_cuda_ms(
        [&]() {
          run_cublas_gemm<T>(handle, d_A.ptr, d_B.ptr, d_ref.ptr, p,
                             cfg.dtype);
        },
        cfg.warmup,
        cfg.repeat);
    result.cublas_tflops = tflops_for(p, result.cublas_ms);
  }

  return result;
}

static RowResult run_one(cublasHandle_t handle,
                         const Config& cfg,
                         const Problem& p,
                         double atol,
                         double rtol) {
  switch (cfg.dtype) {
    case BenchDtype::kFp16:
      return run_one_typed<half>(handle, cfg, p, atol, rtol);
    case BenchDtype::kBf16:
      return run_one_typed<__nv_bfloat16>(handle, cfg, p, atol, rtol);
    case BenchDtype::kTf32:
      return run_one_typed<float>(handle, cfg, p, atol, rtol);
  }
  throw std::runtime_error("Unsupported dtype");
}

static void print_csv_header() {
  std::cout
      << "dtype,M,N,K,custom_ms,custom_tflops,custom_peak_pct,"
      << "ideal_dram_bytes,ai_flop_per_byte,hbm_tbps,"
      << "compute_peak_tflops,mem_roof_tflops,roofline_tflops,"
      << "roofline_bound,custom_roofline_pct,"
      << "cublas_ms,cublas_tflops,cublas_peak_pct,cublas_roofline_pct,"
      << "max_abs,max_rel,bad_count,passed,kernel_ready\n";
}

static void print_csv_row(const Config& cfg,
                          const Problem& p,
                          const RowResult& r,
                          double peak_tflops) {
  const double ai = arithmetic_intensity_for(p, cfg.dtype);
  const double mem_roof = memory_roof_tflops(ai, cfg.hbm_tbps);
  const double roof = roofline_tflops(peak_tflops, mem_roof);
  const double custom_peak_pct = r.custom_tflops / peak_tflops * 100.0;
  const double custom_roof_pct = r.custom_tflops / roof * 100.0;
  const double cublas_peak_pct = r.cublas_tflops / peak_tflops * 100.0;
  const double cublas_roof_pct = r.cublas_tflops / roof * 100.0;

  std::cout << dtype_name(cfg.dtype) << "," << p.M << "," << p.N << ","
            << p.K << "," << std::setprecision(8) << r.custom_ms << ","
            << r.custom_tflops << "," << custom_peak_pct << ","
            << ideal_dram_bytes_for(p, cfg.dtype) << "," << ai << ","
            << cfg.hbm_tbps << "," << peak_tflops << "," << mem_roof << ","
            << roof << "," << roofline_bound(peak_tflops, mem_roof) << ","
            << custom_roof_pct << "," << r.cublas_ms << ","
            << r.cublas_tflops << "," << cublas_peak_pct << ","
            << cublas_roof_pct << "," << r.check.max_abs << ","
            << r.check.max_rel << "," << r.check.bad_count << ","
            << (r.check.passed ? 1 : 0) << ","
            << (custom_kernel_ready(cfg.dtype) ? 1 : 0) << "\n";
}

static void print_human_row(const Config& cfg,
                            const Problem& p,
                            const RowResult& r,
                            double peak_tflops) {
  const double ai = arithmetic_intensity_for(p, cfg.dtype);
  const double mem_roof = memory_roof_tflops(ai, cfg.hbm_tbps);
  const double roof = roofline_tflops(peak_tflops, mem_roof);

  std::cout << "\nProblem: " << p.M << "x" << p.N << "x" << p.K << "\n";
  std::cout << "  dtype: " << dtype_name(cfg.dtype) << "\n";
  std::cout << "  custom: " << std::fixed << std::setprecision(6)
            << r.custom_ms << " ms, " << std::setprecision(4)
            << r.custom_tflops << " TFLOPS, "
            << (r.custom_tflops / peak_tflops * 100.0) << "% peak\n";
  std::cout << "  roofline: AI=" << std::setprecision(3) << ai
            << " FLOP/B, mem_roof=" << mem_roof
            << " TFLOPS, bound=" << roofline_bound(peak_tflops, mem_roof)
            << ", custom_roof=" << (r.custom_tflops / roof * 100.0)
            << "%\n";
  if (cfg.run_cublas) {
    std::cout << "  cuBLAS: " << std::setprecision(6) << r.cublas_ms
              << " ms, " << std::setprecision(4) << r.cublas_tflops
              << " TFLOPS, "
              << (r.cublas_tflops / peak_tflops * 100.0) << "% peak\n";
  }
  if (cfg.check) {
    std::cout << "  check: " << (r.check.passed ? "PASS" : "FAIL")
              << ", max_abs=" << r.check.max_abs
              << ", max_rel=" << r.check.max_rel
              << ", bad_count=" << r.check.bad_count;
    if (!r.check.passed) {
      std::cout << ", worst_index=" << r.check.worst_index
                << ", got=" << r.check.worst_got
                << ", ref=" << r.check.worst_ref;
    }
    std::cout << "\n";
  }
}

int main(int argc, char** argv) {
  try {
    Config cfg = parse_args(argc, argv);

    if (!custom_kernel_ready(cfg.dtype) && !cfg.allow_stub) {
      std::ostringstream oss;
      oss << "The " << dtype_name(cfg.dtype)
          << " custom kernel is still a stub. Implement it, set "
          << "USER_TENSOR_GEMM_" << (cfg.dtype == BenchDtype::kFp16
                                         ? "FP16"
                                         : cfg.dtype == BenchDtype::kBf16
                                               ? "BF16"
                                               : "TF32")
          << "_READY to 1, or pass --allow-stub to test the harness.";
      throw std::runtime_error(oss.str());
    }

    CUDA_CHECK(cudaSetDevice(cfg.device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));

    const double peak_tflops =
        (cfg.peak_tflops > 0.0) ? cfg.peak_tflops
                                : tensor_peak_tflops(prop, cfg.dtype);
    const double atol = (cfg.atol > 0.0) ? cfg.atol : default_atol(cfg.dtype);
    const double rtol = (cfg.rtol > 0.0) ? cfg.rtol : default_rtol(cfg.dtype);

    std::ostream& info_os = cfg.csv ? std::cerr : std::cout;
    print_hardware_info(info_os, prop, cfg.device, cfg, peak_tflops);
    info_os << "Benchmark config:\n";
    info_os << "  warmup: " << cfg.warmup << "\n";
    info_os << "  repeat: " << cfg.repeat << "\n";
    info_os << "  check: " << (cfg.check ? "yes" : "no") << "\n";
    info_os << "  cuBLAS timing: " << (cfg.run_cublas ? "yes" : "no")
            << "\n";
    info_os << "  atol: " << atol << "\n";
    info_os << "  rtol: " << rtol << "\n";
    if (!custom_kernel_ready(cfg.dtype)) {
      info_os << "  warning: running deterministic zero-fill stub; custom "
              << "timing and correctness are not meaningful.\n";
    }

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    if (cfg.csv) {
      print_csv_header();
    }

    bool all_passed = true;
    for (const Problem& p : cfg.problems) {
      validate_shape(p, cfg);
      RowResult result = run_one(handle, cfg, p, atol, rtol);
      if (cfg.check) {
        all_passed = all_passed && result.check.passed;
      }
      if (cfg.csv) {
        print_csv_row(cfg, p, result, peak_tflops);
      } else {
        print_human_row(cfg, p, result, peak_tflops);
      }
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    return (cfg.check && !all_passed) ? 2 : 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
