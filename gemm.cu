// Standalone GEMM benchmark for H100.
//
// Contract for the custom kernel:
//   A, B, C are row-major float matrices.
//   C[M, N] = A[M, K] * B[K, N].
//
// Optimize gemm_kernel below. The benchmark code after the "Do not edit unless
// needed" marker handles timing, cuBLAS reference, correctness, TFLOPS, and
// percentage of theoretical peak.

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define BM 128
#define BN 128
#define BK 32

#define TM 8
#define TN 8

#define OFFSET(row_offset, col_len, col_offset) (row_offset) * (col_len) + col_offset
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

#define MOVONCE 4
// ========================= Optimize this kernel =========================
__device__ void move_shm_out(float* A,
                        float * shm,
                        int start_row,
                        int start_col,
                        int bm,
                        int bn,
                        int M,
                        int N) {
  int num_elements = bm * bn / MOVONCE;
  int nthreads = blockDim.y * blockDim.x;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  for(int i = tid; i < num_elements; i += nthreads) {
    int row = (i << 2) / bn;
    int col = (i << 2) % bn;
    FLOAT4(shm[row * bn + col]) = FLOAT4(A[(start_row + row) * N + (start_col + col)]);
  }
}

__device__ void move_shm_in(float* A,
                        float * shm,
                        int tid,
                        int start_row,
                        int start_col,
                        int bm,
                        int bn,
                        int M,
                        int N) {
  int num_elements = bm * bn / 4;
  int nthreads = blockDim.y * blockDim.x;
  for(int i = tid; i < num_elements; i += nthreads) {
    int row = (i << 2) / bn;
    int col = (i << 2) % bn;
    if (start_row + row < M && start_col + col < N)
      FLOAT4(shm[row * bn + col]) = FLOAT4(A[(start_row + row) * N + (start_col + col)]);
  }
}
// 因为一个线程的寄存器本身对不同元素就是可以共享的，
// 因此可以不用像共享内存一样显示把小tile 放到寄存器中
// 本来共享内存同一个迭代步k 同一行访问同一个元素，
// 因为hbm对不同线程无法共享，使用共享内存共享
// 现在这个线程的同一行的元素的结果直接访问共享内存这个迭代步的值即可
// 所以这边寄存器层次对K 的划分TK 是没必要的
__global__ void gemm_kernel(float* A,
                            float* B,
                            float* C,
                            int M,
                            int N,
                            int K) {
    __shared__ float shm_A[BM * BK];
    __shared__ float shm_B[BK * BN];
    int block_row_start = blockIdx.y * BM;
    int block_col_start = blockIdx.x * BN;

    int row_start = block_row_start + threadIdx.y * TM;
    int col_start = block_col_start + threadIdx.x * TN;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    float tmp_c[TM][TN] = {0.0};
    for (int bk = 0; bk < K; bk += BK){
      // 协作搬运
      move_shm_in(A, shm_A, tid, blockIdx.y * BM, bk, BM, BK, M, K);
      move_shm_in(B, shm_B, tid, bk, blockIdx.x * BN, BK, BN, K, N);
      __syncthreads();
      // 实现这个shm tile 的乘法,分成一些register tile的乘法
      #pragma unroll
      for(int i = 0; i < BK; i ++) {
       #pragma unroll
       for(int tm = 0; tm < TM; tm++) {
        #pragma unroll
        for(int tn = 0; tn < TN; tn++) {
            tmp_c[tm][tn] += shm_A[((threadIdx.y * TM)+ tm) * BK + i ]
            * shm_B[(i) * BN + threadIdx.x * TN + tn];
        }
       }
      }
      __syncthreads();
    }

    for(int i = 0; i < TM; i++) {
      for(int j = 0; j < TN; j+=4) {
        if (row_start + i < M && col_start + j < N)
          FLOAT4(C[(row_start + i)* N + col_start + j]) = FLOAT4(tmp_c[i][j]);
      }
    }
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
  bool csv = false;
  bool run_cublas = true;
  bool cublas_fast_tf32 = false;
  double peak_tflops = 0.0;
  std::string peak_mode = "fp32";
  double hbm_tbps = 3.35;
  double atol = 1.0e-2;
  double rtol = 1.0e-3;
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

static bool contains_case_insensitive(const std::string& haystack,
                                      const std::string& needle) {
  return lower_copy(haystack).find(lower_copy(needle)) != std::string::npos;
}

static int parse_int_value(const char* s, const char* name) {
  char* end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (!s[0] || (end && *end) || v <= 0 || v > std::numeric_limits<int>::max()) {
    std::ostringstream oss;
    oss << "Invalid positive int for " << name << ": " << s;
    throw std::runtime_error(oss.str());
  }
  return static_cast<int>(v);
}

static int parse_nonnegative_int_value(const char* s, const char* name) {
  char* end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (!s[0] || (end && *end) || v < 0 || v > std::numeric_limits<int>::max()) {
    std::ostringstream oss;
    oss << "Invalid nonnegative int for " << name << ": " << s;
    throw std::runtime_error(oss.str());
  }
  return static_cast<int>(v);
}

static double parse_double_value(const char* s, const char* name) {
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
  const std::size_t p2 = (p1 == std::string::npos) ? std::string::npos
                                                   : s.find('x', p1 + 1);
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

static void print_help(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n"
      << "\n"
      << "Default: benchmark square FP32 GEMM sizes 512,1024,2048,4096.\n"
      << "Matrix layout: row-major C[M,N] = A[M,K] * B[K,N].\n"
      << "\n"
      << "Options:\n"
      << "  --device ID              CUDA device id, default 0\n"
      << "  --shape MxNxK            Add one problem shape, repeatable\n"
      << "  --sizes A,B,C            Square sizes, for example 1024,2048,4096\n"
      << "  --warmup N               Warmup launches, default 5\n"
      << "  --repeat N               Timed launches, default 20\n"
      << "  --peak TFLOPS            Override theoretical peak TFLOPS\n"
      << "  --peak-mode MODE         fp32(default), tf32, fp16, bf16, fp8\n"
      << "  --hbm-tbps TBPS          HBM bandwidth for roofline, default 3.35\n"
      << "  --atol X                 Absolute tolerance, default 1e-2\n"
      << "  --rtol X                 Relative tolerance, default 1e-3\n"
      << "  --no-check               Skip cuBLAS correctness check\n"
      << "  --no-cublas              Skip timed cuBLAS baseline\n"
      << "  --cublas-fast-tf32       Time cuBLAS with FAST_TF32 compute\n"
      << "  --csv                    Print CSV rows\n"
      << "  --help                   Show this message\n"
      << "\n"
      << "Examples:\n"
      << "  " << argv0 << " --shape 4096x4096x4096\n"
      << "  " << argv0 << " --sizes 1024,2048,4096,8192 --repeat 50\n"
      << "  " << argv0 << " --peak-mode tf32 --cublas-fast-tf32\n"
      << "  " << argv0 << " --hbm-tbps 3.35 --csv > gemm.csv\n"
      << "  " << argv0 << " --peak 989.4\n";
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
      cfg.device = parse_nonnegative_int_value(need_value("--device"), "--device");
    } else if (arg == "--shape") {
      cfg.problems.push_back(parse_shape(need_value("--shape")));
    } else if (arg == "--sizes") {
      std::vector<Problem> sizes = parse_square_list(need_value("--sizes"));
      cfg.problems.insert(cfg.problems.end(), sizes.begin(), sizes.end());
    } else if (arg == "--warmup") {
      cfg.warmup = parse_nonnegative_int_value(need_value("--warmup"), "--warmup");
    } else if (arg == "--repeat") {
      cfg.repeat = parse_int_value(need_value("--repeat"), "--repeat");
    } else if (arg == "--peak") {
      cfg.peak_tflops = parse_double_value(need_value("--peak"), "--peak");
    } else if (arg == "--peak-mode") {
      cfg.peak_mode = lower_copy(need_value("--peak-mode"));
    } else if (arg == "--hbm-tbps") {
      cfg.hbm_tbps = parse_double_value(need_value("--hbm-tbps"), "--hbm-tbps");
    } else if (arg == "--atol") {
      cfg.atol = parse_double_value(need_value("--atol"), "--atol");
    } else if (arg == "--rtol") {
      cfg.rtol = parse_double_value(need_value("--rtol"), "--rtol");
    } else if (arg == "--no-check") {
      cfg.check = false;
    } else if (arg == "--no-cublas") {
      cfg.run_cublas = false;
    } else if (arg == "--cublas-fast-tf32") {
      cfg.cublas_fast_tf32 = true;
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

static std::size_t checked_count(int a, int b, const char* name) {
  const auto aa = static_cast<std::uint64_t>(a);
  const auto bb = static_cast<std::uint64_t>(b);
  const std::uint64_t max_count =
      std::numeric_limits<std::size_t>::max() / sizeof(float);
  if (aa != 0 && bb > max_count / aa) {
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

static void launch_custom(float* d_A,
                          float* d_B,
                          float* d_C,
                          const Problem& p) {
  dim3 block(BN / TN, BM / TM);
  dim3 grid((p.N + BN - 1) / BN,
            (p.M + BM - 1) / BM);
  gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, p.M, p.N, p.K);
  CUDA_CHECK(cudaGetLastError());
}

static void run_cublas_gemm(cublasHandle_t handle,
                            const float* d_A,
                            const float* d_B,
                            float* d_C,
                            const Problem& p,
                            cublasComputeType_t compute_type) {
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
                            CUDA_R_32F,
                            p.N,
                            d_A,
                            CUDA_R_32F,
                            p.K,
                            &beta,
                            d_C,
                            CUDA_R_32F,
                            p.N,
                            compute_type,
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

static double tflops_for(const Problem& p, double ms) {
  const double flops =
      2.0 * static_cast<double>(p.M) * static_cast<double>(p.N) *
      static_cast<double>(p.K);
  return flops / (ms * 1.0e-3) / 1.0e12;
}

static double flops_for(const Problem& p) {
  return 2.0 * static_cast<double>(p.M) * static_cast<double>(p.N) *
         static_cast<double>(p.K);
}

static double ideal_dram_bytes_for(const Problem& p) {
  // Minimum dense GEMM DRAM traffic: read A once, read B once, write C once.
  // This is a roofline lower bound, not a measured hardware byte count.
  constexpr double bytes_per_float = sizeof(float);
  return bytes_per_float *
         (static_cast<double>(p.M) * static_cast<double>(p.K) +
          static_cast<double>(p.K) * static_cast<double>(p.N) +
          static_cast<double>(p.M) * static_cast<double>(p.N));
}

static double arithmetic_intensity_for(const Problem& p) {
  return flops_for(p) / ideal_dram_bytes_for(p);
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

static double fp32_cuda_core_peak_tflops(const cudaDeviceProp& prop) {
  // H100 has 128 FP32 CUDA cores per SM. The factor 2 counts FMA as 2 FLOPs.
  constexpr double fp32_cores_per_sm = 128.0;
  return static_cast<double>(prop.multiProcessorCount) * fp32_cores_per_sm *
         2.0 * static_cast<double>(prop.clockRate) * 1.0e-9;
}

static double peak_tflops_for_mode(const cudaDeviceProp& prop,
                                   const std::string& mode) {
  const std::string m = lower_copy(mode);
  if (m == "fp32") {
    return fp32_cuda_core_peak_tflops(prop);
  }

  if (!contains_case_insensitive(prop.name, "h100")) {
    throw std::runtime_error(
        "Only fp32 peak is auto-detected on non-H100 devices. Use --peak.");
  }

  // Dense published H100 SXM 80GB peaks. Use --peak to override if your board
  // or datatype contract is different.
  if (m == "tf32") {
    return 494.7;
  }
  if (m == "fp16" || m == "bf16") {
    return 989.4;
  }
  if (m == "fp8") {
    return 1978.9;
  }

  throw std::runtime_error("Unknown --peak-mode: " + mode);
}

static double bytes_to_kib(std::size_t bytes) {
  return static_cast<double>(bytes) / 1024.0;
}

static double bytes_to_gib(std::size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

static void print_hardware_info(const cudaDeviceProp& prop,
                                int device,
                                double peak_tflops,
                                const Config& cfg) {
  std::cout << "Hardware:\n";
  std::cout << "  device: " << device << "\n";
  std::cout << "  name: " << prop.name << "\n";
  std::cout << "  compute capability: " << prop.major << "." << prop.minor
            << "\n";
  std::cout << "  SM count: " << prop.multiProcessorCount << "\n";
  std::cout << "  max threads per SM: " << prop.maxThreadsPerMultiProcessor
            << "\n";
  std::cout << "  max threads per block: " << prop.maxThreadsPerBlock << "\n";
  std::cout << "  warp size: " << prop.warpSize << "\n";
  std::cout << "  SM clock: " << std::fixed << std::setprecision(1)
            << prop.clockRate / 1000.0 << " MHz\n";
  std::cout << "  memory clock: " << std::fixed << std::setprecision(1)
            << prop.memoryClockRate / 1000.0 << " MHz\n";
  std::cout << "  global memory: " << std::fixed << std::setprecision(2)
            << bytes_to_gib(prop.totalGlobalMem) << " GiB\n";
  std::cout << "  memory bus width: " << prop.memoryBusWidth << " bit\n";
  std::cout << "  L2 cache: " << std::fixed << std::setprecision(1)
            << bytes_to_kib(prop.l2CacheSize) << " KiB\n";
  std::cout << "  shared memory per block: " << std::fixed
            << std::setprecision(1) << bytes_to_kib(prop.sharedMemPerBlock)
            << " KiB\n";
  std::cout << "  shared memory per block opt-in: " << std::fixed
            << std::setprecision(1)
            << bytes_to_kib(prop.sharedMemPerBlockOptin) << " KiB\n";
  std::cout << "  shared memory per SM: " << std::fixed
            << std::setprecision(1)
            << bytes_to_kib(prop.sharedMemPerMultiprocessor) << " KiB\n";
  std::cout << "  registers per block: " << prop.regsPerBlock << "\n";
  std::cout << "  registers per SM: " << prop.regsPerMultiprocessor << "\n";
  std::cout << "  theoretical peak used: " << std::fixed
            << std::setprecision(3) << peak_tflops << " TFLOPS";
  if (cfg.peak_tflops > 0.0) {
    std::cout << " (--peak override)";
  } else {
    std::cout << " (--peak-mode " << cfg.peak_mode << ")";
  }
  std::cout << "\n";
  std::cout << "  HBM bandwidth used for roofline: " << std::fixed
            << std::setprecision(3) << cfg.hbm_tbps << " TB/s\n\n";
}

static CheckResult compare_results(const std::vector<float>& got,
                                   const std::vector<float>& ref,
                                   double atol,
                                   double rtol) {
  if (got.size() != ref.size()) {
    throw std::runtime_error("Internal error: result sizes differ");
  }

  CheckResult r;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double g = static_cast<double>(got[i]);
    const double e = static_cast<double>(ref[i]);
    const double abs_err = std::abs(g - e);
    const double denom = std::max(1.0e-12, std::abs(e));
    const double rel_err = abs_err / denom;

    if (abs_err > r.max_abs || rel_err > r.max_rel) {
      r.max_abs = std::max(r.max_abs, abs_err);
      r.max_rel = std::max(r.max_rel, rel_err);
      r.worst_index = i;
      r.worst_got = got[i];
      r.worst_ref = ref[i];
    }

    if (abs_err > atol && rel_err > rtol) {
      ++r.bad_count;
    }
  }
  r.passed = (r.bad_count == 0);
  return r;
}

static void print_text_header() {
  std::cout << std::setw(7) << "M" << std::setw(7) << "N" << std::setw(7)
            << "K" << std::setw(12) << "custom_ms" << std::setw(15)
            << "custom_TF" << std::setw(10) << "peak_%"
            << std::setw(10) << "AI" << std::setw(12) << "roof_TF"
            << std::setw(9) << "bound" << std::setw(10) << "roof_%"
            << std::setw(12) << "cublas_ms" << std::setw(15) << "cublas_TF"
            << std::setw(10) << "peak_%" << std::setw(10) << "roof_%"
            << std::setw(12) << "max_abs" << std::setw(12) << "max_rel"
            << std::setw(10) << "bad" << std::setw(8) << "ok" << "\n";
}

static void print_csv_header() {
  std::cout << "M,N,K,custom_ms,custom_tflops,custom_peak_pct,"
            << "ideal_dram_bytes,ai_flop_per_byte,hbm_tbps,"
            << "compute_peak_tflops,mem_roof_tflops,roofline_tflops,"
            << "roofline_bound,custom_roofline_pct,"
            << "cublas_ms,cublas_tflops,cublas_peak_pct,cublas_roofline_pct,"
            << "max_abs,max_rel,bad_count,passed\n";
}

static bool run_one_problem(const Config& cfg,
                            cublasHandle_t handle,
                            const Problem& p,
                            double peak_tflops,
                            cublasComputeType_t timed_cublas_compute) {
  const std::size_t count_A = checked_count(p.M, p.K, "A");
  const std::size_t count_B = checked_count(p.K, p.N, "B");
  const std::size_t count_C = checked_count(p.M, p.N, "C");

  std::vector<float> h_A(count_A);
  std::vector<float> h_B(count_B);
  std::vector<float> h_C(count_C, 0.0f);
  std::vector<float> h_ref(count_C, 0.0f);

  const std::uint64_t seed_base =
      0x9e3779b97f4a7c15ULL ^ (static_cast<std::uint64_t>(p.M) << 32) ^
      (static_cast<std::uint64_t>(p.N) << 16) ^ static_cast<std::uint64_t>(p.K);
  fill_random(h_A, seed_base);
  fill_random(h_B, seed_base ^ 0xd1b54a32d192ed03ULL);

  DeviceBuffer<float> d_A(count_A);
  DeviceBuffer<float> d_B(count_B);
  DeviceBuffer<float> d_C(count_C);
  DeviceBuffer<float> d_ref(count_C);

  CUDA_CHECK(cudaMemcpy(d_A.ptr, h_A.data(), count_A * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B.ptr, h_B.data(), count_B * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_C.ptr, 0, count_C * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_ref.ptr, 0, count_C * sizeof(float)));

  float custom_ms = time_cuda_ms(
      [&]() { launch_custom(d_A.ptr, d_B.ptr, d_C.ptr, p); }, cfg.warmup,
      cfg.repeat);

  const double custom_tf = tflops_for(p, custom_ms);
  const double custom_pct = 100.0 * custom_tf / peak_tflops;
  const double ideal_bytes = ideal_dram_bytes_for(p);
  const double ai = arithmetic_intensity_for(p);
  const double mem_roof_tf = memory_roof_tflops(ai, cfg.hbm_tbps);
  const double roof_tf = roofline_tflops(peak_tflops, mem_roof_tf);
  const char* bound = roofline_bound(peak_tflops, mem_roof_tf);
  const double custom_roof_pct = 100.0 * custom_tf / roof_tf;

  double cublas_ms = std::numeric_limits<double>::quiet_NaN();
  double cublas_tf = std::numeric_limits<double>::quiet_NaN();
  double cublas_pct = std::numeric_limits<double>::quiet_NaN();
  double cublas_roof_pct = std::numeric_limits<double>::quiet_NaN();
  if (cfg.run_cublas) {
    cublas_ms = time_cuda_ms(
        [&]() {
          run_cublas_gemm(handle, d_A.ptr, d_B.ptr, d_ref.ptr, p,
                          timed_cublas_compute);
        },
        cfg.warmup, cfg.repeat);
    cublas_tf = tflops_for(p, cublas_ms);
    cublas_pct = 100.0 * cublas_tf / peak_tflops;
    cublas_roof_pct = 100.0 * cublas_tf / roof_tf;
  }

  CheckResult check;
  if (cfg.check) {
    CUDA_CHECK(cudaMemset(d_ref.ptr, 0, count_C * sizeof(float)));
    run_cublas_gemm(handle, d_A.ptr, d_B.ptr, d_ref.ptr, p,
                    CUBLAS_COMPUTE_32F_PEDANTIC);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C.ptr, count_C * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ref.data(), d_ref.ptr, count_C * sizeof(float),
                          cudaMemcpyDeviceToHost));
    check = compare_results(h_C, h_ref, cfg.atol, cfg.rtol);
  }

  if (cfg.csv) {
    std::cout << p.M << "," << p.N << "," << p.K << "," << std::setprecision(6)
              << custom_ms << "," << custom_tf << "," << custom_pct << ","
              << ideal_bytes << "," << ai << "," << cfg.hbm_tbps << ","
              << peak_tflops << "," << mem_roof_tf << "," << roof_tf << ","
              << bound << "," << custom_roof_pct << ","
              << cublas_ms << "," << cublas_tf << "," << cublas_pct << ","
              << cublas_roof_pct << ","
              << check.max_abs << "," << check.max_rel << ","
              << check.bad_count << "," << (check.passed ? 1 : 0) << "\n";
  } else {
    std::cout << std::fixed << std::setprecision(3) << std::setw(7) << p.M
              << std::setw(7) << p.N << std::setw(7) << p.K << std::setw(12)
              << custom_ms << std::setw(15) << custom_tf << std::setw(10)
              << custom_pct << std::setw(10) << ai << std::setw(12)
              << roof_tf << std::setw(9) << bound << std::setw(10)
              << custom_roof_pct << std::setw(12) << cublas_ms
              << std::setw(15) << cublas_tf << std::setw(10) << cublas_pct
              << std::setw(10) << cublas_roof_pct << std::scientific
              << std::setprecision(2) << std::setw(12) << check.max_abs
              << std::setw(12) << check.max_rel << std::fixed
              << std::setprecision(0) << std::setw(10)
              << static_cast<double>(check.bad_count) << std::setw(8)
              << (check.passed ? "yes" : "NO") << "\n";

    if (cfg.check && !check.passed) {
      const int row = static_cast<int>(check.worst_index / p.N);
      const int col = static_cast<int>(check.worst_index % p.N);
      std::cerr << "  Worst mismatch at C[" << row << "," << col
                << "]: got=" << std::setprecision(9) << check.worst_got
                << " ref=" << check.worst_ref << " max_abs=" << check.max_abs
                << " max_rel=" << check.max_rel << "\n";
    }
  }

  return !cfg.check || check.passed;
}

int main(int argc, char** argv) {
  try {
    Config cfg = parse_args(argc, argv);
    CUDA_CHECK(cudaSetDevice(cfg.device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.device));

    const double peak_tflops =
        (cfg.peak_tflops > 0.0) ? cfg.peak_tflops
                                : peak_tflops_for_mode(prop, cfg.peak_mode);

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));

    const cublasComputeType_t timed_cublas_compute =
        cfg.cublas_fast_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32
                             : CUBLAS_COMPUTE_32F_PEDANTIC;

    if (!cfg.csv) {
      print_hardware_info(prop, cfg.device, peak_tflops, cfg);
      std::cout << "Timing: warmup=" << cfg.warmup
                << ", repeat=" << cfg.repeat
                << ", cuBLAS="
                << (cfg.run_cublas ? (cfg.cublas_fast_tf32 ? "fast_tf32"
                                                           : "fp32_pedantic")
                                   : "off")
                << ", check=" << (cfg.check ? "on" : "off") << "\n";
      print_text_header();
    } else {
      print_csv_header();
    }

    bool all_ok = true;
    for (const Problem& p : cfg.problems) {
      all_ok = run_one_problem(cfg, handle, p, peak_tflops,
                               timed_cublas_compute) &&
               all_ok;
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaDeviceSynchronize());
    return all_ok ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
