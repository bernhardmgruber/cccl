// Tests the overhead of the NVTX API when enabled. Compile and run from this directory with:
//   nvcc bench_nvtx_perf.cu -I../../cub -I../../thrust -I../../libcudacxx/include && ./a.out
// Variations:
// * Add `-DNVTX_DISABLE` to disable the NVTX API, should be faster
// * Run binary with nsys profile ./a.out to attach Nsight Systems , should be slower

#include <cub/device/device_for.cuh>

#include <thrust/iterator/counting_iterator.h>

#include <chrono>

struct Op
{
  _CCCL_HOST_DEVICE void operator()(int) const
  {
  }
};

constexpr auto reps = 10'000;

int main()
{
  using clock = std::chrono::high_resolution_clock;

  thrust::counting_iterator<int> it{0};

  const auto start = clock::now();
  for(int i = 0; i < reps; i++) {
    cub::DeviceFor::ForEach(it, it + 1, Op{});
  }
  const auto end = clock::now();
  std::cout << "Average launch time: " << std::chrono::duration<double, std::micro>(end - start).count() / reps << "μs\n";

  cudaDeviceSynchronize();
}
