#define CUB_DETAIL_BEFORE_NVTX_RANGE_SCOPE(name) static_assert(false, "");
#define CCCL_DISABLE_NVTX

#include <cub/device/device_for.cuh>

#include <thrust/iterator/counting_iterator.h>

#include <cuda/std/functional>

int main()
{
  thrust::counting_iterator<int> it{0};
  cub::DeviceFor::ForEach(it, it + 16, ::cuda::std::negate<int>{});
  cudaDeviceSynchronize();
}
