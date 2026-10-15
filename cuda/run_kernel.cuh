#pragma once
#include <cuda_runtime.h>
#include "threads_kernel.cuh"

inline cudaError_t run_kernel(const dim3& grid_shape, const dim3& block_shape) {
  kernel<<<grid_shape, block_shape>>>();
  auto kernel_err = cudaGetLastError();
  if (kernel_err != cudaSuccess) {
    return kernel_err;
  }
  return cudaDeviceSynchronize();
}
