#pragma once
#include <cuda_runtime.h>
#include "threads_kernel.cuh"

inline cudaError_t run_kernel(const int3& grid_shape, const int3& block_shape) {
  dim3 grid_dim(grid_shape.x, grid_shape.y, grid_shape.z);
  dim3 block_dim(block_shape.x, block_shape.y, block_shape.z);
  kernel<<<grid_dim, block_dim>>>();
  auto kernel_err = cudaGetLastError();
  if (kernel_err != cudaSuccess) {
    return kernel_err;
  }
  return cudaDeviceSynchronize();
}
