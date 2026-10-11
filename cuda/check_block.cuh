#pragma once
#include <cstdlib>
#include <cuda_runtime.h>

inline int check_block(const dim3& block_shape, const cudaDeviceProp& prop) {
  if (block_shape.x == 0 || block_shape.y == 0 || block_shape.z == 0) {
    return EXIT_FAILURE;
  }
  if (block_shape.x > prop.maxThreadsDim[0] || block_shape.y > prop.maxThreadsDim[1] || block_shape.z > prop.maxThreadsDim[2]) {
    return EXIT_FAILURE;
  }
  auto threads_per_block = block_shape.x * block_shape.y * block_shape.z;
  if (threads_per_block > prop.maxThreadsPerBlock) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
