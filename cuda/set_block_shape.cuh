#pragma once
#include <cstdlib>
#include <cstdint>
#include <string_view>
#include <tuple>
#include <cuda_runtime.h>
#include "parse_num.hpp"

inline int check_block_shape(dim3& block_shape, const cudaDeviceProp& prop) {
  if (block_shape.x == 0 || block_shape.y == 0 || block_shape.z == 0) {
    return EXIT_FAILURE;
  }
  auto threads_per_block = block_shape.x * block_shape.y * block_shape.z;
  if (threads_per_block > prop.maxThreadsPerBlock) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
