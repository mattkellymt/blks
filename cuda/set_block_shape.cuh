#pragma once
#include <cstdlib>
#include <cstdint>
#include <string_view>
#include <tuple>
#include <cuda_runtime.h>
#include "parse_num.hpp"

inline int check_block_shape(uint32_t x, uint32_t y, uint32_t z, const cudaDeviceProp& prop) {
  if (x == 0 || y == 0 || z == 0) {
    return EXIT_FAILURE;
  }
  auto threads_per_block = x * y * z;
  if (threads_per_block > prop.maxThreadsPerBlock) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
