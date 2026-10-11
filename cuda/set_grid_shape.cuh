#pragma once
#include <cstdlib>
#include <cstdint>
#include <string_view>
#include <utility>
#include <cuda_runtime.h>
#include "parse_num.hpp"

inline int check_grid_shape(uint32_t x, uint32_t y, uint32_t z, const cudaDeviceProp& prop) {
  if (x == 0 || y == 0 || z == 0) {
    return EXIT_FAILURE;
  }
  if (x > prop.maxGridSize[0] || y > prop.maxGridSize[1] || z > prop.maxGridSize[2]) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
