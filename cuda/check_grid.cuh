#pragma once
#include <cstdlib>
#include <cuda_runtime.h>

inline int check_grid(const dim3& grid_shape, const cudaDeviceProp& prop) {
  if (grid_shape.x == 0 || grid_shape.y == 0 || grid_shape.z == 0) {
    return EXIT_FAILURE;
  }
  if (grid_shape.x > prop.maxGridSize[0] || grid_shape.y > prop.maxGridSize[1] || grid_shape.z > prop.maxGridSize[2]) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
