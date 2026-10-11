#pragma once
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

inline int get_prop(cudaDeviceProp& prop, uint32_t device_id) {
  cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
  if (err != cudaSuccess) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
