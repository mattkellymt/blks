#pragma once
#include <cstdlib>
#include <cstdint>
#include <tuple>
#include <cuda_runtime.h>

inline std::tuple<cudaDeviceProp, int> get_prop(uint32_t device_id) {
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
  if (err != cudaSuccess) {
    return {cudaDeviceProp{}, EXIT_FAILURE};
  }
  return {prop, EXIT_SUCCESS};
}
