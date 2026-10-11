#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

extern "C" {
  cudaError_t c_set_device(uint32_t device_id);
  void c_welcome();
  size_t c_prop_size();
  cudaError_t c_get_prop(cudaDeviceProp* prop, uint32_t device_id);
  int c_parse_u32(uint32_t* val, const char* str);
  int c_check_grid_shape(dim3 grid_shape, const cudaDeviceProp* prop);
  int c_check_block_shape(dim3 block_shape, const cudaDeviceProp* prop);
  cudaError_t c_run_kernel(dim3 grid_shape, dim3 block_shape);
  const char* c_get_error_string(cudaError_t err);
}
