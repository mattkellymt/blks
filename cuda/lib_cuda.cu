#include <cstddef>
#include <cstdint>
#include <string_view>
#include <cuda_runtime.h>
#include "parse_num.hpp"
#include "run_kernel.cuh"
#include "set_block_shape.cuh"
#include "set_grid_shape.cuh"
#include "threads_host.cuh"

extern "C" {
  cudaError_t c_set_device(uint32_t device_id) {
    return cudaSetDevice(device_id);
  }

  void c_welcome() {
    welcome();
  }

  size_t c_prop_size() {
    return sizeof(cudaDeviceProp);
  }

  cudaError_t c_get_prop(cudaDeviceProp* prop, uint32_t device_id) {
    return cudaGetDeviceProperties(prop, device_id);
  }

  int c_parse_u32(uint32_t* val, const char* str) {
    return parse_num(*val, std::string_view(str));
  }

  int c_check_grid_shape(dim3 grid_shape, const cudaDeviceProp* prop) {
    return check_grid_shape(grid_shape, *prop);
  }

  int c_check_block_shape(dim3 block_shape, const cudaDeviceProp* prop) {
    return check_block_shape(block_shape, *prop);
  }

  cudaError_t c_run_kernel(dim3 grid_shape, dim3 block_shape) {
    return run_kernel(grid_shape, block_shape);
  }

  const char* c_get_error_string(cudaError_t err) {
    return cudaGetErrorString(err);
  }
}
