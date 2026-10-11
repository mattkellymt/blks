#pragma once
#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "get_prop.cuh"
#include "set_block_shape.cuh"
#include "set_grid_shape.cuh"
#include "threads_host.cuh"
#include "threads_kernel.cuh"

inline int run_threads(uint32_t grid_size, uint32_t block_size) {
  constexpr uint32_t device_id = 0;

  auto set_device_err = cudaSetDevice(device_id);
  if (set_device_err != cudaSuccess) {
    std::cerr << "Error setting device" << std::endl;
    return EXIT_FAILURE;
  }

  auto [prop, prop_err] = get_prop(device_id);
  if (prop_err != EXIT_SUCCESS) {
    std::cerr << "Error getting device properties" << std::endl;
    return EXIT_FAILURE;
  }

  auto grid_shape = dim3(grid_size, 1, 1);
  auto grid_shape_err = check_grid_shape(grid_shape, prop);
  if (grid_shape_err != EXIT_SUCCESS) {
    std::cerr << "Invalid grid shape" << std::endl;
    return EXIT_FAILURE;
  }

  auto block_shape = dim3(block_size, 1, 1);
  auto block_shape_err = check_block_shape(block_shape, prop);
  if (block_shape_err != EXIT_SUCCESS) {
    std::cerr << "Invalid block shape" << std::endl;
    return EXIT_FAILURE;
  }

  welcome();
  kernel<<<grid_shape, block_shape>>>();

  auto kernel_err = cudaGetLastError();
  if (kernel_err != cudaSuccess) {
    std::string msg = std::format("Kernel launch failed: {}", cudaGetErrorString(kernel_err));
    std::cerr << msg << std::endl;
    return EXIT_FAILURE;
  }

  auto sync_err = cudaDeviceSynchronize();
  if (sync_err != cudaSuccess) {
    std::string msg = std::format("cudaDeviceSynchronize failed: {}", cudaGetErrorString(sync_err));
    std::cerr << msg << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
