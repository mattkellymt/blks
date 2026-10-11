#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <span>
#include <string>
#include <cuda_runtime.h>
#include "parse_num.hpp"
#include "run_kernel.cuh"
#include "set_block_shape.cuh"
#include "set_grid_shape.cuh"
#include "threads_host.cuh"

int main(int argc, char* argv[]) {
  std::span<char*> args(argv, argc);
  constexpr uint32_t device_id = 0;

  if (args.size() != 3) {
    std::string msg = std::format("Usage: {} <grid size> <block size>", args[0]);
    std::cerr << msg << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto set_device_err = cudaSetDevice(device_id);
  if (set_device_err != cudaSuccess) {
    std::string msg = std::format("Error setting device: {}", cudaGetErrorString(set_device_err));
    std::cerr << msg << std::endl;
    std::exit(EXIT_FAILURE);
  }

  cudaDeviceProp prop{};
  auto prop_err = cudaGetDeviceProperties(&prop, device_id);
  if (prop_err != cudaSuccess) {
    std::string msg = std::format("Error getting device properties: {}", cudaGetErrorString(prop_err));
    std::cerr << msg << std::endl;
    std::exit(EXIT_FAILURE);
  }

  uint32_t grid_size{};
  auto grid_size_err = parse_num(grid_size, args[1]);
  if (grid_size_err != EXIT_SUCCESS) {
    std::cerr << "Invalid grid size" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto grid_shape = dim3{grid_size, 1, 1};
  auto grid_shape_err = check_grid_shape(grid_shape, prop);
  if (grid_shape_err != EXIT_SUCCESS) {
    std::cerr << "Invalid grid shape" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  uint32_t block_size{};
  auto block_size_err = parse_num(block_size, args[2]);
  if (block_size_err != EXIT_SUCCESS) {
    std::cerr << "Invalid block size" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto block_shape = dim3{block_size, 1, 1};
  auto block_shape_err = check_block_shape(block_shape, prop);
  if (block_shape_err != EXIT_SUCCESS) {
    std::cerr << "Invalid block shape" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  welcome();
  auto kernel_err = run_kernel(grid_shape, block_shape);
  if (kernel_err != cudaSuccess) {
    std::string msg = std::format("Kernel failed: {}", cudaGetErrorString(kernel_err));
    std::cerr << msg << std::endl;
    std::exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
