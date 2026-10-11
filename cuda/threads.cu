#include <cstdlib>
#include <format>
#include <iostream>
#include <span>
#include <string>
#include <utility>
#include "get_prop.cuh"
#include "set_block_shape.cuh"
#include "set_grid_shape.cuh"
#include "threads_host.cuh"
#include "threads_kernel.cuh"

int main(int argc, char* argv[]) {
  std::span<char*> args(argv, argc);
  constexpr uint32_t device_id = 0;

  if (args.size() != 3) {
    std::string msg = std::format("Usage: {} <grid size> <block size>", args[0]);
    std::cerr << msg << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto err1 = cudaSetDevice(device_id);
  if (err1 != cudaSuccess) {
    std::cerr << "Error setting device" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto [prop, err2] = get_prop(device_id);
  if (err2 != EXIT_SUCCESS) {
    std::cerr << "Error getting device properties" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto [grid_shape, err3] = set_grid_shape(args[1], "1", "1", prop);
  if (err3 != EXIT_SUCCESS) {
    std::cerr << "Invalid grid shape setup" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto [block_shape, err4] = set_block_shape(args[2], "1", "1", prop);
  if (err4 != EXIT_SUCCESS) {
    std::cerr << "Invalid block shape setup" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  welcome();
  kernel<<<grid_shape, block_shape>>>();

  auto err5 = cudaGetLastError();
  if (err5 != cudaSuccess) {
    std::string msg = std::format("Kernel launch failed: {}", cudaGetErrorString(err5));
    std::cerr << msg << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto err6 = cudaDeviceSynchronize();
  if (err6 != cudaSuccess) {
    std::string msg = std::format("cudaDeviceSynchronize failed: {}", cudaGetErrorString(err6));
    std::cerr << msg << std::endl;
    std::exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}
