#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <span>
#include <string>
#include "parse_num.hpp"
#include "threads_run.cuh"

int main(int argc, char* argv[]) {
  std::span<char*> args(argv, argc);

  if (args.size() != 3) {
    std::string msg = std::format("Usage: {} <grid size> <block size>", args[0]);
    std::cerr << msg << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto [grid_size, grid_size_err] = parse_num<uint32_t>(args[1]);
  if (grid_size_err != EXIT_SUCCESS) {
    std::cerr << "Invalid grid size" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto [block_size, block_size_err] = parse_num<uint32_t>(args[2]);
  if (block_size_err != EXIT_SUCCESS) {
    std::cerr << "Invalid block size" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  return run_threads(grid_size, block_size);
}
