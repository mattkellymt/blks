#pragma once
#include <cstdlib>
#include <cstdint>
#include <string_view>
#include <tuple>
#include <cuda_runtime.h>
#include "parse_num.hpp"

inline int check_block_shape(uint32_t x, uint32_t y, uint32_t z, const cudaDeviceProp& prop) {
  if (x == 0 || y == 0 || z == 0) {
    return EXIT_FAILURE;
  }
  auto threads_per_block = x * y * z;
  if (threads_per_block > prop.maxThreadsPerBlock) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

inline std::tuple<dim3, int> set_block_shape(uint32_t x, uint32_t y, uint32_t z, const cudaDeviceProp& prop) {
  if (check_block_shape(x, y, z, prop) != EXIT_SUCCESS) {
    return {dim3{}, EXIT_FAILURE};
  }
  return {dim3(x, y, z), EXIT_SUCCESS};
}

inline std::tuple<dim3, int> set_block_shape(std::string_view xstr, std::string_view ystr, std::string_view zstr, const cudaDeviceProp& prop) {
  auto [x, err1] = parse_num<uint32_t>(xstr);
  auto [y, err2] = parse_num<uint32_t>(ystr);
  auto [z, err3] = parse_num<uint32_t>(zstr);
  if (err1 != EXIT_SUCCESS || err2 != EXIT_SUCCESS || err3 != EXIT_SUCCESS) {
    return {dim3{}, EXIT_FAILURE};
  }
  return set_block_shape(x, y, z, prop);
}
