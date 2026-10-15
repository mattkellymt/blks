#pragma once
#include <cstdint>

__device__
inline int32_t get_thread() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}
