#pragma once
#include <cstdint>
#include "parse_num.hpp"

struct pair {
  union {
    uint32_t u32;
    int32_t i32;
    float f32;
  };
  int ec;
};

extern "C" {
  pair parse_u32(const char* str);
  pair parse_i32(const char* str);
  pair parse_f32(const char* str);
  int launch_threads(uint32_t grid_size, uint32_t block_size);
}
