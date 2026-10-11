#pragma once
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
}
