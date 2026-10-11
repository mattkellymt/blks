#pragma once
#include <cstdio>

__host__ __device__
void print(const char* msg) {
  printf("%s\n", msg);
}
