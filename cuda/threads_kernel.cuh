#pragma once
#include <cstdint>
#include <cstdio>
#include "threads_device.cuh"

__global__
static void kernel() {
  int32_t thread = get_thread();
  printf("Hello from thread %d!\n", thread);
}
