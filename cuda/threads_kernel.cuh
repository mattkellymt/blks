#pragma once
#include <cstdint>
#include <cstdio>
#include "threads_device.cuh"

__global__
static void kernel() {
  uint32_t thread = get_thread();
  printf("Hello from thread %u!\n", thread);
}
