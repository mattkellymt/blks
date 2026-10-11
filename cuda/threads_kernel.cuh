#pragma once
#include <cstdio>
#include "threads_device.cuh"
#include "threads_homogeneous.cuh"

__global__
void kernel() {
  uint32_t thread = get_thread();
  printf("Hello from thread %u!\n", thread);
}
