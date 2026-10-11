#pragma once
#include <string>
#include "threads_homogeneous.cuh"

__host__
void welcome() {
  std::string msg = "This program will launch a GPU kernel and print its thread number";
  print(msg.c_str());
}
