#pragma once
#include <iostream>
#include <string>

__host__
inline void welcome() {
  std::string msg = "This program will launch a GPU kernel and print its thread number";
  std::cout << msg << std::endl;
}
