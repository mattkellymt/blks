#include <cstdint>
#include <string_view>
#include "lib_cuda.cuh"
#include "threads_run.cuh"

extern "C" {
  pair parse_u32(const char* str) {
    auto [val, ec] = parse_num<uint32_t>(std::string_view(str));
    return pair{.u32 = val, .ec = ec};
  }

  pair parse_i32(const char* str) {
    auto [val, ec] = parse_num<int32_t>(std::string_view(str));
    return pair{.i32 = val, .ec = ec};
  }

  pair parse_f32(const char* str) {
    auto [val, ec] = parse_num<float>(std::string_view(str));
    return pair{.f32 = val, .ec = ec};
  }

  int launch_threads(uint32_t grid_size, uint32_t block_size) {
    return run_threads(grid_size, block_size);
  }
}
