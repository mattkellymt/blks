#pragma once
#include <charconv>
#include <cstdlib>
#include <string_view>
#include <system_error>
#include <tuple>

template <typename T>
std::tuple<T, int> parse_num(const std::string_view arg) {
  auto arg_end = arg.data() + arg.size();
  T temp_val = 0;
  auto [read_end, ec] = std::from_chars(arg.data(), arg_end, temp_val);
  bool parse_error = ec != std::errc{};
  bool partial_read = read_end != arg_end;
  if (parse_error || partial_read) {
    return {T{}, EXIT_FAILURE};
  }
  return {temp_val, EXIT_SUCCESS};
}
