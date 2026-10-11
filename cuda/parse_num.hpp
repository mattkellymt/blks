#pragma once
#include <charconv>
#include <cstdlib>
#include <string_view>
#include <system_error>
#include <tuple>

template <typename T>
std::tuple<T, int> parse_num(const std::string_view arg) {
  auto arg_end = arg.data() + arg.size();
  T val = 0;
  auto [read_end, read_err] = std::from_chars(arg.data(), arg_end, val);
  bool parse_err = read_err != std::errc{};
  bool partial_read = read_end != arg_end;
  if (parse_err || partial_read) {
    return {T{}, EXIT_FAILURE};
  }
  return {val, EXIT_SUCCESS};
}
