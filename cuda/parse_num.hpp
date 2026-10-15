#pragma once
#include <cerrno>
#include <charconv>
#include <string_view>
#include <system_error>

template <typename T>
int parse_num(T& val, const std::string_view arg) {
  auto arg_end = arg.data() + arg.size();
  auto [read_end, read_err] = std::from_chars(arg.data(), arg_end, val);
  bool parse_err = read_err != std::errc{};
  bool partial_read = read_end != arg_end;
  if (parse_err) {
    return static_cast<int>(read_err);
  }
  if (partial_read) {
    return EINVAL;
  }
  return 0;
}
