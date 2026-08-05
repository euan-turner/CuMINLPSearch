#pragma once

#include <cstddef>
#include <cstdio>
#include <iterator>
#include <limits>
#include <string>

// Byte and count formatting for diagnostics. Split out of cuda_utils.cuh so
// that host-only code can use it without <cuda_runtime.h>: the over-budget
// report these two format is now written by config::explain_over_budget
// (design/MODULE_REFACTOR.md §5.6), which is compiled by a plain host
// compiler. cuda_utils.cuh includes this header, so its existing callers see
// the same names in the same namespace.
namespace cuminlp::detail
{

// Byte counts in diagnostics, e.g. "221.2 GiB". Plain bytes below 1 KiB.
inline std::string format_bytes(std::size_t bytes)
{
  if (bytes == std::numeric_limits<std::size_t>::max()) {
    return "more than 2^64 bytes (the size computation saturated)";
  }
  static const char* const units[] = {"B", "KiB", "MiB", "GiB", "TiB", "PiB"};
  auto value = static_cast<double>(bytes);
  std::size_t unit = 0;
  while (value >= 1024.0 && unit + 1 < std::size(units)) {
    value /= 1024.0;
    ++unit;
  }
  char buf[64];
  std::snprintf(
      buf, sizeof(buf), unit == 0 ? "%.0f %s" : "%.1f %s", value, units[unit]);
  return buf;
}

// Thousands separators, so a nine-digit region count is readable at a glance.
inline std::string format_count(std::size_t n)
{
  std::string s = std::to_string(n);
  for (std::size_t pos = s.size() > 3 ? s.size() - 3 : 0; pos > 0; pos -= 3) {
    s.insert(pos, ",");
    if (pos < 3) {
      break;
    }
  }
  return s;
}

}  // namespace cuminlp::detail
