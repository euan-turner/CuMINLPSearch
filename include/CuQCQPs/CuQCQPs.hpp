#pragma once

#include <cstdint>
#include <string>

namespace cuqcqps
{

class library_stub
{
public:
  library_stub();

  auto name() const -> char const*;

private:
  std::string m_name;
};

class driver
{
public:
  driver(uint32_t iter_limit = 1000000, double tolerance = 1e-9);

  auto solve() -> double;

private:
  double GUB_;
  double GLB_;
  double tolerance_;
  uint32_t iter_limit_;
  uint32_t iter_idx_;
};

}  // namespace cuqcqps
