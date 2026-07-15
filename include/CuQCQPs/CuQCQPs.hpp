#pragma once

#include <string>

#include "CuQCQPs/region.hpp"

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
  driver(uint32_t iter_limit = 1000000);

  auto solve() -> double;

private:
  double GUB_;
  double GLB_;
  uint32_t iter_limit_;
  uint32_t iter_idx_;
  uint16_t cycl_idx_;
};
