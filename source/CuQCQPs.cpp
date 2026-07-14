#include <string>

#include "CuQCQPs/CuQCQPs.hpp"

library_stub::library_stub()
    : m_name {"CuQCQPs"}
{
}

auto library_stub::name() const -> char const*
{
  return m_name.c_str();
}
