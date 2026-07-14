#include <string>

#include "CuQCQPs/CuQCQPs.hpp"

auto main() -> int
{
  auto const stub = library_stub {};

  return std::string("CuQCQPs") == stub.name() ? 0 : 1;
}
