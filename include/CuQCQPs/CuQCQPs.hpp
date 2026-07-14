#pragma once

#include <string>

class library_stub
{
public:
  library_stub();

  auto name() const -> char const*;

private:
  std::string m_name;
};
