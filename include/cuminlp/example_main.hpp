#pragma once

#include <exception>
#include <iostream>

#include "cuminlp/errors.hpp"

namespace cuminlp::examples
{

/**
 * @brief Run an example's body, reporting a cuminlp::error as a diagnostic
 *        instead of an uncaught-exception abort.
 *
 * The hand-written drivers in source/ pick their own search shape at compile
 * time, so an over-budget shape is a real possibility for anyone editing one
 * (see nvs09_problem.hpp, whose CYCLE_SIZE had to come down from 10 to 7).
 * Letting that escape main() gets the message printed by std::terminate,
 * buried under a core dump; ResourceExhausted's report is several lines of
 * arithmetic naming the parameter to lower, and is worth reading intact.
 *
 * Exit codes match source/gams/solve.cu: 2 for a bad configuration, 3 for
 * out of device memory, 1 for anything else.
 */
template<typename F>
int guarded(F&& body)
{
  try {
    body();
    return 0;
  } catch (const ResourceExhausted& e) {
    std::cerr << "\nout of device memory:\n" << e.what() << '\n';
    return 3;
  } catch (const InvalidConfiguration& e) {
    std::cerr << "\nconfiguration error: " << e.what() << '\n';
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "\nerror: " << e.what() << '\n';
    return 1;
  }
}

}  // namespace cuminlp::examples
