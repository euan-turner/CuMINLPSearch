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
 * buried under a core dump.
 *
 * What arrives here is the one-line summary a backend::OverBudgetError
 * carries, not the costed report: writing that needs a ProblemProfile, which
 * this wrapper has no way to hold (design/MODULE_REFACTOR.md §5.6). In
 * practice nothing reaches it -- SearchDriver::solve catches an over-budget
 * build inside its own loop and prints the full explanation there -- so this
 * is the backstop for an allocation failure outside a solve, where the
 * summary is all there is to say anyway.
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
