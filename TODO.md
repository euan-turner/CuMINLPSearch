# Implementation Plan


## Interval Operations and Kernels

- [x] Interval struct and intrinstic operations
- [x] Test
- [x] Hard-code rosenbrock kernel (pointwise and interval)
- [x] Test
- [x] Thread sub-interval selection on GPU
- [x] Interval sampling and partitioning (10 variable at a time, 4 regions per variable, so 4^10 subregions)
      Kernel should be able to use device functions to evaluate the function.

## Optimisation Loop

- [ ] Replace explicitregion with interval
- [ ] Reduction of GUB from sampled points
- [ ] Filtering regions from GUB
- [ ] Iterative region selection

## Quadratics

- [ ] Hardcode an easy QP
- [ ] Run full optimisation loop

## Objective Representation

- [ ] Quadratic objectives as a sparse triple
- [ ] Kernel evaluating sparse triple
- [ ] Write-up of potential future work (reduce number of parallel intervals to increase threads per interval, depending on nnz(Q), size of parent region, number of variables, etc.) if work/thread is too high.


## Constraints

- [ ] Represent linear and quadratic constraints as an objective to be evaluated
- [ ] Rule out regions
- [ ] Backprop constraint to restrict domain?

## Integer Constraints

- [ ] Add support for integer variables as in PLAN.md
- [ ] Use examples from MINLPlib

## Parsing Programs