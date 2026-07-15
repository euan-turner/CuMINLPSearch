# Implementation Plan


## Interval Operations and Kernels

- [x] Interval struct and intrinstic operations
- [x] Test
- [ ] Hard-code rosenbrock kernel
- [ ] Test
- [ ] Interval sampling and partitioning (10 variable at a time, 4 regions per variable, so 4^10 subregions)

## Optimisation Loop

- [ ] Replace explicitregion with interval
- [ ] Reduction of GUB from sampled points
- [ ] Filtering regions from GUB
- [ ] Iterative region selection

## Objective Representation

- [ ] Quadratic constraints and objectives as a sparse triple
- [ ] Kernel evaluating sparse triple
- [ ] Write-up of potential future work (reduce number of parallel intervals to increase threads per interval, depending on nnz(Q), size of parent region, number of variables, etc.) if work/thread is too high.
