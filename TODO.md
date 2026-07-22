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

- [x] Replace explicitregion with interval
- [x] Reduction of GUB from sampled points
- [x] Filtering regions from GUB
- [x] Iterative region selection
- [ ] Interval hull over all viable intervals at termination

## DAG Representation

See design/DAG.md for the full design; MC++ was evaluated and rejected (Section 13) in favour of a dependency-free `DAGNode` IR.

- [x] `DAGNode`/`ExprDAG`/`Expr` operator-overloaded frontend (Frontend A, design/DAG.md Section 3.3), shared global `ExprDAG` across all objectives/constraints in a `Problem` (Section 3.1)
- [x] Extend `Op` beyond `{Var, Const, Add, Sub, Mul, Div, Sqr, Neg}` with `Exp, Log, Sqrt, Sin, Cos, Tanh, IPow, Abs, Min, Max` (all already available as `cu::interval` device ops)
- [ ] **Section 4: MAiNGO per-operation CUDA graph reproduction** (baseline before clustering/JIT): one kernel per DAG node, and one kernel for the domain partitioning. Each node has `Interval<T>` scratch buffers, with explicit `cudaGraphAddKernelNode` per `DAGNode` walking a function's root-traversal order, absorbing-zero `feasible[]` + in-graph CUB GUB reduction (Section 10), measured on 4090/5090
- [ ] Sections 5-7 (clustering policy, codegen, NVRTC JIT) — only after Section 4's measured baseline exists
- [ ] `.nl` frontend via AMPL MP (Frontend B, design/DAG.md Section 3.4), for benchmark ingestion

Extensions:

- [ ] Hash-consing at emit time (dedup on `(op, operands, payload)`), scoped within a function's traversal for now
- [ ] Buffer lifetime/reuse for Section 4's per-node scratch buffers (design/DAG.md Section 4.3.1): liveness-based free-list slot allocation instead of one buffer per node forever, plus the required explicit WAR dependency edges on reused slots. Deferred until the naive version actually hits memory limits (already close at `DIMS=100` Rosenbrock: ~1000 nodes × 16 MB/buffer at 10^6 regions ≈ 16 GB)

## Constraints

- [ ] Restrict GUB updates to feasible points
- [ ] Rule out infeasible regions

## Integer Constraints

- [ ] Add support for integer variables via domain partitioning (design/DAG.md Section 3.5: integer variables are not an expression kind, handled by partitioning, not the DAG)
- [ ] Use examples from MINLPlib
