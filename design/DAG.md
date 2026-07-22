# Design Doc: JIT-Compiled, Cluster-per-Kernel Interval Bounding Pipeline

**Status**: Draft for review
**Scope**: Runtime representation and GPU-parallel evaluation of objective and constraint interval bounds for a domain-partitioning solver. This is the **evaluator only**. The search/branch-and-prune driver (region list, refinement, termination, GUB policy) is out of scope and owned separately.
**Owner**: (you)
**Last updated**: 21 July 2026

---

## 1. Purpose and scope

This document specifies how the solver ingests an optimisation problem (one objective, many constraints), compiles it to GPU code at runtime, and evaluates interval bounds over a large set of subregions. The unit of work is: one parent domain in, per-region `feasible[]` and `obj_lb[]` out, `obj_ub` reduced on-device to a scalar. What the driver does with those outputs is not specified here.

**In scope**: expression frontends (programmatic and `.nl`), the internal DAG IR, clustering policy, code generation, NVRTC compilation, module loading, CUDA graph construction, per-launch execution, feasibility and bound reduction, precision handling.

**Out of scope**: the search driver (region list management, refinement/subdivision strategy, termination, incumbent handling), survivor compaction and early-exit pruning, subexpression sharing across clusters, warp- or block-per-region evaluation (Method F), CPU-side interval reduction logic. Deferred items are in Section 13.

**Lean-stack decision (boundary)**: the evaluator depends on the CUDA toolkit (NVRTC, driver API, CUB/CCCL) and AMPL MP (for the `.nl` frontend only). It deliberately does **not** depend on MC++ or any relaxation/AD library. No automatic differentiation, no McCormick/Taylor relaxations. Rationale and the paths that would reverse this are in Section 13.

### 1.1 Key terms

- **Subregion**: one box in the partitioned search domain, evaluated by one thread. Target scale is ~10^6 per launch.
- **DAG**: directed acyclic graph of primitive operations representing a function (leaves are variables and constants, inner nodes are operations).
- **`DAGNode`**: our internal DAG node type (id, op, operands, payload), defined by this project (not an external library). It is the **stable internal IR**: every frontend produces `DAGNode` lists and everything downstream consumes them, so frontends are swappable.
- **Frontend**: a producer of the `DAGNode` IR. Two exist (Section 3): a programmatic builder and an `.nl` parser.
- **Cluster**: a group of constraints (or the objective) compiled into a single kernel.
- **Op-budget** `B`: target cumulative operation count per cluster, the single tuning knob.
- **NVRTC**: NVIDIA Runtime Compilation library. Compiles CUDA C++ source strings to PTX or CUBIN in-process.
- **CUBIN**: compiled device binary for a specific architecture. **PTX**: virtual ISA, JIT-finalised by the driver at load time.
- **CUDA graph**: a captured DAG of GPU work (kernels, copies, memsets) instantiated once and replayed cheaply.
- **Occupancy**: ratio of resident warps per SM to the hardware maximum. Governs latency hiding, especially critical for FP64 on consumer cards.

### 1.2 Target hardware and its consequences (verified July 2026)

| GPU | Arch | Compute cap / `-arch` | FP64:FP32 | L2 |
|---|---|---|---|---|
| RTX 4090 | Ada | 8.9 / `sm_89` | ~1:64 | 72 MB |
| RTX 5090 | Blackwell (GB202) | 12.0 / `sm_120` (needs CUDA 12.8+) | ~1:64 (1.64 vs 105 TFLOPS) | 96 MB |
| H100 / B200 (future) | Hopper / Blackwell DC | 9.0 / 10.0 | ~1:2 | 50-60 MB |

**Consequence**: on current consumer targets, FP64 interval work is compute-bound on scarce FP64 units. Occupancy (resident warps to hide FP64 latency) matters more than raw memory bandwidth. This is why the clustering knob is calibrated against occupancy, not core count (Section 5). When the deployment moves to FP32 or to datacenter FP64 (1:2), the optimum shifts coarser, so the knob stays a runtime parameter.

---

## 2. Architecture overview

```
                          ONCE PER PROBLEM
  Frontend A: programmatic  ┐
  (operator overloading)    ├──▶  DAGNode DAG  ──┬──▶ Cluster ──▶ Codegen ──▶ NVRTC ──▶ Module load
  Frontend B: .nl parser    ┘     (stable IR)      │    greedy B    SSA+tmpl   →CUBIN    +launch cfg
  (AMPL MP mp::Problem)                            │                                          │
                                                    │                                          ▼
                                                    │                                   ┌──────────────┐
                                                    │                                   │ Graph build  │
                                                    │                                   │ instantiate  │
                                                    │                                   └──────────────┘
                                                    │                                          │
                                                    │  (calibration only,              REPLAY LOOP
                                                    │   Section 4)                     (driver-owned, external)
                                                    ▼                                          ▼
                                          ┌────────────────────┐            ┌───────────────────────────────────────┐
                                          │ Per-op precompiled │            │ set domain → cudaGraphLaunch → D2H →   │
                                          │ kernels + CUDA     │            │ [driver: list logic] → new domain      │
                                          │ graph (MAiNGO repro)│           └───────────────────────────────────────┘
                                          └────────────────────┘
```

The **`DAGNode` DAG is the stable internal boundary**. Everything downstream (cluster, codegen, graph) consumes only the DAG and is frontend-agnostic. Two frontends produce it and both are permanent (Section 3): a programmatic builder (used now, no external deps) and the `.nl` parser (added later, for benchmarks). The DAG also feeds a separate calibration path (Section 4): a direct reproduction of paper 1's per-operation CUDA Graph method, using precompiled generic kernels rather than clustering and JIT. That path shares the graph-construction and reduction machinery with the main pipeline but bypasses clustering/codegen/NVRTC entirely, and exists to measure a real baseline before the JIT pipeline is built. The left half runs once per problem; the replay loop runs per launch and touches no compilation. The loop is driven by the out-of-scope search driver; this doc specifies only the per-launch evaluation inside it.

---

## 3. Stage 1: Frontend (produce the DAG)

**Goal**: produce, per objective and per constraint, a `DAGNode` DAG annotated with the metadata the clusterer needs. The DAG is the stable IR; two frontends produce it and everything downstream is frontend-agnostic.

### 3.1 `DAGNode` DAG design (the shared IR)

**Definition.** All objectives and constraints in a `Problem` share one flat, topologically ordered `DAGNode` list (`ExprDAG`), not one list per function. A function (objective or constraint) is identified by its root node id; the function's own topologically ordered subset is recovered on demand by traversing backward from the root through `in` (every operand id is `< id`, so the traversal itself yields a valid topological order — no separate per-function id space or storage is needed). This subsumes the per-function-list design below and is what codegen/clustering walk over per function.

```cpp
struct DAGNode {
  uint32_t id;                  // position in the list; also its SSA temp name "t{id}"
  Op op;                        // Var, Const, Add, Sub, Mul, Div, Neg, Exp, Log, Sqrt,
                                 // Sqr, Sin, Cos, Tanh, IPow, Abs, Min, Max, ...
  std::vector<uint32_t> in;     // operand ids; every id in `in` is strictly < this id
  double constant = 0.0;        // payload for Op::Const
  uint32_t var_index = 0;       // payload for Op::Var
  int int_exp = 0;              // payload for Op::IPow

  uint32_t op_count;            // subtree size (this node + all ancestors' contributions),
                                 // used by clustering (Section 5) to budget cluster size
  uint32_t live_set_estimate;   // peak simultaneously-live interval slots in this subtree
                                 // (~ Strahler number); used to bound register pressure
};

struct ExprDAG {
  std::vector<DAGNode> nodes;  // topologically ordered across every objective and constraint in the Problem
};

struct ConstraintRef {
  std::size_t root_id;  // this constraint's root node id in the shared ExprDAG
  Cmp cmp;               // constraint relation (LE, EQ, ...)
  double rhs;
};

// A Problem holds one ExprDAG, one objective root id, and a list of ConstraintRefs.
// A "function" (the objective, or one constraint) is not its own list: it is
// identified purely by a root node id into the shared ExprDAG.nodes.
```

**Invariants that make this a usable IR, not just a data dump:**

- **Strict topological order, encoded in the id itself.** `id` is the list position and every operand id is smaller. This is what lets codegen (Section 6) emit straight-line SSA by a single linear pass with zero scheduling logic, and what lets Frontend B (Section 3.4) get correct order for free from post-order visitor traversal.
- **Immutable once built.** An `ExprDAG` is a value produced once by a frontend and consumed by clustering and codegen; nothing downstream mutates it. This is what makes clustering "just" a partition of a fixed node list rather than a graph-rewrite problem.
- **Op-count and live-set are computed by the frontend, not derived later.** Both are pure functions of the subtree structure, so it is cheapest to compute them once during construction (a post-order fold) rather than re-walking the DAG in the clusterer.
- **Scoped globally, per-function view by traversal (revised decision, was per-function).** All objectives and constraints share one `ExprDAG.nodes` list rather than each getting an independent list. A function's own topologically ordered node set is not stored separately: it is recovered on demand by walking backward from its root id through `in` (every operand id is `< id` in the *global* list too, so the walk itself yields a valid topological order, no separate id space needed). This was chosen over the original per-function-list design because it gets cross-constraint hash-consing/CSE for free later (Section 13's "global cross-constraint DAG table" is this same structure, not a separate thing to build) while costing nothing today: clustering and codegen still only ever look at one function's traversal-derived subset at a time. Consequence: `nodes.back()` is not a function's result any more (Section 4.3, Section 9); a function's terminal node is its stored root id.
- **No algebraic simplification pass.** `DAGNode` records exactly what the frontend parsed (or what the programmatic builder wrote); it does not fold constants, cancel terms, or normalise commutative operands. Simplification is a legitimate future addition but is out of scope: it would change op-counts (hence clustering) and risks altering rounding behaviour if done carelessly on interval-valued code. Left as a possible optimisation pass between frontend and clustering, not part of the IR's contract.

### 3.2 Why `DAGNode`, not an existing DAG representation

Given two existing candidates were already on the table (MC++'s `FFGraph`, and AMPL MP's own `mp::Expr` tree), it is worth stating explicitly why neither replaces this small custom type:

- **The IR must be frontend-agnostic, and adopting either candidate's type breaks that.** Two independent producers feed the same downstream pipeline (Section 3.3): a programmatic builder and the `.nl` parser. If the IR were MP's `mp::Expr`, the programmatic builder would have to construct MP objects to use the same pipeline, i.e. depend on MP even when no `.nl` file is involved. If the IR were MC++'s `FFVar`, both frontends would depend on MC++. A purpose-built, dependency-free type is what lets each frontend depend on nothing but the IR itself.
- **Neither candidate carries the annotations codegen actually needs.** Clustering needs op-count and a live-set estimate per node (Section 5); codegen needs a stable topological id to emit SSA in one linear pass. Neither `mp::Expr` nor MC++'s `FFVar`/`FFGraph` carries these, because neither was designed for GPU kernel fusion; both would need to be wrapped in a side-table keyed by node identity, which is extra bookkeeping bought for no benefit over just storing the fields on the node.
- **MC++ specifically was evaluated and rejected already** (Section 13) for reasons beyond the IR question, its AD and relaxation machinery are unneeded for the current natural-interval-extension direction, it is host-only (no CUDA), and it is a small, lightly maintained research dependency. Using its DAG type alone, without the AD/relaxations that are its actual value-add, would import the dependency cost for none of the benefit.
- **General symbolic/computation-graph IRs are the wrong grain.** Libraries built for computer algebra (simplification, multiple output backends) carry surface area (rewrite rules, multi-backend lowering) that is dead weight when the only consumer is "emit one line of CUDA per node." Tensor-graph IRs (ONNX, XLA HLO, TVM Relay) operate at whole-tensor granularity and have no native notion of interval arithmetic or directed rounding; adopting one would still require a custom lowering pass from their node types to CUDA source, which is the same amount of work `DAGNode`'s codegen already does, without the dependency.
- **The type is intentionally minimal.** A `DAGNode` is close to POD (plain old data): an id, an op tag, a small operand list, and a couple of scalar payloads. That minimality is what makes hash-consing (equality on `(op, operands, payload)`), serialization for the CUBIN cache key (Section 7.4), and unit-test fixture construction (Section 3.3, the programmatic builder) all trivial. A heavier borrowed type would carry fields and invariants irrelevant to this pipeline.

The honest cost of this choice: two frontends must independently implement the same lowering discipline (topological id assignment, op-count/live-set computation), rather than inheriting it from a shared library. That duplication is small (a few dozen lines per frontend) and is the price paid to keep the IR dependency-free and exactly shaped to what clustering and codegen need.

### 3.3 Frontend A: programmatic builder (primary, build now)

Operator overloading over a lightweight expression handle (the pattern many optimisation libraries use, e.g. MC++/MAiNGO, but with no such dependency here): `auto g = sqr(x[0]) + exp(x[1]);` records nodes into a `DAGNode` list.

- **Why now**: zero external dependencies, so clustering, codegen, NVRTC, and the graph can all be built and tested against hand-constructed DAGs before any parser exists. It is also the fixture generator for unit tests and the numerical oracle (Section 13).
- **Permanent, not a scaffold**: this is the API for embedding the solver in an application with a programmatically defined model. It stays even after the `.nl` parser lands.

### 3.4 Frontend B: `.nl` parser via AMPL MP (secondary, add later, for benchmarks)

Two implementation stages, same `DAGNode` output:

- **Stage B1 (documented approach)**: `mp::Problem` + `ExprVisitor`. Entry point is `mp::Problem`, which retains the nonlinear expression trees ("NL forest"). Traverse with the CRTP visitor `mp::ExprVisitor<Impl, Result>` and hash-cons into the `DAGNode` list, one pass per function. Low boilerplate; the read into `mp::Problem` is one call.
- **Stage B2 (later optimisation, "NL-Reader callback")**: a custom `mp::NLHandler` implementing the ProblemBuilder concept, emitting `DAGNode` directly during the single NL read, skipping the `mp::Problem` intermediate. More boilerplate (the ProblemBuilder concept is large), but zero intermediate storage. Only pursue if the two-pass B1 cost ever shows up in profiling; for benchmark ingestion it almost never will.

Details for B1:

- **Include and read**: `mp/nl.h` + `mp/problem.h`, then `mp::Problem p; ReadNLFile("inst.nl", p);`. (`mp/nl-reader.h` is the lower-level handler path used by B2.)
- **Do not use `mp::ProblemFlattener`.** It decomposes each nonlinear expression into an auxiliary variable and a defining constraint, shattering the fusion target. You want the intact tree.
- **Visitor surface** (verified against ampl/mp source): `VisitNumericConstant(NumericConstant)` with `.value()`, `VisitVariable(Reference)` with `.index()`, `VisitUnary(UnaryExpr)` with `.kind()`/`.arg()`, `VisitBinary(BinaryExpr)` with `.kind()`/`.lhs()`/`.rhs()`, `VisitSum(IteratedExpr)` iterated, plus `VisitCall`, `VisitIf`, `VisitPow`. `.kind()` returns `mp::expr::Kind`, distinguishing ADD/SUB/MUL/DIV/POW inside a binary node. Make `Result` the node id and hash-cons in each visit; post-order emission gives topological order for free.
- **No automatic differentiation.** MP replaced ASL for solvers that do not need AD (lean-stack decision, Section 1). Natural interval extension needs no derivatives. If the Mean Value Form is ever added, run a reverse-mode AD pass over `DAGNode` yourself (Section 13); MP will not supply gradients.

### 3.5 Three adaptor responsibilities the `.nl` frontend must handle

1. **Fold the linear part.** MP stores each objective/constraint as a separate sparse linear part (coeff, var) plus a nonlinear expression tree. Emit the linear terms as a `Var`/`Const`/`Mul`/`Add` chain and add the nonlinear root, or you silently drop the linear terms.
2. **Two-sided ranges.** `.nl` constraints are `lb <= body <= ub` (equality is `lb == ub`, one-sided uses infinity). Carry `[lb, ub]` in the function record. Feasibility becomes "LHS interval intersects `[lb, ub]`": `lhs.lo <= ub && lhs.hi >= lb`, which subsumes LE/GE/EQ in one predicate (Section 10).
3. **Reject non-factorable kinds.** Map the smooth factorable kinds (ADD, SUB, MUL, DIV, POW, exp, log, sqrt, trig, hyperbolic) and the enclosable non-smooth ones (abs, min, max) to your `Op`. Hard-reject discontinuous/discrete/logical kinds (floor, ceil, round, if-then-else, CP constructs) so the benchmark harness skips unsupported instances cleanly rather than emitting wrong bounds. Integer variables are not an expression kind; they are handled by domain partitioning, so factorable MINLP instances are fine. Pull the authoritative `mp::expr::Kind` list from `mp/expr.h` when writing the map; default case rejects.

### 3.6 Data to confirm

- **MINLPLib `.nl` coverage** is sufficient for the target subset (user-confirmed). Batch-convert any gaps from GAMS/`.osil` via AMPL if needed.

**Docs**:
- MP components and NL reader: <https://mp.ampl.com/components.html>
- MP reference (Problem, ExprVisitor, expr::Kind): <https://mp.ampl.com/details.html>
- NL format report: <https://ampl.github.io/nlwrite.pdf>

---

## 4. Stage 2: Benchmark reproduction (MAiNGO per-operation CUDA graph replay)

**Goal**: build a direct reproduction of paper 1's (accelerating_bb / MAiNGO) "CUDA Graph" execution strategy, evaluated over a `DAGNode` DAG, to get a real measured baseline on 4090/5090 before investing further in clustering and JIT codegen (Sections 5–9).

### 4.1 Why this comes before clustering and JIT, not after

Two independent reasons, one of principle and one of engineering economy:

- **Paper 1's conclusion needs re-verification on our hardware.** Their finding that per-operation CUDA Graph beat a fused Single Kernel was measured on an RTX A1000 laptop GPU, an extremely FP64-starved part (roughly 1:64 FP32:FP64, consistent with the 4090/5090's own ratio, but a different SM count, clock, and memory subsystem). The clustering knob (Section 5) and the concurrency-vs-occupancy trade-off discussed there (Section 9.3) are calibrated by argument, not by measurement. Building the paper's own method first and running it on the actual target GPUs turns that argument into a number, and gives a concrete baseline every later cluster granularity can be measured against.
- **The plumbing is not throwaway.** Per-operation execution needs exactly the same execution scaffolding the full pipeline needs: an explicit CUDA graph, one node per unit of work, a feasibility flag with absorbing-zero semantics, an in-graph CUB reduction for the GUB. The only thing that differs between this benchmark and the target pipeline (Section 9) is what a graph node runs: here, one generic precompiled kernel per DAG *operation*; there, one JIT-compiled kernel per *cluster*. Building this first means the graph-construction work (Section 9) is validated and reused, not redone.

### 4.2 The key simplification: no NVRTC needed here

This is what makes the benchmark buildable before the JIT stage at all: the op set (`Op::Add, Mul, Exp, Sqr, ...`) is small and fixed at compile time, unlike an arbitrary fused cluster expression. So instead of generating and compiling CUDA source per problem, write **one generic, precompiled, templated device kernel per `Op` member**, ahead of time, as ordinary `.cu` code:

```cpp
// One of a small, fixed family of generic per-op kernels (precompiled, not JIT'd).
// Each reads its operand(s) from global per-node scratch buffers and writes its
// own result to its own buffer, reproducing the paper's node-per-kernel boundary.
template <class S>
__global__ void op_add_k(const Interval<S>* __restrict__ a,
                          const Interval<S>* __restrict__ b,
                          Interval<S>* __restrict__ out, int n_regions) {
  for (long long r = blockIdx.x * blockDim.x + threadIdx.x;
       r < n_regions; r += (long long)gridDim.x * blockDim.x)
    out[r] = a[r] + b[r];
}
// op_mul_k, op_exp_k, op_sqr_k, ... one per Op member, same shape.
// Leaf nodes (Op::Var, Op::Const) are filled by a single small kernel that
// evaluates partition_subregion once and scatters variable/constant values
// into their node buffers, reusing the same SPSD logic as the main pipeline.
```

No name-mangling, no `nvrtcCompileProgram`, no CUBIN cache. This is a deliberate contrast with Section 7 (JIT compilation): here the "compiler" is `nvcc` at build time, because fusion granularity is fixed at one-node-per-kernel and the operation vocabulary is closed.

### 4.3 Wiring the DAG onto graph nodes

Because `DAGNode.id` is already topologically ordered (Section 3.1), building the graph is a single linear pass with no scheduling logic:

- **Allocate one `Interval<S>` buffer of length `n_regions` per node id.** This is the memory-traffic cost the paper's method pays at every DAG edge, and reproducing it faithfully (not clustering anything) is the point of this benchmark.
- **For each `DAGNode` in order, add one kernel node** (`cudaGraphAddKernelNode`) bound to that op's precompiled kernel, reading its operands' buffers (already-added nodes, since ids are topological) and writing its own buffer. Dependencies are exactly the node's operand ids.
- **Terminal per-function nodes** (Section 3.1, each function's stored root id, not `nodes.back()` since the list is now shared) feed the same feasibility and GUB machinery as the main pipeline (Section 10): a two-sided range test for each constraint's LHS, and the objective's endpoints reduced by an in-graph CUB node. Reuse that machinery verbatim; do not build a second copy for the benchmark.
- **Explicit graph API** (consistent with the Section 9.2 decision), root memset for `feasible[]`, fan-in to the D2H copy. The topology is a straight-line chain per function with fan-out only where a node's output feeds multiple consumers (shared operands in the global `ExprDAG`, Section 3.1 — sharing can now occur across functions too, not just within one).

### 4.3.1 Buffer lifetime and reuse (pending, deferred until it bites)

**Status: not built yet.** The naive version of Section 4.3 allocates one `Interval<T>` buffer of length `n_regions` for every reachable node id and never frees any of them for the duration of the benchmark. That's a real near-term memory problem, not a hypothetical one: at the target scale (`n_regions ~ 10^6`), one buffer is 16 MB (`Interval<double>`), and the Rosenbrock instance already in this repo (`DIMS = 100`, `source/cuminlp.cu`) produces on the order of 1000 DAG nodes, i.e. ~16 GB of buffers for one problem before `feasible[]`, CUB temp storage, or anything else — most of a 4090's 24 GB. This needs addressing before Section 4 can run at its target scale on a non-trivial problem, not just for large future MINLPLib instances.

**Proposed mechanism** (host-side, one-time, at graph-build time — costs nothing per replay and doesn't touch the capture-time restrictions of Section 9.5, since all allocation still happens before capture):

1. For each reachable node, compute its **last consumer**: the max node id among everything that reads it — its downstream ops within a function, or the feasibility/GUB kernel if the node is a function's root. A node with fan-out (shared across constraints, or referenced twice within one expression) stays live until the *last* of its consumers, not the first.
2. Walk nodes in increasing id order with a free-list of buffer slots: allocate a slot for each node's output (reusing a freed slot if one is available), and return an operand's slot to the free list once its last consumer's graph node has been added. This mirrors the register-liveness idea already specified for codegen (Section 6.2), just applied to buffer slots instead of registers, and it's a one-time host-side bookkeeping pass, not a per-launch cost.
3. **Correctness requirement, not just an optimization detail**: reusing slot `S` for a new node's output requires an **explicit WAR dependency edge** from the new writer's graph node to every old consumer of `S`, not just the natural producer→consumer edge. The explicit graph API (Section 9.2) does not serialize unconnected nodes — two nodes with no edge between them may run concurrently — so without this edge the new write can race the old reads.

**Why this doesn't compromise Section 4's fidelity**: reuse changes *which physical address* a node writes to, not the number of kernel launches, dependency edges, or bytes moved per DAG edge. The wall-clock/occupancy/memory-traffic numbers Section 4.4 measures should be unaffected; only peak allocated footprint changes. Adding the WAR edges costs a few extra fan-in edges, which Section 9.3 already argues barely affects the measured concurrency at 10^6-region scale.

**Deferred for now**: build the naive one-buffer-per-node version first; add this pass once it actually blocks running the benchmark at target scale on a real problem (see TODO.md).

### 4.4 What to measure, and what it settles

Run this benchmark on both target GPUs (4090, 5090) over a handful of representative problems (a small one and a large one, in variable and constraint count), and record:

- **Wall-clock per graph replay**, to compare directly against the eventual clustered-JIT pipeline once Sections 5–9 exist.
- **Achieved occupancy per op-kernel** (expect it near 100%, since each kernel's register footprint is tiny, being a single operation). This is the empirical anchor for the occupancy-floor argument in Section 5.3.
- **Memory traffic**, ideally via the profiler's DRAM/L2 throughput counters, to put a real number on the "every DAG edge is a global-memory round trip" cost that motivated moving to clusters at all (Section 5's clustering rationale).

This is what turns Section 9.3's concurrency caveat from an argument into a measurement: if per-op kernels already saturate the device and the wall-clock time is dominated by memory traffic rather than compute stalls, that directly confirms coarser clustering is worth it on this hardware, rather than relying on paper 1's numbers from different silicon.

### 4.5 Boundary with the rest of the doc

This section is a **calibration benchmark, not a deployment path**. It shares the SPSD partition (EP4), the `Interval<S>` library (EP3), the feasibility/GUB reduction (Section 10), and the explicit-graph machinery (Section 9) with the target pipeline. It does **not** share Sections 5–7 (clustering, codegen, NVRTC) at all, by design, since its entire purpose is to measure the case those sections exist to improve on. Once Sections 5–9 are built, this benchmark remains as a permanent regression check: a new problem's clustered-JIT wall-clock should always be compared against its own per-op-kernel number from this stage.

---

## 5. Stage 3: Clustering policy (the one knob)

**Goal**: partition constraints into clusters so each compiled kernel keeps register pressure under an occupancy-driven cap, while minimising kernel count.

### 5.1 Policy: greedy bin-packing by op-count

**Machinery**:
- Fix an op-budget `B`. Optionally sort constraints by op-count descending (first-fit-decreasing) for better runtime balance.
- Pack sequentially: add constraints to the current cluster until the next would exceed `B`, then open a new cluster.
- Any single constraint with op-count `> B` becomes a **singleton kernel**. It cannot be split without retaining intermediates, which we have deliberately excluded. Accept its lower occupancy.
- The objective is packed like any other node (its RHS is the GUB).

**Concise contract**:
```
cluster_constraints(nodes, B) -> list[Cluster]
  """Greedy first-fit-decreasing bin-packing of function nodes by op-count.

  Oversized nodes (op_count > B) become singleton clusters.
  Returns clusters in emission order; each is compiled to one kernel.
  """
```

### 5.2 Why the cap is `max`, not `sum`

Within a cluster each thread evaluates constraints **serially**, folding each constraint's interval into the feasibility accumulator and discarding it before the next (permitted by the no-retain, no-early-exit semantics). Therefore:

- **Peak register liveness** is `max_i(live_set_i)` plus a few accumulator registers, **not** the sum over the cluster.
- Adding more small constraints grows **instruction footprint and runtime**, not peak register pressure.
- The only thing that blows occupancy is a **single very wide constraint**, which clustering cannot fix. That is the singleton case above.

So `B` primarily trades kernel count and I-cache pressure, and the register cap is enforced per widest-member.

### 5.3 Calibrating `B` against occupancy (not a magic constant)

`B` is a proxy; the real target is registers-per-thread versus occupancy. Calibrate on the actual target GPU:

1. Compile a representative cluster kernel.
2. Query its register usage: driver `cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS, func)`.
3. Compute achievable occupancy: `cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, func, blockSize, dynSmem)`.
4. Choose `B` so the compiled kernel keeps occupancy above a floor (start at **50%**) for the current card's register file and warp limits.
5. Recompute per GPU and per precision (FP32 halves interval register width, shifting the budget).

**Emission-side enforcement**: emit `__launch_bounds__(blockSize, minBlocksPerSM)` in the generated kernel (Section 6) to force the compiler to cap registers to hit the target, or fail loudly by spilling (which the register query then reveals).

**Key APIs**:
- `cuFuncGetAttribute` with `CU_FUNC_ATTRIBUTE_NUM_REGS`, `CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`, `CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES` (spill detection).
- `cuOccupancyMaxActiveBlocksPerMultiprocessor`.

**Docs**:
- Driver occupancy: <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html>
- Function attributes: <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html>
- `__launch_bounds__`: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#launch-bounds>

### 5.4 Health metric

Instrument **total emitted ops summed across clusters** divided by **unique ops in the global DAG**. A ratio near 1 means little cross-cluster recomputation, so greedy is fine. A ratio well above 1 is the signal to add subexpression-aware grouping (deferred). Log this per problem.

---

## 6. Stage 4: Code generation

**Goal**: emit one templated CUDA C++ kernel per cluster, as straight-line SSA over your interval type, using a shared header for the partition logic and interval library.

### 6.1 Shared header (injected once, not regenerated)

Put invariant device code in one header string handed to NVRTC:
- The **SPSD partition** logic: thread index to subregion bounds, computed on-device from the selected region and the thread's grid index (the paper 2 technique you already build on). No per-region data is read from global memory.
- Your **interval library**: the `Interval<T>` type and directed-rounding operator overloads.

This header is passed via `nvrtcCreateProgram`'s `headers` / `includeNames` arrays and `#include`d by every cluster kernel. Do not re-emit it per cluster.

### 6.2 Per-cluster kernel body

**Shape** (templated on the scalar type so one generator serves FP64 and FP32):
```cpp
template <class T>
__global__ void __launch_bounds__(BLOCK, MIN_BLOCKS_PER_SM)
cluster_k(const Domain<T> dom, int n_regions,
          unsigned char* feasible /*out, per region*/,
          Interval<T>* obj_bound /*out, per region, objective clusters only*/) {
  /* Evaluate this cluster's constraints over one subregion per thread.
     Writes 0 to feasible[r] on any violation (absorbing, race-free).
     Objective clusters also write the objective interval for GUB reduction.
     Grid-stride over n_regions so a static graph handles variable counts. */
  for (int r = blockIdx.x * blockDim.x + threadIdx.x;
       r < n_regions; r += gridDim.x * blockDim.x) {
    Box<T> box = partition_subregion(dom, r);      // from shared header, no global reads
    // --- straight-line SSA emitted from the cluster DAG ---
    Interval<T> t0 = /* ... */;
    // ...
    if (!contains_feasible(tK, RHS_k)) feasible[r] = 0;   // per constraint
    // objective node (if present): obj_bound[r] = tObj;
  }
}
```

**Codegen rules**:
- Emit each DAG node as a named SSA temporary (`auto tN = op(args);`). Named temporaries let the compiler keep intermediates in **registers**, which is the whole advantage over an interpreter's indexed value stack.
- Free-list temporaries as they die so the compiler sees minimal liveness (mirrors the live-set estimate from Stage 1).
- Emit `__launch_bounds__` from the calibrated occupancy target.
- Grid-stride loop so one static graph tolerates a changing `n_regions` without re-instantiation (Section 9).

### 6.3 Templated kernel names

C++ template instantiations are name-mangled. To launch them from the driver API you must recover the mangled symbol:
- Register the instantiation with `nvrtcAddNameExpression("cluster_k<double>")` **before** compiling.
- After compiling, call `nvrtcGetLoweredName` to get the mangled name, then `cuModuleGetFunction` with that name.

**Docs**:
- Programming guide (codegen target semantics): <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>
- Math API for directed-rounding intrinsics used by your ops: <https://docs.nvidia.com/cuda/cuda-math-api/index.html>

---

## 7. Stage 5: JIT compilation (NVRTC)

**Goal**: compile each cluster source to a loadable CUBIN, cached across runs.

### 7.1 Flow

```
nvrtcCreateProgram(&prog, src, name, nHeaders, headers, includeNames)
nvrtcAddNameExpression(prog, "cluster_k<double>")   // per instantiation
nvrtcCompileProgram(prog, nOpts, opts)
nvrtcGetProgramLogSize / nvrtcGetProgramLog          // always read on failure
nvrtcGetCUBINSize / nvrtcGetCUBIN                     // real arch → CUBIN
nvrtcGetLoweredName(prog, "cluster_k<double>", &mangled)
nvrtcDestroyProgram(&prog)
```

### 7.2 Compile options that matter

- `--gpu-architecture=sm_89` (4090) or `sm_120` (5090). **Build this at runtime** from `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR/MINOR)` rather than hardcoding. A real arch (`sm_`) yields CUBIN; a virtual arch (`compute_`) yields PTX only.
- `--std=c++17` (or later, matching your interval header).
- `--fmad=false` **for correctness**. FMA contraction fuses a multiply-add into one rounding step, which can silently break interval directed rounding if any plain `+`/`*` slips past your intrinsics. Your directed-rounding intrinsics (`__dadd_ru`, `__dmul_rd`, `__fma_rd`, etc.) are not contractable, so this only guards accidental plain arithmetic, but the guard is cheap insurance.
- Include paths for the injected headers.

### 7.3 CUBIN vs PTX

- Prefer **CUBIN** (real `sm_`): no driver-side JIT at load, deterministic. Requires the compiling toolkit to know the target arch (CUDA 12.8+ for `sm_120`).
- Keep **PTX** as a portability fallback: it is driver-JITed at `cuModuleLoad`, which tolerates newer GPUs than the build toolkit but adds load latency.

### 7.4 Caching (critical given high reuse)

- Key a disk cache by `hash(cluster_DAG, precision, arch, nvrtc_version, options)`.
- On hit, skip NVRTC entirely and `cuModuleLoadData` the cached CUBIN.
- Your reuse profile (thousands to millions of replays per compile) makes first-solve compile cost negligible; caching removes it from repeat runs too.

**Key APIs**: `nvrtcCreateProgram`, `nvrtcAddNameExpression`, `nvrtcCompileProgram`, `nvrtcGetProgramLog`, `nvrtcGetCUBIN`/`nvrtcGetCUBINSize`, `nvrtcGetLoweredName`, `nvrtcDestroyProgram`.

**Docs**:
- NVRTC (current, 13.x): <https://docs.nvidia.com/cuda/nvrtc/index.html>
- Compile-option list: same page, "Supported Compile Options" section.

---

## 8. Stage 6: Module loading and launch configuration

**Goal**: turn each CUBIN into a launchable `CUfunction` and pick launch dimensions.

### 8.1 Load

```
cuModuleLoadData(&mod, cubin)                    // or cuModuleLoadDataEx for JIT opts
cuModuleGetFunction(&func, mod, mangled_name)    // mangled name from Stage 5
```

Cache the `CUmodule` and `CUfunction` for the whole solve. Load once, launch thousands of times.

### 8.2 Launch config

- Block size: a multiple of 32; 128 or 256 is a good default. Confirm against occupancy (Section 5.3).
- Grid size: with the grid-stride kernel, size the grid to fill the device (`cuOccupancyMaxActiveBlocksPerMultiprocessor * numSMs`), not to `n_regions`. This decouples grid from region count and keeps the graph static.

**Key APIs**: `cuModuleLoadData` / `cuModuleLoadDataEx`, `cuModuleGetFunction`, `cuLaunchKernel` (used inside the graph), `cuDeviceGetAttribute` (SM count, compute capability).

**Docs**: Driver API modules and execution: <https://docs.nvidia.com/cuda/cuda-driver-api/index.html>

---

## 9. Stage 7: CUDA graph construction

**Goal**: assemble the per-iteration work (partition + all cluster kernels + feasibility init + GUB reduction + copy back) into one graph, instantiated once and replayed.

### 9.1 What goes in the graph

Per iteration the graph contains:
1. **Root memset**: set the per-region `feasible[]` array to 1 (feasible). See Section 10 for why this is race-free with the cluster writes.
2. **Cluster kernels**: one node per cluster, each a grid-stride launch over `n_regions`. Independent clusters have no edges between them, so the runtime may overlap them if resources allow.
3. **GUB reduction**: a CUB `DeviceReduce` node reducing the device-only `obj_ub[]` scratch to a single scalar `candidate` (Section 10). Depends on the objective cluster and the feasibility writes.
4. **GUB fold**: a one-thread min-node `GUB = min(GUB, candidate)` (Section 10.2), keeping the incumbent device-resident. Depends on the reduction.
5. **D2H copy**: `feasible[]` (one byte per region) and `obj_lb[]` (one value per region, for the host objective-dominance filter). Both are constraint-count-independent. Depends on all cluster kernels. The `obj_ub[]` scratch is never copied; the scalar GUB stays on-device (copy it back only for logging or stopping criteria).

### 9.2 Build method: explicit API vs stream capture

Two equivalent construction paths:

- **Stream capture**: enqueue the memset, kernels, reduction, and copy on a stream between `cudaStreamBeginCapture` and `cudaStreamEndCapture`. To express **inter-cluster concurrency**, fork onto side streams and join with events (`cudaEventRecord` on the fork, `cudaStreamWaitEvent` on the join); capture records the fork/join as parallel branches. Simpler to write; the dependency structure is inferred from the recorded stream/event topology.
- **Explicit graph API**: `cudaGraphCreate`, then `cudaGraphAddMemsetNode`, `cudaGraphAddKernelNode` (one per cluster, dependencies listed explicitly), `cudaGraphAddMemcpyNode`. More verbose but gives exact control over the dependency edges, which is cleaner when you want a precise "all clusters depend only on the memset, copy depends on all clusters" fan-out/fan-in.

**Recommendation**: explicit API here. The dependency structure is simple and fixed (fan-out from memset, fan-in to copy), and explicit edges make the intended concurrency unambiguous rather than relying on capture's event bookkeeping.

### 9.3 Concurrency caveat (calibrated expectation)

Do **not** expect large gains from inter-cluster concurrency at 10^6 regions: a single cluster kernel already exposes ~10^6-way parallelism, which saturates the device. The concurrency mainly helps the tail (small or imbalanced clusters, and the singleton kernels). The real lever remains **per-kernel occupancy** from the clustering cap, not overlap. Keep the independent-node structure (it costs nothing and helps the tail), but attribute performance to occupancy first.

### 9.4 Instantiate once, replay

```
cudaGraphInstantiate(&exec, graph, 0)     // once
loop:
  // update the selected domain in device/constant memory (see 8.5)
  cudaGraphLaunch(exec, stream)           // every iteration
  cudaStreamSynchronize(stream)
```

### 9.5 Handling the changing domain and region count without rebuilding

Each iteration operates on a new selected region and possibly a different `n_regions`. Keep the graph **static**:
- Pass the selected region through device or constant memory that the partition logic reads; update its contents between launches (a plain `cudaMemcpy` outside the graph, or a memset/memcpy node inside it). The graph topology does not change.
- Absorb variable `n_regions` with the **grid-stride** kernels (Section 6.2): the grid is fixed to fill the device, and each kernel early-exits past `n_regions`. No relaunch sizing needed.
- If you later need to change kernel **parameters** bound into nodes (for example a different output pointer), use `cudaGraphExecKernelNodeSetParams` (single node) or `cudaGraphExecUpdate` (whole graph) rather than re-instantiating. Structural changes (adding/removing nodes) force a fresh `cudaGraphInstantiate`; parameter-only changes do not.

**Capture-time restrictions to respect**: no `cudaMalloc`, no host synchronisation, and no host-side RNG inside a captured region. Allocate all buffers before capture.

**Key APIs**: `cudaGraphCreate`, `cudaGraphAddMemsetNode`, `cudaGraphAddKernelNode`, `cudaGraphAddMemcpyNode`, `cudaGraphInstantiate`, `cudaGraphLaunch`, `cudaGraphExecKernelNodeSetParams`, `cudaGraphExecUpdate`; capture path: `cudaStreamBeginCapture`, `cudaStreamEndCapture`, `cudaEventRecord`, `cudaStreamWaitEvent`.

**Docs**:
- Programming guide, CUDA Graphs: <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html>
- Runtime API graph group: <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html>
- Intro blog (capture and replay pattern): <https://developer.nvidia.com/blog/cuda-graphs/>

---

## 10. Feasibility and bound reduction

**Goal**: produce, per region, a feasibility flag (AND over all constraints) and the objective bound, with minimal synchronisation.

### 10.1 Feasibility: absorbing-zero shared flag (no atomics)

- One shared `unsigned char feasible[n_regions]`, reset to 1 by the root memset each iteration.
- A cluster kernel writes `feasible[r] = 0` when a constraint's LHS interval does not intersect its range `[lb, ub]`, i.e. `!(lhs.lo <= ub && lhs.hi >= lb)`. This two-sided test comes straight from the `.nl` range form and subsumes LE/GE/EQ. A kernel never writes 1.
- Because 0 is **absorbing** and every writer writes the same value 0, concurrent writes from different cluster kernels to the same address are **race-free without atomics** (all candidate writes agree). This is why clusters can run concurrently and still compute a correct logical AND.
- A region is feasible iff `feasible[r] == 1` after all clusters complete.

This keeps state minimal (one byte per region), consistent with not retaining LHS intervals.

### 10.2 Objective bound and GUB

The objective is single, so its bound handling is light and produces no per-**constraint** host traffic. Two `n_regions` arrays, both independent of constraint count:

- `obj_ub[n_regions]`: **device-only scratch**, CUB input, never copied to the host.
- `obj_lb[n_regions]`: **copied to the host** so the CPU pruning filter can admit only regions with `obj_lb[r] < updated GUB`.

Handling:
- Reduce `obj_ub[]` in-graph to a scalar `candidate`: `cub::DeviceReduce::Reduce` (or `::Min`), `init = +inf`, over a transform iterator that maps infeasible regions to `+inf` so they cannot lower the bound. Then a one-thread node folds `GUB = min(GUB, candidate)`.
- Optional GPU pre-cull: mark `feasible[r] = 0` where `obj_lb[r] > GUB` using the identical `contains_feasible(interval, RHS=GUB)` test. This only shrinks the D2H set; the authoritative objective-dominance filter is the host pass in Section 11, which uses the freshly updated GUB.

**A single GUB buffer is race-free (no double-buffering).** The objective cluster reads `GUB`; the only writer is the incumbent-update node, which sits downstream through the chain objective cluster → CUB reduction → update. The graph therefore orders every read of `GUB` before the single write, so there is no unordered read/write within an iteration, and replays are serialised on one stream across iterations. One incumbent scalar suffices. (`candidate` and `GUB` are two locations only because CUB reduces into its own output before the fold, not for race avoidance.)

**Consequence (benign, worth knowing).** Because the read precedes the write, iteration N prunes against iteration N-1's incumbent, not against the tighter bound N's own reduction produces. This one-iteration lag only makes pruning marginally less aggressive; survivors are re-tested against the updated incumbent when re-partitioned. Closing it would need a second pruning pass after the reduction, rarely worth it.

**When double-buffering or atomics would be needed.** Only if the ordering assumption breaks: (a) folding the incumbent update into the objective kernel via `atomicMin(GUB, ...)`, where per-thread pruning then reads a possibly-stale `GUB` (benign, since `GUB` only decreases, so you under-prune rather than over-prune), or (b) replaying the graph on multiple streams against a shared `GUB` to process several nodes concurrently (a genuine race). Neither is in the current single-stream, single-node design.

**CUB usage** (two-phase, or single-call env overload in CCCL 3.4+):
```
cub::DeviceReduce::Reduce(nullptr, bytes, in, out, n, MinOp{}, init);  // size query
cudaMalloc(&tmp, bytes);                                               // once, before capture
cub::DeviceReduce::Reduce(tmp, bytes, in, out, n, MinOp{}, init);      // in-graph node
```
Allocate CUB temp storage **before** graph capture (no `cudaMalloc` inside capture).

**Key APIs**: `cub::DeviceReduce::Reduce` / `::Min`, custom reduction functor, `thrust`/`cub` transform iterator.

**Docs**:
- CUB DeviceReduce: <https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceReduce.html>
- CCCL overview: <https://nvidia.github.io/cccl/>

---

## 11. Stage 8: Per-launch execution and the driver boundary

**Goal**: define what one evaluation launch does and exactly where the evaluator hands off to the (external) driver.

**Per-launch flow (evaluator, in scope)**:
1. The driver sets the parent domain (and split factors) in the device/constant buffer the partition logic reads. The evaluator exposes this as a small "set domain" entry point.
2. `cudaGraphLaunch(exec, stream)`; synchronise.
3. The graph's D2H node returns `feasible[]` and `obj_lb[]`. The GUB is folded on-device (Section 10.2); it is copied back only if the driver asks for it.

**Handoff (driver, out of scope)**: the driver consumes `feasible[]` and `obj_lb[]` to manage its region list, refinement, GUB policy, and termination. None of that is specified here. The evaluator's contract is exactly: parent domain in; `feasible[]`, `obj_lb[]`, and a scalar GUB out.

**Notes**:
- The evaluator is stateless across launches apart from the incumbent GUB scalar; it holds no region list.
- Keep the set-domain write off the critical path where possible (pinned memory, async copy on the same stream before the graph node that consumes it).

---

## 12. Precision handling

- Generate one templated kernel; instantiate `cluster_k<double>` and `cluster_k<float>` as needed via separate `nvrtcAddNameExpression` calls.
- Compile per target arch (`sm_89`, `sm_120`), built at runtime from the device's compute capability.
- FP32 roughly halves interval register width, which raises the effective op-budget for the same occupancy. Recalibrate `B` per precision (Section 5.3).
- FP32 also moves the compute/memory balance: on the same card, FP32 is far less compute-bound, so favour coarser clusters. Because `B` is a runtime parameter, no code change is required.

---

## 13. Deferred and open items

- **Cross-constraint hash-consing**: the DAG is now a single global `ExprDAG` (Section 3.1), so the storage/id-scoping prerequisite for cross-constraint CSE already exists; only the hash-consing lookup itself (dedup on `(op, operands, payload)` at emit time) remains unbuilt, and it is currently scoped to dedup within one function's emission, not across functions. Extend it to consult the whole shared table, not just the current function's traversal, if Section 5.4's emitted-vs-unique op ratio shows real cross-constraint recomputation.
- **Subexpression sharing across clusters**: the clustering-side consumer of the point above; bias the clusterer to group constraints that share inputs.
- **NL-Reader callback frontend (Stage B2)**: custom `mp::NLHandler`/ProblemBuilder emitting `DAGNode` directly during the NL read, skipping `mp::Problem`. Zero-intermediate but more boilerplate; pursue only if B1's two-pass cost appears in profiling (Section 3.4).
- **`.osil` fallback frontend**: if some target instances lack `.nl`, `.osil` (XML, full expression tree) is an alternative source into the same `DAGNode` IR.
- **Method F (warp/block per region)**: relevant only if region count falls below device saturation while expressions stay heavy. Would nest inside a cluster kernel. Adaptive, later.
- **Datacenter coarsening**: on 1:2 FP64 parts, raise `B` to trade concurrency for fewer launches and less boundary traffic. Same knob.
- **Mean Value Form (derivative bounds)**: a *potential* future direction, not on the current path (main direction is parallel exploration with natural interval extension). If pursued, add a **reverse-mode AD pass over `DAGNode`** (record the forward walk, accumulate adjoints backward as new `DAGNode` nodes); the gradient DAG then flows through the existing codegen unchanged. Reverse mode keeps cost at a constant multiple of `f` rather than scaling with variable count. This is deliberately chosen over adopting MC++ (see below).
- **MC++ / relaxation stack (evaluated, rejected for now)**: MC++ would supply a mature DAG, forward+reverse symbolic AD, cross-constraint CSE, and McCormick/Taylor/Chebyshev relaxations, and could serve as a frontend (via NL-Reader callbacks) plus AD engine. Rejected because (a) the only near-term draw was AD, which a ~50-line reverse-AD pass over `DAGNode` supplies without the dependency; (b) it is a small research library (v4.0 on master, last tag v2.1 2022) with submodule externs and a licence to vet; (c) it is host-only and emits no CUDA, so it would not remove the lowering step. Reverse this decision only if convex **relaxations** for lower bounds become a goal, which is the one capability not cheaply reproducible.
- **Directed-rounding validation**: property-test the JITed kernels against the programmatic-frontend CPU oracle to confirm outward rounding survives codegen and `--fmad=false`.

Note: the search driver (region list, refinement, termination, GUB policy) is out of scope for this doc entirely, not merely deferred. See Section 1.

---

## 14. Documentation reference

| Topic | URL |
|---|---|
| NVRTC | <https://docs.nvidia.com/cuda/nvrtc/index.html> |
| CUDA Driver API | <https://docs.nvidia.com/cuda/cuda-driver-api/index.html> |
| CUDA Runtime API (graphs) | <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html> |
| CUDA Graphs (prog. guide) | <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html> |
| CUDA Graphs (blog) | <https://developer.nvidia.com/blog/cuda-graphs/> |
| Occupancy (driver) | <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html> |
| `__launch_bounds__` | <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#launch-bounds> |
| Math API (rounding intrinsics) | <https://docs.nvidia.com/cuda/cuda-math-api/index.html> |
| CUB DeviceReduce | <https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceReduce.html> |
| CCCL | <https://nvidia.github.io/cccl/> |
| Compute capabilities / GPUs | <https://developer.nvidia.com/cuda/gpus> |

Deep-link anchors on docs.nvidia.com occasionally drift between toolkit versions; if one 404s, navigate from the toolkit index for your installed CUDA version.

---

## 15. Decisions log

**Locked**:
1. **`B` calibration**: ship a **fixed occupancy floor (50%)** first; defer auto-tuning from measured register counts.
2. **Graph build path**: **explicit graph API** (fixed fan-out from memset, fan-in to the feasibility copy).
3. **GUB reduction**: **in-graph CUB node**, reducing the device-only `obj_ub[]` scratch to a scalar. Only `feasible[]` and `obj_lb[]` cross to the host, both constraint-count-independent.
4. **Host output**: **one feasibility flag per region** (`n_regions` bytes) plus `obj_lb[]`. No per-region, per-constraint output.
5. **Frontends**: **both permanent**, different use cases. **A, programmatic builder (primary, now)**: operator overloading, no deps, also the CPU oracle and the embedding API. **B, `.nl` via AMPL MP (later, benchmarks)**: B1 = `mp::Problem` + `ExprVisitor` (documented), B2 = direct `NLHandler` callback (deferred optimisation). The `DAGNode` DAG is the stable boundary between all of them.
6. **DAG scope**: **global, shared `ExprDAG`** across every objective and constraint in a `Problem` (revised from an earlier per-function-list design). A function is identified by a root node id; its own topological subset is recovered by backward traversal from that root, not stored separately. Cross-constraint hash-consing (Section 13) is a lookup-scope extension on top of this, not a structural prerequisite.
7. **Constraint form**: **two-sided range `[lb, ub]`** (from `.nl`), one feasibility predicate `lhs.lo <= ub && lhs.hi >= lb`.
8. **Lean stack**: **no MC++, no AD, no relaxations** (Section 1, Section 13). Reverse-AD-over-`DAGNode` is the chosen path if MVF is ever needed.
9. **Scope**: **evaluator only**. The search driver is owned separately and not specified here.
10. **Build order**: reproduce paper 1's per-operation CUDA Graph method (Section 4) as a measured baseline **before** building clustering/codegen/NVRTC (Sections 5–7), reusing the graph-construction and reduction machinery (Sections 9–10) across both. This replaces reliance on paper 1's own (different-hardware) numbers with a measurement on the actual target GPUs, and front-loads execution-scaffolding work that the JIT pipeline needs anyway.

**Remaining sub-decision**:
- **GUB fold location**: device-side one-thread min-node (recommended, zero host traffic) versus host-side min after a scalar D2H. Default to device-side unless the driver needs the incumbent each iteration anyway.

**Consequence to revisit later**: because one flag carries both elimination reasons (constraint-infeasible and objective-dominated), the driver cannot tell *why* a region died from the flag alone. Acceptable for the current design; if the driver later needs per-reason signal, the evaluator's output shape changes.