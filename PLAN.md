# GPU Interval-Analysis Global Optimization: Implementation Plan

**Target**: reproduce the method and a representative subset of results from Zhang, Shan & Cagan (2026), *Global optimization tailored for graphics processing units*, PNAS Nexus.
**Language**: C++/CUDA, single GPU, FP64.
**Downstream driver**: the design must extend cleanly to MIQPs (QPLIB), without today's simplifications becoming architectural dead ends.

---

## 1. Scope for this phase

**In scope:**
- Box-constrained continuous nonlinear minimization (Eq. 1–2 of the paper), no general linear or quadratic constraints yet.
- A representative subset of the paper's 11 benchmark functions (not full reproduction of Tables 1/2), chosen to exercise the different structural cases: separable multimodal (Ackley or Rastrigin), coupled/dependency-heavy (Rosenbrock).
- **Phase 0 validation target is Rosenbrock only** (see Section 11). Rosenbrock is purely polynomial (add/sub/mul/sqr), so it can be validated with hand-rolled interval ops and no cuinterval dependency. Rastrigin/Ackley need interval `cos`/`sin`, which requires tracking monotonic segments of a periodic function under interval inputs, real work, not a small hand-roll, and exactly what cuinterval already provides correctly. Deferred to post-Phase-1 rather than either hand-rolling interval trig or pulling cuinterval in ahead of schedule just for one function.
- No derivative-based pruning (first-order necessary-condition check from the paper is dropped for now).
- No division in the interval arithmetic (only `+`, `-`, `*`, and `sqr` as a distinct primitive).

**Explicitly stubbed, not built:**
- General linear/quadratic constraint pruning.
- Sparse coupling-term representation for large `Q` (structure defined now, backing implementation deferred).
- Integer variable handling (hybrid branch-and-bound), design fixed, implementation deferred until the continuous engine is validated.
- Compile-time/templated codegen backend for the objective tape (runtime interpreter only for now).

---

## 2. Problem representation

### 2.1 Region

A live region does not store its own coordinates. It stores enough to look up its parent's already-materialized coordinates and take one step from there.

```
struct Region {
    uint32_t subregion_idx;   // Sidx: index within parent's partition
    uint32_t iteration_idx;   // index into region_history: which selected
                              // region is this region's direct parent
    uint16_t cycling_idx;     // which block of dimensions was split to produce this region
    double   lower_bound;     // f's interval lower bound in this region
    bool     alive;           // lazy-deletion flag (see 2.3)
};
```

```
struct RegionHistory {
    // one entry per iteration, appended only when a region is selected
    // and its full boundaries are computed as a side effect of running
    // the sampling/rule-out kernels against it
    std::vector<std::array<double, 2*n>> materialized_bounds;
};
```

**Decision record (revised from arena/parent-pointer design)**: cross-checked against the authors' own reference implementation (GPUGO, `rosenbrock50.py`). Their scheme does not use a parent-pointer tree at all. Instead, every *selected* region's full boundary array is materialized once (as a byproduct of reconstructing it to run that iteration's kernels) and appended to a `region_list`/`region_history` indexed by iteration. Every subregion produced during that iteration then only needs a **single hop** back to its parent's already-materialized entry, not a walk up an arbitrary-depth ancestor chain, because the parent was never itself left as an unresolved chain of indices, it was fully resolved the moment it was selected.

This supersedes the original parent-pointer arena design. The original scheme kept every region's footprint minimal (no coordinates stored anywhere) at the cost of an `O(depth)` reconstruction walk. This scheme instead stores one full `n`-dimensional boundary array per **iteration** (not per region), which is a materially smaller cost than per-region storage (iteration count is orders of magnitude below the live region-list size at the scales in the paper's own tables), and gets `O(1)` reconstruction in exchange, a strictly better trade at this problem's actual access pattern (one selection per iteration, many subregions per selection).

### 2.2 Region -> boundary reconstruction

CPU-side routine, invoked once per iteration on the single selected region:

1. Look up `region_history.materialized_bounds[region.iteration_idx]`, the parent's full `n`-dimensional boundary array (already computed, no walk required).
2. Apply Eqs. 10-11 once, using `region.subregion_idx` and `region.cycling_idx`, to compute this region's own full boundary array from the parent's.
3. **Append the result to `region_history`** (this region is now itself "selected" and materialized; its own future children will reference this new entry). Output: `X_lower[n]`, `X_upper[n]`, transferred to GPU constant memory (or global memory if `n` is large enough that `2n` FP64 values exceed constant memory capacity, matching the paper's own fallback).

`region_history` grows by exactly one entry per iteration, bounded by total iteration count, not by the number of live regions in the priority list.

### 2.3 Region list (priority structure)

- Binary min-heap over `(lower_bound, region_id)`, keyed by `lower_bound`.
- **Lazy deletion**: removing a region when it's pruned (Section 5) just clears `alive`. The heap is not rebuilt. On pop, skip dead entries.
- GLB computation (for output/reporting) must scan only `alive` entries, or maintain a running minimum incrementally as regions are inserted/killed to avoid a full scan.

**Divergence from reference implementation, kept deliberately**: GPUGO's `rosenbrock50.py` does not use a heap at all, it selects the minimum via `np.argmin` over the full live-region array every iteration and physically deletes with `np.delete`, an `O(N)` operation per iteration rather than `O(log N)`. This is presumably adequate at the scales they report. The min-heap with lazy deletion is kept here as the better-scaling choice, this is a deliberate improvement over the reference, not an oversight, flagged so a later correctness comparison against GPUGO's behavior isn't surprised by the difference in selection mechanics (both should select the same region each iteration, just at different asymptotic cost).

---

## 3. Interval arithmetic

### 3.1 Library

Extend **cuinterval** (neilkichler.github.io/cuinterval) rather than writing from scratch.

Confirmed available as correctly-rounded primitives: `add`, `sub`, `mul`, `fma`, `sqr` (all 0-ulp error, i.e. exact outward rounding), plus `neg`, `min`/`max`, comparison/set operations. No division needed for this phase; skip `div`/`recip` entirely, this sidesteps the zero-straddling-denominator complexity entirely for now.

No matrix-level or batched operations exist in the library, everything is scalar interval-in, interval-out. Any quadratic-form or per-dimension reduction loop is hand-written on top of these scalar primitives, not provided.

**Related work note**: the same author (Neil Kichler, RWTH Aachen STCE group) has a companion paper on GPU-parallel interval branch-and-bound for deterministic global optimization (arXiv 2507.20769), built on cuinterval itself. Already reviewed, confirmed to be a branch-and-bound method that uses interval analysis over each node's domain to derive its bound, i.e. the same general family as the paper being reproduced here. Worth keeping in view while building the matrix/vector layer below, in case parts of it are already solved there.

### 3.2 Diagonal (squared) terms

Use `sqr()` as a first-class primitive, not `mul(x, x)`. This is the closed-form tight bound (`[0, max(a²,b²)]` if the interval straddles zero, `[min(a²,b²), max(a²,b²)]` otherwise) and avoids the artificial width inflation of treating the two occurrences of `x_i` as independent.

### 3.3 cuinterval integration steps

1. **Vendor via adapter, not direct calls.** Pull cuinterval in as a git submodule or CMake `FetchContent`, but route every use through a project-owned `interval_ops.h` that wraps only the primitives actually used (`add`, `sub`, `mul`, `sqr`, `neg`, plus the dedicated `mul_scalar` below). No call site references the library's names directly. This is the fallback path if the from-scratch option is ever needed later: only this one file changes.
2. **Validate primitives before building on them.** Hand-compute expected outward-rounded bounds for a handful of test intervals through `add`/`mul`/`sqr`, compare against library output. Cheap gate, catches silent corruption that a claimed 0-ulp error spec doesn't guarantee in practice.
3. **Dedicated `mul_scalar(double c, Interval x)`.** Used constantly (QP coefficients, cycling-loop reduction weights) wherever one operand is a plain real rather than a general interval. Branches once on the sign of `c` and does two correctly-rounded multiplies, cheaper than promoting `c` to a degenerate interval `[c, c]` and paying for the full four-multiply general case. Built as a distinct primitive in `interval_ops.h`, not synthesized from `mul` at call sites.

### 3.4 Matrix/vector interval operations (built on top of 3.3)

4. **`IntervalVector` type.** Structure-of-arrays layout (separate `lo[]` and `hi[]` device arrays), not array-of-structures, for GPU coalescing. Supports elementwise add/sub and a sum-reduction over a vector of intervals, the latter shared by the `reduce_dim` operator (Section 4.1) and by sparse-term accumulation below.
5. **Sparse interval mat-vec.** Given `(i, j, coeff)` triples (COO or CSR), compute `y_i = sum_j coeff_ij * x_j` via `add(acc, mul_scalar(coeff, x_j))` accumulated per row. Natural mapping: each thread (already holding its subregion's `x` bounds) loops over its row's nonzero list. This assumes roughly uniform nonzeros per row; if a QPLIB instance has very uneven row density, this same-thread-per-row mapping causes warp divergence. Flagged, deferred: don't build load-balancing (row reordering, segmented cross-thread reduction) until a real QPLIB instance is in hand to profile against.
6. **Interval quadratic form, lower-triangle convention.** QPLIB stores `Q` as **lower-triangular** (diagonal plus entries with `i > j`), not upper-triangular. The form is:

   ```
   x^T Q x = sum_i Q_ii * sqr(x_i)   +   sum_{i > j} 2 * Q_ij * mul(x_i, x_j)
   ```

   Diagonal term uses `sqr()` directly. Off-diagonal term (`i > j` only, doubled for symmetry) is exactly the sparse mat-vec pattern from Step 5, specialized to a self dot-product. This plugs into the `sparse_terms` slot of the three-part objective (Section 4.1) with no new representation, just a QPLIB-specific filler for that slot that iterates the lower triangle as stored, rather than assuming upper-triangle storage.
7. **Validate against a small dense reference before wiring in.** Small (`n = 5` to `10`) symmetric `Q` in lower-triangular form, hand-computed or CPU dense-loop reference interval quadratic form, compared against the sparse GPU kernel's output. Done before Step 8, isolating "is the sparse kernel correct" from "does the driver correctly consume it."
8. **Wire into the objective interface.** Only after Steps 1-7 pass. Becomes the concrete backend behind `evaluate_objective_interval(region)` for QP-family problems, sitting alongside the tape interpreter backend behind the same `Objective` abstraction (Section 4).

---

## 4. Objective function representation

### 4.1 Fixed three-part structure

```
f(x) = constant
     + reduce_dim(op, per_dim_tape, {dims})      // separable part
     + sum_{(i,j) in sparse_terms} coupling_tape(x_i, x_j, coeff_ij)
```

- **`per_dim_tape`**: a short expression tape (opcodes: `CONST`, `VAR`, `ADD`, `SUB`, `MUL`, `SQR`, plus transcendental ops `SIN`, `COS`, `EXP` as needed per benchmark function), evaluated once per dimension on that dimension's interval.
- **`reduce_dim` operator**: `SUM` or `PRODUCT`, applied across the thread's assigned dimensions in the same loop the variable-cycling partition already requires.
- **`sparse_terms`**: COO-style `(i, j, coeff)` triples with a small coupling tape (e.g. plain `MUL` for QP off-diagonal terms, the actual Rosenbrock neighbor-coupling expression for that function). For QP objectives sourced from QPLIB, `Q` is stored **lower-triangular** (`i > j` entries plus diagonal), matching the library's own convention, not upper-triangular; see Section 3.4 Step 6 for the resulting quadratic form.

This single representation covers all 11 paper benchmarks and the QP quadratic form (diagonal -> `per_dim_tape = SQR`, off-diagonal -> `sparse_terms`), with no separate "QP mode" code path.

**Decision record**: kept as a fixed three-part structure (not a fully general hypergraph of arbitrary-arity coupling terms). Every target function fits this decomposition and it maps directly onto the existing per-dimension cycling loop. Flagged as a mild structural assumption should a future objective not decompose this way.

### 4.2 Execution model

- **Tape storage**: constant memory, read-only, identical across all threads (same reasoning as the paper's own selected-region broadcast).
- **Intermediate values during evaluation**: per-thread local storage, sized from the tape's max live-value count computed once at tape-build time on the CPU.
- **Backend**: runtime interpreter only for this phase (walks the tape via a switch/dispatch loop). A compile-time templated interpreter (specialized per known tape, enabling register allocation instead of local memory) is a valid later optimization for the fixed 11-benchmark validation set, but is not built now. Same tape format supports both backends without a rewrite.

### 4.3 Authoring the benchmark functions

All benchmark functions used for validation are authored as tapes (constant + per-dim tape + reduce op + sparse terms), not hand-written CUDA kernels. This exercises the representation itself as part of validation: if a benchmark's structure cannot be cleanly expressed (Rosenbrock's pairwise coupling is the hardest case), that surfaces now rather than after QPLIB integration.

---

## 5. Pruning pipeline

Structured as an ordered list of independent filters per region/subregion, not a single hard-coded objective comparison:

1. **Objective-bound filter** (built now): drop if `lower_bound(f, region) > GUB`.
2. **Constraint filters** (stubbed, empty list for now): drop if any constraint's interval evaluation over the region proves infeasibility (e.g. for `g(x) <= 0`, drop if `lower_bound(g, region) > 0`; for `h(x) = 0`, drop if `0 ∉ [lower_bound(h), upper_bound(h)]`). Evaluated via the same tape/interval machinery as the objective, just against a different function. Empty list is a no-op today, and the loop structure means adding a constraint later requires no change to the surrounding iteration logic.

Derivative-based pruning (first-order necessary conditions) is not in this pipeline for this phase.

---

## 6. Variable cycling and partitioning

- Selected region is split along a block of `k` dimensions determined by `cycling_idx` (paper's demonstration value: 10 dimensions, 4-way uniform split per dimension, giving `4^10` subregions per iteration; kept as a tunable parameter, not hardcoded).
- Subregion index -> subinterval indices via the modulo/floor-division chain (Eqs. 8-9 generalized to `k` dimensions).
- Subregion bounds computed on-device via the SPSD pattern (Eqs. 10-11): only the selected region's bounds (`2n` floats) are transferred to constant/global memory, each thread computes its own subregion's bounds from its block/thread index, no per-subregion data transfer from host.
- After evaluation, only surviving subregion indices (one integer each) are written back to host, not full bound data, matching the paper's GPU-write minimization.

---

## 7. Sampling (GUB update)

- Diagonal sampling as in the paper: 10 points equally spaced along the selected region's main diagonal.
- Evaluated in plain floating point (not interval), since only an upper bound is needed here, not a rigorous enclosure.
- Smallest observed value updates GUB if improved.

---

## 8. Integer variables (design only, not built this phase)

**Hybrid branch-and-bound**, chosen over unifying integers into the interval-quartering scheme:

- Region struct gains a per-dimension type tag (continuous vs. integer) when this is implemented, not part of the current build.
- On a region containing a fractional integer variable, branch into two children (`x_i <= floor(v)`, `x_i >= ceil(v)`) rather than quartering.
- Variable cycling applies only to the continuous sub-block; already-fixed integer dimensions are skipped by the cycling index logic.
- Stopping criterion for integer dimensions is exact (width == 0 / single integer value), not the tolerance-based width criterion used for continuous dimensions.

This is documented now so the region struct and cycling logic aren't designed in a way that has to be reworked when integers are added.

---

## 9. Sparse representation (design only, not built this phase)

- `sparse_terms` in the objective (Section 4.1) already uses COO triples, this is the hook for sparsity, no dense `Q` matrix is ever assumed.
- When QPLIB integration begins, the per-dimension tape handles the diagonal of `Q` and `sparse_terms` handles the off-diagonal, with cost `O(nnz)` rather than `O(n^2)`. Off-diagonal entries are ingested and iterated in **lower-triangular** form (`i > j`), matching QPLIB's own storage convention (Section 3.4 Step 6), not upper-triangular.
- No dense matrix-vector kernel is planned; this is a hard constraint on the design, not an optimization to add later.

---

## 10. Module breakdown

| Module | Responsibility | Phase |
|---|---|---|
| `region.h/.cu` | Region struct, `region_history` (materialized parent boundaries, indexed by iteration), single-hop reconstruction | Build now |
| `region_list.h/.cu` | Min-heap with lazy deletion, GLB tracking | Build now |
| `interval_ops_min.h` | Hand-rolled `{lo, hi}` struct, outward-rounded `add`/`sub`/`mul`/`sqr` for Rosenbrock only, via CUDA rounding intrinsics; same signatures as the Phase 1 adapter below | Build now (Phase 0) |
| `interval_ops.h` | cuinterval-backed adapter, drop-in replacement for `interval_ops_min.h` once Phase 1 lands (`add`, `sub`, `mul`, `sqr`, `neg`, `mul_scalar`, plus transcendentals for Rastrigin/Ackley) | Build after driver loop |
| `interval_vector.h/.cu` | `IntervalVector` type (SoA lo/hi arrays), elementwise ops, sum-reduction | Build after driver loop |
| `sparse_matvec.cu` | Sparse interval mat-vec and lower-triangular interval quadratic form | Build after driver loop |
| `tape.h/.cu` | Expression tape data structure, opcode enum, runtime interpreter kernel | Build now |
| `objective.h` | Three-part objective structure (constant / per-dim tape+reduce / sparse terms), benchmark function authoring | Build now |
| `constraints.h` | Constraint filter interface, empty implementation | Stub |
| `partition.cu` | Variable-cycling subregion index computation, SPSD kernel | Build now |
| `sampling.cu` | Diagonal sampling kernel, GUB update | Build now |
| `driver.cpp` | Host-side iteration loop: select, sample, prune, partition, insert, stopping check | Build now |
| `integer_branch.h` | Integer branch-and-bound node type and splitting logic | Stub (design only) |
| `sparse_qp.h` | Sparse `Q`/constraint Jacobian ingestion for QPLIB | Stub (design only) |

---

## 11. Build sequence

**Phase 0 (first)**: driver loop end to end with a hand-hardcoded Rosenbrock objective (no tape, no cuinterval adapter layer). Covers `region_history` (iteration-indexed materialized parent boundaries), single-hop reconstruction, min-heap with lazy deletion, variable-cycling partition kernel, sampling kernel, stopping criteria. Goal: one fully working iteration loop, isolating the one true unknown (does the overall loop converge and prune correctly) before introducing any other new subsystem.

**On interval evaluation within Phase 0**: the pruning step genuinely needs a real interval-evaluated `lower_bound(f, region)`, this can't be faked with a plain floating-point evaluation at a corner or center, since that isn't a valid lower bound at all and would hide loop bugs behind bound bugs. Instead, hand-roll the small set of outward-rounded operations Rosenbrock actually needs (interval add, subtract, multiply, `sqr`) directly using CUDA's rounding-mode intrinsics (`__dadd_rd`/`__dadd_ru`, `__dmul_rd`/`__dmul_ru`, etc.) against a bespoke `{lo, hi}` struct, no library dependency yet. Give these hand-rolled functions the exact same signatures planned for `interval_ops.h` in Phase 1, so Phase 1 becomes a backend swap behind those signatures rather than a rewrite of every call site in the objective kernel.

**Phase 1**: cuinterval integration (Section 3.3, steps 1-3), adapter layer, primitive validation, dedicated `mul_scalar`. Can start once Phase 0's interval needs are understood, does not require Phase 0 to be fully complete, but should not be the first thing built. Rastrigin/Ackley (and any other benchmark needing periodic transcendentals) become validation targets once this phase lands.

**Phase 2**: matrix/vector layer on top of Phase 1 (Section 3.4, steps 4-7), `IntervalVector`, sparse mat-vec, lower-triangular interval quadratic form, validated against a small dense reference. Depends on Phase 1 only, independent of the tape interpreter.

**Phase 3**: tape interpreter (Section 4), built and validated standalone against a hand-coded reference evaluation before being wired into the driver loop from Phase 0.

**Phase 4**: wire both the tape interpreter (Phase 3) and the quadratic-form kernel (Phase 2) behind the shared `Objective` interface, replacing Phase 0's hardcoded function in the driver loop.

Phases 1 and 2 can proceed in parallel with Phase 0 once Phase 0 is underway, since neither depends on the driver loop being finished, only on knowing the interval type's shape.

---

## 12. Validation plan

1. Unit-validate the hand-rolled interval primitives (`add`/`sub`/`mul`/`sqr`) against hand-computed enclosures on small test intervals (Phase 0), then re-validate the cuinterval-backed versions the same way once Phase 1 lands.
2. Run the full driver loop on a low-dimension Rosenbrock instance (`n = 2` or `n = 5`) using the Phase 0 hardcoded objective, and manually verify the output region(s) bracket the known global minimum.
3. Once Phase 1 lands, validate the tape interpreter against direct hand-coded evaluation of a separable benchmark with periodic transcendentals (Rastrigin) on a handful of fixed intervals, then run the same low-dimension driver loop check against it.
4. Scale Rosenbrock to `n = 50, 100` and compare iteration count and enclosure against the paper's Table 1/2 figures, as a sanity check rather than full reproduction; repeat for Rastrigin once Phase 1/3 land.
5. Confirm Rosenbrock's cubic-to-quartic runtime scaling (vs. quadratic for separable functions) reproduces qualitatively, this is the paper's own signal that the dependency problem is being handled correctly rather than masked.

---

## 13. Open items to revisit later (not blocking this phase)

- Linear-scan selection (matching GPUGO's reference behavior) vs. the min-heap chosen here, revisit only if profiling shows heap overhead isn't worth it at the scales actually tested.
- Compile-time templated tape interpreter for the fixed benchmark set, once the runtime interpreter is validated.
- Structure-aware bounds for quadratic forms (e.g. exploiting PSD-ness of `Q`), flagged as a known limitation of plain interval evaluation on quadratic forms.
- Derivative-based pruning, dropped for this phase, would need interval-valued analytic derivatives per objective. Note: GPUGO's reference implementation has this active by default (see Section 2.3), so comparisons against it may show tighter bounds/fewer iterations than this phase's engine until derivative pruning is added.+1