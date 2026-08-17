# Findings — what subdivision actually buys

**Last updated**: 14 August 2026. Numbers from `results/`, reproducible with
`run_sweep.sh` (see [README.md](README.md)). Design rationale in
`design/REFINEMENT_STUDY.md`.

---

## Summary

Both solver backends assume that subdividing a box and taking the interval
hull over the children beats relaxing the parent directly. It does, and it
does so at the rate classical interval analysis predicts.

1. **The convergence rate is first-order, as the theory says.** The interval
   hull's *excess* width — what is left after subtracting the objective's true
   range over the box, which is the floor no subdivision can beat — vanishes
   like `N_dim^-1` in the per-dimension refinement, on every instance where it
   can be measured. At the largest reachable `N` the hull is 99.6%+ true
   range. This is not what an earlier reading of this study concluded, and §3
   below explains why that reading was wrong.
2. **The rate is not the problem. The starting width is.** On an instance
   whose root box carries a defaulted `±1,000,000` bound, first-order
   convergence from a width of `2×10^6` still leaves a bracket of order
   `10^30` at the largest `N` a GPU can hold. The same instance, measured on a
   unit-width box, closes to a bracket of `3.4`.

The practical consequence: the return on device budget is real and predictable,
and effort spent on tighter initial bounds dominates effort spent on more
subregions. Doubling `N` on a two-variable problem buys a factor of `√2` off
the bracket; getting a variable's bounds from `±10^6` to `±10` buys `10^5`.

---

## 1. The convergence rate — of the hull's excess width

**The hull width does not converge to zero, and fitting a power law to it is a
category error.** As `N` grows, `lb_r → min f` over subregion `r` and
`ub_r → max f` over `r`, so

```
L_N → min f over P      U_N → max f over P
W_N → the true range of f over P
```

The hull width has a **floor**: the objective's true range over the box, which
no subdivision can go below. `rho = W_N / W0` therefore converges to
`range / W0`, a positive constant, and any fit of `rho ~ N^-alpha` is fitting
a power law to something with a nonzero asymptote. That is why `alpha` comes
out near zero however well the relaxation behaves.

The classical theorem bounds not the width but the **excess width** — the part
of the hull that is relaxation looseness rather than genuine range:

```
W_N = (true range) + excess(N),      excess(N) ≤ K · w(X) / N_dim
```

Neither term is directly observable, but both are bracketed soundly by
quantities already in the CSV, because a hull bound and the opposite extreme
over the subregions sandwich the truth:

```
min_r ub_r  ≥  min f  ≥  L_N          U_N  ≥  max f  ≥  max_r lb_r
```

giving `excess(N) ≤ (min_ub − L_N) + (U_N − max_lb)`, both terms non-negative
by construction. `N` is the total region count, `N_dim^n_live`, so the
theorem's per-dimension first-order prediction is `beta × n_live = 1`.

| config | instance | `n_live` | lower | upper | **total** | **× `n_live`** | excess @ max `N` | as % of range |
|---|---|---|---|---|---|---|---|---|
| declared | `ex4_1_2` | 1 | 1.076 | 0.963 | 0.966 | **0.966** | 1.15e9 | 0.001% |
| declared | `ex4_1_5` | 2 | 0.675 | 0.477 | 0.483 | **0.965** | 6.19e31 | 0.279% |
| declared | `ex8_1_1` | 2 | 0.500 | 0.500 | 0.500 | **1.000** | 1.8e-3 | 0.313% |
| unit-width | `ex4_1_2` | 1 | 1.550 | 0.879 | 0.879 | **0.879** | 2.89e10 | 0.002% |
| unit-width | `ex4_1_5` | 2 | 0.513 | 0.488 | 0.495 | **0.990** | 10.2 | 0.348% |
| unit-width | `ex8_1_1` | 2 | 0.506 | 0.500 | 0.502 | **1.005** | 3.7e-3 | 0.238% |

Corpus mean `beta × n_live`: **0.977** (declared), **0.958** (unit-width),
against a predicted 1.0. The interval hull's excess width vanishes at the
classical first-order rate, and at the largest reachable `N` the hull is
**99.6%+ true range** on every instance measured.

Two things follow. First, nothing here requires a large Lipschitz constant to
explain — an earlier version of this study invoked one, and that explanation
is withdrawn. Second, the hull is wide on these instances because the
*function* varies that much over the box, not because the arithmetic is loose.

**The two sides can converge at different rates.** On `ex4_1_2` the lower side
fits 1.076–1.550 while the upper fits 0.879–0.963. The total tracks the slower
side, so a lower-side-only figure (the optimality bracket alone) overstates
how fast the hull is tightening. Both are reported for that reason.

## 2. Why the bounds are still useless where they are useless

The rate being right does not make the bound good. First-order convergence
from a large enough starting width is still useless at any reachable `N`, and
that is the situation on most of this corpus.

The GAMS frontend defaults an unbounded variable to `±1,000,000`, so `w(X)` is
`2×10^6` per such variable before any nonlinearity is considered. `ex4_1_5`,
`circle` and `ex14_1_1` all warn on this (see the `.log` beside each CSV).

`ex4_1_5` isolates the effect, since it appears in both configurations with
everything but the box width held fixed:

| | parent box | gap at max `N` |
|---|---|---|
| declared | root, `±10^6` on both variables | **7.7e30** |
| unit-width | width 1, placed within `±10` of the origin | **3.36** |

Thirty orders of magnitude, from the box width alone. Both converge at
essentially the same *rate* (`beta × n_live` of 1.35 and 1.03). The rate was
never the difference.

This is the sense in which the original takeaway holds: **an unbounded
variable makes the interval hull useless in absolute terms, and no amount of
subdivision recovers it.** What does not hold is the inference that
refinement is therefore not working. It is working exactly as advertised, from
a hopeless starting point.

## 3. The metric that made this look like a failure

The study's original headline was a refinement exponent of roughly 0.008,
read as "an 8192-fold increase in `N` cuts the width by 3×; subdivision buys
almost nothing". That number is real, and the conclusion drawn from it is not.

`rho = width(refined hull) / width(baseline hull)`, and per §1 the hull width
converges to the objective's true range, not to zero. So `rho` converges to
`range / W0` — a positive constant — and a power-law fit to it must return an
exponent near zero once the excess is small relative to the range. On this
corpus the excess is under 0.4% of the range at the largest `N`, so `rho` is
measuring the range and the range does not move.

Concretely: where the true range is wide, `U_N` is enormous *and correct* — it
cannot shrink, because it is already right — and `rho` stays pinned near 1
however well `L_N` converges.

`ex4_1_2` is the clean case. It is a degree-50 polynomial on `x1 ∈ [1, 2]`, so
its true range is genuinely of order `10^15`: `max f ≈ 1.2e15 ≈ 2^50`, while
`min f ≈ −663.5`. From `results/declared/`, one parent box:

```
ex4_1_2.gms  box 0  sigma=1  partition=uniform
  baseline  N=1  [  -3.32515e+10,    1.21976e+15]  width 1.2198e+15
             N             L_N             U_N         width     rho            dL          min_ub           gap
             2    -3.24456e+10     1.21976e+15   1.21979e+15  1.0000     8.059e+08     8.05899e+08   3.32514e+10
             8         -192430     1.21976e+15   1.21976e+15  1.0000   3.32513e+10         255.218        192685
           128        -744.752     1.21973e+15   1.21973e+15  0.9999   3.32514e+10        -610.991       133.761
          1024        -671.756     1.21973e+15   1.21973e+15  0.9999   3.32514e+10        -655.685       16.0708
          2048        -667.572     1.21973e+15   1.21973e+15  0.9999   3.32514e+10        -659.538       8.03373
          4096        -665.522     1.21973e+15   1.21973e+15  0.9999   3.32514e+10        -661.506        4.0159
          8192        -664.507     1.21973e+15   1.21973e+15  0.9999   3.32514e+10        -662.499       2.00802
         16384        -664.003     1.21973e+15   1.21973e+15  0.9999   3.32514e+10        -662.999       1.00397
         32768        -663.751     1.21973e+15   1.21973e+15  0.9999   3.32514e+10        -663.249      0.501976
```

(rows elided for brevity; the ladder is contiguous in the real output)

`rho` reads 0.9999 for the entire ladder. `U_N` never moves, because 1.2e15
is the right answer. Meanwhile `L_N` climbs from `−3.3e10` to `−663.5`, and
the bracket **halves with every doubling of `N`** — 8.03, 4.02, 2.01, 1.004,
0.502 — textbook first-order convergence, entirely invisible in `rho`.

The `unit-width` configuration makes the divergence starker still: `ex4_1_2`
there fits `alpha = 0.0000` on `rho` (which holds at 0.9999) and
`beta × n_live = 1.55` on the bracket, over the very same launches.

Reproduce this view with:

```sh
study/refinement/report.py --bounds --boxes=1 --instance=ex4_1_2 results/declared
```

`report.py` therefore leads with the bracket exponent (Q1) and demotes `rho`
to Q1b with the confound stated inline. `rho` is still worth reporting — it is
the honest answer to "how much narrower is the hull" — but it answers a
question about *width*, not about *optimization*, and on this corpus the two
diverge completely.

One consequence not re-measured: the earlier observation that `rho` gets worse
as the parent box shrinks is suspect for the same reason. Shrinking a box
shrinks the objective's true range on it, which raises the share of hull width
that is genuine range rather than relaxation looseness — so that trend may be
the metric rather than the geometry.

## 4. Q2 — are some subregions much better than others?

No. Three of the five instances come out "mostly tied" and two "broad, no
outliers"; none show isolated outliers.

| instance | `distinct_frac` | `low_score` | `low_frac` | `low_gap_2` |
|---|---|---|---|---|
| `circle` | 0.0035 | 1.045 | 0.0008 | 0.017 |
| `ex14_1_1` | 0.0002 | 0.491 | 0.0000 | 0.000 |
| `ex4_1_5` | 0.0032 | 0.288 | 0.0000 | 0.000 |
| `ex4_1_2` | 1.0000 | 0.178 | 0.0000 | 0.000 |
| `ex8_1_1` | 1.0000 | 0.985 | 0.0000 | 0.001 |

On `circle`, `ex14_1_1` and `ex4_1_5` the bounds become increasingly *tied* as
`N` grows — most subregions share a lower bound with some other subregion.
`ex4_1_2` and `ex8_1_1` are the opposite in form (every subregion distinct)
and the same in consequence: `low_score` sits below the Tukey fence of 1.5 and
`low_gap_2 ≈ 0`, so the minimising subregion has near neighbours rather than
standing alone.

Either way there is no small set of standout subregions for an adaptive policy
to chase, which bears directly on `AGGREGATE_BOUNDING.md`'s
`SelectionRule::RejectIndex` — it keys on `hull_ub − lb`, which is only
informative if that spread discriminates.

One asymmetry worth noting: `ex4_1_2`'s **high** tail scores 23.5, far above
the fence. Per §2.3 the high tail of `{lb_r}` is the provably-bad regions —
the fraction of the box a single launch eliminates — so a strong high tail
with a flat low tail is the signature of subdivision paying off through
*pruning* rather than through moving `L_N`. This study measures it; acting on
it is a solver decision.

## 5. Constrained instances

On `circle` and `ex14_1_1`, most randomly placed sub-boxes are proven wholly
infeasible — a unit box inside a `2×10^6`-wide square essentially never meets
a bounded feasible region. Those rows carry `excluded_frac = 1` and every
masked statistic is empty. This is a real measurement, not a gap in the data,
but it means the masked columns on those instances average over a handful of
survivors and must be read with their `n`.

The masked/unmasked split (§2.5) exists so that constraint propagation's
contribution is separable from interval refinement's. Where both are
available, a large gap between `rho_masked` and `rho_unmasked` means the
tightening is constraints doing the work, not the objective's relaxation
narrowing.

## 6. The solver's own partition, on the full corpus

`results/budget/` runs `BisectionBudgetCompositionPolicy` — the rule
`gams_solve`/`aggregate_solve` actually use — over all 14 parsing-corpus
instances. Its `N = 2^B` is independent of dimension, so it reaches the
instances the uniform partition cannot. It is *not* comparable to the theorem:
a shared budget `B` spent greedily on the widest live domain leaves dimensions
untouched, so its `N` is not the theorem's `N`.

Two things it shows:

- **On low-dimensional instances the two partitions agree.** `ex8_1_1` fits
  `beta = 0.5003` under the budget partition and `0.5003` under the uniform
  one — identical to four figures. With few live variables the greedy heap
  already cycles through them roughly evenly, so the distinction that
  motivated the uniform policy only bites in higher dimension (`alkyl`, 14
  variables, never bisects more than 7 of them).
- **The bracket is unmeasurable on most of the corpus.** 10 of 14 instances
  are constrained, and `min_ub` is not a sound bound on the optimum when
  constraints are present. The headline rate result therefore rests on
  unconstrained instances only. Extending it to constrained ones needs a
  genuine primal witness — the sampler — not `min_ub`.

The `rho` column is higher here than under the uniform partition (corpus mean
`alpha = 0.040` against `0.008`), which is not evidence that the budget
partition refines better. It reflects which instances each configuration can
reach: the budget corpus includes constrained instances where much of the
apparent tightening is constraint propagation excluding subregions, not the
objective's relaxation narrowing. §5 and the report's Attribution section are
what separate those.

---

## What would change the picture

- **Tighter initial bounds are worth more than more subregions**, by orders of
  magnitude, on any instance with a defaulted bound. Bounds tightening /
  presolve is the higher-leverage investment.
- The bracket `[L_N, min_ub]` is computed already and thrown away on the solve
  path. On unconstrained instances it is a sound optimality bracket obtained
  with no sampling and no function evaluation at a point (§2.4). Whether to
  use it is a solver decision this study does not make.
- Everything here is measured at `k = 1` on low-dimensional instances. The
  uniform partition cannot reach higher dimension — `N = N_dim^n_live` — so
  whether `beta × n_live ≈ 1` survives at ten or twenty live variables is
  untested, and is the obvious next question.
