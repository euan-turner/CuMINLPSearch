# Refinement study

**What it measures.** Whether subdividing a box and taking the interval hull
over the children gives a materially tighter bound than interval analysis on
the parent box alone — and if so, how that improvement scales with the number
of subregions `N`.

Both solver backends rest on the premise that it does. This package attaches
numbers to that premise. It is measurement only: nothing here is on a solve
path, and `gams_solve`/`aggregate_solve` are unaffected by anything in it.

The findings are in [FINDINGS.md](FINDINGS.md). The reasoning behind the
experimental design — the metric definitions, the confounders, the staging —
is in `design/REFINEMENT_STUDY.md`.

## Running it

```sh
cmake --build build/dev --target refinement_study
study/refinement/run_sweep.sh build/dev/refinement_study
study/refinement/report.py --bounds study/refinement/results/declared
```

`run_sweep.sh` takes an optional list of configurations; with none it runs all
three. Each writes `results/<config>/<instance>.csv`, a `.log` holding the
frontend's warnings, and a `MANIFEST.json` recording the git revision, the
exact flags, the GPU and the date. Every configuration is seeded, so a re-run
on the same binary reproduces the CSVs byte for byte.

Needs a CUDA device.

**Runtime.** The ladder is capped at `N = 2^20` subregions by default, because
the last rung costs `2^n_live` times the one before it and without a cap it
dominates the entire sweep — hours rather than minutes. The cap is close to
free: refitting the corpus with it moves the fitted exponent by under 0.04 on
every instance (`ex8_1_1` 0.5003 vs 0.5002, `ex4_1_2` 1.0756 vs 1.0583). For a
run to device exhaustion:

```sh
MAX_REGIONS=0 study/refinement/run_sweep.sh build/dev/refinement_study
```

### The three configurations

| config | partition | parent boxes | what it answers |
|---|---|---|---|
| `declared` | uniform | sigma ladder over each model's declared bounds | takeaway 1 — the corpus as modellers wrote it |
| `unit-width` | uniform | absolute width 1, placed near the origin | takeaway 2 — with the bound-width confound removed |
| `budget` | `BisectionBudgetCompositionPolicy` | sigma ladder | what a real search would see, on the full corpus |

`declared` and `unit-width` use `study::UniformCompositionPolicy`, which splits
every live dimension the same number of times. That is the only partition
shape the classical excess-width theorem is stated over, but it costs
`N = 2^(k·n_live)` regions, so it reaches low-dimensional instances only.
`budget` uses the solver's own rule, whose `N = 2^B` is independent of
dimension and therefore reaches the whole corpus — at the price of not being
the theorem's `N`.

Rows carry a `partition` column, so a directory mixing configurations is still
readable.

## Reading the output

```
study/refinement/report.py [--bounds] [--boxes=N] [--instance=NAME] <dir-or-csv>...
```

- **Bounds** (`--bounds`) — the refined hull `[L_N, U_N]` against the baseline
  `[L0, U0]`, in absolute terms, per parent box, as `N` climbs. Start here.
  Every other section is a ratio, and a ratio cannot tell you whether the
  interval it came from was `[−1.8, 0.2]` or `[−3×10^10, 1×10^15]`.
- **Q1** — how fast the hull's *excess* width vanishes. The hull width itself
  converges to the objective's true range over the box, not to zero, so the
  excess over that floor is what the classical theorem bounds and what a
  power-law fit can meaningfully describe. Reported per side and in total,
  normalised per dimension against the theorem's prediction of 1.0.
- **Q1b** — the older width ratio `rho ~ N^-alpha`, with its confound stated:
  `rho` has a nonzero asymptote, so its exponent is driven toward 0 by
  construction once the excess is small relative to the range.
- **Q2** — whether the per-subregion bounds are all alike or have isolated
  outliers.
- **Attribution** — how much of the tightening is constraint propagation
  rather than interval refinement, from the masked/unmasked pair.

## Files

| | |
|---|---|
| `run_sweep.sh` | the driver: configurations, instance lists, manifests |
| `report.py` | CSV → the four sections above |
| `check_sweep.py` | end-to-end invariant check on the CSV; run by ctest as `refinement_sweep_check` |
| `FINDINGS.md` | the results and what they mean |
| `results/` | sweep output, one directory per configuration |

The instrument itself lives outside this directory, because it is code the
rest of the project builds:

- `source/gams/refinement_study.cu` — the binary
- `include/cuminlp/study/` — the host-side arithmetic (hulls, gains, parent-box
  drawing, the distribution summary, the uniform partition policy)
- `test/source/refinement_study_test.cu`, `study_distribution_test.cpp`,
  `uniform_partition_test.cpp`, `parent_box_fixed_width_test.cpp` — its tests
