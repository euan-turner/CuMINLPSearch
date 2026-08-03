# Using `gams_solve`

Solves a GAMS scalar-format model straight from its `.gms` file.

```sh
gams_solve [--policy=<name>] [--list-policies]
           [--dump-dag[=infix|nodes]] [--dump-only] [-h|--help]
           <model.gms> <iterations>
```

For building, see [BUILDING.md](BUILDING.md); the binary lands at
`build/<preset>/gams_solve`. Sample models live in `test/data/gams/`.

**Start with no flags.** The search shape is selected automatically from the
model's variable kinds and this device's free memory.

```sh
./build/dev/gams_solve /path/to/minlplib/gms/nvs09.gms 1248
```

```
nvs09.gms: 10 variables, 0 constraints, 123 DAG nodes
policy: discrete (auto), partition_num: 7, enumerate_cap: 7, sample_points: 5,
        max_cycle_size: 7 (rung 8)
...
------------ Finished ------------
-43.1343 <= min <= -43.1343
Proven optimal: no pending region can beat the incumbent (the iteration
limit was reached, but every remaining region is already dominated).
RESULT	sense=min	primal=-43.134336918035324	dual=-43.134336918035324
```

`-h` / `--help` prints the flag list, to stdout, without needing a GPU or a
model.

---

## Positional arguments

| Argument | Meaning |
|---|---|
| `<model.gms>` | GAMS scalar-format model to solve. |
| `<iterations>` | Branch-and-bound iteration limit. Required unless `--dump-only`. |

There is no positional shape argument any more — see `--policy` below. A
third positional produces a specific error naming `--policy` instead of a
bare usage dump, since it is the one mistake worth naming explicitly.

---

## Policy selection

Four numbers decide the search shape: `partition_num` (bisection width),
`enumerate_cap` (largest integer domain still enumerated exactly),
`sample_points` (points sampled per subdomain, for the incumbent) and
`max_cycle_size` (how many variables one iteration acts on). Rather than ask
for all four, `gams_solve` picks a **policy** — a named bundle of rules for
choosing them — and a **resolver** turns that policy into concrete numbers
for this problem and this device.

| Flag | Effect |
|---|---|
| `--policy=<name>` | Use this named policy instead of selecting one automatically. |
| `--list-policies` | Print the policy roster (name, rules, evidence, provisional status) and exit 0. Needs no GPU and no model. |

```sh
./build/dev/gams_solve --list-policies
```

```
  all-binary
      partition=pin(2) enumerate=pin(2) sample_points=1 cycle=fit (fallback 20)
      evidence: autocorr_bern20_03.cu
  discrete
      partition=fit enumerate=cover-domains(16) sample_points=5 cycle=fit (fallback 7)
      evidence: nvs09.cu, RUNTIME_SHAPE.md
  mixed-binary
      partition=fit enumerate=follow-partition sample_points=5 cycle=fit (fallback 4)
      evidence: no size split has been measured (provisional -- ...)
  mixed-all-small
      partition=fit enumerate=cover-domains(16) sample_points=10 cycle=fit (fallback 4)
      evidence: ex8_6_2 (continuous-only)
  mixed-all-large
      partition=fit enumerate=cover-domains(16) sample_points=3 cycle=fit (fallback 4)
      evidence: sample_points=3 is a guess, not a measurement (provisional -- ...)
```

With no `--policy`, automatic selection looks at the *lowered* model's
variable kinds and counts (not the `.gms` source) and picks the row that
matches:

| Policy | Selected when |
|---|---|
| `all-binary` | no continuous, no integer variables |
| `discrete` | some integer variable, no continuous |
| `mixed-binary` | some continuous, some binary, no integer |
| `mixed-all-small` | some continuous, and the model has ≤ 64 live variables |
| `mixed-all-large` | some continuous, and more than 64 live variables |

A continuous-only NLP matches none of the discrete rules and falls through to
`mixed-all-{small,large}`, which is correct rather than merely a default: the
rules that would otherwise apply (an enumerate ceiling, a binary fan-out) are
no-ops on variable kinds the problem doesn't have.

The objective variable is not counted if it survived lowering only because
neither elimination pass could solve for it (`--dump-dag` shows whether it
did): such a variable occupies a slot in the search but is not a genuine
degree of freedom, so letting it push an otherwise all-binary or discrete
model into a mixed row would classify on an artefact of the lowering rather
than the model's actual character.

`mixed-binary`, `mixed-all-small` and `mixed-all-large` are marked
provisional in `--list-policies`: their split points and constants are
judgement calls awaiting measurement, not tuned rows like `all-binary` and
`discrete` are. See [design/POLICY_SELECTION.md](design/POLICY_SELECTION.md)
for the full rationale.

**A named `--policy` is checked against the model, not just looked up.**
`--policy=discrete` on a model with a continuous variable, or
`--policy=all-binary` on anything with an integer or continuous variable, is
rejected (exit 2) rather than silently run with a fan-out tuned for a
different variable kind. The mixed-all size split is not part of this check
— `--policy=mixed-all-small` on a 200-variable model is accepted, since
comparing the two rows on the same instance is exactly what pinning one is
for; only a policy whose rules assume a kind the problem lacks is rejected.

### `partition_num` and `max_cycle_size` are fitted, not tabulated

Given a policy (named or auto-selected), the resolver derives the actual
numbers from the problem and the device, rather than looking them up. The
variable count sets a **coverage target** — the point past which one
iteration could in principle act on every live variable — and `partition_num`
is fitted in two phases: first the widest cap reachable at all (fan-out 2,
the cheapest possible), then how far `partition_num` can widen from there
*without giving that cap back*. `enumerate_cap` comes from the problem's
integers alone (the largest domain, capped at a policy-specific ceiling),
independent of `partition_num` unless the policy explicitly couples them.

This is the same problem `--max-cycle-size` used to solve alone: a shape
tabulated from one hand-tuned driver could ask for more memory than exists.
`discrete`'s tabulated `partition_num`/`max_cycle_size` were both 10 once,
inherited verbatim from a revision of `source/nvs09.cu` — a configuration
needing 221 GiB that has never run on a consumer GPU. Fitting both from
device arithmetic re-derives the value that revision was eventually corrected
to (7), with no table.

Two consequences worth stating plainly:

- **A run's shape is not reproducible from its command line alone.** It
  depends on how much memory was free. What *is* reproducible is the
  **policy name** (paired with the commit, since the roster lives in
  `include/`) — that's what `tools/minlp_status.py` records, and what a
  benchmark that must be reproducible should pin with `--policy=<name>`.
- **A busier GPU gets a narrower search**, silently but visibly — the
  resolved numbers on the status line move.

### Experimental overrides

`--partition-num`, `--enumerate-cap`, `--sample-points` and `--max-cycle-size`
still exist, but only as **research instrumentation**, applied on top of the
resolved shape rather than as the primary configuration surface:

| Flag | Effect |
|---|---|
| `--partition-num=N` | Overrides the resolved `partition_num`. Must be ≥ 2. |
| `--enumerate-cap=N` | Overrides the resolved `enumerate_cap`. Must be ≥ 1. Given `--partition-num` without this, `enumerate_cap` follows it instead of staying at what the policy resolved — see the note below. |
| `--sample-points=N` | Overrides the resolved `sample_points`. |
| `--max-cycle-size=N` | Overrides the resolved `max_cycle_size`. Must be ≤ 64. |

A run using any of these reports `source=overridden` (instead of `auto` or
`named`) on the status line and in `PARAMS`. They exist so the roster's value
can be A/B-tested against a pinned shape; without them the resolver's
contribution couldn't be measured. They are not where to reach for day-to-day
tuning — `--policy=<name>` is.

### `--partition-num` vs `--enumerate-cap`, overridden

They are independent by construction now: the resolver's `enumerate_cap`
comes from the problem's integers, not from `partition_num`. That coupling
only comes back if you override `--partition-num` without also overriding
`--enumerate-cap` — the same single-flag shorthand the old template
parameters had, reapplied on top of the resolved shape. On `nvs09`'s ten
`[3, 9]` integers, `--partition-num=2` (alone) drops `enumerate_cap` to 2 and
flips all ten from enumerating to bisecting. The tool says so whenever this
happens:

```
policy: discrete (overridden), partition_num: 2, enumerate_cap: 2,
        sample_points: 5, max_cycle_size: 10 (rung 16)
  note: --partition-num also set enumerate_cap to 2, so 10 integer
        variable(s) -- the widest of domain 7 -- bisect
        instead of enumerating, and are never fathomed exactly.
        --enumerate-cap=7 decouples the two if that was not intended.
```

The note states the mechanism and stops there, deliberately. Bisecting is
cheaper per slot, so the auto-fitted cap usually widens to compensate — on
`nvs09` that run converges, and "fixing" it with `--enumerate-cap=7` narrows
the cap to 6 and stops it converging. Which way to turn the knob is a
judgement about the model, not something the tool can settle.

### `max_cycle_size` and the capacity ladder

The per-thread slot context is register-resident, so its array bound has to
be a compile-time constant. Rather than baking one in, a small ladder of
capacities — **{8, 16, 32, 64}** — is compiled, and the cap in force is
dispatched to the narrowest rung that holds it.

Rounding up is free: unused slots are padding with fan-out 1, so a capacity-8
context running a 4-slot assignment produces exactly the children a
capacity-4 one would. The status line reports both, e.g.
`max_cycle_size: 20 (rung 32)`.

Asking for more than 64 is an error, not a silent clamp — a wider rung would
spill registers to local memory rather than merely cost more of them. Auto-fit
never returns more than 64 for the same reason.

---

## Inspection flags

| Flag | Effect |
|---|---|
| `--dump-dag[=infix\|nodes]` | Print the lowered `Problem` before solving. `infix` (default) reads as an expression; `nodes` lists the DAG node-by-node. |
| `--dump-only` | Print and stop. Implies `--dump-dag`, needs no GPU, and takes no `<iterations>`. |
| `-h`, `--help` | Flag list with defaults, on stdout, exit 0. |

Dumping is useful because `Problem::validate()` proves the DAG is
*well-formed*, not that it is the *right* DAG. Outside the two instances with
a hand-built oracle in the test suite, reading the expression back is the only
check there is.

```sh
./build/dev/gams_solve --dump-only test/data/gams/circle.gms
```

Parser warnings (missing bounds, an objective variable that could not be
solved for) are printed before the dump and are worth reading — a missing
bound becomes a ±1e6 box, which costs search effort and shows up as a dual
bound of exactly ±1e6 that never moves.

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Solved, hit the iteration limit, printed help, or `--list-policies`. |
| `1` | Parse error in the `.gms` file. |
| `2` | Bad flag value; an unknown `--policy` name (with the roster listed); a `--policy` that doesn't apply to this model's variable kinds (e.g. `--policy=discrete` on a model with a continuous variable); or a configuration the solver rejects (e.g. an overridden `--partition-num=1`, `--max-cycle-size` past the widest rung). |
| `3` | Out of device memory. The run was well-formed; a larger GPU, or a smaller experimental override, may succeed. |

`2` and `3` are deliberately distinct: `2` means you asked for something
meaningless, `3` means you asked for something merely too big. With no
overrides, `3` should only appear when even `all-binary`'s fallback shape
can't fit, which needs a very small GPU indeed.

---

## Sizing: the one thing to understand

The number of subregions evaluated per iteration is a **product over slots**:

```
n_regions  =  product of each slot's fan-out
           ≈  partition_num ^ max_cycle_size
```

So `--partition-num` and `--max-cycle-size` both scale the work
*exponentially*, and small increases are not small.

What that product costs is **not one graph's worth of memory**. The driver
caches, per composition it encounters, three replays that are live at the same
time:

| Graph | Elements per region | Bytes per element |
|---|---|---|
| point (incumbent) | `sample_points` | `nodes × sizeof(T)` + 25 |
| interval (bounds, pruning) | 1 | `nodes × sizeof(interval<T>)` + 25 |
| exact (fathoming, enumerable compositions only) | 1 | `nodes × sizeof(T)` + 25 |

That is why a model with a hundred-node expression graph runs out long before
the region count looks alarming, and why `sample_points` matters to sizing as
much as the fan-out does.

Exceeding available memory is caught **before anything is allocated**, and
reported with the multiplication broken down plus a cap that would fit:

```
$ ./build/dev/gams_solve --partition-num=60 --max-cycle-size=4 \
      test/data/gams/ex2_1_1.gms 5

out of device memory: point graph needs 40.7 GiB of device memory,
but only 7.7 GiB is available.
  composition: 4 live slot(s) of 8 compiled
    4 x Continuous (fan-out 60 each)
  -> 12,960,000 regions x 10 sample points
  x 337 B per element (39 DAG-node buffers of 8 B, plus 25 B of
    per-element bookkeeping)
  = 40.7 GiB

  Acting on 3 variable(s) at a time instead of 4 leaves every graph the
  solve holds for one composition -- point, interval -- at 827.9 MiB
  together: try --max-cycle-size=3
  Lowering --partition-num / --enumerate-cap shrinks each slot's fan-out,
  which reduces the product too.
```

The two byte figures are different quantities on purpose. `40.7 GiB` is the
one graph that overflowed; `827.9 MiB` is every graph a solve holds at that
cap, together — which is the number the suggestion has to be right about. It
did not used to be, and the suggestion was correspondingly unfollowable: the
recommendation was costed against the failing graph alone, so following it
produced a second out-of-memory failure on a different graph.

### Worked example: why the co-resident cost is the one that matters

`nvs09` is ten integer variables on [3, 9]. Domain size 7 is within the
default `enumerate_cap`, so every variable enumerates and the region count is
`7 ^ max_cycle_size`. With `sample_points = 5` and 102 buffer-bearing nodes:

| `--max-cycle-size` | regions | exact graph alone | all three graphs |
|---|---|---|---|
| 6 | 117,649 | 94 MiB | 0.73 GiB |
| 7 | 823,543 | 0.65 GiB | **5.14 GiB** |
| 8 | 5,764,801 | 4.52 GiB | 35.99 GiB |
| 10 | 282,475,249 | 221 GiB | 1.72 TiB |

The middle column is what the tool used to size against, and reading it is
enough to see the old bug: on an 8 GiB card, 4.52 GiB at cap 8 looks like a
comfortable fit and is not one. The right-hand column is what actually has to
fit, and on this device it selects cap 7 — the same value
`source/nvs09_problem.hpp` was hand-tuned to. Auto-fit now derives it.

---

## Reading the output

```
test/data/gams/ex2_1_1.gms: 5 variables, 1 constraints, 55 DAG nodes
policy: mixed-all-small (auto), partition_num: 10, enumerate_cap: 10,
        sample_points: 10, max_cycle_size: 5 (rung 8)
PARAMS	policy=mixed-all-small	source=auto	partition_num=10	enumerate_cap=10	sample_points=10	max_cycle_size=5
...
------------ Finished ------------
-29.45 <= min <= -15.9619
Pending size: 8175
Viable regions: 2417
Pruned as interval-infeasible: 2021384
objective (minimise): -15.9619
RESULT	sense=min	primal=-15.961862397412036	dual=-29.450000000000074
  x1 = 0.994049
  ...
```

- The bracket is `GLB <= min <= GUB`. `GUB` is an incumbent attained by a
  real sampled point; `GLB` is a sound bound over everything not yet
  discarded.
- The tab-separated `PARAMS` line is the status line above it in machine form,
  and the two always agree. `policy=` and `source=auto|named|overridden` say
  *which* policy and how it was chosen; the four numbers are what the
  resolver produced for this run specifically (after any experimental
  override), not a reproduction key on their own — see
  [Policy selection](#policy-selection). `tools/minlp_status.py` records the
  policy cell (plus any overrides) beside the bound. The rung is not on the
  `PARAMS` line: it follows from `max_cycle_size` and no flag sets it.
- **`Proven optimal`** appears when no pending region can beat the incumbent,
  and the bracket then collapses to a point. This is reported on three
  distinct endings: the frontier emptied, the search dequeued a region already
  worse than the incumbent, or the iteration limit arrived with every
  remaining region already dominated. The third is the common one and used to
  be mis-reported — see the note below.
- The tab-separated `RESULT` line restates the bracket **in the model's own
  sense**, at full precision, on one grep-able line. It is what
  [`tools/minlp_status.py`](tools/minlp_status.py) records; see
  [Tracking a corpus](#tracking-a-corpus-minlp_statusmd) below. `primal` is the
  incumbent, `dual` is the proven bound, and either may be `none` — `primal`
  when no feasible point was sampled, `dual` when the search never dequeued a
  region. Negation does not swap the two: on a `max` model `primal` is still
  the attained value and `dual` still the bound, so the inequality reads
  `primal <= max <= dual`.
- **Viable regions** are those not yet proven suboptimal. Non-zero means the
  search was cut short, not that it failed. Zero, with an incumbent, means
  optimality was proven — that is exactly the `Proven optimal` condition.
- **Pending size** is the raw queue length and is *not* a measure of progress.
  It can be large while `Viable regions` is zero: dominated regions stay
  queued until they are dequeued and discarded.
- For a maximisation, the objective is negated on the way in and negated
  back on the way out; a `--dump-dag` shows what the solver actually
  minimises, and says so.

> **Note on older logs.** Until recently, hitting the iteration limit set
> `GLB` to the frontier's minimum without clamping it to the incumbent. When
> every pending region was already dominated, that printed a *lower* bound
> above the upper one — `nvs09` reported `-43.1244 <= min <= -43.1343` on a run
> that had in fact proven optimality at −43.134336 — and emitted a `RESULT`
> line whose `dual` was on the wrong side of its `primal`. Since
> `minlp_status.py record` keeps the better of two bounds, any such row in
> `MINLP_STATUS.md` should be re-measured rather than trusted. An inverted
> bracket in a log is the signature.

### If no solution is reported

```
(no feasible sample was found; no solution point to report)
```

The interval bounds pruned the space without random sampling ever landing on
a feasible point. This is expected for **equality-constrained** models — a
uniformly sampled point essentially never satisfies an equality exactly. The
bounds printed are still sound; there is just no witness. Raising
`--sample-points` does not generally fix this. Note that overriding it alone
does not also re-fit `--max-cycle-size` -- overrides apply *after* the
resolver has already fitted the shape to the policy's own `sample_points`, so
a wider `--sample-points` here risks exceeding the device budget the fitted
`max_cycle_size` was sized against; override `--max-cycle-size` alongside it
if you raise this.

---

## Tracking a corpus: `MINLP_STATUS.md`

Everything above solves one instance. [MINLP_STATUS.md](MINLP_STATUS.md) is the
other axis: one row per MINLPLib instance, recording whether the frontend can
parse it at all and the best bounds any run has ever found on it.

| Column | Where it comes from |
|---|---|
| `Parses`, `Sense`, `Vars`, `Cons`, `Nodes`, `Notes` | re-measured from the corpus on every `refresh` |
| `Best primal`, `Best dual`, `Primal @`, `Dual @`, `Primal iters`, `Dual iters`, `Primal policy`, `Dual policy` | accumulated from `gams_solve` runs, one `record` at a time |
| `Ref primal`, `Ref dual` | downloaded from MINLPLib by `reference` |

The split is the whole design. Parse status is cheap, deterministic and true
of the corpus all at once, so it is thrown away and re-derived. A bound is one
expensive GPU run of one instance and only ever improves, so it is carried
across refreshes untouched and never recomputed. The reference bounds are
neither: they are one download, replaceable at any time but not by anything
`refresh` can do offline, so they have a command of their own and are carried
across a `refresh` like a recorded bound.

### Refreshing the parse status

```sh
cmake --build --preset=dev --target gams_report
tools/minlp_status.py refresh
```

`refresh` shells out to `gams_report --per-instance`, which walks the corpus
and emits one TSV row per `.gms` file, then rewrites the table around whatever
bounds are already recorded. Expect about **five minutes** on a debug build for
the full 1633 instances. Run it after any frontend change — that is the number
that says which instances just became attemptable.

The corpus path is baked in as
`/vol/bitbucket/et422/minlplib_gms/minlplib/gms` and overridden with
`--corpus`; `--status` moves the output file. `--reject-discrete` reproduces
the pre-integrality baseline, for attributing a coverage delta to integrality
rather than to everything else a change touched.

`gams_report` is independently useful without the tracker. Bare, it prints the
aggregate coverage tally and a histogram of rejection reasons — *"which
operator should I implement next"* as a measurement rather than a guess — and
`--list` names the files behind each histogram row, since a bare count is not
something you can open and read.

```sh
./build/dev/gams_report --list /vol/bitbucket/et422/minlplib_gms/minlplib/gms
```

### Refreshing the reference bounds

```sh
tools/minlp_status.py reference
```

`Best primal` and `Best dual` say what this solver found; on their own they do
not say whether that is any good. `reference` fills the two columns that do:
the bounds MINLPLib publishes for each instance, the best anyone has reported
to them.

It downloads [`instancedata.csv`](https://www.minlplib.org/instancedata.csv),
the machine-readable form of the per-instance table on
`minlplib.org/instances.html` — same numbers, in a file with eighty-odd
semicolon-separated columns, of which `primalbound`, `dualbound` and
`objsense` are read. `--from` takes a path instead of the URL, for a machine
with no route out or to re-run against a copy already on disk:

```sh
curl -O https://www.minlplib.org/instancedata.csv
tools/minlp_status.py reference --from instancedata.csv
```

The whole corpus is refilled at once and the header line records where the
numbers came from and when — they are only as current as that date, and
MINLPLib's move as other solvers improve them. Rows MINLPLib has no entry for
are left alone and listed; so are rows whose `objsense` disagrees with our
`Sense`, since a disagreement there means the two names are not the same model
and a plausible number in the right column is worse than an empty one.

Both bounds are in the instance's own sense, so together they bracket the
optimum, and reading a row across is the point:

| Reading | Means |
|---|---|
| `Ref primal` = `Ref dual` | solved in the literature; that number is the optimum |
| `Best primal` worse than `Ref primal` | someone else's solver found a better solution than this run |
| `Best primal` better than `Ref primal` | a new best solution — or a bug, and the second is likelier |
| `Ref dual` empty | nobody has published a lower bound; ours is not being compared to anything |
| `Ref dual` infinite | the instance is known **infeasible**, so any primal we record is wrong |

The bracket also checks us. `Best primal` past `Ref dual`, or `Best dual` past
`Ref primal`, contradicts the literature — an infeasible point counted as a
solution, or a relaxation that cut off the optimum — and `record` says so when
it writes one, as does `reference` when a fresh download makes an existing row
contradictory:

```
warning: nvs09 primal -43.2 is past MINLPLib's dual bound -43.13433692,
         which no feasible point can be; one of the two is wrong
```

A bound merely *tighter* than MINLPLib's is not an error and gets a `note:`
rather than a warning, because it is the same event one step short of being
provable: either a result worth reporting upstream or the same bug. Both
comparisons carry a 1e-6 relative slack, since MINLPLib publishes about ten
significant figures and this table stores twelve.

### Recording a bound

```sh
./build/dev/gams_solve <corpus>/alkylation.gms 40 | tee run.log
tools/minlp_status.py record alkylation --log run.log
```

`record` scrapes the last `RESULT` line out of the log, along with the `PARAMS`
line and the `iter` count of that same run, and keeps whichever bound is
better, so re-running a worse configuration cannot lose ground:

```
alkylation (max): primal — -> 4.5; dual 6200 not better than 6107.49999061, kept
```

"Better" is sense-aware — for a `max` row a larger primal and a smaller dual
are the improvements — which is why the table carries a `Sense` column at all.
If the log's sense disagrees with the row's, `record` refuses rather than
attaching real numbers to the wrong instance.

Each bound gets its own `@`, `iters` and `policy` column, because the two
rarely improve in the same run and each is a claim about *that* run. Together
they are the whole recipe: the commit says what code ran, `policy` says which
policy it ran under, and `iters` doubles as the iteration limit to re-run
under — a run that converged at iteration N still converges at N under a
limit of N.

```sh
# reproduce the recorded alkylation primal, exactly
./build/dev/gams_solve <corpus>/alkylation.gms 40 --policy=mixed-all-small
```

The policy name, not the four resolved numbers, is what makes the row
reproducible. `partition_num` and `max_cycle_size` are fitted to whatever the
GPU had free at the time, so pasting *them* back onto a different machine
reproduces a shape but not the decision — the policy name paired with the
commit column (the roster lives in `include/`, so a rule change is a
differing hash) reproduces the decision exactly, which is the record worth
keeping. An overridden run's cell carries the overrides too, e.g.
`--policy=discrete --partition-num=7`, and pasting that back pins the
override the same way.

A `-dirty` suffix on the commit means a **build input** was uncommitted, so the
number is **not** reproducible from that hash alone — treat it as provisional
until re-measured on a clean tree. An empty `policy` cell means the bound was
recorded before the column existed, not that the run used no policy.

The suffix answers to the build, not to `git status`. A bound recorded while a
design note, a log or MINLP_STATUS.md itself was uncommitted *is* reproducible
from the hash alone, gets no suffix, and raises no question — a mark that
appears on every tree with an unsaved paragraph in it is a mark nobody reads.
Both `record` and `refresh` stop only when the uncommitted files could
actually have changed the run:

```
warning: the working tree is dirty, so this bound will be stamped `9f3c633-dirty` --
         a hash that does not identify the code that ran.
  changed, and could change what a run computes:
    source/gams/solve.cu
  changed, but not build inputs:
    USAGE.md
    run.log
proceed? [y/N]
```

A tree dirty only in notes, logs and status files says one line and carries on
— the line exists because a missing `-dirty` on a tree `git status` calls dirty
otherwise reads as a bug in the tool:

```
note: 6 uncommitted file(s), none of them build inputs, so this bound is
      stamped `9f3c633` with no -dirty suffix.
```

"Build input" is a heuristic — paths under `source/`, `include/`, `test/`,
`cmake/`, plus anything that compiles and the CMake files — biased towards
suffixing, since a needless question costs one keystroke and a missed one
costs a bound recorded against code nobody looked at. It is now the only thing
between a bound and a hash that overstates it, so widen it rather than argue
with it if it ever forgives something that matters. Answer `n` to go and
commit first; `-y` skips the question for scripted runs, and it is skipped
automatically when there is no terminal to ask on. Passing `--commit` also
skips it, since that names a revision this tree's dirt says nothing about.

### Overruling a recorded bound

"Keeps whichever is better" is the right default and the wrong one when the
old number has stopped meaning the same thing — a commit that changed what the
search does, or a row recorded against a policy you have since found to be
wrong. Two flags overrule it:

| Flag | Effect |
|---|---|
| `--force` | overwrite even if this run's bound is worse. Only touches the bounds this run actually reported; the other stays as it was. |
| `--replace` | make this run *the* row. Every recorded cell becomes this run's, and a bound this run did not report is **cleared**, not kept. |

```sh
# this commit changed the search; the old bounds are not comparable
tools/minlp_status.py record nvs09 --log run.log --replace
```

The difference only shows when a run reports one bound and not the other:
`--force` leaves the missing one alone, `--replace` clears it, because a bound
carried over from a superseded commit is precisely the one that would be
trusted by mistake. Both also stamp the commit, count and policy of the run
that overruled, so the row keeps describing one run rather than a merge of
several. `--commit` overrides the recorded hash if you are re-recording a run
made at some other revision.

### What the columns are telling you

`Parses: yes` is a statement about the **frontend only**. It means `gams_solve`
will get as far as a `Problem` — not that the search converges, and not that it
converges in a time you would wait for. The sizing section above is what
decides that, and `Vars` / `Nodes` are in the table to let you pick instances
that will fit before spending a run on them.

`Ref primal` and `Ref dual` are the only columns in the table this project did
not produce. They are a target and a sanity check, not a measurement of this
solver: an instance MINLPLib has closed says what number to aim at, and one
they have not says the gap you are looking at may be theirs as much as yours.

`Notes` carries the rejection reason on a `no` row. On a `yes` row it carries
caveats that do not stop a solve but should temper trust in its bounds:
`objvar-kept` (the objective variable survived as a search dimension tied by
an equality — expect a dual bound pinned at the objective variable's own lower
bound, since the objective is then that bare variable), `objvar-ineq` (it was
eliminated through an inequality, which is exact at the optimum rather than
pointwise, and may have dropped a bound stated on it), `default-bound` (some
free variable got the artificial ±1e6 box), and `default-bound-integer` (the
same, on an integer variable, where a 2e6-wide box is a search-quality cliff
rather than merely a cost).

---

## Related

- [MINLP_STATUS.md](MINLP_STATUS.md) — per-instance corpus status and best known
  bounds.
- [BUILDING.md](BUILDING.md) — build presets.
- [TESTING.md](TESTING.md) — running the test suite.
- [design/RUNTIME_SHAPE.md](design/RUNTIME_SHAPE.md) — why these parameters
  are runtime configuration, what constrains the capacity ladder, and the
  replay-soundness contract a custom composition policy must honour.
- [design/POLICY_SELECTION.md](design/POLICY_SELECTION.md) — the policy
  roster, the resolver that fits partition_num/max_cycle_size to the device,
  and the classification rule behind automatic selection.
- [design/GAMS_FRONTEND.md](design/GAMS_FRONTEND.md) — the subset of GAMS
  the parser accepts.
