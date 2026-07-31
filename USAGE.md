# Using `gams_solve`

Solves a GAMS scalar-format model straight from its `.gms` file — no
hand-written problem builder, no recompilation per instance.

```sh
gams_solve [--dump-dag[=infix|nodes]] [--dump-only] [-h|--help]
           [--partition-num=N] [--enumerate-cap=N]
           [--sample-points=N] [--max-cycle-size=N]
           <model.gms> <iterations> [all-binary|discrete|mixed]
```

For building, see [BUILDING.md](BUILDING.md); the binary lands at
`build/<preset>/gams_solve`. Sample models live in `test/data/gams/`.

**Start with no flags.** The defaults are chosen to run on the device you
actually have, so a bare invocation is a real attempt at the model rather than
a first guess to be corrected:

```sh
./build/dev/gams_solve /path/to/minlplib/gms/nvs09.gms 1248
```

```
nvs09.gms: 10 variables, 0 constraints, 123 DAG nodes
shape: discrete, partition_num: 7, enumerate_cap: 7, sample_points: 5,
       max_cycle_size: 7 (auto, rung 8)
...
------------ Finished ------------
-43.1343 <= min <= -43.1343
Proven optimal: no pending region can beat the incumbent (the iteration
limit was reached, but every remaining region is already dominated).
RESULT	sense=min	primal=-43.134336918035324	dual=-43.134336918035324
```

`-h` / `--help` prints the flag list with its defaults, to stdout, without
needing a GPU or a model.

---

## Positional arguments

| Argument | Meaning |
|---|---|
| `<model.gms>` | GAMS scalar-format model to solve. |
| `<iterations>` | Branch-and-bound iteration limit. Required unless `--dump-only`. |
| `[shape]` | Optional. Pins which row of defaults to use; otherwise inferred from the model's variable kinds. |

The shape argument selects **defaults only** — every value it sets is
individually overridable by a flag below. Pin it for a benchmark run that
must not depend on the inference heuristic guessing right.

| Shape | Inferred when | Modelled on |
|---|---|---|
| `all-binary` | no continuous, no integer variables | `source/autocorr_bern20_03.cu` |
| `discrete` | integers but no continuous variables | `source/nvs09.cu` |
| `mixed` | any continuous variable | tuned for `ex8_6_2` |

Inference looks at the *lowered* model, not the `.gms` source. A file whose
variables are all binary but whose objective variable survived as a search
dimension lowers to a continuous variable and infers `mixed` — `--dump-dag`
shows which. That is the case the positional argument exists for.

---

## Search-shape flags

These four were C++ template parameters until recently, so the usable search
shapes were fixed at build time. They are now runtime configuration: one
binary, tuned per instance.

| Flag | Effect | Default |
|---|---|---|
| `--partition-num=N` | Bisection width: how many sub-intervals a continuous or bisected-integer slot splits into. Must be ≥ 2. | 2 / 7 / 10 per shape |
| `--enumerate-cap=N` | Largest integer domain still enumerated exactly rather than bisected. | follows `--partition-num` |
| `--sample-points=N` | Points sampled per subdomain, for the incumbent (upper bound). | 1 / 5 / 10 per shape |
| `--max-cycle-size=N` | How many variables one iteration acts on. Must be ≤ 64. | **fitted to the device** |

Three of the four defaults are the value the shape was originally tuned with.
`--max-cycle-size` is not, and the difference is the point of the next
section.

### `--max-cycle-size` is measured, not tabulated

Left unset, it is the widest cap whose graphs fit in this device's free
memory, and the status line marks it `(auto)`:

```
max_cycle_size: 7 (auto, rung 8)
```

It used to be a per-shape constant, and that was the tool's worst behaviour.
The `discrete` row held 10, inherited verbatim from a revision of
`source/nvs09.cu` whose `CYCLE_SIZE` was 10 — a configuration needing 221 GiB
that has never run on a consumer GPU. So the shape documented as *"modelled
on `nvs09.cu`"* could not solve `nvs09`: a bare `gams_solve nvs09.gms 1248`
failed out of memory, and every recovery was the user re-deriving, by hand and
by trial, a number the tool already had everything it needed to compute.

Auto-fitting spends about two thirds of free device memory on the widest
composition, leaving the rest as headroom for the narrower ones the search
also reaches. Passing `--max-cycle-size=N` explicitly overrides it and
restores the old behaviour, including the old failure mode: an explicit cap
that does not fit is an error, not a request to pick something smaller.

Two consequences worth stating plainly:

- **A run's shape is not reproducible from its command line alone.** It
  depends on how much memory was free. The status line records what was
  actually used; a benchmark that must be reproducible should pass
  `--max-cycle-size` explicitly, exactly as it pins `[shape]`.
- **A busier GPU gets a narrower search**, silently but visibly — the `(auto)`
  number moves.

### `--partition-num` vs `--enumerate-cap`

They are independent on purpose. `--enumerate-cap` decides *when* an integer
variable is small enough to enumerate outright; `--partition-num` decides how
wide a bisection is when it isn't. Raising the enumerate threshold therefore
does not force wider bisection everywhere else.

Left unset, `enumerate-cap` tracks `partition-num`, and **that is the sharpest
edge on this tool**. Lowering `--partition-num` to shrink memory also lowers
`enumerate-cap`, and any integer whose domain no longer fits under it stops
being enumerated and starts being bisected. Bisected integers are never
fathomed exactly, which changes what the search does, not merely how fast it
does it. On `nvs09`'s ten `[3, 9]` integers, `--partition-num=2` drops the cap
to 2 and flips all ten. The tool now says so, whenever `--partition-num` is
given without `--enumerate-cap` and some integer no longer fits:

```
shape: discrete, partition_num: 2, enumerate_cap: 2, sample_points: 5,
       max_cycle_size: 10 (auto, rung 16)
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

### `--max-cycle-size` and the capacity ladder

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
| `0` | Solved, hit the iteration limit, or printed help. |
| `1` | Parse error in the `.gms` file. |
| `2` | Bad flag value, or a configuration the solver rejects (e.g. `--partition-num=1`, `--max-cycle-size` past the widest rung). |
| `3` | Out of device memory. The run was well-formed; a larger GPU or smaller parameters may succeed. |

`2` and `3` are deliberately distinct: `2` means you asked for something
meaningless, `3` means you asked for something merely too big. With
`--max-cycle-size` left to auto-fit, `3` should only appear when you overrode
it.

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
shape: mixed, partition_num: 10, enumerate_cap: 10, sample_points: 10,
       max_cycle_size: 5 (auto, rung 8)
PARAMS	shape=mixed	partition_num=10	enumerate_cap=10	sample_points=10	max_cycle_size=5
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
  and the two always agree. It reports the **resolved** shape — after the
  per-shape defaults, after auto-fitting `--max-cycle-size` — using the same
  names as the flags that set each value, so pasting them back reproduces the
  run on a machine whose free memory would have auto-fitted differently.
  `tools/minlp_status.py` records it beside the bound for exactly that reason.
  The rung is not on it: it follows from `max_cycle_size` and no flag sets it.
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
`--sample-points` does not generally fix this (and costs memory linearly, so
it narrows the auto-fitted `--max-cycle-size`).

---

## Tracking a corpus: `MINLP_STATUS.md`

Everything above solves one instance. [MINLP_STATUS.md](MINLP_STATUS.md) is the
other axis: one row per MINLPLib instance, recording whether the frontend can
parse it at all and the best bounds any run has ever found on it.

| Column | Where it comes from |
|---|---|
| `Parses`, `Sense`, `Vars`, `Cons`, `Nodes`, `Notes` | re-measured from the corpus on every `refresh` |
| `Best primal`, `Best dual`, `Primal @`, `Dual @`, `Primal iters`, `Dual iters`, `Primal params`, `Dual params` | accumulated from `gams_solve` runs, one `record` at a time |
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

Each bound gets its own `@`, `iters` and `params` column, because the two
rarely improve in the same run and each is a claim about *that* run. Together
they are the whole recipe: the commit says what code ran, `params` says what
shape it ran with, and `iters` doubles as the iteration limit to re-run under —
a run that converged at iteration N still converges at N under a limit of N.

```sh
# reproduce the recorded alkylation primal, exactly
./build/dev/gams_solve <corpus>/alkylation.gms 40 \
    --partition-num=10 --enumerate-cap=10 --sample-points=10 --max-cycle-size=5
```

Pasting `params` back matters most for `--max-cycle-size`, which auto-fits to
free device memory and so differs between machines, between GPUs, and between
two runs on the same GPU with something else resident. Recording the resolved
value is what makes the row reproducible without having to remember to pin the
flag by hand beforehand.

A `-dirty` suffix on the commit means a **build input** was uncommitted, so the
number is **not** reproducible from that hash alone — treat it as provisional
until re-measured on a clean tree. An empty `params` cell means the bound was
recorded before the column existed, not that the run used no flags.

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
search does, or a row recorded against a shape you have since found to be
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
trusted by mistake. Both also stamp the commit, count and params of the run
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
an equality), `default-bound` (some free variable got the artificial ±1e6 box),
and `default-bound-integer` (the same, on an integer variable, where a 2e6-wide
box is a search-quality cliff rather than merely a cost).

---

## Related

- [MINLP_STATUS.md](MINLP_STATUS.md) — per-instance corpus status and best known
  bounds.
- [BUILDING.md](BUILDING.md) — build presets.
- [TESTING.md](TESTING.md) — running the test suite.
- [design/RUNTIME_SHAPE.md](design/RUNTIME_SHAPE.md) — why these parameters
  are runtime configuration, what constrains the capacity ladder, and the
  replay-soundness contract a custom composition policy must honour.
- [design/GAMS_FRONTEND.md](design/GAMS_FRONTEND.md) — the subset of GAMS
  the parser accepts.
