#!/usr/bin/env python3
"""End-to-end invariant check on refinement_study's CSV output.

    study/refinement/check_sweep.py <refinement_study binary> <model.gms>

design/REFINEMENT_STUDY.md. The unit tests cover the pure host arithmetic;
this covers the thing they cannot see, which is how the *sweep* is wired.

The invariant that matters here is §3.2's: within one parent box, the hull
must tighten monotonically as the bisection budget grows, because the
budget-(B+1) partition refines the budget-B one. That is only true if every
budget is measured on the *same* boxes. An earlier version of the binary
reseeded its placement RNG per budget, so each budget drew fresh boxes and
rho(N) was confounded with placement rather than being a refinement curve --
a defect invisible to every unit test, because each individual launch was
correct. This script is what catches that class of wiring bug.

Exits non-zero with a description on the first violation.
"""

import collections
import csv
import subprocess
import sys

# Both bounds are reductions over the same doubles across nested partitions,
# so the comparison is exact up to a slack that only absorbs formatting.
SLACK = 1e-9


def number(row, key):
    """CSV value as a float, or None for the empty field an undefined
    quantity is written as (§2.1 -- never NaN)."""
    value = row[key]
    if value == "":
        return None
    return float(value)


def check_widths(text):
    """Every row must have exactly as many fields as the header declares.

    Not pedantry: a Distribution summarising an entirely-excluded parent box
    leaves its gap and quantile vectors empty, and a writer that emitted
    `vector.size()` fields produced short rows whose remaining columns all
    shifted left. Nothing downstream errors on that -- the CSV still parses,
    and every value is simply read under the wrong column name.
    """
    reader = list(csv.reader(text.splitlines()))
    if not reader:
        yield "no output at all"
        return
    want = len(reader[0])
    counts = collections.Counter(len(row) for row in reader[1:])
    for got, how_many in sorted(counts.items()):
        if got != want:
            yield (f"{how_many} row(s) have {got} fields but the header "
                   f"declares {want}; every column past the short one is "
                   f"misaligned")


def check(rows):
    """Yields a description for each violation found."""
    if not rows:
        yield "no rows emitted"
        return

    by_box = collections.defaultdict(list)
    for row in rows:
        by_box[row["box_id"]].append(row)

    for box_id, group in sorted(by_box.items()):
        group.sort(key=lambda r: int(r["budget"]))

        # The parent box must be identical across budgets. Its baseline is a
        # fingerprint: a differing baseline means a differing box.
        baselines = {(r["base_lb"], r["base_ub"]) for r in group}
        if len(baselines) != 1:
            yield (f"box {box_id}: {len(baselines)} distinct baselines across "
                   f"budgets -- the sweep redrew the box per budget, so its "
                   f"rho(N) is confounded with placement")
            continue

        for previous, current in zip(group, group[1:]):
            lo_prev = number(previous, "unmasked_lb")
            lo_curr = number(current, "unmasked_lb")
            hi_prev = number(previous, "unmasked_ub")
            hi_curr = number(current, "unmasked_ub")
            budgets = f"B={previous['budget']}->{current['budget']}"

            if None in (lo_prev, lo_curr, hi_prev, hi_curr):
                continue
            if lo_curr < lo_prev - SLACK:
                yield (f"box {box_id} {budgets}: lower bound loosened, "
                       f"{lo_prev} -> {lo_curr}")
            if hi_curr > hi_prev + SLACK:
                yield (f"box {box_id} {budgets}: upper bound loosened, "
                       f"{hi_prev} -> {hi_curr}")

    for row in rows:
        # Every refined hull sits inside its own baseline.
        base_lo, base_hi = number(row, "base_lb"), number(row, "base_ub")
        lo, hi = number(row, "unmasked_lb"), number(row, "unmasked_ub")
        where = f"box {row['box_id']} B={row['budget']}"
        if None not in (base_lo, lo) and lo < base_lo - SLACK:
            yield f"{where}: hull lb {lo} below baseline lb {base_lo}"
        if None not in (base_hi, hi) and hi > base_hi + SLACK:
            yield f"{where}: hull ub {hi} above baseline ub {base_hi}"

        # Masking only ever discards values inward.
        m_lo, m_hi = number(row, "masked_lb"), number(row, "masked_ub")
        if None not in (m_lo, lo) and m_lo < lo - SLACK:
            yield f"{where}: masked lb {m_lo} below unmasked lb {lo}"
        if None not in (m_hi, hi) and m_hi > hi + SLACK:
            yield f"{where}: masked ub {m_hi} above unmasked ub {hi}"

        # No undefined quantity may reach the CSV as a NaN (§2.1).
        for key, value in row.items():
            if value is None:
                continue  # short row; check_widths reports it precisely
            if value.strip().lower() in ("nan", "-nan"):
                yield f"{where}: column {key} emitted NaN rather than empty"

        # sigma = 1 is the root box and admits exactly one placement (§3.1).
        if number(row, "sigma") == 1.0 and int(row["rep"]) != 0:
            yield f"{where}: sigma=1 emitted rep {row['rep']}; it has one box"


def run(binary, model, partition, max_budget):
    """One sweep; returns (violations, row count) or exits on a hard failure."""
    completed = subprocess.run(
        [binary, f"--partition={partition}", "--sigma-rungs=2", "--reps=2",
         f"--max-budget={max_budget}", model],
        capture_output=True, text=True)
    if completed.returncode != 0:
        print(f"refinement_study --partition={partition} exited "
              f"{completed.returncode}")
        print(completed.stderr)
        return None, 0

    rows = list(csv.DictReader(completed.stdout.splitlines()))
    violations = list(check_widths(completed.stdout)) + list(check(rows))
    return violations, len(rows)


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    binary, model = argv[1], argv[2]

    # Both partitions, because they give `budget` incompatible meanings and
    # the monotonicity invariant has to hold under each. The uniform rung is
    # smaller: N there is 2^(k * n_live), so k = 2 on a seven-variable model
    # is already 2^14 regions where the budget rule's B = 6 is 64.
    failed = False
    for partition, max_budget in (("uniform", 2), ("budget", 6)):
        violations, count = run(binary, model, partition, max_budget)
        if violations is None:
            return 1
        print(f"--partition={partition}: {count} rows")
        for violation in violations:
            print(f"FAIL [{partition}]: {violation}")
        failed = failed or bool(violations)

    if failed:
        return 1

    print("ok: hull monotone in N within every parent box, "
          "hulls inside baselines, no NaN fields, both partitions")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
