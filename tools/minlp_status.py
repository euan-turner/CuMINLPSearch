#!/usr/bin/env python3
"""Maintain MINLP_STATUS.md: which MINLPLib instances this solver can attempt,
and the best bounds it has found on each.

Two halves, because the two columns are measured very differently. Whether an
instance parses is cheap, deterministic, and true of the whole corpus at once,
so `refresh` re-derives it from scratch on every run. A bound is expensive,
comes from one GPU run of one instance, and is only ever meant to improve, so
`record` accumulates it and `refresh` must carry it across untouched.

    tools/minlp_status.py refresh
    tools/minlp_status.py record <instance> --log <gams_solve output>
    tools/minlp_status.py record <instance> --primal <v> --dual <v> --iters <n>

Each recorded bound carries the commit it was found at, the number of search
iterations that run took, and the search shape it was run with, all scraped
from the log alongside the bound itself: the same number reached in fewer
iterations is progress, and the table cannot show that without recording it.

The shape is recorded as the flags that reproduce it, because none of those
four values has a fixed default -- two come from a table keyed on the model's
variable kinds and one is fitted to whatever the GPU had free at the time.
Without them a row says what was found but not how, and the run behind it
cannot be repeated.

Bounds are stored in the instance's own sense: for a `min` model the primal
bound is an upper bound on the optimum and the dual bound a lower one, and for
a `max` model it is the other way round. "Better" follows from that, and it is
why `refresh` records the sense at all -- without it, `record` could not tell
an improvement from a regression.
"""

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATUS = REPO / "MINLP_STATUS.md"
REPORT = REPO / "build" / "dev" / "gams_report"
CORPUS = Path("/vol/bitbucket/et422/minlplib_gms/minlplib/gms")

# One place the column order is written down. The row parser keys off the
# header rather than off positions, so adding a column here does not silently
# reinterpret every recorded bound in an existing file.
COLUMNS = [
    "Instance",
    "Parses",
    "Sense",
    "Vars",
    "Cons",
    "Nodes",
    "Best primal",
    "Primal @",
    "Primal iters",
    "Primal params",
    "Best dual",
    "Dual @",
    "Dual iters",
    "Dual params",
    "Notes",
]

# Only these survive a refresh; everything else is re-measured.
RECORDED = ["Best primal", "Primal @", "Primal iters", "Primal params",
            "Best dual", "Dual @", "Dual iters", "Dual params"]

EMPTY = "—"  # em dash: reads as "nothing here", unlike a blank cell


def die(message):
    sys.exit(f"minlp_status: {message}")


def git(*args):
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def current_commit():
    """Short HEAD, marked if the tree is dirty.

    A bound found against uncommitted changes is not reproducible from the
    hash alone, and silently recording the hash anyway would make the file
    claim more than it knows.
    """
    head = git("rev-parse", "--short", "HEAD")
    dirty = git("status", "--porcelain") != ""
    return head + "-dirty" if dirty else head


def fmt(value):
    return EMPTY if value is None else f"{value:.12g}"


def fmt_iters(value):
    return EMPTY if value is None else str(value)


def fmt_params(value):
    # Escaped like Notes: a pipe would end the cell early and shift every
    # column after it. Nothing gams_solve prints contains one, but --params
    # takes whatever the caller typed.
    return EMPTY if value is None else value.replace("|", "\\|")


def parse_number(text):
    if text in ("", EMPTY, "none"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_iters(text):
    if text in ("", EMPTY, "none"):
        return None
    try:
        return int(text)
    except ValueError:
        return None


# ---------------------------------------------------------------- reading

MEASURED_RE = re.compile(
    r"^- Parse status measured at: `(?P<commit>[^`]*)` on (?P<date>\S+)$", re.MULTILINE)


def read_measured_at(path):
    """The stamp on the parse column, carried across a `record` run.

    `record` rewrites the whole file but re-measures nothing, so re-stamping
    it with today's commit would claim a corpus run that never happened.
    """
    if path.exists():
        m = MEASURED_RE.search(path.read_text())
        if m:
            return m["commit"], m["date"]
    return current_commit(), date.today().isoformat()


def read_status(path):
    """Return {instance: {column: cell}} for the rows already in the file.

    Tolerates the file not existing yet -- the first refresh has nothing to
    carry across, which is not an error.
    """
    if not path.exists():
        return {}

    rows = {}
    header = None
    for line in path.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            if cells and cells[0] == "Instance":
                header = cells
            continue
        if all(set(c) <= {"-", ":"} for c in cells):  # the |---| separator
            continue
        if len(cells) != len(header):
            continue
        row = dict(zip(header, cells))
        # A file written before a column existed is still readable; the new
        # cell is simply empty until something records it. Filled here rather
        # than at each use so `record` can rewrite an old file without
        # tripping over the column it is adding.
        for column in COLUMNS:
            row.setdefault(column, EMPTY)
        rows[row["Instance"]] = row
    return rows


def run_report(report, corpus, reject_discrete=False):
    """Measure the corpus. Returns rows in gams_report's own order (sorted)."""
    if not report.exists():
        die(f"{report} not found; build it with "
            f"`cmake --build --preset=dev --target gams_report`")
    if not corpus.exists():
        die(f"corpus {corpus} not found; pass --corpus")

    cmd = [str(report), "--per-instance"]
    if reject_discrete:
        cmd.append("--reject-discrete")
    cmd.append(str(corpus))
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        die(f"gams_report exited {out.returncode}: {out.stderr.strip()}")

    lines = out.stdout.splitlines()
    if not lines:
        die("gams_report produced no output")
    fields = lines[0].split("\t")
    return [dict(zip(fields, line.split("\t"))) for line in lines[1:] if line]


# ---------------------------------------------------------------- writing

def build_row(measured, carried):
    """One table row: measured columns fresh, recorded columns carried over."""
    parsed = measured["status"] == "parsed"
    row = {
        "Instance": Path(measured["file"]).stem,
        "Parses": "yes" if parsed else "no",
        "Sense": measured["sense"] or EMPTY,
        "Vars": measured["vars"] or EMPTY,
        "Cons": measured["constraints"] or EMPTY,
        "Nodes": measured["nodes"] or EMPTY,
        # A pipe inside a reason would end the cell early and shift every
        # column after it, so it is escaped rather than trusted.
        "Notes": (measured["flags"] if parsed
                  else f"{measured['kind']}: {measured['reason']}").replace("|", "\\|")
                 or EMPTY,
    }
    for column in RECORDED:
        row[column] = (carried or {}).get(column, EMPTY) or EMPTY
    return row


def render(rows, corpus, measured_at, stale=()):
    parsed = [r for r in rows if r["Parses"] == "yes"]
    solved = [r for r in parsed if r["Best primal"] != EMPTY]
    bounded = [r for r in parsed if r["Best dual"] != EMPTY]

    out = []
    out.append("# MINLPLib status")
    out.append("")
    out.append("Which instances this solver can attempt, and the best bounds it has")
    out.append("found on each. Generated by `tools/minlp_status.py`; edit that, not this.")
    out.append("")
    out.append(f"- Corpus: `{corpus}`")
    out.append(f"- Parse status measured at: `{measured_at[0]}` on {measured_at[1]}")
    out.append(f"- Instances: **{len(rows)}** total, "
               f"**{len(parsed)}** parse ({100.0 * len(parsed) / max(len(rows), 1):.1f}%), "
               f"**{len(rows) - len(parsed)}** rejected")
    out.append(f"- Bounds recorded: **{len(solved)}** with a primal bound, "
               f"**{len(bounded)}** with a dual bound")
    out.append("")
    out.append("## Reading the table")
    out.append("")
    out.append("**Parses** is re-measured on every refresh and describes the frontend")
    out.append("only: `yes` means `gams_solve` will get as far as a `Problem`, not that")
    out.append("the search converges or that it converges in reasonable time.")
    out.append("")
    out.append("**Best primal / Best dual** are in the instance's own sense, so for a")
    out.append("`min` row `dual <= optimum <= primal` and for a `max` row it is")
    out.append("reversed. The primal bound is the best feasible objective value the")
    out.append("search actually attained; the dual bound is what the interval relaxation")
    out.append("proved. Equal bounds mean the instance was solved to tolerance. Each has")
    out.append("its own `@` column, the commit it was found at, because the two rarely")
    out.append("improve in the same run.")
    out.append("")
    out.append("**Primal iters / Dual iters** is how many search iterations the run")
    out.append("that produced that bound took -- the last `iter N:` the driver printed,")
    out.append("which is the iteration limit unless the search converged first. It")
    out.append("describes the recorded run, not the cheapest way to reach the number: a")
    out.append("bound is only displaced by a better bound, or by an equal one reached in")
    out.append("fewer iterations.")
    out.append("")
    out.append("**Primal params / Dual params** is the search shape that run used,")
    out.append("written as the flags that set it. Every one of those four has a default")
    out.append("that moves -- two are read off a table keyed on the model's variable")
    out.append("kinds, and `--max-cycle-size` is fitted to whatever the GPU had free --")
    out.append("so re-running the bare command is not the same experiment. Pasting the")
    out.append("cell back pins all four:")
    out.append("")
    out.append("```")
    out.append("gams_solve <corpus>/<instance>.gms <iters> <params>")
    out.append("```")
    out.append("")
    out.append("with `<iters>` the matching iteration count from the column beside it.")
    out.append("That reproduces the run whether it converged or hit the limit: a run")
    out.append("that converged at iteration N converges at N under a limit of N too.")
    out.append("An empty cell means the bound predates this column, not that the run")
    out.append("used no flags.")
    out.append("")
    out.append("**Notes** carries the rejection reason for a `no` row. For a `yes` row it")
    out.append("carries quality caveats that do not stop a solve but should temper trust")
    out.append("in its bounds:")
    out.append("")
    out.append("| Note | Meaning |")
    out.append("| --- | --- |")
    out.append("| `objvar-kept` | the objective variable could not be eliminated, so it "
               "remains a search dimension tied by an equality |")
    out.append("| `default-bound` | some free variable was given the artificial "
               "`ParseOptions::default_bound` box |")
    out.append("| `default-bound-integer` | as above, but on an integer/binary variable, "
               "where a 2e6-wide box is a search-quality cliff |")
    out.append("")
    out.append("## Recording a result")
    out.append("")
    out.append("```")
    out.append("gams_solve <corpus>/<instance>.gms <iterations> | tee run.log")
    out.append("tools/minlp_status.py record <instance> --log run.log")
    out.append("```")
    out.append("")
    out.append("`record` keeps whichever bound is better and stamps the commit, the")
    out.append("iteration count and the search shape it came from, so re-running a worse")
    out.append("configuration cannot lose ground. The count is scraped from the log's")
    out.append("`iter` lines and the shape from its `PARAMS` line; `--iters` and")
    out.append("`--params` set them by hand, which is the only way to record either")
    out.append("alongside a manual `--primal`/`--dual`. A `-dirty` suffix on a hash")
    out.append("means the tree had uncommitted changes and the number is not")
    out.append("reproducible from that hash alone -- the shape is pinned but the code")
    out.append("that ran is not.")
    out.append("")

    if stale:
        # Loud, because the alternative is a bound quietly outliving the
        # instance it was measured on.
        out.append("## Recorded bounds with no matching instance")
        out.append("")
        out.append("These carried bounds no longer correspond to any file in the corpus")
        out.append("(renamed, removed, or a typo in a `record` call). They are dropped")
        out.append("from the table below; re-record them under the right name.")
        out.append("")
        for name in stale:
            out.append(f"- `{name}`")
        out.append("")

    out.append("## Instances")
    out.append("")
    out.append("| " + " | ".join(COLUMNS) + " |")
    out.append("| " + " | ".join("---" for _ in COLUMNS) + " |")
    for row in rows:
        out.append("| " + " | ".join(row[c] for c in COLUMNS) + " |")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------- commands

def cmd_refresh(args):
    existing = read_status(args.status)
    measured = run_report(args.report, args.corpus, args.reject_discrete)

    rows = [build_row(m, existing.get(Path(m["file"]).stem)) for m in measured]

    seen = {r["Instance"] for r in rows}
    stale = sorted(
        name for name, row in existing.items()
        if name not in seen and any(row.get(c, EMPTY) != EMPTY for c in RECORDED)
    )

    measured_at = (current_commit(), date.today().isoformat())
    args.status.write_text(render(rows, args.corpus, measured_at, stale))
    parsed = sum(1 for r in rows if r["Parses"] == "yes")
    print(f"{args.status}: {len(rows)} instances, {parsed} parse, "
          f"{len(rows) - parsed} rejected")
    if stale:
        print(f"warning: {len(stale)} recorded bound(s) no longer match an instance: "
              f"{', '.join(stale)}", file=sys.stderr)


RESULT_RE = re.compile(
    r"^RESULT\tsense=(?P<sense>min|max)\tprimal=(?P<primal>\S+)\tdual=(?P<dual>\S+)$",
    re.MULTILINE,
)

# The driver numbers every iteration it runs, whether the box was fathomed by
# enumeration or expanded, so the highest number printed before a RESULT is
# how many iterations that run took. Cheaper than teaching gams_solve to put
# the count on the RESULT line, and it works on logs already on disk.
ITER_RE = re.compile(r"^iter (?P<n>\d+):", re.MULTILINE)

# gams_solve prints this once, after resolving every default, before the
# search. `shape=` is on the line too but is not recorded: it only selects
# defaults for the four values below, so once those are pinned it changes
# nothing, and a cell that repeats it would invite reproducing a run by the
# shape name instead of by the numbers it happened to imply that day.
PARAMS_RE = re.compile(r"^PARAMS\t(?P<fields>\S.*)$", re.MULTILINE)

# Recorded as flags rather than as bare numbers so a cell can be pasted onto a
# gams_solve command line unedited. The key is the name gams_solve prints; the
# flag is the one that sets it, and they are deliberately the same word.
PARAM_FLAGS = (
    ("partition_num", "--partition-num"),
    ("enumerate_cap", "--enumerate-cap"),
    ("sample_points", "--sample-points"),
    ("max_cycle_size", "--max-cycle-size"),
)


def scrape_params(text):
    """The search shape of the run in `text`, as a pasteable flag string.

    None when the log has no usable PARAMS line -- a log from a build before
    this was printed, or a truncated one. That records as "unknown", which is
    what it is; refusing the whole record would throw away a real bound over a
    missing annotation.
    """
    matches = list(PARAMS_RE.finditer(text))
    if not matches:
        return None
    fields = {}
    for field in matches[-1]["fields"].split("\t"):
        key, sep, value = field.partition("=")
        if sep:
            fields[key] = value
    # All four or nothing: a partial shape reads as reproducible while leaving
    # the reader to guess which default filled the gap, and the guess depends
    # on a GPU they do not have.
    if any(key not in fields for key, _ in PARAM_FLAGS):
        return None
    return " ".join(f"{flag}={fields[key]}" for key, flag in PARAM_FLAGS)


def scrape_log(text):
    """Pull the last RESULT line, and the iteration count that produced it.

    The last, not the first: a log may hold several runs appended, and the
    most recent one is the one being recorded. The iteration count is taken
    from that run only -- the `iter` lines after the last RESULT belong to a
    later, unfinished run, and the ones before the second-to-last belong to a
    run whose bounds are not the ones being recorded.

    A run that reached RESULT without printing an iteration (the whole search
    fitting in the root launch) yields None, which records as "no count" and
    not as zero.

    The search shape is windowed the same way, and for the same reason:
    gams_solve prints it before its own search, so the PARAMS line inside the
    window is the one belonging to this run and not to the run before it.
    """
    matches = list(RESULT_RE.finditer(text))
    if not matches:
        die("no RESULT line in the log; is it gams_solve output from a run that "
            "reached the end of the search?")
    m = matches[-1]
    start = matches[-2].end() if len(matches) > 1 else 0
    window = text[start:m.start()]
    iters = [int(i["n"]) for i in ITER_RE.finditer(window)]
    return m["sense"], parse_number(m["primal"]), parse_number(m["dual"]), \
        (max(iters) if iters else None), scrape_params(window)


def better(kind, sense, new, old):
    """Is `new` an improvement on `old` for this bound in this sense?"""
    if new is None:
        return False
    if old is None:
        return True
    tighter_is_smaller = (kind == "primal") == (sense == "min")
    return new < old if tighter_is_smaller else new > old


def cheaper(new_iters, old_iters):
    """Reaching the same bound in fewer iterations is also a result worth
    keeping, so an equal bound still displaces the record when its run was
    shorter. Only ever consulted for a tie: a worse bound found faster is
    still a worse bound."""
    return new_iters is not None and (old_iters is None or new_iters < old_iters)


def cmd_record(args):
    measured_at = read_measured_at(args.status)
    rows = read_status(args.status)
    if not rows:
        die(f"{args.status} has no table yet; run `refresh` first")

    name = Path(args.instance).stem
    row = rows.get(name)
    if row is None:
        die(f"no instance '{name}' in {args.status}; "
            f"check the name, or run `refresh` if the corpus changed")
    if row["Parses"] != "yes":
        die(f"'{name}' does not parse ({row['Notes']}); there is no bound to record")

    if args.log is not None:
        text = sys.stdin.read() if str(args.log) == "-" else Path(args.log).read_text()
        log_sense, primal, dual, iters, params = scrape_log(text)
        # An explicit --iters wins: the log's count is an inference from the
        # printed trace, and the caller may know better (a truncated log, or a
        # run whose trace was not captured).
        if args.iters is not None:
            iters = args.iters
        if args.params is not None:
            params = args.params
        elif params is None:
            # Warned about rather than passed over: the row will claim a bound
            # nobody can reproduce, and that is worth one line of noise.
            print(f"warning: no PARAMS line in the log, so the search shape "
                  f"behind this bound is not recorded; pass --params to supply "
                  f"it, or re-run with a build that prints it", file=sys.stderr)
        # A sense mismatch means the log is from a different instance than the
        # one named. Recording it would attach real numbers to the wrong row,
        # which is worse than recording nothing.
        if row["Sense"] not in (EMPTY, log_sense):
            die(f"log says the model is a {log_sense} but '{name}' is a "
                f"{row['Sense']}; is this log from a different instance?")
    else:
        primal, dual, iters, params = args.primal, args.dual, args.iters, args.params
    if primal is None and dual is None:
        die("nothing to record: pass --log, or --primal and/or --dual")

    sense = row["Sense"] if row["Sense"] != EMPTY else "min"
    commit = args.commit or current_commit()

    changed = []
    for kind, value, value_col, commit_col, iters_col, params_col in (
        ("primal", primal, "Best primal", "Primal @", "Primal iters",
         "Primal params"),
        ("dual", dual, "Best dual", "Dual @", "Dual iters", "Dual params"),
    ):
        old = parse_number(row[value_col])
        old_iters = parse_iters(row[iters_col])
        # Compared through fmt, not as floats: the stored cell has been
        # rounded to the table's precision, so the run that produced it does
        # not reproduce it bit for bit.
        # A tie is decided at the table's precision, and before `better`,
        # because the stored cell was rounded on the way in: re-recording the
        # very same run can come back a rounding step tighter, which `better`
        # reads as an improvement. Left in that order, a longer run would
        # overwrite a shorter one's iteration count on a bound that did not
        # actually move.
        tied = value is not None and old is not None and fmt(value) == fmt(old)
        # The shape travels with the commit and the count, always: the three
        # together describe one run, and leaving a stale shape beside a new
        # bound would describe a run that never happened.
        if args.force and value is not None:
            row[value_col] = fmt(value)
            row[commit_col] = commit
            row[iters_col] = fmt_iters(iters)
            row[params_col] = fmt_params(params)
            changed.append(f"{kind} {fmt(old)} -> {fmt(value)} "
                           f"in {fmt_iters(iters)} iters (forced)")
        elif tied:
            if cheaper(iters, old_iters):
                row[commit_col] = commit
                row[iters_col] = fmt_iters(iters)
                row[params_col] = fmt_params(params)
                changed.append(f"{kind} {fmt(value)} matched in fewer iters, "
                               f"{fmt_iters(old_iters)} -> {fmt_iters(iters)}")
            else:
                changed.append(f"{kind} {fmt(value)} matched in "
                               f"{fmt_iters(iters)} iters, not fewer than "
                               f"{fmt_iters(old_iters)}, kept")
        elif better(kind, sense, value, old):
            row[value_col] = fmt(value)
            row[commit_col] = commit
            row[iters_col] = fmt_iters(iters)
            row[params_col] = fmt_params(params)
            changed.append(f"{kind} {fmt(old)} -> {fmt(value)} "
                           f"in {fmt_iters(iters)} iters")
        elif value is not None:
            changed.append(f"{kind} {fmt(value)} not better than {fmt(old)}, kept")

    ordered = sorted(rows.values(), key=lambda r: r["Instance"])
    args.status.write_text(render(ordered, args.corpus, measured_at))
    print(f"{name} ({sense}): " + "; ".join(changed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, default=STATUS)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    sub = parser.add_subparsers(dest="command", required=True)

    refresh = sub.add_parser(
        "refresh", help="re-measure the parse status of the whole corpus")
    refresh.add_argument("--report", type=Path, default=REPORT)
    refresh.add_argument("--reject-discrete", action="store_true",
                         help="reproduce the pre-integrality baseline")
    refresh.set_defaults(func=cmd_refresh)

    record = sub.add_parser("record", help="record bounds for one instance")
    record.add_argument("instance")
    record.add_argument("--log", help="gams_solve output to scrape ('-' for stdin)")
    record.add_argument("--primal", type=float)
    record.add_argument("--dual", type=float)
    record.add_argument("--iters", type=int,
                        help="iterations the run took (default: scraped from "
                             "--log; required to record a count without one)")
    record.add_argument("--params",
                        help="search-shape flags the run used, e.g. "
                             "'--partition-num=7 --sample-points=5' (default: "
                             "scraped from --log's PARAMS line)")
    record.add_argument("--commit", help="override the recorded hash "
                                         "(default: current HEAD)")
    record.add_argument("--force", action="store_true",
                        help="overwrite even if the new bound is worse")
    record.set_defaults(func=cmd_record)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
