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
    tools/minlp_status.py reference

A third kind of column joins those two: the bounds MINLPLib itself publishes,
which are neither measured here nor found here but downloaded, and which exist
so a recorded bound can be read against what other solvers have managed. They
have their own command because they have their own cadence -- they change when
minlplib.org changes, which is on nobody's schedule but MINLPLib's, and
fetching them needs a network that `refresh` must not start depending on.

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
import csv
import io
import math
import re
import subprocess
import sys
import urllib.request
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATUS = REPO / "MINLP_STATUS.md"
REPORT = REPO / "build" / "dev" / "gams_report"
CORPUS = Path("/vol/bitbucket/et422/minlplib_gms/minlplib/gms")

# MINLPLib's own per-instance metadata, the machine-readable form of the table
# on minlplib.org/instances.html: one semicolon-separated row per instance,
# `primalbound` and `dualbound` among some eighty columns. Preferred to
# scraping the page because it is the same numbers without an HTML parser
# between us and them.
MINLPLIB_DATA = "https://www.minlplib.org/instancedata.csv"

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
    # Left of our own numbers, and the two together, because they are what the
    # row is being read against: the interval every other solver has already
    # narrowed the optimum to, known before a run is spent and unchanged by it.
    "Ref primal",
    "Ref dual",
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

# Bounds this solver found. Only ever improve, cost a GPU run each, and are
# what a `record` accumulates.
RECORDED = ["Best primal", "Primal @", "Primal iters", "Primal params",
            "Best dual", "Dual @", "Dual iters", "Dual params"]

# Bounds MINLPLib published. Re-derivable at any time from one download, so
# unlike RECORDED they are replaceable rather than precious -- but they still
# survive a refresh, which has no network and no business clearing them.
REFERENCE = ["Ref primal", "Ref dual"]

# Everything a refresh must carry across rather than re-measure.
CARRIED = RECORDED + REFERENCE

EMPTY = "—"  # em dash: reads as "nothing here", unlike a blank cell


def die(message):
    sys.exit(f"minlp_status: {message}")


def git(*args):
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def dirty_paths():
    """Paths git reports as changed, in porcelain order. Empty when clean."""
    paths = []
    for line in git("status", "--porcelain").splitlines():
        path = line[3:].strip()
        # A rename reads "R  old -> new"; the new name is the one that exists.
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path.strip('"'))
    return paths


def affects_a_run(path):
    """Could editing this file change what a solve computes?

    A heuristic, and deliberately a generous one, because it decides whether
    the tool stops to ask: a false "yes" costs one keystroke, a false "no"
    records a bound against code nobody looked at. Anything that compiles, or
    lives where things that compile live, counts.
    """
    p = Path(path)
    if p.name in ("CMakeLists.txt", "CMakePresets.json"):
        return True
    if path.startswith(("source/", "include/", "test/", "cmake/")):
        return True
    return p.suffix in (".cu", ".cuh", ".cpp", ".hpp", ".h", ".cmake")


def current_commit():
    """Short HEAD, suffixed `-dirty` when a build input is uncommitted.

    The suffix means "this hash does not identify the code that ran", so it
    answers to the build and not to `git status`. A bound recorded while a
    design note or a stray log was unsaved *is* reproducible from the hash
    alone, and marking it otherwise spends the reader's suspicion on nothing
    -- a suffix that is always there is a suffix nobody reads.

    The trade is that the file can no longer distinguish "clean tree" from
    "dirty in ways this heuristic forgave", which is why affects_a_run stays
    biased towards saying yes: it is now the thing standing between a bound
    and a hash that overstates it.
    """
    head = git("rev-parse", "--short", "HEAD")
    dirty = any(affects_a_run(path) for path in dirty_paths())
    return head + "-dirty" if dirty else head


def confirm_dirty(what, assume_yes):
    """Show the uncommitted build inputs and ask whether to go ahead.

    The `-dirty` suffix records that a build input was uncommitted but not
    which one or what it said, and by the time anyone reads the row the answer
    is gone. Only the person at the keyboard right now can say whether the
    diff in front of them is a real change to the search or a stray printf, so
    this asks them while they can still tell.

    It asks only when there is something to ask about. A clean tree says
    nothing at all, and a tree dirty in ways that cannot reach a run -- an
    edited design note, a stray log, this file -- says one line and carries
    on. Stopping there would train the answer to be reflexive, which is the
    one way a confirmation prompt can be worse than no prompt.
    """
    paths = dirty_paths()
    if not paths:
        return

    code = [p for p in paths if affects_a_run(p)]
    other = [p for p in paths if not affects_a_run(p)]

    if not code:
        # Says which way the call went, because the interesting part is the
        # hash *without* a suffix on a tree that git calls dirty. Read as a
        # bare "-dirty is missing" that looks like a bug in the tool.
        print(f"note: {len(other)} uncommitted file(s), none of them build "
              f"inputs, so {what} is\n"
              f"      stamped `{current_commit()}` with no -dirty suffix.",
              file=sys.stderr)
        return

    def show(title, group):
        if not group:
            return
        print(f"  {title}", file=sys.stderr)
        for path in group[:12]:
            print(f"    {path}", file=sys.stderr)
        if len(group) > 12:
            print(f"    ... and {len(group) - 12} more", file=sys.stderr)

    print(f"warning: the working tree is dirty, so {what} will be stamped "
          f"`{current_commit()}` --\n"
          f"         a hash that does not identify the code that ran.",
          file=sys.stderr)
    show("changed, and could change what a run computes:", code)
    show("changed, but not build inputs:", other)

    if assume_yes:
        print("proceeding (--yes).", file=sys.stderr)
        return

    answer = ask("proceed? [y/N] ")
    if answer is None:
        # No terminal to ask on -- a cron job, or a pipeline. Refusing here
        # would break a workflow that never asked to be interactive, and the
        # `-dirty` suffix still records the fact permanently.
        print("not a terminal; proceeding without confirmation.", file=sys.stderr)
        return
    if answer.strip().lower() not in ("y", "yes"):
        die("aborted; nothing written")


def ask(prompt):
    """One line of input, or None when there is nothing to read it from.

    Three sources, in the order that gets the answer from whoever actually
    has it. The terminal first, and /dev/tty rather than stdin because
    `record --log -` takes the log on stdin -- a question asked there would be
    answered by the log. Then stdin, so a piped `yes |` still works where
    there is no terminal at all. An empty read means neither exists (a spent
    log pipe, /dev/null, a cron job) and is reported as no answer rather than
    as a refusal.
    """
    sys.stderr.write(prompt)
    sys.stderr.flush()
    if sys.stdin.isatty():
        return sys.stdin.readline()
    try:
        with open("/dev/tty") as tty:
            return tty.readline()
    except OSError:
        pass
    line = sys.stdin.readline()
    if line:
        sys.stderr.write(line if line.endswith("\n") else line + "\n")
        return line
    sys.stderr.write("\n")
    return None


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


# Deliberately unanchored at the end: the rendered line carries a coverage
# count after the date, and that count is re-derived on every write. Anchoring
# here would fail to match the very line this tool just wrote, and the stamp
# would vanish on the next `record`.
REFERENCE_AT_RE = re.compile(
    r"^- Reference bounds: `(?P<source>[^`]*)`, retrieved (?P<date>[0-9-]+)",
    re.MULTILINE)


def read_reference_at(path):
    """Where the reference bounds came from and when, or None if never fetched.

    Carried across `refresh` and `record` for the same reason as the parse
    stamp: neither command re-fetches, so re-dating the line would claim a
    download that did not happen. None means the columns are empty and the
    header line is left out entirely, which is more honest than a stamp on
    nothing.
    """
    if path.exists():
        m = REFERENCE_AT_RE.search(path.read_text())
        if m:
            return m["source"], m["date"]
    return None


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


def fetch_reference(source):
    """The raw instancedata.csv text, from a URL or a local path.

    A path is accepted so the fetch and the fill can be separated: a machine
    with no route to minlplib.org can still be given the file, and a download
    kept on disk makes `reference` repeatable against fixed input.
    """
    source = str(source)
    if source.startswith(("http://", "https://")):
        try:
            with urllib.request.urlopen(source, timeout=60) as response:
                return response.read().decode("utf-8")
        except OSError as error:
            die(f"could not fetch {source}: {error}\n"
                f"  download it by hand and pass the file to --from")
    path = Path(source)
    if not path.exists():
        die(f"{path} not found")
    return path.read_text()


def parse_reference(text):
    """Return {instance: (primal, dual, objsense)} from instancedata.csv.

    Semicolon-separated, and wide -- some eighty columns, of which four are
    read. The header is checked rather than assumed: MINLPLib is free to
    reshape its own file, and a silently missing column would fill the table
    with em dashes that look like "no bound known" instead of "not read".
    """
    reader = csv.DictReader(io.StringIO(text), delimiter=";")
    needed = ("name", "primalbound", "dualbound", "objsense")
    missing = [c for c in needed if c not in (reader.fieldnames or [])]
    if missing:
        die(f"instancedata.csv has no {', '.join(missing)} column; its format "
            f"has changed (columns seen: {', '.join(reader.fieldnames or [])})")

    data = {}
    for row in reader:
        name = (row["name"] or "").strip()
        if name:
            data[name] = (parse_number((row["primalbound"] or "").strip()),
                          parse_number((row["dualbound"] or "").strip()),
                          (row["objsense"] or "").strip())
    if not data:
        die("instancedata.csv had no rows")
    return data


# ---------------------------------------------------------------- writing

def build_row(measured, carried):
    """One table row: measured columns fresh, carried columns carried over."""
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
    for column in CARRIED:
        row[column] = (carried or {}).get(column, EMPTY) or EMPTY
    return row


def render(rows, corpus, measured_at, reference_at=None, stale=()):
    parsed = [r for r in rows if r["Parses"] == "yes"]
    solved = [r for r in parsed if r["Best primal"] != EMPTY]
    bounded = [r for r in parsed if r["Best dual"] != EMPTY]
    referenced = [r for r in rows if r["Ref primal"] != EMPTY or r["Ref dual"] != EMPTY]

    out = []
    out.append("# MINLPLib status")
    out.append("")
    out.append("Which instances this solver can attempt, and the best bounds it has")
    out.append("found on each. Generated by `tools/minlp_status.py`; edit that, not this.")
    out.append("")
    out.append(f"- Corpus: `{corpus}`")
    out.append(f"- Parse status measured at: `{measured_at[0]}` on {measured_at[1]}")
    if reference_at is not None:
        out.append(f"- Reference bounds: `{reference_at[0]}`, "
                   f"retrieved {reference_at[1]} "
                   f"({len(referenced)} of {len(rows)} instances)")
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
    out.append("**Ref primal / Ref dual** are MINLPLib's own published bounds for the")
    out.append("instance, the best any solver has reported to them, in the same sense as")
    out.append("the columns beside them. Together they bracket the optimum, so they say")
    out.append("what a run here is aiming at and what is already known to be reachable:")
    out.append("a `Best primal` no better than `Ref primal` is a solution someone else")
    out.append("already had, and one *better* than it is either a find or a bug. Equal")
    out.append("reference bounds mean the instance is solved in the literature; an empty")
    out.append("cell means MINLPLib publishes no bound of that kind, and an infinite one")
    out.append("means the instance is known infeasible.")
    out.append("")
    out.append("They also bound our own numbers from the other side, which is the cheap")
    out.append("correctness check on this table. In the instance's own sense the optimum")
    out.append("lies between the two reference bounds, so `Best primal` past `Ref dual`,")
    out.append("or `Best dual` past `Ref primal`, is a claim that contradicts the")
    out.append("literature: a bug here, or a bound recorded against the wrong row.")
    out.append("`record` says so when it writes one.")
    out.append("")
    out.append("These are downloaded, not measured -- a `refresh` neither re-fetches nor")
    out.append("clears them -- and they are only as current as the date in the header.")
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
    out.append("| `objvar-ineq` | the objective variable was eliminated through an "
               "inequality, exact at the optimum rather than pointwise; a `.lo` stated "
               "on it (minimising, or `.up` maximising) constrained nothing and was "
               "dropped |")
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
    out.append("means a **build input** was uncommitted, so the number is not")
    out.append("reproducible from that hash alone -- the shape is pinned but the code")
    out.append("that ran is not. Both commands stop and list those files before")
    out.append("stamping one, because only the person looking at the diff can say")
    out.append("whether it reaches the solver; answer `n` to go and commit first, or")
    out.append("pass `-y` to skip the question.")
    out.append("")
    out.append("Uncommitted files that cannot reach a run -- notes, logs, this file --")
    out.append("neither earn the suffix nor raise the question: a hash with no suffix")
    out.append("means the code that ran is at that commit, which is the claim worth")
    out.append("making, and it stays worth making only while it is not made about")
    out.append("every tree with an unsaved paragraph in it. What counts as a build")
    out.append("input is a heuristic in `tools/minlp_status.py`, biased towards")
    out.append("suffixing: `source/`, `include/`, `test/`, `cmake/`, anything that")
    out.append("compiles, and the CMake files.")
    out.append("")
    out.append("Two flags overrule that. `--force` overwrites a bound the run did not")
    out.append("improve; `--replace` goes further and makes the run *the* row, clearing")
    out.append("any bound it did not report rather than leaving it standing. Reach for")
    out.append("`--replace` when the recorded numbers stopped being comparable rather")
    out.append("than merely being beaten -- a commit that changed what the search does,")
    out.append("or a row recorded against the wrong shape -- because \"best ever seen\"")
    out.append("is only a useful record while every entry in it means the same thing.")
    out.append("")
    out.append("## Refreshing the reference bounds")
    out.append("")
    out.append("```")
    out.append("tools/minlp_status.py reference")
    out.append("```")
    out.append("")
    out.append("Downloads `instancedata.csv` from minlplib.org and refills the two `Ref`")
    out.append("columns from its `primalbound` and `dualbound` fields, which is the same")
    out.append("data as the table on `minlplib.org/instances.html` without an HTML parser")
    out.append("in between. `--from` takes a path instead, for a machine with no route")
    out.append("out or for a re-run against a file already downloaded. Rows MINLPLib does")
    out.append("not know, and rows whose objective sense disagrees with ours, are left")
    out.append("alone and listed -- a sense mismatch means the two names are not the same")
    out.append("model, and filling it would be worse than leaving it empty.")
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
    # Asked before the corpus run rather than before the write: the
    # measurement is minutes long, and there is no reason to spend them on a
    # stamp that is about to be abandoned.
    confirm_dirty("the parse status", args.yes)

    existing = read_status(args.status)
    measured = run_report(args.report, args.corpus, args.reject_discrete)

    rows = [build_row(m, existing.get(Path(m["file"]).stem)) for m in measured]

    seen = {r["Instance"] for r in rows}
    stale = sorted(
        name for name, row in existing.items()
        if name not in seen and any(row.get(c, EMPTY) != EMPTY for c in RECORDED)
    )

    measured_at = (current_commit(), date.today().isoformat())
    args.status.write_text(
        render(rows, args.corpus, measured_at, read_reference_at(args.status), stale))
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


REF_TOL = 1e-6


def contradictions(sense, row):
    """Ways this row's recorded bounds disagree with MINLPLib's, as messages.

    The optimum lies between the two reference bounds, so in the instance's own
    sense our primal cannot be past `Ref dual` and our dual cannot be past
    `Ref primal`. Either would mean a bound this solver cannot actually
    justify -- an infeasible point counted as a solution, a relaxation that cut
    off the optimum, or a log recorded against the wrong instance.

    Compared with a relative slack, because both sides are rounded: MINLPLib
    publishes about ten significant figures and this table stores twelve, and a
    disagreement in the last of them is arithmetic, not a bug. An infinite
    reference bound gets no slack -- it is not an approximation of anything.
    """
    sign = -1.0 if sense == "max" else 1.0  # normalise both senses to `min`
    primal = parse_number(row["Best primal"])
    dual = parse_number(row["Best dual"])
    ref_primal = parse_number(row["Ref primal"])
    ref_dual = parse_number(row["Ref dual"])

    def slack(value):
        return 0.0 if math.isinf(value) else REF_TOL * max(1.0, abs(value))

    out = []
    if primal is not None and ref_dual is not None:
        if sign * primal < sign * ref_dual - slack(ref_dual):
            out.append(f"primal {fmt(primal)} is past MINLPLib's dual bound "
                       f"{fmt(ref_dual)}, which no feasible point can be")
    if dual is not None and ref_primal is not None:
        if sign * dual > sign * ref_primal + slack(ref_primal):
            out.append(f"dual {fmt(dual)} is past MINLPLib's primal bound "
                       f"{fmt(ref_primal)}, so it cuts off a known solution")
    return out


def improvements(sense, row):
    """Ways this row's recorded bounds beat MINLPLib's, as messages.

    Not an error and not routine either: a bound tighter than the published one
    is either a result worth reporting upstream or the same bug a
    contradiction would have caught, one step short of being provable. Said out
    loud so it gets looked at either way.
    """
    sign = -1.0 if sense == "max" else 1.0
    out = []
    for kind, ours, theirs, direction in (
        ("primal", parse_number(row["Best primal"]),
         parse_number(row["Ref primal"]), 1.0),
        ("dual", parse_number(row["Best dual"]),
         parse_number(row["Ref dual"]), -1.0),
    ):
        if ours is None or theirs is None or math.isinf(theirs):
            continue
        margin = REF_TOL * max(1.0, abs(theirs))
        if direction * sign * ours < direction * sign * theirs - margin:
            out.append(f"{kind} {fmt(ours)} is tighter than MINLPLib's "
                       f"{fmt(theirs)}")
    return out


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

    # Only when the hash is being taken from this tree. An explicit --commit
    # is an assertion about some other revision, and the dirt here says
    # nothing about that one.
    #
    # Late enough that the log has already been read: `--log -` takes it on
    # stdin, and a question asked before that would be answered by the log.
    if args.commit is None:
        confirm_dirty("this bound", args.yes)

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
        if args.replace:
            # Not a comparison at all: the recorded half of the row becomes
            # this run outright. A bound this run does not have clears rather
            # than survives -- the point of asking for a replace is that the
            # old numbers are no longer to be trusted, and a leftover bound
            # from a superseded commit is exactly what would be trusted.
            row[value_col] = fmt(value)
            keep = value is not None
            row[commit_col] = commit if keep else EMPTY
            row[iters_col] = fmt_iters(iters) if keep else EMPTY
            row[params_col] = fmt_params(params) if keep else EMPTY
            if keep:
                changed.append(f"{kind} {fmt(old)} -> {fmt(value)} "
                               f"in {fmt_iters(iters)} iters (replaced)")
            else:
                changed.append(f"{kind} {fmt(old)} -> cleared, this run "
                               f"reported none (replaced)")
        elif args.force and value is not None:
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
    args.status.write_text(
        render(ordered, args.corpus, measured_at, read_reference_at(args.status)))
    print(f"{name} ({sense}): " + "; ".join(changed))

    # Checked against the row as written, not against this run's numbers, so a
    # bound that was kept rather than taken is still measured against the
    # literature. Reported after the write: the row is what it is, and burying
    # a contradiction by refusing to record it only hides the thing worth
    # seeing.
    for message in improvements(sense, row):
        print(f"note: {name} {message} -- worth checking before believing",
              file=sys.stderr)
    for message in contradictions(sense, row):
        print(f"warning: {name} {message}; one of the two is wrong",
              file=sys.stderr)


def cmd_reference(args):
    rows = read_status(args.status)
    if not rows:
        die(f"{args.status} has no table yet; run `refresh` first")

    data = parse_reference(fetch_reference(args.source))

    filled, unknown, mismatched, cleared = 0, [], [], 0
    for name, row in rows.items():
        entry = data.get(name)
        if entry is None:
            unknown.append(name)
            continue
        primal, dual, objsense = entry
        # A row whose sense disagrees is not this model under another name, so
        # its bounds are not ours to copy -- and the mistake would be invisible
        # afterwards, a plausible-looking number in the right column.
        if row["Sense"] not in (EMPTY, "") and objsense and objsense != row["Sense"]:
            mismatched.append(f"{name} (ours {row['Sense']}, theirs {objsense})")
            continue
        # Assigned unconditionally, including back to EMPTY: this command's job
        # is to make the columns say what MINLPLib says now, and a bound they
        # have withdrawn should disappear here too.
        if primal is None and row["Ref primal"] != EMPTY:
            cleared += 1
        if dual is None and row["Ref dual"] != EMPTY:
            cleared += 1
        row["Ref primal"] = fmt(primal)
        row["Ref dual"] = fmt(dual)
        filled += 1

    ordered = sorted(rows.values(), key=lambda r: r["Instance"])
    reference_at = (str(args.source), date.today().isoformat())
    args.status.write_text(
        render(ordered, args.corpus, read_measured_at(args.status), reference_at))
    print(f"{args.status}: reference bounds for {filled} of {len(rows)} instances"
          + (f", {cleared} cell(s) cleared" if cleared else ""))

    def report(title, names):
        if not names:
            return
        print(f"warning: {len(names)} {title}:", file=sys.stderr)
        for name in names[:12]:
            print(f"    {name}", file=sys.stderr)
        if len(names) > 12:
            print(f"    ... and {len(names) - 12} more", file=sys.stderr)

    report("instance(s) MINLPLib has no row for, left as they were", unknown)
    report("instance(s) whose objective sense disagrees with MINLPLib's, "
           "skipped", mismatched)

    # The whole point of the columns, so it is checked the moment they land
    # rather than waiting for the next `record` on each row.
    for row in ordered:
        sense = row["Sense"] if row["Sense"] != EMPTY else "min"
        for message in contradictions(sense, row):
            print(f"warning: {row['Instance']} {message}; one of the two is wrong",
                  file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, default=STATUS)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--yes", "-y", action="store_true",
                        help="skip the dirty-tree confirmation")
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
                                         "(default: current HEAD; also "
                                         "suppresses the dirty-tree "
                                         "confirmation, which is about this "
                                         "tree and not the revision named)")
    record.add_argument("--force", action="store_true",
                        help="overwrite even if the new bound is worse; only "
                             "touches the bounds this run reported")
    record.add_argument("--replace", action="store_true",
                        help="make this run the row outright: every recorded "
                             "cell becomes this run's, and a bound this run "
                             "did not report is cleared rather than kept. For "
                             "when the old numbers are no longer comparable "
                             "-- a commit that changed the search, a shape "
                             "recorded wrong. Stronger than --force.")
    record.set_defaults(func=cmd_record)

    reference = sub.add_parser(
        "reference", help="refill the MINLPLib reference bounds from "
                          "instancedata.csv")
    reference.add_argument("--from", dest="source", default=MINLPLIB_DATA,
                           help=f"URL or path to instancedata.csv "
                                f"(default: {MINLPLIB_DATA})")
    reference.set_defaults(func=cmd_reference)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
