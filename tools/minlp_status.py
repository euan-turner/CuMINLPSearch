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
iterations that run took, and the policy it was run under, all scraped from
the log alongside the bound itself: the same number reached in fewer
iterations is progress, and the table cannot show that without recording it.

The policy is recorded as its name, not as the shape it resolved to. With
partition_num and max_cycle_size now fitted to whatever the GPU had free at
the time, pasting the resolved numbers back onto a different GPU reproduces a
shape but not the decision -- and the decision, paired with the commit hash
(the roster lives in include/, so a rule change is a differing hash), is what
the row is a record of. `--policy=<name>` is what the cell pastes back onto a
command line; an experimental run's overrides ride alongside it.

Bounds are stored in the instance's own sense: for a `min` model the primal
bound is an upper bound on the optimum and the dual bound a lower one, and for
a `max` model it is the other way round. "Better" follows from that, and it is
why `refresh` records the sense at all -- without it, `record` could not tell
an improvement from a regression.

A run that never samples a feasible point reports no primal bound at all, and
that is a result rather than the absence of one. Which result depends on how
it ended: with a frontier it emptied by proving every region held nothing, it
is a proof that the instance has none, and with places left to look -- regions
still pending, or regions discarded to stay inside its memory budget -- it is a
sampling failure. The primal column carries both, as `infeasible` and
`no sample` -- see the comment on INFEASIBLE.
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
    "Primal policy",
    "Best dual",
    "Dual @",
    "Dual iters",
    "Dual policy",
    "Notes",
]

# Bounds this solver found. Only ever improve, cost a GPU run each, and are
# what a `record` accumulates.
RECORDED = ["Best primal", "Primal @", "Primal iters", "Primal policy",
            "Best dual", "Dual @", "Dual iters", "Dual policy"]

# Bounds MINLPLib published. Re-derivable at any time from one download, so
# unlike RECORDED they are replaceable rather than precious -- but they still
# survive a refresh, which has no network and no business clearing them.
REFERENCE = ["Ref primal", "Ref dual"]

# Everything a refresh must carry across rather than re-measure.
CARRIED = RECORDED + REFERENCE

EMPTY = "—"  # em dash: reads as "nothing here", unlike a blank cell

# The two things a run can report in place of a primal bound, and they are not
# the same thing at all -- which is the whole reason both exist.
#
# A search that never samples a feasible point ends one of two ways. If the
# instance really has no feasible point, every region is eventually discarded
# by the interval feasibility test (or fathomed by an exhaustive enumeration
# that finds nothing) and the frontier *empties*: the search ran out of places
# a solution could hide, which is a proof. If instead the instance is feasible
# and only the point sampling keeps missing -- the usual fate of equality
# constraints, whose feasible set the sampler has measure-zero odds of landing
# on -- then no region is ever discarded, the frontier keeps growing, and the
# run ends at its iteration limit with regions still pending.
#
# A third way to empty a frontier arrived with the host memory budget: a run
# may discard pending regions to stay inside it. Those regions are places
# nobody looked, so a frontier emptied with any of them dropped proves nothing
# and reads as NO_SAMPLE too.
#
# So the terminal state separates them, and the log states it outright: a
# pending size of zero, with no viable region dropped and no incumbent, is
# INFEASIBLE, and anything else is NO_SAMPLE. Neither is a number and neither
# is ever compared as one: the first is a claim about the instance rather than
# a value the objective attains, and the second is a claim about the run.
INFEASIBLE = "infeasible"
NO_SAMPLE = "no sample"

# Ranked, because they share the primal column with each other and with real
# bounds, and something has to say which of two runs the row keeps. A feasible
# point outranks a proof there is none (they contradict, and the point is the
# thing you can hold); a proof outranks a run that merely didn't find one.
PRIMAL_RANK = {NO_SAMPLE: 1, INFEASIBLE: 2}


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


def fmt_policy(value):
    # Escaped like Notes: a pipe would end the cell early and shift every
    # column after it. Nothing gams_solve prints contains one, but --policy
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
    # An outcome is not a bound, so it is not counted as one: a row that found
    # nothing to attain has not been solved, and folding it into the solved
    # count would make the search look better the more instances it failed to
    # find a point in.
    solved = [r for r in parsed
              if r["Best primal"] not in (EMPTY, INFEASIBLE, NO_SAMPLE)]
    infeasible = [r for r in parsed if r["Best primal"] == INFEASIBLE]
    unsampled = [r for r in parsed if r["Best primal"] == NO_SAMPLE]
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
               f"**{len(bounded)}** with a dual bound"
               + (f", **{len(infeasible)}** shown infeasible" if infeasible else "")
               + (f", **{len(unsampled)}** with no feasible point sampled"
                  if unsampled else ""))
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
    out.append("**Best primal** holds two things that are not bounds, for the two ways")
    out.append("a run can find no feasible point at all. They are not the same result")
    out.append("and the run says which:")
    out.append("")
    out.append("| Cell | Meaning |")
    out.append("| --- | --- |")
    out.append(f"| `{INFEASIBLE}` | the frontier emptied: every region was discarded by "
               "the interval feasibility test or fathomed by an exhaustive enumeration "
               "that found nothing, so no feasible point exists to be found |")
    out.append(f"| `{NO_SAMPLE}` | the run stopped with places left to look and no "
               "feasible point sampled -- regions still pending at the iteration limit, "
               "or regions discarded to stay inside the host memory budget; the instance "
               "may well be feasible and the sampler simply missing, which is the "
               "ordinary outcome for equality constraints |")
    out.append("")
    out.append("The difference is the terminal state, not a judgement call. An")
    out.append("infeasible instance runs out of places a solution could hide, so given")
    out.append("iterations enough it *ends*; a sampler that keeps missing keeps")
    out.append("splitting regions instead, and can only ever end by running out of")
    out.append("iterations or of memory. `record` reads the pending count and the count")
    out.append("of regions dropped for memory off the log's summary and writes whichever")
    out.append("applies -- a frontier emptied partly by discarding regions proves")
    out.append("nothing, because a discarded region is a place nobody looked.")
    out.append("")
    out.append("Neither is compared as a number. Any feasible point displaces both,")
    out.append(f"whatever its objective value -- against `{INFEASIBLE}` that is a")
    out.append("contradiction rather than an improvement, and `record` says so, keeping")
    out.append("the point because a point is the half of the pair that can be checked.")
    out.append(f"`{INFEASIBLE}` displaces `{NO_SAMPLE}`, being the stronger claim, and")
    out.append("never the reverse. The `@`, `iters` and `policy` columns beside them")
    out.append("describe the run that got there, and matter more here than for a bound:")
    out.append(f"`{NO_SAMPLE}` is a statement about a policy and a length of search, so")
    out.append("it is displaced by the same outcome from a *longer* run, which says")
    out.append("strictly more -- the one cell in this table where a longer run wins.")
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
    out.append(f"They settle an `{INFEASIBLE}` the same way, and it is the one entry")
    out.append("here they can settle outright. A published `Ref primal` *is* a feasible")
    out.append("point, so it says the search lost one; an infinite `Ref dual` says")
    out.append(f"MINLPLib found the instance infeasible too. (`{NO_SAMPLE}` claims")
    out.append("nothing about the instance, so there is nothing for them to settle.)")
    out.append("Both are reported when the row is written.")
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
    out.append("**Primal policy / Dual policy** is the policy that run used, as a")
    out.append("pasteable `--policy=<name>` -- and, for an experimental run, whatever")
    out.append("`--partition-num`/`--enumerate-cap`/`--sample-points`/`--max-cycle-size`")
    out.append("overrides rode alongside it. The policy name, not the four resolved")
    out.append("numbers, is the reproducible unit: partition_num and max_cycle_size are")
    out.append("now fitted to whatever the GPU had free at the time, so pasting them back")
    out.append("reproduces a shape but not the decision, while the policy name paired")
    out.append("with the commit column reproduces the decision exactly. Pasting the cell")
    out.append("back reruns the same policy:")
    out.append("")
    out.append("```")
    out.append("gams_solve <corpus>/<instance>.gms <iters> <policy>")
    out.append("```")
    out.append("")
    out.append("with `<iters>` the matching iteration count from the column beside it.")
    out.append("That reproduces the run whether it converged or hit the limit: a run")
    out.append("that converged at iteration N converges at N under a limit of N too.")
    out.append("An empty cell means the bound predates this column, not that the run")
    out.append("used no policy.")
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
    out.append("iteration count and the policy it came from, so re-running a worse")
    out.append("configuration cannot lose ground. The count is scraped from the log's")
    out.append("`iter` lines and the policy from its `PARAMS` line; `--iters` and")
    out.append("`--policy` set them by hand, which is the only way to record either")
    out.append("alongside a manual `--primal`/`--dual`. A run that found no feasible")
    out.append("point is classified from the same log, by the pending count in its")
    out.append(f"summary; `--infeasible` records an `{INFEASIBLE}` by hand, and a log")
    out.append("that reports regions still pending refuses it.")
    out.append("A `-dirty` suffix on a hash")
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
    out.append("or a row recorded against the wrong policy -- because \"best ever seen\"")
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

# gams_solve prints this once, after resolving the policy, before the search.
# `source=` is on the line too (auto/named/overridden) but is not recorded
# itself: `--policy=<name>` reproduces a `named` run outright, and an
# `overridden` run is exactly `--policy=<name>` plus whatever `overrides=`
# lists, so the source word adds nothing a reader could paste back that the
# other two fields don't already say.
PARAMS_RE = re.compile(r"^PARAMS\t(?P<fields>\S.*)$", re.MULTILINE)

# How many regions the search still had when it stopped, from the summary the
# driver prints just above the RESULT line. Paired with `primal=none` this is
# the whole discriminator between the two ways of finding no feasible point:
# zero means the frontier emptied, nonzero means the run stopped with places
# left to look.
#
# Preferred to the driver's "Search space exhausted" notice, which is printed
# on exactly this branch and says the same thing, but is a sentence rather than
# an interface. The notice is kept as a fallback only, for a log too old or too
# truncated to carry the summary.
PENDING_RE = re.compile(r"^Pending size: (?P<n>\d+)$", re.MULTILINE)

# How many still-viable regions the search threw away to stay inside its host
# memory budget, from the same summary. It is what stops an emptied frontier
# from being read as a proof: a region evicted while it could still have held
# the optimum is an unexplored place, not an excluded one, so a run that
# emptied its frontier partly by eviction has not shown anything about the
# instance. See design/BOUNDED_FRONTIER.md §4.
#
# Absent from a log written before the driver could drop anything, and that
# reads correctly as zero -- those runs never dropped a region because they
# had no mechanism to.
DROPPED_RE = re.compile(r"^Dropped viable regions: (?P<n>\d+)$", re.MULTILINE)

# Two words, because the sentence after them has already been reworded once:
# the old form went on "...with no feasible point sampled; either the problem
# is infeasible or point sampling never satisfied the constraints", hedging
# between the two outcomes this tool now separates. Logs of both vintages are
# on disk and both mean the frontier emptied, which is all this is asked.
EXHAUSTED_RE = re.compile(r"^Search space exhausted", re.MULTILINE)

# The experimental-override keys gams_solve's `overrides=` field can list, and
# the flag that sets each -- deliberately the same word, so the mapping is
# just spelling. Only ever appended to a scraped policy cell; an ordinary
# auto/named run has no `overrides=` field at all.
OVERRIDE_FLAGS = {
    "partition_num": "--partition-num",
    "enumerate_cap": "--enumerate-cap",
    "sample_points": "--sample-points",
    "max_cycle_size": "--max-cycle-size",
}


def scrape_policy(text):
    """The policy (and any overrides) of the run in `text`, as a pasteable
    `--policy=<name> [--flag=value ...]` string.

    None when the log has no usable `policy=` field -- a log from a build
    before this was printed (including one from before the policy-selection
    change, whose PARAMS line carried `shape=` instead), or a truncated one.
    That records as "unknown", which is what it is; refusing the whole record
    would throw away a real bound over a missing annotation.
    """
    matches = list(PARAMS_RE.finditer(text))
    if not matches:
        return None
    fields = {}
    for field in matches[-1]["fields"].split("\t"):
        key, sep, value = field.partition("=")
        if sep:
            fields[key] = value
    policy = fields.get("policy")
    if policy is None:
        return None
    cell = f"--policy={policy}"
    for pair in fields.get("overrides", "").split(","):
        key, sep, value = pair.partition("=")
        flag = OVERRIDE_FLAGS.get(key)
        if sep and flag:
            cell += f" {flag}={value}"
    return cell


def primal_outcome(primal, pending, said_exhausted, dropped=None):
    """What a run with no primal bound found: INFEASIBLE, NO_SAMPLE, or None.

    None when the log does not say -- no summary line and no exhaustion notice,
    which is a log from a build before either existed. The two outcomes are far
    enough apart that guessing between them is worse than recording neither.

    An empty frontier is only a proof when the search emptied it by *proving*
    every region held nothing. A memory-bounded run can also empty it by
    discarding regions, and those regions are places left to look, so a run
    that dropped a viable one records as NO_SAMPLE however few regions it
    finished with. `dropped` of None is an old log, from a driver that could
    not drop anything, and keeps today's reading byte for byte.
    """
    if primal is not None:
        return None
    if pending is not None:
        return INFEASIBLE if pending == 0 and not dropped else NO_SAMPLE
    return INFEASIBLE if said_exhausted else None


def scrape_log(text):
    """Pull the last RESULT line, the iteration count that produced it, and
    how the search ended if it ended with no feasible point.

    The last, not the first: a log may hold several runs appended, and the
    most recent one is the one being recorded. The iteration count is taken
    from that run only -- the `iter` lines after the last RESULT belong to a
    later, unfinished run, and the ones before the second-to-last belong to a
    run whose bounds are not the ones being recorded.

    A run that reached RESULT without printing an iteration (the whole search
    fitting in the root launch) yields None, which records as "no count" and
    not as zero.

    The policy and the terminal summary are windowed the same way, and for the
    same reason: gams_solve prints both around its own search, so the lines
    inside the window belong to this run and not to the run before it.
    """
    matches = list(RESULT_RE.finditer(text))
    if not matches:
        die("no RESULT line in the log; is it gams_solve output from a run that "
            "reached the end of the search?")
    m = matches[-1]
    start = matches[-2].end() if len(matches) > 1 else 0
    window = text[start:m.start()]
    iters = [int(i["n"]) for i in ITER_RE.finditer(window)]
    pendings = [int(p["n"]) for p in PENDING_RE.finditer(window)]
    droppeds = [int(d["n"]) for d in DROPPED_RE.finditer(window)]
    primal = parse_number(m["primal"])
    pending = pendings[-1] if pendings else None
    dropped = droppeds[-1] if droppeds else None
    said_exhausted = EXHAUSTED_RE.search(window) is not None
    # Both readings of the same run, and they cannot disagree unless the driver
    # has changed under this tool: the notice is printed on exactly the branch
    # the summary reports as empty. Said rather than resolved -- whichever one
    # is now wrong, the other is not automatically right.
    if said_exhausted and pending:
        print(f"warning: the log says the search space was exhausted but also "
              f"reports {pending} pending region(s); the summary is believed "
              f"and this is recorded as a sampling failure, but one of the two "
              f"lines is now wrong", file=sys.stderr)
    return m["sense"], primal, parse_number(m["dual"]), \
        (max(iters) if iters else None), scrape_policy(window), \
        primal_outcome(primal, pending, said_exhausted, dropped)


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


def longer(new_iters, old_iters):
    """The reverse, for `no sample`: there the iteration count is not the cost
    of a result but the extent of the search that failed to produce one, so the
    longer run is the stronger statement and the one worth keeping."""
    return new_iters is not None and (old_iters is None or new_iters > old_iters)


REF_TOL = 1e-6


def contradictions(sense, row):
    """Ways this row's recorded bounds disagree with MINLPLib's, as messages.

    The optimum lies between the two reference bounds, so in the instance's own
    sense our primal cannot be past `Ref dual` and our dual cannot be past
    `Ref primal`. Either would mean a bound this solver cannot actually
    justify -- an infeasible point counted as a solution, a relaxation that cut
    off the optimum, or a log recorded against the wrong instance.

    An `infeasible?` primal is checked the same way, against the same fact: a
    published primal bound is a feasible point, and a feasible point is exactly
    what that cell claims does not exist.

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
    if row["Best primal"] == INFEASIBLE and ref_primal is not None \
            and math.isfinite(ref_primal):
        # They hold a feasible point with that value, so the space this search
        # emptied was not empty: the exhaustion is a lost point (sampling that
        # never satisfied an equality, or a pruning bug), not a property of the
        # instance.
        out.append(f"is recorded as `{INFEASIBLE}`, but MINLPLib publishes a "
                   f"primal bound {fmt(ref_primal)}, so a feasible point "
                   f"exists")
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
        log_sense, primal, dual, iters, policy, outcome = scrape_log(text)
        if args.infeasible:
            # An assertion about a run the log describes, so the log gets to
            # refuse it: a nonzero pending size is the run saying it stopped
            # with places left to look, which is not a proof of anything.
            if outcome == NO_SAMPLE:
                die("the log reports regions still pending, or regions dropped "
                    "to stay inside the host memory budget, so this run "
                    "stopped with places left to look rather than exhausting "
                    "the search space; it does not show the instance is "
                    "infeasible")
            outcome = INFEASIBLE
        # An explicit --iters wins: the log's count is an inference from the
        # printed trace, and the caller may know better (a truncated log, or a
        # run whose trace was not captured).
        if args.iters is not None:
            iters = args.iters
        if args.policy is not None:
            policy = args.policy
        elif policy is None:
            # Warned about rather than passed over: the row will claim a bound
            # nobody can reproduce, and that is worth one line of noise.
            print(f"warning: no PARAMS line in the log, so the policy behind "
                  f"this bound is not recorded; pass --policy to supply it, "
                  f"or re-run with a build that prints it", file=sys.stderr)
        # A sense mismatch means the log is from a different instance than the
        # one named. Recording it would attach real numbers to the wrong row,
        # which is worse than recording nothing.
        if row["Sense"] not in (EMPTY, log_sense):
            die(f"log says the model is a {log_sense} but '{name}' is a "
                f"{row['Sense']}; is this log from a different instance?")
    else:
        primal, dual, iters, policy = args.primal, args.dual, args.iters, args.policy
        outcome = INFEASIBLE if args.infeasible else None
    # A run that found no feasible point still found something out, so it is
    # something to record even though both bounds may be absent.
    if primal is not None and outcome is not None:
        die("a run cannot both attain a primal bound and have found no "
            "feasible point; drop --infeasible, or record the log that "
            "actually says so")
    if primal is None and dual is None and outcome is None:
        die("nothing to record: pass --log, or --primal and/or --dual, or "
            "--infeasible")

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
    conflicts = []
    for kind, value, kind_outcome, value_col, commit_col, iters_col, \
            policy_col in (
        # Only the primal side carries an outcome: both outcomes are statements
        # about whether a feasible point exists to attain a value, and the dual
        # bound the same run printed is an ordinary bound either way -- on a
        # search space that turned out to be empty, or on one it never finished
        # searching.
        ("primal", primal, outcome, "Best primal", "Primal @",
         "Primal iters", "Primal policy"),
        ("dual", dual, None, "Best dual", "Dual @", "Dual iters",
         "Dual policy"),
    ):
        old_cell = row[value_col]
        old = parse_number(old_cell)
        old_iters = parse_iters(row[iters_col])
        # What this run has to say about this bound, already in the table's own
        # spelling, or None when it says nothing. Comparing cells rather than
        # floats is what lets an outcome travel the same paths as a number
        # without ever being arithmetic: it is equal to itself and to nothing
        # else, and `better` never sees it because `value` is None.
        new_cell = fmt(value) if value is not None else kind_outcome
        # A tie is decided at the table's precision, and before `better`,
        # because the stored cell was rounded on the way in: re-recording the
        # very same run can come back a rounding step tighter, which `better`
        # reads as an improvement. Left in that order, a longer run would
        # overwrite a shorter one's iteration count on a bound that did not
        # actually move.
        tied = new_cell is not None and new_cell == old_cell
        # A feasible point displaces a proof there is none, and contradicts it:
        # one of the two runs is wrong, and the point is the half of the pair
        # that can be checked, so it is what the row keeps. `better` already
        # takes it -- an outcome parses as no bound -- so this only has to be
        # noticed, not enforced.
        disproves = value is not None and old_cell == INFEASIBLE
        # The policy travels with the commit and the count, always: the three
        # together describe one run, and leaving a stale policy beside a new
        # bound would describe a run that never happened.
        def take(note):
            row[value_col] = new_cell
            row[commit_col] = commit
            row[iters_col] = fmt_iters(iters)
            row[policy_col] = fmt_policy(policy)
            changed.append(note)

        if args.replace:
            # Not a comparison at all: the recorded half of the row becomes
            # this run outright. A bound this run does not have clears rather
            # than survives -- the point of asking for a replace is that the
            # old numbers are no longer to be trusted, and a leftover bound
            # from a superseded commit is exactly what would be trusted.
            if new_cell is not None:
                take(f"{kind} {old_cell} -> {new_cell} "
                     f"in {fmt_iters(iters)} iters (replaced)")
            else:
                row[value_col] = EMPTY
                row[commit_col] = EMPTY
                row[iters_col] = EMPTY
                row[policy_col] = EMPTY
                changed.append(f"{kind} {old_cell} -> cleared, this run "
                               f"reported none (replaced)")
        elif args.force and new_cell is not None:
            take(f"{kind} {old_cell} -> {new_cell} "
                 f"in {fmt_iters(iters)} iters (forced)")
        elif tied and new_cell == NO_SAMPLE:
            # The one cell where a longer run is the better record. `no sample`
            # is bounded by the search that produced it -- it says the sampler
            # failed for that many iterations -- so a run that failed for more
            # of them says strictly more, and one that gave up sooner says
            # less. Every other cell reads the opposite way, which is why this
            # is not `cheaper`.
            if longer(iters, old_iters):
                take(f"{kind} {new_cell} again, and for longer, "
                     f"{fmt_iters(old_iters)} -> {fmt_iters(iters)} iters")
            else:
                changed.append(f"{kind} {new_cell} again in "
                               f"{fmt_iters(iters)} iters, no longer than "
                               f"{fmt_iters(old_iters)}, kept")
        elif tied:
            if cheaper(iters, old_iters):
                # Reaching the same cell sooner is progress for an exhausted
                # search too: the space was emptied in fewer iterations.
                take(f"{kind} {new_cell} matched in fewer iters, "
                     f"{fmt_iters(old_iters)} -> {fmt_iters(iters)}")
            else:
                changed.append(f"{kind} {new_cell} matched in "
                               f"{fmt_iters(iters)} iters, not fewer than "
                               f"{fmt_iters(old_iters)}, kept")
        elif better(kind, sense, value, old):
            take(f"{kind} {old_cell} -> {new_cell} in {fmt_iters(iters)} iters"
                 + (f" (a feasible point, so the row's `{INFEASIBLE}` was wrong)"
                    if disproves else ""))
            if disproves:
                conflicts.append(
                    f"a run recorded `{INFEASIBLE}` here, so one of the two is "
                    f"wrong: either that search dropped a region holding this "
                    f"point, or this point is not feasible")
        elif new_cell in PRIMAL_RANK:
            # Neither cell is a number, so `better` has nothing to compare and
            # the rank decides: a proof of infeasibility displaces a run that
            # merely failed to sample, and nothing displaces a real bound.
            if PRIMAL_RANK[new_cell] > PRIMAL_RANK.get(old_cell, 0) and old is None:
                take(f"{kind} {old_cell} -> {new_cell} in "
                     f"{fmt_iters(iters)} iters")
            elif old is not None and new_cell == INFEASIBLE:
                # The mirror of `disproves`, decided the other way round for
                # the same reason: the point already recorded is the half that
                # can be checked, so it stays and the proof is reported.
                changed.append(f"{kind} search space exhausted with no "
                               f"feasible point, but {old_cell} was recorded "
                               f"before, kept")
                conflicts.append(
                    f"this run exhausted the search space, which says no "
                    f"feasible point exists, but {old_cell} is recorded as one; "
                    f"either this search dropped a region or that bound is not "
                    f"a feasible point")
            else:
                changed.append(f"{kind} {new_cell} says less than "
                               f"{old_cell}, kept")
        elif new_cell is not None:
            changed.append(f"{kind} {new_cell} not better than {old_cell}, kept")

    ordered = sorted(rows.values(), key=lambda r: r["Instance"])
    args.status.write_text(
        render(ordered, args.corpus, measured_at, read_reference_at(args.status)))
    print(f"{name} ({sense}): " + "; ".join(changed))

    # Two runs of this solver disagreeing about whether a feasible point
    # exists, which the row cannot show because it keeps only one of them.
    for message in conflicts:
        print(f"warning: {name}: {message}", file=sys.stderr)

    # Checked against the row as written, not against this run's numbers, so a
    # bound that was kept rather than taken is still measured against the
    # literature. Reported after the write: the row is what it is, and burying
    # a contradiction by refusing to record it only hides the thing worth
    # seeing.
    if row["Best primal"] == INFEASIBLE:
        # The one claim in this table that a reference bound can *support*
        # rather than merely fail to contradict, and the row cannot show it:
        # MINLPLib records an infeasible instance as an infinite dual bound,
        # which reads as a very weak bound unless you know the convention.
        ref_dual = parse_number(row["Ref dual"])
        sign = -1.0 if sense == "max" else 1.0
        if ref_dual is not None and sign * ref_dual == math.inf:
            print(f"note: {name} MINLPLib's dual bound is infinite too, which "
                  f"is how it records an instance known to be infeasible",
                  file=sys.stderr)
        elif ref_dual is None and row["Ref primal"] == EMPTY:
            print(f"note: {name} MINLPLib publishes no bound of either kind, "
                  f"so there is nothing to check `{INFEASIBLE}` against",
                  file=sys.stderr)

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
    record.add_argument("--policy",
                        help="policy the run used, e.g. '--policy=discrete' or "
                             "'--policy=discrete --partition-num=7' for an "
                             "overridden run (default: scraped from --log's "
                             "PARAMS line)")
    record.add_argument("--infeasible", action="store_true",
                        help=f"record that the search emptied its frontier "
                             f"with no feasible point, which the primal column "
                             f"shows as `{INFEASIBLE}` (default: classified "
                             f"from --log's pending count, which also tells it "
                             f"apart from `{NO_SAMPLE}`; needed only to record "
                             f"one without a log)")
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
                             "-- a commit that changed the search, a policy "
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
