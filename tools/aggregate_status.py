#!/usr/bin/env python3
"""Generate AGGREGATE_STATUS.md: the aggregate backend's results, against the
published reference bounds and against gams_solve on the same instances.

    tools/aggregate_status.py <comparison.json> [--out AGGREGATE_STATUS.md]

A sibling of tools/minlp_status.py rather than a mode of it, for the same
reason aggregate_solve is a sibling of gams_solve: the two backends are
compared, not merged (design/AGGREGATE_BOUNDING.md §1). The reference bounds
are read out of MINLP_STATUS.md rather than re-fetched, so both files quote
the same published numbers from the same retrieval.

The input is the JSON the stage-6 harness writes: one row per instance, each
holding an `aggregate` and a `gams` result with primal / dual / stop / secs.
"""

import argparse
import json
import math
import os
import re
import sys

# `| name | parses | sense | vars | cons | nodes | ref primal | ref dual | ...`
ROW = re.compile(r"^\|\s*([^|\s]+)\s*\|([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|"
                 r"([^|]*)\|([^|]*)\|([^|]*)\|")

# The driver collapses glb onto gub once it declares convergence, so a
# converged dual is only sound to within the run's tolerance. Comparing such a
# dual against a published primal can therefore show a violation of order the
# tolerance and mean nothing. Anything larger is a real inversion.
TOLERANCE = 1e-6


def read_reference(path):
    """{instance: (sense, vars, cons, nodes, ref_primal, ref_dual)}."""
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        m = ROW.match(line)
        if not m:
            continue
        name = m.group(1)
        if name in ("Instance", "---", "Cell", "Note"):
            continue
        fields = [f.strip() for f in m.groups()[1:]]
        parses, sense, nvars, ncons, nodes, refp, refd = fields
        if parses not in ("yes", "no"):
            continue
        out[name] = (sense, nvars, ncons, nodes, refp, refd)
    return out


def num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def clamp_to_optimum(sense, dual, ref_primal):
    """A dual can never legitimately beat the true optimum, so cap it there.

    Without this, a run that converged and had `finalise_bounds` collapse its
    dual onto a slightly-worse incumbent scores as having the *tighter* dual
    than a run that landed exactly on the published optimum -- it "wins" by
    overshooting within the tolerance. On `gear` that put the aggregate ahead
    at 1.5e-10 against a gams dual of 2.7e-12 that equals the published primal
    exactly, and on `ex8_1_1` it did the same at 6.7e-8. Capping both sides at
    the published primal makes the comparison what it should be: how close
    each got to the optimum from the correct side.
    """
    if dual is None or ref_primal is None:
        return dual
    return min(dual, ref_primal) if sense == "min" else max(dual, ref_primal)


def vacuous(sense, a, b, ref_primal):
    """Both duals so far from the optimum that ranking them means nothing.

    Reachable on instances whose variables took the frontend's 1e6 default
    bound: the relaxation over a 2e6-wide domain is astronomically loose, and
    on `ex8_1_3` the two backends differ by an order of magnitude at 1e46 and
    1e47 while the optimum is 3. Declaring a winner there would be recording
    noise as a result.
    """
    if ref_primal is None:
        return False
    scale = 1e6 * max(1.0, abs(ref_primal))
    return all(d is not None and abs(d - ref_primal) > scale for d in (a, b))


def better_dual(sense, a, b, ref_primal=None):
    """Which of two dual bounds is tighter, in the instance's own sense.

    For `min` a dual is a lower bound, so larger is tighter; for `max` it is
    an upper bound and smaller is tighter. Both are first capped at the
    published optimum -- see clamp_to_optimum. Returns 'a', 'b', 'tie' or
    'vacuous'.
    """
    if vacuous(sense, a, b, ref_primal):
        return "vacuous"
    a = clamp_to_optimum(sense, a, ref_primal)
    b = clamp_to_optimum(sense, b, ref_primal)
    if a is None and b is None:
        return "tie"
    if a is None:
        return "b"
    if b is None:
        return "a"
    if math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
        return "tie"
    if sense == "min":
        return "a" if a > b else "b"
    return "a" if a < b else "b"


def unsound(sense, dual, ref_primal):
    """A dual bound strictly past a published feasible objective value.

    Beyond the tolerance this cannot happen for a sound bound, so it is a bug
    report rather than a weak result -- see the module docstring for why a
    small violation on a converged run is not one.
    """
    if dual is None or ref_primal is None:
        return False
    slack = TOLERANCE * max(1.0, abs(ref_primal))
    return dual > ref_primal + slack if sense == "min" else dual < ref_primal - slack


def cell(text):
    return text if text not in (None, "") else "—"


def secs(result):
    """Wall time, or an em-dash. A row recorded by hand may not carry one, and
    inventing a number there would put a measurement in the table that nobody
    took."""
    v = result.get("secs")
    return f"{v:.1f}" if isinstance(v, (int, float)) else "—"


def count(value):
    """A frontier size with thousands separators. These span four orders of
    magnitude between the two backends and are unreadable without them."""
    return f"{value:,}" if isinstance(value, int) else "—"


def fmt(value, width=13):
    if value is None:
        return "—"
    return f"{value:.{width}g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("comparison")
    ap.add_argument("--out", default="AGGREGATE_STATUS.md")
    ap.add_argument("--reference", default="MINLP_STATUS.md")
    ap.add_argument("--commit", default="")
    args = ap.parse_args()

    data = json.load(open(args.comparison))
    ref = read_reference(args.reference)
    rows = data["rows"]

    wins = losses = ties = vacuous_rows = 0
    inversions = []
    uneven = []
    converged = 0

    body = []
    for row in rows:
        name = row["instance"]
        agg, gams = row["aggregate"], row["gams"]
        sense, nvars, ncons, nodes, refp, refd = ref.get(
            name, ("?", "?", "?", "?", "", ""))

        ad, gd = num(agg.get("dual")), num(gams.get("dual"))
        ap_, gp = num(agg.get("primal")), num(gams.get("primal"))
        who = better_dual(sense, ad, gd, num(refp))
        wins += who == "a"
        losses += who == "b"
        ties += who == "tie"
        vacuous_rows += who == "vacuous"
        converged += agg.get("stop") == "converged"

        if unsound(sense, ad, num(refp)):
            inversions.append((name, ad, refp))

        mark = {"a": "**agg**", "b": "gams", "tie": "tie",
                "vacuous": "_both vacuous_"}[who]
        agg_iters = agg.get("iters", data["iters"])
        if str(agg_iters) != str(data["iters"]):
            uneven.append((name, agg_iters, data["iters"]))
        iters_cell = (f"**{int(agg_iters):,}**"
                      if str(agg_iters) != str(data["iters"]) else "—")

        body.append(
            f"| {name} | {sense} | {nvars} | {ncons} | {nodes} "
            f"| {cell(refp)} | {cell(refd)} "
            f"| {fmt(ap_)} | {fmt(ad)} | {agg.get('stop','—')} | {secs(agg)} "
            f"| {count(agg.get('pending'))} | {iters_cell} "
            f"| {fmt(gp)} | {fmt(gd)} | {gams.get('stop','—')} | {secs(gams)} "
            f"| {count(gams.get('pending'))} "
            f"| {mark} |")

    out = []
    out.append("# Aggregate backend status\n")
    out.append("What the aggregate-bounding backend "
               "(design/AGGREGATE_BOUNDING.md) achieves on a low-constraint")
    out.append("MINLPLib subset, beside `gams_solve` on the same instances "
               "and the same iteration budget.")
    out.append("Generated by `tools/aggregate_status.py`; edit that, not "
               "this.\n")
    out.append(f"- Instances: **{len(rows)}**, chosen for zero or very few "
               "inequality constraints (§12)")
    out.append(f"- Iteration budget: **{data['iters']}** per run; wall-clock "
               f"cap **{data['cap']} s**")
    out.append("- Reference bounds: quoted from `MINLP_STATUS.md`, which "
               "records their retrieval date")
    if args.commit:
        out.append(f"- Measured at: `{args.commit}`")
    out.append("")

    out.append("## Reading the table\n")
    out.append("**Ref primal / Ref dual** are the published bounds, in the "
               "instance's own sense. **Agg** and")
    out.append("**gams** columns are this run's own. For a `min` row a dual "
               "is a lower bound, so *larger*")
    out.append("is tighter; for `max` it is an upper bound and smaller is "
               "tighter. The **Better dual**")
    out.append("column names the backend whose dual is tighter, which is the "
               "quantity stage 6 exists to")
    out.append("compare — the primal is reported but is a sampler result "
               "rather than a property of the")
    out.append("bounding scheme.\n")
    out.append("A run that reports `converged` has had its dual collapsed "
               "onto its incumbent by")
    out.append("`finalise_bounds`, so its dual is sound only to within the "
               "run's tolerance. A converged")
    out.append("dual a hair past the published primal is that, not an "
               "unsound bound; anything past it by")
    out.append("more than the tolerance is listed under Inversions below and "
               "is a bug report.\n")
    out.append("Both duals are capped at the published primal before being "
               "compared: a run cannot be")
    out.append("credited with beating the true optimum, which it would "
               "otherwise be whenever the")
    out.append("tolerance let its converged dual overshoot. `_both vacuous_` "
               "marks a row where neither")
    out.append("backend came within a factor of 1e6 of the optimum — "
               "reachable when a variable took")
    out.append("the frontend's 1e6 default bound, and not a result worth "
               "ranking.\n")
    out.append("**Agg iters** is set only where a row did *not* run at the "
               "budget in the header — a row")
    out.append("carrying a number there was given more (or fewer) iterations "
               "than the backend it sits")
    out.append("beside, and its comparison is not like-for-like. **pending** "
               "is the frontier size the run")
    out.append("ended with, which is §2.1's whole claim in one column.\n")

    out.append("## Instances\n")
    out.append("| Instance | Sense | Vars | Cons | Nodes | Ref primal | Ref dual "
               "| Agg primal | Agg dual | Agg stop | Agg s | Agg pending | Agg iters "
               "| gams primal | gams dual | gams stop | gams s | gams pending "
               "| Better dual |")
    out.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
               "--- | --- | --- | --- | --- | --- | --- | --- | --- |")
    out.extend(body)
    out.append("")

    out.append("## Summary\n")
    out.append(f"- Tighter dual: **aggregate {wins}**, gams {losses}, "
               f"tie {ties}, both vacuous {vacuous_rows}, of {len(rows)}")
    out.append(f"- Aggregate runs reaching `converged`: **{converged}** "
               f"of {len(rows)}")
    out.append(f"- Dual bounds past a published primal by more than the "
               f"tolerance: **{len(inversions)}**")
    if uneven:
        out.append("")
        out.append("Rows **not** run at the header's iteration budget, whose "
                   "backend-to-backend comparison")
        out.append("is therefore not like-for-like:")
        out.append("")
        for name, got, base in uneven:
            out.append(f"  - `{name}`: aggregate at {int(got):,} iterations "
                       f"against gams at {int(base):,}")
    if inversions:
        out.append("")
        for name, dual, refp in inversions:
            out.append(f"  - `{name}`: dual {dual} against published primal "
                       f"{refp} — **investigate before quoting this table**")
    out.append("")
    out.append("## Reproducing a row\n")
    out.append("```sh")
    out.append("cmake --build build/dev -t aggregate_solve")
    out.append(f"./build/dev/aggregate_solve ../minlplib/gms/<instance>.gms "
               f"{data['iters']}")
    out.append("```\n")
    out.append("Unlike `MINLP_STATUS.md`'s `--policy=` cell, this is the "
               "whole command: the aggregate")
    out.append("backend's shape comes from `--branch-factor` and "
               "`--partition-budget`, both of which")
    out.append("default to the values these runs used (`k = 4`, `N` fitted to "
               "free device memory).")

    text = "\n".join(out) + "\n"
    open(args.out, "w").write(text)
    print(f"wrote {args.out}: {len(rows)} instances, "
          f"agg {wins} / gams {losses} / tie {ties}, "
          f"{len(inversions)} inversion(s)")
    return 1 if inversions else 0


if __name__ == "__main__":
    sys.exit(main())
