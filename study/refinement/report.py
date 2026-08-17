#!/usr/bin/env python3
"""Turn refinement_study CSVs into the answers the study was built to give.

    study/refinement/report.py [options] <sweep-dir-or-csv>...

      --bounds            show the raw interval bounds ladder (see below)
      --boxes=N           parent boxes per instance in --bounds. Default 2.
      --instance=NAME     restrict every section to instances matching NAME

Reads every CSV written by refinement_study and reports:

  Bounds  the refined hull [L_N, U_N] against the baseline [L0, U0] -- the
      interval evaluation over the parent box itself -- in absolute terms,
      per parent box, as N climbs. Every other section here is a ratio or an
      exponent, and a ratio hides whether the numbers involved are of order
      one or of order 10^15. This section is where a bound that is vacuous in
      absolute terms becomes visible as such.

  Q1  how fast the interval hull's EXCESS width vanishes in N, fitted per
      instance *within* each parent box (§6.7 -- the ladder is only a ladder
      for a fixed box). The hull width itself converges to the objective's
      true range over the box, not to zero, so the excess over that floor is
      the quantity Theorem 6.1 bounds; normalised per dimension it is directly
      comparable to the theorem's prediction of first-order convergence. Both
      sides of the hull are fitted separately, since they can converge at
      different rates and the total tracks the slower one.

  Q1b the older hull-width ratio rho ~ N^-alpha, and how rho varies with
      parent-box width sigma. Kept because it is what the study originally
      measured, and reported with the confound -- rho's nonzero asymptote --
      that makes it read far worse than the truth. See the section's footer.

  Q2  whether the per-subregion lower bounds are all alike or have isolated
      outliers, via layer 1's distinct_frac and layers 3-4's outlier scores
      and isolation gaps.

  Attribution  how much of the tightening is constraint propagation rather
      than interval refinement, from the masked/unmasked pair (§2.5).

Undefined quantities are empty fields in the CSV, never NaN, so every
aggregate here reports the count it was computed over.

See study/refinement/FINDINGS.md for what these numbers came out as, and
design/REFINEMENT_STUDY.md for why they are the numbers being asked for.
"""

import collections
import csv
import math
import os
import sys

# Layer 3's classical Tukey fence. A low_score far above this with a tiny
# low_frac is the "a few particularly good subregions" signature.
TUKEY = 1.5


def number(row, key):
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load(paths):
    """{instance: [row, ...]} over every CSV found under `paths`.

    Rows short of the header are dropped with a warning rather than parsed:
    a sweep read while it is still running ends in a partially flushed line,
    and every column past the truncation would otherwise be read under the
    wrong name. check_sweep.py is what diagnoses a short row precisely; this
    tool only has to refuse to average one in.
    """
    files = []
    for path in paths:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                if name.endswith(".csv"):
                    files.append(os.path.join(path, name))
        else:
            files.append(path)

    by_instance = collections.defaultdict(list)
    dropped = 0
    for path in files:
        with open(path) as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("instance") is None or None in row.values():
                    dropped += 1
                    continue
                by_instance[os.path.basename(row["instance"])].append(row)
    if dropped:
        print(f"warning: dropped {dropped} short row(s); is the sweep still "
              f"running?")
    return by_instance


def fit_exponent(points):
    """Least-squares slope of log(rho) against log(N); returns -slope.

    `points` is [(N, rho), ...]. Needs three distinct N to mean anything.
    """
    points = [(n, r) for n, r in points if n > 0 and r is not None and r > 0]
    if len({n for n, _ in points}) < 3:
        return None
    xs = [math.log(n) for n, _ in points]
    ys = [math.log(r) for _, r in points]
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    denominator = n * sum(x * x for x in xs) - sx * sx
    if abs(denominator) < 1e-12:
        return None
    slope = (n * sum(x * y for x, y in zip(xs, ys)) - sx * sy) / denominator
    return -slope


def largest_budget_rows(rows):
    """One row per parent box, at that box's largest completed budget."""
    best = {}
    for row in rows:
        box = row["box_id"]
        if box not in best or int(row["budget"]) > int(best[box]["budget"]):
            best[box] = row
    return list(best.values())


def mean(values):
    values = [v for v in values if v is not None]
    return (sum(values) / len(values), len(values)) if values else (None, 0)


def report_bounds(by_instance, boxes_per_instance):
    """The refined hull against the baseline, in absolute terms.

    The baseline is the interval evaluation over the parent box as a single
    region -- exactly what the solver would compute without subdividing at
    all -- so this table is the direct "what did subdivision actually buy"
    view that every ratio elsewhere in this report is derived from.

    The *unmasked* hull is what is shown, because the baseline is itself
    unmasked: on a constrained instance the masked hull is tighter partly
    because constraint propagation excluded subregions, and putting it beside
    an unmasked baseline would credit that tightening to interval refinement
    (§2.5). Masked columns are added for constrained instances, alongside
    rather than instead.
    """
    print("=" * 78)
    print("Bounds  refined hull [L_N, U_N] vs baseline [L0, U0] over the "
          "parent box")
    print("=" * 78)

    for instance, rows in sorted(by_instance.items()):
        by_box = collections.defaultdict(list)
        for row in rows:
            by_box[row["box_id"]].append(row)

        chosen = sorted(by_box, key=lambda b: int(b))[:boxes_per_instance]
        for box_id in chosen:
            group = sorted(by_box[box_id], key=lambda r: int(r["budget"]))
            first = group[0]
            base_lo, base_hi = number(first, "base_lb"), number(first, "base_ub")
            if base_lo is None or base_hi is None:
                continue
            base_width = base_hi - base_lo
            constrained = first.get("constrained") == "1"

            print()
            print(f"{instance}  box {box_id}  sigma={first['sigma']}"
                  f"  partition={first.get('partition', '?')}")
            print(f"  baseline  N=1  [{base_lo:>14.6g}, {base_hi:>14.6g}]"
                  f"  width {base_width:.6g}")

            header = (f"  {'N':>12}{'L_N':>16}{'U_N':>16}{'width':>14}"
                      f"{'rho':>8}{'dL':>14}")
            if constrained:
                header += f"{'masked L_N':>16}{'excl':>7}"
            else:
                # min_r ub_r bounds the optimum over the box from above with
                # no sampling (§2.4), so on an unconstrained instance
                # [L_N, min_ub] brackets the optimum and its width is the
                # optimality gap a search would actually face. It is NOT sound
                # under constraints -- an unexcluded subregion is not known to
                # contain a feasible point -- so the column only appears here.
                header += f"{'min_ub':>16}{'gap':>14}"
            print(header)

            for row in group:
                n = number(row, "n_regions")
                lo, hi = number(row, "unmasked_lb"), number(row, "unmasked_ub")
                if n is None or lo is None or hi is None:
                    continue
                width = hi - lo
                # Recomputed from the unmasked pair rather than read from
                # width_ratio, which is the masked ratio; on a constrained
                # instance the two differ and this table is the unmasked one.
                rho = width / base_width if base_width else float("nan")
                line = (f"  {int(n):>12}{lo:>16.6g}{hi:>16.6g}{width:>14.6g}"
                        f"{rho:>8.4f}{lo - base_lo:>14.6g}")
                if constrained:
                    masked = number(row, "masked_lb")
                    excluded = number(row, "excluded_frac")
                    line += (f"{masked:>16.6g}" if masked is not None
                             else f"{'-':>16}")
                    line += (f"{excluded:>7.2f}" if excluded is not None
                             else f"{'-':>7}")
                else:
                    min_ub = number(row, "min_ub")
                    line += (f"{min_ub:>16.6g}" if min_ub is not None
                             else f"{'-':>16}")
                    line += (f"{min_ub - lo:>14.6g}" if min_ub is not None
                             else f"{'-':>14}")
                print(line)

    print()
    print("  dL is the only column a branch-and-bound search can act on: it "
          "is how far\n  the lower bound moved, and pruning keys on nothing "
          "else.")
    print("  rho is the width ratio, and it is confounded when the objective's "
          "TRUE range\n  over the box is wide: U_N can be both enormous and "
          "exactly right, holding\n  rho near 1 while L_N converges. Read rho "
          "against gap, not on its own.")
    print("  gap = min_ub - L_N brackets the optimum on an unconstrained "
          "instance (§2.4)\n  and is the honest 'how much uncertainty is "
          "left' number.")
    print()


def report_excess(by_instance):
    """Q1's primary answer: how fast the interval hull's EXCESS width vanishes.

    The hull width itself does not converge to zero, and fitting a power law
    to it is a category error. As N grows, lb_r -> min f over subregion r and
    ub_r -> max f over r, so

        L_N -> min f over P     U_N -> max f over P
        W_N -> the true range of f over P

    The hull width has a floor, and rho = W_N / W0 therefore converges to
    range/W0 -- a positive constant. A fit of rho ~ N^-alpha is fitting a
    power law to something with a nonzero asymptote, which is why alpha comes
    out near zero however well the relaxation is behaving (report_rho).

    Theorem 6.1 is not about the width. It bounds the EXCESS width, the part
    of the hull that is relaxation looseness rather than genuine range:

        W_N = (true range) + excess(N),   excess(N) <= K * w(X) / N_dim

    Neither the range nor the excess is directly observable -- both need the
    true min and max of f, which is what the whole exercise is trying to
    bound. But both sides are bracketed soundly by quantities already in the
    CSV, because a hull bound and the opposite extreme over the subregions
    sandwich the truth:

        min_r ub_r  >=  min f  >=  L_N        (min_ub, and the hull's lb)
        U_N         >=  max f  >=  max_r lb_r (the hull's ub, and lb_q100)

    so, writing max_lb for max_r lb_r,

        excess(N) <= (min_ub - L_N) + (U_N - max_lb)
                     `-- lower side    `-- upper side

    and the true range is at least max_lb - min_ub. Both terms are
    non-negative by construction, and both are reported: on some instances
    the two sides converge at visibly different rates, and the total is
    governed by the slower one, so a lower-side-only number flatters the
    result.

    The theorem is stated per dimension -- splitting every axis into N_dim
    parts -- while the sweep's N is the total region count, N_dim^n_live under
    the uniform partition. The prediction here is therefore

        excess ~ N^-beta   with   beta * n_live = 1

    and `x nlive` is the column to read against 1.0.

    Constrained instances are excluded, not merely flagged: min_ub is not a
    sound bound on the optimum when constraints are present (an unexcluded
    subregion is not known to contain a feasible point), so the lower side --
    and with it the total -- would not mean what the header says.
    """
    print("=" * 78)
    print("Q1  interval hull excess width   W_N = (true range) + excess")
    print("    excess ~ N^-beta,  bounded by (min_ub - L_N) + (U_N - max_lb)")
    print("=" * 78)
    print(f"{'instance':<24}{'nlive':>6}{'lower':>8}{'upper':>8}{'total':>8}"
          f"{'x nlive':>9}{'boxes':>7}{'exc@minN':>12}{'exc@maxN':>12}"
          f"{'exc/range':>12}")

    table = []
    skipped = []
    for instance, rows in sorted(by_instance.items()):
        if rows[0].get("constrained") == "1":
            skipped.append(instance)
            continue

        lower = collections.defaultdict(list)
        upper = collections.defaultdict(list)
        total = collections.defaultdict(list)
        live = set()
        largest = {}
        for row in rows:
            n = number(row, "n_regions")
            lo = number(row, "unmasked_lb")
            hi = number(row, "unmasked_ub")
            min_ub = number(row, "min_ub")
            max_lb = number(row, "lb_q100")
            slots = number(row, "slots")
            if slots:
                live.add(int(slots))
            if None in (n, lo, hi, min_ub, max_lb) or not n:
                continue
            box = row["box_id"]
            low_side = min_ub - lo
            high_side = hi - max_lb
            if low_side > 0:
                lower[box].append((n, low_side))
            if high_side > 0:
                upper[box].append((n, high_side))
            if low_side + high_side > 0:
                total[box].append((n, low_side + high_side))
            # Kept so the excess can be reported as a share of the range at
            # the box's largest N, where both are tightest. A share rather
            # than the raw range because the sigma ladder spans boxes of
            # wildly different widths, and a mean of raw ranges across them
            # is dominated by the widest box and says nothing.
            if box not in largest or n > largest[box][0]:
                largest[box] = (n, max_lb - min_ub, low_side + high_side)

        def fitted(groups):
            values = [b for b in (fit_exponent(p) for p in groups.values())
                      if b is not None]
            return sum(values) / len(values) if values else None

        beta_lo, beta_hi = fitted(lower), fitted(upper)
        beta_total = fitted(total)
        if beta_total is None:
            continue

        flat = [p for points in total.values() for p in points]
        n_min, n_max = min(n for n, _ in flat), max(n for n, _ in flat)
        exc_lo, _ = mean([e for n, e in flat if n == n_min])
        exc_hi, _ = mean([e for n, e in flat if n == n_max])
        share, _ = mean([(e / r if r > 0 else None)
                         for _, r, e in largest.values()])

        # Under the uniform partition every live dimension gets exactly one
        # slot, so the slot count is n_live. Under the budget partition it is
        # the number of variables the greedy heap happened to touch, which is
        # not the live dimension, and the per-dimension normalisation below
        # does not apply.
        uniform = rows[0].get("partition") == "uniform"
        n_live = max(live) if live else 0
        scaled = beta_total * n_live if (uniform and n_live) else None

        table.append((instance, beta_total, n_live, scaled))
        cell = lambda v: f"{v:.4f}" if v is not None else "-"
        print(f"{instance[:24]:<24}{(str(n_live) if scaled else '-'):>6}"
              f"{cell(beta_lo):>8}{cell(beta_hi):>8}{cell(beta_total):>8}"
              f"{cell(scaled):>9}{len(total):>7}{exc_lo:>12.4g}"
              f"{exc_hi:>12.4g}"
              f"{(f'{share:.3%}' if share is not None else '-'):>12}")

    scaled_all = [t[3] for t in table if t[3] is not None]
    if scaled_all:
        print(f"\n  corpus mean beta*n_live = "
              f"{sum(scaled_all) / len(scaled_all):.4f} over "
              f"{len(scaled_all)} instances")
        print("  Theorem 6.1 predicts 1.0 (first order in the per-dimension "
              "refinement).")
    print("""
  'exc/range' is what remains of the hull at the largest N, as a share of the
  floor it converges to -- the objective's true range over the box, which no
  subdivision can go below. A tiny share means the relaxation is essentially
  exact and the hull is wide because the FUNCTION varies that much over the
  box, not because the arithmetic is loose. That is the case in which rho
  looks flat while nothing is wrong.

  'lower' and 'upper' are the two sides' exponents. Where they differ, the
  total tracks the slower side, and a lower-side-only figure overstates how
  fast the hull is tightening.""")
    if skipped:
        print(f"\n  excluded (constrained, so min_ub is not a sound bound on "
              f"the optimum):\n    {', '.join(sorted(skipped))}")
    print()
    return table


def report_rho(by_instance):
    print("=" * 78)
    print("Q1b  hull width ratio   rho ~ N^-alpha  (confounded -- see below)")
    print("=" * 78)
    print(f"{'instance':<26}{'alpha':>9}{'boxes':>7}{'rho@min':>10}"
          f"{'rho@max':>10}{'Nmax':>9}")

    table = []
    for instance, rows in sorted(by_instance.items()):
        by_box = collections.defaultdict(list)
        for row in rows:
            rho = number(row, "width_ratio")
            n = number(row, "n_regions")
            if rho is not None and n:
                by_box[row["box_id"]].append((n, rho))

        alphas = []
        for points in by_box.values():
            alpha = fit_exponent(points)
            if alpha is not None:
                alphas.append(alpha)
        if not alphas:
            continue

        flat = [p for points in by_box.values() for p in points]
        n_min = min(n for n, _ in flat)
        n_max = max(n for n, _ in flat)
        rho_lo, _ = mean([r for n, r in flat if n == n_min])
        rho_hi, _ = mean([r for n, r in flat if n == n_max])
        alpha_mean = sum(alphas) / len(alphas)

        table.append((instance, alpha_mean, len(alphas), rho_lo, rho_hi, n_max))
        print(f"{instance:<26}{alpha_mean:>9.4f}{len(alphas):>7}"
              f"{rho_lo:>10.4f}{rho_hi:>10.4f}{int(n_max):>9}")

    if table:
        overall = sum(t[1] for t in table) / len(table)
        print(f"\n  corpus mean alpha = {overall:.4f} over {len(table)} "
              f"instances")
        print("  Reads as: doubling N multiplies the hull width by "
              f"2^-alpha = {2 ** -overall:.4f}, i.e. almost nothing.")
    print("""
  This number is confounded and should not be quoted on its own. rho divides
  the refined hull's width by the baseline's, and the hull's width is set by
  whichever of L_N, U_N is further from the optimum. Where the objective's
  TRUE range over the box is wide, U_N is enormous AND correct -- it cannot
  shrink, because it is already right -- so rho stays pinned near 1 however
  well L_N converges. ex4_1_2 is the clean case: U_N = 1.2e15 is genuinely
  max(f) on [1,2] for a degree-50 polynomial, rho holds at 0.9999, and the
  optimality bracket nevertheless halves with every doubling of N.

  The Q1 section above measures the quantity that is not confounded this way.""")
    return table


def report_sigma(by_instance):
    print()
    print("=" * 78)
    print("Q1b  rho against parent-box width, at each instance's largest N")
    print("=" * 78)
    print(f"{'instance':<26}" + "".join(f"{s:>11}" for s in
                                        ("sigma=1", "1/2", "1/4", "1/8",
                                         "<=1/16")))

    def bucket(sigma):
        if sigma >= 1.0:
            return "sigma=1"
        if sigma >= 0.5:
            return "1/2"
        if sigma >= 0.25:
            return "1/4"
        if sigma >= 0.125:
            return "1/8"
        return "<=1/16"

    order = ("sigma=1", "1/2", "1/4", "1/8", "<=1/16")
    for instance, rows in sorted(by_instance.items()):
        buckets = collections.defaultdict(list)
        for row in largest_budget_rows(rows):
            sigma = number(row, "sigma")
            rho = number(row, "width_ratio")
            if sigma is not None and rho is not None:
                buckets[bucket(sigma)].append(rho)
        if not buckets:
            continue
        cells = []
        for key in order:
            value, count = mean(buckets.get(key, []))
            cells.append(f"{value:.4f}" if value is not None else "-")
        print(f"{instance:<26}" + "".join(f"{c:>11}" for c in cells))

    print("\n  Rising left-to-right reads as 'subdivision buys proportionally "
          "LESS on a\n  smaller box'. Before drawing that conclusion, note "
          "that rho carries the\n  confound described above, and that it "
          "bites hardest exactly where the box\n  is small: shrinking the box "
          "shrinks the objective's true range on it, so\n  the fraction of "
          "the hull width that is genuine range rather than "
          "relaxation\n  looseness goes UP. The trend may be the metric, not "
          "the geometry.")


def report_q2(by_instance):
    print()
    print("=" * 78)
    print("Q2  are the lower bounds all alike, or are there outliers?")
    print("=" * 78)
    print(f"{'instance':<26}{'distinct':>10}{'modal':>9}{'low_sc':>8}"
          f"{'hi_sc':>8}{'low_frac':>10}{'gap2':>8}{'gap100':>8}")

    verdicts = collections.Counter()
    for instance, rows in sorted(by_instance.items()):
        final = largest_budget_rows(rows)
        distinct, n = mean([number(r, "lb_distinct_frac") for r in final])
        if distinct is None:
            continue
        modal, _ = mean([number(r, "lb_modal_frac") for r in final])
        low_score, _ = mean([number(r, "lb_low_score") for r in final])
        high_score, _ = mean([number(r, "lb_high_score") for r in final])
        low_frac, _ = mean([number(r, "lb_low_frac") for r in final])
        gap2, _ = mean([number(r, "lb_low_gap_2") for r in final])
        gap100, _ = mean([number(r, "lb_low_gap_100") for r in final])

        def cell(v):
            return f"{v:.4f}" if v is not None else "-"

        print(f"{instance:<26}{cell(distinct):>10}{cell(modal):>9}"
              f"{cell(low_score):>8}{cell(high_score):>8}{cell(low_frac):>10}"
              f"{cell(gap2):>8}{cell(gap100):>8}")

        # The verdict layers 1, 3 and 4 were built to deliver.
        if distinct < 0.5:
            verdicts["mostly tied"] += 1
        elif low_score is not None and low_score > TUKEY and gap2 is not None \
                and gap2 > 1.0:
            verdicts["isolated outliers"] += 1
        elif low_score is not None and low_score > TUKEY:
            verdicts["heavy tail, no lone outlier"] += 1
        else:
            verdicts["broad, no outliers"] += 1

    print("\n  verdict across instances:")
    for verdict, count in verdicts.most_common():
        print(f"    {count:>3}  {verdict}")
    print("\n  'mostly tied'  distinct_frac < 0.5: most subregions share a "
          "bound with\n                 another -- no discrimination to "
          "exploit.")
    print("  'isolated outliers'  low_score > 1.5 AND low_gap_2 > 1 IQR: the "
          "minimum\n                 stands alone. This is the case a "
          "targeted policy could chase.")


def report_attribution(by_instance):
    print()
    print("=" * 78)
    print("Attribution  how much tightening is constraints, not refinement")
    print("=" * 78)
    print(f"{'instance':<26}{'rho_masked':>12}{'rho_unmask':>12}"
          f"{'excluded':>10}{'rows':>7}{'empty':>7}")

    for instance, rows in sorted(by_instance.items()):
        final = largest_budget_rows(rows)
        masked, n_masked = mean([number(r, "width_ratio") for r in final])
        unmasked, _ = mean([number(r, "unmasked_width_ratio") for r in final])
        excluded, _ = mean([number(r, "excluded_frac") for r in final])
        # §6.6: on a heavily constrained instance most randomly placed boxes
        # are wholly infeasible, so the masked means average over few boxes.
        empty = sum(1 for r in final if not number(r, "lb_count"))
        if masked is None and unmasked is None:
            continue

        def cell(v):
            return f"{v:.4f}" if v is not None else "-"

        print(f"{instance:<26}{cell(masked):>12}{cell(unmasked):>12}"
              f"{cell(excluded):>10}{len(final):>7}{empty:>7}")

    print("\n  A large masked/unmasked gap means constraint propagation, not "
          "interval\n  refinement, is doing the work. 'empty' counts parent "
          "boxes proven wholly\n  infeasible -- masked columns average over "
          "the rest, so read them with it.")


def main(argv):
    paths = []
    show_bounds = False
    boxes_per_instance = 2
    instance_filter = None

    for arg in argv[1:]:
        if arg in ("-h", "--help"):
            print(__doc__)
            return 0
        if arg == "--bounds":
            show_bounds = True
        elif arg.startswith("--boxes="):
            boxes_per_instance = int(arg[len("--boxes="):])
            show_bounds = True
        elif arg.startswith("--instance="):
            instance_filter = arg[len("--instance="):]
        elif arg.startswith("-"):
            print(f"unknown option: {arg}")
            return 2
        else:
            paths.append(arg)

    if not paths:
        print(__doc__)
        return 2
    by_instance = load(paths)
    if instance_filter:
        by_instance = {k: v for k, v in by_instance.items()
                       if instance_filter in k}
    if not by_instance:
        print("no rows found")
        return 1
    total = sum(len(v) for v in by_instance.values())
    print(f"{total} rows over {len(by_instance)} instances\n")

    if show_bounds:
        report_bounds(by_instance, boxes_per_instance)
    report_excess(by_instance)
    report_rho(by_instance)
    report_sigma(by_instance)
    report_q2(by_instance)
    report_attribution(by_instance)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
