#!/usr/bin/env python3
"""Tests for the log scraper in tools/minlp_status.py.

Only the reading half: `scrape_log` and `primal_outcome` are what stand
between a driver's summary and a permanent row in MINLP_STATUS.md, and both
the bounded frontier (design/BOUNDED_FRONTIER.md §7) and the later
`Outcome:` line (design/TELEMETRY.md follow-up) changed what stands behind
that classification. The rest of the tool writes files and runs subprocesses,
which is a different kind of test and not this one.

Both directions of compatibility are the point of most of what follows: a log
from a driver that could not drop regions, or that predates `Outcome:`
entirely, must still read exactly as it did, and a new line an old tool never
heard of must not break anything.
"""

import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import minlp_status as ms  # noqa: E402


def log(*, outcome=None, pending=None, dropped=None, primal="none",
        dual="-3.5", exhausted=False, iters=2):
    """A gams_solve log, in the shape the driver actually prints one.

    `outcome` ("infeasible" / "no sample") is the current build's line, always
    present on a real log with no primal bound. `pending`/`dropped` reproduce
    the older summary lines a pre-Outcome: log carried instead -- pass those
    alone (outcome=None) to simulate one.
    """
    lines = ["some/model.gms: 3 variables, 1 constraints, 9 DAG nodes",
             "PARAMS\tpolicy=discrete\tsource=auto\tpartition_num=4"]
    for i in range(1, iters + 1):
        lines.append(f"iter {i}: GUB = 1, Candidate: 1")
    if exhausted:
        lines.append("Search space exhausted: every region was proven to hold "
                     "no feasible point, so the problem is infeasible as far "
                     "as interval analysis can tell.")
    lines.append("------------ Finished ------------")
    if pending is not None:
        lines.append("Stop reason: host-memory")
        lines.append(f"Pending size: {pending}")
        lines.append("Viable regions: 0")
        lines.append("Pruned as interval-infeasible: 7")
    if outcome is not None:
        lines.append(f"Outcome: {outcome}")
    if dropped is not None:
        lines.append(f"Dropped viable regions: {dropped}")
        lines.append("Dropped dominated regions: 12")
        lines.append("Dropped lb floor: none")
    lines.append(f"RESULT\tsense=min\tprimal={primal}\tdual={dual}")
    return "\n".join(lines) + "\n"


class PrimalOutcome(unittest.TestCase):
    """The gate itself, in isolation from the scraping around it."""

    def test_a_primal_bound_is_not_an_outcome(self):
        self.assertIsNone(ms.primal_outcome(1.5, 0, False, 0))

    def test_an_emptied_frontier_with_no_drops_is_infeasible(self):
        self.assertEqual(ms.primal_outcome(None, 0, False, 0), ms.INFEASIBLE)

    def test_an_emptied_frontier_with_viable_drops_is_not(self):
        # The regions were discarded for memory, so they are places left to
        # look rather than places shown to be empty.
        self.assertEqual(ms.primal_outcome(None, 0, False, 3), ms.NO_SAMPLE)

    def test_regions_still_pending_is_a_sampling_failure(self):
        self.assertEqual(ms.primal_outcome(None, 41, False, 0), ms.NO_SAMPLE)

    def test_an_old_log_reads_exactly_as_it_did(self):
        # No dropped line at all: a driver that had no way to drop anything.
        self.assertEqual(ms.primal_outcome(None, 0, False, None),
                         ms.INFEASIBLE)
        self.assertEqual(ms.primal_outcome(None, 7, False, None),
                         ms.NO_SAMPLE)

    def test_no_summary_at_all_falls_back_to_the_notice(self):
        self.assertEqual(ms.primal_outcome(None, None, True, None),
                         ms.INFEASIBLE)
        self.assertIsNone(ms.primal_outcome(None, None, False, None))

    def test_explicit_outcome_wins_outright(self):
        self.assertEqual(
            ms.primal_outcome(None, None, False, None, ms.INFEASIBLE),
            ms.INFEASIBLE)
        self.assertEqual(
            ms.primal_outcome(None, None, False, None, ms.NO_SAMPLE),
            ms.NO_SAMPLE)

    def test_explicit_outcome_overrules_a_disagreeing_reconstruction(self):
        # The driver's own answer beats this tool's guess from pending/dropped,
        # even when they would have disagreed -- Outcome: is authoritative.
        self.assertEqual(
            ms.primal_outcome(None, 0, False, 0, ms.NO_SAMPLE), ms.NO_SAMPLE)

    def test_a_primal_bound_still_beats_an_explicit_outcome(self):
        # Can't happen from a real log (gams_solve never prints both), but the
        # gate's own contract holds regardless: a bound is not an outcome.
        self.assertIsNone(ms.primal_outcome(1.5, None, False, None,
                                            ms.INFEASIBLE))


class ScrapeLog(unittest.TestCase):
    """`pending=`/`dropped=` alone (no `outcome=`) simulate a log from before
    the `Outcome:` line existed -- the legacy reconstruction this tool falls
    back to. `CurrentFormat` below exercises the line current builds actually
    print.
    """

    def scrape(self, text):
        with redirect_stderr(io.StringIO()) as captured:
            result = ms.scrape_log(text)
        return result, captured.getvalue()

    def test_an_old_log_without_dropped_lines(self):
        (sense, primal, dual, iters, policy, outcome), _ = self.scrape(
            log(pending=0, exhausted=True))
        self.assertEqual(sense, "min")
        self.assertIsNone(primal)
        self.assertEqual(dual, -3.5)
        self.assertEqual(iters, 2)
        self.assertEqual(policy, "--policy=discrete")
        self.assertEqual(outcome, ms.INFEASIBLE)

    def test_an_emptied_frontier_with_drops_records_no_sample(self):
        (_, _, _, _, _, outcome), _ = self.scrape(log(pending=0, dropped=4))
        self.assertEqual(outcome, ms.NO_SAMPLE)

    def test_an_emptied_frontier_without_drops_records_infeasible(self):
        (_, _, _, _, _, outcome), _ = self.scrape(log(pending=0, dropped=0))
        self.assertEqual(outcome, ms.INFEASIBLE)

    def test_the_new_summary_lines_do_not_disturb_the_rest(self):
        # `Stop reason:` has no regex here at all, which is the compatibility
        # claim: an unknown line is ignored rather than mis-parsed.
        (sense, primal, dual, iters, policy, outcome), stderr = self.scrape(
            log(pending=5, dropped=0, primal="1.25"))
        self.assertEqual((sense, primal, dual, iters), ("min", 1.25, -3.5, 2))
        self.assertEqual(policy, "--policy=discrete")
        self.assertIsNone(outcome)
        self.assertEqual(stderr, "")

    def test_only_the_last_run_in_the_log_is_read(self):
        # Two runs appended to one file: the earlier one dropped regions, the
        # later one did not, and the row is about the later one.
        text = log(pending=0, dropped=9) + log(pending=0, dropped=0)
        (_, _, _, _, _, outcome), _ = self.scrape(text)
        self.assertEqual(outcome, ms.INFEASIBLE)

    def test_a_log_with_no_result_line_is_refused(self):
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                ms.scrape_log("iter 1: GUB = 1, Candidate: 1\n")


class CurrentFormat(unittest.TestCase):
    """A log carrying the `Outcome:` line current builds print -- no
    `Pending size:`/`Dropped viable regions:` needed at all, since ConsoleReporter
    stopped printing the former unconditionally and the classification no
    longer has to be reconstructed from either (design/TELEMETRY.md follow-up).
    """

    def scrape(self, text):
        with redirect_stderr(io.StringIO()) as captured:
            result = ms.scrape_log(text)
        return result, captured.getvalue()

    def test_outcome_infeasible_needs_no_pending_or_dropped_line(self):
        (_, _, _, _, _, outcome), stderr = self.scrape(
            log(outcome="infeasible", exhausted=True))
        self.assertEqual(outcome, ms.INFEASIBLE)
        self.assertEqual(stderr, "")

    def test_outcome_no_sample_needs_no_pending_or_dropped_line(self):
        (_, _, _, _, _, outcome), _ = self.scrape(log(outcome="no sample"))
        self.assertEqual(outcome, ms.NO_SAMPLE)

    def test_outcome_wins_over_a_disagreeing_legacy_reading(self):
        # A log that (hypothetically) carried both: the explicit line is
        # believed over the pending/dropped reconstruction, which here would
        # have said the opposite.
        (_, _, _, _, _, outcome), _ = self.scrape(
            log(outcome="no sample", pending=0, dropped=0))
        self.assertEqual(outcome, ms.NO_SAMPLE)

    def test_a_primal_bound_leaves_no_outcome_even_with_the_line_present(self):
        # Can't happen from a real log, but scrape_log's own contract should
        # hold: a bound found means there is nothing to classify.
        (_, primal, _, _, _, outcome), _ = self.scrape(
            log(outcome="infeasible", primal="1.25"))
        self.assertEqual(primal, 1.25)
        self.assertIsNone(outcome)


if __name__ == "__main__":
    unittest.main()
