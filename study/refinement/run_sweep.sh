#!/usr/bin/env bash
#
# Reproducible driver for the refinement study (design/REFINEMENT_STUDY.md).
#
#   study/refinement/run_sweep.sh <refinement_study binary> [config...]
#
# Runs one or more named configurations over a fixed instance list, writing
# one CSV and one log per instance into results/<config>/, plus a MANIFEST
# recording the binary's git revision, the exact flags, and the date. The
# manifest is the point: the first generation of this sweep left CSVs whose
# flags were not recorded anywhere, and a number nobody can regenerate is not
# a measurement.
#
# Configurations (see FINDINGS.md for what each one is for):
#
#   declared    uniform partition, sigma ladder over each model's own declared
#               bounds. The corpus as the modeller wrote it -- which for
#               several instances means a variable the frontend defaulted to
#               +-1000000. Takeaway 1.
#   unit-width  uniform partition, every live variable given absolute width 1
#               with placements held near the origin. Removes the defaulted-
#               bound confound so what is left is the model's own nonlinearity.
#               Takeaway 2.
#   budget      the solver's own BisectionBudgetCompositionPolicy over the full
#               corpus, including instances too high-dimensional for the
#               uniform partition to reach. Not comparable to the theorem, but
#               it is what a real search would see.
#
# Every configuration is seeded, so a re-run reproduces the CSVs byte for byte
# on the same binary.

set -euo pipefail

BINARY=${1:-}
if [[ -z "$BINARY" || ! -x "$BINARY" ]]; then
  echo "usage: $0 <refinement_study binary> [declared|unit-width|budget ...]" >&2
  exit 2
fi
shift

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
MODELS="$REPO/test/data/gams"
RESULTS="$HERE/results"

SEED=20260812

# Where the ladder stops. The last rung costs 2^n_live times the one before
# it, so without a cap it dominates the entire sweep's runtime -- the full
# ladder is hours, this is minutes. It is cheap in accuracy: refitting the
# measured corpus with the ladder capped here rather than run to device
# exhaustion moves the fitted exponent by under 0.04 on every instance
# (ex8_1_1 0.5003 vs 0.5002, ex4_1_2 1.0756 vs 1.0583). Override for a
# publication-grade run:
#
#   MAX_REGIONS=0 study/refinement/run_sweep.sh <binary>    # no cap
MAX_REGIONS=${MAX_REGIONS-1048576}

# Low-dimensional only: the uniform partition needs N = 2^(k * n_live)
# regions, so above roughly seven live variables the ladder is over before it
# has enough rungs to fit a slope (REFINEMENT_STUDY.md §7.2).
LOW_DIM=(ex4_1_2 ex4_1_5 ex8_1_1 circle ex14_1_1)

# The full parsing corpus. The budget partition's N = 2^B is independent of
# dimension, so every instance is reachable.
CORPUS=(alkyl autocorr_bern20-03 chem circle ex14_1_1 ex14_1_2 ex2_1_1
        ex3_1_1 ex4_1_2 ex4_1_5 ex5_2_2_case1 ex6_1_1 ex7_2_1 ex8_1_1)

run_config() {
  local config=$1
  shift
  local -a instances=("$@")

  local out="$RESULTS/$config"
  mkdir -p "$out"

  local -a flags
  local -a cap=()
  if [[ "$MAX_REGIONS" != "0" ]]; then
    cap=("--max-regions=$MAX_REGIONS")
  fi

  case "$config" in
    declared)
      flags=(--partition=uniform --sigma-rungs=6 --reps=8 "--seed=$SEED" "${cap[@]}")
      ;;
    unit-width)
      # --center-range keeps placements near the origin. Without it a unit box
      # drawn anywhere inside a defaulted +-1000000 root lands out in the
      # padding, where a polynomial's local derivative is astronomical purely
      # from the exponent -- which measures the padding, not the model
      # (REFINEMENT_STUDY.md §7.3).
      flags=(--partition=uniform --fixed-width=1 --center-range=10 --reps=8
             "--seed=$SEED" "${cap[@]}")
      ;;
    budget)
      flags=(--partition=budget --sigma-rungs=6 --reps=8 "--seed=$SEED" "${cap[@]}")
      ;;
    *)
      echo "unknown config: $config" >&2
      exit 2
      ;;
  esac

  {
    echo "{"
    echo "  \"config\": \"$config\","
    echo "  \"date\": \"$(date -Iseconds)\","
    echo "  \"git_revision\": \"$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)\","
    echo "  \"git_dirty\": $(if git -C "$REPO" diff --quiet 2>/dev/null; then echo false; else echo true; fi),"
    echo "  \"binary\": \"$BINARY\","
    echo "  \"gpu\": \"$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)\","
    echo "  \"flags\": \"${flags[*]}\","
    echo "  \"instances\": [$(printf '"%s", ' "${instances[@]}" | sed 's/, $//')]"
    echo "}"
  } > "$out/MANIFEST.json"

  for instance in "${instances[@]}"; do
    local model="$MODELS/$instance.gms"
    if [[ ! -f "$model" ]]; then
      echo "  skip $instance (no $model)" >&2
      continue
    fi
    echo "  $config/$instance"
    # stderr carries the frontend's warnings -- including which variables got
    # a defaulted bound, which is load-bearing for reading the results -- so
    # it is kept beside the CSV rather than discarded.
    "$BINARY" "${flags[@]}" "--out=$out/$instance.csv" "$model" \
      > /dev/null 2> "$out/$instance.log"
  done
}

CONFIGS=("$@")
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  CONFIGS=(declared unit-width budget)
fi

for config in "${CONFIGS[@]}"; do
  echo "== $config"
  case "$config" in
    budget) run_config "$config" "${CORPUS[@]}" ;;
    *)      run_config "$config" "${LOW_DIM[@]}" ;;
  esac
done

echo
echo "done. Report with:"
echo "  study/refinement/report.py $RESULTS/<config>"
