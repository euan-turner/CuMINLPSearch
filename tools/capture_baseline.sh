#!/usr/bin/env bash
# Captures stdout/stderr/exit-code for the fixed_examples binaries and every
# gams_solve invocation used to gate MODULE_REFACTOR.md's staged migration.
# Filters out the `Host memory budget:` line (live MemAvailable measurement,
# the only known run-to-run nondeterminism). Usage:
#   tools/capture_baseline.sh <build-dir> <output-dir>
set -euo pipefail

BUILD_DIR=${1:?usage: capture_baseline.sh <build-dir> <output-dir>}
OUT_DIR=${2:?usage: capture_baseline.sh <build-dir> <output-dir>}
GAMS_DIR=${3:-test/data/gams}

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

run_capture() {
  local name="$1"
  shift
  local out="$OUT_DIR/$name"
  mkdir -p "$out"
  set +e
  "$@" >"$out/stdout" 2>"$out/stderr"
  echo "$?" >"$out/exit_code"
  grep -v '^Host memory budget:' "$out/stdout" > "$out/stdout.filtered"
  mv "$out/stdout.filtered" "$out/stdout"
  grep -v '^Host memory budget:' "$out/stderr" > "$out/stderr.filtered"
  mv "$out/stderr.filtered" "$out/stderr"
  set -e
}

EXAMPLES="graph_rosenbrock graph_morse graph_power_series graph_qps graph_nvs09 graph_autocorr_bern20_03"
for ex in $EXAMPLES; do
  run_capture "example_${ex}" "$BUILD_DIR/$ex"
done

GAMS_SOLVE="$BUILD_DIR/gams_solve"

for gms in "$GAMS_DIR"/*.gms; do
  base=$(basename "$gms" .gms)
  run_capture "gams_${base}" "$GAMS_SOLVE" "$gms" 200
done

run_capture "flag_list_policies" "$GAMS_SOLVE" --list-policies
run_capture "flag_help" "$GAMS_SOLVE" --help
run_capture "flag_dump_only" "$GAMS_SOLVE" --dump-only "$GAMS_DIR/ex4_1_2.gms"
run_capture "flag_dump_dag_nodes" "$GAMS_SOLVE" --dump-dag=nodes --dump-only "$GAMS_DIR/ex2_1_1.gms"
run_capture "flag_partition_num" "$GAMS_SOLVE" "$GAMS_DIR/ex4_1_2.gms" 50 --partition-num=4
run_capture "flag_max_cycle_size" "$GAMS_SOLVE" "$GAMS_DIR/ex4_1_2.gms" 50 --max-cycle-size=4
run_capture "flag_nvs01_combo" "$GAMS_SOLVE" "$GAMS_DIR/nvs01.gms" 50 --partition-num=3 --enumerate-cap=5 --sample-points=2 --max-cycle-size=6
run_capture "flag_host_budget" "$GAMS_SOLVE" "$GAMS_DIR/ex8_6_2.gms" 200 --host-budget-bytes=2000000
run_capture "flag_host_budget_bounded_frontier" "$GAMS_SOLVE" "$GAMS_DIR/ex8_6_2.gms" 200 --host-budget-bytes=2000000 --bounded-frontier
run_capture "flag_policy_discrete" "$GAMS_SOLVE" "$GAMS_DIR/nvs01.gms" 50 --policy=discrete
run_capture "flag_policy_no_such" "$GAMS_SOLVE" "$GAMS_DIR/nvs01.gms" 50 --policy=no-such-policy
run_capture "flag_partition_num_bad" "$GAMS_SOLVE" "$GAMS_DIR/nvs01.gms" 50 --partition-num=8x

echo "Captured to $OUT_DIR"
