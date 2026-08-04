# Testing

Commands for running linting, formatting, and tests locally. All commands
assume the `dev` preset (from `CMakeUserPresets.json`) is configured:

```sh
cmake --preset=dev
```

## Build

```sh
cmake --build --preset=dev
```

## Tests

```sh
ctest --preset=dev
```

> **Note**
> `cuminlp_cuinterval_test` and `cuminlp_rosenbrock_test` launch real CUDA
> kernels, so they (and configuring at all, since `CMAKE_CUDA_ARCHITECTURES`
> defaults to `native`) require an actual CUDA-capable GPU. Their host-side
> rounding oracle (`CpuRounding`) uses `<xmmintrin.h>`, so those two targets
> are x86-only.
>
> `cuminlp_gams_test`, `cuminlp_dag_test`, `cuminlp_decode_test`,
> `cuminlp_composition_policy_test`, `cuminlp_policy_catalogue_test` and
> `cuminlp_host_budget_test` are the exceptions: the frontend, the DAG, the
> policy, the sidx decode and the bounded frontier are all pure host C++, so
> their tests run anywhere, GPU or not.
>
> `minlp_status_test` is a third kind: `tools/minlp_status.py`'s log scraper,
> run through CTest by whatever Python 3 CMake finds. Configuring without one
> skips it (with a message) rather than failing.

## GAMS frontend

`cuminlp_gams_test` covers expression parsing, objective-variable elimination,
and agreement with the hand-built ex4_1_2/ex8_6_2 DAGs. Its fixtures live in
`test/data/gams/`.

To check coverage over a whole corpus (e.g. an unpacked
[MINLPLib](https://www.minlplib.org) `.gms` archive) and see what is rejected
and why:

```sh
cmake --build build/dev -t gams_report
./build/dev/gams_report path/to/gms/directory
```

The summary groups rejections into a histogram of reasons. A count on its own
is not something you can open, so `--list` names the files behind each row (and
drops the 25-row cap, since the one-instance reasons are the cheapest to fix):

```sh
./build/dev/gams_report --list path/to/gms/directory
```

`--list` also names the instances that *parsed* but gave an integer or binary
variable the 1e6 default bound — they appear in no rejection table, but a
2e6-wide integer domain is a search-quality cliff worth knowing about before
quoting a benchmark built on them. `--reject-discrete` reproduces the
pre-integrality baseline for an A/B.

To solve any parseable instance directly, with no per-instance code:

```sh
cmake --build build/dev -t gams_solve
./build/dev/gams_solve test/data/gams/ex8_6_2.gms 300
```

`Problem::validate()` proves the lowered DAG is well-formed, not that it is the
*right* DAG — outside the two instances with a hand-built oracle in
`cuminlp_gams_test`, reading the expression back is the only check there is.
`--dump-dag` prints it, and `--dump-only` stops before the solve, so it needs no
GPU:

```sh
./build/dev/gams_solve --dump-only test/data/gams/nvs01.gms
./build/dev/gams_solve --dump-only --dump-dag=nodes test/data/gams/nvs01.gms
```

`infix` (the default) reconstructs the expression for comparison against the
`.gms` source; `nodes` lists the DAG in SSA order, showing shared
subexpressions once. Use `nodes` on large instances — infix rendering is capped
and degrades to a placeholder past ~4000 operators.

## Formatting

Check formatting (fails and lists offending files if anything is
misformatted, without changing them):

```sh
cmake --build build/dev -t format-check
```

Fix formatting in place:

```sh
cmake --build build/dev -t format-fix
```

## Static analysis

### clang-tidy

Runs automatically for every compiled translation unit when
`CMAKE_CXX_CLANG_TIDY` is set at configure time. Re-configure the `dev` build
with it enabled, then build:

```sh
cmake --preset=dev -D CMAKE_CXX_CLANG_TIDY="clang-tidy;--header-filter=^$(pwd)/"
cmake --build --preset=dev
```

Unset it (reconfigure without the `-D`) to go back to plain builds.

### cppcheck

Same idea, via `CMAKE_CXX_CPPCHECK`:

```sh
cmake --preset=dev -D CMAKE_CXX_CPPCHECK="cppcheck;--inline-suppr"
cmake --build --preset=dev
```

> **Note**
> cppcheck versions older than ~2.x don't understand some of the flags in
> this project's release compile options (e.g. `-U_FORTIFY_SOURCE`) and will
> fail with `unrecognized command line option`. If that happens locally,
> check `cppcheck --version` and update it.

### Sanitizers (ASan/UBSan)

Separate build directory, since sanitizer instrumentation shouldn't mix with
the regular dev build:

```sh
cmake --preset=ci-sanitize
cmake --build build/sanitize -j2
ctest --test-dir build/sanitize --output-on-failure
```

## Combined preset

`ci-ubuntu` bundles clang-tidy + cppcheck + the standard release flags in one
preset:

```sh
cmake --preset=ci-ubuntu
cmake --build build -j2
ctest --test-dir build --output-on-failure
```
