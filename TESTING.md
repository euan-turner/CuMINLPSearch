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
> `cuminlp_gams_test` is the exception: the GAMS frontend is pure host C++, so
> its tests run anywhere, GPU or not.

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

To solve any parseable instance directly, with no per-instance code:

```sh
cmake --build build/dev -t gams_solve
./build/dev/gams_solve test/data/gams/ex8_6_2.gms 300
```

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
