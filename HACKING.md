# Hacking

Wisdom for building and testing this project as a developer.

## Developer mode

Build system targets that are only useful for developers (tests, lint
targets) are hidden unless the `CuQCQPs_DEVELOPER_MODE` option is enabled.
Developer mode is always on in CI.

### Presets

This project uses [CMake presets][1]. `CMakePresets.json` at the repo root
holds the shared/CI presets; `CMakeUserPresets.json` (gitignored, already
present locally) holds a `dev` preset for day-to-day work:

```sh
cmake --preset=dev
cmake --build --preset=dev
ctest --preset=dev
```

If `CMakeUserPresets.json` ever needs recreating, base it on the `dev-mode`
and `ci-linux` presets from `CMakePresets.json`; see that file for what those
inherit.

> **Note**
> Some editors configure automatically on open and may pick an unwanted
> preset. Make sure your editor only configures when you want it to.

### Developer mode targets

Invoke with the build command plus `-t <target>`:

#### `format-check` and `format-fix`

Run clang-format over the codebase to check or fix formatting.
Customizable via the `FORMAT_PATTERNS` and `FORMAT_COMMAND` cache variables.

See [TESTING.md](TESTING.md) for the full set of commands to run linting,
formatting, and tests.

[1]: https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html
