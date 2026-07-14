# Building with CMake

This project doesn't require any special command-line flags to build to keep
things simple.

Here are the steps for building in release mode:

```sh
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
cmake --build build
```

For day-to-day development (tests, linting, sanitizers), see
[TESTING.md](TESTING.md).
