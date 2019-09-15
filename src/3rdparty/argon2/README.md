# Argon2 [![Build Status](https://travis-ci.org/WOnder93/argon2.svg?branch=master)](https://travis-ci.org/WOnder93/argon2)
A multi-arch library implementing the Argon2 password hashing algorithm.

This project is based on the [original source code](https://github.com/P-H-C/phc-winner-argon2) by the Argon2 authors. The goal of this project is to provide efficient Argon2 implementations for various HW architectures (x86, SSE, ARM, PowerPC, ...).

For the x86_64 architecture, the library implements a simple CPU dispatch which automatically selects the best implementation based on CPU flags and quick benchmarks.

# Building
## Using GNU autotools

To prepare the build environment, run:
```bash
autoreconf -i
./configure
```

After that, just run `make` to build the library.

### Running tests
After configuring the build environment, run `make check` to run the tests.

### Architecture options
You can specify the target architecture by passing the `--host=...` flag to `./configure`.

Supported architectures:
 * `x86_64` &ndash; 64-bit x86 architecture
 * `generic` &ndash; use generic C impementation

## Using CMake

To prepare the build environment, run:
```bash
cmake -DCMAKE_BUILD_TYPE=Release .
```

Then you can run `make` to build the library.

## Using QMake/Qt Creator
A [QMake](http://doc.qt.io/qt-4.8/qmake-manual.html) project is also available in the `qmake` directory. You can open it in the [Qt Creator IDE](http://wiki.qt.io/Category:Tools::QtCreator) or build it from terminal:
```bash
cd qmake
# see table below for the list of possible ARCH and CONFIG values
qmake ARCH=... CONFIG+=...
make
```

### Architecture options
For QMake builds you can configure support for different architectures. Use the `ARCH` variable to choose the architecture and the `CONFIG` variable to set additional options.

Supported architectures:
 * `x86_64` &ndash; 64-bit x86 architecture
   * QMake config flags:
     * `USE_SSE2` &ndash; use SSE2 instructions
     * `USE_SSSE3` &ndash; use SSSE3 instructions
     * `USE_XOP` &ndash; use XOP instructions
     * `USE_AVX2` &ndash; use AVX2 instructions
     * `USE_AVX512F` &ndash; use AVX-512F instructions
 * `generic` &ndash; use generic C impementation
