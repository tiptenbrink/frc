# Changelog

## Unreleased

frc now computes the FRC in Rust, and this repository releases three packages at one version: the Python library `frc`, its compiled core `frc-core` on PyPI, and the Rust crate `frc` on crates.io. frc depends only on NumPy and frc-core. frc-core takes the place of rustfrc, whose code moved here with its history.

### Added

- The Python package `frc-core`: the compiled core of frc, without dependencies. Its functions take any array that supports the buffer protocol, of booleans, integers or floats in native byte order, and return `memoryview`s: `binom_split(a, seed=None)`, `one_frc(img, seed=None)`, `two_frc(img1, img2)`, `ring_counts(n)` and `FrcPlan(n)`, whose `one_frc` and `two_frc` reuse its FFT plans and buffers for images of one size. It has type stubs, and wheels for CPython 3.11 and later through the stable ABI and for free-threaded CPython 3.15t and later.
- The Rust crate `frc`, which frc-core is built from: `binom_split`, `FrcPlan` with `one_frc`, `two_frc` and their ring sums, and `ring_counts`.
- A `seed` for `one_frc`. With a seed, the same image gives the same curve on every CPU and any number of threads; without one, every call splits with a fresh seed.
- `utility.tukey_window(length, alpha)`, the one-dimensional window that `tukey_square` uses.
- Support for Python 3.15 and free-threaded 3.15t.

### Changed

- `one_frc` and `two_frc` run in Rust: the split, the Fourier transforms and the ring sums, in two passes over memory. `one_frc` on a 4096 x 4096 image takes about 30 ms instead of about 1 s, on 8 threads of an AMD EPYC 9V45. Rings hold the same frequencies as DIPlib's `RadialSum`, which frc used before.
- The binomial split samples binomial(n, 0.5) exactly, by counting the ones among n random bits from 16 xoshiro256++ generators at once, with AVX-512 or AVX2 when the CPU has them, chosen at run time.
- `one_frc` truncates float images to int32 counts before the split, for both halves; earlier versions kept the fractional part in the second half.
- `gaussf` filters with NumPy. For sigmas of 0.8 and more it gives what DIPlib gave, to within 1e-14; for smaller sigmas, DIPlib used a Fourier method instead.
- `tukey_square` computes its window with NumPy, as SciPy did.
- frc no longer caps the Python version.

### Removed

- DIPlib, SciPy and rustfrc as dependencies, and with DIPlib, DIPlib images as input: pass NumPy arrays, for example `np.asarray(dip_image)`.
- Python 3.10 support. It has reached its end of life, and frc-core requires Python 3.11, the first stable ABI with the buffer protocol.

## 0.2.0 - 2026-11-14

### Added

- Support for Python 3.13 and 3.14, with DIPlib 3.5.2 and 3.6.

### Changed

- frc requires rustfrc 1.2 or later, which brings wheels for every CPython from 3.9 on.
- The oldest NumPy, SciPy and DIPlib allowed on each Python is the oldest release with wheels for it, so installers no longer try to build one from source.

### Removed

- Python 3.8 and 3.9 support. Both have reached their end of life, and no pytest that runs on them fixes GHSA-6w46-j5rx-g56g.

## 0.1.3 - 2026-11-01

### Changed

- Releases upload PEP 740 attestations, so PyPI shows the workflow that built and published each file. 0.1.2 was published without them.

## 0.1.2 - 2026-10-31

### Fixed

- `half_bit_threshold` and `frc_res(..., threshold="half_bit")` no longer fail with `module 'numpy' has no attribute 'int'` on NumPy 1.24 and newer ([#3](https://github.com/tiptenbrink/frc/issues/3)).
- On Python 3.8-3.10, DIPlib is capped at the last release with wheels for that Python, so installers no longer pick a DIPlib they cannot install.
- The license metadata says MIT, as `LICENSE` does.

### Changed

- NumPy, SciPy, DIPlib and rustfrc no longer have an upper version bound, so NumPy 2 is allowed.
- Releases are approved with a security key through [touchgate](https://github.com/borink-org/touchgate) and published with Trusted Publishing and PEP 740 attestations.

## 0.1.1 - 2024-04-03

### Added

- Python 3.12 support, by updating to rustfrc 1.1.5.

### Removed

- Python 3.7 support.

## 0.1.0 - 2022-04-14

- GitHub release for citation.
