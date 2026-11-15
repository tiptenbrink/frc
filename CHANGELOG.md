# Changelog

## Unreleased

frc now computes the FRC in Rust, and this repository releases three packages at one version: the Python library `frc`, its compiled core `frc-core` on PyPI, and the Rust crate `frc` on crates.io. frc depends only on NumPy and frc-core. frc-core takes the place of rustfrc, whose code moved here with its history.

### Added

- `FrcPlan` in frc: the new main API, on NumPy arrays. Make one plan for a size, and call its `one_frc`, `two_frc` and `resolution` for every image of that size, and its `frequencies` for the spatial frequency of each ring. It makes the corrected choices of frc-core: it applies the window after the split, sums `"nearest"` rings, and interpolates the first crossing. `resolution` returns the resolution in pixels, and raises `NoIntersectionException` if the curve does not drop below the threshold. The functions `one_frc`, `two_frc` and `frc_res` are now the legacy API, and keep their defaults.
- The Python package `frc-core`: the compiled core of frc, without dependencies. Its functions take any array that supports the buffer protocol, of booleans, integers or floats in native byte order, and return `memoryview`s: `binom_split(a, seed=None)`, `one_frc(img, seed=None, tukey_alpha=0.125)`, `two_frc(img1, img2)`, `resolution(curve, n)`, `ring_counts(n)` and `FrcPlan(n)`, whose `one_frc`, `two_frc` and `resolution` reuse its FFT plans and buffers for images of one size. It has type stubs, and wheels for CPython 3.11 and later through the stable ABI and for free-threaded CPython 3.15t and later.
- The Rust crate `frc`, which frc-core is built from: `binom_split`, `FrcPlan` with `one_frc`, `two_frc` and their ring sums, `Window`, `ring_counts`, and `first_crossing` with `Threshold` and `Crossing`.
- A `crossing` argument for frc's `frc_res`. With `crossing="first"`, it finds the first ring whose FRC is below the threshold and interpolates between that ring and the one before it, as frc-core's `resolution` and the crate's `first_crossing` do. With `crossing="band"`, the default, it searches as before: it takes the mean frequency of the first group of rings within 0.08 of the threshold in which the curve crosses. That group often reaches past the first crossing. At 10 counts per pixel, its resolutions were 2 to 9% finer than the first crossing and varied twice as much between splits.
- `Rings` in the crate, and a `rings` argument for frc-core's `FrcPlan`, `one_frc`, `two_frc` and `ring_counts` and for frc's `one_frc`, `two_frc`, `half_bit_threshold` and `frc_res`. It says which frequencies make up ring r. `"floor"` takes those with r <= |f| < r + 1, the bins of DIPlib's `RadialSum`. Their frequencies lie about half a ring above r, where a curve's value r is plotted, and in a simulation the resolution came out about 1% too coarse. `"nearest"` takes the frequencies whose distance |f| rounds to r. In frc-core and in the crate, `"nearest"` is the default. In frc, `"floor"` is the default, as before.
- A `tukey_alpha` for `one_frc`. With it, `one_frc` takes the counts as they were measured, splits them, and multiplies both halves by a Tukey window. Splitting an image that was windowed before, as frc documented until now, lowers the FRC: a count multiplied by a window is not a Poisson count, and the noise of its two halves correlates. In a simulation with an alpha of 1/8, that lowered the FRC of nearly every ring, by 0.01 on average, and gave a resolution that was 0.6 to 1% too coarse. In frc-core and in the crate, the window is the default. In frc, `one_frc` without a `tukey_alpha` applies no window, as before.
- A `seed` for `one_frc`. With a seed, the same image gives the same curve on any number of threads: that of `two_frc` for the two halves that `frc_core.binom_split` makes with the same seed. Without one, every call splits with a new seed.
- `utility.tukey_window(length, alpha)`, the one-dimensional window that `tukey_square` uses.
- Support for Python 3.15 and free-threaded 3.15t.

### Changed

- `one_frc` and `two_frc` run in Rust: the split, the Fourier transforms and the ring sums, in two passes over memory. `one_frc` of a 4096 x 4096 image takes about 26 ms instead of about 1 s, on 8 threads of an AMD EPYC 9V45. Rings hold the same frequencies as DIPlib's `RadialSum`, which frc used before, unless you pass `rings="nearest"`.
- The binomial split samples binomial(n, 0.5) exactly, by counting the ones among n random bits from 16 xoshiro256++ generators at once, with AVX-512 or AVX2 when the CPU has them, chosen at run time.
- `one_frc` without a `tukey_alpha` warns with a `NotCountsWarning`, which can be ignored on its own, when its image holds values that are not whole numbers, as a windowed image does. It splits such an image as before: the first half is a sample of the truncated values, and the second half is the rest of the image. With a `tukey_alpha` it refuses such an image.
- `gaussf` filters with NumPy. For sigmas of 0.8 and more it gives what DIPlib gave, to within 1e-14; for smaller sigmas, DIPlib used a Fourier method instead.
- `tukey_square` computes its window with NumPy, as SciPy did.
- frc no longer caps the Python version.
- frc-core's `ring_counts` and `FrcPlan` count the frequencies of each ring on all threads, over an eighth of the spectrum. At 4096 x 4096, `ring_counts` takes 0.7 ms instead of 5.4 ms, and a new plan 6.7 ms instead of 13.4 ms, on 6 threads of a Core Ultra 7 155H.
- frc is licensed under MIT or Apache-2.0, at your option, instead of MIT alone, as are frc-core and the crate frc.

### Fixed

- The FRC of the first ring, which holds the origin alone, is exactly 1. Earlier versions returned 1 with a rounding error of either sign, and `frc_res` with `threshold='half_bit'` found another resolution, or none, when the value fell below 1.
- `frc_res` with `threshold='half_bit'` takes the threshold of ring r for value r of the curve. It took the threshold of ring `int(x * img_size)` for the frequency x of each value, which is ring r only for `xs` in cycles per pixel. For `xs` in cycles per unit of length, as frc's README showed, it compared the curve with the thresholds of other rings. With 30 nm pixels, the resolution came out 2194 nm instead of 668 nm. For sizes such as 100, the truncation also took ring r - 1 at some rings.
- `half_bit_threshold` and `frc_res` with `threshold='half_bit'` compute the thresholds of a size once. At 4096 x 4096, `frc_res` took 19 ms on every call.
- `utility.intersection` evaluates a first curve that is passed as a function. It evaluated the second curve in its place.
- `utility.intersection` and `frc_res` no longer change the first value of the curve that they are given when the curve starts on its threshold.

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
