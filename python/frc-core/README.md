# frc-core

frc-core is the compiled core of [`frc`](https://pypi.org/project/frc/), a Python library for Fourier Ring Correlation (FRC) and resolution determination in microscopy. It has no dependencies: it takes any array that supports the buffer protocol, such as a NumPy array, an `array.array` or a `memoryview`, and returns a `memoryview`, which `numpy.asarray` wraps without copying it. For an API that takes and returns NumPy arrays, with thresholds and resolution, use `frc`. For Rust, the same code is the crate [`frc`](https://crates.io/crates/frc).

frc-core has wheels for Windows (x86_64), macOS (universal2, which runs on both Apple Silicon and Intel) and Linux (x86_64, glibc 2.28 and later). One wheel per platform covers CPython 3.11 and later, through the stable ABI, and one free-threaded CPython 3.15t and later, so a new Python version needs no new release. On other platforms, pip builds frc-core from the sdist, which needs a Rust toolchain.

frc-core replaces rustfrc, whose last release is 1.2.0. Its API is the one that was planned for rustfrc 2.0.

## Features

- `binom_split(a, seed=None)` replaces every count n of an array of non-negative integers with a sample k of binomial(n, 0.5), in an int32 array of the same shape. It counts the ones among n random bits, from 16 xoshiro256++ generators at once, with AVX-512 or AVX2 when the CPU has them. A Poisson-distributed count with mean λ becomes one with mean λ / 2, and so does n - k, independently of k.
- `one_frc(img, seed=None)` computes the FRC curve of a square image of counts from that one image: it splits every count with `binom_split`, Fourier transforms both halves, and correlates them over rings of equal distance from the origin. Ring r holds the frequencies f with r <= |f| < r + 1, as DIPlib's `RadialSum` bins them. The curve is a float64 array of length n // 2.
- `two_frc(img1, img2)` computes the FRC curve of two square images of one size.
- `ring_counts(n)` gives the number of frequencies in each ring of an n x n spectrum, as a uint64 array.
- `FrcPlan(n)` holds the FFT plans and the buffers for n x n images. `one_frc` and `two_frc` make one per call; for several images of one size, make one plan and use its `one_frc` and `two_frc` methods.

With a seed, the same input gives the same output on every CPU and any number of threads. Without one, every call draws a fresh seed from the operating system.

```python
import numpy as np
import frc_core

img = np.random.default_rng(0).poisson(100, (1024, 1024))
curve = np.asarray(frc_core.one_frc(img, seed=1))
```

## Performance

On 8 threads of an AMD EPYC 9V45 (Zen 5, AVX-512), with mean counts of 100 to 200:

- `binom_split`: its sampler takes ~1 ms on a 2000x2000 array, against ~49 ms for rustfrc 1.2.
- `FrcPlan.one_frc`: ~18 ms on a 4096x4096 image with a reused plan, ~30 ms with `one_frc`, which makes the plan, against ~1000 ms for `frc.one_frc` 0.1 (rustfrc 1.2's split with DIPlib's FFT and `RadialSum`).

[docs/PERFORMANCE.md](https://github.com/tiptenbrink/frc/blob/main/docs/PERFORMANCE.md) describes where the time goes, what was tried to make each step faster, and the literature behind the choices.

## Installation

```shell
pip install frc-core
```

In a project managed by [uv](https://github.com/astral-sh/uv), add it with `uv add frc-core`.

To build a wheel from a checkout of the [repository](https://github.com/tiptenbrink/frc), run `uvx maturin build --release` in `python/frc-core`. This builds one wheel for CPython 3.11 and later; for free-threaded Python, build with a free-threaded interpreter, e.g. `uvx maturin build --release -i python3.15t`. The wheels land in `target/wheels` at the root of the repository.
