# frc-core

frc-core computes Fourier ring correlation (FRC) curves of images and splits images of photon counts in two, in compiled code. It is the core of [`frc`](https://pypi.org/project/frc/), a Python library that determines the resolution of microscopy images. Use `frc` for an API that takes and returns NumPy arrays and adds thresholds and the resolution. For Rust, the same code is the crate [`frc`](https://crates.io/crates/frc).

frc-core has no dependencies. Its functions take any array that supports the buffer protocol, such as a NumPy array, an `array.array` or a `memoryview`. They return a `memoryview`, which `numpy.asarray` wraps without copying it.

## Installation

```shell
pip install frc-core
```

In a project managed by [uv](https://github.com/astral-sh/uv), add it with `uv add frc-core`.

## Usage

```python
import numpy as np
import frc_core

img = np.random.default_rng(0).poisson(100, (1024, 1024))
curve = np.asarray(frc_core.one_frc(img, seed=1))
```

- `one_frc(img, seed=None, tukey_alpha=0.125, rings="nearest")` computes the FRC curve of one square image of counts. It splits every count as `binom_split` does, multiplies both halves by a Tukey window, Fourier transforms them, and correlates them over rings of equal distance from the origin. With `rings="nearest"`, ring r holds the frequencies f whose distance |f| rounds to r. With `rings="floor"`, it holds those with r <= |f| < r + 1, as DIPlib's `RadialSum` and frc 0.1 and 0.2 bin them. The curve holds a float64 value for each ring r < n // 2 of an n x n image.
- `two_frc(img1, img2)` computes the FRC curve of two square images of one size. It applies no window: multiply both images by one first.
- `FrcPlan(n)` holds the FFT plan and the working memory for n x n images. `one_frc` and `two_frc` make a plan on each call. For several images of one size, make one plan and call its `one_frc` and `two_frc` methods.
- `binom_split(a, seed=None)` splits every count c of an array of non-negative integers. It returns an int32 array of the same shape with a sample k of binomial(c, 0.5) for each count: the number of ones among c random bits. If c is Poisson-distributed with mean λ, then k and c - k are independent and Poisson-distributed with mean λ / 2.
- `ring_counts(n)` returns the number of frequencies in each ring of an n x n spectrum, as a uint64 array.

With the same seed, `binom_split` draws for a square image the samples that `one_frc` draws: `one_frc(img, seed=s, tukey_alpha=0)` is `two_frc(k, img - k)` with `k = binom_split(img, seed=s)`.

### The window

The Fourier transform takes an image to repeat, so its opposite edges meet with a jump, which correlates between two images of one scene. A window that falls to zero at the edges removes the jump. `tukey_alpha` is the fraction of each side that the two cosine tapers of the Tukey window take together, from 0 to 1, and 0 gives no window.

Pass `one_frc` the counts as they were measured. It applies the window to both halves, after the split. The split gives two independent images only of Poisson counts, whose variance is their mean. A count that a window has scaled by w has w times the mean and w² times the variance, and the noise of its two halves correlates, which lowers the FRC of every ring.

## Repeatable results

Without a seed, every call draws a new seed from the operating system. With a seed:

- `binom_split` returns the same samples for the same array on every CPU and any number of threads.
- `one_frc` returns the same curve for the same image on any number of threads. Between CPUs with different vector instructions, the Fourier transform rounds differently, and a curve differs in its last bits.

## Performance

Measured on 2026-11-14 through Python, on 8 threads of an AMD EPYC 9V45 (Zen 5, AVX-512), with Poisson counts of mean 150:

- `FrcPlan.one_frc` of a 4096 x 4096 int32 image takes about 15 ms. `one_frc`, which also makes the plan, takes about 25 ms. `frc.one_frc` 0.2 took about 1000 ms, with rustfrc 1.2's split and DIPlib's FFT and `RadialSum`.
- `FrcPlan.two_frc` of two 4096 x 4096 float32 images takes about 14 ms.
- `binom_split` of a 2000 x 2000 int32 array takes about 1 ms. rustfrc 1.2 took about 49 ms.

Counts that are not int32, and images for `two_frc` that are not float32, are converted first. For a 4096 x 4096 int64 image that adds about 11 ms.

The split uses AVX-512 or AVX2 when the CPU has them, and chooses at run time.

## Wheels

frc-core has wheels for Windows (x86_64), macOS (universal2, which runs on both Apple Silicon and Intel) and Linux (x86_64, glibc 2.28 and later). One wheel per platform covers CPython 3.11 and later through the stable ABI, and one covers free-threaded CPython 3.15t and later, so a new Python version needs no new release. On other platforms, pip builds frc-core from the sdist, which needs a Rust toolchain.

To build a wheel from a checkout of the [repository](https://github.com/tiptenbrink/frc), run `uvx maturin build --release` in `python/frc-core`. This builds one wheel for CPython 3.11 and later. For free-threaded Python, build with a free-threaded interpreter, for example `uvx maturin build --release -i python3.15t`. The wheels land in `target/wheels` at the root of the repository.

## rustfrc

frc-core replaces rustfrc, whose last release is 1.2.0.
