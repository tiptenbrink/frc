# frc-core

frc-core computes Fourier ring correlation (FRC) curves of microscopy images, and splits images of photon counts binomially in two, in compiled code. It is the core of the Python library [frc](https://pypi.org/project/frc/), which adds thresholds and the resolution on NumPy arrays. The [repository's README](https://github.com/tiptenbrink/frc#readme) explains the method and which papers to cite.

frc-core has no dependencies. Its functions take any array that supports the buffer protocol, such as a NumPy array, an `array.array` or a `memoryview`. They return a `memoryview`, which `numpy.asarray` wraps without a copy.

## Installation

In a project managed by [uv](https://docs.astral.sh/uv/), run `uv add frc-core`. Otherwise run `pip install frc-core`.

frc-core has wheels for Linux (x86_64, glibc 2.28 and later), macOS (Apple Silicon and Intel) and Windows (x86_64). One wheel per platform covers CPython 3.11 and later, and one covers free-threaded CPython 3.15t and later. On other platforms, pip builds frc-core from source, which needs a Rust toolchain.

## Usage

Pass `one_frc` a square image of photon counts, after offset and gain correction and before any other processing:

```python
import numpy as np
import frc_core

img = np.random.default_rng(0).poisson(100, (1024, 1024))
curve = np.asarray(frc_core.one_frc(img, seed=1))
```

- `one_frc(img, seed=None, tukey_alpha=0.125, rings="nearest")` splits every count of `img` binomially in two, multiplies both halves by a [Tukey window](https://en.wikipedia.org/wiki/Window_function#Tukey_window) with alpha `tukey_alpha`, and returns their FRC curve: a float64 value for each ring r < n // 2 of the n x n image. `tukey_alpha=0` applies no window.
- `two_frc(img1, img2, rings="nearest")` returns the FRC curve of two square images of one size. It applies no window: multiply both images by one first.
- `binom_split(a, seed=None)` returns an int32 array of the shape of `a` with a sample k of binomial(c, 0.5) for each count c. If c is Poisson-distributed with mean λ, then k and c - k are independent and Poisson-distributed with mean λ / 2.
- `resolution(curve, n, threshold="1/7", rings="nearest")` returns the resolution of an n x n image in pixels, and the threshold at the crossing, or `None` if the curve does not drop below the threshold. It finds the first ring whose FRC is below the threshold and interpolates between that ring and the one before it. `threshold` is `"1/7"`, `"half_bit"` or a float.
- `ring_counts(n, rings="nearest")` returns the number of frequencies in each ring of an n x n spectrum, as uint64 values.
- `FrcPlan(n, rings="nearest")` holds the FFT plan and the working memory for n x n images. `one_frc` and `two_frc` make one on each call. For several images of one size, make one plan and call its `one_frc`, `two_frc` and `resolution` methods.

With `rings="nearest"`, ring r holds the frequencies whose distance |f| from the origin rounds to r. With `rings="floor"`, it holds those with r ≤ |f| < r + 1, as earlier versions of frc did.

## Repeatable results

Without a seed, every call draws a new seed from the operating system. With a seed:

- `binom_split` returns the same samples for the same array on every CPU and any number of threads.
- `one_frc` returns the same curve for the same image on any number of threads. Between CPUs with different vector instructions, the Fourier transform rounds differently, and a curve differs in its last bits.
- `one_frc(img, seed=s, tukey_alpha=0)` is `two_frc(k, img - k)` with `k = binom_split(img, seed=s)`, for a square image.

## Performance

`FrcPlan.one_frc` of a 4096 x 4096 int32 image takes about 15 ms, and `one_frc`, which also makes the plan, about 25 ms. This was measured on 2026-11-14 through Python, on 8 threads of an AMD EPYC 9V45. Counts that are not int32, and images for `two_frc` that are not float32, are converted first. For a 4096 x 4096 int64 image that adds about 11 ms.

frc-core replaces [rustfrc](https://pypi.org/project/rustfrc/), whose last release is 1.2.0.
