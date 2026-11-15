# frc-core

frc-core computes Fourier ring correlation (FRC) curves of microscopy images, and the resolution that they give, in compiled code. It is the core of the Python library [frc](https://pypi.org/project/frc/), which offers the same plan on NumPy arrays. The [repository's README](https://github.com/tiptenbrink/frc#readme) explains the method and which papers to cite.

frc-core has no dependencies. It takes any array that supports the buffer protocol, such as a NumPy array, an `array.array` or a `memoryview`. It returns a `memoryview`, which `numpy.asarray` wraps without a copy.

## Installation

In a project managed by [uv](https://docs.astral.sh/uv/), run `uv add frc-core`. Otherwise run `pip install frc-core`.

frc-core has wheels for Linux (x86_64, glibc 2.28 and later), macOS (Apple Silicon and Intel) and Windows (x86_64). One wheel per platform covers CPython 3.11 and later, and one covers free-threaded CPython 3.15t and later (the latter also works on non-free-threaded Python). On other platforms, pip builds frc-core from source, which needs a Rust toolchain.

## Usage

Make an `FrcPlan` for the size of your images, and use it for every image of that size. Pass its `one_image` a square image of photon counts, after offset and gain correction and before any other processing:

```python
import numpy as np
import frc_core

img = np.random.default_rng(0).poisson(100, (1024, 1024))
plan = frc_core.FrcPlan(1024)
curve = np.asarray(plan.one_image(img, seed=1))
crossing = plan.first_crossing(curve)  # a Crossing, or None if the curve does not cross
```

`FrcPlan(n, binning="nearest")` holds the Fourier transform plan and the working memory for images of n × n pixels. Its methods:

- `one_image(img, seed=None, tukey_alpha=0.125)` splits every count of `img` binomially in two, multiplies both halves by a [Tukey window](https://en.wikipedia.org/wiki/Window_function#Tukey_window) with alpha `tukey_alpha`, and returns their FRC curve: a float64 value for each ring r < n // 2. `tukey_alpha=0` applies no window.
- `two_images(img1, img2)` returns the FRC curve of two images. It applies no window: multiply both images by one first.
- `first_crossing(curve, threshold="1/7")` finds the first ring whose FRC is below the threshold and interpolates between that ring and the one before it. It returns a `Crossing` with the `resolution` in pixels, the `ring` at which the curve crosses and the `threshold` there, or `None` if the curve does not drop below the threshold. `threshold` is `"1/7"`, `"half_bit"` or a float.
- `rings` holds the plan's `Rings`.

`Rings(n, binning="nearest")` holds the rings of the spectrum of n × n images: their `binning`, their `size` and, from `counts()`, the number of frequencies in each ring as uint64 values. It needs no Fourier transform, so make one to find the resolution from a curve without a plan. Its `first_crossing` is the plan's. With `binning="nearest"`, ring r holds the frequencies whose distance |f| from the origin rounds to r. With `binning="floor"`, it holds those with r ≤ |f| < r + 1, as earlier versions of frc did.

A plan computes one curve at a time, and threads that share a plan take turns. `binom_split(a, seed=None)` returns an int32 array of the shape of `a` with a sample k of binomial(c, 0.5) for each count c. If c is Poisson-distributed with mean λ, then k and c - k are independent and Poisson-distributed with mean λ / 2.

## Repeatable results

Without a seed, every call draws a new seed from the operating system. With a seed:

- `binom_split` returns the same samples for the same array on every CPU and any number of threads.
- `one_image` returns the same curve for the same image on any number of threads. Between CPUs with different vector instructions, the Fourier transform rounds differently, and a curve differs in its last bits.
- `plan.one_image(img, seed=s, tukey_alpha=0)` is `plan.two_images(k, img - k)` with `k = binom_split(img, seed=s)`.

## Performance

`FrcPlan.one_image` of a 4096 x 4096 int32 image takes about 15 ms, and making the plan about 10 ms more. This was measured on 2026-11-14 through Python, on 8 threads of an AMD EPYC 9V45. Counts that are not int32, and images for `two_images` that are not float32, are converted first. For a 4096 x 4096 int64 image that adds about 11 ms.

frc-core replaces [rustfrc](https://pypi.org/project/rustfrc/), whose last release is 1.2.0.
