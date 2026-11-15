# frc

frc computes Fourier ring correlation (FRC) curves of microscopy images and the resolution that follows from them. It works from two images of the same object, or from one image of photon counts (1FRC). The [repository's README](https://github.com/tiptenbrink/frc#readme) explains the method, when 1FRC applies, and which papers to cite.

## Installation

In a project managed by [uv](https://docs.astral.sh/uv/), run `uv add frc`. Otherwise run `pip install frc`, also in a conda environment.

frc needs Python 3.11 or later. It depends on NumPy and on [frc-core](https://pypi.org/project/frc-core/), its compiled core, of the same version.

## Usage

Pass `one_frc` a square image of photon counts, after offset and gain correction and before any other processing. Then pass the curve to `frc_res`:

```python
import numpy as np
import frc

counts = frc.util.square_image(counts, add_padding=False)
curve = frc.one_frc(counts, tukey_alpha=0.125, rings="nearest")
n = counts.shape[0]
xs = np.arange(len(curve)) / n  # the frequency of each ring, in cycles per pixel
resolution, _, _ = frc.frc_res(xs, curve, n, rings="nearest", crossing="first")  # in pixels
```

- `one_frc(img, method=1, seed=None, tukey_alpha=None, rings="floor")` splits every count of `img` binomially in two, multiplies both halves by a [Tukey window](https://en.wikipedia.org/wiki/Window_function#Tukey_window) with alpha `tukey_alpha`, and returns their FRC curve. Method 2 splits the image twice, independently. The same image and `seed` give the same curve.
- `two_frc(img1, img2, rings="floor")` returns the FRC curve of two images. Multiply both by `frc.util.tukey_square(n)` first.
- `frc_res(xs, frc_curve, img_size, threshold=None, rings="floor", crossing="band")` returns the resolution, the value of the threshold where the curve crosses it, and the threshold as a function. `xs` holds the frequency of each ring in any unit, and the resolution comes out in the reciprocal unit. `threshold` is `"1/7"`, the default, `"half_bit"`, or a function of the frequency. Pass the rings that the curve was computed with. `crossing="first"` interpolates the first point at which the curve drops below the threshold.
- `frc.util` holds `apply_tukey`, `tukey_square`, `tukey_window`, `square_image`, `gaussf` and `intersection`.

`frc_res` raises `frc.NoIntersectionException` if the curve does not cross its threshold. Plot `curve` against `xs` to see why.

## Defaults that reproduce earlier versions

Earlier versions of frc made three choices that shift the resolution by one to a few percent. So that earlier results can be reproduced, frc still makes these choices by default:

- `one_frc` applies no window unless you pass `tukey_alpha`. In earlier versions, you windowed the image yourself before the split, which makes the resolution slightly coarser. With `tukey_alpha`, `one_frc` windows both halves after the split.
- `rings` is `"floor"`, which places ring r about half a ring too low. Pass `rings="nearest"` for rings centred on r.
- `frc_res` estimates the crossing from a band of rings around the threshold, which often gives a slightly finer resolution. Pass `crossing="first"` to interpolate the first crossing instead.

The [repository's README](https://github.com/tiptenbrink/frc#differences-from-earlier-versions) gives the size of each effect and how it was measured.

When `one_frc` splits an image whose values are not whole numbers, such as a windowed image, it warns with a `frc.NotCountsWarning`. With `tukey_alpha`, it refuses such an image. To keep the results of a windowed image without the warning, switch it off:

```python
import warnings

warnings.simplefilter("ignore", frc.NotCountsWarning)
```
