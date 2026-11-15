# frc

frc measures the resolution of microscopy images with Fourier ring correlation (FRC). It works from two images of the same object, or from one image of photon counts (1FRC). The [repository's README](https://github.com/tiptenbrink/frc#readme) explains the method, when 1FRC applies, and which papers to cite.

## Installation

In a project managed by [uv](https://docs.astral.sh/uv/), run `uv add frc`. Otherwise run `pip install frc`, also in a conda environment.

frc needs Python 3.11 or later. It depends on NumPy and on [frc-core](https://pypi.org/project/frc-core/), its compiled core, of the same version.

## Usage

Make an `FrcPlan` for the size of your images, and use it for every image of that size. Pass its `one_frc` a square image of photon counts, after offset and gain correction and before any other processing:

```python
import frc

counts = frc.util.square_image(counts, add_padding=False)
plan = frc.FrcPlan(counts.shape[0])
curve = plan.one_frc(counts)
resolution = plan.resolution(curve)  # in pixels
```

`FrcPlan(n, rings="nearest")` holds the Fourier transform plan and the working memory for images of n × n pixels. Its methods:

- `one_frc(img, seed=None, tukey_alpha=0.125)` splits every count of `img` binomially in two, multiplies both halves by a [Tukey window](https://en.wikipedia.org/wiki/Window_function#Tukey_window), and returns their FRC curve. The same image and `seed` give the same curve.
- `two_frc(img1, img2)` returns the FRC curve of two images of the same object with independent noise. Multiply both by `frc.util.tukey_square(n)` first.
- `resolution(curve, threshold="1/7")` returns the resolution in pixels. It interpolates where the curve first drops below the threshold, which is `"1/7"`, `"half_bit"` or a number. It raises `frc.NoIntersectionException` if the curve does not drop below the threshold.
- `frequencies()` returns the spatial frequency of each ring in 1/pixel. Plot the curve against it to see where it crosses the threshold.

`frc.util` holds `apply_tukey`, `tukey_square`, `tukey_window`, `square_image`, `gaussf` and `intersection`.

## Legacy API

The functions `one_frc`, `two_frc` and `frc_res` are the API of earlier versions of frc. They keep its defaults, so that earlier results can be reproduced. Each default shifts the resolution by one to a few percent:

- `one_frc(img, method=1, seed=None, tukey_alpha=None, rings="floor")` applies no window unless you pass `tukey_alpha`. In earlier versions, you windowed the image yourself before the split, which makes the resolution slightly coarser. With `tukey_alpha`, `one_frc` windows both halves after the split, as `FrcPlan` does. Method 2 splits the image twice, independently.
- `rings` is `"floor"`, which places ring r about half a ring too low. Pass `rings="nearest"` for rings centred on r. `two_frc(img1, img2, rings="floor")` takes the same argument.
- `frc_res(xs, frc_curve, img_size, threshold=None, rings="floor", crossing="band")` estimates the crossing from a band of rings around the threshold, which often gives a slightly finer resolution. Pass `crossing="first"` to interpolate the first crossing instead. `frc_res` also takes `xs` in any unit of frequency, and a threshold that is a function of the frequency.

The [repository's README](https://github.com/tiptenbrink/frc#differences-from-earlier-versions) gives the size of each effect and how it was measured.

When `one_frc` splits an image whose values are not whole numbers, such as a windowed image, it warns with a `frc.NotCountsWarning`. With `tukey_alpha`, it refuses such an image. To keep the results of a windowed image without the warning, switch it off:

```python
import warnings

warnings.simplefilter("ignore", frc.NotCountsWarning)
```
