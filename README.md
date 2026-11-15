# frc

frc measures the resolution of microscopy images with Fourier ring correlation (FRC). Super-resolution microscopy goes beyond the diffraction limit, so its resolution no longer follows from the optics. FRC determines the resolution from the images themselves, by comparing two images of the same object with independent noise [[2]][ref2]. This is also called 2FRC, and in three dimensions Fourier shell correlation (FSC).

frc can also compute the FRC of a single image of photon counts, by splitting it binomially into two images [[1]][ref1]. This is called 1FRC, or single image FRC. frc was originally written for a bachelor thesis on 1FRC at Delft University of Technology [[6]][ref6].

## Measuring the resolution of an image

Install the Python library frc. In a project managed by [uv](https://docs.astral.sh/uv/), run `uv add frc`. `pip install frc` also works, in a conda environment too.

1. Convert the image to photon counts: subtract the camera offset, divide by the gain and round to whole numbers. Do not process the image any further before the split, not even with a window.
2. Crop the image to a square.
3. Make an `frc.FrcPlan` for the size of the image, and compute the 1FRC curve and the resolution with it. Use the same plan for every image of that size.

```python
import numpy as np
import frc

# raw is one camera image, and offset and gain are the camera's calibration.
counts = np.clip(np.rint((raw - offset) / gain), 0, None).astype(np.int32)
counts = frc.util.square_image(counts, add_padding=False)

plan = frc.FrcPlan(counts.shape[0])
curve = plan.one_image(counts)
resolution = plan.resolution(curve)
print(f"resolution: {resolution:.2f} pixels")
```

`one_image` splits the counts and applies a [Tukey window](https://en.wikipedia.org/wiki/Window_function#Tukey_window) to both halves. `resolution` finds where the curve first drops below the threshold and returns one over that spatial frequency, in pixels. Multiply it by the pixel size for a length.

The default threshold is the fixed value 1/7 [[3]][ref3]. Pass `threshold="half_bit"` for the 1/2-bit threshold [[4]][ref4], or a number. If the curve never drops below the threshold, `resolution` raises `frc.NoIntersectionException`. Plotting `curve` against `plan.frequencies()` usually shows why.

If you have two independently measured images, multiply both by `frc.util.tukey_square(n)` and use `plan.two_images(img1, img2)` instead of `one_image`.

## The FRC curve

For two images with Fourier transforms F₁ and F₂, the FRC at spatial frequency q is [[2]][ref2], [[5]][ref5]

```math
\mathrm{FRC}(q) = \frac{\mathrm{Re} \sum_{\mathbf{q} \in \text{ring}} F_1(\mathbf{q})\, F_2(\mathbf{q})^*}{\sqrt{\sum_{\mathbf{q} \in \text{ring}} |F_1(\mathbf{q})|^2 \, \sum_{\mathbf{q} \in \text{ring}} |F_2(\mathbf{q})|^2}},
```

where the sums run over the ring of frequencies **q** at distance q from the origin. At low frequencies the object dominates both images, and the FRC is close to 1. At high frequencies the independent noise dominates, and the FRC drops to around 0. The resolution is one over the spatial frequency at which the curve crosses a threshold [[1]][ref1], [[3]][ref3].

For an image of n × n pixels, frc computes the FRC of the rings r = 0, 1, …, n/2 − 1, and ring r has spatial frequency r/n. With `rings="nearest"`, ring r holds the frequencies whose distance to the origin rounds to r. With `rings="floor"`, it holds the frequencies at a distance d with r ≤ d < r + 1, as in [DIPlib](https://diplib.org)'s `RadialSum`.

## When 1FRC works

Suppose a pixel value N is Poisson distributed with mean λ. The binomial split samples a value X from a binomial distribution with N trials and probability 1/2, and puts N − X in the second image. X and N − X are then independent and both Poisson distributed with mean λ/2 [[1]][ref1]. The two new images are therefore equivalent to two independent measurements with half the exposure, and their FRC is the 1FRC of the original image.

This only holds if the pixel values are Poisson-distributed photon counts. Cameras limited by shot noise, such as sCMOS and EMCCD cameras, give such counts after offset and gain correction. If the pixel values deviate from this, the 1FRC curve no longer drops to 0 at high frequencies, but levels off at a plateau above or below 0:

- Readout noise, or an error in the offset or the gain, causes such a plateau. Rieger et al. derive its height in equations 28 to 32 of [[1]][ref1]. A gain that is a few percent off can already change the resolution. The thesis [[6]][ref6] found that the plateau lies at 1 − μ/σ², where μ is the mean and σ² the variance of a pixel after the split.
- Any processing before the split, such as filtering, deconvolution or reconstruction, mixes the noise of neighbouring pixels. The two halves of the split are then no longer independent. Rieger et al. write: "It is imperative to split the image data in the processing pipeline when Poisson statistics applies. This is at the raw data level, after gain and offset correction." [[1]][ref1]
- A window multiplies each count by a weight w between 0 and 1, which acts like a gain of w. By equation 31 of [[1]][ref1], the plateau is then (w − 1)/(w + 1), which is below 0. This is why `FrcPlan.one_image` applies its window after the split.

## Differences from earlier versions

Earlier versions of frc made three choices that shift the resolution by one to a few percent. `FrcPlan`, frc-core and the Rust crate make the new choices. The legacy functions `one_frc`, `two_frc` and `frc_res` of frc keep the old choices by default, so that they reproduce earlier results. They make the new choices with `tukey_alpha`, `rings="nearest"` and `crossing="first"`.

| | Earlier versions | Now | Effect of the old choice on the resolution |
|---|---|---|---|
| Window | applied by you, before the split | applied by `one_image` to both halves, after the split | 0.6 to 1% coarser at 10 to 200 counts per pixel, up to a factor 2 below 1 count per pixel |
| Rings | `"floor"` | `"nearest"` | about 1% coarser |
| Crossing | the mean frequency of a band of rings near the threshold | the first crossing, interpolated between two rings | 2 to 9% finer at 10 counts per pixel, with twice the spread between splits |

We measured these effects in November 2026 on simulated images of 256 × 256 pixels: sparse points blurred by a Gaussian of 2 pixels, with Poisson noise. We compared each 1FRC with the 2FRC of two independent images with half the counts each. With the window applied after the split, the 1FRC matched the 2FRC at every count level.

With `rings="floor"`, the frequencies in ring r lie on average half a ring above r, but the curve places the ring at r. The band of rings that earlier versions averaged often reaches past the first crossing, where a noisy curve rises above the threshold again.

When the legacy `one_frc` splits an image whose values are not whole numbers, such as an image that was already windowed, it warns with a `frc.NotCountsWarning`. `FrcPlan.one_image` refuses such an image.

## Packages

This repository holds three packages, which are released together under one version number:

| Package | Where | What it is |
|---|---|---|
| [`frc`](python/frc) | [PyPI](https://pypi.org/project/frc/) | The Python library: `FrcPlan` on NumPy arrays, and the legacy functions of earlier versions. |
| [`frc-core`](python/frc-core) | [PyPI](https://pypi.org/project/frc-core/) | The compiled core of frc, without dependencies. It takes any array that supports the buffer protocol and returns `memoryview`s. It replaces rustfrc. |
| [`frc`](crates/frc) | [crates.io](https://crates.io/crates/frc) | The Rust library that both Python packages are built from. |

The 1FRC of a 4096 × 4096 image takes about 15 ms from Python with a plan, and about 25 ms with a new plan for each image, on 8 threads of an AMD EPYC 9V45 (measured in November 2026). Earlier versions of frc took about a second.

## References

1. B. Rieger, I. Droste, F. Gerritsma, T. ten Brink and S. Stallinga, "Single image Fourier ring correlation," Optics Express 32(12), 21767-21782 (2024). [doi:10.1364/OE.524683](https://doi.org/10.1364/OE.524683)
2. R. P. J. Nieuwenhuizen, K. A. Lidke, M. Bates, et al., "Measuring image resolution in optical nanoscopy," Nature Methods 10(6), 557-562 (2013). [doi:10.1038/nmeth.2448](https://doi.org/10.1038/nmeth.2448)
3. P. B. Rosenthal and R. Henderson, "Optimal determination of particle orientation, absolute hand, and contrast loss in single-particle electron cryomicroscopy," Journal of Molecular Biology 333(4), 721-745 (2003). [doi:10.1016/j.jmb.2003.07.013](https://doi.org/10.1016/j.jmb.2003.07.013)
4. M. van Heel and M. Schatz, "Fourier shell correlation threshold criteria," Journal of Structural Biology 151(3), 250-262 (2005). [doi:10.1016/j.jsb.2005.05.009](https://doi.org/10.1016/j.jsb.2005.05.009)
5. M. van Heel, "Similarity measures between images," Ultramicroscopy 21(1), 95-100 (1987). [doi:10.1016/0304-3991(87)90010-6](https://doi.org/10.1016/0304-3991(87)90010-6)
6. T. ten Brink, "Fourier Ring Correlation with a single image," bachelor thesis, Delft University of Technology (2022). [resolver.tudelft.nl](https://resolver.tudelft.nl/uuid:abdc31f6-6ecd-4c1b-a0c4-53711829467a)

To cite frc itself, use the metadata in [`CITATION.cff`](CITATION.cff). To cite the 1FRC method, cite [[1]][ref1].

## History

frc started in September 2021 as part of a bachelor thesis on 1FRC at Delft University of Technology [[6]][ref6]. The thesis used three packages written for it: frc for the FRC itself, [rustfrc](https://pypi.org/project/rustfrc/), an extension module in Rust for the binomial split, and [AnalyzeFRC](https://doi.org/10.5281/zenodo.6457673) for the analysis of the curves. [frc 0.1.0](https://doi.org/10.5281/zenodo.6461444) was released in April 2022 so that the thesis could cite it, and the thesis was completed in May 2022. In 2024, Rieger et al. published the 1FRC method [[1]][ref1].

frc 0.1.1 added Python 3.12 in 2024. In late 2026, frc 0.1.2 to 0.2.0 brought it up to date with current versions of Python, NumPy and DIPlib. After that, the FRC computation was moved to Rust, and rustfrc was replaced by frc-core and the Rust crate frc. These now live in this repository together with frc, including the history of rustfrc. frc no longer depends on DIPlib or SciPy.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) describes the layout of the repository, how to build and test it, and how releases work.

## License

All three packages are licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.

[ref1]: https://doi.org/10.1364/OE.524683
[ref2]: https://doi.org/10.1038/nmeth.2448
[ref3]: https://doi.org/10.1016/j.jmb.2003.07.013
[ref4]: https://doi.org/10.1016/j.jsb.2005.05.009
[ref5]: https://doi.org/10.1016/0304-3991(87)90010-6
[ref6]: https://resolver.tudelft.nl/uuid:abdc31f6-6ecd-4c1b-a0c4-53711829467a
