# frc

frc computes Fourier ring correlation (FRC) curves of microscopy images, and the image resolution that follows from them. FRC compares two images of the same object whose noise is independent: classical FRC, or 2FRC. frc also computes it from one image, which is called single image FRC, or 1FRC. It splits the photon count of every pixel binomially into two images with independent Poisson noise [1].

## Measuring the resolution of one image

1. Install the Python library frc. In a project managed by [uv](https://docs.astral.sh/uv/), run `uv add frc`. frc is on PyPI, so `pip install frc` also works, in a conda environment too.
2. Convert your image to photon counts: subtract the camera offset, divide by the gain, and round to whole numbers. Do nothing else to the image before the split, not even a window.
3. Crop the image to a square.
4. Call `frc.one_frc` with the counts and a `tukey_alpha`. It splits the counts, multiplies both halves by a Tukey window with that alpha, and returns the FRC of each ring of frequencies.
5. Call `frc.frc_res` with the curve and `crossing="first"`. It finds the first ring whose FRC is below the threshold, interpolates between that ring and the one before it, and returns the resolution: one over the frequency of the crossing.

```python
import numpy as np
import frc

# raw holds one camera image, and offset and gain the camera's calibration. one_frc refuses
# negative counts.
counts = np.clip(np.rint((raw - offset) / gain), 0, None).astype(np.int32)
counts = frc.util.square_image(counts, add_padding=False)

curve = frc.one_frc(counts, tukey_alpha=0.125, rings="nearest")
n = counts.shape[0]
xs = np.arange(len(curve)) / n  # the frequency of each ring, in cycles per pixel
resolution, _, _ = frc.frc_res(xs, curve, n, rings="nearest", crossing="first")
print(f"resolution: {resolution:.2f} pixels")
```

Multiply the resolution by the pixel size for a length, or pass `xs` in cycles per unit of length, and the resolution comes out in that unit. For two images of the same object, multiply each by `frc.util.tukey_square(n)` and call `frc.two_frc(img1, img2, rings="nearest")` in place of `one_frc`.

`frc_res` compares the curve with the threshold 1/7 [3]. Pass `threshold="half_bit"` for the 1/2-bit threshold [4], or a function of the frequency for another one. It raises `frc.NoIntersectionException` if the curve does not cross the threshold. Plot `curve` against `xs` to see why.

## The curve

The FRC of ring r is

FRC(r) = Re Σ F1(f) F2(f)* / √(Σ |F1(f)|² Σ |F2(f)|²)

where F1 and F2 are the Fourier transforms of the two images and the sums run over the frequencies f of the ring [2, 5]. At low frequencies, where the object dominates both images, the FRC is near 1. At high frequencies, where independent noise dominates, it falls to about 0. The resolution is the length scale at which the curve drops below its threshold [1, 3].

frc computes the curve for every ring r < n / 2 of an n x n image, and puts ring r at the frequency r / n. With `rings="nearest"`, ring r holds the frequencies whose distance |f| from the origin, in pixels of the spectrum, rounds to r. With `rings="floor"`, it holds those with r <= |f| < r + 1, as DIPlib's `RadialSum` bins them.

## When 1FRC applies

If a pixel's count n is Poisson distributed with mean μ, the binomial split takes k from binomial(n, 1/2) and keeps n − k. Then k and n − k are independent and Poisson distributed with mean μ / 2 each [1]. The two halves are two images of the same object with independent noise, and their FRC is the 1FRC of the image.

The split needs pixel values that are Poisson-distributed counts. Shot-noise limited cameras, such as sCMOS and EMCCD cameras, give them after offset and gain correction. Any deviation lifts the curve to a plateau above 0 at high frequencies, or lowers it below 0:

- Readout noise, and errors in the offset or the gain, give a plateau whose height Rieger et al. derive [1, eqs. 28-32]. A gain that is a few percent off can already change the resolution. A bachelor thesis on 1FRC found the plateau 1 − μ/σ² for pixels with mean μ and variance σ² after the split [6].
- Processing before the split does the same. Rieger et al. state: "It is imperative to split the image data in the processing pipeline when Poisson statistics applies. This is at the raw data level, after gain and offset correction." [1] A filter, a deconvolution or a reconstruction mixes the noise of pixels, and the halves of the split are no longer independent.
- A window multiplies a count by a weight w between 0 and 1. That acts on the pixel as a gain error with g′ = w. Equation 31 of [1] then gives the plateau (w − 1) / (w + 1), which is negative. `one_frc` with a `tukey_alpha` therefore applies the window after the split.

## Choices that change the resolution

Both choices below were measured on simulated images of 256 x 256 pixels, on 2026-11-14. The images held sparse points blurred by a Gaussian of 2 pixels, with Poisson noise. Each choice was compared with two independent acquisitions of half the counts each.

**The window and the split.** Splitting an image that was multiplied by a Tukey window with an alpha of 1/8 lowered the FRC of nearly every ring, by 0.01 on average. The resolution came out 0.6 to 1% coarser at 10 to 200 counts per pixel. Below 1 count per pixel the error grew to a factor 2. Splitting first and windowing both halves gave the curve of the two acquisitions at every count.

**The rings.** With `rings="floor"`, the frequencies of ring r lie about half a ring above r on average, where the curve puts the ring. The resolution came out about 1% coarser than with `rings="nearest"`, whose frequencies lie around r.

**The crossing.** frc 0.1 and 0.2 took the rings whose FRC lies within 0.08 of the threshold, and returned the mean frequency of the first group of consecutive ones in which the curve crosses. That group often reaches past the first crossing, where a noisy curve comes back up. At 10 counts per pixel its resolutions were 2 to 9% finer than the interpolated first crossing, and varied twice as much between splits of one image. `crossing="first"` interpolates the first crossing.

frc 0.1 and 0.2 split the image that you had windowed, summed `"floor"` rings, and searched a band around the threshold. To keep their results, frc's `one_frc` applies no window without a `tukey_alpha`, its rings are `"floor"` unless you pass `rings="nearest"`, and `frc_res` searches the band unless you pass `crossing="first"`. It warns with a `frc.NotCountsWarning` when it splits an image that holds values that are not whole numbers, such as a windowed image. frc-core and the Rust crate apply a window, sum `"nearest"` rings, and interpolate the first crossing by default.

## Packages

This repository holds three software packages, released together at one version:

| Package | Where | What it is |
|---|---|---|
| [`frc`](python/frc) | [PyPI](https://pypi.org/project/frc/) | The Python library: `one_frc`, `two_frc`, thresholds and the resolution, on NumPy arrays. |
| [`frc-core`](python/frc-core) | [PyPI](https://pypi.org/project/frc-core/) | The compiled core of frc, without dependencies: it takes any array through the buffer protocol and returns `memoryview`s. It replaces rustfrc. |
| [`frc`](crates/frc) | [crates.io](https://crates.io/crates/frc) | The Rust library that both Python packages are built from. |

`one_frc` of a 4096 x 4096 image of counts takes about 15 to 25 ms through Python, on 8 threads of an AMD EPYC 9V45 (measured on 2026-11-14). frc 0.2 took about a second.

## References

1. B. Rieger, I. Droste, F. Gerritsma, T. ten Brink and S. Stallinga, "Single image Fourier ring correlation," Optics Express 32(12), 21767-21782 (2024). [doi:10.1364/OE.524683](https://doi.org/10.1364/OE.524683)
2. R. P. J. Nieuwenhuizen, K. A. Lidke, M. Bates, et al., "Measuring image resolution in optical nanoscopy," Nature Methods 10(6), 557-562 (2013). [doi:10.1038/nmeth.2448](https://doi.org/10.1038/nmeth.2448)
3. P. B. Rosenthal and R. Henderson, "Optimal determination of particle orientation, absolute hand, and contrast loss in single-particle electron cryomicroscopy," Journal of Molecular Biology 333(4), 721-745 (2003). [doi:10.1016/j.jmb.2003.07.013](https://doi.org/10.1016/j.jmb.2003.07.013)
4. M. van Heel and M. Schatz, "Fourier shell correlation threshold criteria," Journal of Structural Biology 151(3), 250-262 (2005). [doi:10.1016/j.jsb.2005.05.009](https://doi.org/10.1016/j.jsb.2005.05.009)
5. M. van Heel, "Similarity measures between images," Ultramicroscopy 21(1), 95-100 (1987). [doi:10.1016/0304-3991(87)90010-6](https://doi.org/10.1016/0304-3991(87)90010-6)
6. T. ten Brink, "Fourier Ring Correlation with a single image," bachelor thesis, Delft University of Technology (2022). [resolver.tudelft.nl](http://resolver.tudelft.nl/uuid:abdc31f6-6ecd-4c1b-a0c4-53711829467a)

To cite frc itself, use the metadata in [`CITATION.cff`](CITATION.cff). To cite the 1FRC method, cite [1].

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) describes the layout of the repository, how to build and test it, and how releases work.

## License

All three packages are licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option.
