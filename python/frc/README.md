# frc

`frc` computes Fourier ring correlation (FRC) curves of images and the resolution that follows from them. Microscopy uses it to determine the resolution of images, in particular of images taken with super-resolution techniques. It works from two images of one scene, or from one image of photon counts (1FRC).

## Installation

```shell
pip install frc
```

frc depends on [NumPy](https://numpy.org/) and on [frc-core](https://pypi.org/project/frc-core/), its compiled core, of the same version. frc-core has no dependencies, and one wheel per platform covers every CPython from 3.11 on.

## Usage

The example below computes the FRC curve of one image and its resolution at the 1/7 threshold. It assumes that `img` holds the photon counts of your image and that `scale` is the number of pixels per unit of length, and it plots the result with matplotlib.

```python
import frc
import numpy as np
import matplotlib.pyplot as plt

... # get an image and scale

img = np.array(img)
img = frc.util.square_image(img, add_padding=False)
# Apply 1FRC technique: one_frc splits the counts, and windows both halves afterwards
frc_curve = frc.one_frc(img, tukey_alpha=0.125)

img_size = img.shape[0]
xs_pix = np.arange(len(frc_curve)) / img_size
# scale has units [pixels <length unit>^-1] corresponding to original image
xs_nm_freq = xs_pix * scale
frc_res, res_y, thres = frc.frc_res(xs_nm_freq, frc_curve, img_size)
plt.plot(xs_nm_freq, thres(xs_nm_freq))
plt.plot(xs_nm_freq, frc_curve)
plt.show()
```

For two images of one scene, window both with `frc.util.apply_tukey` and call `frc.two_frc(img1, img2)` in place of `frc.one_frc`.

### The window and the split

A window that falls to zero at the edges of an image keeps those edges out of its Fourier transform, where they would correlate between the two images. For `two_frc`, apply it yourself.

For `one_frc`, pass the counts as they were measured and a `tukey_alpha`. `one_frc` then applies the window after it has split the counts. The order matters: the split gives two independent images only of Poisson counts, whose variance is their mean. A count that a window has scaled by w has w times the mean and w² times the variance, and the noise of its two halves correlates. In a simulation with a Tukey window with an alpha of 1/8, splitting windowed images lowered the FRC of nearly every ring, by 0.01 on average, and gave a resolution that was 0.6 to 1% too coarse. Splitting first gave the curve of two acquisitions with half the counts each.

frc 0.1 and 0.2 documented the other order, `frc.one_frc(frc.util.apply_tukey(img))`. That call still works as it did, and warns with a `frc.NotCountsWarning`. Without a `tukey_alpha`, `one_frc` applies no window. To keep the results of the old order without the warning, for example to compare with earlier measurements, switch it off:

```python
import warnings

warnings.simplefilter("ignore", frc.NotCountsWarning)
```

### Troubleshooting

`frc_res` raises `NoIntersectionException` if the curve does not cross the threshold. Plot the curve without `frc_res` to see whether it does:

```python
import frc
import numpy as np
import matplotlib.pyplot as plt

img = ... # get an image and scale

# Apply 1FRC technique
frc_curve = frc.one_frc(img, tukey_alpha=0.125)
img_size = img.shape[0]
xs_pix = np.arange(len(frc_curve)) / img_size
# scale has units [pixels <length unit>^-1] corresponding to original image
xs_nm_freq = xs_pix * scale
plt.plot(xs_nm_freq, frc_curve)
plt.show()
```

1FRC fails for images with much noise that is not Poisson noise. The paper below describes a correction.

## Fourier ring correlation

Fourier ring correlation, introduced in 1982, compares two images of one scene whose noise is independent. In the Fourier transforms of the images, the structure of the scene dominates the low spatial frequencies, and noise dominates the high ones.

The frequencies at one distance from the origin form a ring. The correlation between the two transforms over a ring, as a function of the ring's frequency, is the FRC curve.

Above some frequency, the signal cannot be told from the noise. That frequency depends on the threshold that the curve is compared with. frc has the 1/7 (0.143) threshold and the 1/2-bit threshold, and it finds where a curve crosses its threshold, which gives the resolution.

For more about Fourier ring correlation, see [R.P.J. Nieuwenhuizen et al. (2013)](https://doi.org/10.1038/nmeth.2448).

### 1FRC

FRC needs two images with independent noise. Modern cameras are often limited by shot noise, which is Poisson noise. 1FRC, a method by Bernd Rieger and Sjoerd Stallinga (Department of Imaging Physics, TU Delft), splits each pixel's count of one such image binomially. The two images that result have independent Poisson noise.

I wrote this library for my bachelor thesis in 2022, on 1FRC, with Bernd Rieger as supervisor. [The thesis is here](http://resolver.tudelft.nl/uuid:abdc31f6-6ecd-4c1b-a0c4-53711829467a). Bernd Rieger and Sjoerd Stallinga continued the research, and published a peer-reviewed paper in Optics Express in 2024, [which is here](https://doi.org/10.1364/OE.524683). The paper describes the limitations of 1FRC and when it applies. Its abstract:

> We address resolution assessment for (light super-resolution) microscopy imaging. In modalities where imaging is not diffraction limited, correlation between two noise independent images is the standard way to infer the resolution. Here we take away the need for two noise independent images by computationally splitting one image acquisition into two noise independent realizations. This procedure generates two Poisson noise distributed images if the input is Poissonian distributed. As most modern cameras are shot-noise limited this procedure is directly applicable. However, also in the presence of readout noise we can compute the resolution faithfully via a correction factor. We evaluate our method on simulations and experimental data of widefield microscopy, STED microscopy, rescan confocal microscopy, image scanning microscopy, conventional confocal microscopy, and transmission electron microscopy. In all situations we find that using one image instead of two results in the same computed image resolution.

## Performance

The binomial split and the FRC itself, with its Fourier transforms and ring sums, run in Rust, in [frc-core](https://pypi.org/project/frc-core/). `one_frc` of a 4096 x 4096 image of int32 counts takes about 26 ms on 8 threads of an AMD EPYC 9V45 (measured on 2026-11-14), where frc 0.2 took about a second. frc-core's README has more measurements. The same code is also a Rust crate, [`frc`](https://crates.io/crates/frc).

## Development

frc, frc-core and the crate frc live in one [repository](https://github.com/tiptenbrink/frc), whose README describes how to build and test them.
