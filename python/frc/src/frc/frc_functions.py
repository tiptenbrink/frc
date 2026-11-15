"""The legacy API of frc: FRC curves and resolutions with the defaults of
earlier versions. For new code, use FrcPlan."""

# Copyright (C) 2021                Department of Imaging Physics
# All rights reserved               Faculty of Applied Sciences
#                                   TU Delft
# Tip ten Brink

import functools
import warnings

import frc_core
import numpy as np

import frc._internal_utility as _util
import frc.utility as frc_util
from frc.deps_types import Func1D, Img, NoIntersectionException, NotCountsWarning

__all__ = ['one_frc', 'two_frc', 'frc_res']


def two_frc(img1: Img, img2: Img, rings: str = "floor") -> np.ndarray:
    """
    Computes the FRC curve of two images of one scene whose noise is
    independent.

    This function belongs to the legacy API, which keeps the defaults of
    earlier versions of frc. For new code, use FrcPlan.

    The images must be square and of one size. Window them first, for
    example with apply_tukey, so that their edges leave no artifacts in the
    Fourier transforms.

    The FRC of ring r is Re(sum F1 conj F2) / sqrt(sum |F1|^2 sum |F2|^2)
    over the frequencies f of the ring, where F1 and F2 are the Fourier
    transforms of the images. See Van Heel, M. (1987). Similarity
    measures between images. Ultramicroscopy, 21(1), 95–100.
    doi:10.1016/0304-3991(87)90010-6

    :param img1: The first image.
    :param img2: The second image.
    :param rings: Which frequencies f make up ring r of the curve. With
    'nearest', ring r holds the frequencies whose distance |f| from the
    origin rounds to r. With 'floor', the default, it holds those with
    r ≤ |f| < r + 1, as earlier versions did with DIPlib's RadialSum. The
    frequencies of a 'floor' ring lie on average half a ring above r, but
    the curve places the ring at r. In a simulation, this made the
    resolution about 1% coarser. Pass the same rings to frc_res.
    :return: A 1D array with the FRC of each ring r < n / 2 of the n x n
    images. The index is the distance of the ring from the origin, in
    pixels of the Fourier transform.
    """
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)
    if not _util.are_sizes_equal([img1, img2]):
        raise ValueError(
            f"img1 has shape {img1.shape} and img2 has shape {img2.shape}, but both must be square and of one size. "
            "Trim or pad them first.")

    return np.asarray(frc_core.two_frc(img1, img2, rings))


def one_frc(img: Img, method: int = 1, seed: int | None = None, tukey_alpha: float | None = None,
            rings: str = "floor") -> np.ndarray:
    """
    Computes the FRC curve of one image of counts.

    This function belongs to the legacy API, which keeps the defaults of
    earlier versions of frc. For new code, use FrcPlan.

    Pass a square 2D image of the counts as they were measured, and a
    tukey_alpha, such as 0.125. one_frc splits the counts in two images,
    multiplies both by a Tukey window, so that their edges leave no
    artifacts in the Fourier transforms, and correlates them.

    The window must come after the split. The split gives two independent
    images only of Poisson counts, whose variance is their mean. A count
    that a window has scaled by w has w times the mean and w * w times the
    variance, and the noise of its two halves correlates, which lowers the
    FRC of every ring.

    Without a tukey_alpha, one_frc applies no window, as in earlier
    versions, where you windowed the image yourself before the split. It then
    warns with a NotCountsWarning if the image holds values that are not
    whole numbers.

    Method 1 splits the image once: every count c becomes a sample k of
    binomial(c, 0.5) in one image and c - k in the other. Method 2 splits
    the image twice, independently, and correlates the two samples. Prefer
    method 1.

    :param img: The image.
    :param method: 1 or 2. The default is 1.
    :param seed: A seed for the split, from 0 to 2**64 - 1. The same image
    and seed give the same curve. Without a seed, every call splits with a
    new seed from the operating system.
    :param tukey_alpha: The alpha of the Tukey window that both images are
    multiplied by after the split, from 0 to 1: the fraction of each side
    that the two tapers take together, as in apply_tukey. With 0, one_frc
    applies no window. With None, the default, it applies no window and
    accepts an image that was windowed before, as earlier versions did.
    :param rings: Which frequencies f make up ring r of the curve. With
    'nearest', ring r holds the frequencies whose distance |f| from the
    origin rounds to r. With 'floor', the default, it holds those with
    r ≤ |f| < r + 1, as earlier versions did with DIPlib's RadialSum. The
    frequencies of a 'floor' ring lie on average half a ring above r, but
    the curve places the ring at r. In a simulation, this made the
    resolution about 1% coarser. Pass the same rings to frc_res.
    :return: A 1D array with the FRC of each ring r < n / 2 of the n x n
    image. The index is the distance of the ring from the origin, in pixels
    of the Fourier transform.
    :raises ValueError: If tukey_alpha is given and the image holds values
    that are not whole numbers.
    """
    img = np.asarray(img)
    if img.ndim != 2 or img.shape[0] != img.shape[1]:
        raise ValueError(f"img has shape {img.shape}, but must be a square 2D image. Trim or pad it first.")
    if method not in (1, 2):
        raise ValueError(f"method is {method!r}, but must be 1 or 2.")
    counts, holds_counts = _util.counts_of(img)

    if tukey_alpha is None and not holds_counts:
        warnings.warn(
            "img holds values that are not whole numbers, such as those of a windowed image, so it does not hold "
            "photon counts. Splitting such an image lowers the FRC. "
            "Pass the counts as they were measured, with tukey_alpha=0.125, and one_frc applies the window after "
            "the split. To keep this result without the warning, ignore frc.NotCountsWarning with the warnings "
            "module.",
            NotCountsWarning,
            stacklevel=2,
        )
        if method == 1:
            # As in earlier versions: the first half is a sample of the truncated values, and the
            # second half is the rest of the image, with its fractions.
            half1 = np.asarray(frc_core.binom_split(counts, seed))
            return np.asarray(frc_core.two_frc(half1, img - half1, rings))
    if tukey_alpha is not None and not holds_counts:
        raise ValueError(
            "img holds values that are not whole numbers, but one_frc with a tukey_alpha splits counts. Pass the "
            "counts as they were measured, without a window. Round counts that were converted from camera units."
        )

    if method == 1:
        return np.asarray(frc_core.one_frc(counts, seed, tukey_alpha or 0.0, rings))
    seeds = (None, None) if seed is None else np.random.SeedSequence(seed).generate_state(2, np.uint64).tolist()
    halves = [np.asarray(frc_core.binom_split(counts, seed)) for seed in seeds]
    if tukey_alpha:
        halves = [frc_util.apply_tukey(half, tukey_alpha) for half in halves]
    return np.asarray(frc_core.two_frc(*halves, rings))


@functools.cache
def _half_bit_thresholds(ring_count: int, rings: str) -> np.ndarray:
    """Returns the 1/2-bit threshold of each of the first ring_count rings."""
    # A spectrum of twice the size holds all of these rings whole.
    ring_sizes = np.asarray(frc_core.ring_counts(2 * ring_count, rings), dtype=np.float64)
    ring_sizes_sqrt = np.sqrt(ring_sizes)
    snr_half_set = 0.5 * np.sqrt(2) - 0.5
    factor = np.sqrt(snr_half_set) * 2
    thresholds = (snr_half_set * ring_sizes_sqrt + (factor + 1)) / ((snr_half_set + 1) * ring_sizes_sqrt + factor)
    thresholds.setflags(write=False)
    return thresholds


def _ring_of(x, first: float, step: float) -> np.ndarray:
    """Returns the ring of each frequency x, for rings at first, first + step, and so on."""
    # The small addition keeps a ring's own frequency, which a float may hold a little below
    # it, in that ring.
    return np.floor((np.asarray(x) - first) / step + 1e-9).astype(int)


def half_bit_threshold(img_size: int, rings: str = "floor") -> Func1D:
    """
    Makes the 1/2-bit threshold curve of the FRC of a square image.

    See Van Heel, M., & Schatz, M. (2005). Fourier shell correlation
    threshold criteria. Journal of Structural Biology, 151(3), 250–262.
    doi:10.1016/j.jsb.2005.05.009

    :param img_size: The side of the image, in pixels.
    :param rings: The rings of the curve, 'floor' or 'nearest', as one_frc
    takes them. The threshold of a ring depends on its number of
    frequencies.
    :return: A function that maps x to the threshold of the ring at
    x * img_size pixels from the origin, for x from 0 up to, but not
    including, 1.
    """
    thresholds = _half_bit_thresholds(img_size, rings)

    def threshold_at(x):
        return thresholds[_ring_of(x, 0.0, 1 / img_size)]

    return threshold_at


def frc_res(xs: np.ndarray, frc_curve: np.ndarray, img_size: int,
            threshold: str | Func1D | None = None, rings: str = "floor",
            crossing: str = "band") -> tuple[float, float, Func1D]:
    """
    Computes the resolution from an FRC curve: one over the frequency at
    which the curve crosses its threshold.

    This function belongs to the legacy API, which keeps the defaults of
    earlier versions of frc. For new code, use FrcPlan.

    With crossing="first", frc_res finds the first ring i whose FRC is
    below its threshold. It then interpolates linearly between rings i - 1
    and i to find the frequency at which the curve crosses the threshold.
    With crossing="band", the default, it takes the rings whose FRC lies
    within 0.08 of the threshold, and returns the mean frequency of the
    first group of consecutive rings in which the curve crosses, as
    earlier versions did. That group often reaches past the first crossing.
    In a simulation, it gave resolutions 2 to 9% finer than the first
    crossing, with twice the spread between splits.

    :param xs: The frequency of each value of the curve, in any unit:
    xs[r] for ring r. The resolution comes out in the reciprocal unit.
    :param img_size: The side of the square image, in pixels.
    :param threshold: '1/7', the default, for a threshold of 1/7,
    'half_bit' for the 1/2-bit threshold curve, or a function that maps
    the frequencies to their thresholds.
    :param rings: The rings of the curve, 'floor' or 'nearest', as one_frc
    and two_frc took them. The half-bit threshold depends on them.
    :param crossing: 'band', the default, or 'first'.
    :return: The resolution, the value of the threshold where the curve
    crosses it, and the threshold as a function.
    :raises NoIntersectionException: If the curve does not cross the
    threshold.
    """
    xs = np.asarray(xs, dtype=np.float64)
    if crossing not in ("band", "first"):
        raise ValueError(f"crossing is {crossing!r}, but must be 'band' or 'first'.")
    if threshold is None or threshold == '1/7':
        def threshold_f(x):
            return x * 0 + 1. / 7
    elif threshold == 'half_bit':
        # Value r of the curve belongs to ring r, at xs[r], in whatever unit xs has.
        thresholds = _half_bit_thresholds(img_size, rings)
        step = xs[1] - xs[0] if len(xs) > 1 else 1 / img_size

        def threshold_f(x):
            return thresholds[_ring_of(x, xs[0], step)]
    elif callable(threshold):
        threshold_f = threshold
    else:
        raise ValueError(f"threshold is {threshold!r}, but must be '1/7', 'half_bit' or a function.")

    if crossing == "band":
        intersect_x, intersect_y = frc_util.intersection(xs, frc_curve, threshold_f)
        return 1. / intersect_x, intersect_y, threshold_f

    curve = np.asarray(frc_curve, dtype=np.float64)
    ring_thresholds = np.broadcast_to(np.asarray(threshold_f(xs), dtype=np.float64), curve.shape)
    # The curve crosses its threshold where the curve minus the threshold crosses 0.
    found = frc_core.resolution(curve - ring_thresholds, img_size, 0.0, rings)
    if found is None:
        raise NoIntersectionException("the curve does not cross its threshold")
    ring = img_size / found[0]
    rings_of_xs = np.arange(len(xs))
    intersect_x = np.interp(ring, rings_of_xs, xs)
    return 1. / intersect_x, float(np.interp(ring, rings_of_xs, ring_thresholds)), threshold_f
