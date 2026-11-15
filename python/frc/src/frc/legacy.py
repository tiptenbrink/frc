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
from frc.deps_types import Func1D, Img, NoIntersectionException

__all__ = ['one_frc', 'two_frc', 'half_bit_threshold', 'frc_res', 'NotCountsWarning']


class NotCountsWarning(UserWarning):
    """The warning of one_frc for an image that holds values that are not
    whole numbers, such as an image that was multiplied by a window. To
    switch it off, ignore this class with the warnings module."""


def two_frc(img1: Img, img2: Img, binning: str = "floor") -> np.ndarray:
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
    :param binning: Which frequencies f make up ring r of the curve. With
    'nearest', ring r holds the frequencies whose distance |f| from the
    origin rounds to r. With 'floor', the default, it holds those with
    r ≤ |f| < r + 1, as earlier versions did with DIPlib's RadialSum. The
    frequencies of a 'floor' ring lie on average half a ring above r, but
    the curve places the ring at r. In a simulation, this made the
    resolution about 1% coarser. Pass the same binning to frc_res.
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

    return np.asarray(frc_core.FrcPlan(img1.shape[0], binning).two_images(img1, img2))


def one_frc(img: Img, method: int = 1, seed: int | None = None, binning: str = "floor") -> np.ndarray:
    """
    Computes the FRC curve of one image of counts.

    This function belongs to the legacy API, which keeps the defaults of
    earlier versions of frc. For new code, use FrcPlan.

    Pass a square 2D image of counts. one_frc splits the counts in two
    images and correlates them. It applies no window, as in earlier
    versions, where you windowed the image yourself before the split. It
    warns with a NotCountsWarning if the image holds values that are not
    whole numbers, as a windowed image does.

    A window before the split lowers the FRC. The split gives two
    independent images only of Poisson counts, whose variance is their mean.
    A count that a window has scaled by w has w times the mean and w * w
    times the variance, and the noise of its two halves correlates.
    FrcPlan.one_image applies the window after the split.

    Method 1 splits the image once: every count c becomes a sample k of
    binomial(c, 0.5) in one image and c - k in the other. Method 2 splits
    the image twice, independently, and correlates the two samples. Prefer
    method 1.

    :param img: The image.
    :param method: 1 or 2. The default is 1.
    :param seed: A seed for the split, from 0 to 2**64 - 1. The same image
    and seed give the same curve. Without a seed, every call splits with a
    new seed from the operating system.
    :param binning: Which frequencies f make up ring r of the curve. With
    'nearest', ring r holds the frequencies whose distance |f| from the
    origin rounds to r. With 'floor', the default, it holds those with
    r ≤ |f| < r + 1, as earlier versions did with DIPlib's RadialSum. The
    frequencies of a 'floor' ring lie on average half a ring above r, but
    the curve places the ring at r. In a simulation, this made the
    resolution about 1% coarser. Pass the same binning to frc_res.
    :return: A 1D array with the FRC of each ring r < n / 2 of the n x n
    image. The index is the distance of the ring from the origin, in pixels
    of the Fourier transform.
    """
    img = np.asarray(img)
    if img.ndim != 2 or img.shape[0] != img.shape[1]:
        raise ValueError(f"img has shape {img.shape}, but must be a square 2D image. Trim or pad it first.")
    if method not in (1, 2):
        raise ValueError(f"method is {method!r}, but must be 1 or 2.")
    counts, holds_counts = _util.counts_of(img)
    plan = frc_core.FrcPlan(img.shape[0], binning)

    if not holds_counts:
        warnings.warn(
            "img holds values that are not whole numbers, such as those of a windowed image, so it does not hold "
            "photon counts. Splitting such an image lowers the FRC. "
            "Pass the counts as they were measured to FrcPlan.one_image, which applies the window after the split. "
            "To keep this result without the warning, ignore frc.NotCountsWarning with the warnings module.",
            NotCountsWarning,
            stacklevel=2,
        )
        if method == 1:
            # As in earlier versions: the first half is a sample of the truncated values, and the
            # second half is the rest of the image, with its fractions.
            half1 = np.asarray(frc_core.binom_split(counts, seed))
            return np.asarray(plan.two_images(half1, img - half1))

    if method == 1:
        return np.asarray(plan.one_image(counts, seed, 0.0))
    seeds = (None, None) if seed is None else np.random.SeedSequence(seed).generate_state(2, np.uint64).tolist()
    halves = [np.asarray(frc_core.binom_split(counts, seed)) for seed in seeds]
    return np.asarray(plan.two_images(*halves))


@functools.cache
def _half_bit_thresholds(ring_count: int, binning: str) -> np.ndarray:
    """Returns the 1/2-bit threshold of each of the first ring_count rings."""
    # A spectrum of twice the size holds all of these rings whole.
    ring_sizes = np.asarray(frc_core.Rings(2 * ring_count, binning).counts(), dtype=np.float64)
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


def half_bit_threshold(img_size: int, binning: str = "floor") -> Func1D:
    """
    Makes the 1/2-bit threshold curve of the FRC of a square image.

    See Van Heel, M., & Schatz, M. (2005). Fourier shell correlation
    threshold criteria. Journal of Structural Biology, 151(3), 250–262.
    doi:10.1016/j.jsb.2005.05.009

    :param img_size: The side of the image, in pixels.
    :param binning: The ring binning of the curve, 'floor' or 'nearest', as
    one_frc takes it. The threshold of a ring depends on its number of
    frequencies.
    :return: A function that maps x to the threshold of the ring at
    x * img_size pixels from the origin, for x from 0 up to, but not
    including, 1.
    """
    thresholds = _half_bit_thresholds(img_size, binning)

    def threshold_at(x):
        return thresholds[_ring_of(x, 0.0, 1 / img_size)]

    return threshold_at


def intersection(xs: np.ndarray, ys1_f: np.ndarray | Func1D, ys2_f: Func1D, tol=0.08):
    """
    Finds the first point at which a curve crosses a second curve, such as
    the point at which an FRC curve crosses its threshold.

    The function evaluates both curves at xs. It then collects the indices
    at which the first curve is within a tolerance of the second, the
    indices just before the first curve crosses the second, and the indices
    just after. The tolerance is tol times the largest value of both curves.
    It splits these indices into groups of consecutive ones, and takes the
    first group in which the first curve is both at or above and at or below
    the second. The intersection is at the mean of the group's xs, which is
    accurate if xs are evenly spaced.

    The two curves do not play the same role: swapping them can give
    another point.

    :param xs: The x values of both curves.
    :param ys1_f: The first curve, as its values at xs or as a function.
    :param ys2_f: The second curve, as a function.
    :param tol: The tolerance, as a fraction of the largest value of both
    curves.
    :return: The x of the intersection, and the value of the second curve
    there.
    :raises NoIntersectionException: If no group of indices holds a
    crossing.
    """
    # A copy, because the first value may change below.
    ys1 = np.array(ys1_f(xs) if callable(ys1_f) else ys1_f, dtype=np.float64)
    ys2: np.ndarray = ys2_f(xs)
    tolerance = tol * max(np.amax(ys1), np.amax(ys2))

    close_indices = np.asarray((ys1 < ys2 + tolerance) & (ys1 > ys2 - tolerance)).nonzero()[0]
    next_y = np.diff(ys1)
    thres_y = (ys2 - ys1)[:-1]

    # A first curve that starts on the second one starts above it instead, so
    # that the crossings below go downwards.
    if ys1[0] == ys2[0]:
        ys1[0] += tolerance * 0.5

    if ys1[0] > ys2[0]:
        # The first curve comes from above: it crosses where its next step
        # down is larger than its distance to the second curve.
        crossings = np.asarray((next_y < thres_y) & (thres_y < 0)).nonzero()[0]
    else:
        crossings = np.asarray((next_y > thres_y) & (thres_y > 0)).nonzero()[0]

    of_interest = np.unique(np.concatenate((close_indices, crossings, crossings + 1)))

    for group in _util.separate_noncontiguous(of_interest):
        difference: np.ndarray = ys1[group] - ys2[group]
        if np.any(difference >= 0) and np.any(difference <= 0):
            x = np.average(xs[group])
            return x, ys2_f(x)

    raise NoIntersectionException("the curve does not cross its threshold")


def frc_res(xs: np.ndarray, frc_curve: np.ndarray, img_size: int,
            threshold: str | Func1D | None = None, binning: str = "floor") -> tuple[float, float, Func1D]:
    """
    Computes the resolution from an FRC curve: one over the frequency at
    which the curve crosses its threshold.

    This function belongs to the legacy API, which keeps the defaults of
    earlier versions of frc. For new code, use FrcPlan.

    frc_res takes the rings whose FRC lies within 0.08 of the threshold,
    and returns the mean frequency of the first group of consecutive rings
    in which the curve crosses, as earlier versions did. That group often
    reaches past the first crossing. In a simulation, it gave resolutions 2
    to 9% finer than the first crossing, with twice the spread between
    splits. FrcPlan.first_crossing interpolates the first crossing.

    :param xs: The frequency of each value of the curve, in any unit:
    xs[r] for ring r. The resolution comes out in the reciprocal unit.
    :param img_size: The side of the square image, in pixels.
    :param threshold: '1/7', the default, for a threshold of 1/7,
    'half_bit' for the 1/2-bit threshold curve, or a function that maps
    the frequencies to their thresholds.
    :param binning: The ring binning of the curve, 'floor' or 'nearest', as
    one_frc and two_frc took it. The half-bit threshold depends on it.
    :return: The resolution, the value of the threshold where the curve
    crosses it, and the threshold as a function.
    :raises NoIntersectionException: If the curve does not cross the
    threshold.
    """
    xs = np.asarray(xs, dtype=np.float64)
    if threshold is None or threshold == '1/7':
        def threshold_f(x):
            return x * 0 + 1. / 7
    elif threshold == 'half_bit':
        # Value r of the curve belongs to ring r, at xs[r], in whatever unit xs has.
        thresholds = _half_bit_thresholds(img_size, binning)
        step = xs[1] - xs[0] if len(xs) > 1 else 1 / img_size

        def threshold_f(x):
            return thresholds[_ring_of(x, xs[0], step)]
    elif callable(threshold):
        threshold_f = threshold
    else:
        raise ValueError(f"threshold is {threshold!r}, but must be '1/7', 'half_bit' or a function.")

    intersect_x, intersect_y = intersection(xs, frc_curve, threshold_f)
    return 1. / intersect_x, intersect_y, threshold_f
