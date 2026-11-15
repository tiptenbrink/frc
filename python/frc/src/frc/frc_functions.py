"""Fourier ring correlation curves of images, and the resolution they give."""

# Copyright (C) 2021                Department of Imaging Physics
# All rights reserved               Faculty of Applied Sciences
#                                   TU Delft
# Tip ten Brink

import frc_core
import numpy as np

import frc._internal_utility as _util
import frc.utility as frc_util
from frc.deps_types import Func1D, Img

__all__ = ['one_frc', 'two_frc', 'frc_res']


def two_frc(img1: Img, img2: Img) -> np.ndarray:
    """
    Computes the FRC curve of two images of one scene whose noise is
    independent.

    The images must be square and of one size. Window them first, for
    example with apply_tukey, so that their edges leave no artifacts in the
    Fourier transforms.

    The FRC of ring r, the frequencies f with r <= |f| < r + 1, is
    Re(sum F1 conj F2) / sqrt(sum |F1|^2 sum |F2|^2), where F1 and F2 are the
    Fourier transforms of the images. See Van Heel, M. (1987). Similarity
    measures between images. Ultramicroscopy, 21(1), 95–100.
    doi:10.1016/0304-3991(87)90010-6

    :param img1: The first image.
    :param img2: The second image.
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

    return np.asarray(frc_core.two_frc(img1, img2))


def one_frc(img: Img, method: int = 1, seed: int | None = None) -> np.ndarray:
    """
    Computes the FRC curve of one image of counts.

    The image must be a square 2D image of counts. Floats are cast to int32,
    which truncates them. Window the image first, for example with
    apply_tukey, so that its edges leave no artifacts in the Fourier
    transforms.

    Method 1 splits the image once: every count c becomes a sample k of
    binomial(c, 0.5) in one image and c - k in the other. Method 2 splits
    the image twice, independently, and correlates the two samples. Prefer
    method 1.

    :param img: The image.
    :param method: 1 or 2. The default is 1.
    :param seed: A seed for the split, from 0 to 2**64 - 1. The same image
    and seed give the same curve. Without a seed, every call splits with a
    new seed from the operating system.
    :return: A 1D array with the FRC of each ring r < n / 2 of the n x n
    image. The index is the distance of the ring from the origin, in pixels
    of the Fourier transform.
    """
    img = np.asarray(img)
    if img.ndim != 2 or img.shape[0] != img.shape[1]:
        raise ValueError(f"img has shape {img.shape}, but must be a square 2D image. Trim or pad it first.")
    if not np.issubdtype(img.dtype, np.integer):
        img = img.astype(np.int32)

    if method == 1:
        return np.asarray(frc_core.one_frc(img, seed))
    if method == 2:
        seeds = (None, None) if seed is None else np.random.SeedSequence(seed).generate_state(2, np.uint64).tolist()
        half1 = frc_core.binom_split(img, seeds[0])
        half2 = frc_core.binom_split(img, seeds[1])
        return np.asarray(frc_core.two_frc(half1, half2))
    raise ValueError(f"method is {method!r}, but must be 1 or 2.")


def half_bit_threshold(img_size: int) -> Func1D:
    """
    Makes the 1/2-bit threshold curve of the FRC of a square image.

    See Van Heel, M., & Schatz, M. (2005). Fourier shell correlation
    threshold criteria. Journal of Structural Biology, 151(3), 250–262.
    doi:10.1016/j.jsb.2005.05.009

    :param img_size: The side of the image, in pixels.
    :return: A function that maps x to the threshold of the ring at
    x * img_size pixels from the origin, for x from 0 up to, but not
    including, 1.
    """
    # The number of frequencies in each ring up to img_size pixels from the
    # origin. A spectrum of twice the size holds all of these rings whole.
    ring_sizes = np.asarray(frc_core.ring_counts(img_size * 2))[:img_size].astype(np.float64)
    ring_sizes_sqrt = np.sqrt(ring_sizes)

    snr_half_set = 0.5 * np.sqrt(2) - 0.5
    factor = np.sqrt(snr_half_set) * 2
    thresholds = (snr_half_set * ring_sizes_sqrt + (factor + 1)) / ((snr_half_set + 1) * ring_sizes_sqrt + factor)

    def threshold_at(x):
        return thresholds[np.array(x * img_size).astype(int)]

    return threshold_at


def frc_res(xs: np.ndarray, frc_curve: np.ndarray, img_size: int,
            threshold: str | Func1D | None = None) -> tuple[float, float, Func1D]:
    """
    Computes the resolution from an FRC curve: one over the frequency at
    which the curve crosses its threshold.

    :param xs: The frequency of each value of the curve.
    :param frc_curve: The FRC curve, as one_frc and two_frc return it.
    :param img_size: The side of the square image, in pixels.
    :param threshold: '1/7', the default, for a threshold of 1/7,
    'half_bit' for the 1/2-bit threshold curve, or a function that maps
    the frequencies to their thresholds.
    :return: The resolution, the value of the threshold where the curve
    crosses it, and the threshold as a function.
    :raises NoIntersectionException: If the curve does not cross the
    threshold.
    """
    if threshold is None or threshold == '1/7':
        def threshold_f(x):
            return x * 0 + 1. / 7
    elif threshold == 'half_bit':
        threshold_f = half_bit_threshold(img_size)
    elif callable(threshold):
        threshold_f = threshold
    else:
        raise ValueError(f"threshold is {threshold!r}, but must be '1/7', 'half_bit' or a function.")

    intersect_x, intersect_y = frc_util.intersection(xs, frc_curve, threshold_f)

    return 1. / intersect_x, intersect_y, threshold_f
