"""Functions that prepare an image for an FRC curve, and that find where a
curve crosses its threshold."""

# Copyright (C) 2021                Department of Imaging Physics
# All rights reserved               Faculty of Applied Sciences
#                                   TU Delft
# Tip ten Brink

import numpy as np

import frc._internal_utility as _util
from frc.deps_types import Func1D, Img, NoIntersectionException

__all__ = ['gaussf', 'square_image', 'tukey_window', 'tukey_square', 'apply_tukey', 'intersection']


def square_image(img: Img, add_padding=True) -> Img:
    """
    Makes a 2D image square, by padding it with zeros or by trimming it.

    Padding keeps the longer axis and puts the image in the middle of the
    shorter one, one pixel to the right or to the bottom if the padding is
    odd. Trimming keeps the top left of the image.

    :param img: The image.
    :param add_padding: If True, pad the image with zeros. Otherwise trim it.
    :return: The square image.
    """
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError(f"img has shape {img.shape}, but must be a 2D image.")

    short, long = sorted(img.shape)
    if not add_padding:
        return img[:short, :short]

    padded = np.zeros((long, long), dtype=img.dtype)
    before = (long - short + 1) // 2
    if img.shape[0] == long:
        padded[:, before:before + short] = img
    else:
        padded[before:before + short, :] = img
    return padded


def gaussf(img: Img, sigmas=1) -> Img:
    """
    Filters an image with a Gaussian.

    Convolves the image along each axis with a Gaussian of standard
    deviation sigmas pixels, truncated at ceil(3 sigmas) pixels and
    normalized. Beyond its border the image continues as its mirror image
    about the edge pixels. For sigmas of 0.8 and more this is what DIPlib's
    Derivative computed, which frc used before 0.3. For smaller sigmas,
    DIPlib used a Fourier method.

    :param img: The image.
    :param sigmas: The standard deviation of the Gaussian, in pixels.
    :return: The filtered image, as an array of float64.
    """
    out = np.asarray(img, dtype=np.float64)
    if sigmas <= 0:
        raise ValueError(f"sigmas is {sigmas}, but must be positive.")
    radius = int(np.ceil(3 * sigmas))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-offsets ** 2 / (2.0 * sigmas ** 2))
    kernel /= kernel.sum()
    for axis in range(out.ndim):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="reflect")
        length = out.shape[axis]
        out = sum(k * np.take(padded, np.arange(i, i + length), axis=axis) for i, k in enumerate(kernel))
    return out


def tukey_window(length: int, alpha=0.125) -> np.ndarray:
    """
    Makes a symmetric Tukey window, as scipy.signal.windows.tukey does.

    :param length: The number of points of the window.
    :param alpha: The fraction of the window that the two cosine tapers take
    together. The window is flat in between.
    :return: The window, as a 1D array.
    """
    if length <= 1:
        return np.ones(max(length, 0))
    if alpha <= 0:
        return np.ones(length)
    n = np.arange(length)
    if alpha >= 1:
        return 0.5 - 0.5 * np.cos(2 * np.pi * n / (length - 1))
    width = int(np.floor(alpha * (length - 1) / 2.0))
    n1 = n[0:width + 1]
    n3 = n[length - width - 1:]
    w1 = 0.5 * (1 + np.cos(np.pi * (-1 + 2.0 * n1 / alpha / (length - 1))))
    w2 = np.ones(length - 2 * (width + 1))
    w3 = 0.5 * (1 + np.cos(np.pi * (-2.0 / alpha + 1 + 2.0 * n3 / alpha / (length - 1))))
    return np.concatenate((w1, w2, w3))


def tukey_square(length, alpha=0.125) -> np.ndarray:
    """Makes a Tukey window of length x length points: the product of
    tukey_window(length, alpha) along both axes."""
    tukey = tukey_window(length, alpha=alpha)
    return np.outer(tukey, tukey)


def apply_tukey(img: np.ndarray, alpha=0.125) -> np.ndarray:
    """Multiplies a square 2D image by the Tukey window of its size."""
    if img.ndim != 2 or img.shape[0] != img.shape[1]:
        raise ValueError(f"img has shape {img.shape}, but must be a square 2D image.")
    return tukey_square(img.shape[0], alpha) * img


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
