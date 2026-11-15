"""FrcPlan: FRC curves and resolutions of images of one size."""

import frc_core
import numpy as np
from frc_core import Crossing, Rings

import frc._internal_utility as _util
from frc.deps_types import Img

__all__ = ['FrcPlan', 'Rings', 'Crossing']


class FrcPlan:
    """
    Computes FRC curves and resolutions of square images of n x n pixels.

    Make one plan for a size, and use it for every image of that size. The
    plan keeps the Fourier transform plan and the working memory between
    curves.

    A plan uses the corrected defaults of frc-core: one_image applies a
    Tukey window after the split, ring r holds the frequencies whose
    distance from the origin rounds to r, and first_crossing interpolates
    the first crossing of the threshold. The functions one_frc, two_frc and
    frc_res keep the defaults of earlier versions of frc.

    A plan computes one curve at a time. Threads that share a plan take
    turns.

    :param n: The side of the images, in pixels.
    :param binning: Which frequencies f make up ring r. With 'nearest', the
    default, ring r holds the frequencies whose distance |f| from the
    origin rounds to r. With 'floor', it holds those with r ≤ |f| < r + 1,
    as earlier versions of frc did.
    """

    def __init__(self, n: int, binning: str = "nearest"):
        self._plan = frc_core.FrcPlan(n, binning)

    @property
    def size(self) -> int:
        """The side n of the images, in pixels."""
        return self._plan.size

    @property
    def rings(self) -> Rings:
        """The rings that the plan's curves sum over: their binning and the
        number of frequencies in each."""
        return self._plan.rings

    def frequencies(self) -> np.ndarray:
        """
        Returns the spatial frequency of each ring r < n / 2, in 1/pixel:
        r / n for ring r. Plot a curve against these values.
        """
        n = self.size
        return np.arange(n // 2) / n

    def one_image(self, img: Img, seed: int | None = None, tukey_alpha: float = 0.125) -> np.ndarray:
        """
        Computes the 1FRC curve of an n x n image of photon counts.

        Pass the counts after offset and gain correction, and before any
        other processing. one_image splits every count c into a sample k of
        binomial(c, 0.5) and c - k, applies a Tukey window to both halves,
        and computes the FRC of the two halves.

        :param img: The image of counts. Floats are accepted if they hold
        whole numbers.
        :param seed: A seed for the split, from 0 to 2**64 - 1. The same
        image and seed give the same curve. Without a seed, every call
        splits with a new seed from the operating system.
        :param tukey_alpha: The fraction of each side that the two tapers of
        the Tukey window take together, from 0 to 1. With 0, one_image
        applies no window.
        :return: The FRC of each ring r < n / 2.
        :raises ValueError: If the image holds values that are not whole
        numbers, or is not of the plan's size.
        """
        counts, holds_counts = _util.counts_of(np.asarray(img))
        if not holds_counts:
            raise ValueError(
                "img holds values that are not whole numbers, but FrcPlan.one_image splits photon counts. Pass the "
                "counts as they were measured, without a window. Round counts that were converted from camera "
                "units."
            )
        return np.asarray(self._plan.one_image(counts, seed, tukey_alpha))

    def two_images(self, img1: Img, img2: Img) -> np.ndarray:
        """
        Computes the FRC curve of two n x n images of the same object with
        independent noise.

        two_images applies no window. Multiply both images by one first, for
        example with frc.util.apply_tukey.

        :return: The FRC of each ring r < n / 2.
        """
        return np.asarray(self._plan.two_images(np.asarray(img1), np.asarray(img2)))

    def first_crossing(self, curve: np.ndarray, threshold: str | float = "1/7") -> Crossing | None:
        """
        Finds the point at which a curve of this plan first drops below its
        threshold.

        first_crossing finds the first ring i whose FRC is below the
        threshold, and interpolates linearly between rings i - 1 and i to
        find where the curve crosses the threshold. The crossing's
        resolution is one over its spatial frequency, in pixels. Multiply it
        by the pixel size for a length.

        :param curve: A curve that one_image or two_images of this plan
        returned.
        :param threshold: '1/7', the default, 'half_bit' for the 1/2-bit
        threshold, or a number.
        :return: The crossing, with its ring, its threshold and the
        resolution, or None if the curve does not drop below the threshold.
        """
        return self._plan.first_crossing(np.asarray(curve, dtype=np.float64), threshold)
