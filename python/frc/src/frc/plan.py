"""FrcPlan: FRC curves and resolutions of images of one size."""

import frc_core
import numpy as np

import frc._internal_utility as _util
from frc.deps_types import Img, NoIntersectionException

__all__ = ['FrcPlan']


class FrcPlan:
    """
    Computes FRC curves and resolutions of square images of n x n pixels.

    Make one plan for a size, and use it for every image of that size. The
    plan keeps the Fourier transform plan and the working memory between
    curves.

    A plan uses the corrected defaults of frc-core: one_frc applies a Tukey
    window after the split, ring r holds the frequencies whose distance from
    the origin rounds to r, and resolution interpolates the first crossing
    of the threshold. The functions one_frc, two_frc and frc_res keep the
    defaults of earlier versions of frc.

    A plan computes one curve at a time. Threads that share a plan take
    turns.

    :param n: The side of the images, in pixels.
    :param rings: Which frequencies f make up ring r. With 'nearest', the
    default, ring r holds the frequencies whose distance |f| from the
    origin rounds to r. With 'floor', it holds those with r ≤ |f| < r + 1,
    as earlier versions of frc did.
    """

    def __init__(self, n: int, rings: str = "nearest"):
        self._plan = frc_core.FrcPlan(n, rings)

    @property
    def size(self) -> int:
        """The side n of the images, in pixels."""
        return self._plan.size

    @property
    def rings(self) -> str:
        """The rings of the plan's curves: 'nearest' or 'floor'."""
        return self._plan.rings

    def frequencies(self) -> np.ndarray:
        """
        Returns the spatial frequency of each ring r < n / 2, in 1/pixel:
        r / n for ring r. Plot a curve against these values.
        """
        n = self.size
        return np.arange(n // 2) / n

    def ring_counts(self) -> np.ndarray:
        """Returns the number of frequencies in each ring r < n / 2."""
        return np.asarray(self._plan.ring_counts())

    def one_frc(self, img: Img, seed: int | None = None, tukey_alpha: float = 0.125) -> np.ndarray:
        """
        Computes the 1FRC curve of an n x n image of photon counts.

        Pass the counts after offset and gain correction, and before any
        other processing. one_frc splits every count c into a sample k of
        binomial(c, 0.5) and c - k, applies a Tukey window to both halves,
        and computes the FRC of the two halves.

        :param img: The image of counts. Floats are accepted if they hold
        whole numbers.
        :param seed: A seed for the split, from 0 to 2**64 - 1. The same
        image and seed give the same curve. Without a seed, every call
        splits with a new seed from the operating system.
        :param tukey_alpha: The fraction of each side that the two tapers of
        the Tukey window take together, from 0 to 1. With 0, one_frc
        applies no window.
        :return: The FRC of each ring r < n / 2.
        :raises ValueError: If the image holds values that are not whole
        numbers, or is not of the plan's size.
        """
        counts, holds_counts = _util.counts_of(np.asarray(img))
        if not holds_counts:
            raise ValueError(
                "img holds values that are not whole numbers, but FrcPlan.one_frc splits photon counts. Pass the "
                "counts as they were measured, without a window. Round counts that were converted from camera "
                "units."
            )
        return np.asarray(self._plan.one_frc(counts, seed, tukey_alpha))

    def two_frc(self, img1: Img, img2: Img) -> np.ndarray:
        """
        Computes the FRC curve of two n x n images of the same object with
        independent noise.

        two_frc applies no window. Multiply both images by one first, for
        example with frc.util.apply_tukey.

        :return: The FRC of each ring r < n / 2.
        """
        return np.asarray(self._plan.two_frc(np.asarray(img1), np.asarray(img2)))

    def resolution(self, curve: np.ndarray, threshold: str | float = "1/7") -> float:
        """
        Computes the resolution, in pixels, from a curve of this plan.

        resolution finds the first ring i whose FRC is below the threshold,
        and interpolates linearly between rings i - 1 and i to find where
        the curve crosses the threshold. The resolution is one over the
        spatial frequency of the crossing. Multiply it by the pixel size for
        a length.

        :param curve: A curve that one_frc or two_frc of this plan returned.
        :param threshold: '1/7', the default, 'half_bit' for the 1/2-bit
        threshold, or a number.
        :raises NoIntersectionException: If the curve does not drop below
        the threshold.
        """
        found = self._plan.resolution(np.asarray(curve, dtype=np.float64), threshold)
        if found is None:
            raise NoIntersectionException("the curve does not drop below its threshold")
        return found[0]
