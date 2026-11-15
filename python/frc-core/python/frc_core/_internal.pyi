from collections.abc import Buffer
from typing import Literal

RingBinning = Literal["nearest", "floor"]

def binom_split(a: Buffer, seed: int | None = None) -> memoryview:
    """Splits every count of an array of non-negative integers in two.

    Returns a memoryview of int32 values with the shape of the array. It holds, for each
    count c, a sample k of binomial(c, 0.5): the number of ones among c random bits. If c is
    Poisson-distributed with mean m, then k and c - k are independent and Poisson-distributed
    with mean m / 2.

    The same array and seed give the same samples on every CPU. For a square image they are
    the samples that FrcPlan.one_image draws with the same seed, so
    plan.one_image(img, seed, tukey_alpha=0) is plan.two_images(k, img - k). Without a seed,
    every call draws a new seed from the operating system.
    """

class Crossing:
    """The point at which an FRC curve first drops below its threshold."""

    @property
    def ring(self) -> float:
        """The position of the crossing in rings: from ring 0 at the origin up, with fractions
        between two rings."""
    @property
    def threshold(self) -> float:
        """The threshold at the crossing."""
    @property
    def resolution(self) -> float:
        """The resolution in pixels: n / ring for n x n images. It is infinite for a curve that
        starts below its threshold, which crosses at ring 0."""

class Rings:
    """The rings r < n / 2 of the spectrum of n x n images: their binning and the number of
    frequencies in each.

    Make one to find where the FRC curve of an n x n image crosses its threshold. Computing the
    rings needs no Fourier transform: an FrcPlan holds one, and you can also make one on its
    own. binning says which frequencies f make up ring r. With "nearest", ring r holds the
    frequencies whose distance |f| from the origin rounds to r. With "floor", it holds those
    with r ≤ |f| < r + 1, as DIPlib's RadialSum bins them and as earlier versions of frc did.
    """

    def __init__(self, n: int, binning: RingBinning = "nearest") -> None: ...
    @property
    def size(self) -> int:
        """The side n of the images."""
    @property
    def binning(self) -> RingBinning:
        """Which frequencies make up each ring: "nearest" or "floor"."""
    def counts(self) -> memoryview:
        """Counts the frequencies in each ring r < n / 2.

        Returns a memoryview of uint64 values: the number of frequencies f in ring r.
        """
    def first_crossing(self, curve: Buffer, threshold: str | float = "1/7") -> Crossing | None:
        """Finds the point at which a curve first drops below its threshold.

        curve holds the FRC of ring r at index r, as FrcPlan.one_image and two_images return it
        for images of this size and binning. The method finds the first ring i >= 1 whose FRC is
        below the threshold, and draws a straight line through the FRC minus the threshold at
        rings i - 1 and i. It returns the point at which the line is zero as a Crossing, or None
        if the curve does not drop below the threshold.

        threshold is "1/7", the default, "half_bit" for the 1/2-bit threshold, which depends on
        the number of frequencies in each ring, or a float.
        """

class FrcPlan:
    """The FFT plan and the working memory for FRC curves of n x n images.

    Make one plan for a size and its ring binning, and use it for every image of that size.
    binning is "nearest" or "floor", as for Rings. A plan computes one curve at a time: threads
    that share a plan take turns.
    """

    def __init__(self, n: int, binning: RingBinning = "nearest") -> None: ...
    @property
    def size(self) -> int:
        """The side n of the images."""
    @property
    def rings(self) -> Rings:
        """The rings that the plan's curves sum over."""
    def first_crossing(self, curve: Buffer, threshold: str | float = "1/7") -> Crossing | None:
        """Finds the point at which a curve of this plan first drops below its threshold, as
        Rings.first_crossing does."""
    def one_image(self, img: Buffer, seed: int | None = None, tukey_alpha: float = 0.125) -> memoryview:
        """Computes the Fourier ring correlation curve of one n x n image of counts.

        Pass the counts as they were measured, without a window. The method splits every count
        c of the image into a sample k of binomial(c, 0.5) and c - k, as binom_split does with
        the same seed. It multiplies both images that result by a Tukey window, which falls to
        zero at the edges, and correlates them, as two_images does. Returns a memoryview of
        float64 values, one for each ring r < n / 2.

        tukey_alpha is the fraction of each side that the two tapers of the window take
        together, from 0 to 1. With 0 the method applies no window. The window comes after the
        split because the split gives two independent images only of Poisson counts, and a
        count multiplied by a window is not one.

        The same image, seed and tukey_alpha give the same curve. Without a seed, every call
        draws a new seed from the operating system.
        """
    def two_images(self, img1: Buffer, img2: Buffer) -> memoryview:
        """Computes the Fourier ring correlation curve of two n x n images.

        Returns a memoryview of float64 values, one for each ring r < n / 2. The value of ring r
        is Re(sum F1 conj F2) / sqrt(sum |F1|^2 sum |F2|^2) over the frequencies f of ring r,
        where F1 and F2 are the Fourier transforms of the images. The images are read as
        float32. The method applies no window: multiply both images by one first.
        """
