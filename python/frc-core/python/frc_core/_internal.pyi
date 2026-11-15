from collections.abc import Buffer
from typing import Literal

Rings = Literal["nearest", "floor"]

def binom_split(a: Buffer, seed: int | None = None) -> memoryview:
    """Splits every count of an array of non-negative integers in two.

    Returns a memoryview of int32 values with the shape of the array. It holds, for each
    count c, a sample k of binomial(c, 0.5): the number of ones among c random bits. If c is
    Poisson-distributed with mean m, then k and c - k are independent and Poisson-distributed
    with mean m / 2.

    The same array and seed give the same samples on every CPU. For a square image they are
    the samples that one_frc draws with the same seed, so one_frc(img, seed, tukey_alpha=0) is
    two_frc(k, img - k). Without a seed, every call draws a new seed from the operating system.
    """

def one_frc(img: Buffer, seed: int | None = None, tukey_alpha: float = 0.125, rings: Rings = "nearest") -> memoryview:
    """Computes the Fourier ring correlation curve of one square image of counts.

    Pass the counts as they were measured, without a window. The function splits every count
    c of the image into a sample k of binomial(c, 0.5) and c - k, as binom_split does with the
    same seed. It multiplies both images that result by a Tukey window, which falls to zero at
    the edges, and correlates them, as two_frc does. Returns a memoryview of float64 values,
    one for each ring r < n / 2 of the n x n image.

    tukey_alpha is the fraction of each side that the two tapers of the window take together,
    from 0 to 1. With 0 the function applies no window. The window comes after the split
    because the split gives two independent images only of Poisson counts, and a count
    multiplied by a window is not one.

    rings says which frequencies f make up ring r. With "nearest", ring r holds those whose
    distance |f| from the origin rounds to r, and its frequencies lie around r. With "floor", it
    holds those with r <= |f| < r + 1, as DIPlib's RadialSum and frc 0.1 and 0.2 bin them.

    The same image, seed and tukey_alpha give the same curve. Without a seed, every call draws
    a new seed from the operating system.

    Each call makes an FrcPlan, which takes time and memory. For several images of one size,
    make one FrcPlan and call its one_frc.
    """

def two_frc(img1: Buffer, img2: Buffer, rings: Rings = "nearest") -> memoryview:
    """Computes the Fourier ring correlation curve of two square images of one size.

    Returns a memoryview of float64 values, one for each ring r < n / 2 of the n x n images.
    The value of ring r is Re(sum F1 conj F2) / sqrt(sum |F1|^2 sum |F2|^2) over the
    frequencies f of ring r, where F1 and F2 are the Fourier transforms of the images. rings
    says which frequencies those are, as for one_frc. The images are read as float32. The
    function applies no window: multiply both images by one first.

    Each call makes an FrcPlan, which takes time and memory. For several pairs of images of
    one size, make one FrcPlan and call its two_frc.
    """

def ring_counts(n: int, rings: Rings = "nearest") -> memoryview:
    """Counts the frequencies in each ring of an n x n spectrum.

    Returns a memoryview of uint64 values, one for each ring r < n / 2: the number of
    frequencies f in ring r. rings says which frequencies those are, as for one_frc.
    """

class FrcPlan:
    """The FFT plan and the working memory for FRC curves of n x n images.

    Make one plan for a size and its rings, "nearest" or "floor" as for one_frc, and use it for
    every image of that size. A plan computes one curve at a time: threads that share a plan
    take turns.
    """

    def __init__(self, n: int, rings: Rings = "nearest") -> None: ...
    @property
    def size(self) -> int:
        """The side n of the images."""
    @property
    def rings(self) -> Rings:
        """The rings that the plan's curves sum over: "nearest" or "floor"."""
    def ring_counts(self) -> memoryview:
        """Counts the frequencies in each ring r < n / 2, as ring_counts(n, rings) does."""
    def one_frc(self, img: Buffer, seed: int | None = None, tukey_alpha: float = 0.125) -> memoryview:
        """Computes the FRC curve of an n x n image of counts, as the function one_frc does."""
    def two_frc(self, img1: Buffer, img2: Buffer) -> memoryview:
        """Computes the FRC curve of two n x n images, as the function two_frc does."""
