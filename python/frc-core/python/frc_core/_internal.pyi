from collections.abc import Buffer

def binom_split(a: Buffer, seed: int | None = None) -> memoryview:
    """Splits every count of an array of non-negative integers in two.

    Returns a memoryview of int32 values with the shape of the array. It holds, for each
    count c, a sample k of binomial(c, 0.5): the number of ones among c random bits. If c is
    Poisson-distributed with mean m, then k and c - k are independent and Poisson-distributed
    with mean m / 2.

    The same array and seed give the same samples on every CPU. For a square image they are
    the samples that one_frc draws with the same seed, so one_frc(img, seed) is
    two_frc(k, img - k). Without a seed, every call draws a new seed from the operating system.
    """

def one_frc(img: Buffer, seed: int | None = None) -> memoryview:
    """Computes the Fourier ring correlation curve of one square image of counts.

    Splits every count c of the image into a sample k of binomial(c, 0.5) and c - k, as
    binom_split does with the same seed, and correlates the two images that result, as two_frc
    does. Returns a memoryview of float64 values, one for each ring r < n / 2 of the n x n
    image.

    The same image and seed give the same curve. Without a seed, every call draws a new seed
    from the operating system.

    Each call makes an FrcPlan, which takes time and memory. For several images of one size,
    make one FrcPlan and call its one_frc.
    """

def two_frc(img1: Buffer, img2: Buffer) -> memoryview:
    """Computes the Fourier ring correlation curve of two square images of one size.

    Returns a memoryview of float64 values, one for each ring r < n / 2 of the n x n images.
    The value of ring r is Re(sum F1 conj F2) / sqrt(sum |F1|^2 sum |F2|^2) over the
    frequencies f with r <= |f| < r + 1, where F1 and F2 are the Fourier transforms of the
    images. The images are read as float32.

    Each call makes an FrcPlan, which takes time and memory. For several pairs of images of
    one size, make one FrcPlan and call its two_frc.
    """

def ring_counts(n: int) -> memoryview:
    """Counts the frequencies in each ring of an n x n spectrum.

    Returns a memoryview of uint64 values, one for each ring r < n / 2: the number of
    frequencies f with r <= |f| < r + 1.
    """

class FrcPlan:
    """The FFT plan and the working memory for FRC curves of n x n images.

    Make one plan for a size and use it for every image of that size. A plan computes one
    curve at a time: threads that share a plan take turns.
    """

    def __init__(self, n: int) -> None: ...
    @property
    def size(self) -> int:
        """The side n of the images."""
    def ring_counts(self) -> memoryview:
        """Counts the frequencies in each ring r < n / 2, as ring_counts(n) does."""
    def one_frc(self, img: Buffer, seed: int | None = None) -> memoryview:
        """Computes the FRC curve of an n x n image of counts, as the function one_frc does."""
    def two_frc(self, img1: Buffer, img2: Buffer) -> memoryview:
        """Computes the FRC curve of two n x n images, as the function two_frc does."""
