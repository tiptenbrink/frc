"""The compiled core of frc: Fourier ring correlation with an exact binomial split.

Make an ``FrcPlan`` for the size of your images, and use it for every image of that size. To
find the resolution from a curve without a plan, make ``Rings`` for its size. frc-core has no
dependencies. It takes any array that supports the buffer protocol, such as a NumPy array, an
``array.array`` or a ``memoryview``, and returns a ``memoryview``, which ``numpy.asarray`` wraps
without copying it.
"""

from ._internal import Crossing, FrcPlan, Rings, binom_split

__all__ = ["Crossing", "FrcPlan", "Rings", "binom_split"]
