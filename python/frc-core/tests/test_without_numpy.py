"""frc-core has no dependencies. CI also runs these tests without NumPy installed."""

import array

import frc_core


def test_split_an_array():
    k = frc_core.binom_split(array.array("i", [5] * 64), seed=3)
    assert k.format == "i" and k.shape == (64,)
    assert all(0 <= v <= 5 for v in k)


def test_frc_of_a_memoryview():
    square = memoryview(array.array("q", [7] * 256)).cast("B").cast("q", [16, 16])
    curve = frc_core.one_frc(square, seed=4)
    assert curve.format == "d" and len(curve) == 8
    counts = frc_core.ring_counts(16)
    assert counts.format == "Q" and counts.tolist()[:2] == [1, 8]
