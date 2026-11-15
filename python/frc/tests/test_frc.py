import warnings

import frc_core
import pytest
import frc
import numpy as np


def test_simple_no_err():
    a = np.ones((5, 5))*20
    a[3, 2] = 16
    a[4, 3] = 18
    frc.one_frc(a)


def test_half_bit_threshold():
    from frc.frc_functions import half_bit_threshold

    threshold = half_bit_threshold(80)
    values = threshold(np.array([0.1, 0.2]))

    assert values.shape == (2,)
    assert np.all((values > 0) & (values < 1))


def test_frc_res_half_bit():
    rng = np.random.default_rng(0)
    img = rng.poisson(50, (64, 64))
    frc_curve = frc.one_frc(img, tukey_alpha=0.125)
    xs = np.arange(len(frc_curve)) / img.shape[0]

    try:
        frc.frc_res(xs, frc_curve, img.shape[0], threshold='half_bit')
    except frc.deps_types.NoIntersectionException:
        pass


def pattern_image(n, mean, seed):
    y, x = np.mgrid[0:n, 0:n]
    waves = 0.4 * np.cos(2 * np.pi * (3 * x + 5 * y) / n) + 0.2 * np.cos(2 * np.pi * (17 * x - 9 * y) / n)
    return np.random.default_rng(seed).poisson(mean * (1 + waves))


def test_one_frc_sees_the_pattern():
    curve = frc.one_frc(pattern_image(128, 100.0, 1), seed=3)
    assert curve.shape == (64,) and curve.dtype == np.float64
    assert curve[5] > 0.9 and curve[19] > 0.9
    assert np.median(np.abs(curve[30:])) < 0.15


def test_the_first_ring_correlates_exactly():
    # Ring 0 holds the origin alone. A value just below or just above 1 would decide where
    # frc_res looks for the half-bit resolution, whose threshold is exactly 1 there.
    for n in (64, 101, 256):
        img = pattern_image(n, 80.0, n)
        assert frc.one_frc(img, seed=1)[0] == 1.0
        windowed = frc.util.apply_tukey(img.astype(np.float64))
        other = frc.util.apply_tukey(pattern_image(n, 80.0, n + 1).astype(np.float64))
        assert frc.two_frc(windowed, other)[0] == 1.0


def test_seeds_repeat_curves():
    img = pattern_image(64, 30.0, 2)
    for method in (1, 2):
        assert np.array_equal(frc.one_frc(img, method, seed=5), frc.one_frc(img, method, seed=5))
        assert not np.array_equal(frc.one_frc(img, method, seed=5), frc.one_frc(img, method, seed=6))


@pytest.mark.parametrize("method", [1, 2])
def test_one_frc_windows_both_halves_after_the_split(method):
    counts = pattern_image(64, 40.0, 3)
    curve = frc.one_frc(counts, method, seed=4, tukey_alpha=0.125)
    if method == 1:
        half = np.asarray(frc_core.binom_split(counts, seed=4))
        want = frc.two_frc(frc.util.apply_tukey(half), frc.util.apply_tukey(counts - half))
        # frc-core multiplies in float32.
        assert np.allclose(curve, want, atol=1e-5)
    assert curve[5] > 0.9 and curve[19] > 0.8
    assert not np.allclose(curve, frc.one_frc(counts, method, seed=4, tukey_alpha=0), atol=1e-5)
    # Counts in floats are counts, and nothing warns.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.array_equal(frc.one_frc(counts.astype(np.float64), method, seed=4, tukey_alpha=0.125), curve)
        assert np.array_equal(frc.one_frc(counts.astype(np.float64), method, seed=4), frc.one_frc(counts, method, seed=4))
        assert np.array_equal(frc.one_frc(counts, method, seed=4), frc.one_frc(counts, method, seed=4, tukey_alpha=0))


def test_windowed_images_are_split_as_in_frc_0_1_with_a_warning():
    counts = pattern_image(64, 40.0, 5)
    windowed = frc.util.apply_tukey(counts)
    truncated = windowed.astype(np.int32)
    with pytest.warns(frc.NotCountsWarning, match="not whole numbers"):
        curve = frc.one_frc(windowed, seed=6)
    # The second half keeps the fractions of the image.
    half = np.asarray(frc_core.binom_split(truncated, seed=6))
    assert np.array_equal(curve, frc.two_frc(half, windowed - half))
    assert not np.array_equal(curve, frc.one_frc(truncated, seed=6))
    # Method 2 splits the truncated image twice.
    with pytest.warns(frc.NotCountsWarning, match="not whole numbers"):
        curve = frc.one_frc(windowed, 2, seed=6)
    assert np.array_equal(curve, frc.one_frc(truncated, 2, seed=6))
    # The warning has a class of its own, so that a caller can switch off this one alone.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.simplefilter("ignore", frc.NotCountsWarning)
        assert np.array_equal(frc.one_frc(windowed, 2, seed=6), curve)


def test_windowed_images_are_refused_with_a_tukey_alpha():
    windowed = frc.util.apply_tukey(pattern_image(32, 40.0, 7))
    for method in (1, 2):
        with pytest.raises(ValueError, match="not whole numbers"):
            frc.one_frc(windowed, method, tukey_alpha=0.125)
    with pytest.raises(ValueError, match="method"):
        frc.one_frc(windowed, 3)


def test_rings_are_floor_unless_asked_for_nearest():
    counts = pattern_image(64, 40.0, 8)
    floats = counts.astype(np.float64)
    assert np.array_equal(frc.one_frc(counts, seed=2), np.asarray(frc_core.one_frc(counts, 2, 0.0, "floor")))
    assert np.array_equal(
        frc.one_frc(counts, seed=2, tukey_alpha=0.125, rings="nearest"), np.asarray(frc_core.one_frc(counts, 2))
    )
    assert np.array_equal(frc.two_frc(floats, floats[::-1]), np.asarray(frc_core.two_frc(floats, floats[::-1], "floor")))
    assert np.array_equal(
        frc.two_frc(floats, floats[::-1], rings="nearest"), np.asarray(frc_core.two_frc(floats, floats[::-1]))
    )
    # The waves at |f| = 5.83 and 19.24 lie in rings 6 and 19 with "nearest" rings.
    curve = frc.one_frc(counts, seed=2, tukey_alpha=0.125, rings="nearest")
    assert curve[6] > 0.9 and curve[19] > 0.8
    for method in (1, 2):
        assert not np.array_equal(frc.one_frc(counts, method, seed=2), frc.one_frc(counts, method, seed=2, rings="nearest"))


def test_the_half_bit_threshold_follows_the_rings():
    from frc.frc_functions import half_bit_threshold

    xs = np.arange(32) / 64
    floor, nearest = half_bit_threshold(64)(xs), half_bit_threshold(64, "nearest")(xs)
    # Ring 2 holds 16 frequencies with "floor" rings and 12 with "nearest" rings.
    assert floor[0] == nearest[0] and floor[2] < nearest[2]
    curve = frc.one_frc(pattern_image(64, 40.0, 9), seed=3, tukey_alpha=0.125, rings="nearest")
    _, _, threshold = frc.frc_res(xs, curve, 64, "half_bit", rings="nearest")
    assert np.array_equal(threshold(xs), nearest)


def test_two_frc_of_an_image_with_itself_is_one():
    img = frc.util.apply_tukey(pattern_image(32, 50.0, 4).astype(np.float64))
    assert np.allclose(frc.two_frc(img, img), 1.0)
    with pytest.raises(ValueError):
        frc.two_frc(img, img[:16, :16])


def test_half_bit_threshold_counts_the_rings_of_twice_the_size():
    from frc.frc_functions import half_bit_threshold

    # The threshold at ring 0 has one frequency: (s + f + 1) / (s + 1 + f), with s = sqrt(2)/2 - 1/2.
    assert half_bit_threshold(16)(0.0) == pytest.approx(1.0)
    # DIPlib's RadialSum counted 8 frequencies in ring 1.
    s = 0.5 * np.sqrt(2) - 0.5
    f = np.sqrt(s) * 2
    want = (s * np.sqrt(8) + (f + 1)) / ((s + 1) * np.sqrt(8) + f)
    assert half_bit_threshold(16)(1 / 16) == pytest.approx(want)
