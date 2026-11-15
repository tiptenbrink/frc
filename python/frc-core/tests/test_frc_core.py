import pytest

import frc_core

np = pytest.importorskip("numpy")


def pattern_image(n: int, mean: float, seed: int) -> np.ndarray:
    """Poisson counts around mean with three cosine waves, at |f| = 5.83, 19.24 and 51.86: rings
    6, 19 and 52 with "nearest" rings, and 5, 19 and 51 with "floor" rings."""
    y, x = np.mgrid[0:n, 0:n]
    waves = sum(a * np.cos(2 * np.pi * (u * x + v * y) / n) for u, v, a in [(3, 5, 0.4), (17, -9, 0.2), (40, 33, 0.1)])
    return np.random.default_rng(seed).poisson(mean * (1 + waves)).astype(np.int32)


def test_pattern_rings_correlate():
    img = pattern_image(256, 100.0, 1)
    frc = np.asarray(frc_core.one_frc(img, seed=3))
    assert frc.shape == (128,) and frc.dtype == np.float64
    assert np.all(frc[[6, 19, 52]] > 0.9)
    assert np.median(np.abs(frc[60:])) < 0.1
    floor = np.asarray(frc_core.one_frc(img, seed=3, rings="floor"))
    assert np.all(floor[[5, 19, 51]] > 0.9)


def test_same_seed_same_curve():
    img = pattern_image(64, 30.0, 2)
    plan = frc_core.FrcPlan(64)
    assert plan.size == 64
    assert plan.one_frc(img, seed=5) == plan.one_frc(img, seed=5)
    assert plan.one_frc(img, seed=5) != plan.one_frc(img, seed=6)
    # Without a seed, every call splits anew.
    assert plan.one_frc(img) != plan.one_frc(img)


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.uint16, np.int64, np.uint64])
def test_every_integer_type_gives_the_same_curve(dtype):
    img = pattern_image(48, 20.0, 3)
    assert frc_core.one_frc(img.astype(dtype), seed=1) == frc_core.one_frc(img, seed=1)


def test_strided_and_contiguous_images_give_the_same_curve():
    big = pattern_image(64, 20.0, 4)
    view = big[::2, 1::2]
    assert not view.flags.c_contiguous
    assert frc_core.one_frc(view, seed=2) == frc_core.one_frc(np.ascontiguousarray(view), seed=2)
    assert frc_core.one_frc(big.T, seed=2) == frc_core.one_frc(np.ascontiguousarray(big.T), seed=2)


def test_reversed_images_read_from_before_their_first_element():
    big = pattern_image(64, 20.0, 6)
    for view in (big[::-1, ::-1], big[::-2, 1::2].astype(np.uint16)[::-1]):
        assert view.strides[0] < 0
        copy = np.ascontiguousarray(view)
        assert frc_core.one_frc(view, seed=3) == frc_core.one_frc(copy, seed=3)
        assert np.array_equal(frc_core.binom_split(view, seed=3), frc_core.binom_split(copy, seed=3))
        floats = view.astype(np.float64)
        assert frc_core.two_frc(floats, floats[::-1]) == frc_core.two_frc(
            np.ascontiguousarray(floats), np.ascontiguousarray(floats[::-1])
        )


def test_two_frc_of_the_halves_of_a_split_sees_the_pattern():
    img = pattern_image(64, 50.0, 5)
    half = np.asarray(frc_core.binom_split(img, seed=9))
    curve = np.asarray(frc_core.two_frc(half, img - half))
    assert curve.shape == (32,)
    assert np.all(curve[[6, 19]] > 0.8)


@pytest.mark.parametrize("n", [64, 300, 1030])
def test_one_frc_is_two_frc_of_the_halves_that_binom_split_makes(n):
    img = pattern_image(n, 50.0, 7)
    half = np.asarray(frc_core.binom_split(img, seed=11))
    assert frc_core.one_frc(img, seed=11, tukey_alpha=0) == frc_core.two_frc(half, img - half)
    # Another element type and a view with strides hold the same image.
    assert np.array_equal(frc_core.binom_split(img.astype(np.uint16), seed=11), half)
    assert np.array_equal(frc_core.binom_split(img.T, seed=11), frc_core.binom_split(np.ascontiguousarray(img.T), seed=11))


def tukey_square(n: int, alpha: float) -> np.ndarray:
    """The window of scipy.signal.windows.tukey(n, alpha) along both axes, for alpha < 1."""
    from_edge = np.minimum(np.arange(n), n - 1 - np.arange(n))
    rise = alpha * (n - 1) / 2
    window = np.where(from_edge <= np.floor(rise), 0.5 * (1 - np.cos(np.pi * from_edge / rise)), 1.0)
    return np.outer(window, window)


@pytest.mark.parametrize(("n", "alpha"), [(64, 0.125), (101, 0.125), (300, 0.5)])
def test_one_frc_windows_both_halves_after_the_split(n, alpha):
    img = pattern_image(n, 50.0, 8)
    half = np.asarray(frc_core.binom_split(img, seed=12))
    window = tukey_square(n, alpha)
    want = np.asarray(frc_core.two_frc(window * half, window * (img - half)))
    # one_frc multiplies in float32.
    assert np.allclose(frc_core.one_frc(img, seed=12, tukey_alpha=alpha), want, atol=1e-5)
    assert np.allclose(frc_core.FrcPlan(n).one_frc(img, seed=12, tukey_alpha=alpha), want, atol=1e-5)
    assert not np.allclose(frc_core.one_frc(img, seed=12, tukey_alpha=0), want, atol=1e-5)


def test_the_default_window_is_a_tukey_window_with_an_alpha_of_an_eighth():
    img = pattern_image(64, 50.0, 9)
    assert frc_core.one_frc(img, seed=1) == frc_core.one_frc(img, seed=1, tukey_alpha=0.125)
    assert frc_core.one_frc(img, seed=1) != frc_core.one_frc(img, seed=1, tukey_alpha=0)


@pytest.mark.parametrize("alpha", [-0.1, 1.5, float("nan")])
def test_alphas_outside_zero_to_one_are_refused(alpha):
    with pytest.raises(ValueError, match="tukey_alpha"):
        frc_core.one_frc(np.ones((8, 8), dtype=np.int32), tukey_alpha=alpha)


def test_two_frc_of_an_image_with_itself_is_one():
    img = pattern_image(32, 50.0, 6).astype(np.float64)
    assert np.allclose(frc_core.two_frc(img, img), 1.0)


def test_binom_split_keeps_the_shape_and_halves_the_mean():
    a = np.full((3, 50, 70), 1000, dtype=np.int64)
    k = np.asarray(frc_core.binom_split(a, seed=1))
    assert k.shape == a.shape and k.dtype == np.int32
    assert np.all((k >= 0) & (k <= 1000))
    assert abs(k.mean() - 500) < 1.5
    assert frc_core.binom_split(a, seed=1) == frc_core.binom_split(a, seed=1)
    # The result is the caller's own array, to change in place.
    assert k.flags.writeable and k.flags.c_contiguous
    assert np.asarray(frc_core.binom_split(np.zeros((0, 4), dtype=np.int32))).shape == (0, 4)
    odd = np.asarray(frc_core.binom_split(np.full(7, 9, dtype=np.uint8), seed=1))
    assert odd.shape == (7,) and odd.dtype == np.int32 and np.all(odd <= 9)


@pytest.mark.parametrize(("rings", "ring_of"), [("nearest", lambda r: int(r + 0.5)), ("floor", int)])
def test_ring_counts_cover_the_disk(rings, ring_of):
    plan = frc_core.FrcPlan(16, rings)
    assert plan.rings == rings
    counts = np.asarray(plan.ring_counts())
    want = np.zeros(8, dtype=np.uint64)
    for fy in range(-8, 8):
        for fx in range(-8, 8):
            r = ring_of((fx * fx + fy * fy) ** 0.5)
            if r < 8:
                want[r] += 1
    assert counts.tolist() == want.tolist()
    assert frc_core.ring_counts(16, rings) == plan.ring_counts()


def test_the_default_rings_are_nearest():
    assert frc_core.FrcPlan(16).rings == "nearest"
    assert frc_core.ring_counts(16) == frc_core.ring_counts(16, "nearest") != frc_core.ring_counts(16, "floor")
    img = pattern_image(64, 50.0, 10)
    assert frc_core.one_frc(img, seed=1) == frc_core.one_frc(img, seed=1, rings="nearest")
    assert frc_core.one_frc(img, seed=1, rings="floor") == frc_core.FrcPlan(64, "floor").one_frc(img, seed=1)
    floats = img.astype(np.float32)
    assert frc_core.two_frc(floats, floats[::-1], rings="floor") == frc_core.FrcPlan(64, "floor").two_frc(floats, floats[::-1])


def test_resolution_interpolates_the_first_crossing():
    curve = np.array([1.0, 0.6, 0.3, 0.2, 0.1, 0.2])
    # Between ring 3 at 0.2 and ring 4 at 0.1, the curve crosses 0.15 at ring 3.5.
    assert frc_core.resolution(curve, 12, 0.15) == (12 / 3.5, 0.15)
    resolution, threshold = frc_core.resolution(curve, 12)
    assert threshold == 1 / 7 and resolution == pytest.approx(12 / (3 + (0.2 - 1 / 7) / 0.1))
    assert frc_core.resolution(curve, 12, 0.05) is None
    assert frc_core.resolution(memoryview(curve.astype(np.float32)), 12, 0.15)[0] == pytest.approx(12 / 3.5)


def test_resolution_of_a_plan_uses_its_rings():
    img = pattern_image(128, 40.0, 11)
    for rings in ("nearest", "floor"):
        plan = frc_core.FrcPlan(128, rings)
        curve = plan.one_frc(img, seed=2)
        for threshold in ("1/7", "half_bit", 0.5):
            assert plan.resolution(curve, threshold) == frc_core.resolution(curve, 128, threshold, rings)
    # The 1/2-bit threshold depends on the number of frequencies in each ring.
    curve = np.linspace(1.0, 0.0, 64)
    assert frc_core.resolution(curve, 128, "half_bit", "nearest") != frc_core.resolution(curve, 128, "half_bit", "floor")
    # The threshold of ring 0, which holds the origin alone, is exactly 1, the FRC there.
    assert frc_core.resolution(curve, 128, 1.0) == (float("inf"), 1.0)


def test_bad_resolution_arguments_are_refused():
    curve = np.linspace(1.0, 0.0, 8)
    with pytest.raises(ValueError, match="threshold"):
        frc_core.resolution(curve, 16, "1/8")
    with pytest.raises(ValueError, match="rings"):
        frc_core.resolution(curve, 14)
    with pytest.raises(TypeError, match="floats"):
        frc_core.resolution(np.arange(8), 16)
    with pytest.raises(ValueError, match="1D"):
        frc_core.resolution(np.ones((2, 4)), 16)


def test_unknown_rings_are_refused():
    with pytest.raises(ValueError, match="rings"):
        frc_core.FrcPlan(16, "round")
    with pytest.raises(ValueError, match="rings"):
        frc_core.ring_counts(16, "Nearest")


@pytest.mark.parametrize("shape", [(8, 9), (8,), (2, 2, 2), (1, 1)])
def test_non_square_images_are_refused(shape):
    with pytest.raises(ValueError):
        frc_core.one_frc(np.ones(shape, dtype=np.int32))


def test_bad_counts_are_refused():
    plan = frc_core.FrcPlan(16)
    with pytest.raises(ValueError, match="plan is for"):
        plan.one_frc(np.ones((8, 8), dtype=np.int32))
    img = np.ones((16, 16), dtype=np.int32)
    img[3, 4] = -1
    with pytest.raises(ValueError, match="negative"):
        plan.one_frc(img)
    with pytest.raises(ValueError, match="above"):
        frc_core.binom_split(np.array([2**31], dtype=np.int64))
    with pytest.raises(TypeError, match="integer counts"):
        plan.one_frc(np.ones((16, 16)))
    with pytest.raises(TypeError, match="buffer protocol"):
        frc_core.binom_split([1, 2, 3])
