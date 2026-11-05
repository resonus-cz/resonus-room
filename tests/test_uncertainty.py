import numpy as np
import pytest

from resonus_room.core.uncertainty import (
    N_DECAYS_IR_METHOD,
    UNCERTAINTY_CONSTANTS,
    SpatialAverage,
    bandwidth,
    ensemble_average,
    iso354_repeatability,
    reverberation_std,
    spatial_average,
)


def test_constants_iso3382_2():
    assert UNCERTAINTY_CONSTANTS[20] == (0.88, 1.90)  # Eq. (2)
    assert UNCERTAINTY_CONSTANTS[30] == (0.55, 1.52)  # Eq. (3)
    assert UNCERTAINTY_CONSTANTS[10] == (1.75, 3.32)  # Table A.1, D = 10 dB, gamma = 5
    assert N_DECAYS_IR_METHOD == 10  # A.3


def test_bandwidth_iso3382_2_7_1():
    np.testing.assert_allclose(bandwidth([125.0, 1000.0], 1), [0.71 * 125, 710.0])
    np.testing.assert_allclose(bandwidth(1000.0, 3), 230.0)
    with pytest.raises(ValueError):
        bandwidth(1000.0, 2)


def test_reverberation_std_eq2_eq3():
    # Eq. (2): T20 = 1 s, B = 710 Hz, N = 6, n = 10:
    # sigma = 0.88 * sqrt((1 + 0.19) / (6 * 710)) = 0.88 * sqrt(1.19 / 4260)
    s = reverberation_std(1.0, 710.0, 6)
    assert s == pytest.approx(0.88 * np.sqrt(1.19 / 4260), rel=1e-12)
    # Eq. (3): T30 = 2 s, B = 23 Hz, N = 12, n = 3
    s = reverberation_std(2.0, 23.0, 12, range_db=30, n_decays=3)
    assert s == pytest.approx(0.55 * 2.0 * np.sqrt((1 + 1.52 / 3) / (12 * 23 * 2.0)), rel=1e-12)
    out = reverberation_std([1.0, np.nan], [710.0, 710.0], 1)
    assert out.shape == (2,) and np.isnan(out[1])
    with pytest.raises(ValueError):
        reverberation_std(1.0, 710.0, 6, range_db=25)
    with pytest.raises(ValueError):
        reverberation_std(1.0, 710.0, 0)
    with pytest.raises(ValueError):
        reverberation_std(1.0, 710.0, 1, n_decays=0)


def test_iso354_repeatability_eq10():
    # T = 5 s at 100 Hz, N = 36 decays: eps = 5 * sqrt((2.42 + 3.59/36) / 500)
    e = iso354_repeatability(5.0, 100.0, 36)
    assert e == pytest.approx(5.0 * np.sqrt((2.42 + 3.59 / 36) / 500.0), rel=1e-12)
    np.testing.assert_allclose(
        iso354_repeatability([2.0, 4.0], [1000.0, 125.0], 12),
        [2 * np.sqrt((2.42 + 3.59 / 12) / 2000), 4 * np.sqrt((2.42 + 3.59 / 12) / 500)],
    )
    with pytest.raises(ValueError):
        iso354_repeatability(1.0, 100.0, 0)


def test_spatial_average():
    v = np.array([[1.0, 2.0, np.nan], [3.0, np.nan, np.nan], [2.0, 4.0, np.nan]])
    a = spatial_average(v)
    assert isinstance(a, SpatialAverage)
    np.testing.assert_allclose(a.mean[:2], [2.0, 3.0])
    assert np.isnan(a.mean[2])
    np.testing.assert_allclose(a.std[:2], [1.0, np.sqrt(2.0)])
    assert np.isnan(a.std[2])
    np.testing.assert_array_equal(a.count, [3, 2, 0])
    one = spatial_average([1.0, 2.0, 3.0])
    np.testing.assert_allclose(one.mean, [2.0])
    np.testing.assert_allclose(one.std, [1.0])
    np.testing.assert_array_equal(one.count, [3])
    single = spatial_average([[1.5, 2.5]])
    np.testing.assert_allclose(single.mean, [1.5, 2.5])
    assert np.all(np.isnan(single.std))
    with pytest.raises(ValueError):
        spatial_average(np.zeros((0, 3)))
    with pytest.raises(ValueError):
        spatial_average(np.zeros((2, 2, 2)))


def test_ensemble_average_iso354_eq2():
    levels = np.array([[0.0, 0.0], [-10.0, -20.0], [-30.0, -30.0]])
    out = ensemble_average(levels)
    np.testing.assert_allclose(out, [0.0, 10 * np.log10((0.1 + 0.01) / 2), -30.0])
    with pytest.raises(ValueError):
        ensemble_average(np.zeros(3))
    with pytest.raises(ValueError):
        ensemble_average(np.zeros((3, 0)))
