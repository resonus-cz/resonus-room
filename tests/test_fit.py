import numpy as np
import pytest

from resonus_room.core.fit import (
    EVALUATION_RANGES,
    DecayFit,
    curvature,
    curvature_flags,
    fit_decay,
    fit_flags,
)


def test_evaluation_ranges():
    assert EVALUATION_RANGES == {"EDT": (0.0, -10.0), "T20": (-5.0, -25.0), "T30": (-5.0, -35.0)}


@pytest.mark.parametrize("name", ["EDT", "T20", "T30"])
def test_linear_decay_exact(name):
    t = np.arange(0, 2.0, 1e-3)
    level = -60 * t / 1.3
    fit = fit_decay(t, level, *EVALUATION_RANGES[name], name=name)
    assert isinstance(fit, DecayFit) and fit.name == name
    assert fit.decay_time == pytest.approx(1.3, rel=1e-12)
    assert fit.slope == pytest.approx(-60 / 1.3, rel=1e-12)
    assert fit.intercept == pytest.approx(0.0, abs=1e-9)
    assert fit.r2 == pytest.approx(1.0, abs=1e-12) and fit.xi == pytest.approx(0.0, abs=1e-9)
    start, stop = EVALUATION_RANGES[name]
    # first samples at or below the range limits
    assert fit.t_start == t[np.argmax(level <= start)]
    assert fit.t_stop == t[np.argmax(level <= stop)]
    assert fit.n_points == np.argmax(level <= stop) - np.argmax(level <= start) + 1


def test_annex_c_hand_computed():
    # 4 points: mean t = 1.5, mean L = -9.25, Sxy = -33.5, Sxx = 5 -> b = -6.7, a = 0.8;
    # SSreg = b**2 * Sxx = 224.45, SStot = 234.75.
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    level = np.array([0.0, -6.0, -10.0, -21.0, -30.0])
    fit = fit_decay(t, level, 0.0, -20.0)
    assert fit.n_points == 4 and fit.t_start == 0.0 and fit.t_stop == 3.0
    assert fit.slope == pytest.approx(-6.7) and fit.intercept == pytest.approx(0.8)
    assert fit.decay_time == pytest.approx(60 / 6.7)
    assert fit.r2 == pytest.approx(224.45 / 234.75)
    assert fit.xi == pytest.approx(1000 * (1 - 224.45 / 234.75))


def test_first_reaching_with_non_monotonic_curve():
    t = np.arange(8.0)
    level = np.array([0.0, -4.0, -6.0, -4.5, -12.0, -26.0, -30.0, -40.0])
    fit = fit_decay(t, level, -5.0, -25.0)
    assert fit.t_start == 2.0 and fit.t_stop == 5.0 and fit.n_points == 4


def test_range_not_covered():
    t = np.arange(0, 1.0, 1e-3)
    level = -20 * t  # reaches only -20 dB
    fit = fit_decay(t, level, -5.0, -35.0, "T30")
    assert fit.n_points == 0 and np.isnan(fit.decay_time) and np.isnan(fit.slope)
    assert fit.t_start == pytest.approx(0.25) and np.isnan(fit.t_stop)
    assert fit_flags(fit, 80.0) == ("range",)
    # fewer than 3 points also counts as not covered
    fit = fit_decay([0.0, 1.0, 2.0], [0.0, -6.0, -30.0], -5.0, -25.0)
    assert fit.n_points == 0 and np.isnan(fit.decay_time)


def test_rising_curve_gives_nan_time():
    fit = fit_decay([0.0, 1.0, 2.0, 3.0], [-6.0, -30.0, -29.0, -28.0], -5.0, -25.0)
    # samples 0..1 only -> fewer than 3 points
    assert np.isnan(fit.decay_time)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        fit_decay([0.0, 1.0], [0.0], -5.0, -25.0)
    with pytest.raises(ValueError):
        fit_decay(np.zeros((2, 2)), np.zeros((2, 2)), -5.0, -25.0)
    with pytest.raises(ValueError):
        fit_decay([0.0, 1.0], [0.0, -1.0], -25.0, -5.0)
    with pytest.raises(ValueError):
        fit_decay([0.0, 1.0], [0.0, -1.0], 1.0, -5.0)


def test_curvature_iso3382_2_b4():
    assert curvature(1.0, 1.05) == pytest.approx(5.0)
    assert curvature(2.0, 1.9) == pytest.approx(-5.0)
    assert np.isnan(curvature(float("nan"), 1.0))
    assert np.isnan(curvature(0.0, 1.0))
    assert curvature_flags(5.0) == ()
    assert curvature_flags(10.5) == ("curvature",)
    assert curvature_flags(-0.1) == ("curvature",)
    assert curvature_flags(float("nan")) == ()


def test_two_slope_decay_has_positive_curvature():
    # Steeper early decay (T = 1 s) followed by a slower one (T = 2 s) after -15 dB.
    t = np.arange(0, 4.0, 1e-3)
    level = np.where(t < 0.25, -60 * t, -15 - 30 * (t - 0.25))
    t20 = fit_decay(t, level, *EVALUATION_RANGES["T20"]).decay_time
    t30 = fit_decay(t, level, *EVALUATION_RANGES["T30"]).decay_time
    assert 1.0 < t20 < t30 < 2.0
    assert curvature(t20, t30) > 0


def test_fit_flags():
    t = np.arange(0, 2.0, 1e-3)
    fit = fit_decay(t, -60 * t / 0.5, -5.0, -35.0, "T30")  # T = 0.5 s
    assert fit_flags(fit, 46.0) == ()
    assert fit_flags(fit, 44.9) == ("dynamic_range",)
    # one-third octave at 100 Hz: B = 0.23 * 100 = 23 Hz, BT = 11.5
    assert fit_flags(fit, 60.0, bandwidth=23.0) == ("bt",)
    assert fit_flags(fit, 60.0, bandwidth=23.0, reverse=True) == ()
    assert fit_flags(fit, 60.0, bandwidth=32.0) == ("bt",)  # BT = 16 is not > 16
    assert fit_flags(fit, 60.0, bandwidth=33.0) == ()
    edt = fit_decay(t, -60 * t / 0.5, 0.0, -10.0, "EDT")
    assert fit_flags(edt, 25.0) == ()  # 10 dB range + 15 dB
    assert fit_flags(edt, 24.0) == ("dynamic_range",)


def test_nonlinear_flag():
    t = np.arange(0, 3.0, 1e-3)
    level = -20 * t + 4 * np.sin(2 * np.pi * 2 * t)  # wavy decay
    fit = fit_decay(t, level, -5.0, -25.0)
    assert fit.xi > 10
    assert "nonlinear" in fit_flags(fit, 80.0)
