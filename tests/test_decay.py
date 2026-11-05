import numpy as np
import pytest

from resonus_room.core.decay import (
    METHODS,
    DecayCurve,
    Truncation,
    decay_curve,
    tail_correction,
    truncation_point,
)

FS = 48000.0
T60 = 1.0
K = 6 * np.log(10) / T60  # energy exp(-K t) decays by 60 dB in T60


def decay(noise_energy: float = 0.0, seconds: float = 2.0) -> np.ndarray:
    """Deterministic decay sqrt(exp(-K t) + noise): closed-form energy, no carrier."""
    t = np.arange(int(seconds * FS)) / FS
    return np.sqrt(np.exp(-K * t) + noise_energy)


@pytest.mark.parametrize("nf_db", [-50.0, -40.0, -30.0])
def test_truncation_point_at_noise_intersection(nf_db):
    nf = 10 ** (nf_db / 10)
    tr = truncation_point(decay(nf), FS)
    assert isinstance(tr, Truncation)
    assert tr.background_db == pytest.approx(nf_db, abs=0.01)
    # Decay meets the noise at t1 = -nf_db / 60 dB/s; noise bends the fitted line a little.
    assert tr.index / FS == pytest.approx(-nf_db / 60, rel=0.025)
    assert tr.slope == pytest.approx(-60.0, rel=0.03)


def test_truncation_point_without_usable_decay():
    p = np.ones(4800)  # flat: envelope never falls 5 dB below its maximum
    tr = truncation_point(p, FS)
    assert tr.index == 4800 and np.isnan(tr.slope) and np.isnan(tr.intercept)
    with pytest.raises(ValueError):
        truncation_point(decay(), FS, top_db=0.0)
    with pytest.raises(ValueError):
        truncation_point(decay(), FS, margin_db=-1.0)


def test_tail_correction_with_measured_line():
    # With the measured line, C approximates the true remaining decay energy nf/K.
    tr = truncation_point(decay(1e-5), FS)
    assert tail_correction(tr.background_db, tr.slope) == pytest.approx(1e-5 / K, rel=0.03)


@pytest.mark.parametrize("method", METHODS)
def test_decay_curve_noise_free_is_linear(method):
    d = decay_curve(decay(), FS, method)
    assert isinstance(d, DecayCurve)
    assert d.time.shape == d.level.shape == (d.truncation,)
    assert d.level[0] == 0.0
    sel = d.level > -40
    np.testing.assert_allclose(d.level[sel], -60 * d.time[sel] / T60, atol=0.02)
    assert d.peak_to_noise_db > 100


def test_decay_curve_noise_compensation():
    p = decay(1e-5)  # background 50 dB below the peak
    ideal_at_05 = -30.0  # level of the ideal decay at t = 0.5 s
    j = int(0.5 * FS)
    none = decay_curve(p, FS, "none")
    trunc = decay_curve(p, FS, "truncate")
    corr = decay_curve(p, FS, "correct")
    assert none.truncation == p.size and none.correction == 0.0
    assert trunc.truncation == pytest.approx(0.8333 * FS, rel=0.025) and trunc.correction == 0.0
    assert corr.truncation == trunc.truncation and corr.correction > 0
    assert none.level[j] - ideal_at_05 > 0.6  # noise lifts the uncompensated tail
    assert abs(corr.level[j] - ideal_at_05) < 0.3
    assert abs(trunc.level[j] - ideal_at_05) < 0.3
    for d in (none, trunc, corr):
        assert d.background_db == pytest.approx(-50.0, abs=0.01)
        assert d.peak_to_noise_db == pytest.approx(50.0, abs=0.2)
    with pytest.raises(ValueError):
        decay_curve(p, FS, "lundeby")


def test_digital_silence_tail():
    # Exact zeros at the end (e.g. a windowed or synthetic IR): background level -inf,
    # no truncation, the curve is the plain backward integral.
    p = decay()
    p[int(1.5 * FS) :] = 0.0
    tr = truncation_point(p, FS)
    assert tr.index == p.size and tr.background_db == -np.inf and np.isnan(tr.slope)
    d = decay_curve(p, FS, "correct")
    assert d.truncation == p.size and d.correction == 0.0
    sel = d.level > -60
    np.testing.assert_allclose(d.level[sel], -60 * d.time[sel] / T60, atol=0.05)


def test_truncation_point_with_exact_zeros_inside_the_decay():
    # Quantized IRs (e.g. REW 24-bit WAV exports) contain runs of exact zeros once the
    # decay drops below one LSB; such samples (level -inf) must not enter the line fit.
    # Background far below the decay (filter ringing in the padding, about -240 dB), so the
    # first point below background + margin is an exact zero inside the decay.
    p = decay()
    p[int(0.30 * FS) : int(0.35 * FS)] = 0.0
    p[int(1.5 * FS) :] = 1e-12
    tr = truncation_point(p, FS)
    assert np.isfinite(tr.slope) and tr.slope < 0
    assert 0 < tr.index <= p.size
    d = decay_curve(p, FS, "correct")
    assert np.all(np.isfinite(d.level[: d.truncation // 2]))
