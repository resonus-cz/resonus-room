import numpy as np
import pytest

from resonus_room.core.decay import (
    background_level,
    energy_envelope,
    schroeder,
    tail_correction,
)

FS = 48000.0
T60 = 1.0
K = 6 * np.log(10) / T60  # energy exp(-K t) decays by 60 dB in T60


def decay(noise_energy: float = 0.0, seconds: float = 2.0) -> np.ndarray:
    """Deterministic decay sqrt(exp(-K t) + noise): closed-form energy, no carrier."""
    t = np.arange(int(seconds * FS)) / FS
    return np.sqrt(np.exp(-K * t) + noise_energy)


def test_schroeder_closed_form():
    # p[j] = r**(j/2): sum_{j=i}^{n-1} r**j = (r**i - r**n) / (1 - r).
    n = 1000
    r = np.exp(-2 * 30.0 / FS)
    p = r ** (np.arange(n) / 2)
    i = np.arange(n)
    np.testing.assert_allclose(schroeder(p, FS), (r**i - r**n) / (1 - r) / FS, rtol=1e-10)
    e = schroeder(p, FS, stop=600, correction=0.25)
    assert e.shape == (600,)
    np.testing.assert_allclose(e, (r ** i[:600] - r**600) / (1 - r) / FS + 0.25, rtol=1e-10)


def test_schroeder_errors():
    with pytest.raises(ValueError):
        schroeder(np.zeros(10), FS)
    with pytest.raises(ValueError):
        schroeder(np.ones(10), FS, stop=0)
    with pytest.raises(ValueError):
        schroeder(np.ones(10), FS, stop=11)
    with pytest.raises(ValueError):
        schroeder(np.ones(10), FS, correction=-1.0)
    with pytest.raises(ValueError):
        schroeder(np.ones((10, 2)), FS)
    with pytest.raises(ValueError):
        schroeder(np.ones(10), 0.0)


def test_energy_envelope():
    np.testing.assert_allclose(energy_envelope(np.full(100, 2.0), FS), 4.0)  # no edge loss
    x = np.random.default_rng(0).standard_normal(50)
    np.testing.assert_allclose(energy_envelope(x, FS, window=1 / FS), x**2)
    # h = int(3 * 1) // 2 = 1: 3-sample centred average, partial at the ends.
    x = np.array([0.0, 0.0, 3.0, 0.0, 0.0])
    np.testing.assert_allclose(energy_envelope(x, 1.0, window=3.0), [0, 3, 3, 3, 0])
    np.testing.assert_allclose(energy_envelope([2.0, 0.0], 1.0, window=3.0), [2, 2])
    np.testing.assert_allclose(energy_envelope([2.0, 0.0], 1.0, window=10.0), [2, 2])
    with pytest.raises(ValueError):
        energy_envelope(np.ones(10), FS, window=0.0)


def test_background_level():
    env = np.concatenate([np.full(90, 1.0), np.full(10, 1e-4)])
    assert background_level(env) == pytest.approx(-40.0)
    assert background_level(env, 0.2) == pytest.approx(10 * np.log10((10 + 10e-4) / 20))
    with pytest.raises(ValueError):
        background_level(env, 0.0)
    with pytest.raises(ValueError):
        background_level(env, 1.5)


def test_tail_correction_closed_form():
    # Exponential continuation from level B at t1 with slope b: C = 10**(B/10)*10/(-b ln 10).
    assert tail_correction(-50.0, -60.0) == pytest.approx(1e-5 / K, rel=1e-12)
    assert tail_correction(-30.0, -20.0) == pytest.approx(1e-3 * 10 / (20 * np.log(10)))
    assert tail_correction(-30.0, 0.0) == 0.0
    assert tail_correction(-30.0, float("nan")) == 0.0


def test_energy_envelope_precision_over_large_dynamic_range():
    # Exponential decay over 300 dB: the envelope keeps its relative precision at the end
    # (each window sum is a fixed fraction of the remaining decay energy).
    t = np.arange(int(5 * FS)) / FS
    p = np.exp(-K * t / 2)  # p**2 = exp(-K t), 60 dB/s
    env = energy_envelope(p, FS)
    h = int(0.010 * FS) // 2
    i = np.arange(h, p.size - h, 997)
    exact = np.exp(-K * t[i]) * np.sinh(K * (h + 0.5) / FS) / ((2 * h + 1) * np.sinh(K / (2 * FS)))
    np.testing.assert_allclose(env[i], exact, rtol=1e-6)
    assert background_level(np.zeros(10)) == -np.inf
