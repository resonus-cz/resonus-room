import numpy as np
import pytest
from resonus_core.bands import band_set, filter_bands

from resonus_room.core.energy import (
    EARLY_50,
    EARLY_80,
    PAD,
    EnergyParameters,
    centre_time,
    early_late_energy,
    energy_parameters,
)

FS = 48000.0
K = 6 * np.log(10) / 1.0  # energy exp(-K t): T = 1 s


def exp_ir(seconds: float = 3.0) -> np.ndarray:
    t = np.arange(int(seconds * FS)) / FS
    return np.exp(-K * t / 2)  # p**2 = exp(-K t)


def test_constants():
    assert (EARLY_50, EARLY_80) == (0.050, 0.080)
    assert PAD == 0.5


def test_broadband_geometric_closed_form():
    # p[j]**2 = r**j: early = (1 - r**ne)/(1 - r)/fs, late = (r**ne - r**n)/(1 - r)/fs.
    x = exp_ir()
    n = x.size
    r = np.exp(-K / FS)
    for te in (EARLY_50, EARLY_80):
        ne = round(te * FS)
        e, l_ = early_late_energy(x, FS, 0, te)
        assert e.shape == l_.shape == (1,)
        np.testing.assert_allclose(e, (1 - r**ne) / (1 - r) / FS, rtol=1e-10)
        np.testing.assert_allclose(l_, (r**ne - r**n) / (1 - r) / FS, rtol=1e-10)


def test_broadband_parameters_continuous_limit():
    # Continuous exponential decay, infinite length: C_te = 10 lg(exp(K te) - 1),
    # D50 = 1 - exp(-K * 0.05), Ts = 1/K (discretization error ~ 1/(2 fs)).
    p = energy_parameters(exp_ir(), FS, 0)
    assert isinstance(p, EnergyParameters)
    assert p.c50.shape == p.c80.shape == p.d50.shape == p.ts.shape == (1,)
    np.testing.assert_allclose(p.c50, 10 * np.log10(np.exp(K * 0.05) - 1), atol=2e-3)
    np.testing.assert_allclose(p.c80, 10 * np.log10(np.exp(K * 0.08) - 1), atol=2e-3)
    np.testing.assert_allclose(p.d50, 1 - np.exp(-K * 0.05), atol=1e-4)
    np.testing.assert_allclose(p.ts, 1 / K, rtol=1e-3)
    # C50 and D50 are related exactly by Eq. (A.12).
    np.testing.assert_allclose(p.c50, 10 * np.log10(p.d50 / (1 - p.d50)), rtol=1e-10)


def test_onset_shift():
    x = exp_ir()
    shifted = np.concatenate([np.zeros(1234), x])
    a = energy_parameters(x, FS, 0)
    b = energy_parameters(shifted, FS, 1234)
    for name in ("c50", "c80", "d50", "ts"):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=1e-12)


def test_band_energies_follow_window_before_filtering():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(int(1.5 * FS)) * exp_ir(1.5)
    bands = band_set(1, 250, 2000)
    e, l_ = early_late_energy(x, FS, 100, EARLY_80, bands, order=2, reverse=True)
    assert e.shape == l_.shape == (4,)
    seg = x[100:]
    ne = round(EARLY_80 * FS)
    npad = round(PAD * FS)
    z = np.zeros(npad)
    ref_e = np.sum(filter_bands(np.concatenate([z, seg[:ne], z]), bands, FS, 2, True) ** 2, 0)
    ref_l = np.sum(filter_bands(np.concatenate([z, seg[ne:], z]), bands, FS, 2, True) ** 2, 0)
    np.testing.assert_allclose(e, ref_e / FS, rtol=1e-10)
    np.testing.assert_allclose(l_, ref_l / FS, rtol=1e-10)


def test_band_values_near_broadband_expectation():
    # Noise carrier with T = 1 s in every band: energies pooled over 32 seeded
    # realizations approach the continuous-limit values of an exponential decay.
    bands = band_set(1, 500, 4000)
    e80 = l80 = e50 = l50 = 0.0
    for seed in range(32):
        x = np.random.default_rng(seed).standard_normal(int(1.2 * FS)) * exp_ir(1.2)
        e, l_ = early_late_energy(x, FS, 0, EARLY_80, bands)
        e80, l80 = e80 + e, l80 + l_
        e, l_ = early_late_energy(x, FS, 0, EARLY_50, bands)
        e50, l50 = e50 + e, l50 + l_
    np.testing.assert_allclose(
        10 * np.log10(e80 / l80), 10 * np.log10(np.exp(K * 0.08) - 1), atol=0.6
    )
    np.testing.assert_allclose(e50 / (e50 + l50), 1 - np.exp(-K * 0.05), atol=0.04)
    p = energy_parameters(x, FS, 0, bands)
    assert p.c80.shape == p.ts.shape == (4,)
    np.testing.assert_allclose(p.ts, 1 / K, atol=0.015)


def test_centre_time_band_time_axis():
    # A unit impulse at the onset: band Ts equals the energy centroid of the filter's
    # impulse response (positive delay forward, negative with time reversal).
    x = np.zeros(int(0.5 * FS))
    x[0] = 1.0
    bands = band_set(1, 1000, 1000)
    fwd = centre_time(x, FS, 0, bands)
    rev = centre_time(x, FS, 0, bands, reverse=True)
    assert fwd.shape == (1,) and fwd[0] > 0
    np.testing.assert_allclose(rev, -fwd, rtol=1e-6)
    np.testing.assert_allclose(centre_time(x, FS, 0), [0.0])


def test_invalid_inputs():
    x = exp_ir(0.2)
    with pytest.raises(ValueError):
        early_late_energy(x, FS, 0, 0.0)
    with pytest.raises(ValueError):
        early_late_energy(x, FS, 0, 0.5)  # no late samples
    with pytest.raises(ValueError):
        early_late_energy(x, FS, x.size, EARLY_50)
    with pytest.raises(ValueError):
        early_late_energy(x[:, None], FS, 0, EARLY_50)
    with pytest.raises(ValueError):
        centre_time(x, 0.0, 0)
    with pytest.raises(ValueError):
        energy_parameters(np.array([]), FS, 0)
