"""Compound signal: calibrated levels, IR and background noise (PROVISIONAL).

Statistical tolerances were set from a private reference over four seeds
(2026-11-05): noise only at S/N 0 and +5 dB, STI within 0.0045 of [2] Eq. (2) via
sti_from_mtf (the tests allow 0.009); exponential reverberation T = 1 s plus noise within
0.028 of [2] Eq. (6) over ten seeds (seed 20, used here, is the worst; without noise
within 0.011), the test allows 0.04.
"""

import numpy as np
import pytest

from resonus_room.core.sti import (
    MODULATION_FREQUENCIES,
    P0,
    band_level,
    compound_signal,
    measure_mtf_compound,
    modulated_noise,
    mtf_noise,
    mtf_reverberation,
    sti_from_mtf,
)

FS = 32000.0
DURATION = 4.0
N = int(DURATION * FS)
F = np.array(MODULATION_FREQUENCIES)
DELTA = [1.0]


def test_reference_pressure():
    assert P0 == 20e-6  # ISO 1683:2015 Table 1


def test_band_level_of_a_sine():
    # 1 Pa amplitude at the 1 kHz centre: 20 lg(1 / sqrt(2) / 20e-6) = 90.97 dB
    x = np.sin(2 * np.pi * 1000.0 * np.arange(int(2 * FS)) / FS)
    assert band_level(x, FS, 3, start_s=0.5) == pytest.approx(90.969, abs=0.05)
    assert band_level(x, FS, 0, start_s=0.5) < 50.0
    assert isinstance(band_level(x, FS, 3), float)


def test_band_level_edge_cases():
    assert band_level(np.zeros(1000), FS, 2) == -np.inf
    with pytest.raises(ValueError, match="no samples after start_s"):
        band_level(np.ones(100), FS, 2, start_s=1.0)
    with pytest.raises(ValueError, match="band must be 0...6"):
        band_level(np.ones(100), FS, 7)


def test_compound_signal_levels():
    test = modulated_noise(3, 2.0, 2.0, FS, np.random.default_rng(1))
    n = test.size
    y = compound_signal(test, FS, 3, 65.0, DELTA, DELTA, np.zeros(n))
    assert y.shape == (n,) and y.dtype == np.float64
    assert band_level(y, FS, 3) == pytest.approx(65.0, abs=1e-6)
    # in-situ response at half the 1 m pressure: -6.02 dB
    y = compound_signal(test, FS, 3, 65.0, DELTA, [0.5], np.zeros(n))
    assert band_level(y, FS, 3) == pytest.approx(65.0 - 20 * np.log10(2), abs=1e-6)
    # a source twice as loud needs half the drive: same at the same room response
    y = compound_signal(test, FS, 3, 65.0, [2.0], [1.0], np.zeros(n))
    assert band_level(y, FS, 3) == pytest.approx(65.0 - 20 * np.log10(2), abs=1e-6)
    # noise is added unscaled and truncated to the test length
    noise = np.random.default_rng(2).standard_normal(n + 10)
    y0 = compound_signal(test, FS, 3, 65.0, DELTA, DELTA, np.zeros(n))
    y1 = compound_signal(test, FS, 3, 65.0, DELTA, DELTA, noise)
    np.testing.assert_allclose(y1 - y0, noise[:n], atol=1e-12)


def test_compound_signal_errors():
    test = np.ones(100)
    with pytest.raises(ValueError, match="noise shorter than the test signal"):
        compound_signal(test, FS, 3, 60.0, DELTA, DELTA, np.zeros(99))
    with pytest.raises(ValueError, match="band must be 0...6"):
        compound_signal(test, FS, -1, 60.0, DELTA, DELTA, np.zeros(100))


def _noise(seed):
    noise = np.random.default_rng(seed).standard_normal(N) * 0.02  # Pa, 57 dB broadband
    return noise, np.array([band_level(noise, FS, k) for k in range(7)])


@pytest.mark.parametrize("snr", [0.0, 5.0])
def test_noise_only(snr):
    noise, ln = _noise(10)
    m = measure_mtf_compound(FS, DURATION, np.random.default_rng(11), ln + snr, DELTA, DELTA, noise)
    assert m.shape == (7, 14)
    expected = sti_from_mtf(np.full((7, 14), mtf_noise(snr))).sti  # [2] Eq. (2)
    assert sti_from_mtf(m).sti == pytest.approx(expected, abs=0.009)


def test_reverberation_and_noise():
    rng = np.random.default_rng(20)
    t60 = 1.0
    t = np.arange(int(1.5 * FS)) / FS
    h = rng.standard_normal(t.size) * np.exp(-3 * np.log(10) * t / t60)  # -60 dB at t60
    h /= np.sqrt(np.sum(h**2))
    noise, _ = _noise(21)
    levels = np.full(7, 70.0)
    # received speech and noise levels per band, after the build-up (start 1 s)
    lr = []
    for k in range(7):
        test = modulated_noise(k, 1.0, DURATION, FS, np.random.default_rng(22))
        y = compound_signal(test, FS, k, levels[k], DELTA, h, np.zeros(N))
        lr.append(band_level(y, FS, k, start_s=1.0))
    ln = np.array([band_level(noise, FS, k, start_s=1.0) for k in range(7)])
    expected = mtf_reverberation(F, t60)[None, :] * mtf_noise(np.array(lr) - ln)[:, None]
    m = measure_mtf_compound(FS, DURATION, rng, levels, DELTA, h, noise, start_s=1.0)
    assert sti_from_mtf(m).sti == pytest.approx(sti_from_mtf(expected).sti, abs=0.04)


def test_measure_mtf_compound_levels_shape():
    with pytest.raises(ValueError, match="speech_levels_db must have 7 values"):
        measure_mtf_compound(
            FS, 1.0, np.random.default_rng(0), [60.0] * 6, DELTA, DELTA, np.zeros(N)
        )
