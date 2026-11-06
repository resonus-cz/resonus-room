"""Route (b): modulated test signal and envelope analysis (PROVISIONAL).

Statistical tolerances were set from a private reference over four seeds
(2026-11-05): identity channel m >= 0.92 and STI >= 0.997; noise at S/N = 0 dB STI within
0.003 of 0.5; exponential reverberation T = 1.5 s STI within 0.021 of the closed form.
The tests allow about twice that.
"""

import numpy as np
import pytest
import scipy.signal as ss

from resonus_room.core.sti import (
    BAND_CENTERS,
    MODULATION_FREQUENCIES,
    OCTAVE_BANDS,
    measure_mtf,
    modulated_noise,
    modulation_index,
    mtf_reverberation,
    sti_from_mtf,
)

FS = 32000.0
F = np.array(MODULATION_FREQUENCIES)


def test_band_centers_are_exact_base_ten():
    # IEC 61260-1:2014 Eq. (2) via resonus-core: 1000 * 10**(0.3 x)
    expected = 1000 * 10 ** (0.3 * np.arange(-3, 4))
    np.testing.assert_allclose(BAND_CENTERS, expected)
    assert len(BAND_CENTERS) == len(OCTAVE_BANDS)


def test_modulated_noise():
    x = modulated_noise(3, 2.0, 4.0, FS, np.random.default_rng(0))
    assert x.shape == (128000,) and x.dtype == np.float64
    assert np.mean(x**2) == pytest.approx(1.0, rel=0.05)  # carrier RMS 1, mean(1 + cos) = 1
    assert modulation_index(x, FS, 3, 2.0) == pytest.approx(1.0, abs=0.08)
    a = modulated_noise(0, 1.0, 1.0, FS, np.random.default_rng(5))
    b = modulated_noise(0, 1.0, 1.0, FS, np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)  # only rng draws the noise


def test_modulation_index_of_a_sine_carrier():
    # deterministic: a tone at the band centre with intensity (1 + m0 cos 2 pi F t)
    t = np.arange(int(6 * FS)) / FS
    for band, frequency, m0 in [(3, 2.5, 0.4), (5, 0.63, 0.8), (1, 12.5, 0.1)]:
        tone = np.sin(2 * np.pi * BAND_CENTERS[band] * t)
        y = tone * np.sqrt(1 + m0 * np.cos(2 * np.pi * frequency * t))
        assert modulation_index(y, FS, band, frequency, start_s=0.5) == pytest.approx(m0, abs=0.01)
    unmodulated = np.sin(2 * np.pi * BAND_CENTERS[4] * t)
    assert modulation_index(unmodulated, FS, 4, 3.15, start_s=0.5) < 0.01
    # whole periods only: 5.3 s at 0.63 Hz still gives the exact index
    tone = np.sin(2 * np.pi * BAND_CENTERS[2] * t[: int(5.3 * FS)])
    y = tone * np.sqrt(1 + 0.5 * np.cos(2 * np.pi * 0.63 * t[: tone.size]))
    assert modulation_index(y, FS, 2, 0.63, start_s=0.5) == pytest.approx(0.5, abs=0.01)


def test_modulation_index_is_capped_at_one():
    t = np.arange(int(4 * FS)) / FS
    y = np.sin(2 * np.pi * BAND_CENTERS[3] * t) * (1 + np.cos(2 * np.pi * 2 * t))  # m > 1
    assert modulation_index(y, FS, 3, 2.0, start_s=0.5) == 1.0


def test_invalid_arguments():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match=r"^band must be 0\.\.\.6$"):
        modulated_noise(7, 1.0, 1.0, FS, rng)
    with pytest.raises(ValueError, match=r"^band must be 0\.\.\.6$"):
        modulation_index(np.ones(100), FS, -1, 1.0)
    with pytest.raises(ValueError, match="^frequency must be > 0$"):
        modulated_noise(0, 0.0, 1.0, FS, rng)
    with pytest.raises(ValueError, match="^frequency must be > 0$"):
        modulation_index(np.ones(100), FS, 0, -2.0)
    with pytest.raises(ValueError, match="^duration must be > 0$"):
        modulated_noise(0, 1.0, 0.0, FS, rng)
    with pytest.raises(ValueError, match="^signal shorter than one modulation period$"):
        modulation_index(np.ones(int(FS)), FS, 0, 0.63)
    with pytest.raises(ValueError):
        modulated_noise(6, 1.0, 1.0, 16000.0, rng)  # 8 kHz band above Nyquist


def test_identity_channel():
    m = measure_mtf(lambda x: x, FS, 8.0, np.random.default_rng(1), start_s=0.5)
    assert m.shape == (7, 14)
    assert np.all(m >= 0.85) and np.all(m <= 1.0)
    assert sti_from_mtf(m).sti >= 0.99


def test_noise_channel():
    rng = np.random.default_rng(2)
    noise_rng = np.random.default_rng(3)
    centres = iter(np.repeat(np.arange(7), 14))

    def add_band_noise(x):
        # independent noise of the same band and power: S/N = 0 dB ([2] Eq. (2): m = 0.5)
        k = next(centres)
        n = ss.sosfilt(ss.butter(3, [BAND_CENTERS[k] / 2**0.5, BAND_CENTERS[k] * 2**0.5],
                                 "bandpass", fs=FS, output="sos"),
                       noise_rng.standard_normal(x.size))  # fmt: skip
        return x + n * np.sqrt(np.mean(x**2) / np.mean(n**2))

    m = measure_mtf(add_band_noise, FS, 8.0, rng, start_s=0.5)
    assert np.all(np.abs(m - 0.5) < 0.25)
    assert sti_from_mtf(m).sti == pytest.approx(0.5, abs=0.01)


def test_reverberation_channel():
    # diffuse exponential decay: noise with intensity falling 60 dB in T ([2] Eq. (1))
    T = 1.5
    rng = np.random.default_rng(4)
    t = np.arange(int(3 * T * FS)) / FS
    h = rng.standard_normal(t.size) * np.exp(-3 * np.log(10) * t / T)
    m = measure_mtf(lambda x: ss.fftconvolve(x, h)[: x.size], FS, 8.0, rng, start_s=T)
    closed = sti_from_mtf(np.tile(mtf_reverberation(F, T), (7, 1))).sti
    assert sti_from_mtf(m).sti == pytest.approx(closed, abs=0.04)
    assert np.all(np.diff(m.mean(axis=0)) < 0.05)  # falls with F, up to estimation noise
