"""STI from the modulation transfer matrix (PROVISIONAL, docs/PLAN_STI.md).

Reference values: Houtgast and Steeneken (1985) [2], Eqs. (1)-(3) and the text of
sec. I C and the caption of Fig. 2 (S/N = 0 dB and T = 1.5 s each give m = 0.5 near
F = 2.5 Hz and STI = 0.5); Steeneken and Houtgast (1980) [1] for the weights.
"""

import numpy as np
import pytest

from resonus_room.core import sti
from resonus_room.core.sti import (
    MODULATION_FREQUENCIES,
    OCTAVE_BANDS,
    SNR_LIMIT,
    WEIGHTS,
    StiResult,
    apparent_snr,
    mtf_noise,
    mtf_reverberation,
    sti_from_mtf,
)

F = np.array(MODULATION_FREQUENCIES)


def matrix(row):
    return np.tile(np.asarray(row, dtype=float), (7, 1))


def test_constants():
    assert OCTAVE_BANDS == (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
    assert len(MODULATION_FREQUENCIES) == 14
    assert MODULATION_FREQUENCIES[0] == 0.63 and MODULATION_FREQUENCIES[-1] == 12.5
    # one-third octave steps: each ratio within 2 % of 10**0.1
    ratios = F[1:] / F[:-1]
    np.testing.assert_allclose(ratios, 10**0.1, rtol=0.02)
    assert WEIGHTS == (0.129, 0.143, 0.114, 0.114, 0.186, 0.171, 0.143)
    assert sum(WEIGHTS) == pytest.approx(1.0)
    assert SNR_LIMIT == 15.0


def test_mtf_noise():
    # [2] sec. I C 2: "for an S/N ratio of 0 dB, m = 0.5 for all modulation frequencies"
    assert mtf_noise(0.0) == pytest.approx(0.5)
    np.testing.assert_allclose(mtf_noise([10.0, -10.0]), [1 / 1.1, 1 / 11])
    assert mtf_noise(np.inf) == 1.0
    assert mtf_noise(np.zeros((2, 3))).shape == (2, 3)


def test_mtf_reverberation():
    # [2] Eq. (1); sec. I C: F = 2.5 Hz and T = 1.5 s give about m = 0.5
    assert mtf_reverberation(2.5, 1.5) == pytest.approx(0.5, abs=0.01)
    expected = (1 + (2 * np.pi * F * 1.5 / 13.8) ** 2) ** -0.5
    np.testing.assert_allclose(mtf_reverberation(F, 1.5), expected)
    np.testing.assert_array_equal(mtf_reverberation(F, 0.0), np.ones(14))
    assert np.all(np.diff(mtf_reverberation(F, 1.0)) < 0)  # low-pass
    with pytest.raises(ValueError, match="^t60 must be >= 0$"):
        mtf_reverberation(F, -0.1)


def test_apparent_snr():
    np.testing.assert_allclose(apparent_snr([0.5, 1 / 1.1, 1 / 11]), [0.0, 10.0, -10.0])
    np.testing.assert_array_equal(apparent_snr([0.0, 1.0, 0.999, 0.001]), [-15, 15, 15, -15])
    assert apparent_snr(np.full((7, 14), 0.5)).shape == (7, 14)
    for bad in ([1.1], [-0.01], [np.nan]):
        with pytest.raises(ValueError, match=r"^m must be in 0\.\.\.1$"):
            apparent_snr(bad)


def test_sti_limits():
    r = sti_from_mtf(np.ones((7, 14)))
    assert isinstance(r, StiResult) and r.sti == pytest.approx(1.0)
    assert sti_from_mtf(np.zeros((7, 14))).sti == pytest.approx(0.0)
    r = sti_from_mtf(np.full((7, 14), 0.5))  # S/N = 0 dB everywhere
    assert r.sti == pytest.approx(0.5)
    assert r.mti.shape == (7,) and r.apparent_snr.shape == (7, 14)
    np.testing.assert_allclose(r.mti, 0.5)


def test_sti_figure_2_conditions():
    # [2] Fig. 2 caption: reverberation only (T = 1.5 s) or noise only (S/N = 0 dB),
    # "these conditions both correspond with STI ~ 0.5"
    assert sti_from_mtf(matrix(mtf_noise(np.zeros(14)))).sti == pytest.approx(0.5)
    assert sti_from_mtf(matrix(mtf_reverberation(F, 1.5))).sti == pytest.approx(0.5, abs=0.01)


def test_sti_steps():
    # one band clipped, the rest at known S/N: checks the per-band mean and the weights
    snr = np.array([20.0, 9.0, 6.0, 3.0, 0.0, -3.0, -20.0])  # dB, 125 Hz ... 8 kHz
    m = mtf_noise(snr)[:, None] * np.ones((7, 14))
    r = sti_from_mtf(m)
    clipped = np.clip(snr, -15, 15)
    np.testing.assert_allclose(r.apparent_snr, np.repeat(clipped[:, None], 14, axis=1))
    np.testing.assert_allclose(r.mti, (clipped + 15) / 30)
    assert r.sti == pytest.approx((np.dot(WEIGHTS, clipped) + 15) / 30)
    assert r.sti == pytest.approx(np.dot(WEIGHTS, r.mti))
    # a frequency-dependent row: the band value is the mean over the 14 frequencies
    row = mtf_reverberation(F, 1.0) * mtf_noise(5.0)  # [2] Eq. (6)
    r = sti_from_mtf(matrix(row))
    expected = np.clip(10 * np.log10(row / (1 - row)), -15, 15).mean()
    np.testing.assert_allclose(r.mti, (expected + 15) / 30)


def test_sti_shape():
    for shape in [(7, 13), (14, 7), (7,)]:
        with pytest.raises(ValueError, match=r"^m must have shape \(7, 14\)$"):
            sti_from_mtf(np.full(shape, 0.5))


def test_module_is_marked_provisional():
    assert "PROVISIONAL" in sti.__doc__
