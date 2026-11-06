"""Speech Transmission Index (STI) from a modulation transfer matrix: PROVISIONAL.

The original STI of Steeneken and Houtgast (1980), in the five steps of Houtgast and
Steeneken (1985), section II B (docs/PLAN_STI.md, draft 3). IEC 60268-16:2020 is not
available yet (Q-STI1); its revised STIr (band weights with redundancy, level-dependent
masking, reception threshold) replaces this when the standard or the 1999 paper is filed
(Q-STI9). Every value here is PROVISIONAL.

The modulation transfer matrix ``m`` has one row per octave band (:data:`OCTAVE_BANDS`)
and one column per modulation frequency (:data:`MODULATION_FREQUENCIES`); each entry is
the modulation reduction factor in 0...1.

References
----------
.. [1] H. J. M. Steeneken, T. Houtgast, "A physical method for measuring
   speech-transmission quality", J. Acoust. Soc. Am. 67(1), 318-326, 1980.
.. [2] T. Houtgast, H. J. M. Steeneken, "A review of the MTF concept in room acoustics
   and its use for estimating speech intelligibility in auditoria", J. Acoust. Soc. Am.
   77(3), 1069-1077, 1985.
.. [3] ISO 1683:2015, Acoustics - Preferred reference values for acoustical and
   vibratory levels, Table 1.
.. [4] J. H. Rindel, "ODEON Application Note - Calculation of Speech Transmission Index
   in rooms", February 2014 (PDF of 2021-01-08), p. 3 and Table 1 (speech SPL at 1 m
   after ANSI S3.5), p. 4 (ISO 9921:2003 Annex A level reduction).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
import scipy.signal as ss
from numpy.typing import ArrayLike, NDArray
from resonus_core.bands import band_filter, band_set
from resonus_core.calibration import P0  # 20 uPa, ISO 1683:2015 Table 1

# Octave-band centre frequencies in Hz, 125 Hz-8 kHz ([2] sec. II B).
OCTAVE_BANDS: Final = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
# Modulation frequencies in Hz, 0.63-12.5 Hz in one-third octave steps ([2] sec. II B);
# the intermediate values are the one-third-octave preferred numbers of ISO 266:1997.
MODULATION_FREQUENCIES: Final = (
    0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5,
)  # fmt: skip
# Octave-band weighting factors W1...W7, 125 Hz-8 kHz: the final set of [1] (sec. I,
# parameter values for the PB-word relation). [2] Eq. (4) rounds them, with 0.12 for
# 1 kHz; the 1980 values are used (docs/PLAN_STI.md, Q-STI8).
WEIGHTS: Final = (0.129, 0.143, 0.114, 0.114, 0.186, 0.171, 0.143)
# Apparent signal-to-noise ratios are truncated to +-15 dB ([2] step 2; [1]: range
# R = 30 dB, zero contribution at S = -15 dB).
SNR_LIMIT: Final = 15.0
# Reverberation: 13.8 = 6 ln 10, the decay of T by 60 dB ([2] Eq. (1)).
_DECAY: Final = 13.8
# Exact octave-band centres (base ten) of OCTAVE_BANDS, for the band filters.
BAND_CENTERS: Final = tuple(float(f) for f in band_set(1, 125.0, 8000.0).center)
# Fraction b of the 1/b-octave band filters of the carrier and the analysis.
_OCTAVE: Final = 1

# Speech spectra per vocal effort (Q-STI3, user decision 2026-11-06: the spectra of [4]).
# [4] Table 1, ANSI S3.5 SPL at 1 m in front of the mouth, octaves 250 Hz-8 kHz, dB re
# 20 uPa, and the A-weighted level of each row (the last column of the table).
SPEECH_SPL_250_8K: Final = {
    "normal": (57.2, 59.8, 53.5, 48.8, 43.8, 38.6),
    "raised": (61.5, 65.6, 62.3, 56.8, 51.3, 42.6),
    "loud": (64.0, 70.3, 70.6, 65.9, 59.9, 48.9),
    "shouted": (65.0, 74.7, 79.8, 75.8, 68.9, 58.2),
}
SPEECH_SPL_A: Final = {"normal": 59.5, "raised": 66.5, "loud": 73.7, "shouted": 82.3}
# 125 Hz is not in ANSI S3.5: "the speech level at 125 Hz has been chosen to be 6 dB
# lower than the 250 Hz level" ([4] p. 3; user decision 2026-11-06, Q-STI3).
SPEECH_125_BELOW_250_DB: Final = 6.0
# ISO 9921:2003 Annex A (normative), as quoted in [4] p. 4: above 75 dB(A) at 1 m the
# speech level is reduced by 0.4 (L_S,A,1m - 75) dB (applied: user decision 2026-11-06).
LOUD_SPEECH_THRESHOLD_DBA: Final = 75.0
LOUD_SPEECH_FACTOR: Final = 0.4
# Vocal efforts of the settings; "custom": the user's own levels (sti_speech_levels).
CUSTOM_EFFORT: Final = "custom"
VOCAL_EFFORTS: Final = (CUSTOM_EFFORT, *SPEECH_SPL_250_8K)


def speech_level_reduction(a_weighted_db: float) -> float:
    """Reduction in dB of the speech level for loud speech (ISO 9921:2003 Annex A, [4]).

    ``LOUD_SPEECH_FACTOR * (a_weighted_db - LOUD_SPEECH_THRESHOLD_DBA)`` above the
    threshold, else 0.0.
    """
    if a_weighted_db <= LOUD_SPEECH_THRESHOLD_DBA:
        return 0.0
    return LOUD_SPEECH_FACTOR * (a_weighted_db - LOUD_SPEECH_THRESHOLD_DBA)


def vocal_effort_levels(effort: str) -> tuple[float, ...]:
    """Speech SPL at 1 m in the 7 octave bands 125 Hz-8 kHz for a vocal effort ([4]).

    ``levels = SPEECH_SPL_250_8K[effort]``; the 125 Hz level is ``levels[0] -
    SPEECH_125_BELOW_250_DB``, followed by ``levels``; every value is then lowered by
    ``speech_level_reduction(SPEECH_SPL_A[effort])`` (only "shouted" is above 75
    dB(A)). Returns a tuple of 7 floats.

    Raises
    ------
    ValueError
        ``f"unknown vocal effort {effort!r}; use one of {', '.join(SPEECH_SPL_250_8K)}"``
        (also for :data:`CUSTOM_EFFORT`, which has no table).
    """
    if effort not in SPEECH_SPL_250_8K:
        raise ValueError(
            f"unknown vocal effort {effort!r}; use one of {', '.join(SPEECH_SPL_250_8K)}"
        )
    levels = SPEECH_SPL_250_8K[effort]
    reduction = speech_level_reduction(SPEECH_SPL_A[effort])
    full = (levels[0] - SPEECH_125_BELOW_250_DB, *levels)
    return tuple(float(x - reduction) for x in full)


def _check(band: int, frequency: float) -> None:
    """Validate band index and modulation frequency."""
    if not 0 <= band <= 6:
        raise ValueError("band must be 0...6")
    if frequency <= 0:
        raise ValueError("frequency must be > 0")


@dataclass(frozen=True)
class StiResult:
    """STI with its intermediate values.

    Parameters
    ----------
    sti
        Speech Transmission Index, 0...1 ([2] step 5).
    mti
        Per octave band ``((S/N)'_k + 15) / 30``, shape ``(7,)``; ``sti`` is their
        weighted sum with :data:`WEIGHTS` (a linear rewriting of [2] steps 4-5).
    apparent_snr
        Truncated apparent S/N in dB per band and modulation frequency, shape ``(7, 14)``
        ([2] steps 1-2).
    """

    sti: float
    mti: NDArray[np.float64]
    apparent_snr: NDArray[np.float64]


def mtf_noise(snr_db: ArrayLike) -> NDArray[np.float64]:
    """Modulation reduction by steady noise: ``m = (1 + 10**(-S/N / 10))**-1``.

    [2] Eq. (2); independent of the modulation frequency. ``S/N`` is the octave-band
    signal-to-noise ratio in dB at the receiving end; ``+inf`` gives 1.

    Returns
    -------
    numpy.ndarray
        ``m``, same shape as ``snr_db``.
    """
    snr_db_arr = np.asarray(snr_db, float)
    return 1.0 / (1.0 + 10.0 ** (-snr_db_arr / 10.0))


def mtf_reverberation(frequency: ArrayLike, t60: float) -> NDArray[np.float64]:
    """Modulation reduction by exponential reverberation, direct sound disregarded.

    ``m(F) = (1 + (2 pi F T / 13.8)**2)**-1/2`` ([2] Eq. (1)).

    Parameters
    ----------
    frequency
        Modulation frequencies F in Hz.
    t60
        Reverberation time T in s, >= 0 (0 gives m = 1).

    Raises
    ------
    ValueError
        ``"t60 must be >= 0"``.
    """
    if t60 < 0:
        raise ValueError("t60 must be >= 0")
    freq_arr = np.asarray(frequency, float)
    return (1.0 + (2.0 * np.pi * freq_arr * t60 / _DECAY) ** 2) ** -0.5


def apparent_snr(m: ArrayLike) -> NDArray[np.float64]:
    """Apparent S/N ``10 lg(m / (1 - m))`` in dB, truncated to +-:data:`SNR_LIMIT`.

    [2] Eq. (3) and step 2; ``m = 0`` gives -15 dB, ``m = 1`` gives +15 dB (no warning
    for the division).

    Raises
    ------
    ValueError
        ``"m must be in 0...1"`` if any value is outside [0, 1] or NaN.
    """
    m_arr = np.asarray(m, float)
    if np.any(np.isnan(m_arr)) or np.any(m_arr < 0) or np.any(m_arr > 1):
        raise ValueError("m must be in 0...1")
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = 10.0 * np.log10(m_arr / (1.0 - m_arr))
    return np.clip(snr, -SNR_LIMIT, SNR_LIMIT)


def sti_from_mtf(m: ArrayLike) -> StiResult:
    """STI from a 7 x 14 modulation transfer matrix ([2] sec. II B, steps 1-5).

    Steps: apparent S/N per entry (:func:`apparent_snr`), plain mean over the 14
    modulation frequencies per band, weighted mean over the bands with :data:`WEIGHTS`,
    ``STI = ((S/N)' + 15) / 30``.

    Parameters
    ----------
    m
        Shape ``(7, 14)``: rows :data:`OCTAVE_BANDS`, columns
        :data:`MODULATION_FREQUENCIES`.

    Raises
    ------
    ValueError
        ``"m must have shape (7, 14)"``; from :func:`apparent_snr`.
    """
    m_arr = np.asarray(m, float)
    if m_arr.shape != (7, 14):
        raise ValueError("m must have shape (7, 14)")
    snr = apparent_snr(m_arr)
    band = snr.mean(axis=1)
    mti = (band + SNR_LIMIT) / (2 * SNR_LIMIT)
    sti = float(np.dot(WEIGHTS, mti))
    return StiResult(sti=sti, mti=mti, apparent_snr=snr)


def modulated_noise(
    band: int, frequency: float, duration: float, sample_rate: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Test signal of route (b): octave-band noise with 100 % sinusoidal intensity modulation.

    ``x(t) = c(t) sqrt(1 + cos(2 pi F t))``, so the intensity envelope is
    ``I(t) = I_k (1 + cos 2 pi F t)`` with modulation index 1 ([1] Eq. (1); [2] sec. I A).
    The carrier ``c`` is white noise ``rng.standard_normal(n)`` filtered with
    ``resonus_core.bands.band_filter(BAND_CENTERS[band], 1, sample_rate)``
    (``scipy.signal.sosfilt``) and scaled to RMS 1. ``n = round(duration * sample_rate)``.

    Parameters
    ----------
    band
        Index into :data:`OCTAVE_BANDS`, 0...6.
    frequency
        Modulation frequency F in Hz, > 0.
    duration
        Length in s, > 0.
    sample_rate
        In Hz; the band's upper edge must be below Nyquist (``band_filter`` raises).
    rng
        Source of the noise.

    Raises
    ------
    ValueError
        ``"band must be 0...6"``, ``"frequency must be > 0"``, ``"duration must be > 0"``.
    """
    _check(band, frequency)
    if duration <= 0:
        raise ValueError("duration must be > 0")
    n = round(duration * sample_rate)
    c = ss.sosfilt(
        band_filter(BAND_CENTERS[band], _OCTAVE, sample_rate),
        rng.standard_normal(n),
    )
    c /= np.sqrt(np.mean(c**2))
    t = np.arange(n) / sample_rate
    return c * np.sqrt(1 + np.cos(2 * np.pi * frequency * t))


def modulation_index(
    y: ArrayLike, sample_rate: float, band: int, frequency: float, start_s: float = 0.0
) -> float:
    """Modulation index at ``frequency`` of the octave-band intensity envelope of ``y``.

    ``y`` is filtered with the band filter of :func:`modulated_noise` and squared
    (intensity, [2] sec. II A). From sample ``i0 = round(start_s * sample_rate)`` the
    longest whole number of modulation periods is used: ``n = floor(N F / fs) fs / F``
    samples (``int`` of it), ``N`` the samples left after ``i0``. The index is the
    correlation with the sinusoidal envelope ([1] sec. I A), phase disregarded ([2] sec.
    I B): ``m = 2 |sum I(t) exp(-j 2 pi F t)| / sum I(t)``, ``t = (i0 + i) / fs``, capped
    at 1 (estimation noise can exceed it; design choice).

    Raises
    ------
    ValueError
        ``"band must be 0...6"``, ``"frequency must be > 0"``; ``"signal shorter than
        one modulation period"`` if ``n < 1`` or ``n`` is below ``fs / F``.
    """
    _check(band, frequency)
    z = ss.sosfilt(
        band_filter(BAND_CENTERS[band], _OCTAVE, sample_rate),
        np.asarray(y, float),
    )
    i0 = round(start_s * sample_rate)
    intensity = z[i0:] ** 2
    n = int(np.floor(intensity.size * frequency / sample_rate) * sample_rate / frequency)
    if n < 1 or n < sample_rate / frequency - 1:
        raise ValueError("signal shorter than one modulation period")
    t = (i0 + np.arange(n)) / sample_rate
    m = 2 * abs(np.sum(intensity[:n] * np.exp(-2j * np.pi * frequency * t))) / np.sum(intensity[:n])
    return min(1.0, float(m))


def measure_mtf(
    channel: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    sample_rate: float,
    duration: float,
    rng: np.random.Generator,
    start_s: float = 0.0,
) -> NDArray[np.float64]:
    """Modulation transfer matrix of ``channel`` by route (b) (docs/PLAN_STI.md).

    For each band ``k`` (rows, :data:`OCTAVE_BANDS`) and modulation frequency ``F``
    (columns, :data:`MODULATION_FREQUENCIES`), in that loop order:
    ``m[k, j] = modulation_index(channel(modulated_noise(k, F, duration, sample_rate,
    rng)), sample_rate, k, F, start_s)``. ``channel`` maps the test signal to the received
    signal (e.g. convolution with an impulse response plus noise).

    Returns
    -------
    numpy.ndarray
        Shape ``(7, 14)``, for :func:`sti_from_mtf`.
    """
    m = np.empty((7, 14), dtype=float)
    for k in range(7):
        for j, F in enumerate(MODULATION_FREQUENCIES):
            test_sig = modulated_noise(k, F, duration, sample_rate, rng)
            received = channel(test_sig)
            m[k, j] = modulation_index(received, sample_rate, k, F, start_s)
    return m


def band_level(x: ArrayLike, sample_rate: float, band: int, start_s: float = 0.0) -> float:
    """Equivalent octave-band sound pressure level of ``x`` (in Pa), in dB re :data:`P0`.

    ``x`` is filtered with the band filter of :func:`modulated_noise`; from sample
    ``i0 = round(start_s * sample_rate)`` on, ``L = 10 lg(mean(z**2) / P0**2)`` ([3]).
    A zero signal gives ``-inf`` (no warning).

    Raises
    ------
    ValueError
        ``"band must be 0...6"``; ``"no samples after start_s"``.
    """
    _check(band, 1.0)
    z = ss.sosfilt(
        band_filter(BAND_CENTERS[band], _OCTAVE, sample_rate),
        np.asarray(x, float),
    )
    i0 = round(start_s * sample_rate)
    z = z[i0:]
    if z.size == 0:
        raise ValueError("no samples after start_s")
    with np.errstate(divide="ignore"):
        return float(10 * np.log10(np.mean(z**2) / P0**2))


def compound_signal(
    test: ArrayLike,
    sample_rate: float,
    band: int,
    speech_level_db: float,
    ir_source: ArrayLike,
    ir_room: ArrayLike,
    noise: ArrayLike,
) -> NDArray[np.float64]:
    """Received signal of work-package item 5 (docs/PLAN_STI.md section 3), in Pa.

    The test signal (drive units) is scaled so that the source produces
    ``speech_level_db`` in octave band ``band`` at 1 m, convolved with the in-situ impulse
    response and added to the captured background noise:

    - ``s = scipy.signal.fftconvolve(test, ir_source)[:n]``, ``n = len(test)``;
    - ``g = 10**((speech_level_db - band_level(s, sample_rate, band)) / 20)``;
    - result ``g * fftconvolve(test, ir_room)[:n] + noise[:n]``.

    The noise is added once, at its calibrated level; the impulse responses should be
    measured at a high S/N so that their own noise floor does not count twice
    (docs/PLAN_STI.md section 4).

    Parameters
    ----------
    test
        Test signal in drive units, e.g. from :func:`modulated_noise`; not all zero in
        the band.
    sample_rate
        In Hz, common to all signals.
    band
        Index into :data:`OCTAVE_BANDS`, 0...6.
    speech_level_db
        Octave-band speech level at 1 m in dB re :data:`P0` (a user setting, Q-STI3).
    ir_source
        Free-field impulse response of the source at 1 m, Pa per drive unit (item 3).
    ir_room
        Calibrated in-situ impulse response, Pa per drive unit (item 4).
    noise
        Calibrated background noise in Pa (item 2), at least ``n`` samples.

    Returns
    -------
    numpy.ndarray
        Shape ``(n,)``.

    Raises
    ------
    ValueError
        ``"noise shorter than the test signal"``; from :func:`band_level`.
    """
    test = np.asarray(test, float)
    n = test.size
    noise_arr = np.asarray(noise, float)
    if noise_arr.size < n:
        raise ValueError("noise shorter than the test signal")
    s = ss.fftconvolve(test, np.asarray(ir_source, float))[:n]
    g = 10 ** ((speech_level_db - band_level(s, sample_rate, band)) / 20)
    return g * ss.fftconvolve(test, np.asarray(ir_room, float))[:n] + noise_arr[:n]


def measure_mtf_compound(
    sample_rate: float,
    duration: float,
    rng: np.random.Generator,
    speech_levels_db: ArrayLike,
    ir_source: ArrayLike,
    ir_room: ArrayLike,
    noise: ArrayLike,
    start_s: float = 0.0,
) -> NDArray[np.float64]:
    """Modulation transfer matrix from the compound signal (work-package item 6, route b).

    As :func:`measure_mtf`, same loop order, with
    ``compound_signal(modulated_noise(k, F, duration, sample_rate, rng), sample_rate, k,
    speech_levels_db[k], ir_source, ir_room, noise)`` as the received signal.

    Parameters
    ----------
    speech_levels_db
        Shape ``(7,)``: octave-band speech levels at 1 m, dB re :data:`P0`.
    start_s
        Passed to :func:`modulation_index`; skip at least the build-up of the
        reverberation.

    Returns
    -------
    numpy.ndarray
        Shape ``(7, 14)``, for :func:`sti_from_mtf`.

    Raises
    ------
    ValueError
        ``"speech_levels_db must have 7 values"``; from :func:`compound_signal`.
    """
    levels = np.asarray(speech_levels_db, float)
    if levels.shape != (7,):
        raise ValueError("speech_levels_db must have 7 values")
    m = np.empty((7, 14), dtype=float)
    for k in range(7):
        for j, F in enumerate(MODULATION_FREQUENCIES):
            test_sig = modulated_noise(k, F, duration, sample_rate, rng)
            received = compound_signal(
                test_sig, sample_rate, k, levels[k], ir_source, ir_room, noise
            )
            m[k, j] = modulation_index(received, sample_rate, k, F, start_s)
    return m
