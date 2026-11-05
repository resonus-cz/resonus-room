"""Synthetic room impulse responses with known decay times (tests, simulation, examples).

Model: white Gaussian noise, split into the bands of a :class:`BandSet` by an ideal
(rectangular) FFT mask between the band edges, each band component multiplied by the
amplitude envelope ``10**(-3 t / T_k)`` (energy falls by 60 dB in ``T_k``), summed,
preceded by ``delay`` seconds of silence, plus optional stationary white background
noise. Frequencies outside the band set are absent. This is a test signal, not a room
model.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from resonus_core.bands import BandSet


def exponential_decay_ir(
    sample_rate: float,
    duration: float,
    decay_times: ArrayLike,
    bands: BandSet,
    delay: float = 0.0,
    noise_rms: float = 0.0,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Impulse response whose band ``k`` decays with reverberation time ``decay_times[k]``.

    1. ``n = round(duration * fs)``, ``nd = round(delay * fs)``, ``m = n - nd``;
       ``rng = np.random.default_rng(seed)``; ``w = rng.standard_normal(m)``;
       ``W = np.fft.rfft(w)``, ``f = np.fft.rfftfreq(m, 1/fs)``, ``t = arange(m)/fs``.
    2. For each band ``k``: ``mask = (f >= bands.lower[k]) & (f < bands.upper[k])``;
       ``out += np.fft.irfft(W * mask, m) * 10**(-3 * t / decay_times[k])``.
    3. ``ir = concatenate([zeros(nd), out])``; if ``noise_rms > 0``:
       ``ir += noise_rms * rng.standard_normal(n)`` (drawn after ``w``).

    Parameters
    ----------
    sample_rate
        Sample rate in Hz, > 0; the highest upper band edge must be below ``fs/2``.
    duration
        Total length in s (including ``delay``), > ``delay``.
    decay_times
        Reverberation time per band in s, ``len(bands)`` values, > 0.
    bands
        Band set.
    delay
        Leading silence in s, >= 0 (the onset is at sample ``round(delay * fs)``).
    noise_rms
        RMS of stationary white background noise, >= 0.
    seed
        Seed of the random generator.

    Returns
    -------
    numpy.ndarray
        Impulse response, ``n`` samples.

    Raises
    ------
    ValueError
        On invalid parameters.
    """
    fs = float(sample_rate)
    t60 = np.asarray(decay_times, dtype=float)
    if fs <= 0 or duration <= delay or delay < 0 or noise_rms < 0:
        raise ValueError("invalid parameters")
    if t60.shape != (len(bands),) or np.any(t60 <= 0):
        raise ValueError("invalid decay_times")
    if bands.upper[-1] >= fs / 2:
        raise ValueError("band upper edge exceeds Nyquist")
    n = round(duration * fs)
    nd = round(delay * fs)
    m = n - nd
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(m)
    spec = np.fft.rfft(w)
    f = np.fft.rfftfreq(m, 1 / fs)
    t = np.arange(m) / fs
    out = np.zeros(m)
    lower, upper = bands.lower, bands.upper
    for k in range(len(bands)):
        mask = (f >= lower[k]) & (f < upper[k])
        out += np.fft.irfft(spec * mask, m) * 10 ** (-3 * t / t60[k])
    ir = np.concatenate([np.zeros(nd), out])
    if noise_rms > 0:
        ir = ir + noise_rms * rng.standard_normal(n)
    return ir
