"""Decay curves by backward integration of squared impulse responses (Schroeder).

ISO 3382-1:2009 5.3.3 (and ISO 3382-2:2008 5.3.3, ISO 354:2003 7.3.4): the decay curve is
the backward integral of the squared (band-filtered) impulse response, Eq. (1)/(2). To
limit the influence of background noise, integration starts at the truncation point
``t1`` (intersection of the background level with a line fitted to the decay) and a
correction ``C`` for the energy after ``t1`` is added, Eq. (3); ``C`` assumes an
exponential decay with the rate of the squared IR between ``t0`` and ``t1``, ``t0`` being
10 dB above the level at ``t1``.

PROVISIONAL (docs/DECISIONS.md D-6): the standard does not specify how
the background level, the "representative part" of the decay and the levels at ``t0``,
``t1`` are determined. This module uses a moving-average energy envelope, the mean energy
of the last part of the response as background, and single-pass least-squares lines
(no Lundeby iteration). All of these are parameters.

Interpretation of ``C``: by definition of ``t1`` the decay level at ``t1`` equals the
background level ``B``, and ``t0`` lies 10 dB higher on the decay. The rate is taken from
the least-squares line of the representative part (the same line that defines ``t1``),
because a regression on the squared IR between ``t0`` and ``t1`` is biased by the
background noise near ``t1`` (up to +3 dB at ``t1``; about 2x too large ``C`` in tests).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

# PROVISIONAL (D-6): length of the moving average forming the energy envelope, in s.
DEFAULT_ENVELOPE_WINDOW = 0.010
# PROVISIONAL (D-6): fraction of the response at its end used for the background level
# (as resonus_core.window.onset).
DEFAULT_TAIL_FRACTION = 0.1
# PROVISIONAL (D-6): the line through the "representative part" of the decay starts this
# many dB below the envelope maximum ...
DEFAULT_FIT_TOP_DB = -5.0
# ... and ends this many dB above the background level.
DEFAULT_FIT_MARGIN_DB = 10.0
METHODS = ("none", "truncate", "correct")


@dataclass(frozen=True)
class DecayCurve:
    """Normalized decay curve of one (band-filtered) impulse response.

    Parameters
    ----------
    time
        Time in s from the first sample of the analysed signal, length ``truncation``.
    level
        Decay level ``10*lg(E(t)/E(0))`` in dB (0 dB at ``t = 0``).
    truncation
        Truncation index ``t1`` (samples); the integration covers samples ``0...t1-1``.
    correction
        Correction ``C`` added to the backward integral, in (unit of the IR)^2 * s.
    background_db
        Background level of the energy envelope, ``10*lg`` of the mean squared value, dB re
        (unit of the IR)^2.
    peak_to_noise_db
        Envelope maximum minus background level, in dB.
    """

    time: NDArray[np.float64]
    level: NDArray[np.float64]
    truncation: int
    correction: float
    background_db: float
    peak_to_noise_db: float


@dataclass(frozen=True)
class Truncation:
    """Result of :func:`truncation_point`.

    Parameters
    ----------
    index
        Truncation index ``t1`` in samples, ``1 <= index <= len(p)``.
    background_db
        Background level of the energy envelope in dB re (unit of the IR)^2.
    slope
        Slope ``b`` of the line through the representative part in dB/s (NaN if no line
        was fitted).
    intercept
        Intercept ``a`` of that line in dB (NaN if no line was fitted).
    """

    index: int
    background_db: float
    slope: float
    intercept: float


def _check_signal(p: ArrayLike, sample_rate: float) -> NDArray[np.float64]:
    """Return ``p`` as a 1-D float array; ValueError if invalid."""
    x = np.asarray(p, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.any(x):
        raise ValueError("Signal must be a non‑empty 1‑D array with non‑zero values")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    return x


def _line_fit(t: NDArray[np.float64], level: NDArray[np.float64]) -> tuple[float, float]:
    """Least-squares line ``level = a + b t`` (ISO 3382-2:2008 Annex C); returns ``(a, b)``."""
    b = float(np.sum((t - t.mean()) * (level - level.mean())) / np.sum((t - t.mean()) ** 2))
    a = float(level.mean() - b * t.mean())
    return a, b


def energy_envelope(
    p: ArrayLike, sample_rate: float, window: float = DEFAULT_ENVELOPE_WINDOW
) -> NDArray[np.float64]:
    """Centred moving average of ``p**2`` over ``2h + 1`` samples.

    ``h = int(window * sample_rate) // 2``; at the ends only the available samples are
    averaged: ``env[i] = mean(p[max(0, i-h) : min(n, i+h+1)]**2)`` (no attenuation at the
    ends). Computed as differences of the *reverse* cumulative sum
    ``r[i] = sum(p[i:]**2)``, which are small where the decay is small; a forward
    cumulative sum loses all precision about 130 dB below the peak.

    Parameters
    ----------
    p
        Impulse response, 1-D.
    sample_rate
        Sample rate in Hz, > 0.
    window
        Averaging length in s, > 0.

    Returns
    -------
    numpy.ndarray
        Energy envelope, same length as ``p``, in (unit of ``p``)^2.

    Raises
    ------
    ValueError
        If ``p`` is not 1-D, is empty or all zero, or ``sample_rate``/``window`` <= 0.
    """
    x = _check_signal(p, sample_rate)
    if window <= 0:
        raise ValueError("Window must be positive")
    h = int(window * sample_rate) // 2
    r = np.concatenate([np.cumsum(x[::-1] ** 2)[::-1], [0.0]])
    i = np.arange(x.size)
    lo = np.maximum(i - h, 0)
    hi = np.minimum(i + h + 1, x.size)
    return np.maximum(r[lo] - r[hi], 0.0) / (hi - lo)


def background_level(envelope: ArrayLike, tail_fraction: float = DEFAULT_TAIL_FRACTION) -> float:
    """Background level ``10*lg(mean(envelope[-k:]))`` with ``k = max(1, round(f * n))``.

    Digital silence at the end gives ``-inf`` (no warning).

    Raises
    ------
    ValueError
        If ``tail_fraction`` is not in (0, 1] or the envelope is empty.
    """
    env = np.asarray(envelope, dtype=float)
    if env.size == 0:
        raise ValueError("Envelope must not be empty")
    if not (0 < tail_fraction <= 1):
        raise ValueError("tail_fraction must be in (0, 1]")
    k = max(1, round(tail_fraction * env.size))
    with np.errstate(divide="ignore"):
        return float(10 * np.log10(np.mean(env[-k:])))


def truncation_point(
    p: ArrayLike,
    sample_rate: float,
    window: float = DEFAULT_ENVELOPE_WINDOW,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
    top_db: float = DEFAULT_FIT_TOP_DB,
    margin_db: float = DEFAULT_FIT_MARGIN_DB,
) -> Truncation:
    """Find the truncation point ``t1`` (ISO 3382-1:2009 5.3.3).

    1. ``env = energy_envelope(p, sample_rate, window)``, ``L = 10*lg(env)``;
       ``B = background_level(env, tail_fraction)``; ``m = argmax(env)``.
    2. ``i_top``: first index ``>= m`` with ``L <= L[m] + top_db``; ``i_bot``: first index
       ``>= i_top`` with ``L <= B + margin_db``. If either does not exist or
       ``i_bot - i_top < 2``, there is no usable decay above the background: return
       ``Truncation(len(p), B, nan, nan)``. The same applies if ``B`` is not finite
       (digital silence at the end: no background noise to truncate at).
    3. Least-squares line ``L = a + b*t`` over samples ``i_top...i_bot`` (``t = i/fs``).
       If ``b >= 0``: return ``Truncation(len(p), B, b, a)``. Else ``t1 = (B - a)/b`` and
       the index is ``min(len(p), max(1, round(t1 * fs)))``.

    Samples with level -inf (exact zeros of quantized responses) are excluded from the
    line fit (D-13).

    Parameters
    ----------
    p
        (Band-filtered) impulse response, 1-D, starting at the onset.
    sample_rate
        Sample rate in Hz.
    window, tail_fraction
        See :func:`energy_envelope`, :func:`background_level`.
    top_db
        Start of the fitted part re the envelope maximum, in dB, < 0.
    margin_db
        End of the fitted part above the background, in dB, >= 0.

    Returns
    -------
    Truncation

    Raises
    ------
    ValueError
        On an invalid signal or parameters (``top_db >= 0``, ``margin_db < 0``).

    References
    ----------
    ISO 3382-1:2009, 5.3.3; ISO 354:2003, 7.3.4.
    """
    # Validate parameters
    if top_db >= 0:
        raise ValueError("top_db must be negative")
    if margin_db < 0:
        raise ValueError("margin_db must be non‑negative")

    # Prepare signal and envelope
    x = _check_signal(p, sample_rate)
    env = energy_envelope(x, sample_rate, window)
    with np.errstate(divide="ignore"):
        level = 10 * np.log10(env)

    bg = background_level(env, tail_fraction)
    none = Truncation(x.size, bg, float("nan"), float("nan"))
    if not np.isfinite(bg):
        return none

    m = int(np.argmax(env))
    top = np.flatnonzero(level[m:] <= level[m] + top_db)
    if top.size == 0:
        return none
    i_top = m + int(top[0])

    bot = np.flatnonzero(level[i_top:] <= bg + margin_db)
    if bot.size == 0:
        return none
    i_bot = i_top + int(bot[0])

    if i_bot - i_top < 2:
        return none

    idx = np.arange(i_top, i_bot + 1)
    idx = idx[np.isfinite(level[idx])]
    if idx.size < 2:
        return none
    a, b = _line_fit(idx / sample_rate, level[idx])

    if b >= 0:
        return Truncation(x.size, bg, b, a)

    t1 = (bg - a) / b
    return Truncation(min(x.size, max(1, round(t1 * sample_rate))), bg, b, a)


def tail_correction(background_db: float, slope: float) -> float:
    """Correction ``C`` for the energy after the truncation point (ISO 3382-1:2009 5.3.3).

    Exponential continuation of the decay from the level ``background_db`` at ``t1`` with
    the decay rate ``slope`` (dB/s): ``C = 10**(background_db/10) * 10 / (-slope * ln 10)``,
    the integral of ``10**((background_db + slope*(t - t1))/10)`` from ``t1`` to infinity.
    Returns 0.0 if ``slope`` is NaN or ``>= 0`` (no decay to extrapolate).

    Parameters
    ----------
    background_db
        Decay level at ``t1`` (= background level), dB re (unit of the IR)^2.
    slope
        Decay rate in dB/s (negative).

    Returns
    -------
    float
        ``C`` in (unit of the IR)^2 * s, >= 0.

    References
    ----------
    ISO 3382-1:2009, 5.3.3, Eq. (3); ISO 354:2003, 7.3.4, Eq. (4).
    """
    if not np.isfinite(slope) or slope >= 0:
        return 0.0
    return float(10 ** (background_db / 10) * 10 / (-slope * np.log(10)))


def schroeder(
    p: ArrayLike, sample_rate: float, stop: int | None = None, correction: float = 0.0
) -> NDArray[np.float64]:
    """Backward-integrated squared impulse response ``E``, ISO 3382-1:2009 Eq. (3).

    ``E[i] = sum(p[i:stop]**2) / sample_rate + correction`` for ``i = 0...stop-1``
    (``stop = len(p)`` if None), computed with a reversed cumulative sum.

    Parameters
    ----------
    p
        (Band-filtered) impulse response, 1-D.
    sample_rate
        Sample rate in Hz.
    stop
        Truncation index ``t1``, ``1 <= stop <= len(p)``.
    correction
        ``C`` in (unit of ``p``)^2 * s, >= 0.

    Returns
    -------
    numpy.ndarray
        ``E``, length ``stop``, in (unit of ``p``)^2 * s.

    Raises
    ------
    ValueError
        On an invalid signal, ``stop`` outside ``1...len(p)`` or ``correction < 0``.

    References
    ----------
    ISO 3382-1:2009, 5.3.3, Eqs. (1)-(3); ISO 354:2003, 7.3.4, Eqs. (3)-(4).
    """
    x = _check_signal(p, sample_rate)
    n = x.size if stop is None else stop
    if not (1 <= n <= x.size):
        raise ValueError("stop must be in 1...len(p)")
    if correction < 0:
        raise ValueError("correction must be non‑negative")
    rev_cumsum = np.cumsum(x[:n][::-1] ** 2)[::-1]
    return rev_cumsum / sample_rate + correction


def decay_curve(
    p: ArrayLike,
    sample_rate: float,
    method: str = "correct",
    window: float = DEFAULT_ENVELOPE_WINDOW,
    tail_fraction: float = DEFAULT_TAIL_FRACTION,
    top_db: float = DEFAULT_FIT_TOP_DB,
    margin_db: float = DEFAULT_FIT_MARGIN_DB,
) -> DecayCurve:
    """Compute the normalized decay curve with optional truncation and correction.

    1. ``env = energy_envelope(p, sample_rate, window)``;
       ``peak_to_noise_db = 10*lg(max(env)) - background_db``.
    2. ``method``: ``"none"``: ``stop = len(p)``, ``C = 0``, ``background_db =
       background_level(env, tail_fraction)``; ``"truncate"``: ``tr = truncation_point(...)``,
       ``stop = tr.index``, ``background_db = tr.background_db``, ``C = 0``; ``"correct"``:
       as ``"truncate"`` and ``C = tail_correction(tr.background_db, tr.slope)``.
    3. ``E = schroeder(p, sample_rate, stop, C)``; ``level = 10*lg(E / E[0])``;
       ``time = arange(stop) / sample_rate``.

    Raises
    ------
    ValueError
        On an unknown method or invalid signal/parameters.

    References
    ----------
    ISO 3382-1:2009, 5.3.3; ISO 3382-2:2008, 5.3.3; ISO 354:2003, 7.3.4.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}")

    x = _check_signal(p, sample_rate)
    env = energy_envelope(x, sample_rate, window)
    c = 0.0

    if method == "none":
        stop, bg = x.size, background_level(env, tail_fraction)
    else:
        tr = truncation_point(x, sample_rate, window, tail_fraction, top_db, margin_db)
        stop, bg = tr.index, tr.background_db
        if method == "correct":
            c = tail_correction(tr.background_db, tr.slope)

    e = schroeder(x, sample_rate, stop, c)
    with np.errstate(divide="ignore"):
        level = 10 * np.log10(e / e[0])

    return DecayCurve(
        time=np.arange(stop) / sample_rate,
        level=level,
        truncation=stop,
        correction=c,
        background_db=bg,
        peak_to_noise_db=float(10 * np.log10(np.max(env)) - bg),
    )
