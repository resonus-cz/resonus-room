"""Measurement uncertainty of reverberation times and averaging over positions.

References
----------
ISO 3382-2:2008, 5.2.2, 7.1-7.2 Eqs. (2)-(3), clause 8, Annex A Eq. (A.4) and Table A.1.
ISO 354:2003, 7.2.2 Eq. (2), 8.1.1, 8.2.2 Eq. (10).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ISO 3382-2:2008 7.1: filter bandwidth B = 0.71 f_c (octave), 0.23 f_c (one-third octave).
BANDWIDTH_FACTOR = {1: 0.71, 3: 0.23}
# ISO 3382-2:2008 Table A.1, gamma = 5 (as assumed by Eqs. (2) and (3)): evaluation range
# D in dB -> (G as a fraction, H). D = 20 and 30 give Eqs. (2) and (3).
UNCERTAINTY_CONSTANTS = {10: (1.75, 3.32), 20: (0.88, 1.90), 30: (0.55, 1.52)}
# ISO 3382-2:2008 7.2 and A.3: the integrated impulse response method corresponds to
# n = 10 decays per position.
N_DECAYS_IR_METHOD = 10


@dataclass(frozen=True)
class SpatialAverage:
    """Arithmetic mean over positions (ISO 3382-2:2008 clause 8 a, ISO 354:2003 8.1.1).

    Parameters
    ----------
    mean
        Mean of the finite values per band.
    std
        Sample standard deviation (``ddof=1``) of the finite values; NaN if fewer than 2.
    count
        Number of finite values per band.
    """

    mean: NDArray[np.float64]
    std: NDArray[np.float64]
    count: NDArray[np.int64]


def bandwidth(center: ArrayLike, fraction: int) -> NDArray[np.float64]:
    """Filter bandwidth ``B = 0.71 f_c`` (octave) or ``0.23 f_c`` (one-third octave), in Hz.

    Raises
    ------
    ValueError
        If ``fraction`` is not 1 or 3.

    References
    ----------
    ISO 3382-2:2008, 7.1.
    """
    if fraction not in BANDWIDTH_FACTOR:
        raise ValueError(f"Unsupported fraction {fraction!r}")
    return BANDWIDTH_FACTOR[fraction] * np.asarray(center, dtype=float)


def reverberation_std(
    decay_time: ArrayLike,
    bandwidth: ArrayLike,
    n_positions: int,
    range_db: int = 20,
    n_decays: int = N_DECAYS_IR_METHOD,
) -> NDArray[np.float64]:
    """Return the standard deviation of a mean reverberation time, ISO 3382-2 Eq. (A.4).

    ``sigma(T) = G * T * sqrt((1 + H/n) / (N * B * T))`` with ``(G, H) =
    UNCERTAINTY_CONSTANTS[range_db]`` (Table A.1, gamma = 5); ``range_db`` 20 and 30 are
    Eqs. (2) and (3). For the integrated impulse response method use ``n = 10`` (A.3).

    D-11 (confirmed by the user 2026-11-03): ``range_db = 10`` (Table A.1, D = 10 dB)
    may be used for EDT; the standard does not state that Table A.1 applies to EDT
    (0 to -10 dB).

    Parameters
    ----------
    decay_time
        Reverberation time ``T`` in s (array), NaN propagates.
    bandwidth
        Filter bandwidth ``B`` in Hz (see :func:`bandwidth`).
    n_positions
        Number ``N`` of independent source-receiver combinations, >= 1.
    range_db
        Evaluation range ``D``: 10, 20 or 30 dB.
    n_decays
        Number ``n`` of decays per position, >= 1.

    Returns
    -------
    numpy.ndarray
        ``sigma(T)`` in s.

    Raises
    ------
    ValueError
        On an unknown ``range_db`` or ``n_positions``/``n_decays`` < 1.
    """
    if range_db not in UNCERTAINTY_CONSTANTS:
        raise ValueError(f"Unsupported range_db {range_db!r}")
    if n_positions < 1:
        raise ValueError("n_positions must be >= 1")
    if n_decays < 1:
        raise ValueError("n_decays must be >= 1")
    g, h = UNCERTAINTY_CONSTANTS[range_db]
    t = np.asarray(decay_time, dtype=float)
    b = np.asarray(bandwidth, dtype=float)
    return g * t * np.sqrt((1 + h / n_decays) / (n_positions * b * t))


def iso354_repeatability(
    decay_time: ArrayLike, frequency: ArrayLike, n_decays: int
) -> NDArray[np.float64]:
    """Return the standard deviation of T20 in a reverberation room, ISO 354 Eq. (10).

    ``epsilon20(T) = T * sqrt((2.42 + 3.59/N) / (f * T))``, ``f`` the one-third-octave
    centre frequency, ``N`` the number of decay curves evaluated.

    Raises
    ------
    ValueError
        If ``n_decays < 1``.
    """
    if n_decays < 1:
        raise ValueError("n_decays must be >= 1")
    t = np.asarray(decay_time, dtype=float)
    f = np.asarray(frequency, dtype=float)
    return t * np.sqrt((2.42 + 3.59 / n_decays) / (f * t))


def spatial_average(values: ArrayLike) -> SpatialAverage:
    """Arithmetic mean and standard deviation over positions (axis 0), ignoring NaN.

    ``values`` has shape ``(n_positions, n_bands)`` (1-D: one band). Per band, the finite
    values are averaged; ``std`` uses ``ddof=1`` and is NaN for fewer than 2 values; the
    mean is NaN for 0 values.

    Raises
    ------
    ValueError
        If ``values`` is empty or has more than 2 dimensions.

    References
    ----------
    ISO 3382-2:2008, clause 8 a); ISO 3382-1:2009, clause 8; ISO 354:2003, 8.1.1.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0 or v.ndim > 2:
        raise ValueError("values must be 1- or 2-D and non-empty")
    v2 = v[:, None] if v.ndim == 1 else v
    ok = np.isfinite(v2)
    count = ok.sum(axis=0)
    total = np.where(ok, v2, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    dev = np.where(ok, v2 - mean, 0.0)
    std = np.where(
        count > 1,
        np.sqrt((dev**2).sum(axis=0) / np.maximum(count - 1, 1)),
        np.nan,
    )
    if v.ndim == 1:
        return SpatialAverage(mean[:1], std[:1], count[:1].astype(np.int64))
    return SpatialAverage(mean, std, count.astype(np.int64))


def ensemble_average(levels: ArrayLike) -> NDArray[np.float64]:
    """Energetic average of decay curves: ``10 lg(mean(10**(L/10)))`` over axis 1.

    ``levels`` has shape ``(n_samples, n_decays)`` (synchronized starts).

    Raises
    ------
    ValueError
        If ``levels`` is not 2-D or has no decays.

    References
    ----------
    ISO 354:2003, 7.2.2 Eq. (2); ISO 3382-2:2008, 5.2.2 and clause 8 b).
    """
    lv = np.asarray(levels, dtype=float)
    if lv.ndim != 2 or lv.shape[1] == 0:
        raise ValueError("levels must be 2-D with at least one decay")
    return 10 * np.log10(np.mean(10 ** (lv / 10), axis=1))
