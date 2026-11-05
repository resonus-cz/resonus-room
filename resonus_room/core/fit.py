"""Decay-time evaluation: least-squares fits of decay curves and their quality indicators.

References
----------
ISO 3382-1:2009, 3.5 (T20, T30), 5.3.3, 6, A.2.2 (EDT).
ISO 3382-2:2008, 6, 7.3, Annex B (non-linearity xi, curvature C), Annex C (least squares).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np  # noqa: F401
from numpy.typing import ArrayLike

# Evaluation ranges (start, stop) in dB re the initial level of the decay curve.
EVALUATION_RANGES: dict[str, tuple[float, float]] = {
    "EDT": (0.0, -10.0),  # ISO 3382-1:2009 A.2.2 (initial 10 dB, 0 dB to -10 dB)
    "T20": (-5.0, -25.0),  # ISO 3382-2:2008 3.5 NOTE 2 and clause 6
    "T30": (-5.0, -35.0),  # ISO 3382-2:2008 3.5 NOTE 2 and clause 6
}
# ISO 3382-2:2008 B.2: xi above 10 permille indicates a decay far from a straight line.
XI_LIMIT = 10.0
# ISO 3382-2:2008 B.4: C above 10 % is suspicious; negative values should not occur.
CURVATURE_LIMIT = 10.0
# ISO 3382-1:2009 5.3.3: without correction (C = 0), a background at least the evaluation
# range plus 15 dB below the maximum keeps the underestimation below 5 %; this gives the
# 35 dB (T20) and 45 dB (T30) of ISO 3382-2:2008 5.3.2 / ISO 3382-1:2009 5.3.2.
DYNAMIC_RANGE_MARGIN_DB = 15.0
# ISO 3382-2:2008 7.3 Eq. (4): BT > 16 for forward filtering; NOTE: BT > 4 with the
# time-reversal technique.
MIN_BT_FORWARD = 16.0
MIN_BT_REVERSED = 4.0


@dataclass(frozen=True)
class DecayFit:
    """Least-squares line through one evaluation range of a decay curve.

    Parameters
    ----------
    name
        Label, e.g. ``"T20"``.
    start_db, stop_db
        Evaluation range in dB re the initial level (``start_db > stop_db``).
    t_start, t_stop
        Times in s at which the curve first reaches ``start_db`` and ``stop_db`` (NaN if
        not reached).
    intercept
        ``a`` in dB, ISO 3382-2:2008 Eq. (C.2).
    slope
        ``b`` in dB/s, Eq. (C.3).
    decay_time
        ``-60 / b`` in s, Eq. (C.6) (NaN if the range is not covered or ``b >= 0``).
    r2
        Squared correlation coefficient, Eq. (B.2).
    xi
        Non-linearity ``1000 * (1 - r2)`` in permille, Eq. (B.3).
    n_points
        Number of samples in the fit (0 if not fitted).
    """

    name: str
    start_db: float
    stop_db: float
    t_start: float
    t_stop: float
    intercept: float
    slope: float
    decay_time: float
    r2: float
    xi: float
    n_points: int


def fit_decay(
    time: ArrayLike, level: ArrayLike, start_db: float, stop_db: float, name: str = ""
) -> DecayFit:
    """Fit a straight line to the part of a decay curve between ``start_db`` and ``stop_db``.

    1. ``i_start``: first index with ``level <= start_db``; ``i_stop``: first index with
       ``level <= stop_db`` (the times at which the curve *first reaches* these levels,
       ISO 3382-2:2008 3.5 NOTE 2). If either does not exist or ``i_stop - i_start < 2``,
       return a fit with NaN values (``t_start``/``t_stop`` set where found) and
       ``n_points = 0``.
    2. Samples ``i_start...i_stop`` (inclusive), ``m`` of them: least squares per Annex C:
       ``b = (sum(t*L) - m*mean(t)*mean(L)) / (sum(t**2) - m*mean(t)**2)``,
       ``a = mean(L) - b*mean(t)``; ``decay_time = -60/b`` if ``b < 0`` else NaN.
    3. ``Lhat = a + b*t``; ``r2 = sum((Lhat - mean(L))**2) / sum((L - mean(L))**2)``
       (Eq. B.2); ``xi = 1000*(1 - r2)`` (Eq. B.3).

    Parameters
    ----------
    time
        Time in s, 1-D, increasing.
    level
        Decay curve in dB, 1-D, same length.
    start_db, stop_db
        Evaluation range in dB, ``0 >= start_db > stop_db``.
    name
        Label stored in the result.

    Returns
    -------
    DecayFit

    Raises
    ------
    ValueError
        If the arrays are not 1-D of equal length, or the range is invalid.

    References
    ----------
    ISO 3382-2:2008, clause 6, Annex B (B.1)-(B.3), Annex C (C.1)-(C.6).
    """
    t = np.asarray(time, dtype=float)
    lv = np.asarray(level, dtype=float)
    if t.ndim != 1 or lv.ndim != 1 or t.size != lv.size:
        raise ValueError("time and level must be 1-D arrays of equal length")
    if not (0.0 >= start_db > stop_db):
        raise ValueError("need 0 >= start_db > stop_db")
    nan = float("nan")
    hit_start = np.flatnonzero(lv <= start_db)
    hit_stop = np.flatnonzero(lv <= stop_db)
    t_start = float(t[hit_start[0]]) if hit_start.size else nan
    t_stop = float(t[hit_stop[0]]) if hit_stop.size else nan
    if hit_start.size == 0 or hit_stop.size == 0 or hit_stop[0] - hit_start[0] < 2:
        return DecayFit(name, start_db, stop_db, t_start, t_stop, nan, nan, nan, nan, nan, 0)
    sel = slice(int(hit_start[0]), int(hit_stop[0]) + 1)
    ts, ls = t[sel], lv[sel]
    m = ts.size
    tm, lm = ts.mean(), ls.mean()
    b = (np.sum(ts * ls) - m * tm * lm) / (np.sum(ts**2) - m * tm**2)
    a = lm - b * tm
    lhat = a + b * ts
    r2 = np.sum((lhat - lm) ** 2) / np.sum((ls - lm) ** 2)
    decay_time = float(-60.0 / b) if b < 0 else nan
    return DecayFit(
        name=name,
        start_db=start_db,
        stop_db=stop_db,
        t_start=t_start,
        t_stop=t_stop,
        intercept=float(a),
        slope=float(b),
        decay_time=decay_time,
        r2=float(r2),
        xi=float(1000.0 * (1.0 - r2)),
        n_points=m,
    )


def curvature(t20: float, t30: float) -> float:
    """Curvature ``C = 100 * (T30/T20 - 1)`` in percent, ISO 3382-2:2008 Eq. (B.4).

    Returns NaN if either value is NaN or ``t20 <= 0``.
    """
    if not (np.isfinite(t20) and np.isfinite(t30)) or t20 <= 0:
        return float("nan")
    return float(100.0 * (t30 / t20 - 1.0))


def fit_flags(
    fit: DecayFit,
    peak_to_noise_db: float,
    bandwidth: float | None = None,
    reverse: bool = False,
) -> tuple[str, ...]:
    """Return the validity flags of one fit (empty tuple: no objection).

    - ``"range"``: no decay time (curve does not cover the range); no other flag is added.
    - ``"dynamic_range"``: ``peak_to_noise_db < (start_db - stop_db) + DYNAMIC_RANGE_MARGIN_DB``
      (ISO 3382-1:2009 5.3.3; 35 dB for T20, 45 dB for T30 as in ISO 3382-2:2008 5.3.2).
    - ``"nonlinear"``: ``xi > XI_LIMIT`` (ISO 3382-2:2008 B.2).
    - ``"bt"``: ``bandwidth * decay_time <= MIN_BT_REVERSED if reverse else MIN_BT_FORWARD``
      (ISO 3382-2:2008 7.3 Eq. (4) and NOTE); skipped if ``bandwidth`` is None.

    Flags appear in this order.
    """
    if not np.isfinite(fit.decay_time):
        return ("range",)
    flags: list[str] = []
    if peak_to_noise_db < (fit.start_db - fit.stop_db) + DYNAMIC_RANGE_MARGIN_DB:
        flags.append("dynamic_range")
    if fit.xi > XI_LIMIT:
        flags.append("nonlinear")
    if bandwidth is not None:
        limit = MIN_BT_REVERSED if reverse else MIN_BT_FORWARD
        if bandwidth * fit.decay_time <= limit:
            flags.append("bt")
    return tuple(flags)


def curvature_flags(c: float) -> tuple[str, ...]:
    """Return ``("curvature",)`` if ``c > CURVATURE_LIMIT`` or ``c < 0``, else ``()``.

    ISO 3382-2:2008 B.4. NaN gives ``()``.
    """
    if np.isfinite(c) and (c > CURVATURE_LIMIT or c < 0):
        return ("curvature",)
    return ()
