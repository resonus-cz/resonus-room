"""Binaural parameters from a pair of ear impulse responses, ISO 3382-1:2009 Annex B.

Annex B is informative. Definitions (checked on the rendered page of the standard):

- Eq. (B.1): ``IACF(t1, t2; tau) = int_{t1}^{t2} pl(t) pr(t + tau) dt /
  sqrt(int_{t1}^{t2} pl^2 dt * int_{t1}^{t2} pr^2 dt)``,
- Eq. (B.2): ``IACC(t1, t2) = max |IACF(t1, t2; tau)|`` for ``-1 ms < tau < +1 ms``,

with ``pl``, ``pr`` the impulse responses at the entrances of the left and right ear canals.
B.4 names the intervals used here: early (``t1 = 0``, ``t2 = 0.08 s``), late (``t1 = 0.08
s``, ``t2`` beyond the reverberation time) and the general form (``t1 = 0``, ``t2`` of the
order of the reverberation time), "generally measured in octave bands ranging from 125 Hz
to 4 000 Hz"; JND 0.075.

Decisions (docs/DECISIONS.md D-17, user 2026-11-04, D-17):

- ``t = 0`` is the onset (ISO 3382-1:2009 A.3.4, ``resonus_core.window.onset``) of the
  *earlier* ear, used for both ears, so the interaural time difference is kept.
- ``t2`` of the late and general intervals is the end of the analysed responses.
- Band values use window-before-filtering, as the energy parameters (``core/energy.py``;
  B.5: the procedure "should parallel that given in Annex A"): each ear is cut to the
  interval first, zero-padded by ``energy.PAD`` on both sides, then band-filtered; the
  integrals of Eq. (B.1) run over the whole filtered, padded segment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from resonus_core.bands import DEFAULT_FILTER_ORDER, BandSet, filter_bands
from resonus_core.window import onset as find_onset

from resonus_room.core.energy import PAD

# ISO 3382-1:2009 B.2, Eq. (B.2): lag range -1 ms < tau < +1 ms (strict).
TAU_MAX = 1e-3
# ISO 3382-1:2009 B.4: limit between the early and the late interval, in s.
EARLY_LIMIT = 0.080
# ISO 3382-1:2009 B.4: "generally measured in octave bands ranging from 125 Hz to 4 000 Hz".
IACC_BAND_RANGE = (125.0, 4000.0)
# ISO 3382-1:2009 B.4: just noticeable difference of IACC.
IACC_JND = 0.075


@dataclass(frozen=True)
class BinauralParameters:
    """IACC of one ear pair, one value per band (or one broadband value).

    Parameters
    ----------
    early, late, full
        IACC for the intervals ``[0, EARLY_LIMIT)``, ``[EARLY_LIMIT, end)`` and ``[0,
        end)`` after the common onset, Eq. (B.2).
    onset
        Common onset index (earlier ear), ISO 3382-1:2009 A.3.4.
    """

    early: NDArray[np.float64]
    late: NDArray[np.float64]
    full: NDArray[np.float64]
    onset: int


def max_lag(sample_rate: float) -> int:
    """Largest lag ``K`` in samples with ``K / sample_rate < TAU_MAX``.

    ``K = ceil(TAU_MAX * sample_rate) - 1`` (strict inequality of Eq. (B.2)).

    Raises
    ------
    ValueError
        If ``sample_rate <= 0``.

    References
    ----------
    ISO 3382-1:2009, B.2, Eq. (B.2).
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    return math.ceil(TAU_MAX * sample_rate) - 1


def iacf(
    left: ArrayLike, right: ArrayLike, sample_rate: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the normalized interaural cross-correlation function, Eq. (B.1), |tau| < 1 ms.

    The integrals run over the whole given (already windowed) signals, sampled at
    ``sample_rate``: with ``K = max_lag(sample_rate)`` and ``k = -K..K``,
    ``values[k] = sum_n l[n] * r[n + k] / sqrt(sum l**2 * sum r**2)``, where ``r[m] = 0``
    outside ``0..len(r)-1``. A positive lag means the right ear lags (``pr(t + tau)``).

    Parameters
    ----------
    left, right
        Ear signals ``pl``, ``pr``, 1-D, equal length.
    sample_rate
        Sample rate in Hz.

    Returns
    -------
    lags, values
        Lags ``k / sample_rate`` in s (increasing) and IACF values (-1...1).

    Raises
    ------
    ValueError
        If the signals are not 1-D, empty, of different length, or one has zero energy,
        or ``sample_rate <= 0``.

    References
    ----------
    ISO 3382-1:2009, B.2, Eq. (B.1).
    """
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if left_arr.ndim != 1 or right_arr.ndim != 1:
        raise ValueError("signals must be 1-D")
    if left_arr.size == 0 or right_arr.size == 0:
        raise ValueError("signals must not be empty")
    if left_arr.size != right_arr.size:
        raise ValueError("signals must have same length")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    el = np.sum(left_arr**2)
    er = np.sum(right_arr**2)
    if el == 0 or er == 0:
        raise ValueError("signals must have non-zero energy")
    K = max_lag(sample_rate)
    n = left_arr.size
    values = np.empty(2 * K + 1, dtype=float)
    for idx, k in enumerate(range(-K, K + 1)):
        if k >= 0:
            if k < n:
                val = np.dot(left_arr[: n - k], right_arr[k:])
            else:
                val = 0.0
        else:
            if -k < n:
                val = np.dot(left_arr[-k:], right_arr[: n + k])
            else:
                val = 0.0
        values[idx] = val / math.sqrt(el * er)
    lags = np.arange(-K, K + 1) / sample_rate
    return lags, values


def iacc(left: ArrayLike, right: ArrayLike, sample_rate: float) -> float:
    """Interaural cross-correlation coefficient ``max |IACF|``, Eq. (B.2).

    Raises
    ------
    ValueError
        As :func:`iacf`.

    References
    ----------
    ISO 3382-1:2009, B.2, Eq. (B.2).
    """
    _, values = iacf(left, right, sample_rate)
    return float(np.max(np.abs(values)))


def binaural_parameters(
    left: ArrayLike,
    right: ArrayLike,
    sample_rate: float,
    bands: BandSet | None = None,
    order: int = DEFAULT_FILTER_ORDER,
    reverse: bool = False,
    onset: int | None = None,
) -> BinauralParameters:
    """Early, late and full IACC of an ear pair, broadband or per band.

    1. ``l``, ``r`` as float arrays (1-D, equal length, else ValueError); ``i0 =
       min(find_onset(l), find_onset(r))`` if ``onset`` is None (ValueError if ``onset``
       is outside ``0..len-1``).
    2. ``ne = round(EARLY_LIMIT * sample_rate)``; intervals relative to ``i0``: early
       ``[i0, i0 + ne)``, late ``[i0 + ne, len)``, full ``[i0, len)`` (ValueError if the
       late interval is empty).
    3. ``bands is None``: ``iacc(l[a:b], r[a:b], sample_rate)`` per interval, arrays of
       shape ``(1,)``. Otherwise each ear's segment is zero-padded with ``npad =
       round(PAD * sample_rate)`` samples on both sides and filtered with
       ``filter_bands(padded, bands, sample_rate, order, reverse)``; per band ``j``:
       ``iacc(yl[:, j], yr[:, j], sample_rate)``; shape ``(len(bands),)``.

    Parameters
    ----------
    left, right
        Broadband ear impulse responses ``pl``, ``pr``.
    sample_rate
        Sample rate in Hz.
    bands, order, reverse
        Band set and filter settings (:func:`resonus_core.bands.filter_bands`).
    onset
        Common onset index; None: earlier ear per A.3.4.

    Returns
    -------
    BinauralParameters

    Raises
    ------
    ValueError
        On invalid input (see the steps).

    References
    ----------
    ISO 3382-1:2009, Annex B (B.2 Eqs. (B.1), (B.2); B.4 intervals); A.3.4 (onset,
    window before filtering).
    """
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if left_arr.ndim != 1 or right_arr.ndim != 1:
        raise ValueError("signals must be 1-D")
    if left_arr.size == 0 or right_arr.size == 0:
        raise ValueError("signals must not be empty")
    if left_arr.size != right_arr.size:
        raise ValueError("signals must have same length")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")

    if onset is None:
        i0 = min(find_onset(left_arr), find_onset(right_arr))
    else:
        if not (0 <= onset < left_arr.size):
            raise ValueError("onset must be in [0, len-1]")
        i0 = onset

    n = left_arr.size
    ne = round(EARLY_LIMIT * sample_rate)
    if i0 + ne >= n:
        raise ValueError("late interval is empty")

    intervals = {
        "early": (i0, i0 + ne),
        "late": (i0 + ne, n),
        "full": (i0, n),
    }

    def _compute_interval(a: int, b: int) -> NDArray[np.float64]:
        if bands is None:
            val = iacc(left_arr[a:b], right_arr[a:b], sample_rate)
            return np.array([val], dtype=float)
        npad = round(PAD * sample_rate)
        seg_l = np.concatenate([np.zeros(npad), left_arr[a:b], np.zeros(npad)])
        seg_r = np.concatenate([np.zeros(npad), right_arr[a:b], np.zeros(npad)])
        fl = filter_bands(seg_l, bands, sample_rate, order, reverse)
        fr = filter_bands(seg_r, bands, sample_rate, order, reverse)
        vals = np.empty(len(bands), dtype=float)
        for j in range(len(bands)):
            vals[j] = iacc(fl[:, j], fr[:, j], sample_rate)
        return vals

    early_vals = _compute_interval(*intervals["early"])
    late_vals = _compute_interval(*intervals["late"])
    full_vals = _compute_interval(*intervals["full"])

    return BinauralParameters(
        early=early_vals,
        late=late_vals,
        full=full_vals,
        onset=i0,
    )
