"""Energy ratios and centre time: C50, C80, D50, Ts (ISO 3382-1:2009 A.2.3).

Band values use the window-before-filtering approach of ISO 3382-1:2009 A.3.4: the
broadband impulse response is split at ``onset + te`` *before* filtering; early and late
parts are filtered separately, each zero-padded by ``PAD`` seconds on both sides so that
the energy delayed (or, with time-reversed filtering, advanced) by the filter is
included in the integrals.

Decision (docs/DECISIONS.md D-8, user 2026-11-03, D-8): the late integral runs to the end
of the given impulse response, i.e. it includes background noise; band Ts is computed
from the filtered response with time measured from the onset (filter delay included).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from resonus_core.bands import DEFAULT_FILTER_ORDER, BandSet, filter_bands  # noqa: F401

# Early time limits t_e in s: ISO 3382-1:2009 A.2.3 (50 ms for speech, 80 ms for music).
EARLY_50 = 0.050
EARLY_80 = 0.080
# Design choice (D-8): zero padding on both sides of filtered segments, in s; long enough for
# the ringing of one-third-octave filters down to 25 Hz (design choice).
PAD = 0.5


@dataclass(frozen=True)
class EnergyParameters:
    """Energy parameters, one value per band (or one broadband value).

    Parameters
    ----------
    c50, c80
        Early-to-late index ``C_te = 10 lg(E_early/E_late)`` in dB, Eq. (A.10).
    d50
        Definition ``E_early,50 / (E_early,50 + E_late,50)``, Eq. (A.11) (0...1).
    ts
        Centre time in s, Eq. (A.13).
    """

    c50: NDArray[np.float64]
    c80: NDArray[np.float64]
    d50: NDArray[np.float64]
    ts: NDArray[np.float64]


def _segments(ir: ArrayLike, sample_rate: float, onset: int) -> NDArray[np.float64]:
    """Return ``ir[onset:]`` as float; ValueError on invalid input."""
    x = np.asarray(ir, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("ir must be a non-empty 1-D array")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    if not (0 <= onset < x.size):
        raise ValueError("onset must be in [0, len(ir))")
    return x[onset:]


def early_late_energy(
    ir: ArrayLike,
    sample_rate: float,
    onset: int,
    te: float,
    bands: BandSet | None = None,
    order: int = DEFAULT_FILTER_ORDER,
    reverse: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Early and late energies ``E_early = int_0^te p^2 dt``, ``E_late = int_te^end p^2 dt``.

    ``x = ir[onset:]``, ``ne = round(te * sample_rate)`` (``1 <= ne < len(x)``),
    ``early = x[:ne]``, ``late = x[ne:]``.

    - ``bands is None``: ``E = sum(segment**2) / sample_rate``; arrays of shape ``(1,)``.
    - Otherwise each segment is zero-padded with ``npad = round(PAD * sample_rate)`` samples
      on both sides, filtered with ``filter_bands(padded, bands, sample_rate, order,
      reverse)``, and ``E = sum(filtered**2, axis=0) / sample_rate``; shape ``(len(bands),)``.

    Parameters
    ----------
    ir
        Broadband impulse response, 1-D.
    sample_rate
        Sample rate in Hz.
    onset
        Index of the start of the impulse response (ISO 3382-1:2009 A.3.4).
    te
        Early time limit in s, > 0.
    bands, order, reverse
        Band set and filter settings (see :func:`resonus_core.bands.filter_bands`).

    Returns
    -------
    early, late
        Energies in (unit of ``ir``)^2 * s.

    Raises
    ------
    ValueError
        On invalid input or if ``te`` does not leave at least one early and one late sample.

    References
    ----------
    ISO 3382-1:2009, A.2.3 Eq. (A.10); A.3.4 (window before filtering).
    """
    x = _segments(ir, sample_rate, onset)
    ne = round(te * sample_rate)
    if not (1 <= ne < x.size):
        raise ValueError("te must leave at least one early and one late sample")
    early, late = x[:ne], x[ne:]
    if bands is None:
        early_energy = np.array([np.sum(early**2)]) / sample_rate
        late_energy = np.array([np.sum(late**2)]) / sample_rate
    else:
        npad = round(PAD * sample_rate)
        energies = []
        for seg in (early, late):
            padded = np.concatenate([np.zeros(npad), seg, np.zeros(npad)])
            y = filter_bands(padded, bands, sample_rate, order, reverse)
            energies.append(np.sum(y**2, axis=0) / sample_rate)
        early_energy, late_energy = energies
    return early_energy, late_energy


def centre_time(
    ir: ArrayLike,
    sample_rate: float,
    onset: int,
    bands: BandSet | None = None,
    order: int = DEFAULT_FILTER_ORDER,
    reverse: bool = False,
) -> NDArray[np.float64]:
    """Centre time ``Ts = sum(t * p**2) / sum(p**2)`` in s, ISO 3382-1:2009 Eq. (A.13).

    ``x = ir[onset:]``. Broadband (``bands is None``): ``t = arange(len(x)) / sample_rate``,
    shape ``(1,)``. Bands: ``x`` zero-padded with ``npad = round(PAD * sample_rate)`` samples
    on both sides and filtered with :func:`filter_bands`; ``t = (arange(len(padded)) -
    npad) / sample_rate`` (time from the onset); shape ``(len(bands),)``.

    Raises
    ------
    ValueError
        On invalid input.
    """
    x = _segments(ir, sample_rate, onset)
    if bands is None:
        t = np.arange(x.size) / sample_rate
        return np.array([np.sum(t * x**2) / np.sum(x**2)])
    npad = round(PAD * sample_rate)
    padded = np.concatenate([np.zeros(npad), x, np.zeros(npad)])
    y2 = filter_bands(padded, bands, sample_rate, order, reverse) ** 2
    t = (np.arange(padded.size) - npad) / sample_rate
    return np.sum(t[:, None] * y2, axis=0) / np.sum(y2, axis=0)


def energy_parameters(
    ir: ArrayLike,
    sample_rate: float,
    onset: int,
    bands: BandSet | None = None,
    order: int = DEFAULT_FILTER_ORDER,
    reverse: bool = False,
) -> EnergyParameters:
    """C50, C80, D50 and Ts of an impulse response (broadband or per band).

    ``e50, l50 = early_late_energy(..., EARLY_50, ...)``, ``e80, l80`` likewise;
    ``c50 = 10*lg(e50/l50)``, ``c80 = 10*lg(e80/l80)`` (Eq. A.10),
    ``d50 = e50/(e50 + l50)`` (Eq. A.11), ``ts = centre_time(...)`` (Eq. A.13).

    Raises
    ------
    ValueError
        On invalid input (see :func:`early_late_energy`).

    References
    ----------
    ISO 3382-1:2009, A.2.3, Eqs. (A.10), (A.11), (A.13); A.3.4.
    """
    e50, l50 = early_late_energy(ir, sample_rate, onset, EARLY_50, bands, order, reverse)
    e80, l80 = early_late_energy(ir, sample_rate, onset, EARLY_80, bands, order, reverse)
    with np.errstate(divide="ignore"):
        c50 = 10 * np.log10(e50 / l50)
        c80 = 10 * np.log10(e80 / l80)
    return EnergyParameters(
        c50=c50,
        c80=c80,
        d50=e50 / (e50 + l50),
        ts=centre_time(ir, sample_rate, onset, bands, order, reverse),
    )
