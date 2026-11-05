"""Room-acoustic analysis of one impulse response: all bands, all parameters.

Pipeline (integrated impulse response method, docs/DECISIONS.md D-2):

1. Onset from the broadband IR (``resonus_core.window.onset``, ISO 3382-1:2009 A.3.4).
2. Broadband and band-filtered responses from the onset (``core.bands``; filters
   PROVISIONAL, IEC 61260-1 compliance not verified).
3. Decay curves by backward integration with truncation/correction (``core.decay``).
4. EDT, T20, T30 by least squares, xi, curvature C, validity flags (``core.fit``).
5. C50, C80, D50, Ts with window-before-filtering (``core.energy``).
6. Standard deviation of T20/T30 for one position, ISO 3382-2:2008 Eqs. (2)-(3) with
   n = 10 (``core.uncertainty``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from resonus_core.bands import DEFAULT_FILTER_ORDER, BandSet, filter_bands  # noqa: F401
from resonus_core.window import onset as find_onset  # noqa: F401

from resonus_room.core.decay import DecayCurve, decay_curve  # noqa: F401
from resonus_room.core.energy import energy_parameters  # noqa: F401
from resonus_room.core.fit import (
    EVALUATION_RANGES,  # noqa: F401
    DecayFit,
    curvature,  # noqa: F401
    curvature_flags,  # noqa: F401
    fit_decay,  # noqa: F401
    fit_flags,  # noqa: F401
)
from resonus_room.core.uncertainty import bandwidth, reverberation_std  # noqa: F401

STANDARDS = (
    "ISO 3382-1:2009 5.3.3, A.2.2, A.2.3, A.3.4",
    "ISO 3382-2:2008 6, 7.1, 7.3, Annex B, Annex C",
)
PARAMETERS = ("EDT", "T20", "T30")


@dataclass(frozen=True)
class ParameterSet:
    """Room-acoustic parameters of one signal (broadband) or one band.

    Parameters
    ----------
    curve
        Decay curve.
    fits
        ``{"EDT": DecayFit, "T20": ..., "T30": ...}``.
    curvature
        ``C = 100 (T30/T20 - 1)`` in percent (ISO 3382-2:2008 B.4).
    flags
        Validity flags, each prefixed with the parameter, e.g. ``"T30:dynamic_range"``,
        plus ``"curvature"``.
    c50, c80, d50, ts
        Energy parameters (dB, dB, 1, s).
    """

    curve: DecayCurve
    fits: dict[str, DecayFit]
    curvature: float
    flags: tuple[str, ...]
    c50: float
    c80: float
    d50: float
    ts: float

    @property
    def edt(self) -> float:
        """Early decay time in s."""
        return self.fits["EDT"].decay_time

    @property
    def t20(self) -> float:
        """T20 in s."""
        return self.fits["T20"].decay_time

    @property
    def t30(self) -> float:
        """T30 in s."""
        return self.fits["T30"].decay_time


@dataclass(frozen=True)
class RoomAnalysis:
    """Analysis of one impulse response.

    Parameters
    ----------
    sample_rate
        Sample rate in Hz.
    onset
        Index of the IR start in the given signal.
    bands
        Band set analysed.
    broadband
        Parameters of the unfiltered IR (no BT flag).
    band_results
        One :class:`ParameterSet` per band.
    t20_std, t30_std
        Standard deviation per band for one position (n = 10), s.
    settings
        Analysis settings used (filter order, direction, decay method).
    standards
        Standards and clauses applied.
    """

    sample_rate: float
    onset: int
    bands: BandSet
    broadband: ParameterSet
    band_results: list[ParameterSet]
    t20_std: NDArray[np.float64]
    t30_std: NDArray[np.float64]
    settings: dict[str, object] = field(default_factory=dict)
    standards: tuple[str, ...] = STANDARDS

    def values(self, name: str) -> NDArray[np.float64]:
        """Return the per-band values of one parameter as an array.

        Names: ``"EDT"``, ``"T20"``, ``"T30"``, ``"C"``, ``"C50"``, ``"C80"``, ``"D50"``,
        ``"Ts"``, ``"PNR"`` (peak-to-noise ratio, dB).

        Raises
        ------
        KeyError
            On an unknown name.
        """
        getters = {
            "EDT": lambda p: p.edt,
            "T20": lambda p: p.t20,
            "T30": lambda p: p.t30,
            "C": lambda p: p.curvature,
            "C50": lambda p: p.c50,
            "C80": lambda p: p.c80,
            "D50": lambda p: p.d50,
            "Ts": lambda p: p.ts,
            "PNR": lambda p: p.curve.peak_to_noise_db,
        }
        if name not in getters:
            raise KeyError(name)
        return np.array([getters[name](p) for p in self.band_results], dtype=float)


def parameter_set(
    p: ArrayLike,
    sample_rate: float,
    energy: tuple[float, float, float, float],
    band_width: float | None,
    reverse: bool,
    method: str,
) -> ParameterSet:
    """Decay curve, fits, curvature and flags of one signal starting at the onset.

    ``curve = decay_curve(p, sample_rate, method)``; for each name in ``PARAMETERS``:
    ``fits[name] = fit_decay(curve.time, curve.level, *EVALUATION_RANGES[name], name)``,
    flags ``f"{name}:{flag}"`` for ``flag in fit_flags(fit, curve.peak_to_noise_db,
    band_width, reverse)``; ``c = curvature(T20, T30)``, flags extended by
    ``curvature_flags(c)``. ``energy = (c50, c80, d50, ts)``.
    """
    curve = decay_curve(p, sample_rate, method)
    fits: dict[str, DecayFit] = {}
    flags: list[str] = []
    for name in PARAMETERS:
        fit = fit_decay(
            curve.time,
            curve.level,
            *EVALUATION_RANGES[name],
            name=name,
        )
        fits[name] = fit
        flags += [
            f"{name}:{f}" for f in fit_flags(fit, curve.peak_to_noise_db, band_width, reverse)
        ]
    c = curvature(fits["T20"].decay_time, fits["T30"].decay_time)
    flags += list(curvature_flags(c))
    c50, c80, d50, ts = energy
    return ParameterSet(
        curve=curve,
        fits=fits,
        curvature=c,
        flags=tuple(flags),
        c50=c50,
        c80=c80,
        d50=d50,
        ts=ts,
    )


def analyze(
    ir: ArrayLike,
    sample_rate: float,
    bands: BandSet,
    order: int = DEFAULT_FILTER_ORDER,
    reverse: bool = False,
    method: str = "correct",
    onset: int | None = None,
) -> RoomAnalysis:
    """Analyse one impulse response in all bands (see the module docstring).

    1. ``x = np.asarray(ir, float)`` (1-D); trailing exact zeros (export padding) are
       removed first (D-13); ``onset = find_onset(x)`` if None. ``y = x[onset:]``.
    2. Broadband: ``e = energy_parameters(x, fs, onset)``;
       ``broadband = parameter_set(y, fs, (e.c50[0], e.c80[0], e.d50[0], e.ts[0]), None,
       reverse, method)``.
    3. Bands: ``yb = filter_bands(y, bands, fs, order, reverse)``;
       ``eb = energy_parameters(x, fs, onset, bands, order, reverse)``;
       ``bw = bandwidth(bands.center, bands.fraction)``; per band ``k``:
       ``parameter_set(yb[:, k], fs, (eb.c50[k], ...), float(bw[k]), reverse, method)``.
    4. ``t20_std = reverberation_std(T20, bw, 1, 20)``, ``t30_std = reverberation_std(T30,
       bw, 1, 30)``.
    5. ``settings = {"filter_order": order, "reverse": reverse, "decay_method": method}``.

    Raises
    ------
    ValueError
        If ``ir`` is not 1-D, the onset is outside the signal, or on errors of the steps.
    """
    x = np.asarray(ir, dtype=float)
    if x.ndim != 1:
        raise ValueError("ir must be a 1-D array")
    nz = np.flatnonzero(x)
    if nz.size == 0:
        raise ValueError("ir is all zero")
    x = x[: nz[-1] + 1]
    i0 = find_onset(x) if onset is None else onset
    if not (0 <= i0 < x.size):
        raise ValueError("onset must be within the signal")
    y = x[i0:]
    e = energy_parameters(x, sample_rate, i0)
    broadband = parameter_set(
        y,
        sample_rate,
        (float(e.c50[0]), float(e.c80[0]), float(e.d50[0]), float(e.ts[0])),
        None,
        reverse,
        method,
    )
    yb = filter_bands(y, bands, sample_rate, order, reverse)
    eb = energy_parameters(x, sample_rate, i0, bands, order, reverse)
    bw = bandwidth(bands.center, bands.fraction)
    results = [
        parameter_set(
            yb[:, k],
            sample_rate,
            (
                float(eb.c50[k]),
                float(eb.c80[k]),
                float(eb.d50[k]),
                float(eb.ts[k]),
            ),
            float(bw[k]),
            reverse,
            method,
        )
        for k in range(len(bands))
    ]
    t20 = np.array([r.t20 for r in results], dtype=float)
    t30 = np.array([r.t30 for r in results], dtype=float)
    return RoomAnalysis(
        sample_rate=float(sample_rate),
        onset=int(i0),
        bands=bands,
        broadband=broadband,
        band_results=results,
        t20_std=reverberation_std(t20, bw, 1, 20),
        t30_std=reverberation_std(t30, bw, 1, 30),
        settings={"filter_order": order, "reverse": reverse, "decay_method": method},
    )
