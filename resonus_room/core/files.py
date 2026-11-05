"""Import of impulse responses and export of room-acoustic results.

Results are stored as resonus-core ``BandResults`` (``.h5``, readable by every Re:sonus
app, docs/DECISIONS.md D-10) and as human-readable JSON and CSV reports (D-7).
"""

from __future__ import annotations

import csv  # noqa: F401
import dataclasses
import io  # noqa: F401
import math  # noqa: F401
from pathlib import Path
from typing import Any

import numpy as np  # noqa: F401
from resonus_core import data  # noqa: F401
from resonus_core.bands import BandSet, band_set  # noqa: F401
from resonus_core.data import BandResults, FrequencyResponse, ImpulseResponse, Metadata  # noqa: F401
from scipy.io import wavfile  # noqa: F401

from resonus_room import __version__  # noqa: F401
from resonus_room.core.absorption import AbsorptionResult
from resonus_room.core.analysis import PARAMETERS, RoomAnalysis  # noqa: F401
from resonus_room.core.binaural import EARLY_LIMIT, TAU_MAX, BinauralParameters  # noqa: F401

APP_NAME = "resonus-room"
# Statements attached to every exported result (docs/COMPLIANCE.md).
NOTES = (
    "Calculation method only; not a type-approved instrument (IEC 61672-1/IEC 61260-1).",
    "PROVISIONAL: band filters are Butterworth; IEC 61260-1 class compliance not verified.",
)
CSV_COLUMNS = (
    "band_hz", "center_hz", "EDT_s", "T20_s", "T30_s", "T20_std_s", "T30_std_s",
    "curvature_pct", "xi_EDT", "xi_T20", "xi_T30", "PNR_dB", "C50_dB", "C80_dB", "D50",
    "Ts_s", "flags",
)  # fmt: skip
# BandResults quantities: name -> unit. xi in per mille (ISO 3382-2:2008 B.3), curvature
# C in percent (B.4).
ANALYSIS_QUANTITIES = {
    "EDT": "s", "T20": "s", "T30": "s", "T20_std": "s", "T30_std": "s", "C": "%",
    "xi_EDT": "permille", "xi_T20": "permille", "xi_T30": "permille",
    "r2_EDT": "1", "r2_T20": "1", "r2_T30": "1",
    "PNR": "dB", "C50": "dB", "C80": "dB", "D50": "1", "Ts": "s",
}  # fmt: skip
ABSORPTION_QUANTITIES = {
    "T1": "s", "T2": "s", "A1": "m2", "A2": "m2", "AT": "m2", "alpha_s": "1",
    "A_obj": "m2", "alpha_std": "1", "AT_std": "m2",
}  # fmt: skip
# IACC per interval of ISO 3382-1:2009 B.4 (early 0-80 ms, late 80 ms-end, full 0-end
# after the common onset; core/binaural.py). Descriptive names, not the symbols of the
# standard.
BINAURAL_QUANTITIES = {"IACC_early": "1", "IACC_late": "1", "IACC_full": "1"}
BINAURAL_COLUMNS = ("band_hz", "center_hz", "IACC_early", "IACC_late", "IACC_full")
BINAURAL_STANDARDS = ("ISO 3382-1:2009 Annex B (informative), B.2 Eqs. (B.1), (B.2), B.4",)
ABSORPTION_COLUMNS = (
    "band_hz", "T1_s", "T2_s", "A1_m2", "A2_m2", "AT_m2", "alpha_s", "A_obj_m2",
    "alpha_std", "AT_std_m2",
)  # fmt: skip


def load_impulse_response(path: str | Path, channel: int = 0) -> ImpulseResponse:
    """Load an impulse response from a resonus ``.h5`` file or a WAV file.

    - ``.h5``: ``resonus_core.data.load``; ValueError if it is not an ImpulseResponse.
    - ``.wav``: ``scipy.io.wavfile.read``; integer samples are scaled to [-1, 1):
      ``int16``/``int32`` divided by ``2**15``/``2**31``, ``uint8`` as ``(x - 128) / 128``;
      float samples are kept. Multichannel files: column ``channel``. Unit ``"1"``,
      metadata ``Metadata(APP_NAME, __version__, description=<file name>)``.

    Raises
    ------
    ValueError
        On an unknown suffix, a non-IR ``.h5`` file, an unsupported sample type or a
        channel outside the file.
    OSError
        If the file cannot be read.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".h5":
        result = data.load(p)
        if not isinstance(result, ImpulseResponse):
            raise ValueError(f"{p}: not an impulse response")
        return result
    if suffix != ".wav":
        raise ValueError(f"{p}: unsupported file type (use .h5 or .wav)")
    rate, samples = wavfile.read(p)
    x = np.asarray(samples)
    if np.issubdtype(x.dtype, np.integer):
        if x.dtype == np.int16:
            y = x / 2.0**15
        elif x.dtype == np.int32:
            y = x / 2.0**31
        elif x.dtype == np.uint8:
            y = (x.astype(float) - 128) / 128
        else:
            raise ValueError(f"{p}: unsupported integer sample type {x.dtype}")
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(float)
    else:
        raise ValueError(f"{p}: unsupported sample type {x.dtype}")
    if y.ndim == 2:
        if not (0 <= channel < y.shape[1]):
            raise ValueError(f"{p}: channel {channel} out of range")
        y = y[:, channel]
    elif channel != 0:
        raise ValueError(f"{p}: channel {channel} out of range")
    return ImpulseResponse(y, float(rate), "1", Metadata(APP_NAME, __version__, description=p.name))


def impulse_to_frequency_response(ir: ImpulseResponse) -> FrequencyResponse:
    """Transfer function of an impulse response: the DFT of the whole response.

    ``frequency = np.fft.rfftfreq(n, 1 / ir.sample_rate)`` and ``values =
    np.fft.rfft(ir.data)`` with ``n = ir.data.size``; unit and metadata of ``ir`` are kept.
    The phase is referenced to the first sample of the file (for a measured response t = 0;
    a delay before the direct sound shows as a linear phase). No window and no smoothing
    (fractional-octave smoothing is deferred in resonus-core).

    Parameters
    ----------
    ir
        Impulse response, e.g. loaded from an external file.

    Returns
    -------
    FrequencyResponse
        ``n // 2 + 1`` bins from 0 Hz to the Nyquist frequency (or just below for odd n).
    """
    n = ir.data.size
    return FrequencyResponse(
        np.fft.rfftfreq(n, 1 / ir.sample_rate),
        np.fft.rfft(ir.data),
        ir.unit,
        ir.metadata,
    )


def _clean(value: Any) -> Any:
    """Convert numpy values to JSON types; NaN and infinities become None."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def analysis_to_dict(analysis: RoomAnalysis) -> dict[str, Any]:
    """JSON-serializable summary of a :class:`RoomAnalysis` (NaN -> None).

    Keys: ``software``, ``software_version``, ``standards`` (list), ``notes`` (``NOTES``),
    ``settings``, ``sample_rate``, ``onset_s`` (``onset / sample_rate``), ``bands``
    (``{"fraction", "nominal_hz", "center_hz"}``), ``values`` (``{name:
    analysis.values(name)}`` for EDT, T20, T30, C, C50, C80, D50, Ts, PNR plus ``T20_std``,
    ``T30_std``), ``fits`` (``{name: {"r2", "xi", "slope", "intercept", "t_start",
    "t_stop", "n_points"}}`` per band lists for each name in ``PARAMETERS``), ``flags``
    (list of lists), ``broadband`` (``{"EDT", "T20", "T30", "C", "C50", "C80", "D50",
    "Ts", "PNR", "flags"}``).
    """
    a = analysis
    values = {n: a.values(n) for n in ("EDT", "T20", "T30", "C", "C50", "C80", "D50", "Ts", "PNR")}
    values["T20_std"] = a.t20_std
    values["T30_std"] = a.t30_std
    fits = {}
    for n in PARAMETERS:
        fits[n] = {
            key: [getattr(p.fits[n], key) for p in a.band_results]
            for key in ("r2", "xi", "slope", "intercept", "t_start", "t_stop", "n_points")
        }
    onset_s = a.onset / a.sample_rate
    bands = {
        "fraction": a.bands.fraction,
        "nominal_hz": a.bands.nominal,
        "center_hz": a.bands.center,
    }
    flags = [list(p.flags) for p in a.band_results]
    b = a.broadband
    broadband = {
        "EDT": b.edt,
        "T20": b.t20,
        "T30": b.t30,
        "C": b.curvature,
        "C50": b.c50,
        "C80": b.c80,
        "D50": b.d50,
        "Ts": b.ts,
        "PNR": b.curve.peak_to_noise_db,
        "flags": list(b.flags),
    }
    out = {
        "software": APP_NAME,
        "software_version": __version__,
        "standards": list(a.standards),
        "notes": list(NOTES),
        "settings": dict(a.settings),
        "sample_rate": a.sample_rate,
        "onset_s": onset_s,
        "bands": bands,
        "values": values,
        "fits": fits,
        "flags": flags,
        "broadband": broadband,
    }
    return _clean(out)


def _fmt(value: float, digits: int) -> str:
    """Fixed-point text with ``digits`` decimals; empty for NaN."""
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def analysis_to_csv(analysis: RoomAnalysis) -> str:
    """Band table as CSV text: header ``CSV_COLUMNS``, one row per band.

    Formatting: band_hz ``f"{nominal:g}"``, center_hz 2 decimals, times (EDT, T20, T30,
    stds, Ts) 3 decimals, curvature and xi 1 decimal, PNR/C50/C80 2 decimals, D50 3
    decimals, NaN as empty field, flags joined with ``" "``.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    a = analysis
    for k, p in enumerate(a.band_results):
        row = [
            f"{a.bands.nominal[k]:g}",
            _fmt(a.bands.center[k], 2),
            _fmt(p.edt, 3),
            _fmt(p.t20, 3),
            _fmt(p.t30, 3),
            _fmt(a.t20_std[k], 3),
            _fmt(a.t30_std[k], 3),
            _fmt(p.curvature, 1),
            _fmt(p.fits["EDT"].xi, 1),
            _fmt(p.fits["T20"].xi, 1),
            _fmt(p.fits["T30"].xi, 1),
            _fmt(p.curve.peak_to_noise_db, 2),
            _fmt(p.c50, 2),
            _fmt(p.c80, 2),
            _fmt(p.d50, 3),
            _fmt(p.ts, 3),
            " ".join(p.flags),
        ]
        writer.writerow(row)
    return buf.getvalue()


def absorption_to_dict(result: AbsorptionResult, rounded: bool = True) -> dict[str, Any]:
    """JSON-serializable ISO 354 result (NaN -> None).

    Keys: ``software``, ``software_version``, ``standards`` (``["ISO 354:2003"]``),
    ``notes``, ``frequency_hz``, ``T1``, ``T2``, ``A1``, ``A2``, ``AT``, ``alpha``,
    ``A_obj``, ``alpha_std``, ``AT_std``, ``issues``. With ``rounded``, ``AT``/``A_obj``
    are rounded to 0.1 m2 and ``alpha`` to 0.01 (ISO 354:2003 8.3) via
    ``absorption.round_to``.
    """
    from resonus_room.core.absorption import ALPHA_RESOLUTION, AREA_RESOLUTION, round_to

    r = result
    at = r.area
    alpha = r.alpha
    aobj = r.area_per_object
    if rounded:
        at = round_to(at, AREA_RESOLUTION)
        aobj = round_to(aobj, AREA_RESOLUTION)
        alpha = round_to(alpha, ALPHA_RESOLUTION)
    out = {
        "software": APP_NAME,
        "software_version": __version__,
        "standards": ["ISO 354:2003"],
        "notes": list(NOTES),
        "frequency_hz": r.frequency,
        "T1": r.t1,
        "T2": r.t2,
        "A1": r.a1,
        "A2": r.a2,
        "AT": at,
        "alpha": alpha,
        "A_obj": aobj,
        "alpha_std": r.alpha_std,
        "AT_std": r.area_std,
        "issues": list(r.issues),
    }
    return _clean(out)


def absorption_to_csv(result: AbsorptionResult) -> str:
    """Return the ISO 354 band table as CSV text.

    ISO 354 band table as CSV: header ``ABSORPTION_COLUMNS``; T 2 decimals (8.1.1),
    areas 1 decimal (rounded per 8.3 for AT/A_obj; A1/A2 2 decimals), alpha 2 decimals,
    stds 3 decimals; NaN as empty field.
    """
    from resonus_room.core.absorption import ALPHA_RESOLUTION, AREA_RESOLUTION, round_to

    r = result
    at = round_to(r.area, AREA_RESOLUTION)
    aobj = round_to(r.area_per_object, AREA_RESOLUTION)
    alpha = round_to(r.alpha, ALPHA_RESOLUTION)
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(ABSORPTION_COLUMNS)
    for k in range(len(r.frequency)):
        row = [
            f"{r.frequency[k]:g}",
            _fmt(r.t1[k], 2),
            _fmt(r.t2[k], 2),
            _fmt(r.a1[k], 2),
            _fmt(r.a2[k], 2),
            _fmt(at[k], 1),
            _fmt(alpha[k], 2),
            _fmt(aobj[k], 1),
            _fmt(r.alpha_std[k], 3),
            _fmt(r.area_std[k], 3),
        ]
        writer.writerow(row)
    return buf.getvalue()


def analysis_to_band_results(analysis: RoomAnalysis, metadata: Metadata) -> BandResults:
    """Per-band results of a :class:`RoomAnalysis` as resonus-core ``BandResults``.

    ``fraction``, ``center`` and ``labels`` from ``analysis.bands``. ``values`` has the
    keys of ``ANALYSIS_QUANTITIES`` in that order: ``EDT``, ``T20``, ``T30``, ``C``,
    ``PNR``, ``C50``, ``C80``, ``D50``, ``Ts`` from ``analysis.values(name)``;
    ``T20_std``/``T30_std`` from ``analysis.t20_std``/``t30_std``; ``xi_X`` and ``r2_X``
    for X in ``PARAMETERS`` from ``p.fits[X].xi`` / ``p.fits[X].r2`` of each band result
    ``p``. ``units = dict(ANALYSIS_QUANTITIES)``; ``flags`` = ``p.flags`` per band.
    Metadata: ``dataclasses.replace(metadata, standards=analysis.standards,
    extra=metadata.extra | {"notes": list(NOTES), "settings": dict(analysis.settings),
    "sample_rate": analysis.sample_rate, "onset_s": analysis.onset /
    analysis.sample_rate})``.
    """
    values = {}
    for name in ANALYSIS_QUANTITIES:
        if name in ("T20_std", "T30_std"):
            values[name] = getattr(analysis, f"{name.lower()}")
        elif name in ("xi_EDT", "xi_T20", "xi_T30"):
            key = name.split("_")[1]
            values[name] = np.array([p.fits[key].xi for p in analysis.band_results])
        elif name in ("r2_EDT", "r2_T20", "r2_T30"):
            key = name.split("_")[1]
            values[name] = np.array([p.fits[key].r2 for p in analysis.band_results])
        else:
            values[name] = analysis.values(name)
    units = dict(ANALYSIS_QUANTITIES)
    flags = tuple(p.flags for p in analysis.band_results)
    meta = dataclasses.replace(
        metadata,
        standards=analysis.standards,
        extra=metadata.extra
        | {
            "notes": list(NOTES),
            "settings": dict(analysis.settings),
            "sample_rate": analysis.sample_rate,
            "onset_s": analysis.onset / analysis.sample_rate,
        },
    )
    return BandResults(
        analysis.bands.fraction,
        analysis.bands.center,
        tuple(analysis.bands.labels),
        values,
        units,
        meta,
        flags,
    )


def absorption_to_band_results(result: AbsorptionResult, metadata: Metadata) -> BandResults:
    """ISO 354 result as resonus-core ``BandResults`` (unrounded values).

    Bands ``b = band_set(3, 100, 5000)`` (the 18 bands of ISO 354:2003 clause 5):
    ``fraction = 3``, ``center = b.center``, ``labels = b.labels``. ``values`` with the
    keys of ``ABSORPTION_QUANTITIES`` in that order from ``result.t1``, ``t2``, ``a1``,
    ``a2``, ``area``, ``alpha``, ``area_per_object``, ``alpha_std``, ``area_std``;
    ``units = dict(ABSORPTION_QUANTITIES)``; no flags. Metadata:
    ``dataclasses.replace(metadata, standards=("ISO 354:2003",), extra=metadata.extra |
    {"notes": list(NOTES), "issues": list(result.issues)})``.

    Raises
    ------
    ValueError
        If ``result.frequency`` does not have 18 bands.
    """
    if len(result.frequency) != 18:
        raise ValueError("expected the 18 bands of ISO 354")
    b = band_set(3, 100, 5000)
    values = {
        "T1": result.t1,
        "T2": result.t2,
        "A1": result.a1,
        "A2": result.a2,
        "AT": result.area,
        "alpha_s": result.alpha,
        "A_obj": result.area_per_object,
        "alpha_std": result.alpha_std,
        "AT_std": result.area_std,
    }
    units = dict(ABSORPTION_QUANTITIES)
    flags = ()
    meta = dataclasses.replace(
        metadata,
        standards=("ISO 354:2003",),
        extra=metadata.extra | {"notes": list(NOTES), "issues": list(result.issues)},
    )
    return BandResults(
        b.fraction,
        b.center,
        tuple(b.labels),
        values,
        units,
        meta,
        flags,
    )


def _check_binaural(params: BinauralParameters, bands: BandSet) -> None:
    """Raise ``ValueError`` unless every IACC array has one value per band."""
    if any(len(v) != len(bands) for v in (params.early, params.late, params.full)):
        raise ValueError("one IACC value per band required")


def binaural_to_csv(params: BinauralParameters, bands: BandSet) -> str:
    """IACC table as CSV text: header ``BINAURAL_COLUMNS``, one row per band.

    band_hz ``f"{nominal:g}"``, center_hz 2 decimals (as :func:`analysis_to_csv`),
    ``params.early``, ``.late``, ``.full`` with 3 decimals (``_fmt``; design choice: the
    JND of IACC is 0.075, ISO 3382-1:2009 B.4), NaN as empty field.

    Raises
    ------
    ValueError
        ``"one IACC value per band required"`` if the lengths of ``params.early`` (or
        ``.late``, ``.full``) and ``bands`` differ.
    """
    _check_binaural(params, bands)
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(BINAURAL_COLUMNS)
    for k in range(len(bands)):
        row = [
            f"{bands.nominal[k]:g}",
            _fmt(bands.center[k], 2),
            _fmt(params.early[k], 3),
            _fmt(params.late[k], 3),
            _fmt(params.full[k], 3),
        ]
        writer.writerow(row)
    return buf.getvalue()


def binaural_to_band_results(
    params: BinauralParameters, bands: BandSet, sample_rate: float, metadata: Metadata
) -> BandResults:
    """IACC per band as resonus-core ``BandResults``.

    ``fraction``, ``center``, ``labels`` from ``bands``; ``values`` with the keys of
    ``BINAURAL_QUANTITIES`` in that order from ``params.early``, ``.late``, ``.full``
    (float arrays); ``units = dict(BINAURAL_QUANTITIES)``; no flags. Metadata:
    ``dataclasses.replace(metadata, standards=BINAURAL_STANDARDS, extra=metadata.extra |
    {"notes": list(NOTES), "sample_rate": sample_rate, "onset_s": params.onset /
    sample_rate, "early_limit_s": EARLY_LIMIT, "tau_max_s": TAU_MAX})``.

    Raises
    ------
    ValueError
        As :func:`binaural_to_csv`.
    """
    _check_binaural(params, bands)
    values = {
        "IACC_early": np.asarray(params.early, dtype=float),
        "IACC_late": np.asarray(params.late, dtype=float),
        "IACC_full": np.asarray(params.full, dtype=float),
    }
    units = dict(BINAURAL_QUANTITIES)
    meta = dataclasses.replace(
        metadata,
        standards=BINAURAL_STANDARDS,
        extra=metadata.extra
        | {
            "notes": list(NOTES),
            "sample_rate": sample_rate,
            "onset_s": params.onset / sample_rate,
            "early_limit_s": EARLY_LIMIT,
            "tau_max_s": TAU_MAX,
        },
    )
    return BandResults(
        bands.fraction,
        bands.center,
        tuple(bands.labels),
        values,
        units,
        meta,
    )
