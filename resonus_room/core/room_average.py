"""Room average over measurement positions, ISO 3382-2:2008 clause 8 a) and Annex A.

Every impulse response of a :class:`~resonus_room.core.positions.PositionSetup` is
analysed with the same settings; per band, the arithmetic mean and the sample standard
deviation over the positions are formed (clause 8 a), NaN values ignored). The standard
deviation of the mean T20/T30 follows Eq. (A.4) with N = number of distinct
source-microphone combinations and n = 10 (integrated impulse response method, A.3).
"""

from __future__ import annotations

import csv  # noqa: F401
import dataclasses  # noqa: F401
import io  # noqa: F401
import math  # noqa: F401
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path  # noqa: F401

import numpy as np
from numpy.typing import NDArray
from resonus_core.bands import BandSet
from resonus_core.data import BandResults, Metadata

from resonus_room.core.analysis import RoomAnalysis
from resonus_room.core.engine import analyze_ir, settings_band_set  # noqa: F401
from resonus_room.core.files import NOTES, load_impulse_response  # noqa: F401
from resonus_room.core.positions import (  # noqa: F401
    METHODS,
    POSITION_LABEL,
    Position,
    PositionSetup,
    check_positions,
    label_position,
)
from resonus_room.core.session import Entry
from resonus_room.core.settings import AppSettings
from resonus_room.core.uncertainty import (  # noqa: F401
    bandwidth,
    reverberation_std,
    spatial_average,
)

# Quantities averaged over positions -> unit.
AVERAGED = {"EDT": "s", "T20": "s", "T30": "s", "C50": "dB", "C80": "dB", "D50": "1", "Ts": "s"}
STANDARDS = ("ISO 3382-2:2008 4.3, 5.1, 8 a), Annex A Eq. (A.4)",)
AVERAGE_COLUMNS = (
    "band_hz", "N", "EDT_s", "EDT_std_s", "T20_s", "T20_std_s", "T20_sigma_s",
    "T30_s", "T30_std_s", "T30_sigma_s", "C50_dB", "C50_std_dB", "C80_dB", "C80_std_dB",
    "D50", "D50_std", "Ts_s", "Ts_std_s",
)  # fmt: skip


@dataclass(frozen=True)
class RoomAverage:
    """Spatial average of one room.

    Parameters
    ----------
    setup
        The positions.
    bands
        Band set analysed.
    analyses
        One analysis per position, in the order of ``setup.positions``.
    mean, std
        Quantity of ``AVERAGED`` -> mean / sample standard deviation per band.
    count
        Quantity -> number of finite values per band.
    sigma
        ``"T20"``/``"T30"`` -> standard deviation of the mean, Eq. (A.4).
    issues
        Findings of :func:`~resonus_room.core.positions.check_positions`.
    """

    setup: PositionSetup
    bands: BandSet
    analyses: tuple[RoomAnalysis, ...]
    mean: dict[str, NDArray[np.float64]]
    std: dict[str, NDArray[np.float64]]
    count: dict[str, NDArray[np.int64]]
    sigma: dict[str, NDArray[np.float64]]
    issues: tuple[str, ...]


def average_positions(setup: PositionSetup, settings: AppSettings) -> RoomAverage:
    """Analyse every position and average over them.

    1. ``bands = settings_band_set(settings)``; for each position:
       ``analyze_ir(load_impulse_response(position.file), settings)``.
    2. For each ``name`` in ``AVERAGED``: ``s = spatial_average(np.array([a.values(name)
       for a in analyses]))``; ``mean[name] = s.mean``, ``std[name] = s.std``,
       ``count[name] = s.count``.
    3. ``n = len({(p.source, p.microphone) for p in setup.positions})``; ``bw =
       bandwidth(bands.center, bands.fraction)``; ``sigma["T20"] =
       reverberation_std(mean["T20"], bw, n, 20)``, ``sigma["T30"] =
       reverberation_std(mean["T30"], bw, n, 30)``.
    4. ``issues = tuple(check_positions(setup, bands.fraction, list(bands.nominal)))``.

    Raises
    ------
    ValueError, OSError
        From loading or analysing a file.
    """
    bands = settings_band_set(settings)
    analyses = [analyze_ir(load_impulse_response(p.file), settings) for p in setup.positions]
    return average_analyses(setup, analyses, bands)


def average_analyses(
    setup: PositionSetup, analyses: Sequence[RoomAnalysis], bands: BandSet
) -> RoomAverage:
    """Average analyses that are already computed, one per position of ``setup``.

    Steps 2-4 of :func:`average_positions` with these ``analyses`` and ``bands`` (also
    used by the GUI room average, D-19).

    Raises
    ------
    ValueError
        ``"one analysis per position required"`` if the lengths differ.
    """
    if len(analyses) != len(setup.positions):
        raise ValueError("one analysis per position required")
    mean: dict[str, NDArray[np.float64]] = {}
    std: dict[str, NDArray[np.float64]] = {}
    count: dict[str, NDArray[np.int64]] = {}
    for name in AVERAGED:
        s = spatial_average(np.array([a.values(name) for a in analyses]))
        mean[name] = s.mean
        std[name] = s.std
        count[name] = s.count
    # Eq. (A.4): N = number of distinct source-microphone combinations
    n = len({(p.source, p.microphone) for p in setup.positions})
    bw = bandwidth(bands.center, bands.fraction)
    sigma = {
        "T20": reverberation_std(mean["T20"], bw, n, 20),
        "T30": reverberation_std(mean["T30"], bw, n, 30),
    }
    issues = tuple(check_positions(setup, bands.fraction, list(bands.nominal)))
    return RoomAverage(setup, bands, tuple(analyses), mean, std, count, sigma, issues)


def average_entries(entries: Sequence[Entry], method: str) -> RoomAverage:
    """Room average of session entries that are already analysed (GUI, D-19).

    1. ``ValueError("no entries")`` if ``entries`` is empty; ``f"unknown method
       {method!r}"`` if ``method`` is not in ``METHODS``.
    2. Labels: entries whose ``label`` does not match ``POSITION_LABEL``
       (``fullmatch``) -> one ``ValueError`` naming all of them: ``"not one source and one
       microphone position (s#m#): " + ", ".join(repr(e.title) for each)``.
    3. ``ValueError(f"{e.title}: not analysed")`` for the first entry with ``analysis is
       None``; ``ValueError("entries analysed with different bands")`` if the
       ``analysis.bands`` differ (``!=`` against the first entry's).
    4. ``setup = PositionSetup(method, tuple(Position(Path(e.title), *label_position(
       e.label)) for e in entries))`` (no files: the title names the position in
       messages); return ``average_analyses(setup, [e.analysis ...], bands)`` with the
       first entry's ``bands``.

    Raises
    ------
    ValueError
        As listed above.
    """
    if not entries:
        raise ValueError("no entries")
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    bad = [e.title for e in entries if POSITION_LABEL.fullmatch(e.label) is None]
    if bad:
        raise ValueError(
            "not one source and one microphone position (s#m#): " + ", ".join(repr(t) for t in bad)
        )
    first = entries[0]
    if first.analysis is None:
        raise ValueError(f"{first.title}: not analysed")
    bands = first.analysis.bands
    for e in entries[1:]:
        if e.analysis is None:
            raise ValueError(f"{e.title}: not analysed")
        if e.analysis.bands != bands:
            raise ValueError("entries analysed with different bands")
    setup = PositionSetup(
        method,
        tuple(Position(Path(e.title), *label_position(e.label)) for e in entries),
    )
    analyses = [e.analysis for e in entries if e.analysis is not None]  # all, checked above
    return average_analyses(setup, analyses, bands)


def average_to_band_results(average: RoomAverage, metadata: Metadata) -> BandResults:
    """Return the average as resonus-core ``BandResults``.

    ``values``: for each ``name`` in ``AVERAGED``: ``name`` (mean), ``f"{name}_std"``,
    ``f"{name}_n"`` (count as float); then ``"T20_sigma"``, ``"T30_sigma"``. Units: the
    unit of ``name`` for mean, std and sigma, ``"1"`` for counts. Bands from
    ``average.bands`` (``fraction``, ``center``, ``tuple(labels)``), no flags. Metadata:
    ``dataclasses.replace(metadata, standards=STANDARDS, extra=metadata.extra |
    {"notes": list(NOTES), "method": average.setup.method, "issues":
    list(average.issues), "positions": [{"file": p.file.name, "source": p.source,
    "microphone": p.microphone} for p in average.setup.positions]})``.
    """
    # Prepare values and units
    values: dict[str, NDArray[np.float64]] = {}
    units: dict[str, str] = {}
    for name in AVERAGED:
        values[name] = average.mean[name]
        values[f"{name}_std"] = average.std[name]
        values[f"{name}_n"] = average.count[name].astype(float)
        units[name] = AVERAGED[name]
        units[f"{name}_std"] = AVERAGED[name]
        units[f"{name}_n"] = "1"
    values["T20_sigma"] = average.sigma["T20"]
    values["T30_sigma"] = average.sigma["T30"]
    units["T20_sigma"] = AVERAGED["T20"]
    units["T30_sigma"] = AVERAGED["T30"]

    # Build metadata with extra info
    extra = dict(metadata.extra)
    extra["notes"] = list(NOTES)
    extra["method"] = average.setup.method
    extra["issues"] = list(average.issues)
    extra["positions"] = [
        {"file": p.file.name, "source": p.source, "microphone": p.microphone}
        for p in average.setup.positions
    ]
    new_meta = dataclasses.replace(metadata, standards=STANDARDS, extra=extra)

    b = average.bands
    return BandResults(b.fraction, b.center, tuple(b.labels), values, units, new_meta)


def average_to_csv(average: RoomAverage) -> str:
    """Band table as CSV text: header ``AVERAGE_COLUMNS``, one row per band.

    ``band_hz`` ``f"{nominal:g}"``; ``N`` = ``count["T20"]``; times and their std/sigma
    3 decimals, C50/C80 and std 2 decimals, D50 and std 3 decimals; NaN as empty field.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")

    def fmt(x: float, d: int) -> str:
        return "" if not math.isfinite(x) else f"{x:.{d}f}"

    writer.writerow(AVERAGE_COLUMNS)
    for k, nominal in enumerate(average.bands.nominal):
        row = [
            f"{nominal:g}",
            str(int(average.count["T20"][k])),
            fmt(average.mean["EDT"][k], 3),
            fmt(average.std["EDT"][k], 3),
            fmt(average.mean["T20"][k], 3),
            fmt(average.std["T20"][k], 3),
            fmt(average.sigma["T20"][k], 3),
            fmt(average.mean["T30"][k], 3),
            fmt(average.std["T30"][k], 3),
            fmt(average.sigma["T30"][k], 3),
            fmt(average.mean["C50"][k], 2),
            fmt(average.std["C50"][k], 2),
            fmt(average.mean["C80"][k], 2),
            fmt(average.std["C80"][k], 2),
            fmt(average.mean["D50"][k], 3),
            fmt(average.std["D50"][k], 3),
            fmt(average.mean["Ts"][k], 3),
            fmt(average.std["Ts"][k], 3),
        ]
        writer.writerow(row)
    return buf.getvalue()
