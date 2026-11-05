"""Import of REW (Room EQ Wizard) RT60 text exports, for cross-checks (docs/VERIFICATION.md).

REW "RT60 data" export ("File > Export > RT60 data as text"; format line of REW V5.31):
header lines ``key: value``, a line starting with ``Format is`` naming the columns, then
one comma-separated row per band: ``freq (Hz), BW (octaves), EDT (s), r, T20 (s), r, T30
(s), r, Topt (s), r, ToptStart (dB), ToptEnd (dB), T60M (s), reverse/forward/zero phase
filtered, C50 (dB), C80 (dB), D50 (%), TS (s)``. Octave rows have ``BW = 1/1``, one-third
octave rows ``1/3``; the row with ``freq = full`` is the broadband result. The file is
parsed as plain text (never evaluated). Implemented from the sample exports in
refs/rew-test-data (user, 2026-11-03); REW documents no format specification.
"""

from __future__ import annotations

import math  # noqa: F401
from dataclasses import dataclass
from pathlib import Path

import numpy as np  # noqa: F401
from resonus_core.bands import (  # noqa: F401
    band_index,
    exact_center,
    frequency_label,
    nominal_frequency,
)
from resonus_core.data import BandResults, Metadata  # noqa: F401

# Columns of a band row (0-based) -> quantity name and unit. "r" columns are the
# correlation coefficients REW reports for each fit. D50 is converted from percent to a
# ratio (ISO 3382-1:2009 A.2.3 defines D50 as a ratio).
REW_COLUMNS = {
    2: ("EDT", "s"),
    3: ("r_EDT", "1"),
    4: ("T20", "s"),
    5: ("r_T20", "1"),
    6: ("T30", "s"),
    7: ("r_T30", "1"),
    8: ("Topt", "s"),
    9: ("r_Topt", "1"),
    10: ("Topt_start", "dB"),
    11: ("Topt_end", "dB"),
    12: ("T60M", "s"),
    14: ("C50", "dB"),
    15: ("C80", "dB"),
    16: ("D50", "1"),
    17: ("Ts", "s"),
}
N_COLUMNS = 18
FILTER_COLUMN = 13


@dataclass(frozen=True)
class RewRT60:
    """Content of one REW RT60 export.

    Parameters
    ----------
    header
        ``key: value`` lines before the format line (keys as in the file, e.g.
        ``"Measurement"``, ``"Dated"``, ``"Source"``); the first line is stored under
        ``"Saved by"`` (e.g. ``"REW V5.31.3"``).
    octave, third
        Octave and one-third-octave results (None if the file has no such rows).
    broadband
        Quantity -> value of the ``full`` row (empty if absent).
    filtering
        Filter type REW reports (e.g. ``"Zero Phase"``), from the first band row.
    """

    header: dict[str, str]
    octave: BandResults | None
    third: BandResults | None
    broadband: dict[str, float]
    filtering: str


def parse_value(text: str) -> float:
    """Float value of one field; empty fields and ``"NaN"``/``"-"`` give NaN.

    Raises
    ------
    ValueError
        If the text is neither a number nor one of the empty markers.
    """
    s = text.strip()
    if s in ("", "-") or s.lower() == "nan":
        return math.nan
    return float(s)


def read_rew_rt60(path: str | Path) -> RewRT60:
    """Read a REW RT60 text export.

    1. Read the file as UTF-8 (``errors="replace"``); strip each line.
    2. Header: the first line ``"RT60 data saved by X"`` -> ``header["Saved by"] = X``;
       every following line before the ``Format is`` line that contains ``": "`` is
       split at the first ``": "`` into key and value. ValueError if the first line does
       not start with ``"RT60 data saved by"`` or there is no ``Format is`` line.
    3. Rows: every non-blank line after the format line must have ``N_COLUMNS``
       comma-separated fields (stripped), else ValueError. Field 1 is ``"1/1"`` (octave)
       or ``"1/3"`` (third), else ValueError. Values via :func:`parse_value` for the
       columns of ``REW_COLUMNS``; D50 divided by 100. ``filtering`` = field
       ``FILTER_COLUMN`` of the first row.
    4. Field 0 ``"full"``: ``broadband = {name: value}``. Otherwise the nominal
       frequency ``f = float(field 0)``; per resolution, collect rows in file order.
    5. Per resolution with rows: ``b = 1`` or ``3``; ``indices = band_index(f, b)``;
       ValueError unless ``nominal_frequency(indices, b) == f`` for every row (REW and
       our nominal labels must agree) and the indices are strictly increasing.
       Bands need not be consecutive. ``BandResults(b, exact_center(indices, b),
       tuple(frequency_label(float(x)) for x in nominal_frequency(indices, b)), values,
       units, meta)`` with values (float arrays, one per quantity) in the order
       of ``REW_COLUMNS``, ``units`` from ``REW_COLUMNS`` and ``meta = Metadata("REW",
       header.get("Saved by", ""), description=header.get("Measurement", ""),
       device=header.get("Source", ""), extra={"dated": header.get("Dated", ""),
       "filtering": filtering, "file": Path(path).name})``.

    Raises
    ------
    ValueError
        On a malformed file (see the steps).
    OSError
        If the file cannot be read.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    lines = [ln.strip() for ln in lines]
    if not lines:
        raise ValueError("Empty file")
    first = lines[0]
    if not first.startswith("RT60 data saved by"):
        raise ValueError("Missing header start")
    header: dict[str, str] = {"Saved by": first[len("RT60 data saved by ") :]}
    try:
        format_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Format is"))
    except StopIteration as err:
        raise ValueError("Missing format line") from err
    for ln in lines[1:format_idx]:
        if ": " in ln:
            key, value = ln.split(": ", 1)
            header[key] = value
    rows = lines[format_idx + 1 :]
    broadband: dict[str, float] = {}
    res_rows: dict[int, list[tuple[float, dict[str, float]]]] = {1: [], 3: []}
    filtering: str | None = None
    for ln in rows:
        if not ln:
            continue
        fields = [f.strip() for f in ln.split(",")]
        if len(fields) != N_COLUMNS:
            raise ValueError(f"Expected {N_COLUMNS} columns, got {len(fields)}")
        if filtering is None:
            filtering = fields[FILTER_COLUMN]
        if fields[0] == "full":
            for idx, (name, _) in REW_COLUMNS.items():
                val = parse_value(fields[idx])
                if name == "D50":
                    val /= 100.0
                broadband[name] = val
            continue
        f = float(fields[0])
        res = 1 if fields[1] == "1/1" else 3 if fields[1] == "1/3" else None
        if res is None:
            raise ValueError(f"Unknown resolution {fields[1]}")
        row_vals: dict[str, float] = {}
        for idx, (name, _) in REW_COLUMNS.items():
            val = parse_value(fields[idx])
            if name == "D50":
                val /= 100.0
            row_vals[name] = val
        res_rows[res].append((f, row_vals))

    def build_band(b: int) -> BandResults | None:
        rows_b = res_rows[b]
        if not rows_b:
            return None
        freqs = [f for f, _ in rows_b]
        indices = band_index(freqs, b)
        nominal = nominal_frequency(indices, b)
        if any(abs(n - f) > 1e-6 for n, f in zip(nominal, freqs, strict=False)):
            raise ValueError("Nominal frequencies do not match")
        if any(indices[i] >= indices[i + 1] for i in range(len(indices) - 1)):
            raise ValueError("Band indices not strictly increasing")
        values: dict[str, np.ndarray] = {}
        units: dict[str, str] = {}
        for _idx, (name, unit) in REW_COLUMNS.items():
            values[name] = np.array([row[1][name] for row in rows_b], dtype=float)
            units[name] = unit
        meta = Metadata(
            "REW",
            header.get("Saved by", ""),
            description=header.get("Measurement", ""),
            device=header.get("Source", ""),
            extra={
                "dated": header.get("Dated", ""),
                "filtering": filtering or "",
                "file": Path(path).name,
            },
        )
        return BandResults(
            b,
            exact_center(indices, b),
            tuple(frequency_label(float(x)) for x in nominal),
            values,
            units,
            meta,
        )

    octave = build_band(1)
    third = build_band(3)
    if filtering is None:
        filtering = ""
    return RewRT60(header, octave, third, broadband, filtering)
