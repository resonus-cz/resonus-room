"""Export of the STI at a position: JSON, CSV and resonus-core ``BandResults`` (D-31).

Every export carries :data:`METHOD_NOTE` and the app's :data:`~resonus_room.core.files.NOTES`
(PROVISIONAL: original STI, not IEC 60268-16:2020, Q-STI1).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Final

import numpy as np
from resonus_core.bands import band_set
from resonus_core.data import BandResults, Metadata

from resonus_room.core import files
from resonus_room.core.sti import MODULATION_FREQUENCIES, OCTAVE_BANDS
from resonus_room.core.sti_inputs import StiReport

# Statement on the method (docs/PLAN_STI.md draft 3).
METHOD_NOTE: Final = (
    "PROVISIONAL: original STI (Steeneken & Houtgast 1980; Houtgast & Steeneken 1985, "
    "sec. II B), modulation transfer of the compound signal (route b); "
    "not IEC 60268-16:2020 (STIr, masking and reception threshold not applied)."
)
# BandResults quantities: name -> unit.
STI_QUANTITIES: Final = {"MTI": "1", "speech_level_1m": "dB", "noise_level": "dB"}
CSV_HEADER: Final = "band_hz,MTI,speech_level_1m_dB,noise_level_dB"


def sti_to_dict(report: StiReport) -> dict[str, Any]:
    """Return the report as a JSON-ready dict.

    Keys in this order: ``"STI"`` (float), ``"bands_hz"`` (``[int(f) for f in
    OCTAVE_BANDS]``), ``"MTI"``, ``"speech_level_1m_dB"``, ``"noise_level_dB"`` (lists of
    floats; a non-finite noise level becomes None), ``"modulation_frequencies_hz"``
    (``list(MODULATION_FREQUENCIES)``), ``"m"`` (7 lists of 14 floats), ``"seed"``,
    ``"method"`` (:data:`METHOD_NOTE`), ``"notes"`` (``list(files.NOTES)``). Values are
    not rounded.
    """
    # Convert noise levels: non-finite values become None
    noise_levels = [float(v) if np.isfinite(v) else None for v in report.noise_levels_db]
    # Convert modulation matrix to plain lists of floats
    m_lists = [[float(v) for v in row] for row in report.m]
    return {
        "STI": float(report.sti),
        "bands_hz": [int(f) for f in OCTAVE_BANDS],
        "MTI": [float(v) for v in report.mti],
        "speech_level_1m_dB": [float(v) for v in report.speech_levels_db],
        "noise_level_dB": noise_levels,
        "modulation_frequencies_hz": list(MODULATION_FREQUENCIES),
        "m": m_lists,
        "seed": int(report.seed),
        "method": METHOD_NOTE,
        "notes": list(files.NOTES),
    }


def sti_to_csv(report: StiReport) -> str:
    """Return the per-band table as CSV, then a summary line.

    Line 1 :data:`CSV_HEADER`; one line per band ``f"{int(f)},{mti:.3f},{speech:.1f},
    {noise}"`` with ``noise`` formatted ``.1f``, or empty if not finite; last line
    ``f"STI,{sti:.3f},,"``. Every line, the last included, ends with a newline.
    """
    lines = [CSV_HEADER]
    for f, mti, speech, noise in zip(
        OCTAVE_BANDS, report.mti, report.speech_levels_db, report.noise_levels_db, strict=True
    ):
        noise_str = f"{noise:.1f}" if np.isfinite(noise) else ""
        lines.append(f"{int(f)},{mti:.3f},{speech:.1f},{noise_str}")
    lines.append(f"STI,{report.sti:.3f},,")
    return "\n".join(lines) + "\n"


def sti_to_band_results(report: StiReport, metadata: Metadata) -> BandResults:
    """Return the per-band values as ``BandResults`` (octave bands 125 Hz-8 kHz).

    ``bands = resonus_core.bands.band_set(1, 125.0, 8000.0)``; ``BandResults(1,
    bands.center, tuple(bands.labels), values, dict(STI_QUANTITIES),
    meta)`` with ``values = {"MTI": report.mti, "speech_level_1m":
    report.speech_levels_db, "noise_level": noise}`` (copies, float; non-finite noise
    levels as NaN) and ``meta = dataclasses.replace(metadata, extra=metadata.extra |
    {"STI": report.sti, "seed": report.seed, "method": METHOD_NOTE, "notes":
    list(files.NOTES)})``.
    """
    bands = band_set(1, 125.0, 8000.0)
    noise = np.where(np.isfinite(report.noise_levels_db), report.noise_levels_db, np.nan)
    values = {
        "MTI": np.array(report.mti, dtype=np.float64),
        "speech_level_1m": np.array(report.speech_levels_db, dtype=np.float64),
        "noise_level": noise.astype(np.float64),
    }
    meta = dataclasses.replace(
        metadata,
        extra=metadata.extra
        | {
            "STI": report.sti,
            "seed": report.seed,
            "method": METHOD_NOTE,
            "notes": list(files.NOTES),
        },
    )
    return BandResults(1, bands.center, tuple(bands.labels), values, dict(STI_QUANTITIES), meta)
