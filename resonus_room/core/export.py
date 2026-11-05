"""Export of the current results, format chosen by the file extension (one Export action).

Like Inkscape's "Save as", the user names the file and the extension selects what is
written:

- ``.png``, ``.svg``, ``.pdf``: the plot of the current tab (matplotlib ``savefig``).
- ``.json``: full analysis (:func:`~resonus_room.core.files.analysis_to_dict` plus
  ``source`` and ``device``).
- ``.csv``: band table (:func:`~resonus_room.core.files.analysis_to_csv`).
- ``.ir.h5``: the impulse response in the resonus HDF5 format with its full metadata
  (software and version, time, device, all measurement and analysis settings); the app
  opens it again. Structure: resonus-core docs/FILE_FORMAT.md (user decision D-16).
- ``.h5`` (any other name): band results (resonus-core ``BandResults``, readable by the
  viewer; the same content as ``analysis.h5`` of the CLI).
- ``.wav``: the impulse response as 32-bit float WAV for other tools (no rescaling;
  metadata is not kept, D-16).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from resonus_core import data
from resonus_core.data import ImpulseResponse, Metadata
from resonus_core.plot import MplPlotView
from scipy.io import wavfile

from resonus_room.core.analysis import RoomAnalysis
from resonus_room.core.files import (
    analysis_to_band_results,
    analysis_to_csv,
    analysis_to_dict,
)

# Supported file extensions (lower case) -> description shown in the file dialog.
EXPORT_FORMATS = {
    ".png": "Plot of the current tab (PNG)",
    ".svg": "Plot of the current tab (SVG)",
    ".pdf": "Plot of the current tab (PDF)",
    ".json": "Analysis (JSON)",
    ".csv": "Band table (CSV)",
    ".ir.h5": "Impulse response with metadata (HDF5)",
    ".h5": "Band results (HDF5)",
    ".wav": "Impulse response (32-bit float WAV)",
}
PLOT_SUFFIXES = (".png", ".svg", ".pdf")
# Double extension of the impulse-response export (user decision D-16, 2026-11-04).
IR_SUFFIX = ".ir.h5"


def export_results(
    path: str | Path,
    analysis: RoomAnalysis,
    ir: ImpulseResponse,
    view: MplPlotView,
    metadata: Metadata,
) -> Path:
    """Write the results to ``path`` in the format given by its extension.

    ``suffix = IR_SUFFIX`` if ``Path(path).name.lower().endswith(IR_SUFFIX)``, else
    ``Path(path).suffix.lower()``:

    - ``IR_SUFFIX``: ``data.save(path, ir)`` (the impulse response with its own metadata,
      unchanged);
    - in ``PLOT_SUFFIXES``: ``view.save(str(path))``;
    - ``".json"``: ``analysis_to_dict(analysis) | {"source": metadata.description,
      "device": metadata.device}`` written with ``json.dumps(..., indent=2,
      allow_nan=False)``, UTF-8;
    - ``".csv"``: ``analysis_to_csv(analysis)``, UTF-8;
    - ``".h5"``: ``data.save(path, analysis_to_band_results(analysis, metadata))``;
    - ``".wav"``: ``wavfile.write(path, int(round(ir.sample_rate)),
      ir.data.astype(np.float32))``.

    Parameters
    ----------
    path
        Target file; overwritten if it exists.
    analysis, ir
        Current results.
    view
        Plot of the current tab (used for plot formats only).
    metadata
        Software, version, description (source) and device for JSON and HDF5.

    Returns
    -------
    Path
        ``Path(path)``.

    Raises
    ------
    ValueError
        If the extension is not in ``EXPORT_FORMATS``; the message is ``f"unsupported file
        extension {suffix!r}; use one of {', '.join(EXPORT_FORMATS)}"``.
    """
    p = Path(path)
    suffix = IR_SUFFIX if p.name.lower().endswith(IR_SUFFIX) else p.suffix.lower()
    if suffix not in EXPORT_FORMATS:
        raise ValueError(
            f"unsupported file extension {suffix!r}; use one of {', '.join(EXPORT_FORMATS)}"
        )
    if suffix == IR_SUFFIX:
        data.save(p, ir)
    elif suffix in PLOT_SUFFIXES:
        view.save(str(p))
    elif suffix == ".json":
        data_dict = analysis_to_dict(analysis)
        data_dict.update({"source": metadata.description, "device": metadata.device})
        with p.open("w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=2, allow_nan=False)
    elif suffix == ".csv":
        with p.open("w", encoding="utf-8") as f:
            f.write(analysis_to_csv(analysis))
    elif suffix == ".h5":
        data.save(p, analysis_to_band_results(analysis, metadata))
    elif suffix == ".wav":
        wavfile.write(p, int(round(ir.sample_rate)), ir.data.astype(np.float32))
    return p


# Formats that hold one impulse response; with several channels one file per channel is
# written (user decision D-17).
IR_FORMATS = (IR_SUFFIX, ".wav")


def export_suffix(path: str | Path) -> str:
    """Return ``IR_SUFFIX`` if the name ends with it (case-insensitive), else the last suffix."""
    path = Path(path)
    return IR_SUFFIX if path.name.lower().endswith(IR_SUFFIX) else path.suffix.lower()


def channel_paths(path: str | Path, labels: Sequence[str]) -> list[Path]:
    """File names for one impulse response per channel: ``name_<label>`` + suffix.

    One label: ``[Path(path)]`` unchanged. Several: the suffix is ``IR_SUFFIX`` if the
    name ends with it (case-insensitive), else ``Path(path).suffix``; with ``stem`` the
    name without that suffix (original case kept), each path is
    ``p.with_name(f"{stem}_{label}{suffix}")`` where ``suffix`` keeps the original case
    of the given name. Example: ``room.ir.h5``, ``["s1m1L", "s1m1R"]`` ->
    ``room_s1m1L.ir.h5``, ``room_s1m1R.ir.h5``.

    Raises
    ------
    ValueError
        ``"no channels"`` if ``labels`` is empty; ``f"duplicate channel label
        {label!r}"`` if a label repeats (files would overwrite each other).
    """
    if not labels:
        raise ValueError("no channels")
    seen: set[str] = set()
    for label in labels:
        if label in seen:
            raise ValueError(f"duplicate channel label {label!r}")
        seen.add(label)
    if len(labels) == 1:
        return [Path(path)]
    p = Path(path)
    suffix = export_suffix(p)
    n = len(IR_SUFFIX) if suffix == IR_SUFFIX else len(p.suffix)
    stem = p.name[: len(p.name) - n]
    suffix = p.name[len(p.name) - n :]
    return [p.with_name(f"{stem}_{label}{suffix}") for label in labels]


def export_channels(
    path: str | Path, channels: Sequence[tuple[str, ImpulseResponse]]
) -> list[Path]:
    """Write the impulse response of every channel, one file each.

    The format comes from the extension as in :func:`export_results` and must be one of
    ``IR_FORMATS`` (``.ir.h5``: ``data.save`` with the IR's own metadata; ``.wav``:
    float32 as in :func:`export_results`). File names from :func:`channel_paths` with the
    labels of ``channels``.

    Returns
    -------
    list of Path
        The files written, in channel order.

    Raises
    ------
    ValueError
        ``f"one file per channel is written for {', '.join(IR_FORMATS)} only, not
        {suffix!r}"`` for other extensions; errors of :func:`channel_paths`.
    """
    suffix = export_suffix(path)
    if suffix not in IR_FORMATS:
        raise ValueError(
            f"one file per channel is written for {', '.join(IR_FORMATS)} only, not {suffix!r}"
        )
    paths = channel_paths(path, [label for label, _ in channels])
    for q, (_, ir) in zip(paths, channels, strict=True):
        if suffix == IR_SUFFIX:
            data.save(q, ir)
        else:  # .wav
            wavfile.write(q, int(round(ir.sample_rate)), ir.data.astype(np.float32))
    return paths
