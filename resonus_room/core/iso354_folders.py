"""ISO 354 evaluation from two folders of impulse responses (D-32).

The ISO 354 result needs two room averages, of the empty reverberation room (T1) and with
the specimen (T2) (ISO 354:2003 8.1.1, :mod:`core.iso354`). Workflow (user
request 2026-11-06): one folder per condition; every impulse response in it is one
position. A folder may hold

- impulse-response files at the top level: ``.wav`` (REW exports such as ``" s1m 1.wav"``,
  first channel) and resonus ``.h5`` files that hold an ``ImpulseResponse`` without a
  ``role`` (so ``measure --out`` folders work: ``analysis.h5``, ``frequency_response.h5``
  and ``source_calibration.h5`` are skipped), and
- channel subfolders with :data:`~resonus_room.core.positions.CHANNEL_FILE`
  (``measure --out`` of a multichannel run, one subfolder per channel label, D-17).

Position labels ``s#m#`` come from the file names (:func:`label_from_filename`) or the
subfolder names; with labels for every file the number of microphone and source
positions is checked (ISO 354:2003 7.1.4, :func:`check_positions`). The evaluated setup
can be written as a setup file for ``resonus-room absorption`` (:func:`setup_to_toml`),
so a GUI evaluation is reproducible from the command line.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from resonus_core import data
from resonus_core.data import ImpulseResponse

from resonus_room.core.absorption import EMPTY_NAME, SPECIMEN_NAME, Climate, check_positions
from resonus_room.core.iso354 import AbsorptionSetup
from resonus_room.core.positions import (
    CHANNEL_FILE,
    POSITION_LABEL,
    label_from_filename,
    label_position,
)

# Top-level file types read as impulse responses (design choice D-32).
IR_SUFFIXES: Final = (".wav", ".h5")
# Names of the two conditions in notes and errors (defined with the climate findings).


@dataclass(frozen=True)
class FolderPositions:
    """Impulse-response files of one folder, one per position.

    Parameters
    ----------
    files
        Paths, top-level files first (sorted by name), then the ``CHANNEL_FILE`` of each
        channel subfolder (sorted by subfolder name).
    labels
        Position label ``s#m#`` of each file, or None if its name has none.
    """

    files: tuple[Path, ...]
    labels: tuple[str | None, ...]


def folder_positions(folder: str | Path) -> FolderPositions:
    """Collect the impulse responses of ``folder`` (rules in the module docstring).

    Entries whose names start with ``"."`` are ignored. Top-level files (``is_file``,
    sorted by name) with a suffix in :data:`IR_SUFFIXES` (case-insensitive): a ``.wav`` is
    always taken; a ``.h5`` is loaded with ``resonus_core.data.load`` and taken only if
    it is an ``ImpulseResponse`` and ``not obj.metadata.extra.get("role")``. Label:
    ``label_from_filename(path.name)``. Then every subfolder (sorted by name) that
    contains ``CHANNEL_FILE``: the file ``sub / CHANNEL_FILE``, label ``sub.name`` if
    ``POSITION_LABEL.fullmatch(sub.name)``, else None. Other entries are ignored.

    Raises
    ------
    ValueError
        ``f"{folder}: not a folder"`` unless it is a directory;
        ``f"{folder}: no impulse responses (.wav, .h5 or <label>/{CHANNEL_FILE})"`` if
        nothing was taken; ``f"{folder}: several files for position(s) {names}: one file
        per position (put different rooms in separate folders)"`` if a label (not None)
        occurs more than once, ``names`` the sorted repeated labels joined by ``", "``;
        ``f"{path}: {exc}"`` (from the ``ValueError`` or ``OSError`` ``exc``) if a ``.h5``
        file cannot be loaded.
    """
    p = Path(folder)
    if not p.is_dir():
        raise ValueError(f"{p}: not a folder")
    files: list[Path] = []
    labels: list[str | None] = []

    # Top‑level impulse‑response files
    for entry in sorted(p.iterdir()):
        if entry.is_file() and not entry.name.startswith("."):
            suffix = entry.suffix.lower()
            if suffix in IR_SUFFIXES:
                if suffix == ".wav":
                    files.append(entry)
                    labels.append(label_from_filename(entry.name))
                elif suffix == ".h5":
                    try:
                        obj = data.load(entry)
                    except (ValueError, OSError) as exc:
                        raise ValueError(f"{entry}: {exc}") from exc
                    if isinstance(obj, ImpulseResponse) and not obj.metadata.extra.get("role"):
                        files.append(entry)
                        labels.append(label_from_filename(entry.name))

    # Channel subfolders
    for sub in sorted(p.iterdir()):
        if sub.is_dir() and not sub.name.startswith("."):
            channel_file = sub / CHANNEL_FILE
            if channel_file.is_file():
                files.append(channel_file)
                labels.append(sub.name if POSITION_LABEL.fullmatch(sub.name) else None)

    if not files:
        raise ValueError(f"{p}: no impulse responses (.wav, .h5 or <label>/{CHANNEL_FILE})")

    # Detect duplicate labels
    label_counts: dict[str, int] = {}
    for lbl in labels:
        if lbl is not None:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
    duplicates = sorted([lbl for lbl, cnt in label_counts.items() if cnt > 1])
    if duplicates:
        raise ValueError(
            f"{p}: several files for position(s) {', '.join(duplicates)}: one file per "
            "position (put different rooms in separate folders)"
        )

    return FolderPositions(tuple(files), tuple(labels))


def position_counts(labels: Sequence[str | None]) -> tuple[int, int] | None:
    """Numbers of distinct microphone and source positions: ``(microphones, sources)``.

    From ``label_position(label)`` of every label; None if ``labels`` is empty or any
    label is None.
    """
    if not labels or any(label is None for label in labels):
        return None
    microphones: set[str] = set()
    sources: set[str] = set()
    for lbl in labels:
        assert lbl is not None
        src, mic = label_position(lbl)  # (source, microphone), e.g. ("s1", "m2")
        microphones.add(mic)
        sources.add(src)
    return len(microphones), len(sources)


def setup_from_folders(
    empty_folder: str | Path,
    specimen_folder: str | Path,
    volume: float,
    empty_climate: Climate,
    specimen_climate: Climate,
    specimen_area: float | None = None,
    n_objects: int | None = None,
    max_length: float | None = None,
) -> tuple[AbsorptionSetup, list[str]]:
    """ISO 354 setup from the two folders, and notes on the position check.

    ``e = folder_positions(empty_folder)``, ``s = folder_positions(specimen_folder)``;
    ``ce, cs = position_counts(e.labels), position_counts(s.labels)``.

    - If ``ce is not None and ce == cs``: ``microphones, sources = ce`` in the setup (the
      check then runs in ``run_absorption``), notes ``[]``.
    - Otherwise the setup has ``microphones = sources = None`` and the notes are, for the
      empty room then the specimen (names :data:`EMPTY_NAME`, :data:`SPECIMEN_NAME`):
      with counts ``(m, n)``: ``f"{name}: {issue}"`` for each issue of
      ``check_positions(m, n)``; without counts: ``f"{name}: positions not checked (ISO
      354:2003 7.1.4): {k} of {total} file(s) without an s#m# label"``.
    - Then, if both folders have labels for every file and the label sets differ (the
      normal case is the same positions in both conditions; design choice D-32, not a
      clause): ``f"positions differ between the conditions (only {EMPTY_NAME}: {a};
      only {SPECIMEN_NAME}: {b})"``, ``a``/``b`` the sorted labels of one set only
      joined by ``", "``, or ``"-"`` if none.

    The setup: ``AbsorptionSetup(volume, e.files, empty_climate, s.files,
    specimen_climate, specimen_area, n_objects, max_length, microphones, sources)``.

    Raises
    ------
    ValueError
        ``"give exactly one of specimen_area and n_objects"`` (checked first); from
        :func:`folder_positions`.
    """
    if (specimen_area is None) == (n_objects is None):
        raise ValueError("give exactly one of specimen_area and n_objects")

    e = folder_positions(empty_folder)
    s = folder_positions(specimen_folder)

    ce = position_counts(e.labels)
    cs = position_counts(s.labels)

    notes: list[str] = []
    microphones: int | None = None
    sources: int | None = None
    if ce is not None and ce == cs:
        microphones, sources = ce
    else:
        # Empty room
        if ce is not None:
            for issue in check_positions(*ce):
                notes.append(f"{EMPTY_NAME}: {issue}")
        else:
            notes.append(
                f"{EMPTY_NAME}: positions not checked (ISO 354:2003 7.1.4): "
                f"{sum(x is None for x in e.labels)} of {len(e.files)} file(s) "
                "without an s#m# label"
            )
        # Specimen
        if cs is not None:
            for issue in check_positions(*cs):
                notes.append(f"{SPECIMEN_NAME}: {issue}")
        else:
            notes.append(
                f"{SPECIMEN_NAME}: positions not checked (ISO 354:2003 7.1.4): "
                f"{sum(x is None for x in s.labels)} of {len(s.files)} file(s) "
                "without an s#m# label"
            )

    # Compare label sets if all labels present
    if ce is not None and cs is not None:
        set_e = {x for x in e.labels if x is not None}
        set_s = {x for x in s.labels if x is not None}
        if set_e != set_s:
            only_e = sorted(set_e - set_s)
            only_s = sorted(set_s - set_e)
            a = ", ".join(only_e) if only_e else "-"
            b = ", ".join(only_s) if only_s else "-"
            notes.append(
                f"positions differ between the conditions (only {EMPTY_NAME}: {a}; "
                f"only {SPECIMEN_NAME}: {b})"
            )

    setup = AbsorptionSetup(
        volume,
        e.files,
        empty_climate,
        s.files,
        specimen_climate,
        specimen_area,
        n_objects,
        max_length,
        microphones,
        sources,
    )
    return setup, notes


def setup_to_toml(setup: AbsorptionSetup, path: str | Path) -> None:
    """Write ``setup`` as a setup file of ``resonus-room absorption`` (``load_setup``).

    Keys in this order, each on its own line as ``key = value``: ``volume``, then
    ``specimen_area``, ``n_objects``, ``max_length``, ``microphones``, ``sources`` where
    not None; then the tables ``[empty]`` and ``[specimen]`` with ``temperature``,
    ``relative_humidity``, ``pressure`` and ``files``. Floats are written with ``repr``
    (exact round trip), integers with ``str``, strings with ``json.dumps`` (a valid TOML
    basic string), ``files`` as a list on one line. File paths are relative to the
    setup file's folder (``os.path.relpath(f, Path(path).parent)``, written with ``/``
    as separator), or absolute if no relative path exists (``relpath`` raises
    ``ValueError``). UTF-8, final newline.
    """
    out = Path(path)

    def num(value: float | int) -> str:
        return str(value) if isinstance(value, int) else repr(float(value))

    def rel(f: Path) -> str:
        try:
            return Path(os.path.relpath(f, out.parent)).as_posix()
        except ValueError:  # no relative path (e.g. another drive)
            return Path(f).as_posix()

    lines = [f"volume = {num(setup.volume)}"]
    for key in ("specimen_area", "n_objects", "max_length", "microphones", "sources"):
        value = getattr(setup, key)
        if value is not None:
            lines.append(f"{key} = {num(value)}")
    for table, climate, files in (
        ("empty", setup.empty_climate, setup.empty_files),
        ("specimen", setup.specimen_climate, setup.specimen_files),
    ):
        lines += [
            f"[{table}]",
            f"temperature = {num(climate.temperature)}",
            f"relative_humidity = {num(climate.relative_humidity)}",
            f"pressure = {num(climate.pressure)}",
            f"files = {json.dumps([rel(f) for f in files], ensure_ascii=False)}",
        ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "EMPTY_NAME",
    "IR_SUFFIXES",
    "SPECIMEN_NAME",
    "FolderPositions",
    "folder_positions",
    "position_counts",
    "setup_from_folders",
    "setup_to_toml",
]
