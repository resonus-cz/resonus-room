"""ISO 354 workflow: impulse responses -> mean T1, T2 -> absorption.

ISO 354 workflow: measured impulse responses of the empty room and with the specimen
-> mean reverberation times T1, T2 -> absorption (``core.absorption``).

Setup file (TOML), paths relative to the file::

    volume = 210.0            # m3
    specimen_area = 10.8      # m2, plane absorber / array (or: n_objects = 4)
    max_length = 10.5         # m, optional (ISO 354 6.1.2 check)
    microphones = 3           # optional, with sources (ISO 354 7.1.4 check)
    sources = 2
    [empty]
    temperature = 20.0        # degC
    relative_humidity = 50.0  # %
    pressure = 101.325        # kPa, optional
    files = ["empty/m1s1.wav", "empty/m2s1.wav"]
    [specimen]
    temperature = 20.5
    relative_humidity = 48.0
    files = ["specimen/m1s1.wav"]

Each IR is analysed in the one-third-octave bands 100 Hz-5 kHz (ISO 354:2003 clause 5);
T is T20 (evaluation from -5 dB over 20 dB, ISO 354:2003 7.4.1); T1/T2 are arithmetic
means over the positions (8.1.1). Standard uncertainty of each mean: sample standard
deviation / sqrt(N) (confirmed by the user 2026-11-03, docs/DECISIONS.md D-11).
"""

from __future__ import annotations

import tomllib  # noqa: F401
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from resonus_room.core.absorption import (
    ISO354_FREQUENCIES,  # noqa: F401
    AbsorptionResult,
    Climate,
    absorption,  # noqa: F401
    check_positions,  # noqa: F401
    check_room,  # noqa: F401
)
from resonus_room.core.engine import analyze_ir  # noqa: F401
from resonus_room.core.files import load_impulse_response  # noqa: F401
from resonus_room.core.settings import AppSettings
from resonus_room.core.uncertainty import spatial_average  # noqa: F401


@dataclass(frozen=True)
class AbsorptionSetup:
    """Contents of an ISO 354 setup file (see the module docstring)."""

    volume: float
    empty_files: tuple[Path, ...]
    empty_climate: Climate
    specimen_files: tuple[Path, ...]
    specimen_climate: Climate
    specimen_area: float | None = None
    n_objects: int | None = None
    max_length: float | None = None
    microphones: int | None = None
    sources: int | None = None


@dataclass(frozen=True)
class AbsorptionRun:
    """Result of :func:`run_absorption`.

    Parameters
    ----------
    result
        ISO 354 result (``issues`` include all checks of this run).
    t1_positions, t2_positions
        T20 per position and band, shape ``(n_files, 18)``, s.
    """

    result: AbsorptionResult
    t1_positions: NDArray[np.float64]
    t2_positions: NDArray[np.float64]


def _climate(table: dict[str, Any]) -> Climate:
    """Climate from a TOML table (``pressure`` optional)."""
    return Climate(
        float(table["temperature"]),
        float(table["relative_humidity"]),
        float(table.get("pressure", 101.325)),
    )


def load_setup(path: str | Path) -> AbsorptionSetup:
    """Read an ISO 354 setup file (TOML); file paths are resolved relative to it.

    Raises
    ------
    ValueError
        On missing keys (``volume``, ``[empty]``/``[specimen]`` with ``temperature``,
        ``relative_humidity``, ``files``), empty file lists, or neither/both of
        ``specimen_area`` and ``n_objects``.
    OSError
        If the file cannot be read.
    """
    p = Path(path)
    with p.open("rb") as fh:
        cfg = tomllib.load(fh)
    try:
        empty, spec = cfg["empty"], cfg["specimen"]
        e_files = tuple(p.parent / f for f in empty["files"])
        s_files = tuple(p.parent / f for f in spec["files"])
        setup = AbsorptionSetup(
            volume=float(cfg["volume"]),
            empty_files=e_files,
            empty_climate=_climate(empty),
            specimen_files=s_files,
            specimen_climate=_climate(spec),
            specimen_area=float(cfg["specimen_area"]) if "specimen_area" in cfg else None,
            n_objects=int(cfg["n_objects"]) if "n_objects" in cfg else None,
            max_length=float(cfg["max_length"]) if "max_length" in cfg else None,
            microphones=int(cfg["microphones"]) if "microphones" in cfg else None,
            sources=int(cfg["sources"]) if "sources" in cfg else None,
        )
    except KeyError as exc:
        raise ValueError(f"{p}: missing key {exc}") from exc
    if not e_files or not s_files:
        raise ValueError(f"{p}: empty file list")
    if (setup.specimen_area is None) == (setup.n_objects is None):
        raise ValueError(f"{p}: give exactly one of specimen_area and n_objects")
    return setup


def position_t20(files: tuple[Path, ...], settings: AppSettings) -> NDArray[np.float64]:
    """T20 per file and ISO 354 band, shape ``(len(files), 18)``.

    Each file: ``load_impulse_response(f)`` then ``analyze_ir(ir, s)`` with
    ``s = settings.replace(bands="third", f_min=100.0, f_max=5000.0)``;
    row = ``analysis.values("T20")``.
    """
    s = settings.replace(bands="third", f_min=100.0, f_max=5000.0)
    return np.array([analyze_ir(load_impulse_response(f), s).values("T20") for f in files])


def run_absorption(setup: AbsorptionSetup, settings: AppSettings) -> AbsorptionRun:
    """Evaluate an ISO 354 measurement.

    1. ``t1p = position_t20(setup.empty_files, settings)``, ``t2p`` likewise.
    2. ``a1 = spatial_average(t1p)``, ``a2 = spatial_average(t2p)``;
       ``s_i = a_i.std / sqrt(a_i.count)``.
    3. ``res = absorption(a1.mean, a2.mean, setup.volume, setup.empty_climate,
       setup.specimen_climate, setup.specimen_area, setup.n_objects, s1, s2)``.
    4. Extra issues, in this order: ``check_room(volume, max_length)`` if ``max_length``
       is given; ``check_positions(microphones, sources)`` if both are given; for each band
       where any position has no T20 (NaN): ``f"T20 missing at {f:g} Hz in {n} position(s)"``.
       ``issues = res.issues + extra`` (a new AbsorptionResult via
       ``dataclasses.replace``).
    """
    from dataclasses import replace

    t1p = position_t20(setup.empty_files, settings)
    t2p = position_t20(setup.specimen_files, settings)
    a1 = spatial_average(t1p)
    a2 = spatial_average(t2p)
    s1 = a1.std / np.sqrt(a1.count)
    s2 = a2.std / np.sqrt(a2.count)
    res = absorption(
        a1.mean,
        a2.mean,
        setup.volume,
        setup.empty_climate,
        setup.specimen_climate,
        setup.specimen_area,
        setup.n_objects,
        s1,
        s2,
    )
    extra: list[str] = []
    if setup.max_length is not None:
        extra += check_room(setup.volume, setup.max_length)
    if setup.microphones is not None and setup.sources is not None:
        extra += check_positions(setup.microphones, setup.sources)
    missing = np.isnan(t1p).sum(axis=0) + np.isnan(t2p).sum(axis=0)
    for f, n in zip(ISO354_FREQUENCIES, missing, strict=True):
        if n:
            extra.append(f"T20 missing at {f:g} Hz in {n} position(s)")
    return AbsorptionRun(replace(res, issues=res.issues + tuple(extra)), t1p, t2p)
