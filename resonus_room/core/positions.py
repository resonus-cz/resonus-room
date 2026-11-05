"""Measurement positions and room averages, ISO 3382-2:2008 4.3, 5.1 and clause 8.

A position setup file (TOML) lists the impulse responses with their source and
microphone labels::

    method = "precision"          # survey | engineering | precision | correction
    volume = 50.0                 # m3, optional: minimum distance check, Eq. (1)
    reverberation_time = 1.0      # s, expected T for Eq. (1), optional
    temperature = 20.0            # degrees C, speed of sound for Eq. (1), default 20

    [[position]]
    file = "s1m 1.wav"            # relative to the setup file
    source = "S1"
    microphone = "M1"
    distance = 2.0                # source-microphone distance in m, optional

``"correction"`` is footnote a of Table 1 (result used as a correction term to other
engineering-level measurements: one source position and three microphone positions).

The room average is the arithmetic mean of the per-position values (clause 8 a)); the
standard deviation of the mean reverberation time follows Eq. (A.4) with N = number of
source-microphone combinations and n = 10 (integrated impulse response method, A.3).
"""

from __future__ import annotations

import math  # noqa: F401
import re
import tomllib  # noqa: F401
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np  # noqa: F401
from resonus_core.air import speed_of_sound  # noqa: F401

# ISO 3382-2:2008 Table 1: minimum numbers (source-microphone combinations, source
# positions, microphone positions); "correction" = footnote a (1 source, 3 microphones).
POSITION_REQUIREMENTS = {
    "survey": (2, 1, 2),
    "engineering": (6, 2, 2),
    "precision": (12, 2, 3),
    "correction": (3, 1, 3),
}
METHODS = tuple(POSITION_REQUIREMENTS)
# ISO 3382-2:2008 5.1: frequency range to cover at least (nominal Hz), per band fraction.
# Survey: 250 Hz-2 kHz (octave bands only, 4.3.2); engineering/precision: 125 Hz-4 kHz in
# octaves or 100 Hz-5 kHz in one-third octaves. "correction" follows engineering.
FREQUENCY_RANGE = {
    "survey": {1: (250.0, 2000.0)},
    "engineering": {1: (125.0, 4000.0), 3: (100.0, 5000.0)},
    "precision": {1: (125.0, 4000.0), 3: (100.0, 5000.0)},
    "correction": {1: (125.0, 4000.0), 3: (100.0, 5000.0)},
}
DEFAULT_TEMPERATURE = 20.0


@dataclass(frozen=True)
class Position:
    """One source-microphone measurement.

    Parameters
    ----------
    file
        Impulse response file (.wav or .h5), absolute or relative to the working directory.
    source, microphone
        Position labels.
    distance
        Source-microphone distance in m, or None if not given.
    """

    file: Path
    source: str
    microphone: str
    distance: float | None = None


@dataclass(frozen=True)
class PositionSetup:
    """Positions of one room measurement (see the module docstring for the file format).

    Parameters
    ----------
    method
        One of ``METHODS``.
    positions
        The measurements, at least one.
    volume
        Room volume in m3, or None.
    reverberation_time
        Expected reverberation time for Eq. (1) in s, or None.
    temperature
        Air temperature in degrees C (speed of sound for Eq. (1)).
    """

    method: str
    positions: tuple[Position, ...]
    volume: float | None = None
    reverberation_time: float | None = None
    temperature: float = DEFAULT_TEMPERATURE


def minimum_source_distance(volume: float, speed: float, reverberation_time: float) -> float:
    """Minimum source-microphone distance ``d_min = 2 sqrt(V / (c T))`` in m.

    Parameters
    ----------
    volume
        Room volume V in m3, > 0.
    speed
        Speed of sound c in m/s, > 0.
    reverberation_time
        Estimate of the expected reverberation time T in s, > 0.

    Raises
    ------
    ValueError
        If any argument is not positive.

    References
    ----------
    ISO 3382-2:2008, 4.3.1, Eq. (1).
    """
    if volume <= 0 or speed <= 0 or reverberation_time <= 0:
        raise ValueError("volume, speed and reverberation_time must be > 0")
    return 2 * math.sqrt(volume / (speed * reverberation_time))


def load_positions(path: str | Path) -> PositionSetup:
    """Read a position setup file (TOML, see the module docstring).

    ``file`` entries are resolved relative to the setup file's folder. Optional keys:
    ``volume``, ``reverberation_time``, ``temperature`` (default ``DEFAULT_TEMPERATURE``),
    per position ``distance``. Labels are converted with ``str``, numbers with ``float``.

    Raises
    ------
    ValueError
        On a missing key (``method``, ``position``, per position ``file``, ``source``,
        ``microphone``), an unknown method, or an empty position list.
    OSError
        If the file cannot be read.
    """
    p = Path(path)
    with p.open("rb") as fh:
        cfg = tomllib.load(fh)

    if "method" not in cfg:
        raise ValueError(f"{p}: missing key 'method'")
    method = cfg["method"]
    if method not in METHODS:
        raise ValueError(f"{p}: unknown method {method!r}")

    if "position" not in cfg:
        raise ValueError(f"{p}: missing key 'position'")
    pos_cfg = cfg["position"]
    if not pos_cfg:
        raise ValueError(f"{p}: empty position list")

    positions: list[Position] = []
    for e in pos_cfg:
        if "file" not in e or "source" not in e or "microphone" not in e:
            raise ValueError(f"{p}: missing key in position entry")
        file_path = p.parent / str(e["file"])
        source = str(e["source"])
        microphone = str(e["microphone"])
        distance = float(e["distance"]) if "distance" in e else None
        positions.append(Position(file_path, source, microphone, distance))

    volume = float(cfg["volume"]) if "volume" in cfg else None
    reverberation_time = float(cfg["reverberation_time"]) if "reverberation_time" in cfg else None
    temperature = float(cfg["temperature"]) if "temperature" in cfg else DEFAULT_TEMPERATURE

    return PositionSetup(method, tuple(positions), volume, reverberation_time, temperature)


def check_positions(setup: PositionSetup, fraction: int, nominal: list[float]) -> list[str]:
    """Check a setup against ISO 3382-2:2008 Table 1, 4.3.1 Eq. (1), 4.3.2 and 5.1.

    ``fraction`` and ``nominal`` (nominal band frequencies analysed, in Hz) describe the
    band set. Returns the findings in this order (empty list: all requirements met):

    1. Table 1, with ``(n_comb, n_src, n_mic) = POSITION_REQUIREMENTS[setup.method]``:
       the number of distinct ``(source, microphone)`` pairs, of distinct sources and of
       distinct microphones, each compared with its minimum. Messages
       ``f"ISO 3382-2 Table 1: {n} source-microphone combinations, {method} requires
       {n_comb}"``, ``f"ISO 3382-2 Table 1: {n} source positions, {method} requires
       {n_src}"``, ``f"ISO 3382-2 Table 1: {n} microphone positions, {method} requires
       {n_mic}"``.
    2. Survey with ``fraction != 1``: ``"ISO 3382-2 4.3.2: survey measurements are made
       in octave bands only"``.
    3. Frequency range: if ``fraction`` is in ``FREQUENCY_RANGE[method]`` with ``(lo,
       hi)`` and ``min(nominal) > lo`` or ``max(nominal) < hi``: ``f"ISO 3382-2 5.1: the
       frequency range should cover at least {lo:g}-{hi:g} Hz"``.
    4. Eq. (1), only if ``volume`` and ``reverberation_time`` are given: ``d =
       minimum_source_distance(volume, speed_of_sound(temperature),
       reverberation_time)``; for each position with ``distance is not None and distance
       < d``: ``f"ISO 3382-2 4.3.1 Eq. (1): {file name}: distance {distance:g} m is below
       d_min = {d:.2f} m"``.
    """
    issues: list[str] = []

    # 1. Table 1 checks
    comb_set = {(q.source, q.microphone) for q in setup.positions}
    src_set = {q.source for q in setup.positions}
    mic_set = {q.microphone for q in setup.positions}
    n_comb, n_src, n_mic = POSITION_REQUIREMENTS[setup.method]
    if len(comb_set) < n_comb:
        issues.append(
            f"ISO 3382-2 Table 1: {len(comb_set)} source-microphone combinations, "
            f"{setup.method} requires {n_comb}"
        )
    if len(src_set) < n_src:
        issues.append(
            f"ISO 3382-2 Table 1: {len(src_set)} source positions, {setup.method} requires {n_src}"
        )
    if len(mic_set) < n_mic:
        issues.append(
            f"ISO 3382-2 Table 1: {len(mic_set)} microphone positions, "
            f"{setup.method} requires {n_mic}"
        )

    # 2. Survey octave band check
    if setup.method == "survey" and fraction != 1:
        issues.append("ISO 3382-2 4.3.2: survey measurements are made in octave bands only")

    # 3. Frequency range check
    if fraction in FREQUENCY_RANGE[setup.method]:
        lo, hi = FREQUENCY_RANGE[setup.method][fraction]
        if min(nominal) > lo or max(nominal) < hi:
            issues.append(
                f"ISO 3382-2 5.1: the frequency range should cover at least {lo:g}-{hi:g} Hz"
            )

    # 4. Eq. (1) distance check
    if setup.volume is not None and setup.reverberation_time is not None:
        d_min = minimum_source_distance(
            setup.volume,
            float(speed_of_sound(setup.temperature)),
            setup.reverberation_time,
        )
        for q in setup.positions:
            if q.distance is not None and q.distance < d_min:
                issues.append(
                    f"ISO 3382-2 4.3.1 Eq. (1): {q.file.name}: distance {q.distance:g} m "
                    f"is below d_min = {d_min:.2f} m"
                )

    return issues


# Channel labels that name one source and one microphone position (user convention s#m#,
# D-17). Dummy-head ears (m1L/m1R) and simultaneous sources (s1+s2m1) are not one
# position each and are refused for the room average (D-19, user decision D-19).
POSITION_LABEL = re.compile(r"(s[1-9][0-9]*)(m[1-9][0-9]*)")
# File of one channel in a `resonus-room measure --out` folder (cli.py).
CHANNEL_FILE = "impulse_response.h5"


# File names of single impulse responses in the user's convention, e.g. REW exports
# " s1m 1.wav" (user decision 2026-11-04, D-19): source and microphone number only, case
# and white space ignored. Extensions removed first (case-insensitive), longest first.
FILE_POSITION = re.compile(r"s\s*([1-9][0-9]*)\s*m\s*([1-9][0-9]*)", re.IGNORECASE)
FILE_SUFFIXES = (".ir.h5", ".h5", ".wav")


def label_from_filename(name: str) -> str | None:
    """Channel label ``s#m#`` from a file name that names one position, else None.

    Remove the first of ``FILE_SUFFIXES`` that ends ``name`` (case-insensitive), strip
    white space; if the rest matches ``FILE_POSITION`` completely (``fullmatch``), return
    ``f"s{source}m{microphone}"`` (lower case, numbers as written), else None.

    Examples: ``" s1m 1.wav"`` -> ``"s1m1"``, ``"S2M3.WAV"`` -> ``"s2m3"``,
    ``"s1 m12.ir.h5"`` -> ``"s1m12"``, ``"s1m1"`` -> ``"s1m1"``; None for
    ``"room_s1m1.wav"``, ``"s1m1L.wav"`` (dummy-head ear, D-19), ``"s0m1.wav"``,
    ``"s1m1_take2.wav"``.
    """
    stem = name
    for suffix in FILE_SUFFIXES:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    m = FILE_POSITION.fullmatch(stem.strip())
    if m is None:
        return None
    return f"s{m.group(1)}m{m.group(2)}"


def label_position(label: str) -> tuple[str, str]:
    """Source and microphone of a channel label: ``"s1m2"`` -> ``("s1", "m2")``.

    The whole label must match ``POSITION_LABEL`` (``re.fullmatch``).

    Raises
    ------
    ValueError
        ``f"{label!r} is not one source and one microphone position (s#m#)"``.
    """
    m = POSITION_LABEL.fullmatch(label)
    if m is None:
        raise ValueError(f"{label!r} is not one source and one microphone position (s#m#)")
    return m.group(1), m.group(2)


def positions_from_folders(folders: Sequence[str | Path], method: str) -> PositionSetup:
    """Position setup from ``measure --out`` folders of multichannel runs (D-19).

    Every subfolder (sorted by name) of every folder (in the given order) that contains
    ``CHANNEL_FILE`` is one position: ``Position(sub / CHANNEL_FILE, source,
    microphone)`` with ``source, microphone = label_position(sub.name)`` (the subfolder
    is named by the channel label). Other subfolders and files are ignored. No volume,
    reverberation time or distances (Eq. (1) is not checked).

    Raises
    ------
    ValueError
        ``f"unknown method {method!r}"`` if ``method`` is not in ``METHODS``;
        ``f"{folder}: no channel folders with {CHANNEL_FILE}"`` for a folder without
        any; for labels that are not positions one message naming all of them:
        ``"not one source and one microphone position (s#m#): " + ", ".join(repr(name)
        for each such subfolder)``, checked before anything else is returned.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")

    found: list[Path] = []
    for folder in folders:
        subs = sorted(
            q for q in Path(folder).iterdir() if q.is_dir() and (q / CHANNEL_FILE).is_file()
        )
        if not subs:
            raise ValueError(f"{folder}: no channel folders with {CHANNEL_FILE}")
        found += subs
    # all invalid labels of all folders in one message
    invalid = [q.name for q in found if POSITION_LABEL.fullmatch(q.name) is None]
    if invalid:
        raise ValueError(
            "not one source and one microphone position (s#m#): "
            + ", ".join(repr(name) for name in invalid)
        )
    positions: list[Position] = []
    for q in found:
        source, microphone = label_position(q.name)
        positions.append(Position(q / CHANNEL_FILE, source, microphone))
    return PositionSetup(method, tuple(positions))
