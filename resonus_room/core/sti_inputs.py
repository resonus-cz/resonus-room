"""Inputs of the STI estimate and the STI at one listener position (PROVISIONAL, D-31).

Work-package items 2, 3 and 6 of docs/PLAN_STI.md: the calibrated background-noise
recording, the source calibration (free-field impulse response at 1 m), and the STI at a
listener position from the calibrated in-situ impulse response
(:func:`resonus_room.core.sti.measure_mtf_compound`).

Files (``.h5``, resonus-core ``data.save``) carry a role in ``metadata.extra[ROLE_KEY]``
(design choice D-31): a noise recording is a ``TimeSignal`` (resonus-core >= 0.11.0,
Q-C9) with unit ``"Pa"`` and role :data:`NOISE_ROLE`; recordings written before were an
``ImpulseResponse`` with that role and are still read (:data:`NoiseRecording`). A
source calibration is an ``ImpulseResponse`` with unit ``"Pa/FS"`` and role
:data:`SOURCE_ROLE`. An in-situ impulse response has no role.

Known limits (docs/PLAN_STI.md section 4): the impulse responses' own noise floor lowers m
once more (negligible at a high measurement S/N); the source calibration and the in-situ
impulse response must use the same drive chain; the speech levels are a user input
(Q-STI3).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from resonus_core.audio import AudioIO
from resonus_core.data import ImpulseResponse, Metadata, TimeSignal

from resonus_room import __version__
from resonus_room.core.calibrate import CALIBRATED_IR_UNIT, OFFSET_KEY, calibrated_recording
from resonus_room.core.settings import AppSettings
from resonus_room.core.sti import (
    CUSTOM_EFFORT,
    OCTAVE_BANDS,
    band_level,
    measure_mtf_compound,
    sti_from_mtf,
    vocal_effort_levels,
)

APP_NAME: Final = "resonus-room"
# Metadata.extra keys and values of the roles (design choice D-31).
ROLE_KEY: Final = "role"
LABEL_KEY: Final = "label"
DISTANCE_KEY: Final = "distance_m"
NOISE_ROLE: Final = "background-noise"
SOURCE_ROLE: Final = "source-calibration"
NOISE_UNIT: Final = "Pa"
# Source calibration distance in m: free field at 1 m (docs/PLAN_STI.md item 3, user).
SOURCE_DISTANCE_M: Final = 1.0
# Length of the modulation analysis in s after the build-up (design choice D-31: at least
# five periods of the lowest modulation frequency, 0.63 Hz).
STI_WINDOW_S: Final = 8.0
# Default length of a noise recording in s (design choice D-31: covers an in-situ impulse
# response of up to 12 s plus STI_WINDOW_S).
DEFAULT_NOISE_DURATION: Final = 20.0
# A noise recording: a TimeSignal, or an ImpulseResponse with NOISE_ROLE in files written
# before resonus-core 0.11.0 (Q-C9).
NoiseRecording = TimeSignal | ImpulseResponse


def _now() -> str:
    """Return the current UTC time, ISO 8601 to the second."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def role(result: ImpulseResponse | TimeSignal) -> str:
    """Return ``result.metadata.extra.get(ROLE_KEY, "")`` ("" for an in-situ IR)."""
    return str(result.metadata.extra.get(ROLE_KEY, ""))


def record_noise(
    settings: AppSettings,
    io: AudioIO,
    offsets: Sequence[float | None],
    duration: float = DEFAULT_NOISE_DURATION,
) -> list[TimeSignal]:
    """Record the background noise (no excitation) on every measurement input, in Pa.

    ``k = len(settings.input_channels)``; ``rec = io.rec(round(duration *
    io.sample_rate))[:, :k]`` (a reference channel, recorded last, is dropped);
    ``pa = calibrated_recording(rec, offsets, settings.input_labels)``. One result per
    input ``i``: ``TimeSignal(pa[:, i], io.sample_rate, NOISE_UNIT, Metadata(APP_NAME,
    __version__, created=now, description=f"Background noise {label}", extra={ROLE_KEY:
    NOISE_ROLE, LABEL_KEY: label, OFFSET_KEY: offsets[i]}), start_utc=now)`` with ``label
    = settings.input_labels[i]`` and ``now`` the UTC time before recording, ISO 8601 to
    the second.

    Raises
    ------
    ValueError
        ``"duration must be > 0"``; from ``calibrated_recording`` (e.g. an uncalibrated
        input).
    """
    if duration <= 0:
        raise ValueError("duration must be > 0")
    created = _now()
    k = len(settings.input_channels)
    rec = io.rec(round(duration * io.sample_rate))[:, :k]
    pa = calibrated_recording(rec, offsets, settings.input_labels)
    out: list[TimeSignal] = []
    for i, label in enumerate(settings.input_labels):
        out.append(
            TimeSignal(
                pa[:, i].copy(),
                io.sample_rate,
                NOISE_UNIT,
                Metadata(
                    APP_NAME,
                    __version__,
                    created=created,
                    description=f"Background noise {label}",
                    extra={ROLE_KEY: NOISE_ROLE, LABEL_KEY: label, OFFSET_KEY: offsets[i]},
                ),
                start_utc=created,
            )
        )
    return out


def source_calibration(
    ir: ImpulseResponse, distance_m: float = SOURCE_DISTANCE_M
) -> ImpulseResponse:
    """Mark a calibrated free-field impulse response as the source calibration.

    A copy of ``ir`` with ``extra = {**ir.metadata.extra, ROLE_KEY: SOURCE_ROLE,
    DISTANCE_KEY: distance_m}`` (``dataclasses.replace``); data unchanged.

    Raises
    ------
    ValueError
        ``f"source calibration needs a calibrated impulse response (unit 'Pa/FS'), got
        {ir.unit!r}"``; ``"distance must be > 0"``.
    """
    if ir.unit != CALIBRATED_IR_UNIT:
        raise ValueError(
            "source calibration needs a calibrated impulse response (unit 'Pa/FS'), "
            f"got {ir.unit!r}"
        )
    if distance_m <= 0:
        raise ValueError("distance must be > 0")
    return dataclasses.replace(
        ir,
        metadata=dataclasses.replace(
            ir.metadata,
            extra={**ir.metadata.extra, ROLE_KEY: SOURCE_ROLE, DISTANCE_KEY: distance_m},
        ),
    )


def noise_band_levels(noise: NoiseRecording) -> NDArray[np.float64]:
    """Octave-band levels ``band_level(noise.data, noise.sample_rate, k)``, k = 0...6.

    Raises
    ------
    ValueError
        ``"not a noise recording"`` unless ``role(noise) == NOISE_ROLE``.
    """
    if role(noise) != NOISE_ROLE:
        raise ValueError("not a noise recording")
    return np.array(
        [band_level(noise.data, noise.sample_rate, k) for k in range(7)], dtype=np.float64
    )


@dataclass(frozen=True)
class StiReport:
    """STI at one listener position (PROVISIONAL, original STI of docs/PLAN_STI.md).

    Parameters
    ----------
    sti
        Speech Transmission Index, 0...1.
    mti
        Per octave band (:data:`resonus_room.core.sti.OCTAVE_BANDS`), shape ``(7,)``.
    m
        Modulation transfer matrix, shape ``(7, 14)``.
    speech_levels_db
        Speech levels at 1 m used, dB re 20 µPa, shape ``(7,)``.
    noise_levels_db
        Octave-band levels of the noise recording, dB re 20 µPa, shape ``(7,)``.
    seed
        Seed of the test-signal noise.
    """

    sti: float
    mti: NDArray[np.float64]
    m: NDArray[np.float64]
    speech_levels_db: NDArray[np.float64]
    noise_levels_db: NDArray[np.float64]
    seed: int


def settings_speech_levels(settings: AppSettings) -> tuple[float, ...]:
    """Speech levels at 1 m of the settings (7 octave bands, or ``()``; Q-STI3).

    ``settings.sti_speech_levels`` if ``settings.sti_vocal_effort == CUSTOM_EFFORT``
    (``()`` if none were entered), else
    ``resonus_room.core.sti.vocal_effort_levels(settings.sti_vocal_effort)``.
    """
    if settings.sti_vocal_effort == CUSTOM_EFFORT:
        return tuple(settings.sti_speech_levels)
    return vocal_effort_levels(settings.sti_vocal_effort)


def sti_at_position(
    room_ir: ImpulseResponse,
    source: ImpulseResponse,
    noise: NoiseRecording,
    speech_levels_db: ArrayLike,
    seed: int = 0,
) -> StiReport:
    """STI at the position of ``room_ir`` (work-package item 6, route b).

    Checks in this order (each a ``ValueError``):

    1. ``room_ir.unit == CALIBRATED_IR_UNIT`` and ``role(room_ir) == ""``: else
       ``"the in-situ impulse response must be calibrated (unit 'Pa/FS')"``;
    2. ``role(source) == SOURCE_ROLE``: else ``"not a source calibration"``;
    3. ``role(noise) == NOISE_ROLE``: else ``"not a noise recording"``;
    4. equal sample rates of the three: else ``"sample rates differ"``;
    5. ``speech_levels_db`` of shape ``(7,)``: else ``"speech_levels_db must have 7
       values"``;
    6. ``start_s = room_ir.data.size / fs`` (the whole impulse response builds up),
       ``duration = start_s + STI_WINDOW_S``; ``noise.data.size >= round(duration *
       fs)``: else ``f"noise recording too short: {noise.data.size / fs:.1f} s, need
       {duration:.1f} s"``.

    Then ``m = measure_mtf_compound(fs, duration, np.random.default_rng(seed),
    speech_levels_db, source.data, room_ir.data, noise.data, start_s)``, ``r =
    sti_from_mtf(m)``; ``noise_levels_db = noise_band_levels(noise)``.
    """
    if room_ir.unit != CALIBRATED_IR_UNIT or role(room_ir) != "":
        raise ValueError("the in-situ impulse response must be calibrated (unit 'Pa/FS')")
    if role(source) != SOURCE_ROLE:
        raise ValueError("not a source calibration")
    if role(noise) != NOISE_ROLE:
        raise ValueError("not a noise recording")
    if room_ir.sample_rate != source.sample_rate or room_ir.sample_rate != noise.sample_rate:
        raise ValueError("sample rates differ")
    levels = np.asarray(speech_levels_db, float)
    if levels.shape != (7,):
        raise ValueError("speech_levels_db must have 7 values")
    fs = room_ir.sample_rate
    start_s = room_ir.data.size / fs
    duration = start_s + STI_WINDOW_S
    if noise.data.size < round(duration * fs):
        raise ValueError(
            f"noise recording too short: {noise.data.size / fs:.1f} s, need {duration:.1f} s"
        )
    rng = np.random.default_rng(seed)
    m = measure_mtf_compound(
        fs, duration, rng, levels, source.data, room_ir.data, noise.data, start_s
    )
    r = sti_from_mtf(m)
    return StiReport(r.sti, r.mti, m, levels, noise_band_levels(noise), seed)


__all__ = [
    "DEFAULT_NOISE_DURATION",
    "NOISE_ROLE",
    "OCTAVE_BANDS",
    "SOURCE_ROLE",
    "STI_WINDOW_S",
    "StiReport",
    "noise_band_levels",
    "record_noise",
    "role",
    "source_calibration",
    "sti_at_position",
]
