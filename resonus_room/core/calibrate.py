"""Level calibration of the measurement inputs (resonus-core ``calibration``, D-C1; D-31).

The method, the quality checks and the history file are resonus-core's (the method of
resonus-meter). This module binds them to the app: one :class:`InputProfile` per
measurement input of the settings, the app's own history file, the tone capture through
the app's audio device, and the conversion of recordings and impulse responses to Pa.

Units (resonus-core ``measure``, checked 2026-11-05): a measured impulse response has unit
``"1"``, input full scale per output full scale (or per reference-input full scale with a
loopback reference); its in-band gain is the gain of the path. Calibrated with the
input's offset ``C`` it is in Pa per full-scale drive, unit :data:`CALIBRATED_IR_UNIT`.
Two calibrated impulse responses are comparable only with the same drive chain (output,
amplifier gain, reference configuration): design note of D-31.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from resonus_core.audio import AudioIO
from resonus_core.calibration import (
    CAPTURE_DURATION,
    CalibrationProposal,
    CalibrationStore,
    InputProfile,
    propose,
    to_pascal,
)
from resonus_core.data import ImpulseResponse
from resonus_core.devices import DeviceInfo

from resonus_room.core.settings import AppSettings

# History file name, in the app's data directory (design choice D-31: the app keeps its
# own history; resonus-meter identifies devices differently, by its cpal names).
STORE_FILE: Final = "calibrations.toml"
# Directory of the app's data below the XDG data home.
APP_DIR: Final = "resonus-room"
# Device identity of the simulated device (settings.simulate).
SIMULATED_DEVICE_ID: Final = "simulated"
SIMULATED_DEVICE_NAME: Final = "Simulated room"
# Unit of a calibrated impulse response: Pa per full-scale drive (D-31).
CALIBRATED_IR_UNIT: Final = "Pa/FS"
# Unit of an uncalibrated impulse response (resonus-core measure).
UNCALIBRATED_IR_UNIT: Final = "1"
# Metadata.extra key of the applied calibration offset in dB.
OFFSET_KEY: Final = "calibration_offset_db"


def store_path(data_home: str | Path | None = None) -> Path:
    """Path of the calibration history: ``<data home>/resonus-room/calibrations.toml``.

    ``data_home`` if given, else the environment variable ``XDG_DATA_HOME`` if set and
    non-empty, else ``Path.home() / ".local" / "share"`` (XDG Base Directory
    Specification 0.8). The file is not created here.
    """
    base = (
        Path(data_home)
        if data_home is not None
        else Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    )
    return base / APP_DIR / STORE_FILE


def input_profiles(settings: AppSettings, devices: Sequence[DeviceInfo]) -> list[InputProfile]:
    """One profile per measurement input, in the order of ``settings.input_channels``.

    The reference channel is not included (it is not acoustic). ``InputProfile(
    device_id, device_name, channel - 1, int(settings.sample_rate))`` for each 1-based
    ``channel``, where with ``settings.simulate`` the device is
    (:data:`SIMULATED_DEVICE_ID`, :data:`SIMULATED_DEVICE_NAME`), else the one device of
    ``devices`` whose ``key`` equals ``settings.input_device`` (its ``key`` and ``name``).

    Raises
    ------
    ValueError
        ``"choose the input device in Preferences to use calibration"`` if
        ``settings.input_device`` is empty and not simulated (the default device may
        change, so it is not calibrated); ``f"input device {key!r} not found"`` if no
        device has that key.
    """
    if settings.simulate:
        device_id = SIMULATED_DEVICE_ID
        device_name = SIMULATED_DEVICE_NAME
    else:
        input_device_key = settings.input_device
        if not input_device_key:
            raise ValueError("choose the input device in Preferences to use calibration")
        device = next((d for d in devices if d.key == input_device_key), None)
        if device is None:
            raise ValueError(f"input device {input_device_key!r} not found")
        device_id, device_name = device.key, device.name

    rate = int(settings.sample_rate)
    return [InputProfile(device_id, device_name, ch - 1, rate) for ch in settings.input_channels]


def input_offsets(
    settings: AppSettings, devices: Sequence[DeviceInfo], store: CalibrationStore
) -> list[float | None]:
    """Active calibration offset ``C`` in dB of each input of :func:`input_profiles`.

    ``store.active(profile).offset_db``, or None for an input without a record.

    Raises
    ------
    ValueError
        From :func:`input_profiles`.
    """
    offsets: list[float | None] = []
    for profile in input_profiles(settings, devices):
        record = store.active(profile)
        offsets.append(None if record is None else record.offset_db)
    return offsets


def capture_tone(
    settings: AppSettings,
    io: AudioIO,
    input_index: int,
    frequency: float,
    calibrator_level_db: float,
    ambient_correction_db: float = 0.0,
    adjustment_correction_db: float = 0.0,
    duration: float = CAPTURE_DURATION,
) -> CalibrationProposal:
    """Record the calibrator tone on one input and return the calibration proposal.

    ``x = io.rec(round(duration * io.sample_rate))[:, input_index]``, then
    ``resonus_core.calibration.propose(x, io.sample_rate, frequency, calibrator_level_db,
    ambient_correction_db, adjustment_correction_db)``. Nothing is played; nothing is
    saved (the user confirms the proposal first).

    Parameters
    ----------
    input_index
        Zero-based index into ``settings.input_channels``.

    Raises
    ------
    ValueError
        ``"input_index out of range"`` unless ``0 <= input_index <
        len(settings.input_channels)``.
    resonus_core.calibration.CalibrationError
        From ``propose`` (quality checks).
    """
    if not (0 <= input_index < len(settings.input_channels)):
        raise ValueError("input_index out of range")

    n_samples = round(duration * io.sample_rate)
    recording = io.rec(n_samples)
    x = recording[:, input_index]
    return propose(
        x,
        io.sample_rate,
        frequency,
        calibrator_level_db,
        ambient_correction_db,
        adjustment_correction_db,
    )


def calibrated_ir(ir: ImpulseResponse, offset_db: float) -> ImpulseResponse:
    """Return the impulse response in Pa per full-scale drive.

    ``data = resonus_core.calibration.to_pascal(ir.data, offset_db)``, unit
    :data:`CALIBRATED_IR_UNIT`, same sample rate; the metadata is a copy
    (``dataclasses.replace``) with ``extra = {**ir.metadata.extra, OFFSET_KEY:
    offset_db}``.

    Raises
    ------
    ValueError
        ``f"impulse response unit {ir.unit!r}: expected '1' (uncalibrated)"`` unless
        ``ir.unit == UNCALIBRATED_IR_UNIT``.
    """
    if ir.unit != UNCALIBRATED_IR_UNIT:
        raise ValueError(f"impulse response unit {ir.unit!r}: expected '1' (uncalibrated)")

    data = to_pascal(ir.data, offset_db)
    meta = dataclasses.replace(ir.metadata, extra={**ir.metadata.extra, OFFSET_KEY: offset_db})
    return dataclasses.replace(ir, data=data, unit=CALIBRATED_IR_UNIT, metadata=meta)


def calibrated_recording(
    recording: ArrayLike, offsets: Sequence[float | None], labels: Sequence[str]
) -> NDArray[np.float64]:
    """Return the recording in Pa: column ``i`` is ``to_pascal(recording[:, i], offsets[i])``.

    ``recording`` has shape ``(n, k)``; ``offsets`` and ``labels`` have ``k`` items.

    Raises
    ------
    ValueError
        ``"recording must be 2-D"``; ``"one offset and one label per column"``;
        ``f"input {labels[i]} is not calibrated"`` for the first None offset.
    """
    recording = np.asarray(recording, dtype=np.float64)

    if recording.ndim != 2:
        raise ValueError("recording must be 2-D")

    n_channels = recording.shape[1]
    if len(offsets) != n_channels or len(labels) != n_channels:
        raise ValueError("one offset and one label per column")

    result = np.empty(recording.shape, dtype=np.float64)
    for i, (offset, label) in enumerate(zip(offsets, labels, strict=True)):
        if offset is None:
            raise ValueError(f"input {label} is not calibrated")
        result[:, i] = to_pascal(recording[:, i], offset)
    return result
