"""Measurement and analysis engine of Resonus Room (used by CLI, GUI and notebooks)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from resonus_core.audio import AudioIO, SimulatedIO  # noqa: F401
from resonus_core.bands import BandSet, band_set, filter_bands  # noqa: F401
from resonus_core.data import ImpulseResponse
from resonus_core.measure import MeasurementResult, open_io
from resonus_core.measure import measure as core_measure  # noqa: F401

from resonus_room import __version__  # noqa: F401
from resonus_room.core.analysis import RoomAnalysis, analyze  # noqa: F401
from resonus_room.core.settings import AppSettings
from resonus_room.core.simulate import exponential_decay_ir  # noqa: F401

APP_NAME = "resonus-room"


def settings_band_set(settings: AppSettings) -> BandSet:
    """Band set of the settings: ``band_set(1 if bands == "octave" else 3, f_min, f_max)``."""
    return band_set(1 if settings.bands == "octave" else 3, settings.f_min, settings.f_max)


def simulated_room(settings: AppSettings, seed: int = 0) -> NDArray[np.float64]:
    """Impulse response of the simulated room (no background noise; the device adds it).

    ``exponential_decay_ir(fs, duration, [sim_decay_time] * n, bands, delay=0.005,
    seed=0)`` with one-third-octave bands from 25 Hz up to the highest band whose upper
    edge is below ``0.45 * fs`` (``band_set(3, 25, f)`` with ``f`` the largest nominal
    frequency allowed) and ``duration = 2 * sim_decay_time + 0.5`` s. ``seed`` selects
    the noise realisation (one per simulated source/microphone path).
    Normalised to unit energy, ``h / sqrt(sum(h**2))`` (user decision D-21, 2026-11-05): the
    room neither amplifies nor attenuates white noise power, so simulated input levels
    are realistic (before: about +34 dB).
    """
    fs = float(settings.sample_rate)
    bs = band_set(3, 25.0, 20000.0)
    keep = [x for x, up in zip(bs.indices, bs.upper, strict=True) if up < 0.45 * fs]
    bands = BandSet(3, tuple(keep))
    t60 = settings.sim_decay_time
    h = exponential_decay_ir(
        fs, 2 * t60 + 0.5, np.full(len(bands), t60), bands, delay=0.005, seed=seed
    )
    return h / np.sqrt(np.sum(h**2))


def open_device(settings: AppSettings) -> AudioIO:
    """Return the audio device of the settings.

    Simulated room if ``settings.simulate``, else the selected input/output devices and
    channels (``resonus_core.measure.open_io``).

    The simulated device has the channel layout of the settings, as ``open_io`` would
    (resonus-core 0.5.0: io inputs = ``input_channels``, then the reference): with
    ``n_in = len(input_channels)``, ``ref = bool(reference_channel)``, ``n_out =
    len(output_channels)``, ``SimulatedIO(sample_rate, h, noise_rms=10**(sim_noise/20))``
    with ``h`` of shape ``(n_in + ref, n_out, len(simulated_room(settings)))``:
    ``h[i, j] = simulated_room(settings, seed=i * n_out + j)`` (one room path per
    microphone and source; path 0 is the former single-channel room), and with a
    reference ``h[n_in, 0, 0] = 1.0`` (a wire from the first output: the loopback is
    connected to one output, resonus-core ``MeasurementSettings`` relation 3).
    """
    if settings.simulate:
        n_in, n_out = len(settings.input_channels), len(settings.output_channels)
        ref = bool(settings.reference_channel)
        room = simulated_room(settings)
        h = np.zeros((n_in + ref, n_out, room.size))
        for i in range(n_in):
            for j in range(n_out):
                h[i, j] = simulated_room(settings, seed=i * n_out + j)
        if ref:
            # the reference is the last io input, whatever its device channel number
            h[n_in, 0, 0] = 1.0
        return SimulatedIO(
            settings.sample_rate,
            h,
            noise_rms=10 ** (settings.sim_noise / 20),
        )
    else:
        return open_io(settings)


def measure(settings: AppSettings, io: AudioIO) -> MeasurementResult:
    """Run a measurement with ``resonus_core.measure.measure`` (app name and version stored).

    Raises
    ------
    ValueError
        On a sample-rate mismatch or invalid settings (from resonus-core).
    """
    return core_measure(settings, io, software=APP_NAME, software_version=__version__)


def analyze_ir(ir: ImpulseResponse, settings: AppSettings) -> RoomAnalysis:
    """Analyse an impulse response with the analysis settings.

    ``analyze(ir.data, ir.sample_rate, settings_band_set(settings), settings.filter_order,
    settings.reverse_filter, settings.decay_method)``.
    """
    return analyze(
        ir.data,
        ir.sample_rate,
        settings_band_set(settings),
        settings.filter_order,
        settings.reverse_filter,
        settings.decay_method,
    )


def band_impulse_responses(
    ir: ImpulseResponse, analysis: RoomAnalysis, settings: AppSettings
) -> NDArray[np.float64]:
    """Band-filtered impulse responses from the onset, as used by the analysis.

    ``x = ir.data`` with trailing exact zeros removed (as in ``analyze``, D-13); ``y =
    x[analysis.onset:]``; returns ``filter_bands(y, analysis.bands, ir.sample_rate,
    settings.filter_order, settings.reverse_filter)``, shape ``(len(y), len(bands))``.
    ``settings`` must be those of the analysis.

    Raises
    ------
    ValueError
        If ``ir`` is all zero.
    """
    x = ir.data
    nz = np.flatnonzero(x)
    if nz.size == 0:
        raise ValueError("Impulse response is all zero")
    y = x[: nz[-1] + 1][analysis.onset :]
    return filter_bands(
        y,
        analysis.bands,
        ir.sample_rate,
        settings.filter_order,
        settings.reverse_filter,
    )
