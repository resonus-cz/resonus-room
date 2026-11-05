"""Level test before a measurement: test noise, input levels, level-meter zones.

The level test (GUI Test button, CLI ``test-levels``, notebook ``level_test``) plays
white noise for ``TEST_DURATION`` on each configured output in turn, records the inputs and shows
each input's peak and RMS level on a vertical meter with a clipping indicator (user
requests 2026-11-05, docs/DECISIONS.md D-21). The meter marks are the user's (-12, -42,
-60, -100 dB); the zones between them and the clipping threshold are design choices (no
standard applies).

Levels are in dB re full scale (amplitude 1): peak ``20 lg max|x|`` and RMS
``20 lg sqrt(mean(x**2))`` (resonus-core ``signals`` convention; AES17 dBFS of an RMS
value would be 3.01 dB higher).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from resonus_core.audio import AudioIO
from resonus_core.signals import white_noise

from resonus_room.core.settings import AppSettings

# Length of the test noise in s (user request 2026-11-05).
TEST_DURATION: Final = 1.0
# Meter scale in dB re full scale, bottom to top; marks given by the user (2026-11-05).
METER_RANGE: Final = (-100.0, 0.0)
METER_MARKS: Final = (-12.0, -42.0, -60.0, -100.0)
# Zones of the meter, lower bound in dB, top down (design choice, D-21): "hot" above the
# -12 dB mark (little headroom), "good" -42...-12 dB, "low" -60...-42 dB, "quiet" below.
METER_ZONES: Final = (("hot", -12.0), ("good", -42.0), ("low", -60.0), ("quiet", -np.inf))
# A sample counts as clipped from the top code of a 16-bit converter on (design choice,
# D-21): 1 - 2**-15 is reached by any converter of 16 bits or more at full scale.
CLIP_THRESHOLD: Final = 1.0 - 2.0**-15


@dataclass(frozen=True)
class InputLevel:
    """Levels of one recorded input during the test.

    Attributes
    ----------
    peak_db
        Peak level ``20 lg max|x|`` in dB re full scale; ``-inf`` for silence.
    rms_db
        RMS level ``20 lg sqrt(mean(x**2))`` in dB re full scale; ``-inf`` for silence.
    clipped
        True if any ``|x| >= CLIP_THRESHOLD``.
    """

    peak_db: float
    rms_db: float
    clipped: bool


def level_noise(
    level_db: float, sample_rate: float, duration: float = TEST_DURATION
) -> NDArray[np.float64]:
    """Return white test noise with the peak at ``level_db`` (1-D, played on every output).

    ``n = round(duration * sample_rate)``; ``x = white_noise(n, sample_rate, rms_db=-40.0,
    periodic=False).signal`` (any level that does not clip, rescaled next); returns ``x *
    10**(level_db / 20) / max|x|``. The peak, not the RMS, is set to the level, so the test
    never clips the output and reaches the same peak as a sweep at that level (D-21).

    Raises
    ------
    ValueError
        ``"level must be <= 0 dB re full scale"`` if ``level_db > 0``;
        ``"duration and sample rate must be positive"`` if ``n < 1``.
    """
    # Validate level
    if level_db > 0.0:
        raise ValueError("level must be <= 0 dB re full scale")
    # Compute number of samples
    n = round(duration * sample_rate)
    if n < 1:
        raise ValueError("duration and sample rate must be positive")
    # Generate white noise at a low RMS to avoid clipping before scaling
    noise = white_noise(n, sample_rate, rms_db=-40.0, periodic=False).signal
    # Scale to desired peak level
    peak = np.max(np.abs(noise))
    if peak == 0:
        return noise  # silent noise
    scale = 10 ** (level_db / 20.0) / peak
    return noise * scale


def input_levels(recording: ArrayLike) -> list[InputLevel]:
    """Peak, RMS and clipping of each column of ``recording`` (shape ``(n, n_inputs)``).

    A 1-D recording is one input. Levels of an all-zero column are ``-inf``.

    Raises
    ------
    ValueError
        ``"recording must be a non-empty 1-D or 2-D array"``.
    """
    rec = np.asarray(recording, dtype=np.float64)  # integer samples would overflow in rec**2
    if rec.ndim == 0:
        raise ValueError("recording must be a non-empty 1-D or 2-D array")
    if rec.ndim == 1:
        rec = rec[:, None]
    if rec.ndim != 2:
        raise ValueError("recording must be a non-empty 1-D or 2-D array")
    if rec.shape[0] == 0 or rec.shape[1] == 0:
        raise ValueError("recording must be a non-empty 1-D or 2-D array")
    peaks = np.max(np.abs(rec), axis=0)
    rms = np.sqrt(np.mean(rec**2, axis=0))
    with np.errstate(divide="ignore"):
        peak_db = 20 * np.log10(peaks)
        rms_db = 20 * np.log10(rms)
    clipped = np.any(np.abs(rec) >= CLIP_THRESHOLD, axis=0)
    levels = []
    for p, r, c in zip(peak_db, rms_db, clipped, strict=False):
        levels.append(InputLevel(float(p), float(r), bool(c)))
    return levels


def meter_zone(level_db: float) -> str:
    """Return the zone of ``level_db``: first of ``METER_ZONES`` with ``level_db >= bound``.

    ``-inf`` and NaN give ``"quiet"``.
    """
    if not np.isfinite(level_db):
        return "quiet"
    for zone, bound in METER_ZONES:
        if level_db >= bound:
            return zone
    return "quiet"


def meter_fraction(level_db: float) -> float:
    """Return the height of ``level_db`` on the meter, 0 (bottom) to 1 (top).

    ``(level_db - METER_RANGE[0]) / (METER_RANGE[1] - METER_RANGE[0])`` clipped to
    [0, 1] (linear in dB); ``-inf`` and NaN give 0.
    """
    if not np.isfinite(level_db):
        return 0.0
    frac = (level_db - METER_RANGE[0]) / (METER_RANGE[1] - METER_RANGE[0])
    return float(np.clip(frac, 0.0, 1.0))


def meter_labels(settings: AppSettings) -> list[str]:
    """Labels of the recorded inputs, in the order of the recording.

    ``list(settings.input_labels)``, plus ``"ref"`` if ``settings.reference_channel`` is
    not 0 (the loopback reference is recorded last, resonus-core ``open_io``).
    """
    labels = list(settings.input_labels)
    if settings.reference_channel != 0:
        labels.append("ref")
    return labels


# Result of the level test for one output: its label and the level of each input.
OutputLevels = tuple[str, list[InputLevel]]


def level_test(
    settings: AppSettings,
    io: AudioIO,
    on_block: Callable[[str, list[InputLevel]], None] | None = None,
) -> list[OutputLevels]:
    """Play the test noise on each output in turn; return the input levels per output.

    User decision D-21 (2026-11-05): one output at a time, in the order of
    ``settings.output_labels`` (whatever the output mode). ``x = level_noise(
    settings.level, io.sample_rate)``; for output ``j`` the signal is ``(n, n_outputs)``
    zeros with column ``j`` = ``x``; ``recording = io.playrec_blocks(signal, block)``,
    where ``block(b)`` calls ``on_block(output_labels[j], input_levels(b))`` if
    ``on_block`` is given (live meter; on a sound card it runs in the audio thread).
    Returns ``[(output_labels[j], input_levels(recording)), ...]``; the input levels are
    in the order of :func:`meter_labels`.

    Raises
    ------
    ValueError
        ``f"{io.n_outputs} outputs, but {k} output labels"`` if the counts differ; from
        :func:`level_noise`.
    """
    labels = list(settings.output_labels)
    if len(labels) != io.n_outputs:
        raise ValueError(f"{io.n_outputs} outputs, but {len(labels)} output labels")
    x = level_noise(settings.level, io.sample_rate)
    results: list[OutputLevels] = []
    for j, label in enumerate(labels):
        signal = np.zeros((x.size, io.n_outputs))
        signal[:, j] = x

        def block(b: NDArray[np.float64], label: str = label) -> None:
            if on_block is not None:
                on_block(label, input_levels(b))

        recording = io.playrec_blocks(signal, block)
        results.append((label, input_levels(recording)))
    return results


def pair_labels(settings: AppSettings) -> list[list[str]]:
    """Labels ``output + input`` of the level-test pairs, e.g. ``"s1m1"``, ``"s2ref"``.

    One list per output label, each in the order of :func:`meter_labels` (the user's
    naming convention s#m#, D-17).
    """
    return [[out + inp for inp in meter_labels(settings)] for out in settings.output_labels]


def combined_levels(results: Sequence[OutputLevels]) -> list[InputLevel]:
    """Per input, the highest peak and RMS over all outputs, clipped if any clipped.

    Shown on the meters after a test (one meter per input). ``results`` must hold the
    same number of inputs for every output.

    Raises
    ------
    ValueError
        ``"no results"`` for an empty ``results``.
    """
    if not results:
        raise ValueError("no results")
    per_input = zip(*(levels for _label, levels in results), strict=True)
    return [
        InputLevel(
            max(v.peak_db for v in vs), max(v.rms_db for v in vs), any(v.clipped for v in vs)
        )
        for vs in per_input
    ]
