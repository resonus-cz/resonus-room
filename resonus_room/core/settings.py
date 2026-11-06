"""Settings of Resonus Room: one dataclass for GUI preferences, CLI (TOML) and notebooks.

Measurement fields come from :class:`resonus_core.measure.MeasurementSettings` (some
defaults changed for rooms); analysis and simulation fields are added here.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from resonus_core.bands import DEFAULT_FILTER_ORDER
from resonus_core.devices import DeviceInfo
from resonus_core.measure import MeasurementSettings
from resonus_core.settings import param
from resonus_core.style import TOKENS

from resonus_room.core.decay import METHODS
from resonus_room.core.positions import METHODS as POSITION_METHODS
from resonus_room.core.sti import VOCAL_EFFORTS

BAND_CHOICES = ("octave", "third")
# Channel labels, naming convention s#m# (user decisions D-17): outputs
# (sources) "s<n>", inputs (microphones) "m<n>", a dummy head's ears "m<n>L" / "m<n>R".
OUTPUT_LABEL = re.compile(r"s[1-9][0-9]*")
INPUT_LABEL = re.compile(r"m[1-9][0-9]*[LR]?")


@dataclass(frozen=True)
class AppSettings(MeasurementSettings):
    """Measurement, analysis and simulation settings.

    Analysis defaults: ISO 3382-2:2008 5.1 asks for at least 125 Hz-4 kHz (octaves) or
    100 Hz-5 kHz (one-third octaves) for engineering/precision measurements; ``f_min =
    100``, ``f_max = 5000`` select exactly these bands for both resolutions.

    Channel labels (checked after the core relations): every ``output_labels`` item must
    match ``OUTPUT_LABEL`` and every ``input_labels`` item ``INPUT_LABEL`` in full
    (``re.fullmatch``); else ``ValueError(f"output label {label!r}: use s1, s2, ...")``
    resp. ``ValueError(f"input label {label!r}: use m1, m2, ... (m1L / m1R for a dummy
    head)")``. Example: source s1, dummy head at position 1 -> results ``s1m1L``,
    ``s1m1R``.

    The audio device is chosen by ``input_device`` / ``output_device`` (core keys
    ``"<host API>: <name>"``); the former single ``device`` field (a PortAudio name) is
    migrated by :func:`migrate_settings`.
    """

    # Room measurements: longer silence and IR than resonus-core's defaults (design choice:
    # 3 s covers T up to about 3 s at a 60 dB decay).
    tail: float = param(
        3.0, label="Silence after sweep", unit="s", minimum=0.0, maximum=60.0, group="Signal"
    )
    ir_length: float = param(
        3.0,
        label="IR length",
        unit="s",
        minimum=0.001,
        maximum=600.0,
        group="Processing",
        help="Sweeps: at most the silence after the sweep; noise/MLS: at most the period",
    )
    simulate: bool = param(
        True,
        label="Simulated device",
        help="Use the built-in simulated room",
        group="Simulation",
    )
    bands: str = param("octave", label="Bands", choices=BAND_CHOICES, group="Analysis")
    f_min: float = param(
        100.0, label="Lowest band", unit="Hz", minimum=25.0, maximum=20000.0, group="Analysis"
    )
    f_max: float = param(
        5000.0, label="Highest band", unit="Hz", minimum=25.0, maximum=20000.0, group="Analysis"
    )
    filter_order: int = param(
        DEFAULT_FILTER_ORDER,
        label="Filter order",
        minimum=1,
        maximum=8,
        group="Analysis",
        help="Butterworth order (PROVISIONAL, IEC 61260-1 not verified)",
    )
    reverse_filter: bool = param(
        False,
        label="Time-reversed filtering",
        group="Analysis",
        help="ISO 3382-2 7.3 NOTE: reliable down to BT > 4",
    )
    decay_method: str = param(
        "correct",
        label="Noise compensation",
        choices=METHODS,
        group="Analysis",
        help="ISO 3382-1 5.3.3: truncation and correction",
    )
    # Default by user decision D-19 (D-19): method for the Table 1 check of the GUI room average.
    position_method: str = param(
        "engineering",
        label="Room average method",
        choices=POSITION_METHODS,
        group="Analysis",
        help="ISO 3382-2:2008 Table 1: minimum numbers of positions",
    )
    # Text in the bottom-right corner of every plot (user request 2026-11-05, D-24):
    # replaceable, default resonus-core's "Re:sonus", empty for none.
    plot_mark: str = param(
        str(TOKENS["brand_text"]),
        label="Plot mark",
        group="Plots",
        help="Text in the bottom-right corner of every plot and exported image; empty: none",
    )
    # IACC tab (user request 2026-11-05, D-28): off by default, binaural measurements only.
    show_iacc: bool = param(
        False,
        label="IACC tab",
        group="Plots",
        help="Show the IACC tab (ISO 3382-1:2009 Annex B) for ear pairs labelled s#m#L / s#m#R",
    )
    # STI tab (docs/PLAN_STI.md item 7): off by default, needs calibrated inputs (D-31).
    show_sti: bool = param(
        False,
        label="STI tab",
        group="Plots",
        help="Show the STI tab (PROVISIONAL: original STI, not IEC 60268-16:2020)",
    )
    # Absorption tab (user request 2026-11-06, D-32): ISO 354 from two folders; off by
    # default, reverberation-room measurements only.
    show_absorption: bool = param(
        False,
        label="Absorption tab",
        group="Plots",
        help="Show the Absorption tab (ISO 354:2003 from two folders: empty room, with specimen)",
    )
    # Vocal effort of the talker for STI (Q-STI3, user decision 2026-11-06): the speech
    # spectra of the Odeon application note (core/sti.py [4]); "custom" uses
    # sti_speech_levels. Default "normal": design choice D-34.
    sti_vocal_effort: str = param(
        "normal",
        label="STI vocal effort",
        choices=VOCAL_EFFORTS,
        group="STI",
        help="Speech level at 1 m: ANSI S3.5 spectra (Odeon application note) or custom",
    )
    # Speech levels at 1 m per octave band 125 Hz-8 kHz: a user input, no default from
    # the filed sources (QUESTIONS.md Q-STI3, D-31).
    sti_speech_levels: tuple[float, ...] = param(
        (),
        item=float,
        label="STI speech levels at 1 m",
        unit="dB",
        minimum=0.0,
        maximum=140.0,
        group="STI",
        help="Custom: 7 octave-band levels 125 Hz-8 kHz in dB re 20 µPa",
    )
    # Calibrated measurements (D-31): impulse responses in Pa/FS with the inputs' offsets.
    calibrated: bool = param(
        False,
        label="Calibrated impulse responses",
        group="STI",
        help="Measure in Pa/FS with the inputs' calibration (needed for STI)",
    )
    sim_decay_time: float = param(
        1.0, label="Simulated T", unit="s", minimum=0.05, maximum=20.0, group="Simulation"
    )
    sim_noise: float = param(
        -80.0,
        label="Simulated background",
        unit="dB re 1",
        minimum=-160.0,
        maximum=0.0,
        group="Simulation",
        help="RMS level of the simulated background noise",
    )

    def __post_init__(self) -> None:
        """Validate fields, then the relations between them."""
        super().__post_init__()
        if self.f_max < self.f_min:
            raise ValueError("f_max must not be below f_min")
        if self.sti_speech_levels and len(self.sti_speech_levels) != 7:
            raise ValueError("sti_speech_levels: give 7 octave-band levels (125 Hz-8 kHz) or none")
        for label in self.output_labels:
            if not OUTPUT_LABEL.fullmatch(label):
                raise ValueError(f"output label {label!r}: use s1, s2, ...")
        for label in self.input_labels:
            if not INPUT_LABEL.fullmatch(label):
                raise ValueError(
                    f"input label {label!r}: use m1, m2, ... (m1L / m1R for a dummy head)"
                )


def migrate_settings(
    data: dict[str, object], devices: Callable[[], Sequence[DeviceInfo]]
) -> tuple[dict[str, object], list[str]]:
    """Convert settings written by older versions (before resonus-core 0.5.0, D-17).

    The key ``"device"`` (a PortAudio device *name*, empty = default) is removed. If it
    is non-empty, ``devices()`` is called (only then) and the devices with that
    ``name`` are looked up: exactly one -> its ``key`` becomes ``input_device`` and
    ``output_device`` (unless the data already has these keys); none or several -> both
    stay at their defaults (default device) and a message ``f"audio device {name!r}
    not found: using the default device; choose it in Preferences"`` resp. ``f"audio
    device {name!r} is listed by several host APIs: using the default device; choose
    it in Preferences"`` is returned (no guessing between host APIs).

    Parameters
    ----------
    data
        Settings as read from TOML; not modified.
    devices
        Returns the current device list (e.g. ``resonus_core.devices.list_devices``).

    Returns
    -------
    data, messages
        Migrated copy, and messages for the user (empty if nothing to report).
    """
    new = dict(data)
    name = new.pop("device", None)
    if not name:
        return new, []
    devs = devices()
    matches = [d for d in devs if d.name == name]
    if len(matches) == 1:
        key = matches[0].key
        new.setdefault("input_device", key)
        new.setdefault("output_device", key)
        return new, []
    if len(matches) == 0:
        msg = f"audio device {name!r} not found: using the default device; choose it in Preferences"
    else:
        msg = (
            f"audio device {name!r} is listed by several host APIs: using the "
            "default device; choose it in Preferences"
        )
    return new, [msg]


def load_settings(
    path: str | Path, devices: Callable[[], Sequence[DeviceInfo]] | None = None
) -> tuple[AppSettings, list[str]]:
    """Read settings from a TOML file, migrating older files (:func:`migrate_settings`).

    ``tomllib`` reads the file, :func:`migrate_settings` converts it (``devices``
    defaults to ``resonus_core.devices.list_devices``), ``AppSettings.from_dict``
    validates it.

    Returns
    -------
    settings, messages
        The settings and the migration messages.

    Raises
    ------
    ValueError
        On unknown keys or invalid values (from ``AppSettings.from_dict``).
    """
    if devices is None:
        from resonus_core.devices import list_devices

        devices = list_devices
    with Path(path).open("rb") as fh:
        data = tomllib.load(fh)
    new, messages = migrate_settings(data, devices)
    settings = AppSettings.from_dict(new)
    return settings, messages
