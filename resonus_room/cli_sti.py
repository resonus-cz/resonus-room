"""CLI commands of the calibrated STI chain: calibrate, record-noise, sti (D-31).

``measure --calibrated`` (in :mod:`resonus_room.cli`) uses :func:`calibrate_irs`.
Everything about STI is PROVISIONAL (docs/PLAN_STI.md).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from resonus_core import data
from resonus_core.calibration import CAPTURE_DURATION, CalibrationRecord, CalibrationStore
from resonus_core.data import ImpulseResponse, Metadata, TimeSignal
from resonus_core.devices import DeviceInfo
from resonus_core.plot import MplPlotView

from resonus_room import __version__
from resonus_room.core import engine
from resonus_room.core.calibrate import (
    calibrated_ir,
    capture_tone,
    input_offsets,
    input_profiles,
    store_path,
)
from resonus_room.core.settings import AppSettings
from resonus_room.core.sti_export import sti_to_band_results, sti_to_csv, sti_to_dict
from resonus_room.core.sti_inputs import (
    APP_NAME,
    DEFAULT_NOISE_DURATION,
    noise_band_levels,
    record_noise,
    settings_speech_levels,
    sti_at_position,
)
from resonus_room.plotting import band_ticks, setup_band_axes, show_band_results

# Message when no speech levels are given (no default from the filed sources, Q-STI3).
NO_SPEECH_LEVELS = (
    "give the speech levels at 1 m: 7 octave-band levels 125 Hz-8 kHz in dB, "
    "--speech-levels, or the setting sti_speech_levels with sti_vocal_effort = custom"
)


def add_parsers(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the sub-commands ``calibrate``, ``record-noise`` and ``sti``."""
    c = sub.add_parser(
        "calibrate",
        help="calibrate an input with an acoustic calibrator (or enter an offset, or show them)",
    )
    c.add_argument("--settings", type=Path, help="settings TOML file")
    c.add_argument(
        "--simulate", action=argparse.BooleanOptionalAction, help="override the simulate setting"
    )
    c.add_argument("--store", type=Path, help="calibration history (default: the app's file)")
    c.add_argument("--input", help="input label, e.g. m1")
    c.add_argument("--level", type=float, help="calibrator level in dB re 20 uPa (certificate)")
    c.add_argument("--frequency", type=float, default=1000.0, help="calibrator frequency in Hz")
    c.add_argument(
        "--ambient", type=float, default=0.0, help="ambient correction in dB (IEC 61672-3 10.2)"
    )
    c.add_argument(
        "--adjustment", type=float, default=0.0, help="adjustment data in dB (IEC 61672-3 9.1)"
    )
    c.add_argument("--duration", type=float, default=CAPTURE_DURATION, help="capture length in s")
    c.add_argument("--manual", type=float, help="enter this offset in dB instead of measuring")
    c.add_argument("--note", default="", help="note stored with the record")
    c.add_argument("--save", action="store_true", help="save the result to the history")
    c.add_argument("--show", action="store_true", help="show the active offsets and exit")
    c.add_argument("--json", action="store_true", help="print a machine-readable result")

    n = sub.add_parser(
        "record-noise", help="record the calibrated background noise on every input (STI)"
    )
    n.add_argument("--settings", type=Path, help="settings TOML file")
    n.add_argument(
        "--simulate", action=argparse.BooleanOptionalAction, help="override the simulate setting"
    )
    n.add_argument("--store", type=Path, help="calibration history (default: the app's file)")
    n.add_argument(
        "--duration", type=float, default=DEFAULT_NOISE_DURATION, help="recording length in s"
    )
    n.add_argument("--out", type=Path, required=True, help="output folder")
    n.add_argument("--json", action="store_true", help="print the band levels as JSON")

    s = sub.add_parser("sti", help="STI at a position (PROVISIONAL, original STI)")
    s.add_argument("room_ir", type=Path, help="calibrated in-situ impulse response (.h5)")
    s.add_argument("--source", type=Path, required=True, help="source calibration (.h5)")
    s.add_argument("--noise", type=Path, required=True, help="noise recording (.h5)")
    s.add_argument("--speech-levels", help="7 levels at 1 m in dB, comma-separated")
    s.add_argument("--settings", type=Path, help="settings TOML file (speech levels, plot mark)")
    s.add_argument("--seed", type=int, default=0, help="seed of the test signal")
    s.add_argument("--out", type=Path, help="output folder (default: print CSV)")
    s.add_argument("--json", action="store_true", help="print the result as JSON")


def devices_for(settings: AppSettings) -> list[DeviceInfo]:
    """Return ``[]`` if ``settings.simulate``, else ``resonus_core.devices.list_devices()``."""
    if settings.simulate:
        return []
    from resonus_core.devices import list_devices

    return list_devices()


def speech_levels(text: str | None, settings: AppSettings) -> NDArray[np.float64]:
    """Return the 7 speech levels: from ``text`` if given, else from the settings.

    Settings: ``settings_speech_levels(settings)`` (the vocal-effort spectrum of
    ``sti_vocal_effort``, or ``sti_speech_levels`` for "custom"; Q-STI3, D-34).

    ``text``: comma-separated numbers (``float()`` each, spaces allowed).

    Raises
    ------
    ValueError
        ``f"--speech-levels: not a number: {item.strip()!r}"``; :data:`NO_SPEECH_LEVELS`
        if neither gives any level; ``"speech levels: give 7 values (125 Hz-8 kHz)"`` if
        not exactly 7.
    """
    if text is not None:
        values: list[float] = []
        for item in text.split(","):
            try:
                values.append(float(item))
            except ValueError:
                raise ValueError(f"--speech-levels: not a number: {item.strip()!r}") from None
    else:
        values = list(settings_speech_levels(settings))
    if not values:
        raise ValueError(NO_SPEECH_LEVELS)
    if len(values) != 7:
        raise ValueError("speech levels: give 7 values (125 Hz-8 kHz)")
    return np.array(values, dtype=np.float64)


def calibrate_irs(
    settings: AppSettings,
    irs: Sequence[ImpulseResponse],
    store: CalibrationStore,
    devices: Sequence[DeviceInfo],
) -> list[ImpulseResponse]:
    """Return the measured impulse responses in Pa/FS (``measure --calibrated``).

    ``offsets = input_offsets(settings, devices, store)``; each ``ir`` belongs to the input
    ``i = settings.input_labels.index(ir.metadata.extra["channel"]["input_label"])``
    (resonus-core ``measure`` metadata) and becomes ``calibrated_ir(ir, offsets[i])``.

    Raises
    ------
    ValueError
        ``f"input {label} is not calibrated: run resonus-room calibrate --input {label}"``
        for an input without an offset; from ``input_offsets``.
    """
    offsets = input_offsets(settings, devices, store)
    out: list[ImpulseResponse] = []
    for ir in irs:
        label = ir.metadata.extra["channel"]["input_label"]
        offset = offsets[settings.input_labels.index(label)]
        if offset is None:
            raise ValueError(
                f"input {label} is not calibrated: run resonus-room calibrate --input {label}"
            )
        out.append(calibrated_ir(ir, offset))
    return out


def _common(args: argparse.Namespace) -> tuple[AppSettings, CalibrationStore, list[DeviceInfo]]:
    """Return settings (``--settings``, ``--simulate``), history and devices of a command."""
    from resonus_room.cli import _settings  # here: resonus_room.cli imports this module

    settings = _settings(args.settings)
    if args.simulate is not None:
        settings = settings.replace(simulate=args.simulate)
    store = CalibrationStore.open(args.store or store_path())
    return settings, store, devices_for(settings)


def calibrate_command(args: argparse.Namespace) -> int:
    """Command ``calibrate``.

    Settings from ``--settings`` (``resonus_room.cli._settings``), then ``--simulate``;
    ``store = CalibrationStore.open(args.store or store_path())``; ``devices =
    devices_for(settings)``; ``profiles = input_profiles(settings, devices)``.

    1. ``--show``: for each input label and its active offset (:func:`input_offsets`),
       print ``f"{label}: {offset:.2f} dB"`` or ``f"{label}: not calibrated"``; with
       ``--json`` print ``json.dumps({label: offset_or_None, ...})`` instead. Return 0.
    2. Otherwise ``--input`` is required (``ValueError("--input is required")``) and must
       be one of ``settings.input_labels`` (``ValueError(f"no input {label!r} in the
       settings")``); ``i`` its index.
    3. ``--manual``: ``record = CalibrationRecord.manual(profiles[i], args.manual,
       now, args.note)``; ``offset = args.manual``.
       Else ``--level`` is required (``ValueError("--level is required (or --manual)")``);
       ``proposal = capture_tone(settings, open_device(settings), i, args.frequency,
       args.level, args.ambient, args.adjustment, args.duration)`` (``open_device`` from
       :mod:`resonus_room.core.engine`, called as ``engine.open_device`` so tests can
       replace it);
       ``record = CalibrationRecord.from_proposal(profiles[i], proposal, now,
       note=args.note)``; ``offset = proposal.offset_db``. ``now = datetime.now(UTC)``.
    4. Print ``f"{label}: offset {offset:.2f} dB"``; then with ``--save``
       ``store.append(record)`` and print ``f"saved to {store.path}"``, else print
       ``"not saved: run again with --save to keep it"``. With ``--json`` print instead
       ``json.dumps({"input": label, "offset_db": offset, "saved": args.save})``. Return 0.
    """
    settings, store, devices = _common(args)
    profiles = input_profiles(settings, devices)
    if args.show:
        offsets = input_offsets(settings, devices, store)
        if args.json:
            print(json.dumps(dict(zip(settings.input_labels, offsets, strict=True))))
        else:
            for label, offset in zip(settings.input_labels, offsets, strict=True):
                if offset is None:
                    print(f"{label}: not calibrated")
                else:
                    print(f"{label}: {offset:.2f} dB")
        return 0
    label = args.input
    if label is None:
        raise ValueError("--input is required")
    if label not in settings.input_labels:
        raise ValueError(f"no input {label!r} in the settings")
    i = settings.input_labels.index(label)
    now = datetime.now(UTC)
    if args.manual is not None:
        record = CalibrationRecord.manual(profiles[i], args.manual, now, args.note)
        offset = float(args.manual)
    else:
        if args.level is None:
            raise ValueError("--level is required (or --manual)")
        proposal = capture_tone(
            settings,
            engine.open_device(settings),
            i,
            args.frequency,
            args.level,
            args.ambient,
            args.adjustment,
            args.duration,
        )
        record = CalibrationRecord.from_proposal(profiles[i], proposal, now, note=args.note)
        offset = proposal.offset_db
    if args.save:
        store.append(record)
    if args.json:
        print(json.dumps({"input": label, "offset_db": offset, "saved": bool(args.save)}))
        return 0
    print(f"{label}: offset {offset:.2f} dB")
    print(f"saved to {store.path}" if args.save else "not saved: run again with --save to keep it")
    return 0


def record_noise_command(args: argparse.Namespace) -> int:
    """Command ``record-noise``.

    Settings, ``--simulate``, store and devices as :func:`calibrate_command`;
    ``offsets = input_offsets(...)``; ``noises = record_noise(settings,
    engine.open_device(settings), offsets, args.duration)``. ``args.out`` is created; each noise
    is saved as ``out / f"noise_{label}.h5"`` (``resonus_core.data.save``). The band
    levels (:func:`noise_band_levels`) go to ``out / "noise.csv"``: header
    ``"input,125,250,500,1000,2000,4000,8000"``, one line per input ``label`` then the 7
    levels ``.1f`` (empty if not finite); printed too, or with ``--json``
    ``json.dumps({label: [level or None, ...]})``. Return 0.
    """
    settings, store, devices = _common(args)
    offsets = input_offsets(settings, devices, store)
    noises = record_noise(settings, engine.open_device(settings), offsets, args.duration)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    lines = ["input,125,250,500,1000,2000,4000,8000"]
    levels: dict[str, list[float | None]] = {}
    for noise, label in zip(noises, settings.input_labels, strict=True):
        data.save(out / f"noise_{label}.h5", noise)
        band = noise_band_levels(noise)
        lines.append(",".join([label] + [f"{x:.1f}" if np.isfinite(x) else "" for x in band]))
        levels[label] = [float(x) if np.isfinite(x) else None for x in band]
    text = "\n".join(lines) + "\n"
    out.joinpath("noise.csv").write_text(text, encoding="utf-8")
    if args.json:
        print(json.dumps(levels))
    else:
        print(text, end="")
    return 0


def sti_command(args: argparse.Namespace) -> int:
    """Command ``sti``.

    ``room = data.load(args.room_ir)``, ``source = data.load(args.source)``, ``noise =
    data.load(args.noise)``; room and source must be an ``ImpulseResponse``
    (``ValueError(f"{path}: not an impulse response file")``), the noise a ``TimeSignal``
    or (files before resonus-core 0.11.0) an ``ImpulseResponse`` (``ValueError(f"{path}:
    not a noise recording file")``). ``levels =
    speech_levels(args.speech_levels, settings)`` (settings from ``--settings``).
    ``report = sti_at_position(room, source, noise, levels, args.seed)``.

    - ``--json``: print ``json.dumps(sti_to_dict(report))``.
    - no ``--out``: print ``sti_to_csv(report)`` (unless ``--json``).
    - ``--out``: created; ``sti.json`` (``indent=2``), ``sti.csv``, ``sti.h5``
      (``sti_to_band_results(report, Metadata(APP_NAME, __version__,
      description=args.room_ir.stem))``) and ``sti.png``: ``MplPlotView(brand_text=
      settings.plot_mark)``, ``setup_band_axes(view, "MTI", "1")``,
      ``show_band_results(view, bres, "MTI", "MTI", f"MTI (STI {report.sti:.2f})")``,
      ``band_ticks(view, bres.center, 1)``, ``view.save(...)``.

    Return 0.
    """
    from resonus_room.cli import _settings

    settings = _settings(args.settings)
    loaded: list[ImpulseResponse] = []
    for path in (args.room_ir, args.source):
        obj = data.load(path)
        if not isinstance(obj, ImpulseResponse):
            raise ValueError(f"{path}: not an impulse response file")
        loaded.append(obj)
    room, source = loaded
    noise = data.load(args.noise)
    if not isinstance(noise, (TimeSignal, ImpulseResponse)):
        raise ValueError(f"{args.noise}: not a noise recording file")
    levels = speech_levels(args.speech_levels, settings)
    report = sti_at_position(room, source, noise, levels, args.seed)
    if args.out:
        out: Path = args.out
        out.mkdir(parents=True, exist_ok=True)
        out.joinpath("sti.json").write_text(
            json.dumps(sti_to_dict(report), indent=2), encoding="utf-8"
        )
        out.joinpath("sti.csv").write_text(sti_to_csv(report), encoding="utf-8")
        meta = Metadata(APP_NAME, __version__, description=args.room_ir.stem)
        bres = sti_to_band_results(report, meta)
        data.save(out / "sti.h5", bres)
        view = MplPlotView(brand_text=settings.plot_mark)
        setup_band_axes(view, "MTI", "1")
        show_band_results(view, bres, "MTI", "MTI", f"MTI (STI {report.sti:.2f})")
        band_ticks(view, bres.center, 1)
        view.save(str(out / "sti.png"))
    if args.json:
        print(json.dumps(sti_to_dict(report)))
    elif not args.out:
        print(sti_to_csv(report), end="")
    return 0
