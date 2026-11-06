"""Command-line front-end of Resonus Room.

Subcommands: ``measure`` (measure an IR and analyse it), ``analyze`` (analyse an IR file),
``absorption`` (ISO 354 evaluation of a setup file), ``settings`` (write default
settings), ``plot`` (plot an IR or FR result file). Non-zero exit status on any error.
"""

from __future__ import annotations

import argparse
import json  # noqa: F401
import math
import sys
from pathlib import Path
from typing import Any

from resonus_core import data  # noqa: F401
from resonus_core.bands import band_set
from resonus_core.calibration import CalibrationStore
from resonus_core.data import BandResults, FrequencyResponse, ImpulseResponse, Metadata
from resonus_core.measure import measurement_warnings
from resonus_core.plot import MplPlotView  # noqa: F401

from resonus_room import __version__, cli_sti
from resonus_room.core.analysis import RoomAnalysis
from resonus_room.core.binaural import IACC_BAND_RANGE, binaural_parameters
from resonus_room.core.calibrate import store_path
from resonus_room.core.compare import compare, comparisons_to_csv, summary
from resonus_room.core.engine import analyze_ir, measure, open_device  # noqa: F401
from resonus_room.core.files import (
    APP_NAME,
    absorption_to_band_results,
    absorption_to_csv,  # noqa: F401
    absorption_to_dict,  # noqa: F401
    analysis_to_band_results,
    analysis_to_csv,  # noqa: F401
    analysis_to_dict,  # noqa: F401
    binaural_to_band_results,
    binaural_to_csv,
    load_impulse_response,  # noqa: F401
)
from resonus_room.core.iso354 import load_setup, run_absorption  # noqa: F401
from resonus_room.core.levels import level_test, meter_labels, meter_zone, pair_labels
from resonus_room.core.positions import METHODS, load_positions, positions_from_folders
from resonus_room.core.rew import read_rew_rt60
from resonus_room.core.room_average import (
    average_positions,
    average_to_band_results,
    average_to_csv,
)
from resonus_room.core.settings import BAND_CHOICES, AppSettings, load_settings
from resonus_room.core.sti_inputs import source_calibration
from resonus_room.plotting import (
    band_ticks,
    default_quantity,
    setup_absorption_axes,  # noqa: F401
    setup_band_axes,
    setup_ir_axes,  # noqa: F401
    setup_response_axes,  # noqa: F401
    setup_rt_axes,  # noqa: F401
    show_absorption,  # noqa: F401
    show_band_results,
    show_decay,  # noqa: F401
    show_ir,  # noqa: F401
    show_response,  # noqa: F401
    show_rt,  # noqa: F401
)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser."""
    p = argparse.ArgumentParser(prog="resonus-room", description="Resonus Room")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    m = sub.add_parser("measure", help="measure an impulse response and analyse it")
    m.add_argument("--settings", type=Path, help="settings TOML file")
    m.add_argument(
        "--simulate", action=argparse.BooleanOptionalAction, help="override the simulate setting"
    )
    m.add_argument("--duration", type=float, help="excitation length in s (overrides the setting)")
    m.add_argument("--repetitions", type=int, help="number of averages (overrides the setting)")
    m.add_argument(
        "--inputs", help="input channels, comma-separated, e.g. 1,2 (overrides the setting)"
    )
    m.add_argument(
        "--input-labels",
        help="labels of the input channels, e.g. m1L,m1R (default with --inputs: m1, m2, ...)",
    )
    m.add_argument("--out", type=Path, required=True, help="output folder")
    m.add_argument("--json", action="store_true", help="print a machine-readable summary")
    m.add_argument(
        "--calibrated",
        action="store_true",
        help="save the impulse responses in Pa/FS with the inputs' calibration (D-31)",
    )
    m.add_argument(
        "--source-calibration",
        action="store_true",
        help="with --calibrated: also save source_calibration.h5 (free field, 1 m; STI)",
    )
    m.add_argument("--store", type=Path, help="calibration history (default: the app's file)")

    tl = sub.add_parser(
        "test-levels",
        help="play 1 s of white noise on each output in turn and report the input levels",
    )
    tl.add_argument("--settings", type=Path, help="settings TOML file")
    tl.add_argument(
        "--simulate", action=argparse.BooleanOptionalAction, help="override the simulate setting"
    )
    tl.add_argument(
        "--level", type=float, help="peak level of the noise in dBFS (overrides the setting)"
    )
    tl.add_argument("--inputs", help="input channels, comma-separated (overrides the setting)")
    tl.add_argument("--input-labels", help="labels of the input channels, e.g. m1,m2")
    tl.add_argument("--json", action="store_true", help="print a machine-readable summary")

    a = sub.add_parser("analyze", help="analyse an impulse response file (.h5 or .wav)")
    a.add_argument("ir", type=Path)
    a.add_argument("--settings", type=Path, help="settings TOML file")
    a.add_argument("--bands", choices=BAND_CHOICES, help="override the band setting")
    a.add_argument("--channel", type=int, default=0, help="WAV channel (0-based)")
    a.add_argument("--out", type=Path, help="output folder (default: print CSV)")
    a.add_argument("--json", action="store_true", help="print the full result as JSON")

    b = sub.add_parser("absorption", help="ISO 354 evaluation of a setup TOML file")
    b.add_argument("setup", type=Path)
    b.add_argument("--settings", type=Path, help="settings TOML file (analysis options)")
    b.add_argument("--out", type=Path, help="output folder (default: print CSV)")
    b.add_argument("--json", action="store_true", help="print the result as JSON")

    s = sub.add_parser("settings", help="write the default settings to a TOML file")
    s.add_argument("path", type=Path)

    pl = sub.add_parser(
        "plot",
        help="plot result files (impulse/freq: one file; band results: multiple overlaid)",
    )
    pl.add_argument("result", type=Path, nargs="+")
    pl.add_argument(
        "--quantity",
        action="append",
        help="band-result quantity, e.g. T20 (repeatable; default: T20 or alpha_s)",
    )
    pl.add_argument("-o", "--output", type=Path, required=True, help="image file (png, pdf, svg)")
    pl.add_argument("--settings", type=Path, help="settings TOML file (plot mark)")
    c = sub.add_parser(
        "compare-rew",
        help="compare the analysis of an IR with a REW RT60 export",
    )
    c.add_argument("ir", type=Path)
    c.add_argument("rt60", type=Path)
    c.add_argument("--settings", type=Path, help="settings TOML file")
    c.add_argument("--bands", choices=BAND_CHOICES, help="override the band setting")
    c.add_argument("--out", type=Path, help="CSV file (default: print)")

    bi = sub.add_parser(
        "binaural",
        help="IACC of a left/right ear impulse-response pair (ISO 3382-1:2009 Annex B)",
    )
    bi.add_argument("left", type=Path, help="left-ear impulse response (.h5 or .wav)")
    bi.add_argument("right", type=Path, help="right-ear impulse response (.h5 or .wav)")
    bi.add_argument("--left-channel", type=int, default=0, help="WAV channel of LEFT (0-based)")
    bi.add_argument("--right-channel", type=int, default=0, help="WAV channel of RIGHT (0-based)")
    bi.add_argument("--settings", type=Path, help="settings TOML file (filter order, direction)")
    bi.add_argument("--out", type=Path, help="output folder (default: print CSV)")

    av = sub.add_parser("average", help="average over measurement positions (ISO 3382-2 clause 8)")
    av.add_argument(
        "setup",
        type=Path,
        nargs="+",
        metavar="SETUP_OR_FOLDER",
        help="position setup file (TOML), or measure --out folders with one subfolder "
        "per channel label s#m# (D-19)",
    )
    av.add_argument(
        "--method",
        choices=METHODS,
        help="ISO 3382-2 measurement method for folders (Table 1 check); required for folders",
    )
    av.add_argument("--settings", type=Path, help="settings TOML file")
    av.add_argument("--bands", choices=BAND_CHOICES, help="override the band setting")
    av.add_argument("--out", type=Path, help="output folder (default: print CSV)")

    i = sub.add_parser(
        "import-rew",
        help="convert a REW RT60 export to band-result files",
    )
    i.add_argument("rt60", type=Path)
    i.add_argument("--out", type=Path, required=True, help="output folder")

    cli_sti.add_parsers(sub)
    return p


def _settings(path: Path | None) -> AppSettings:
    if path is None:
        return AppSettings()
    settings, messages = load_settings(path)
    for message in messages:
        print(f"warning: {message}", file=sys.stderr)
    return settings


def _finite(value: float) -> float | None:
    """``value``, or None for -inf/NaN (not valid in JSON)."""
    return value if math.isfinite(value) else None


def _test_levels(args: argparse.Namespace) -> int:
    """Command ``test-levels``: the level test of the GUI's Test button (D-21).

    Settings from ``--settings``, then ``--simulate``, ``--level`` and ``--inputs`` /
    ``--input-labels`` (:func:`_input_override`); ``results = level_test(settings,
    open_device(settings))`` (one output after the other, D-21); pair labels from
    :func:`pair_labels`, input labels from :func:`meter_labels`. One row per pair (output
    ``j``, input ``k``): ``--json`` gives a list of ``{"output", "input", "label",
    "peak_db", "rms_db", "zone", "clipped"}`` (levels -inf as null), otherwise lines
    ``f"{label:>8}  peak {peak_db:6.1f} dB  RMS {rms_db:6.1f} dB  {zone}"`` plus
    ``"  CLIPPED"``. In both modes each clipped pair also gives ``f"warning: {label}
    clipped"`` on stderr. Returns 0.
    """
    settings = _settings(args.settings)
    if args.simulate is not None:
        settings = settings.replace(simulate=args.simulate)
    if args.level is not None:
        settings = settings.replace(level=args.level)
    settings = _input_override(settings, args.inputs, args.input_labels)
    results = level_test(settings, open_device(settings))
    inputs = meter_labels(settings)
    rows = [
        (output, inp, label, v)
        for (output, levels), labels in zip(results, pair_labels(settings), strict=True)
        for inp, label, v in zip(inputs, labels, levels, strict=True)
    ]
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "output": output,
                        "input": inp,
                        "label": label,
                        "peak_db": _finite(v.peak_db),
                        "rms_db": _finite(v.rms_db),
                        "zone": meter_zone(v.peak_db),
                        "clipped": v.clipped,
                    }
                    for output, inp, label, v in rows
                ],
                indent=2,
                allow_nan=False,
            )
        )
    else:
        for _output, _inp, label, v in rows:
            line = f"{label:>8}  peak {v.peak_db:6.1f} dB  RMS {v.rms_db:6.1f} dB  "
            print(line + meter_zone(v.peak_db) + ("  CLIPPED" if v.clipped else ""))
    for _output, _inp, label, v in rows:
        if v.clipped:
            print(f"warning: {label} clipped", file=sys.stderr)
    return 0


def _input_override(settings: AppSettings, inputs: str | None, labels: str | None) -> AppSettings:
    """Apply ``--inputs`` / ``--input-labels`` to the settings.

    ``inputs``: comma-separated channel numbers (spaces allowed, ``int()`` each);
    ``labels``: comma-separated labels (stripped). Neither given: ``settings``
    unchanged. ``labels`` without ``inputs``: only ``input_labels`` replaced. ``inputs``
    without ``labels``: labels ``m1, m2, ...`` (one per channel). The new values go
    through ``settings.replace(...)``, so the settings checks apply (label rules s#m#,
    one label per channel).

    Raises
    ------
    ValueError
        ``f"--inputs: not a channel number: {item.strip()!r}"`` for an item that is not an
        integer; errors of ``settings.replace``.
    """
    changes: dict[str, Any] = {}
    if inputs is not None:
        channels: list[int] = []
        for item in inputs.split(","):  # empty items are errors, not skipped (typos)
            try:
                channels.append(int(item))
            except ValueError:
                raise ValueError(f"--inputs: not a channel number: {item.strip()!r}") from None
        changes["input_channels"] = tuple(channels)
        changes["input_labels"] = tuple(f"m{i + 1}" for i in range(len(channels)))
    if labels is not None:
        # empty labels are kept, so the settings check reports them
        changes["input_labels"] = tuple(label.strip() for label in labels.split(","))
    if changes:
        return settings.replace(**changes)
    return settings


def _save_channel(
    out: Path,
    ir: ImpulseResponse,
    fr: FrequencyResponse,
    analysis: RoomAnalysis,
    mark: str | None = None,
    source: bool = False,
) -> None:
    """Write the files of one channel into ``out`` (created if missing).

    ``impulse_response.h5`` and ``frequency_response.h5`` (``data.save``),
    :func:`_write_analysis` with ``{"device": ir.metadata.device}``,
    ``transfer_function.png`` (:func:`_plot_fr`) and ``impulse_response.png``
    (:func:`_plot_ir` with the analysis), as for a single-channel measurement. ``mark``:
    text of the plot mark (``AppSettings.plot_mark``; None: "Re:sonus"). ``source``:
    also ``source_calibration.h5`` (``source_calibration(ir)``, STI input, D-31).
    """
    out.mkdir(parents=True, exist_ok=True)
    data.save(out.joinpath("impulse_response.h5"), ir)
    if source:
        data.save(out.joinpath("source_calibration.h5"), source_calibration(ir))
    data.save(out.joinpath("frequency_response.h5"), fr)
    _write_analysis(analysis, out, {"device": ir.metadata.device}, mark)
    _plot_fr(fr, out.joinpath("transfer_function.png"), mark)
    _plot_ir(ir, out.joinpath("impulse_response.png"), analysis, mark)


def _write_analysis(
    analysis: RoomAnalysis, out: Path, extra: dict[str, Any], mark: str | None = None
) -> None:
    """Write analysis.json (with ``extra`` keys), analysis.csv, analysis.h5, reverberation.png.

    ``mark``: text of the plot mark (``AppSettings.plot_mark``; None: "Re:sonus").
    """
    out.mkdir(parents=True, exist_ok=True)
    d = analysis_to_dict(analysis) | extra
    out.joinpath("analysis.json").write_text(
        json.dumps(d, indent=2, allow_nan=False), encoding="utf-8"
    )
    out.joinpath("analysis.csv").write_text(analysis_to_csv(analysis), encoding="utf-8")
    meta = Metadata(
        APP_NAME,
        __version__,
        description=str(extra.get("source", "")),
        device=str(extra.get("device", "")),
    )
    data.save(out.joinpath("analysis.h5"), analysis_to_band_results(analysis, meta))
    view = MplPlotView(brand_text=mark)
    setup_rt_axes(view)
    for name in ("EDT", "T20", "T30"):
        show_rt(view, analysis, name, name, name)
    band_ticks(view, analysis.bands.nominal, analysis.bands.fraction)
    view.save(str(out.joinpath("reverberation.png")))


def _summary(analysis: RoomAnalysis) -> dict[str, Any]:
    d = analysis_to_dict(analysis)
    return {
        "bands_hz": d["bands"]["nominal_hz"],
        "EDT": d["values"]["EDT"],
        "T20": d["values"]["T20"],
        "T30": d["values"]["T30"],
        "flags": d["flags"],
    }


def _plot_ir(
    ir: ImpulseResponse,
    path: Path,
    analysis: RoomAnalysis | None = None,
    mark: str | None = None,
) -> None:
    view = MplPlotView(brand_text=mark)
    setup_ir_axes(view)
    show_ir(view, ir, "ir", "Impulse response")
    if analysis is not None:
        show_decay(
            view,
            analysis.broadband.curve,
            analysis.onset / ir.sample_rate,
            "decay",
            "Decay curve",
        )
    view.save(str(path))


def _plot_fr(fr: FrequencyResponse, path: Path, mark: str | None = None) -> None:
    view = MplPlotView(brand_text=mark)
    setup_response_axes(view)
    show_response(view, fr, "fr", fr.metadata.description or "measurement")
    view.save(str(path))


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; returns the exit status."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "calibrate":
            return cli_sti.calibrate_command(args)
        if args.command == "record-noise":
            return cli_sti.record_noise_command(args)
        if args.command == "sti":
            return cli_sti.sti_command(args)

        if args.command == "settings":
            AppSettings().save(args.path)
            return 0

        if args.command == "plot":
            mark = _settings(args.settings).plot_mark
            results = [data.load(p) for p in args.result]
            bands = [res for res in results if isinstance(res, BandResults)]
            if len(bands) == len(results):
                quantities = args.quantity or [default_quantity(bands[0])]
                for path, res in zip(args.result, bands, strict=True):
                    for q in quantities:
                        if q not in res.values:
                            raise ValueError(f"{path}: no quantity {q!r}")
                units = {res.units[q] for res in bands for q in quantities}
                if len(units) > 1:
                    raise ValueError("quantities have different units")
                view = MplPlotView(brand_text=mark)
                setup_band_axes(view, ", ".join(quantities), units.pop())
                for i, (path, res) in enumerate(zip(args.result, bands, strict=True)):
                    name = res.metadata.description or path.stem
                    for q in quantities:
                        label = f"{name}: {q}" if len(bands) > 1 else q
                        show_band_results(view, res, q, f"{i}:{q}", label)
                band_ticks(view, bands[0].center, bands[0].fraction)
                view.save(str(args.output))
                return 0
            if len(results) != 1:
                raise ValueError("impulse and frequency responses are plotted one file at a time")
            result = results[0]
            if isinstance(result, FrequencyResponse):
                _plot_fr(result, args.output, mark)
            elif isinstance(result, ImpulseResponse):
                _plot_ir(result, args.output, mark=mark)
            else:
                raise ValueError(
                    f"{args.result[0]}: plotting {type(result).__name__} not supported"
                )
            return 0

        if args.command == "average":
            settings = _settings(args.settings)
            if args.bands:
                settings = settings.replace(bands=args.bands)
            paths: list[Path] = args.setup
            if len(paths) == 1 and paths[0].is_file():
                if args.method:
                    raise ValueError("--method applies to measurement folders only")
                setup = load_positions(paths[0])
                description = paths[0].stem
            elif all(q.is_dir() for q in paths):
                if not args.method:
                    raise ValueError("--method is required for measurement folders")
                setup = positions_from_folders(paths, args.method)
                description = ", ".join(q.name for q in paths)
            else:
                raise ValueError("give one setup file or measurement folders")
            avg = average_positions(setup, settings)
            for issue in avg.issues:
                print(f"warning: {issue}", file=sys.stderr)
            if not args.out:
                print(average_to_csv(avg), end="")
                return 0
            args.out.mkdir(parents=True, exist_ok=True)
            args.out.joinpath("average.csv").write_text(average_to_csv(avg), encoding="utf-8")
            meta = Metadata(APP_NAME, __version__, description=description)
            bres = average_to_band_results(avg, meta)
            data.save(args.out.joinpath("average.h5"), bres)
            view = MplPlotView(brand_text=settings.plot_mark)
            setup_band_axes(view, "T20, T30", "s")
            for q in ("T20", "T30"):
                show_band_results(view, bres, q, q, q)
            band_ticks(view, bres.center, bres.fraction)
            view.save(str(args.out.joinpath("average.png")))
            return 0

        if args.command == "import-rew":
            rew = read_rew_rt60(args.rt60)
            args.out.mkdir(parents=True, exist_ok=True)
            for name, part in (("octave", rew.octave), ("third", rew.third)):
                if part is not None:
                    data.save(args.out.joinpath(f"rew_{name}.h5"), part)
            return 0

        if args.command == "compare-rew":
            settings = _settings(args.settings)
            if args.bands:
                settings = settings.replace(bands=args.bands)
            rew = read_rew_rt60(args.rt60)
            ref = rew.octave if settings.bands == "octave" else rew.third
            if ref is None:
                raise ValueError(f"{args.rt60}: no {settings.bands} bands")
            settings = settings.replace(f_min=float(ref.center[0]), f_max=float(ref.center[-1]))
            ir = load_impulse_response(args.ir)
            ours = analysis_to_band_results(
                analyze_ir(ir, settings),
                Metadata(APP_NAME, __version__, description=args.ir.stem),
            )
            comps = compare(ours, ref)
            out_text = comparisons_to_csv(comps)
            if args.out:
                args.out.write_text(out_text, encoding="utf-8")
            else:
                print(out_text, end="")
            for line in summary(comps):
                print(line, file=sys.stderr)
            return 0

        if args.command == "test-levels":
            return _test_levels(args)

        if args.command == "measure":
            settings = _settings(args.settings)
            if args.simulate is not None:
                settings = settings.replace(simulate=args.simulate)
            if args.duration is not None:
                settings = settings.replace(duration=args.duration)
            if args.repetitions is not None:
                settings = settings.replace(repetitions=args.repetitions)
            settings = _input_override(settings, args.inputs, args.input_labels)
            for warning in measurement_warnings(settings):
                print(f"warning: {warning}", file=sys.stderr)
            if args.source_calibration and not args.calibrated:
                build_parser().error("--source-calibration needs --calibrated")
            r = measure(settings, open_device(settings))
            irs = r.impulse_responses
            if args.calibrated:
                store = CalibrationStore.open(args.store or store_path())
                irs = cli_sti.calibrate_irs(settings, irs, store, cli_sti.devices_for(settings))
            src = bool(args.source_calibration)
            out = args.out
            out.mkdir(parents=True, exist_ok=True)
            settings.save(out.joinpath("settings.toml"))
            pairs = list(zip(irs, r.frequency_responses, strict=True))
            device = pairs[0][0].metadata.device
            if len(pairs) == 1:  # one channel: files directly in the output folder
                ir, fr = pairs[0]
                analysis = analyze_ir(ir, settings)
                _save_channel(out, ir, fr, analysis, settings.plot_mark, src)
                if args.json:
                    print(json.dumps({"out": str(out), "device": device} | _summary(analysis)))
                return 0
            # several channels: one subfolder per channel, named by its label s#m# (D-17)
            summaries: dict[str, Any] = {}
            for ir, fr in pairs:
                label = ir.metadata.extra["channel"]["label"]
                analysis = analyze_ir(ir, settings)
                _save_channel(out / label, ir, fr, analysis, settings.plot_mark, src)
                summaries[label] = _summary(analysis)
            if args.json:
                print(json.dumps({"out": str(out), "device": device, "channels": summaries}))
            return 0

        if args.command == "binaural":
            settings = _settings(args.settings)
            left = load_impulse_response(args.left, args.left_channel)
            right = load_impulse_response(args.right, args.right_channel)
            if left.sample_rate != right.sample_rate:
                raise ValueError(
                    f"sample rates differ: {left.sample_rate:g} Hz (left), "
                    f"{right.sample_rate:g} Hz (right)"
                )
            # ISO 3382-1:2009 B.4: "generally measured in octave bands ranging from 125 Hz
            # to 4 000 Hz"
            iacc_bands = band_set(1, *IACC_BAND_RANGE)
            params = binaural_parameters(
                left.data,
                right.data,
                left.sample_rate,
                iacc_bands,
                settings.filter_order,
                settings.reverse_filter,
            )
            text = binaural_to_csv(params, iacc_bands)
            if args.out is None:
                print(text, end="")
                return 0
            args.out.mkdir(parents=True, exist_ok=True)
            args.out.joinpath("binaural.csv").write_text(text, encoding="utf-8")
            meta = Metadata(
                APP_NAME, __version__, description=f"{args.left.name} / {args.right.name}"
            )
            data.save(
                args.out.joinpath("binaural.h5"),
                binaural_to_band_results(params, iacc_bands, left.sample_rate, meta),
            )
            return 0

        if args.command == "analyze":
            settings = _settings(args.settings)
            if args.bands:
                settings = settings.replace(bands=args.bands)
            ir = load_impulse_response(args.ir, args.channel)
            analysis = analyze_ir(ir, settings)
            if args.out:
                _write_analysis(analysis, args.out, {"source": str(args.ir)}, settings.plot_mark)
            if args.json:
                print(json.dumps(analysis_to_dict(analysis) | {"source": str(args.ir)}))
            elif not args.out:
                print(analysis_to_csv(analysis), end="")
            return 0

        # absorption
        settings = _settings(args.settings)
        run = run_absorption(load_setup(args.setup), settings)
        for issue in run.result.issues:
            print(f"warning: {issue}", file=sys.stderr)
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            args.out.joinpath("absorption.json").write_text(
                json.dumps(absorption_to_dict(run.result), indent=2, allow_nan=False),
                encoding="utf-8",
            )
            args.out.joinpath("absorption.csv").write_text(
                absorption_to_csv(run.result), encoding="utf-8"
            )
            data.save(
                args.out.joinpath("absorption.h5"),
                absorption_to_band_results(
                    run.result, Metadata(APP_NAME, __version__, description=args.setup.stem)
                ),
            )
            view = MplPlotView(brand_text=settings.plot_mark)
            setup_absorption_axes(view)
            show_absorption(view, run.result, "alpha", args.setup.stem)
            band_ticks(view, run.result.frequency, 3)  # ISO 354: one-third octaves
            view.save(str(args.out.joinpath("absorption.png")))
        if args.json:
            print(json.dumps(absorption_to_dict(run.result)))
        elif not args.out:
            print(absorption_to_csv(run.result), end="")
        return 0

    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
