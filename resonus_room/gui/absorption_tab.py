"""Absorption tab (setting ``show_absorption``, D-32): ISO 354 from two folders.

The ISO 354 result needs two room averages, T1 of the empty reverberation room and T2
with the specimen (ISO 354:2003 8.1.1). The tab takes one folder of impulse responses
per condition (:mod:`resonus_room.core.iso354_folders`, D-32), the room
and specimen data and the climate of both measurements, and evaluates them with
:func:`resonus_room.core.iso354.run_absorption` off the main loop. Results: alpha_s (or
A_obj for discrete objects) per one-third-octave band, a band table, the findings of the
ISO 354 checks, and export, including the setup file for ``resonus-room absorption``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gtk  # noqa: E402
from resonus_core import data  # noqa: E402
from resonus_core.data import Metadata  # noqa: E402
from resonus_core.gui.overlays import (  # noqa: E402
    band_table,
    exclusive,
    fill_table,
    findings_panel,
)
from resonus_core.plot import MplPlotView, navigable  # noqa: E402

from resonus_room import __version__  # noqa: E402
from resonus_room.core.absorption import (  # noqa: E402
    ALPHA_RESOLUTION,
    AREA_RESOLUTION,
    ISO354_FREQUENCIES,
    AbsorptionResult,
    Climate,
    round_to,
)
from resonus_room.core.files import (  # noqa: E402
    APP_NAME,
    absorption_to_band_results,
    absorption_to_csv,
    absorption_to_dict,
)
from resonus_room.core.iso354 import AbsorptionRun, AbsorptionSetup, run_absorption  # noqa: E402
from resonus_room.core.iso354_folders import (  # noqa: E402
    EMPTY_NAME,
    SPECIMEN_NAME,
    setup_from_folders,
    setup_to_toml,
)
from resonus_room.core.settings import AppSettings  # noqa: E402
from resonus_room.plotting import band_ticks, setup_absorption_axes  # noqa: E402

# The two conditions: key -> name (core.iso354_folders).
CONDITIONS: Final = {"empty": EMPTY_NAME, "specimen": SPECIMEN_NAME}
# Subtitle of a folder row before a folder is chosen.
NO_FOLDER: Final = "Not chosen"
# Plot titles.
TITLE: Final = "Sound absorption (ISO 354:2003)"
NO_RESULT: Final = "Choose both folders, fill in the data and press Evaluate"
EVALUATING: Final = "Evaluating…"
# Text of the findings label without findings.
NO_FINDINGS: Final = "No findings of the ISO 354 checks"
# Trace id of the plotted quantity.
TRACE_ID: Final = "absorption"
# Default atmospheric pressure in kPa, as core.iso354.load_setup (ISO 9613-1:1993
# reference pressure 101.325 kPa).
DEFAULT_PRESSURE: Final = "101.325"
# Climate rows: key -> title.
CLIMATE_ROWS: Final = {
    "temperature": "Temperature [°C]",
    "relative_humidity": "Relative humidity [%]",
    "pressure": "Pressure [kPa]",
}
# Band table columns: plane absorber / discrete objects.
AREA_COLUMNS: Final = ("Band [Hz]", "T1 [s]", "T2 [s]", "A_T [m²]", "αs", "u(αs)")
OBJECT_COLUMNS: Final = ("Band [Hz]", "T1 [s]", "T2 [s]", "A_T [m²]", "A_obj [m²]", "u(A_T) [m²]")
# Export suffixes of the tab -> description.
EXPORT_SUFFIXES: Final = {
    ".png": "Plot (PNG)",
    ".svg": "Plot (SVG)",
    ".pdf": "Plot (PDF)",
    ".json": "ISO 354 result (JSON)",
    ".csv": "Band table (CSV)",
    ".h5": "Band results (HDF5)",
    ".toml": "Setup file for resonus-room absorption (TOML)",
}

# run_in_thread(work, on_done, on_error) of resonus_core.gui.host.Host
RunInThread = Callable[
    [Callable[[], Any], Callable[[Any], None], Callable[[BaseException], None]], None
]


def parse_number(row: Adw.EntryRow, required: bool = True) -> float | None:
    """Return the number in ``row``: ``float(text.strip().replace(",", "."))`` (decimal comma).

    Empty text: None if not ``required``.

    Raises
    ------
    ValueError
        ``f"{row.get_title()}: required"`` (empty and required);
        ``f"{row.get_title()}: not a number: {text!r}"`` (``text`` stripped), also for
        non-finite values.
    """
    text = row.get_text().strip()
    if not text:
        if required:
            raise ValueError(f"{row.get_title()}: required")
        return None
    try:
        value = float(text.replace(",", "."))
    except ValueError:
        value = float("nan")
    if not value == value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{row.get_title()}: not a number: {text!r}")
    return value


def _cell(value: float, fmt: str) -> str:
    return "–" if value != value else format(value, fmt)


def findings_title(count: int) -> str:
    """Label of the findings toggle: ``f"Findings ({count})"``.

    Parameters
    ----------
    count
        Number of folder notes plus ISO 354 issues.

    Returns
    -------
    str
        ``"Findings (0)"`` and so on.
    """
    return f"Findings ({count})"


def absorption_rows(result: AbsorptionResult, objects: bool) -> list[list[str]]:
    """Rows of the band table (columns :data:`OBJECT_COLUMNS` or :data:`AREA_COLUMNS`).

    Per band: ``f"{f:g}"``; T1, T2 with 2 decimals; A_T rounded to 0.1 m² (ISO 354:2003
    8.3, ``round_to``) with 1 decimal; then with ``objects`` A_obj rounded to 0.1 m²
    (1 decimal) and ``area_std`` (2 decimals), else alpha_s rounded to 0.01 (2
    decimals) and ``alpha_std`` (3 decimals). NaN is ``"–"`` (en dash).
    """
    at = round_to(result.area, AREA_RESOLUTION)
    if objects:
        fifth = round_to(result.area_per_object, AREA_RESOLUTION)
        sixth, f5, f6 = result.area_std, ".1f", ".2f"
    else:
        fifth = round_to(result.alpha, ALPHA_RESOLUTION)
        sixth, f5, f6 = result.alpha_std, ".2f", ".3f"
    return [
        [
            f"{f:g}",
            _cell(result.t1[i], ".2f"),
            _cell(result.t2[i], ".2f"),
            _cell(at[i], ".1f"),
            _cell(fifth[i], f5),
            _cell(sixth[i], f6),
        ]
        for i, f in enumerate(result.frequency)
    ]


class AbsorptionTab:
    """Widgets and state of the Absorption tab.

    Parameters
    ----------
    run_in_thread
        The host's ``run_in_thread``.
    toast
        The host's ``toast``.
    settings
        Returns the current settings (analysis settings for the evaluation).
    brand_text
        Plot mark (``AppSettings.plot_mark``).

    Attributes
    ----------
    view
        ``MplPlotView(brand_text=brand_text)``.
    widget
        ``Gtk.Paned`` (horizontal): start child a ``Gtk.ScrolledWindow`` (no horizontal
        scrolling, width 360) with ``form``; end child a vertical box with a bar
        (``findings_toggle``, ``table_toggle``) and ``navigable(view)`` (vexpand) with
        ``table_frame`` and ``findings_frame`` as pop-up overlays
        (:mod:`resonus_room.gui.overlays`), each shown while its toggle is active, one
        at a time (:func:`~resonus_room.gui.overlays.exclusive`).
    form
        Vertical ``Gtk.Box`` (spacing 12, margins 12) of ``Adw.PreferencesGroup`` s:
        "Measurements" (``folder_rows``), "Room and specimen" (``volume_row``,
        ``area_row``, ``objects_row``, ``max_length_row``), "Climate, empty room" and
        "Climate, with specimen" (``climate_rows``), then a box with ``evaluate_button``
        and ``save_button``.
    folders
        ``{"empty": None, "specimen": None}``, then the chosen paths.
    folder_rows
        ``Adw.ActionRow`` per condition, title the condition name, subtitle the folder
        or :data:`NO_FOLDER`; ``folder_buttons`` the "Choose…" suffix buttons (the
        plugin connects them to a folder dialog).
    volume_row, area_row, objects_row, max_length_row
        ``Adw.EntryRow`` s titled "Room volume [m³]", "Specimen area [m²] (plane
        absorber)", "Number of objects (discrete objects)", "Largest room dimension [m]
        (optional)".
    climate_rows
        ``{condition: {key: Adw.EntryRow}}`` with the titles :data:`CLIMATE_ROWS`;
        pressure preset to :data:`DEFAULT_PRESSURE`.
    evaluate_button, save_button
        "Evaluate" (suggested action) and "Export…" (insensitive until a result; the
        plugin opens a save dialog for :meth:`export`).
    findings_label, findings_frame, findings_toggle
        Wrapped, selectable label of the findings in ``findings_frame``; the toggle
        labelled :func:`findings_title`, insensitive without a result.
    table, table_frame, table_toggle
        ``Gtk.Grid`` of the band table, in ``table_frame`` (overlay), shown by ``table_toggle``.
    setup, notes, run
        Of the result shown, or None / ``[]`` / None.
    """

    def __init__(
        self,
        run_in_thread: RunInThread,
        toast: Callable[[str], None],
        settings: Callable[[], AppSettings],
        brand_text: str | None = None,
    ) -> None:
        self._run = run_in_thread
        self._toast = toast
        self._settings = settings
        self.view = MplPlotView(brand_text=brand_text)
        self.folders: dict[str, Path | None] = dict.fromkeys(CONDITIONS)
        self.setup: AbsorptionSetup | None = None
        self.notes: list[str] = []
        self.run: AbsorptionRun | None = None

        self.form = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            margin_start=12,
            margin_end=12,
            margin_top=12,
            margin_bottom=12,
        )
        measurements = Adw.PreferencesGroup(
            title="Measurements",
            description="One folder of impulse responses per condition (s#m# names)",
        )
        self.folder_rows: dict[str, Adw.ActionRow] = {}
        self.folder_buttons: dict[str, Gtk.Button] = {}
        for key, name in CONDITIONS.items():
            row = Adw.ActionRow(title=name.capitalize(), subtitle=NO_FOLDER)
            button = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER)
            row.add_suffix(button)
            measurements.add(row)
            self.folder_rows[key] = row
            self.folder_buttons[key] = button
        self.form.append(measurements)

        room = Adw.PreferencesGroup(
            title="Room and specimen",
            description="Specimen: either the area (plane absorber) or the number of objects",
        )
        self.volume_row = Adw.EntryRow(title="Room volume [m³]")
        self.area_row = Adw.EntryRow(title="Specimen area [m²] (plane absorber)")
        self.objects_row = Adw.EntryRow(title="Number of objects (discrete objects)")
        self.max_length_row = Adw.EntryRow(title="Largest room dimension [m] (optional)")
        for row in (self.volume_row, self.area_row, self.objects_row, self.max_length_row):
            room.add(row)
        self.form.append(room)

        self.climate_rows: dict[str, dict[str, Adw.EntryRow]] = {}
        for key, name in CONDITIONS.items():
            group = Adw.PreferencesGroup(title=f"Climate, {name}")
            rows = {k: Adw.EntryRow(title=t) for k, t in CLIMATE_ROWS.items()}
            rows["pressure"].set_text(DEFAULT_PRESSURE)
            for row in rows.values():
                group.add(row)
            self.climate_rows[key] = rows
            self.form.append(group)

        self.evaluate_button = Gtk.Button(label="Evaluate")
        self.evaluate_button.add_css_class("suggested-action")
        self.evaluate_button.connect("clicked", lambda _b: self.evaluate())
        self.save_button = Gtk.Button(label="Export…", sensitive=False)
        self.save_button.set_tooltip_text(
            "Results (JSON, CSV, HDF5), plot, or the setup file (TOML) for resonus-room absorption"
        )
        actions = Gtk.Box(spacing=6, halign=Gtk.Align.END)
        actions.append(self.save_button)
        actions.append(self.evaluate_button)
        self.form.append(actions)

        side = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER, width_request=360)
        side.set_child(self.form)
        # band table and compliance findings: pop-up panels at the top right of the plot
        # behind "Table" and "Findings (N)" toggles, one at a time, so the plot fills the
        # right part (user, 2026-11-06); findings never block the evaluation (D-35)
        self.table, self.table_frame = band_table()
        self.findings_label, self.findings_frame = findings_panel()
        self.table_toggle = Gtk.ToggleButton(label="Table")
        self.table_toggle.set_tooltip_text("Show the band table")
        self.table_toggle.connect(
            "toggled", lambda btn: self.table_frame.set_visible(btn.get_active())
        )
        self.findings_toggle = Gtk.ToggleButton(label=findings_title(0), sensitive=False)
        self.findings_toggle.set_tooltip_text(
            "Folder notes and ISO 354 checks; shown, not blocking; recorded in the JSON and "
            "HDF5 exports"
        )
        self.findings_toggle.connect(
            "toggled", lambda btn: self.findings_frame.set_visible(btn.get_active())
        )
        exclusive(self.table_toggle, self.findings_toggle)
        bar = Gtk.Box(spacing=6, margin_start=12, margin_end=12, margin_top=6)
        bar.append(Gtk.Box(hexpand=True))
        bar.append(self.findings_toggle)
        bar.append(self.table_toggle)
        overlay = Gtk.Overlay(vexpand=True)
        overlay.set_child(navigable(self.view))
        overlay.add_overlay(self.table_frame)
        overlay.add_overlay(self.findings_frame)
        main = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        main.append(bar)
        main.append(overlay)
        self.widget = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        self.widget.set_start_child(side)
        self.widget.set_end_child(main)
        self.widget.set_resize_start_child(False)
        self.widget.set_shrink_start_child(False)
        self.empty(NO_RESULT)

    def set_folder(self, key: str, path: str | Path) -> None:
        """Choose the folder of condition ``key`` (``"empty"`` or ``"specimen"``).

        ``folders[key] = Path(path)``, row subtitle ``str(path)``; the result shown is
        cleared (:meth:`empty` with :data:`NO_RESULT`).

        Raises
        ------
        KeyError
            For another ``key``.
        """
        if key not in CONDITIONS:
            raise KeyError(key)
        self.folders[key] = Path(path)
        self.folder_rows[key].set_subtitle(str(path))
        self.empty(NO_RESULT)

    def read_inputs(self) -> dict[str, Any]:
        """Keyword arguments of ``setup_from_folders`` from the form (no file access).

        Keys ``empty_folder``, ``specimen_folder``, ``volume``, ``empty_climate``,
        ``specimen_climate``, ``specimen_area``, ``n_objects``, ``max_length``. Numbers by
        :func:`parse_number`: volume, temperatures and humidities required, pressure,
        area, objects and largest dimension optional (empty pressure: 101.325). The
        number of objects must be a whole number. Climates ``Climate(temperature,
        relative_humidity, pressure)``.

        Raises
        ------
        ValueError
            In this order: ``f"choose the folder: {name}"`` for the first condition
            without a folder; from :func:`parse_number` (volume, area, objects, largest
            dimension, then the climate rows, empty room first, in the order of
            :data:`CLIMATE_ROWS`); ``f"{objects_row.get_title()}: not a whole number:
            {text!r}"``.
        """
        for key, name in CONDITIONS.items():
            if self.folders[key] is None:
                raise ValueError(f"choose the folder: {name}")
        volume = parse_number(self.volume_row)
        area = parse_number(self.area_row, required=False)
        objects = parse_number(self.objects_row, required=False)
        max_length = parse_number(self.max_length_row, required=False)
        climates: dict[str, Climate] = {}
        for key in CONDITIONS:
            rows = self.climate_rows[key]
            t = parse_number(rows["temperature"])
            rh = parse_number(rows["relative_humidity"])
            p = parse_number(rows["pressure"], required=False)
            assert t is not None and rh is not None
            climates[key] = Climate(t, rh, float(DEFAULT_PRESSURE) if p is None else p)
        n_objects = None
        if objects is not None:
            if objects != int(objects):
                text = self.objects_row.get_text().strip()
                raise ValueError(f"{self.objects_row.get_title()}: not a whole number: {text!r}")
            n_objects = int(objects)
        return {
            "empty_folder": self.folders["empty"],
            "specimen_folder": self.folders["specimen"],
            "volume": volume,
            "empty_climate": climates["empty"],
            "specimen_climate": climates["specimen"],
            "specimen_area": area,
            "n_objects": n_objects,
            "max_length": max_length,
        }

    def evaluate(self) -> bool:
        """Evaluate the inputs off the main loop; return False if the form is invalid.

        ``kwargs = read_inputs()`` (a ``ValueError`` is a toast, return False). Then
        :meth:`empty` with :data:`EVALUATING`, ``evaluate_button`` insensitive and
        ``run_in_thread(work, done, failed)``: ``work`` returns ``(setup, notes,
        run_absorption(setup, settings))`` with ``setup, notes =
        setup_from_folders(**kwargs)`` and ``settings = self._settings()`` taken before
        the thread starts; ``done`` makes the button sensitive and calls :meth:`show`;
        ``failed(exc)`` makes it sensitive, :meth:`empty` with :data:`NO_RESULT` and
        ``toast(f"Evaluation failed: {exc}")``. Returns True once started.
        """
        try:
            kwargs = self.read_inputs()
        except ValueError as exc:
            self._toast(str(exc))
            return False
        settings = self._settings()
        self.empty(EVALUATING)
        self.evaluate_button.set_sensitive(False)

        def work() -> tuple[AbsorptionSetup, list[str], AbsorptionRun]:
            setup, notes = setup_from_folders(**kwargs)
            return setup, notes, run_absorption(setup, settings)

        def done(result: tuple[AbsorptionSetup, list[str], AbsorptionRun]) -> None:
            self.evaluate_button.set_sensitive(True)
            self.show(*result)

        def failed(exc: BaseException) -> None:
            self.evaluate_button.set_sensitive(True)
            self.empty(NO_RESULT)
            self._toast(f"Evaluation failed: {exc}")

        self._run(work, done, failed)
        return True

    def empty(self, title: str) -> None:
        """Show no result, with ``title`` as the plot title.

        Remove the trace, absorption axes with ``title``, ISO 354 band ticks
        (``band_ticks(view, ISO354_FREQUENCIES, 3)``), empty findings and table,
        ``setup``/``notes``/``run`` reset, ``save_button`` insensitive.
        """
        if TRACE_ID in self.view.trace_ids():
            self.view.remove_trace(TRACE_ID)
        setup_absorption_axes(self.view)
        self.view.axes.set_title(title)
        band_ticks(self.view, ISO354_FREQUENCIES, 3)
        self.view.refresh()
        self.findings_label.set_text("")
        self.findings_toggle.set_label(findings_title(0))
        self.findings_toggle.set_active(False)
        self.findings_toggle.set_sensitive(False)
        self._fill_table((), [])
        self.setup, self.notes, self.run = None, [], None
        self.save_button.set_sensitive(False)

    def show(self, setup: AbsorptionSetup, notes: list[str], run: AbsorptionRun) -> None:
        r"""Show a result.

        ``objects = setup.n_objects is not None``. Plot: :func:`setup_absorption_axes`,
        title :data:`TITLE`; with ``objects`` the y label ``"Equivalent absorption area
        per object A_obj [m²]"`` and ``set_trace(TRACE_ID, frequency, area_per_object,
        label="A_obj")``, else ``set_trace(TRACE_ID, frequency, alpha, label="αs")``;
        ``band_ticks(view, frequency, 3)``; ``refresh()``. ``findings_label``:
        ``"\\n".join(notes + list(result.issues))`` or :data:`NO_FINDINGS`. Table
        :func:`absorption_rows` under :data:`OBJECT_COLUMNS` / :data:`AREA_COLUMNS`.
        ``self.setup, self.notes, self.run`` set; ``save_button`` sensitive.
        """
        result = run.result
        objects = setup.n_objects is not None
        setup_absorption_axes(self.view)
        self.view.axes.set_title(TITLE)
        if objects:
            self.view.axes.set_ylabel("Equivalent absorption area per object A_obj [m²]")
            self.view.set_trace(TRACE_ID, result.frequency, result.area_per_object, label="A_obj")
        else:
            self.view.set_trace(TRACE_ID, result.frequency, result.alpha, label="αs")
        band_ticks(self.view, result.frequency, 3)
        self.view.refresh()
        findings = notes + list(result.issues)
        self.findings_label.set_text("\n".join(findings) if findings else NO_FINDINGS)
        self.findings_toggle.set_label(findings_title(len(findings)))
        self.findings_toggle.set_sensitive(True)
        self._fill_table(
            OBJECT_COLUMNS if objects else AREA_COLUMNS, absorption_rows(result, objects)
        )
        self.setup, self.notes, self.run = setup, list(notes), run
        self.save_button.set_sensitive(True)

    def _fill_table(self, columns: tuple[str, ...], rows: list[list[str]]) -> None:
        """Replace the table: heading ``columns``, then ``rows`` (:func:`fill_table`)."""
        fill_table(self.table, columns, rows)

    def export(self, path: str | Path) -> bool:
        """Export the result shown; the suffix (lower case) selects the format.

        :data:`EXPORT_SUFFIXES`: plots ``view.save(str(path))``; ``.json``
        ``absorption_to_dict(result) | {"notes_folders": notes}`` with ``json.dumps(...,
        indent=2, allow_nan=False)``; ``.csv`` ``absorption_to_csv``; ``.h5``
        ``data.save(path, absorption_to_band_results(result, Metadata(APP_NAME,
        __version__, description="ISO 354 from folders")))``; ``.toml``
        ``setup_to_toml(setup, path)``. Toast ``f"Exported {name}"`` and True; without a
        result toast ``"Nothing to export: evaluate first"``; an unknown suffix toast
        ``f"unsupported file extension {suffix!r}; use one of {', '.join(EXPORT_SUFFIXES)}"``;
        ``OSError``/``ValueError`` toast ``f"Export failed: {exc}"``; each False.
        """
        p = Path(path)
        if self.run is None or self.setup is None:
            self._toast("Nothing to export: evaluate first")
            return False
        suffix = p.suffix.lower()
        if suffix not in EXPORT_SUFFIXES:
            self._toast(
                f"unsupported file extension {suffix!r}; use one of {', '.join(EXPORT_SUFFIXES)}"
            )
            return False
        result = self.run.result
        try:
            if suffix in (".png", ".svg", ".pdf"):
                self.view.save(str(p))
            elif suffix == ".json":
                out = absorption_to_dict(result) | {"notes_folders": list(self.notes)}
                p.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
            elif suffix == ".csv":
                p.write_text(absorption_to_csv(result), encoding="utf-8")
            elif suffix == ".h5":
                meta = Metadata(
                    APP_NAME,
                    __version__,
                    description="ISO 354 from folders",
                    extra={"notes_folders": list(self.notes)},
                )
                data.save(p, absorption_to_band_results(result, meta))
            else:
                setup_to_toml(self.setup, p)
        except (OSError, ValueError) as exc:
            self._toast(f"Export failed: {exc}")
            return False
        self._toast(f"Exported {p.name}")
        return True

    def set_brand(self, text: str | None) -> None:
        """Set the plot mark (``view.set_brand``)."""
        self.view.set_brand(text)
