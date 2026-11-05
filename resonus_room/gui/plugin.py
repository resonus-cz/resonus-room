"""Resonus Room as an AppPlugin: hosted standalone or inside the viewer bundle.

Header bar: the brand "Re:sonus" (left, Lexend SemiBold), the title "Room" (centre) and
the main menu (D-23). Below it (D-20, a deliberate departure from
the GNOME HIG, user request 2026-11-05) a minimal ribbon with the groups Measure (Measure,
excitation length, repetitions, playback level, device and channels, busy spinner), Files
(Open, Export) and Results (results list: active and overlaid measurements and channels,
D-19); then a tab-like view switcher (Transfer function | Impulse response | Reverberation
time) and the tab content in a frame with a slight accent border. The transfer-function
tab shows magnitude and phase with a shared frequency axis; the impulse-response tab the
IR level, the decay curve and fitted lines of the selected band; the reverberation-time
tab EDT/T20/T30 per band and a table with the fit quality (r², ξ, curvature C,
peak-to-noise ratio, flags; ISO 3382-2:2008 Annex B). Settings in the preferences window.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from gi.repository import Adw, Gdk, Gio, GLib, Gtk, PangoCairo
from numpy.typing import NDArray
from resonus_core.audio import AudioIO, MeasurementAborted
from resonus_core.data import FrequencyResponse, ImpulseResponse, Metadata
from resonus_core.devices import DeviceInfo, default_device, list_devices
from resonus_core.gui.devices import DeviceGroups, device_groups
from resonus_core.gui.host import Action, Host
from resonus_core.gui.preferences import settings_page
from resonus_core.measure import measurement_warnings
from resonus_core.plot import MplPlotView, navigable, set_trace_visible
from resonus_core.style import TOKENS, font_files

from resonus_room import __version__
from resonus_room.core.analysis import PARAMETERS, RoomAnalysis  # noqa: F401
from resonus_room.core.engine import (
    analyze_ir,
    band_impulse_responses,
    measure,
    open_device,
    settings_band_set,
)
from resonus_room.core.export import (
    EXPORT_FORMATS,
    IR_FORMATS,
    export_channels,
    export_results,
    export_suffix,
)
from resonus_room.core.files import APP_NAME, impulse_to_frequency_response, load_impulse_response  # noqa: F401
from resonus_room.core.levels import (
    InputLevel,
    OutputLevels,
    combined_levels,
    level_test,
    meter_labels,
    pair_labels,
)
from resonus_room.core.positions import label_from_filename
from resonus_room.core.room_average import average_entries
from resonus_room.core.session import AVERAGE_COLOR, Entry, Session
from resonus_room.core.settings import AppSettings, load_settings
from resonus_room.gui.meter import LevelMeters
from resonus_room.gui.results import ResultsList
from resonus_room.plotting import (
    EMPTY_FREQUENCY_LIMITS,
    EMPTY_LEVEL_LIMITS,
    EMPTY_MAGNITUDE_LIMITS,
    EMPTY_TIME_LIMITS,
    PHASE_LIMITS,
    band_ticks,
    empty_limits,
    empty_rt_axes,
    hide_legend,
    keep_traces,
    peak_delay,
    setup_ir_axes,
    setup_phase_axes,
    setup_response_axes,
    setup_rt_axes,
    show_average,
    show_band_ir,
    show_decay,  # noqa: F401
    show_fit,  # noqa: F401
    show_ir,  # noqa: F401
    show_parameters,
    show_phase_entry,
    show_response_entry,
    show_rt_entry,
    show_time_windows,
    time_windows,
    update_legend,
)

# AppSettings fields that change the band analysis
# (not the measurement); changing one re-analyses the shown IR.
# Preferences pages (D-17): settings groups per page; the groups "Device" and
# "Channels" are shown by resonus_core.gui.devices.device_groups on the page "Device".
MEASUREMENT_GROUPS = ("Signal", "Processing", "Simulation")
ANALYSIS_GROUPS = ("Analysis", "Plots")

IR_TRACES = (
    "Impulse response",
    "Decay curve",
    "EDT fit",
    "T20 fit",
    "T30 fit",
)  # trace labels of the impulse-response tab, toggled by check buttons
ANALYSIS_FIELDS = (
    "bands",
    "f_min",
    "f_max",
    "filter_order",
    "reverse_filter",
    "decay_method",
    "position_method",
)
# Ribbon below the header bar (D-20): group captions, left to right.
RIBBON_GROUPS = ("Measure", "Device", "Files", "Results")
LEVEL_TOOLTIP = (
    "Playback level of the excitation, dB re full scale (amplitude 1): peak for sweeps "
    "and MLS, RMS for noise (AES17 dBFS of noise = this value + 3.01 dB)"
)
# Ribbon, captions and the accent frame around the tab content (D-20, design choice).
APP_CSS = """
.resonus-ribbon { padding: 6px 12px 3px 12px; }
.resonus-ribbon .ribbon-caption { font-size: smaller; opacity: 0.6; }
/* Stop while measuring: solid red like the solid blue Measure (D-22) */
.resonus-ribbon button.destructive-action {
  background-color: var(--destructive-bg-color);
  color: var(--destructive-fg-color);
}
/* brand in the header bar: Lexend SemiBold, the weight of the plot mark (D1, D-23);
   5px: the text starts at the Measure button's left edge (header 6px + 1px side
   bearing + 5px = ribbon padding 12px) */
.resonus-brand { font-family: Lexend; font-weight: 600; font-size: 1.2em; margin-left: 5px; }
.resonus-panel {
  border: 1px solid color-mix(in srgb, var(--accent-bg-color) 45%, transparent);
  border-radius: 8px;
}
"""
# Header bar (user request 2026-11-05, D-23): brand left, app name without it centred.
HEADER_TITLE = "Room"
BAND_LABELS = ("Octave", "1/3 octave")
BAND_VALUES = ("octave", "third")
TABLE_COLUMNS = (
    "Band [Hz]", "EDT [s]", "T20 [s]", "T30 [s]", "r² T20", "ξ T20 [‰]", "ξ T30 [‰]",
    "C [%]", "PNR [dB]", "Flags",
)  # fmt: skip


def channel_label(ir: ImpulseResponse, fallback: str) -> str:
    """Return the label of a result, e.g. ``"s1m1L"``.

    ``ir.metadata.extra["channel"]["label"]`` (resonus-core 0.5.0 measurements) if
    ``extra["channel"]`` is a dict and its ``"label"`` a string, else ``fallback``.
    """
    channel = ir.metadata.extra.get("channel")
    if isinstance(channel, dict) and isinstance(channel.get("label"), str):
        return channel["label"]
    return fallback


def settings_path() -> Path:
    """Return the per-user settings file."""
    return Path(GLib.get_user_config_dir()) / "resonus-room" / "settings.toml"


def _num(value: float, digits: int) -> str:
    """Format a number, "–" for NaN."""
    return "–" if not math.isfinite(value) else f"{value:.{digits}f}"


def table_rows(analysis: RoomAnalysis) -> list[list[str]]:
    """Rows of the fit-quality table, one per band, columns as ``TABLE_COLUMNS``."""
    rows: list[list[str]] = []
    for label, p in zip(analysis.bands.labels, analysis.band_results, strict=True):
        rows.append(
            [
                label,
                _num(p.edt, 2),
                _num(p.t20, 2),
                _num(p.t30, 2),
                _num(p.fits["T20"].r2, 4),
                _num(p.fits["T20"].xi, 1),
                _num(p.fits["T30"].xi, 1),
                _num(p.curvature, 1),
                _num(p.curve.peak_to_noise_db, 1),
                " ".join(p.flags),
            ]
        )
    return rows


class Plugin:
    """Room-acoustic measurement: transfer function, impulse response, reverberation time."""

    id = "resonus_room"
    title = "Resonus Room"  # window, About dialog; the header bar shows HEADER_TITLE
    icon_name = "audio-x-generic-symbolic"

    def __init__(self) -> None:
        path = settings_path()
        self._load_messages: list[str] = []
        try:
            if path.exists():
                self.settings, self._load_messages = load_settings(path)
            else:
                self.settings = AppSettings()
        except ValueError as exc:
            self.settings = AppSettings()
            self._load_messages = [f"settings not loaded ({exc}): using the defaults"]
        self.ir: ImpulseResponse | None = None
        self.fr: FrequencyResponse | None = None
        self.analysis: RoomAnalysis | None = None
        mark = self.settings.plot_mark  # D-24: user-replaceable plot mark
        self.tf_view = MplPlotView(brand_text=mark)
        setup_response_axes(self.tf_view)
        self.tf_view.axes.set_xlabel("")  # the phase plot below carries the frequency label
        # phase below the magnitude in the same figure: shared x axis, one toolbar (D-22);
        # half the magnitude's height (user request 2026-11-05, D-20)
        self.phase_view = self.tf_view.add_panel((2, 1))
        setup_phase_axes(self.phase_view)
        self.ir_view = MplPlotView(brand_text=mark)
        setup_ir_axes(self.ir_view)
        self.rt_view = MplPlotView(brand_text=mark)
        setup_rt_axes(self.rt_view)
        self.stack = Adw.ViewStack()
        self._switcher = Adw.ViewSwitcher(stack=self.stack, policy=Adw.ViewSwitcherPolicy.WIDE)
        self._measure = Gtk.Button(label="Measure")
        self._measure.add_css_class("suggested-action")
        self._measure.set_tooltip_text("Run a measurement (Ctrl+M)")
        self._open = ribbon_button("document-open-symbolic", "Open")
        self._open.set_tooltip_text("Open impulse responses (Ctrl+O)")
        self._length = Gtk.SpinButton.new_with_range(0.1, 600.0, 0.5)
        self._length.set_digits(1)
        self._length.set_value(self.settings.duration)
        self._length.set_tooltip_text("Excitation length (s)")
        self._repetitions = Gtk.SpinButton.new_with_range(1, 1000, 1)
        self._repetitions.set_value(self.settings.repetitions)
        self._repetitions.set_tooltip_text("Number of repetitions (averaged)")
        self._level = Gtk.SpinButton.new_with_range(-100.0, 0.0, 1.0)
        self._level.set_digits(1)
        self._level.set_value(self.settings.level)
        self._level.set_tooltip_text(LEVEL_TOOLTIP)
        self._device = ribbon_button("audio-card-symbolic", "Device")
        self._device.set_tooltip_text("Audio device and channels (Preferences)")
        # level test (D-21): 1 s of white noise at the level, live input meters
        self._test = ribbon_button("audio-volume-high-symbolic", "Test")
        self._test.set_tooltip_text(
            "Play 1 s of white noise at the level (peak) on each output in turn and show the "
            "input levels (Ctrl+T)"
        )
        self._meters = LevelMeters(meter_labels(self.settings))
        self._bands = Gtk.DropDown.new_from_strings(list(BAND_LABELS))
        self._bands.set_selected(BAND_VALUES.index(self.settings.bands))
        self._bands.set_tooltip_text("Band resolution")
        self._band_select = Gtk.DropDown.new_from_strings(["–"])
        # every measurement / file adds one entry per channel; one active, others
        # overlaid in the TF and RT tabs (D-19, replaces the channel drop-down of D-17)
        self.session = Session()
        self._results = ResultsList(
            self.session,
            self._on_active,
            self._on_overlay,
            self._on_remove,
            self.clear,
            on_rename=self._on_rename,
        )
        self._results.button.set_child(
            Adw.ButtonContent(icon_name="view-list-symbolic", label="Measurements")
        )
        self._results.button.add_css_class("flat")
        self._band_select.set_tooltip_text("Band shown in the impulse-response tab")
        self._table = Gtk.Grid(column_spacing=18, row_spacing=4, margin_start=12, margin_end=12)
        self._table_frame = Gtk.Frame(
            visible=False,
            halign=Gtk.Align.END,
            valign=Gtk.Align.START,
            margin_top=12,
            margin_end=12,
        )
        scroller = Gtk.ScrolledWindow(
            min_content_height=160,
            min_content_width=480,
            max_content_height=400,
            propagate_natural_height=True,
        )
        scroller.set_child(self._table)
        self._table_frame.set_child(scroller)
        self._table_toggle = Gtk.ToggleButton(label="Table")
        # room average of the shown entries in the RT tab (D-19)
        self._average_toggle = Gtk.ToggleButton(label="Room average")
        self._average_toggle.set_tooltip_text(
            "Mean of the shown entries (ISO 3382-2:2008 clause 8 a)); positions from the "
            "s#m# labels, Table 1 check with the method set in Preferences"
        )
        self._average_toggle.connect("toggled", lambda _b: self._redraw())
        self._average_messages: list[str] = []
        self._band_irs: NDArray[np.float64] | None = None
        self._ir_toggles: dict[str, Gtk.CheckButton] = {}
        for label in IR_TRACES:
            btn = Gtk.CheckButton(label=label, active=True)
            btn.connect(
                "toggled",
                lambda btn, lbl=label: set_trace_visible(self.ir_view, lbl, btn.get_active()),
            )
            self._ir_toggles[label] = btn
        self._window_artists: list[Any] = []
        self._windows_toggle = Gtk.CheckButton(label="Time windows", active=False)
        self._windows_toggle.set_tooltip_text("Onset, 50 ms, 80 ms and the truncation point t1")
        self._windows_toggle.connect("toggled", lambda _b: self._set_windows_visible())
        self._table_toggle.set_tooltip_text("Show the band table")
        self._table_toggle.connect(
            "toggled", lambda btn: self._table_frame.set_visible(btn.get_active())
        )
        self._trace_toggles: dict[str, Gtk.CheckButton] = {}
        for name in PARAMETERS:
            btn = Gtk.CheckButton(label=name, active=True)
            btn.connect(
                "toggled",
                lambda _btn: show_parameters(self.rt_view, self._visible_parameters()),
            )
            self._trace_toggles[name] = btn
        self._export = ribbon_button("document-save-as-symbolic", "Export")
        self._export.set_sensitive(False)
        self._export.set_tooltip_text("Export (Ctrl+E): the file extension selects the format")
        self._spinner = Adw.Spinner(visible=False)
        self._host: Host | None = None
        self._device_groups: DeviceGroups[AppSettings] | None = None
        self._preferences_dialog: Adw.PreferencesDialog | None = None
        # device of the running measurement (Stop calls its abort(), D-22); None when idle
        self._measure_io: AudioIO | None = None
        self._brand: Gtk.Label | None = None  # header-bar brand (D-23)
        self._empty_views()

    # AppPlugin protocol -------------------------------------------------------------

    def header_widgets(self, host: Host) -> list[Gtk.Widget]:
        """Return the top-panel controls."""
        self._host = host
        for message in self._load_messages:
            host.toast(message)
        self._measure.connect("clicked", lambda _b: self.toggle_measurement())
        self._open.connect("clicked", lambda _b: self.open_dialog())
        self._export.connect("clicked", lambda _b: self.export_dialog())
        self._bands.connect("notify::selected", lambda *_a: self._on_bands())
        self._band_select.connect("notify::selected", lambda *_a: self._show_band())
        self._length.connect("value-changed", lambda *_a: self._on_signal())
        self._repetitions.connect("value-changed", lambda *_a: self._on_signal())
        self._level.connect("value-changed", lambda *_a: self._on_signal())
        self._device.connect("clicked", lambda _b: self.show_device_preferences())
        self._test.connect("clicked", lambda _b: self.start_test())
        # header bar: brand, title and main menu only; the controls are in the ribbon (D-20)
        install_fonts()
        install_css()
        brand = Gtk.Label(label=str(TOKENS["brand_text"]))
        brand.add_css_class("resonus-brand")
        brand.connect("notify::parent", lambda w, _p: _centre_title(w))
        self._brand = brand
        return [brand]

    def main_view(self, host: Host) -> Gtk.Widget:
        """Return the ribbon, the tab switcher and the three tabs (D-20).

        Top to bottom: :meth:`_ribbon`; ``self._switcher`` (left-aligned); ``self.stack``
        in a box with the CSS class ``resonus-panel`` (``APP_CSS``, installed once).
        Transfer-function tab: ``navigable(self.tf_view)``; the phase is a panel of the
        same figure (``add_panel``, D-22), so one toolbar serves both plots.
        """
        install_css()
        self.stack.add_titled_with_icon(
            navigable(self.tf_view),
            "tf",
            "Transfer function",
            "media-playlist-shuffle-symbolic",
        )
        ir_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        bar = Gtk.Box(spacing=6, margin_start=12, margin_top=6)
        bar.append(Gtk.Label(label="Band"))
        bar.append(self._band_select)
        for btn in self._ir_toggles.values():
            bar.append(btn)
        bar.append(self._windows_toggle)
        ir_box.append(bar)
        ir_box.append(navigable(self.ir_view))
        self.stack.add_titled_with_icon(
            ir_box,
            "ir",
            "Impulse response",
            "find-location-symbolic",
        )
        rt_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        bar = Gtk.Box(
            spacing=6,
            margin_start=12,
            margin_end=12,
            margin_top=6,
        )
        for _name, btn in self._trace_toggles.items():
            bar.append(btn)
        spacer = Gtk.Box(hexpand=True)
        bar.append(spacer)
        bar.append(self._average_toggle)
        bar.append(self._bands)
        bar.append(self._table_toggle)
        overlay = Gtk.Overlay(vexpand=True)
        overlay.set_child(navigable(self.rt_view))
        overlay.add_overlay(self._table_frame)
        rt_box.append(bar)
        rt_box.append(overlay)
        self.stack.add_titled_with_icon(
            rt_box,
            "rt",
            "Reverberation time",
            "x-office-spreadsheet-symbolic",
        )
        self.stack.set_vexpand(True)
        tabs = Gtk.Box(margin_start=12, margin_end=12, margin_top=3, margin_bottom=6)
        self._switcher.set_halign(Gtk.Align.START)
        tabs.append(self._switcher)
        panel = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            vexpand=True,
            margin_start=12,
            margin_end=12,
            margin_bottom=12,
            overflow=Gtk.Overflow.HIDDEN,
        )
        panel.add_css_class("resonus-panel")
        panel.append(self.stack)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        box.append(self._ribbon())
        box.append(tabs)
        box.append(panel)
        return box

    def _ribbon(self) -> Gtk.Widget:
        """Return the ribbon: one :func:`ribbon_group` per ``RIBBON_GROUPS`` (D-20, D-21).

        Each group is a ``Gtk.Grid`` of three rows. Measure: ``self._measure`` over all
        rows, beside it "Length [s]" / "Repetitions" / "Level [dBFS]" with their spin
        buttons, one per row, and ``self._spinner``. Device: ``self._device`` (row 1),
        ``self._test`` (rows 2-3), beside them ``self._meters.box`` over all rows. Files:
        ``self._open`` over ``self._export``. Results: ``self._results.button``. Vertical
        separators between the groups.
        """
        measure = ribbon_grid()
        self._measure.set_valign(Gtk.Align.FILL)
        measure.attach(self._measure, 0, 0, 1, 3)
        for row, (text, spin) in enumerate(
            [
                ("Length [s]", self._length),
                ("Repetitions", self._repetitions),
                ("Level [dBFS]", self._level),
            ]
        ):
            label = Gtk.Label(label=text, xalign=0.0)
            label.add_css_class("dim-label")
            measure.attach(label, 1, row, 1, 1)
            measure.attach(spin, 2, row, 1, 1)
        measure.attach(self._spinner, 3, 0, 1, 3)
        device = ribbon_grid()
        device.attach(self._device, 0, 0, 1, 1)
        self._test.set_valign(Gtk.Align.FILL)
        device.attach(self._test, 0, 1, 1, 2)
        device.attach(self._meters.box, 1, 0, 1, 3)
        files = ribbon_grid()
        files.attach(self._open, 0, 0, 1, 1)
        files.attach(self._export, 0, 1, 1, 1)
        files.attach(Gtk.Label(), 0, 2, 1, 1)  # third row: same height, captions on one line
        results = ribbon_grid()
        results.attach(self._results.button, 0, 0, 1, 1)
        results.attach(Gtk.Label(), 0, 1, 1, 1)
        results.attach(Gtk.Label(), 0, 2, 1, 1)
        content: dict[str, Gtk.Widget] = {
            "Measure": measure,
            "Device": device,
            "Files": files,
            "Results": results,
        }
        ribbon = Gtk.Box(spacing=12)
        ribbon.add_css_class("resonus-ribbon")
        for k, title in enumerate(RIBBON_GROUPS):
            if k:
                ribbon.append(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL))
            ribbon.append(ribbon_group(title, content[title]))
        return ribbon

    def preferences(self, host: Host) -> list[Adw.PreferencesPage]:
        """Return the pages Measurement, Device and Analysis (D-17).

        All pages edit one settings object: every page gets ``current=lambda:
        self.settings`` and ``on_change=self._on_settings``, ``on_error=host.toast``.

        1. ``settings_page(..., title="Measurement", icon_name="media-record-symbolic",
           groups=MEASUREMENT_GROUPS)``: applies to the next measurement.
        2. ``Adw.PreferencesPage(title="Device", icon_name="audio-card-symbolic")`` with
           the groups of ``device_groups(self.settings, self._on_settings, current,
           devices, defaults, on_error=host.toast)`` (kept in ``self._device_groups``),
           where ``devices, defaults = self._device_list(host)``.
        3. ``settings_page(..., title="Analysis", icon_name="x-office-spreadsheet-symbolic",
           groups=ANALYSIS_GROUPS)``: re-analyses the shown IR at once
           (:meth:`_on_settings`).
        """

        def current() -> AppSettings:
            return self.settings

        measurement = settings_page(
            self.settings,
            self._on_settings,
            title="Measurement",
            icon_name="media-record-symbolic",
            groups=MEASUREMENT_GROUPS,
            on_error=host.toast,
            current=current,
        )
        devices, defaults = self._device_list(host)
        self._device_groups = device_groups(
            self.settings,
            self._on_settings,
            current,
            devices,
            defaults,
            on_error=host.toast,
        )
        device_page = Adw.PreferencesPage(title="Device", icon_name="audio-card-symbolic")
        for group in self._device_groups.groups:
            device_page.add(group)
        analysis = settings_page(
            self.settings,
            self._on_settings,
            title="Analysis",
            icon_name="x-office-spreadsheet-symbolic",
            groups=ANALYSIS_GROUPS,
            on_error=host.toast,
            current=current,
        )
        return [measurement, device_page, analysis]

    def _device_list(
        self, host: Host
    ) -> tuple[list[DeviceInfo], tuple[DeviceInfo | None, DeviceInfo | None]]:
        """Return the audio devices and the default input/output device.

        ``devices = list_devices()``, ``defaults = (default_device("input", devices),
        default_device("output", devices))``. If PortAudio is not available or fails
        (any ``Exception``: ``OSError`` when the library is missing, ``PortAudioError``),
        ``host.toast(f"audio devices not available: {exc}")`` and ``([], (None, None))``
        (the simulated device still works).
        """
        try:
            devices = list_devices()
            defaults = (
                default_device("input", devices),
                default_device("output", devices),
            )
            return devices, defaults
        except Exception as exc:  # PortAudio missing (OSError) or failing (PortAudioError)
            host.toast(f"audio devices not available: {exc}")
            return [], (None, None)

    def actions(self, host: Host) -> list[Action]:
        """Return the menu actions and shortcuts."""
        return [
            Action("measure", self.toggle_measurement, "Measure / Stop", ["<Control>m"]),
            Action("open", self.open_dialog, "Open Impulse Responses…", ["<Control>o"]),
            Action("export", self.export_dialog, "Export…", ["<Control>e"]),
            Action("test", self.start_test, "Test Levels", ["<Control>t"]),
            Action("clear", self.clear, "Clear Results"),
        ]

    def start_test(self) -> None:
        """Level test off the main loop (D-21): each output in turn, live meters.

        Nothing without a host or while busy. ``self._meters.set_labels(meter_labels(
        settings))`` (resets them); with several outputs a toast ``"Testing s1, s2 (1 s
        each)"``; :meth:`_busy` True. Worker: ``level_test(settings,
        open_device(settings), live)``, where ``live(output, levels)`` hands each block's
        levels to ``self._meters.update`` with ``GLib.idle_add`` (audio thread); peak
        hold and clipping accumulate over the outputs. Done: ``self._meters.update(
        combined_levels(results))``, busy off, toast ``"Clipped: s1m1, s2ref"`` (labels
        of :func:`pair_labels` that clipped) or ``"Level test done"``. Failed: busy off,
        toast ``f"Level test failed: {exc}"``.
        """
        host = self._host
        if host is None or not self._test.get_sensitive():
            return
        settings = self.settings
        self._meters.set_labels(meter_labels(settings))
        outputs = list(settings.output_labels)
        if len(outputs) > 1:
            host.toast(f"Testing {', '.join(outputs)} (1 s each)")
        self._busy(True)

        def show(levels: list[InputLevel]) -> bool:
            self._meters.update(levels)
            return False

        def live(_output: str, levels: list[InputLevel]) -> None:
            GLib.idle_add(show, levels)

        def work() -> list[OutputLevels]:
            return level_test(settings, open_device(settings), live)

        def done(results: list[OutputLevels]) -> None:
            self._busy(False)
            self._meters.update(combined_levels(results))
            clipped = [
                label
                for (_out, levels), labels in zip(results, pair_labels(settings), strict=True)
                for label, v in zip(labels, levels, strict=True)
                if v.clipped
            ]
            host.toast(f"Clipped: {', '.join(clipped)}" if clipped else "Level test done")

        def failed(exc: BaseException) -> None:
            self._busy(False)
            host.toast(f"Level test failed: {exc}")

        host.run_in_thread(work, done, failed)

    def show_device_preferences(self) -> None:
        """Open the preferences on the page "Device" (ribbon button, D-20).

        Nothing before :meth:`header_widgets` (no host). ``dialog =
        Adw.PreferencesDialog()``; add every page of ``self.preferences(host)``;
        ``dialog.set_visible_page(pages[1])`` (the Device page); ``dialog.present(
        host.window)``; the dialog is kept in ``self._preferences_dialog``.
        """
        host = self._host
        if host is None:
            return
        dialog = Adw.PreferencesDialog()
        pages = self.preferences(host)
        for page in pages:
            dialog.add(page)
        dialog.set_visible_page(pages[1])
        dialog.present(host.window)
        self._preferences_dialog = dialog

    # Actions --------------------------------------------------------------------------

    def toggle_measurement(self) -> None:
        """Measure button and Ctrl+M: start a measurement, or stop the running one."""
        if self._measure_io is not None:
            self.stop_measurement()
        else:
            self.start_measurement()

    def stop_measurement(self) -> None:
        """Stop the running measurement (D-22): ``abort()`` its device; no result is kept.

        The button stays insensitive until the measurement thread has ended
        (``failed`` with ``MeasurementAborted``, toast "Measurement stopped").
        """
        if self._measure_io is None:
            return
        self._measure_io.abort()
        self._measure.set_sensitive(False)

    def start_measurement(self) -> None:
        """Measure and analyse off the main loop, then show the results.

        The device is opened here (main loop) and kept in ``self._measure_io`` while the
        measurement runs, so :meth:`stop_measurement` can abort it; the Measure button
        shows "Stop" meanwhile.
        """
        host = self._host
        if host is None or not self._measure.get_sensitive() or self._measure_io is not None:
            return
        settings = self.settings
        # settings that are valid but probably not intended (resonus-core 0.5.0)
        for warning in measurement_warnings(settings):
            host.toast(warning)
        try:
            io = open_device(settings)
        except Exception as exc:
            host.toast(f"Measurement failed: {exc}")
            return
        self._measure_io = io
        self._busy(True, measuring=True)

        Channels = list[tuple[str, ImpulseResponse, FrequencyResponse | None]]

        def work() -> tuple[Channels, RoomAnalysis]:
            r = measure(settings, io)
            channels: Channels = [
                (channel_label(ir, f"channel {k + 1}"), ir, fr)
                for k, (ir, fr) in enumerate(
                    zip(r.impulse_responses, r.frequency_responses, strict=True)
                )
            ]
            return channels, analyze_ir(r.impulse_responses[0], settings)

        def done(result: tuple[Channels, RoomAnalysis]) -> None:
            self._measure_io = None
            self._busy(False)
            self.add_results(self.session.next_measurement_name(), *result)
            host.toast("Measurement done")

        def failed(exc: BaseException) -> None:
            self._measure_io = None
            self._busy(False)
            if isinstance(exc, MeasurementAborted):
                host.toast("Measurement stopped")
            else:
                host.toast(f"Measurement failed: {exc}")

        host.run_in_thread(work, done, failed)

    def open_file(self, path: Path) -> None:
        """Load one impulse response file (see :meth:`open_files`)."""
        self.open_files([path])

    def open_files(self, paths: Sequence[Path]) -> None:
        """Load and analyse impulse response files off the main loop, all in one job.

        Return if ``self._host`` is None or ``paths`` is empty. ``settings =
        self.settings``; ``self._busy(True)``; ``host.run_in_thread(work, done, failed)``:

        - ``work()``: for each path in order, ``ir = load_impulse_response(path)``, ``fr =
          impulse_to_frequency_response(ir)``, ``analysis = analyze_ir(ir, settings)``;
          an ``OSError`` or ``ValueError`` of one file is caught and kept instead of its
          result, so the other files still load. Returns a list of ``(path, (ir, fr,
          analysis) or the exception)``.
        - ``done(results)``: ``self._busy(False)``; for each loaded file, ``name =
          self.session.unique_name(path.name)``, ``fallback = label_from_filename(
          path.name) or name`` (position from an ``s#m#`` file name, D-19) and
          ``self.add_results(name, [(channel_label(ir, fallback), ir, fr)], analysis)``;
          then make the first loaded file's entry active (``self._on_active(key)``) if
          more than one loaded. Toasts: ``f"Cannot open {path.name}: {exc}"`` per failed
          file, then ``f"Opened {path.name}"`` for one loaded file or ``f"Opened {n}
          files"`` for ``n > 1``.
        - ``failed(exc)`` (unexpected error of the whole job): ``self._busy(False)``,
          toast ``f"Cannot open files: {exc}"``.
        """
        host = self._host
        if host is None or not paths:
            return
        settings = self.settings
        self._busy(True)

        def work() -> list[
            tuple[Path, tuple[ImpulseResponse, FrequencyResponse, RoomAnalysis] | Exception]
        ]:
            results: list[
                tuple[Path, tuple[ImpulseResponse, FrequencyResponse, RoomAnalysis] | Exception]
            ] = []
            for path in paths:
                try:
                    ir = load_impulse_response(path)
                    fr = impulse_to_frequency_response(ir)
                    analysis = analyze_ir(ir, settings)
                    results.append((path, (ir, fr, analysis)))
                except (OSError, ValueError) as exc:
                    results.append((path, exc))
            return results

        def done(
            results: list[
                tuple[Path, tuple[ImpulseResponse, FrequencyResponse, RoomAnalysis] | Exception]
            ],
        ) -> None:
            self._busy(False)
            loaded: list[str] = []
            first_key = None
            for path, res in results:
                if isinstance(res, Exception):
                    host.toast(f"Cannot open {path.name}: {res}")
                else:
                    ir, fr, analysis = res
                    name = self.session.unique_name(path.name)
                    fallback = label_from_filename(path.name) or name
                    self.add_results(name, [(channel_label(ir, fallback), ir, fr)], analysis)
                    if first_key is None:
                        first_key = self.session.active
                    loaded.append(path.name)
            if not loaded:
                return
            if len(loaded) > 1 and first_key is not None:
                self._on_active(first_key)
            if len(loaded) == 1:
                host.toast(f"Opened {loaded[0]}")
            else:
                host.toast(f"Opened {len(loaded)} files")

        def failed(exc: BaseException) -> None:
            self._busy(False)
            host.toast(f"Cannot open files: {exc}")

        host.run_in_thread(work, done, failed)

    def open_dialog(self) -> None:
        """Ask for impulse response files (.h5 or .wav); several can be selected."""
        host = self._host
        if host is None:
            return
        flt = Gtk.FileFilter(name="Impulse responses (.h5, .wav)")
        flt.add_pattern("*.h5")
        flt.add_pattern("*.wav")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(flt)
        dialog = Gtk.FileDialog(title="Open Impulse Responses", filters=filters)

        def on_done(d: Gtk.FileDialog, res: Gio.AsyncResult) -> None:
            try:
                files = d.open_multiple_finish(res)
            except GLib.Error:
                return  # cancelled
            paths = []
            for k in range(files.get_n_items()):
                path = files.get_item(k).get_path()
                if path:
                    paths.append(Path(path))
            self.open_files(paths)

        dialog.open_multiple(host.window, None, on_done)

    def export_to(self, path: Path) -> None:
        """Export the current results; the extension of ``path`` selects the format.

        The plot formats write the plot of the visible tab with the overlay (core/export.py,
        D-16). If the active entry's measurement has several channels, the IR formats
        (``IR_FORMATS``) write one file per channel of that measurement
        (``export_channels``, D-17); the other formats export the active entry, its label
        named in the toast. Errors are shown as a toast.
        """
        active = self.session.active_entry
        if self.analysis is None or self.ir is None or active is None:
            return
        group = [(e.label, e.ir) for e in self.session.group(active.name)]
        views = {"tf": self.tf_view, "ir": self.ir_view, "rt": self.rt_view}
        view = views.get(self.stack.get_visible_child_name() or "", self.rt_view)
        m = self.ir.metadata
        meta = Metadata(APP_NAME, __version__, description=m.description, device=m.device)
        several = len(group) > 1
        try:
            if several and export_suffix(path) in IR_FORMATS:
                paths = export_channels(path, group)
                message = f"Exported {len(paths)} files: {', '.join(q.name for q in paths)}"
            else:
                export_results(path, self.analysis, self.ir, view, meta)
                shown = active.label if several else ""
                message = f"Exported {path.name}" + (f" (channel {shown})" if shown else "")
        except (ValueError, OSError) as exc:
            if self._host is not None:
                self._host.toast(f"Export failed: {exc}")
            return
        if self._host is not None:
            self._host.toast(message)

    def export_dialog(self) -> None:
        """Ask for the export file name; its extension selects the format."""
        host = self._host
        if host is None or self.analysis is None or self.ir is None:
            return
        filters = Gio.ListStore.new(Gtk.FileFilter)
        allf = Gtk.FileFilter(name="All supported formats")
        for s in EXPORT_FORMATS:
            allf.add_pattern(f"*{s}")
        filters.append(allf)
        for s, desc in EXPORT_FORMATS.items():
            flt = Gtk.FileFilter(name=f"{desc} (*{s})")
            flt.add_pattern(f"*{s}")
            filters.append(flt)
        description = self.ir.metadata.description
        stem = Path(description).stem if description else "measurement"
        dialog = Gtk.FileDialog(title="Export", filters=filters, initial_name=f"{stem}.png")

        def on_done(d: Gtk.FileDialog, res: Gio.AsyncResult) -> None:
            try:
                f = d.save_finish(res)
            except GLib.Error:
                return  # cancelled
            if f is not None and f.get_path():
                self.export_to(Path(f.get_path()))

        dialog.save(host.window, None, on_done)

    def clear(self) -> None:
        """Remove all results and traces (also "Clear all" of the results list)."""
        self.session.clear()
        self.show_session()

    # Results --------------------------------------------------------------------------

    def add_results(
        self,
        name: str,
        channels: list[tuple[str, ImpulseResponse, FrequencyResponse | None]],
        analysis: RoomAnalysis,
    ) -> None:
        """Add one measurement or file to the session and show it (D-19).

        ``entries = self.session.add(name, channels)`` (``ValueError`` "no channels"
        passes through); ``entries[0].analysis = analysis`` (the analysis of the first
        channel's IR, already computed in the worker); then :meth:`show_session`. The new
        first entry is active; earlier entries stay in the list (and in the overlay).
        """
        entries = self.session.add(name, channels)
        entries[0].analysis = analysis
        self.show_session()

    def show_session(self) -> None:
        """Draw the session: the active entry in all tabs, the overlaid ones in TF and RT.

        1. ``self._results.refresh(self.session)``; ``shown = self.session.shown()``.
        2. Empty: remove all traces of the four views (``keep_traces(view, [])``,
           ``hide_legend(self.phase_view)``), ``self._fill_table([])``, ``self._band_irs =
           None``, ``self._remove_windows()``, ``self.ir = self.fr = self.analysis = None``,
           ``self._export`` insensitive, :meth:`_empty_views`; return.
        3. For each entry of ``shown`` with ``analysis is None``: ``e.analysis =
           analyze_ir(e.ir, self.settings)``.
        4. Transfer function: for each ``e`` in ``shown`` with ``e.fr is not None``:
           ``show_response_entry(self.tf_view, e.fr, f"tf:{e.key}", e.title, e.color)``
           and ``show_phase_entry(self.phase_view, e.fr, f"ph:{e.key}", e.title,
           e.color, delay=peak_delay(e.ir))`` (phase re the IR peak, D-20); then
           ``keep_traces`` with those ids, ``update_legend(self.tf_view)``,
           ``hide_legend(self.phase_view)``.
        5. RT: for each ``e`` in ``shown`` and ``name`` in ``PARAMETERS``:
           ``show_rt_entry(self.rt_view, e.analysis, name, f"rt:{e.key}:{name}", e.title,
           e.color)``; plus the ids of :meth:`_average_traces` (room average, D-19);
           ``keep_traces`` with those ids; ``show_parameters(self.rt_view,
           self._visible_parameters())``.
        6. ``self._show_active(shown[0])``.
        """
        self._results.refresh(self.session)
        shown = self.session.shown()
        if not shown:
            for view in (self.tf_view, self.phase_view, self.ir_view, self.rt_view):
                keep_traces(view, [])
            hide_legend(self.phase_view)
            self._fill_table([])
            self._band_irs = None
            self._remove_windows()
            self.ir = self.fr = self.analysis = None
            self._export.set_sensitive(False)
            self._empty_views()
            return

        for e in shown:
            if e.analysis is None:
                e.analysis = analyze_ir(e.ir, self.settings)

        tf_ids: list[str] = []
        phase_ids: list[str] = []
        for e in shown:
            if e.fr is not None:
                show_response_entry(self.tf_view, e.fr, f"tf:{e.key}", e.title, e.color)
                show_phase_entry(
                    self.phase_view, e.fr, f"ph:{e.key}", e.title, e.color, peak_delay(e.ir)
                )
                tf_ids.append(f"tf:{e.key}")
                phase_ids.append(f"ph:{e.key}")
        keep_traces(self.tf_view, tf_ids)
        keep_traces(self.phase_view, phase_ids)
        update_legend(self.tf_view)
        hide_legend(self.phase_view)

        rt_ids: list[str] = []
        for e in shown:
            assert e.analysis is not None  # step 3
            for name in PARAMETERS:
                show_rt_entry(
                    self.rt_view, e.analysis, name, f"rt:{e.key}:{name}", e.title, e.color
                )
                rt_ids.append(f"rt:{e.key}:{name}")
        rt_ids += self._average_traces(shown)
        keep_traces(self.rt_view, rt_ids)
        show_parameters(self.rt_view, self._visible_parameters())
        active = shown[0].analysis
        assert active is not None  # step 3
        bands = active.bands
        band_ticks(self.rt_view, bands.nominal, bands.fraction)

        self._show_active(shown[0])

    def _average_traces(self, shown: list[Entry]) -> list[str]:
        """Draw the room average of ``shown`` in the RT tab if the toggle is on (D-19).

        Toggle ``self._average_toggle`` active: ``avg = average_entries(shown,
        self.settings.position_method)``; for each ``name`` in ``PARAMETERS``:
        ``show_average(self.rt_view, avg, name, f"rt:avg:{name}", AVERAGE_COLOR)``;
        ``messages = list(avg.issues)``; return the three trace ids. On ``ValueError``
        from ``average_entries``: ``messages = [f"Room average: {exc}"]``, return [].
        Toggle off: ``messages = []``, return [].

        The caller keeps only the returned ids (``keep_traces``). The messages are shown
        as toasts (``self._host.toast``, if the host is set) only when they differ from
        ``self._average_messages``; then ``self._average_messages = messages``. So a
        redraw (band switch, overlay change with the same findings) does not repeat them.
        """
        ids: list[str] = []
        messages: list[str] = []
        if self._average_toggle.get_active():
            try:
                avg = average_entries(shown, self.settings.position_method)
            except ValueError as exc:
                messages = [f"Room average: {exc}"]
            else:
                for name in PARAMETERS:
                    show_average(self.rt_view, avg, name, f"rt:avg:{name}", AVERAGE_COLOR)
                    ids.append(f"rt:avg:{name}")
                messages = list(avg.issues)
        if messages != self._average_messages and self._host is not None:
            for m in messages:
                self._host.toast(m)
        self._average_messages = messages
        return ids

    def _on_rename(self, key: int, label: str) -> None:
        """Results list: new channel label; a rejected label is reported as a toast."""
        try:
            self.session.rename(key, label)
        except ValueError as exc:
            if self._host is not None:
                self._host.toast(f"Not renamed: {exc}")
        self.show_session()

    def _empty_views(self) -> None:
        """Axes of the empty plots (D-20): limits that show where results will appear.

        ``empty_limits(self.tf_view, EMPTY_FREQUENCY_LIMITS, EMPTY_MAGNITUDE_LIMITS)``,
        ``empty_limits(self.phase_view, EMPTY_FREQUENCY_LIMITS, PHASE_LIMITS)``,
        ``empty_limits(self.ir_view, EMPTY_TIME_LIMITS, EMPTY_LEVEL_LIMITS)`` with the
        title "Impulse response", ``empty_rt_axes(self.rt_view,
        settings_band_set(self.settings))``.
        """
        empty_limits(self.tf_view, EMPTY_FREQUENCY_LIMITS, EMPTY_MAGNITUDE_LIMITS)
        empty_limits(self.phase_view, EMPTY_FREQUENCY_LIMITS, PHASE_LIMITS)
        self.ir_view.axes.set_title("Impulse response")
        empty_limits(self.ir_view, EMPTY_TIME_LIMITS, EMPTY_LEVEL_LIMITS)
        empty_rt_axes(self.rt_view, settings_band_set(self.settings))

    def _redraw(self) -> None:
        """Draw the session again if it has entries (room-average toggle)."""
        if self.session.entries:
            self.show_session()

    def _visible_parameters(self) -> list[str]:
        """Names of ``self._trace_toggles`` whose check button is active."""
        return [name for name, btn in self._trace_toggles.items() if btn.get_active()]

    def _show_active(self, entry: Entry) -> None:
        """Show the active entry in the impulse-response tab and the band table.

        ``self.ir, self.fr, self.analysis = entry.ir, entry.fr, entry.analysis``;
        ``self._band_irs = None``; band drop-down model ``["Broadband"] +
        analysis.bands.labels`` with entry 0 selected; ``self._show_band()``;
        ``self._fill_table(table_rows(analysis))``; ``self._export`` sensitive.
        """
        analysis = entry.analysis
        assert analysis is not None  # analysed by show_session
        self.ir, self.fr, self.analysis = entry.ir, entry.fr, analysis
        self._band_irs = None
        model = Gtk.StringList.new(["Broadband"] + analysis.bands.labels)
        self._band_select.set_model(model)
        self._band_select.set_selected(0)
        self._show_band()
        self._fill_table(table_rows(analysis))
        self._export.set_sensitive(True)

    def _on_active(self, key: int) -> None:
        """Results list: make ``key`` active, then :meth:`show_session`."""
        self.session.set_active(key)
        self.show_session()

    def _on_overlay(self, key: int, on: bool) -> None:
        """Results list: tick/untick ``key``, then :meth:`show_session`."""
        self.session.set_overlay(key, on)
        self.show_session()

    def _on_remove(self, key: int) -> None:
        """Results list: remove ``key``, then :meth:`show_session`."""
        self.session.remove(key)
        self.show_session()

    def _show_band(self) -> None:
        """Decay curve and fits of the selected band in the impulse-response tab."""
        a, ir = self.analysis, self.ir
        if a is None or ir is None:
            return
        k = self._band_select.get_selected()
        p = a.broadband if k in (0, Gtk.INVALID_LIST_POSITION) else a.band_results[k - 1]
        start = a.onset / ir.sample_rate
        if k in (0, Gtk.INVALID_LIST_POSITION):
            show_ir(self.ir_view, ir, "ir", "Impulse response")
            title = "Impulse response, broadband"
        else:
            if self._band_irs is None:
                self._band_irs = band_impulse_responses(ir, a, self.settings)
            show_band_ir(
                self.ir_view,
                self._band_irs[:, k - 1],
                ir.sample_rate,
                start,
                "ir",
                "Impulse response",
            )
            title = f"Impulse response, {a.bands.labels[k - 1]} Hz band"
        show_decay(self.ir_view, p.curve, start, "decay", "Decay curve")
        for name in PARAMETERS:
            show_fit(self.ir_view, p.fits[name], start, f"fit-{name}", f"{name} fit")
        self.ir_view.axes.set_title(title)
        for label, button in self._ir_toggles.items():
            set_trace_visible(self.ir_view, label, button.get_active())
        self._remove_windows()
        self._window_artists = show_time_windows(
            self.ir_view, time_windows(start, p.curve, ir.sample_rate)
        )
        self._set_windows_visible()

    def _remove_windows(self) -> None:
        for art in self._window_artists:
            art.remove()
        self._window_artists = []

    def _set_windows_visible(self) -> None:
        for art in self._window_artists:
            art.set_visible(self._windows_toggle.get_active())
        self.ir_view.refresh()

    def _fill_table(self, rows: list[list[str]]) -> None:
        child = self._table.get_first_child()
        while child:
            next_child = child.get_next_sibling()
            self._table.remove(child)
            child = next_child
        for col, title in enumerate(TABLE_COLUMNS):
            head = Gtk.Label(label=title, xalign=1.0 if col else 0.0)
            head.add_css_class("heading")
            self._table.attach(head, col, 0, 1, 1)
        for r, row in enumerate(rows, start=1):
            for col, text in enumerate(row):
                cell = Gtk.Label(label=text, xalign=1.0 if 0 < col < len(row) - 1 else 0.0)
                cell.add_css_class("numeric")
                self._table.attach(cell, col, r, 1, 1)

    # Settings -------------------------------------------------------------------------

    def _on_bands(self) -> None:
        k = self._bands.get_selected()
        if k < len(BAND_VALUES) and BAND_VALUES[k] != self.settings.bands:
            self._on_settings(self.settings.replace(bands=BAND_VALUES[k]))

    def reanalyze(self) -> None:
        """Analyse the shown entries again with the current settings (synchronous)."""
        self.session.invalidate()
        if self.session.entries:
            self.show_session()

    def _on_settings(self, settings: AppSettings) -> None:
        """Take new settings: save them, update the ribbon controls, re-analyse if needed.

        ``self.settings = settings``; save to ``settings_path()``; set ``self._length``,
        ``self._repetitions``, ``self._level`` (``settings.level``) and ``self._bands``
        where they differ; a new ``plot_mark`` goes to the three views (``set_brand``,
        D-24). If an ``ANALYSIS_FIELDS`` field changed: :meth:`reanalyze`,
        and with an empty session also :meth:`_empty_views` (band range of the empty RT
        plot).
        """
        old = self.settings
        self.settings = settings
        path = settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        settings.save(path)
        if self._length.get_value() != settings.duration:
            self._length.set_value(settings.duration)
        if self._repetitions.get_value() != settings.repetitions:
            self._repetitions.set_value(settings.repetitions)
        k = BAND_VALUES.index(settings.bands)
        if self._bands.get_selected() != k:
            self._bands.set_selected(k)
        if self._level.get_value() != settings.level:
            self._level.set_value(settings.level)
        if settings.plot_mark != old.plot_mark:
            for view in (self.tf_view, self.ir_view, self.rt_view):
                view.set_brand(settings.plot_mark)
        if meter_labels(settings) != self._meters.labels:
            self._meters.set_labels(meter_labels(settings))
        if any(getattr(old, f) != getattr(settings, f) for f in ANALYSIS_FIELDS):
            self.reanalyze()
            if not self.session.entries:
                self._empty_views()  # band range of the empty RT plot

    def _on_signal(self) -> None:
        """Take excitation length, repetitions and playback level from the ribbon.

        ``duration = round(self._length.get_value(), 3)``, ``repetitions =
        int(self._repetitions.get_value())``, ``level = round(self._level.get_value(),
        1)``; if any differs from ``self.settings``: ``self._on_settings(
        self.settings.replace(duration=..., repetitions=..., level=...))``.
        """
        duration = round(self._length.get_value(), 3)
        repetitions = int(self._repetitions.get_value())
        level = round(self._level.get_value(), 1)
        s = self.settings
        if (duration, repetitions, level) != (s.duration, s.repetitions, s.level):
            self._on_settings(s.replace(duration=duration, repetitions=repetitions, level=level))

    def _busy(self, busy: bool, measuring: bool = False) -> None:
        """Lock the ribbon during a measurement or file job.

        Measure, Open, length, repetitions, level, Device and Test insensitive while ``busy``;
        the spinner visible while ``busy``; Export sensitive only when not busy and an
        analysis is shown. While ``measuring`` the Measure button stays sensitive and
        becomes a red "Stop" (CSS ``destructive-action``, D-22).
        """
        self._measure.set_sensitive(not busy or measuring)
        self._measure.set_label("Stop" if measuring else "Measure")
        self._measure.set_tooltip_text(
            "Stop the measurement (Ctrl+M)" if measuring else "Run a measurement (Ctrl+M)"
        )
        if measuring:
            self._measure.remove_css_class("suggested-action")
            self._measure.add_css_class("destructive-action")
        else:
            self._measure.remove_css_class("destructive-action")
            self._measure.add_css_class("suggested-action")
        self._open.set_sensitive(not busy)
        self._spinner.set_visible(busy)
        self._export.set_sensitive(not busy and self.analysis is not None)
        self._length.set_sensitive(not busy)
        self._repetitions.set_sensitive(not busy)
        self._level.set_sensitive(not busy)
        self._device.set_sensitive(not busy)
        self._test.set_sensitive(not busy)


_css_installed = False


_fonts_installed = False


def install_fonts() -> None:
    """Make the bundled Lexend fonts (resonus-core) available to GTK, once per process.

    ``PangoCairo.FontMap.get_default().add_font_file(path)`` for each of
    ``resonus_core.style.font_files()`` (Pango >= 1.56); without that call, or if a file
    cannot be added, GTK falls back to the default font.
    """
    global _fonts_installed
    if _fonts_installed:
        return
    font_map = PangoCairo.FontMap.get_default()
    if hasattr(font_map, "add_font_file"):
        for path in font_files():
            try:
                font_map.add_font_file(str(path))
            except GLib.Error:
                pass
    _fonts_installed = True


def _centre_title(widget: Gtk.Widget) -> None:
    """Show ``HEADER_TITLE`` in the centre of the header bar that holds ``widget``.

    The host's header bar shows the window title (``Plugin.title``) by default; the
    window title stays "Resonus Room" for the window list and the About dialog.
    """
    header = widget.get_ancestor(Adw.HeaderBar)
    if isinstance(header, Adw.HeaderBar):
        header.set_title_widget(Adw.WindowTitle(title=HEADER_TITLE))


def install_css() -> None:
    """Load ``APP_CSS`` for the default display, once per process."""
    global _css_installed
    display = Gdk.Display.get_default()
    if _css_installed or display is None:
        return
    provider = Gtk.CssProvider()
    provider.load_from_string(APP_CSS)
    Gtk.StyleContext.add_provider_for_display(
        display, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
    )
    _css_installed = True


def ribbon_button(icon_name: str, label: str) -> Gtk.Button:
    """Flat ribbon button with icon and label (``Adw.ButtonContent``)."""
    button = Gtk.Button(child=Adw.ButtonContent(icon_name=icon_name, label=label))
    button.add_css_class("flat")
    return button


def labelled(text: str, widget: Gtk.Widget) -> Gtk.Widget:
    """Return a box with a dimmed label before ``widget`` (ribbon spin buttons)."""
    box = Gtk.Box(spacing=6, valign=Gtk.Align.CENTER)
    label = Gtk.Label(label=text)
    label.add_css_class("dim-label")
    box.append(label)
    box.append(widget)
    return box


def ribbon_grid() -> Gtk.Grid:
    """Return the grid of one ribbon group (three rows of equal height)."""
    return Gtk.Grid(row_spacing=3, column_spacing=6, row_homogeneous=True)


def ribbon_group(title: str, content: Gtk.Widget) -> Gtk.Widget:
    """Return one ribbon group: ``content`` with the caption ``title`` below."""
    caption = Gtk.Label(label=title)
    caption.add_css_class("ribbon-caption")
    group = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
    group.append(content)
    group.append(caption)
    return group
