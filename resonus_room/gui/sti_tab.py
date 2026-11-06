"""STI tab (setting ``show_sti``, docs/PLAN_STI.md item 7, D-31): PROVISIONAL.

Shows the modulation transmission index per octave band and the STI of the active entry,
from three inputs: the entry's calibrated impulse response (unit ``"Pa/FS"``), a source
calibration file and a noise recording (``resonus-room measure --calibrated
--source-calibration`` and ``resonus-room record-noise``, or the GUI later), and the
speech levels at 1 m of the settings. The computation runs off the main loop; results are
cached per input combination.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import gi

gi.require_version("Gtk", "4.0")  # before any gi.repository import of the implementation
from gi.repository import Gtk  # noqa: E402
from resonus_core import data  # noqa: E402
from resonus_core.data import ImpulseResponse, TimeSignal  # noqa: E402
from resonus_core.plot import MplPlotView, navigable  # noqa: E402

from resonus_room.core.sti import BAND_CENTERS  # noqa: E402
from resonus_room.core.sti_inputs import (  # noqa: E402
    NOISE_ROLE,
    SOURCE_ROLE,
    StiReport,
    role,
    sti_at_position,
)
from resonus_room.plotting import band_ticks, setup_band_axes  # noqa: E402

# Cache key: entry name, source path, noise path, speech levels.
_Key = tuple[str | None, Path | None, Path | None, tuple[float, ...]]

# Status texts, in the order :meth:`StiTab.message` checks them (design choice D-31).
NO_ENTRY: Final = "Open or measure a calibrated impulse response"
NOT_CALIBRATED: Final = "Not calibrated: measure with calibrated inputs (unit Pa/FS)"
NO_SOURCE: Final = "Choose the source calibration"
NO_NOISE: Final = "Choose the noise recording"
NO_LEVELS: Final = "Set the custom speech levels at 1 m (Preferences, Calibration)"
TITLE: Final = "Modulation transmission index"
# Trace id of the MTI line.
TRACE_ID: Final = "sti:mti"

# run_in_thread(work, on_done, on_error) of resonus_core.gui.host.Host
RunInThread = Callable[
    [Callable[[], Any], Callable[[Any], None], Callable[[BaseException], None]], None
]


class StiTab:
    """Widgets and state of the STI tab.

    Parameters
    ----------
    run_in_thread
        The host's ``run_in_thread``.
    toast
        The host's ``toast`` (short messages to the user).
    brand_text
        Plot mark (``AppSettings.plot_mark``).

    Attributes
    ----------
    view
        ``MplPlotView(brand_text=brand_text)``, set up with :meth:`empty` at creation.
    widget
        ``Gtk.Box`` (vertical, spacing 6): a bar (``Gtk.Box``, spacing 6, margins start/end
        12, top 6) with ``source_button`` (label ``"Source calibration…"``),
        ``noise_button`` (label ``"Noise…"``), an expanding spacer and ``result_label``
        (empty); below it ``resonus_core.plot.navigable(self.view)`` (vexpand).
    source, noise
        Loaded inputs, or None. ``source_path``, ``noise_path``: their paths.
    speech_levels
        Tuple of 7 floats, or ``()``.
    report
        The report shown, or None.
    """

    def __init__(
        self,
        run_in_thread: RunInThread,
        toast: Callable[[str], None],
        brand_text: str | None = None,
    ) -> None:
        self._run = run_in_thread
        self._toast = toast
        self.view = MplPlotView(brand_text=brand_text)
        self.source: ImpulseResponse | None = None
        self.noise: ImpulseResponse | TimeSignal | None = None
        self.source_path: Path | None = None
        self.noise_path: Path | None = None
        self.speech_levels: tuple[float, ...] = ()
        self.report: StiReport | None = None
        self._name: str | None = None
        self._ir: ImpulseResponse | None = None
        self._cache: dict[_Key, tuple[ImpulseResponse, StiReport | str]] = {}
        self.source_button = Gtk.Button(label="Source calibration…")
        self.noise_button = Gtk.Button(label="Noise…")
        self.result_label = Gtk.Label(label="")
        bar = Gtk.Box(spacing=6, margin_start=12, margin_end=12, margin_top=6)
        bar.append(self.source_button)
        bar.append(self.noise_button)
        bar.append(Gtk.Box(hexpand=True))
        bar.append(self.result_label)
        self.widget = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.widget.append(bar)
        plot = navigable(self.view)
        plot.set_vexpand(True)
        self.widget.append(plot)
        self.empty(NO_ENTRY)

    def load_source(self, path: Path) -> bool:
        """Load a source calibration file; return True on success.

        ``obj = resonus_core.data.load(path)``; it must be an ``ImpulseResponse`` with
        ``role(obj) == SOURCE_ROLE``, else ``toast(f"{path.name}: not a source
        calibration")`` and False (also for an ``OSError`` or ``ValueError`` while
        loading: ``toast(f"{path.name}: {exc}")``). On success ``self.source``,
        ``self.source_path``, ``source_button`` label ``f"Source: {path.name}"``, then
        :meth:`update`.
        """
        obj = self._load(path, SOURCE_ROLE, "source calibration", (ImpulseResponse,))
        if not isinstance(obj, ImpulseResponse):
            return False
        self.source, self.source_path = obj, path
        self.source_button.set_label(f"Source: {path.name}")
        self.update()
        return True

    def load_noise(self, path: Path) -> bool:
        """Load a noise recording; return True on success.

        As :meth:`load_source` with ``NOISE_ROLE``, ``"not a noise recording"``, label
        ``f"Noise: {path.name}"``, ``self.noise`` and ``self.noise_path``; a
        ``TimeSignal`` (resonus-core >= 0.11.0) or, in older files, an ``ImpulseResponse``.
        """
        obj = self._load(path, NOISE_ROLE, "noise recording", (TimeSignal, ImpulseResponse))
        if obj is None:
            return False
        self.noise, self.noise_path = obj, path
        self.noise_button.set_label(f"Noise: {path.name}")
        self.update()
        return True

    def _load(
        self, path: Path, wanted: str, what: str, kinds: tuple[type, ...]
    ) -> ImpulseResponse | TimeSignal | None:
        """Return the file's result if one of ``kinds`` with role ``wanted``; else toast, None."""
        try:
            obj = data.load(path)
        except (OSError, ValueError) as exc:
            self._toast(f"{path.name}: {exc}")
            return None
        if not isinstance(obj, kinds) or not isinstance(obj, (ImpulseResponse, TimeSignal)):
            self._toast(f"{path.name}: not a {what}")
            return None
        if role(obj) != wanted:
            self._toast(f"{path.name}: not a {what}")
            return None
        return obj

    def set_speech_levels(self, levels: tuple[float, ...]) -> None:
        """Set ``self.speech_levels = tuple(levels)``; :meth:`update` if they changed."""
        new = tuple(levels)
        if new != self.speech_levels:
            self.speech_levels = new
            self.update()

    def show_entry(self, name: str | None, ir: ImpulseResponse | None) -> None:
        """Show the entry ``name`` with impulse response ``ir`` (None: no entry); :meth:`update`."""
        self._name, self._ir = name, ir
        self.update()

    def message(self) -> str:
        """Return the first status text that applies, or "" if STI can be computed.

        :data:`NO_ENTRY` (no entry), :data:`NOT_CALIBRATED` (``ir.unit != "Pa/FS"``),
        :data:`NO_SOURCE`, :data:`NO_NOISE`, :data:`NO_LEVELS` (``len(speech_levels) !=
        7``), in this order.
        """
        if self._ir is None:
            return NO_ENTRY
        if self._ir.unit != "Pa/FS":
            return NOT_CALIBRATED
        if self.source is None:
            return NO_SOURCE
        if self.noise is None:
            return NO_NOISE
        if len(self.speech_levels) != 7:
            return NO_LEVELS
        return ""

    def empty(self, title: str) -> None:
        """Show no result, with ``title`` as the plot title.

        Remove the trace, ``setup_band_axes(view, "MTI", "1")``,
        ``view.axes.set_title(title)``, ``view.axes.set_ylim(0, 1)``, ``band_ticks(view,
        BAND_CENTERS, 1)``, ``view.refresh()``; ``result_label`` text "",
        ``self.report = None``.
        """
        if TRACE_ID in self.view.trace_ids():
            self.view.remove_trace(TRACE_ID)
        setup_band_axes(self.view, "MTI", "1")
        self.view.axes.set_title(title)
        self.view.axes.set_ylim(0, 1)
        band_ticks(self.view, BAND_CENTERS, 1)
        self.view.refresh()
        self.result_label.set_text("")
        self.report = None

    def update(self) -> None:
        """Recompute or redraw for the current inputs.

        ``msg = self.message()``; if ``msg``: :meth:`empty` with ``msg``. Else ``key =
        (self._name, self.source_path, self.noise_path, self.speech_levels)``. The cache
        ``self._cache: dict[key, tuple[ImpulseResponse, StiReport | str]]`` holds the
        impulse response it was computed for: a hit needs ``cached_ir is self._ir`` (the
        object is kept, so its identity is not reused). A cached report is drawn with
        :meth:`draw`; a cached error string is shown with :meth:`empty`. Otherwise
        :meth:`empty` with ``"Computing STI…"`` and ``run_in_thread(work, done, failed)``:
        ``work`` returns ``sti_at_position(ir, source, noise, speech_levels)`` for the
        inputs at the call; ``done(report)`` caches ``(ir, report)`` under ``key`` and
        draws it if ``key`` and ``ir`` are still current; ``failed(exc)`` caches ``(ir,
        str(exc))``, and if still current shows it with :meth:`empty` and
        ``toast(f"STI failed: {exc}")``.
        """
        msg = self.message()
        if msg:
            self.empty(msg)
            return
        ir, source, noise = self._ir, self.source, self.noise
        assert ir is not None and source is not None and noise is not None  # message() ""
        levels = self.speech_levels
        key: _Key = (self._name, self.source_path, self.noise_path, levels)
        hit = self._cache.get(key)
        if hit is not None and hit[0] is ir:
            result = hit[1]
            if isinstance(result, str):
                self.empty(result)
            else:
                self.draw(result)
            return
        self.empty("Computing STI…")

        def current() -> bool:
            now = (self._name, self.source_path, self.noise_path, self.speech_levels)
            return now == key and self._ir is ir

        def work() -> StiReport:
            return sti_at_position(ir, source, noise, levels)

        def done(report: StiReport) -> None:
            self._cache[key] = (ir, report)
            if current():
                self.draw(report)

        def failed(exc: BaseException) -> None:
            self._cache[key] = (ir, str(exc))
            if current():
                self.empty(str(exc))
                self._toast(f"STI failed: {exc}")

        self._run(work, done, failed)

    def draw(self, report: StiReport) -> None:
        """Draw ``report``: MTI per band against ``BAND_CENTERS``.

        ``setup_band_axes(view, "MTI", "1")``; ``view.set_trace(TRACE_ID, BAND_CENTERS,
        report.mti, label=self._name)``; title ``f"{self._name}: STI {report.sti:.2f}
        (PROVISIONAL)"``; ``set_ylim(0, 1)``; ``band_ticks(view, BAND_CENTERS, 1)``;
        ``view.refresh()``; ``result_label`` text ``f"STI {report.sti:.2f}"``;
        ``self.report = report``.
        """
        setup_band_axes(self.view, "MTI", "1")
        self.view.set_trace(TRACE_ID, BAND_CENTERS, report.mti, label=self._name or "")
        self.view.axes.set_title(f"{self._name}: STI {report.sti:.2f} (PROVISIONAL)")
        self.view.axes.set_ylim(0, 1)
        band_ticks(self.view, BAND_CENTERS, 1)
        self.view.refresh()
        self.result_label.set_text(f"STI {report.sti:.2f}")
        self.report = report

    def set_brand(self, text: str | None) -> None:
        """Set the plot mark (``view.set_brand``)."""
        self.view.set_brand(text)
