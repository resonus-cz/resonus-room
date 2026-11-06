"""Preferences page "Calibration": input offsets, STI speech levels (D-31).

One row per measurement input with its active calibration (resonus-core history, D-C1)
and a "Calibrate…" button (the dialog is the plugin's); the STI group edits the speech
levels at 1 m (a tuple field, not shown by ``settings_page``) and the switch for
calibrated measurements.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final

import gi

gi.require_version("Adw", "1")
gi.require_version("Gtk", "4.0")
from gi.repository import Adw, Gtk  # noqa: E402
from resonus_core.calibration import CalibrationRecord, CalibrationStore  # noqa: E402
from resonus_core.devices import DeviceInfo  # noqa: E402

from resonus_room.core.calibrate import input_profiles  # noqa: E402
from resonus_room.core.settings import AppSettings  # noqa: E402
from resonus_room.core.sti import CUSTOM_EFFORT, VOCAL_EFFORTS  # noqa: E402
from resonus_room.core.sti_inputs import settings_speech_levels  # noqa: E402

NOT_CALIBRATED: Final = "Not calibrated"
LEVELS_ERROR: Final = "speech levels: give 7 numbers in dB, comma-separated (125 Hz-8 kHz)"


# Titles of the vocal efforts (core.sti.VOCAL_EFFORTS); shouted is lowered per ISO 9921
# Annex A (Odeon application note p. 4).
EFFORT_TITLES: Final = {
    "custom": "Custom levels",
    "normal": "Normal",
    "raised": "Raised",
    "loud": "Loud",
    "shouted": "Shouted (−2.9 dB, ISO 9921)",
}


def record_subtitle(record: CalibrationRecord | None) -> str:
    """Return the row subtitle of an input's active record.

    None: :data:`NOT_CALIBRATED`. Else ``f"{record.offset_db:.2f} dB, {record.kind},
    {record.calibrated_utc:%Y-%m-%d %H:%M} UTC"``.
    """
    if record is None:
        return NOT_CALIBRATED
    return f"{record.offset_db:.2f} dB, {record.kind}, {record.calibrated_utc:%Y-%m-%d %H:%M} UTC"


def parse_levels(text: str) -> tuple[float, ...]:
    """Return the speech levels of an entry text: ``()`` for blank text, else 7 floats.

    Items are separated by commas (spaces allowed).

    Raises
    ------
    ValueError
        :data:`LEVELS_ERROR` for a non-number or a count other than 7.
    """
    if not text.strip():
        return ()
    try:
        levels = tuple(float(x) for x in text.split(","))
    except ValueError:
        raise ValueError(LEVELS_ERROR) from None
    if len(levels) != 7:
        raise ValueError(LEVELS_ERROR)
    return levels


def format_levels(levels: Sequence[float]) -> str:
    """Return ``", ".join(f"{x:g}" for x in levels)`` (``""`` for none)."""
    return ", ".join(f"{x:g}" for x in levels)


class CalibrationPage:
    """The page and its rows.

    Parameters
    ----------
    current
        Returns the current settings.
    on_change
        Takes new settings (the plugin's ``_on_settings``).
    store
        The calibration history.
    devices
        Returns the audio devices (for ``input_profiles``).
    calibrate
        Called with the zero-based input index when its "Calibrate…" button is clicked.
    toast
        Shows a short message.

    Attributes
    ----------
    page
        ``Adw.PreferencesPage(title="Calibration",
        icon_name="audio-input-microphone-symbolic")`` holding ``inputs_group`` and
        ``sti_group``.
    inputs_group
        ``Adw.PreferencesGroup(title="Inputs", description="Offsets of the acoustic
        calibrator (IEC 60942), per input")``.
    rows
        The ``Adw.ActionRow`` of each input (title its label, subtitle
        :func:`record_subtitle` of ``store.active(profile)``, suffix
        ``Gtk.Button(label="Calibrate…", valign=Gtk.Align.CENTER)``), in input order.
        Their buttons are kept in ``self._buttons`` (same order). If ``input_profiles``
        raises ``ValueError``, one row ``self._message_row`` with the message as title and
        no button instead (``rows`` and ``_buttons`` are then empty; else
        ``_message_row`` is None).
    sti_group
        ``Adw.PreferencesGroup(title="STI")`` with ``effort_row``, ``levels_row`` and
        ``calibrated_row``.
    effort_row
        ``Adw.ComboRow`` "Vocal effort", one item per ``VOCAL_EFFORTS`` (titles
        :data:`EFFORT_TITLES`), selected ``current().sti_vocal_effort``; a change calls
        ``on_change(current().replace(sti_vocal_effort=...))`` (Q-STI3, D-34).
    levels_row
        ``Adw.EntryRow(title="Speech levels at 1 m [dB], 125 Hz–8 kHz",
        show_apply_button=True)``: with the "custom" effort editable, text
        ``format_levels(current().sti_speech_levels)``; with a table effort insensitive,
        text ``format_levels(settings_speech_levels(current()))`` (the levels used).
        On "apply": ``parse_levels(text)``; valid: ``on_change(current().replace(
        sti_speech_levels=levels))``; invalid: ``toast(str(exc))``.
    calibrated_row
        ``Adw.SwitchRow(title="Calibrated impulse responses", subtitle="Measure in Pa/FS
        with the inputs' calibration")``, active ``current().calibrated``; on
        ``notify::active``: ``on_change(current().replace(calibrated=active))`` if it
        differs.
    """

    def __init__(
        self,
        current: Callable[[], AppSettings],
        on_change: Callable[[AppSettings], None],
        store: CalibrationStore,
        devices: Callable[[], Sequence[DeviceInfo]],
        calibrate: Callable[[int], None],
        toast: Callable[[str], None],
    ) -> None:
        self._current = current
        self._on_change = on_change
        self._store = store
        self._devices = devices
        self._calibrate = calibrate
        self._toast = toast
        self.page = Adw.PreferencesPage(
            title="Calibration", icon_name="audio-input-microphone-symbolic"
        )
        self.inputs_group = Adw.PreferencesGroup(
            title="Inputs", description="Offsets of the acoustic calibrator (IEC 60942), per input"
        )
        self.sti_group = Adw.PreferencesGroup(title="STI")
        self.effort_row = Adw.ComboRow(
            title="Vocal effort",
            subtitle="Speech spectrum at 1 m (ANSI S3.5, Odeon application note)",
            model=Gtk.StringList.new([EFFORT_TITLES[e] for e in VOCAL_EFFORTS]),
        )
        self.effort_row.connect("notify::selected", self._on_effort)
        self.sti_group.add(self.effort_row)
        self.levels_row = Adw.EntryRow(
            title="Speech levels at 1 m [dB], 125 Hz–8 kHz", show_apply_button=True
        )
        self.calibrated_row = Adw.SwitchRow(
            title="Calibrated impulse responses",
            subtitle="Measure in Pa/FS with the inputs' calibration",
        )
        self.levels_row.connect("apply", self._on_apply)
        self.calibrated_row.connect("notify::active", self._on_switch)
        self.sti_group.add(self.levels_row)
        self.sti_group.add(self.calibrated_row)
        self.page.add(self.inputs_group)
        self.page.add(self.sti_group)
        self.rows: list[Adw.ActionRow] = []
        self._buttons: list[Gtk.Button] = []
        self._message_row: Adw.ActionRow | None = None
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the input rows from the current settings and history.

        Removes the old rows from ``inputs_group`` (``remove``), then adds the new ones as
        described for ``rows``. Also sets ``levels_row`` text and ``calibrated_row``
        active from the settings where they differ.
        """
        for row in self.rows:
            self.inputs_group.remove(row)
        if self._message_row is not None:
            self.inputs_group.remove(self._message_row)
        self.rows, self._buttons, self._message_row = [], [], None
        settings = self._current()
        try:
            profiles = input_profiles(settings, self._devices())
        except ValueError as exc:
            self._message_row = Adw.ActionRow(title=str(exc))
            self.inputs_group.add(self._message_row)
        else:
            for i, (label, profile) in enumerate(zip(settings.input_labels, profiles, strict=True)):
                row = Adw.ActionRow(
                    title=label, subtitle=record_subtitle(self._store.active(profile))
                )
                button = Gtk.Button(label="Calibrate…", valign=Gtk.Align.CENTER)
                button.connect("clicked", lambda _b, i=i: self._calibrate(i))
                row.add_suffix(button)
                self.inputs_group.add(row)
                self.rows.append(row)
                self._buttons.append(button)
        k = VOCAL_EFFORTS.index(settings.sti_vocal_effort)
        if self.effort_row.get_selected() != k:
            self.effort_row.set_selected(k)
        custom = settings.sti_vocal_effort == CUSTOM_EFFORT
        self.levels_row.set_sensitive(custom)
        shown = settings.sti_speech_levels if custom else settings_speech_levels(settings)
        text = format_levels(shown)
        if self.levels_row.get_text() != text:
            self.levels_row.set_text(text)
        if self.calibrated_row.get_active() != settings.calibrated:
            self.calibrated_row.set_active(settings.calibrated)

    def _on_apply(self, row: Adw.EntryRow) -> None:
        """Entry row "apply": take valid speech levels, else toast the error."""
        try:
            levels = parse_levels(row.get_text())
        except ValueError as exc:
            self._toast(str(exc))
            return
        self._on_change(self._current().replace(sti_speech_levels=levels))

    def _on_effort(self, row: Adw.ComboRow, _pspec: object) -> None:
        """Combo row: take a changed vocal effort, then show its levels."""
        effort = VOCAL_EFFORTS[row.get_selected()]
        if effort != self._current().sti_vocal_effort:
            self._on_change(self._current().replace(sti_vocal_effort=effort))
            self.refresh()

    def _on_switch(self, row: Adw.SwitchRow, _pspec: object) -> None:
        """Switch row: take a changed ``calibrated`` setting."""
        active = row.get_active()
        if active != self._current().calibrated:
            self._on_change(self._current().replace(calibrated=active))
