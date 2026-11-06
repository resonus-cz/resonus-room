"""Preferences page Calibration (D-31)."""

from datetime import UTC, datetime

import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no display available", allow_module_level=True)

from resonus_core.calibration import CalibrationRecord, CalibrationStore  # noqa: E402

from resonus_room.core.calibrate import input_profiles  # noqa: E402
from resonus_room.core.settings import AppSettings  # noqa: E402
from resonus_room.gui.calibration_page import (  # noqa: E402
    LEVELS_ERROR,
    NOT_CALIBRATED,
    CalibrationPage,
    format_levels,
    parse_levels,
    record_subtitle,
)

Adw.init()
T0 = datetime(2026, 11, 5, 14, 3, tzinfo=UTC)


def test_record_subtitle():
    assert record_subtitle(None) == NOT_CALIBRATED == "Not calibrated"
    p = input_profiles(AppSettings(simulate=True), [])[0]
    r = CalibrationRecord.manual(p, 117.014, T0)
    assert record_subtitle(r) == "117.01 dB, manual, 2026-11-05 14:03 UTC"


def test_parse_and_format_levels():
    assert parse_levels("") == () and parse_levels("  ") == ()
    assert parse_levels("60, 61,62 ,63,58,52,47.5") == (60.0, 61.0, 62.0, 63.0, 58.0, 52.0, 47.5)
    for bad in ("60,61", "60,x,1,1,1,1,1", "1,1,1,1,1,1,1,1"):
        with pytest.raises(ValueError) as e:
            parse_levels(bad)
        assert str(e.value) == LEVELS_ERROR
    assert format_levels(()) == ""
    assert (
        format_levels((60.0, 61.5, 62.0, 63.0, 58.0, 52.0, 47.0)) == "60, 61.5, 62, 63, 58, 52, 47"
    )


class Harness:
    def __init__(self, settings):
        self.settings = settings
        self.store = CalibrationStore.unsaved()
        self.calls = []
        self.toasts = []
        self.page = CalibrationPage(
            lambda: self.settings,
            self.on_change,
            self.store,
            lambda: [],
            self.calls.append,
            self.toasts.append,
        )

    def on_change(self, s):
        self.settings = s


def test_rows_and_calibrate_button():
    h = Harness(AppSettings(simulate=True, input_channels=(1, 2), input_labels=("m1", "m2")))
    page = h.page
    assert isinstance(page.page, Adw.PreferencesPage) and page.page.get_title() == "Calibration"
    assert page.inputs_group.get_title() == "Inputs"
    assert [r.get_title() for r in page.rows] == ["m1", "m2"]
    assert [r.get_subtitle() for r in page.rows] == [NOT_CALIBRATED] * 2
    p2 = input_profiles(h.settings, [])[1]
    h.store.append(CalibrationRecord.manual(p2, 100.0, T0))
    page.refresh()
    assert [r.get_title() for r in page.rows] == ["m1", "m2"]  # rebuilt, not doubled
    assert page.rows[1].get_subtitle() == "100.00 dB, manual, 2026-11-05 14:03 UTC"
    button = page._buttons[1]  # noqa: SLF001
    assert button.get_label() == "Calibrate…"
    button.emit("clicked")
    assert h.calls == [1]


def test_rows_without_device():
    h = Harness(AppSettings(simulate=False, input_device=""))
    assert h.page.rows == []
    assert h.page._message_row.get_title().startswith("choose the input device")  # noqa: SLF001


def test_effort_row():
    # Q-STI3, D-34: vocal effort; a table effort shows its levels, read-only
    from resonus_room.core.sti import VOCAL_EFFORTS, vocal_effort_levels

    h = Harness(AppSettings(simulate=True))
    combo, row = h.page.effort_row, h.page.levels_row
    assert combo.get_model().get_n_items() == len(VOCAL_EFFORTS)
    assert VOCAL_EFFORTS[combo.get_selected()] == "normal"
    assert not row.get_sensitive()
    assert row.get_text() == format_levels(vocal_effort_levels("normal"))
    combo.set_selected(VOCAL_EFFORTS.index("raised"))
    assert h.settings.sti_vocal_effort == "raised"
    assert row.get_text() == format_levels(vocal_effort_levels("raised"))
    combo.set_selected(VOCAL_EFFORTS.index("custom"))
    assert h.settings.sti_vocal_effort == "custom"
    assert row.get_sensitive() and row.get_text() == ""


def test_levels_row():
    h = Harness(AppSettings(simulate=True, sti_vocal_effort="custom"))
    row = h.page.levels_row
    assert isinstance(row, Adw.EntryRow) and row.get_text() == ""
    row.set_text("60,60,60,60,60,60,60")
    row.emit("apply")
    assert h.settings.sti_speech_levels == (60.0,) * 7
    row.set_text("60")
    row.emit("apply")
    assert h.toasts == [LEVELS_ERROR] and h.settings.sti_speech_levels == (60.0,) * 7
    h.settings = h.settings.replace(sti_speech_levels=(50.0,) * 7)
    h.page.refresh()
    assert row.get_text() == "50, 50, 50, 50, 50, 50, 50"


def test_calibrated_row():
    h = Harness(AppSettings(simulate=True))
    row = h.page.calibrated_row
    assert isinstance(row, Adw.SwitchRow) and row.get_active() is False
    row.set_active(True)
    assert h.settings.calibrated is True
    h.settings = h.settings.replace(calibrated=False)
    h.page.refresh()
    assert row.get_active() is False and h.settings.calibrated is False
