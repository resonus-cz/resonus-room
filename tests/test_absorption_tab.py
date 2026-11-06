"""Absorption tab (D-32).

Synthetic reverberation room as in tests/test_iso354_folders.py (A1 = 6 m2, 10.8 m2
specimen, alpha_s from 0.1 to 0.9; 6 microphones x 2 sources, 12 decays: ISO 354:2003
7.1.4 met). The thread runner is synchronous, so results are shown at once.
"""

import json
import tomllib

import numpy as np
import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no display available", allow_module_level=True)

from resonus_core import data  # noqa: E402
from resonus_core.air import power_attenuation_coefficient  # noqa: E402
from resonus_core.bands import band_set  # noqa: E402
from resonus_core.data import BandResults  # noqa: E402
from resonus_core.gui.overlays import OVERLAY_CLASS  # noqa: E402
from scipy.io import wavfile  # noqa: E402

from resonus_room.core.absorption import Climate  # noqa: E402
from resonus_room.core.iso354 import load_setup, run_absorption  # noqa: E402
from resonus_room.core.settings import AppSettings  # noqa: E402
from resonus_room.core.simulate import exponential_decay_ir  # noqa: E402
from resonus_room.gui.absorption_tab import (  # noqa: E402
    AREA_COLUMNS,
    EVALUATING,
    NO_FINDINGS,
    NO_FOLDER,
    NO_RESULT,
    OBJECT_COLUMNS,
    TITLE,
    TRACE_ID,
    AbsorptionTab,
    absorption_rows,
    findings_title,
    parse_number,
)

Adw.init()

FS = 16000
VOLUME, AREA = 200.0, 10.8
BANDS = band_set(3, 100, 5000)
ALPHA_TRUE = np.linspace(0.1, 0.9, 18)


def t_of(a):
    m = power_attenuation_coefficient(BANDS.center, 20.0, 50.0)
    return 55.3 * VOLUME / (343.0 * (a + 4 * VOLUME * m))


@pytest.fixture(scope="module")
def room(tmp_path_factory):
    base = tmp_path_factory.mktemp("room")
    for kind, t60 in (("empty", t_of(6.0)), ("specimen", t_of(6.0 + ALPHA_TRUE * AREA))):
        folder = base / kind
        folder.mkdir()
        k = 0
        for s in (1, 2):
            for m in range(1, 7):
                h = exponential_decay_ir(
                    FS, 2 * t60.max() + 0.5, t60, BANDS, delay=0.01, seed=10 * k + len(kind)
                )
                wavfile.write(
                    folder / f"s{s}m{m}.wav", FS, (0.5 * h / np.abs(h).max()).astype(np.float32)
                )
                k += 1
    return base / "empty", base / "specimen"


class Runner:
    def __init__(self):
        self.calls = 0

    def __call__(self, work, done, failed):
        self.calls += 1
        try:
            result = work()
        except Exception as exc:
            failed(exc)
        else:
            done(result)


@pytest.fixture
def tab():
    toasts = []
    t = AbsorptionTab(Runner(), toasts.append, AppSettings, brand_text="mark")
    t.toasts = toasts
    return t


def fill(tab, room, area="10,8", objects=""):
    tab.set_folder("empty", room[0])
    tab.set_folder("specimen", room[1])
    tab.volume_row.set_text("200")
    tab.area_row.set_text(area)
    tab.objects_row.set_text(objects)
    for key in ("empty", "specimen"):
        tab.climate_rows[key]["temperature"].set_text("20")
        tab.climate_rows[key]["relative_humidity"].set_text("50")


def test_layout(tab):
    assert isinstance(tab.widget, Gtk.Paned)
    assert tab.folders == {"empty": None, "specimen": None}
    assert [r.get_subtitle() for r in tab.folder_rows.values()] == [NO_FOLDER, NO_FOLDER]
    assert [b.get_label() for b in tab.folder_buttons.values()] == ["Choose…", "Choose…"]
    assert [r.get_text() for r in tab.climate_rows["empty"].values()] == ["", "", "101.325"]
    assert list(tab.climate_rows["specimen"]) == ["temperature", "relative_humidity", "pressure"]
    assert tab.evaluate_button.get_label() == "Evaluate"
    assert not tab.save_button.get_sensitive()
    assert tab.view.axes.get_title() == NO_RESULT
    assert TRACE_ID not in tab.view.trace_ids()


def test_parse_number():
    row = Adw.EntryRow(title="Room volume [m³]")
    row.set_text(" 12,5 ")
    assert parse_number(row) == 12.5
    row.set_text("")
    assert parse_number(row, required=False) is None
    with pytest.raises(ValueError, match=r"^Room volume \[m³\]: required$"):
        parse_number(row)
    for text in ("abc", "nan", "inf"):
        row.set_text(text)
        with pytest.raises(ValueError, match=rf"^Room volume \[m³\]: not a number: '{text}'$"):
            parse_number(row)


def test_read_inputs(tab, room):
    with pytest.raises(ValueError, match="^choose the folder: empty room$"):
        tab.read_inputs()
    tab.set_folder("empty", room[0])
    with pytest.raises(ValueError, match="^choose the folder: with specimen$"):
        tab.read_inputs()
    fill(tab, room)
    tab.max_length_row.set_text("9.5")
    tab.climate_rows["specimen"]["pressure"].set_text("")
    tab.climate_rows["specimen"]["temperature"].set_text("21,5")
    kwargs = tab.read_inputs()
    assert kwargs == {
        "empty_folder": room[0],
        "specimen_folder": room[1],
        "volume": 200.0,
        "empty_climate": Climate(20.0, 50.0, 101.325),
        "specimen_climate": Climate(21.5, 50.0, 101.325),
        "specimen_area": 10.8,
        "n_objects": None,
        "max_length": 9.5,
    }
    tab.objects_row.set_text("2.5")
    with pytest.raises(ValueError, match=r"^Number of objects .*: not a whole number: '2.5'$"):
        tab.read_inputs()
    tab.objects_row.set_text("3")
    assert tab.read_inputs()["n_objects"] == 3
    tab.climate_rows["empty"]["relative_humidity"].set_text("")
    with pytest.raises(ValueError, match=r"^Relative humidity \[%\]: required$"):
        tab.read_inputs()


def test_evaluate_and_show(tab, room):
    fill(tab, room)
    assert tab.evaluate()
    assert tab.run is not None and tab.setup is not None
    assert tab.notes == []
    assert tab.evaluate_button.get_sensitive() and tab.save_button.get_sensitive()
    expected = run_absorption(tab.setup, AppSettings()).result
    np.testing.assert_allclose(tab.run.result.alpha, expected.alpha)
    res = tab.run.result
    assert np.all(np.abs(res.alpha - ALPHA_TRUE) <= 4 * res.alpha_std + 0.04)
    assert TRACE_ID in tab.view.trace_ids()
    assert tab.view.axes.get_title() == TITLE
    assert tab.findings_label.get_text() == ("\n".join(res.issues) if res.issues else NO_FINDINGS)
    heads = [tab.table.get_child_at(c, 0).get_label() for c in range(6)]
    assert heads == list(AREA_COLUMNS)
    assert tab.table.get_child_at(0, 1).get_label() == "100"
    assert tab.table.get_child_at(4, 18).get_label() == absorption_rows(res, False)[17][4]
    # a new folder clears the result
    tab.set_folder("specimen", room[1])
    assert tab.run is None and not tab.save_button.get_sensitive()
    assert tab.view.axes.get_title() == NO_RESULT


def test_evaluate_errors(tab, room, tmp_path):
    assert not tab.evaluate()
    assert tab.toasts == ["choose the folder: empty room"]
    assert tab._run.calls == 0
    fill(tab, room)
    tab.set_folder("specimen", tmp_path)  # empty folder: fails in the worker
    assert tab.evaluate()
    assert tab.toasts[-1].startswith("Evaluation failed: ")
    assert "no impulse responses" in tab.toasts[-1]
    assert tab.run is None and tab.evaluate_button.get_sensitive()
    assert tab.view.axes.get_title() == NO_RESULT


def test_objects(tab, room):
    fill(tab, room, area="", objects="4")
    assert tab.evaluate()
    res = tab.run.result
    assert np.all(np.isnan(res.alpha))
    heads = [tab.table.get_child_at(c, 0).get_label() for c in range(6)]
    assert heads == list(OBJECT_COLUMNS)
    assert "A_obj" in tab.view.axes.get_ylabel()
    rows = absorption_rows(res, True)
    assert rows[0][4] == f"{np.round(res.area_per_object[0] / 0.1) * 0.1:.1f}"


def test_absorption_rows_format(tab, room):
    fill(tab, room)
    tab.evaluate()
    res = tab.run.result
    rows = absorption_rows(res, False)
    assert len(rows) == 18
    assert rows[0][0] == "100" and rows[-1][0] == "5000"
    assert rows[3][1] == f"{res.t1[3]:.2f}"
    assert rows[3][4] == f"{np.round(res.alpha[3] / 0.01) * 0.01:.2f}"
    assert rows[3][5] == f"{res.alpha_std[3]:.3f}"
    import dataclasses

    nan = dataclasses.replace(res, t2=np.full(18, np.nan))
    assert absorption_rows(nan, False)[0][2] == "–"


def test_findings_and_notes(tab, room, tmp_path):
    # specimen folder with 3 files (2 microphones x 2 sources = 4 combinations, 7.1.4)
    small = tmp_path / "small"
    small.mkdir()
    for name in ("s1m1.wav", "s1m2.wav", "s2m1.wav"):
        (small / name).write_bytes((room[1] / name).read_bytes())
    fill(tab, room)
    tab.set_folder("specimen", small)
    assert tab.evaluate()
    text = tab.findings_label.get_text()
    assert tab.notes and all(n in text for n in tab.notes)
    assert "with specimen: 7.1.4: 4 decay curves < 12" in text


def test_non_compliant_setup_is_evaluated(tab, room, tmp_path):
    # user, 2026-11-06: compliance issues are shown and recorded but never block. A 1 m3
    # room with a 0.5 m2 specimen fails 6.1.4 / 6.2.1.1 and is still evaluated.
    fill(tab, room, area="0,5")
    tab.volume_row.set_text("1")
    assert tab.evaluate()
    assert tab.toasts == []
    assert TRACE_ID in tab.view.trace_ids()
    assert tab.view.axes.get_title() == TITLE
    issues = list(tab.run.result.issues)
    assert any("6.2.1.1" in i for i in issues)
    assert tab.findings_label.get_text() == "\n".join(tab.notes + issues)
    assert tab.findings_toggle.get_label() == findings_title(len(tab.notes) + len(issues))
    assert tab.findings_toggle.get_sensitive()
    assert tab.save_button.get_sensitive()
    assert tab.export(tmp_path / "a.h5")
    extra = data.load(tmp_path / "a.h5").metadata.extra
    assert list(extra["issues"]) == issues
    assert list(extra["notes_folders"]) == tab.notes
    tab.findings_toggle.set_active(True)
    tab.empty(NO_RESULT)
    assert tab.findings_toggle.get_label() == "Findings (0)"
    assert not tab.findings_toggle.get_sensitive() and not tab.findings_frame.get_visible()


def test_findings_title():
    assert findings_title(0) == "Findings (0)"
    assert findings_title(3) == "Findings (3)"


def test_export(tab, room, tmp_path):
    assert not tab.export(tmp_path / "a.json")
    assert tab.toasts[-1] == "Nothing to export: evaluate first"
    fill(tab, room)
    tab.evaluate()
    for name in ("a.png", "a.json", "a.csv", "a.h5", "setup.toml"):
        assert tab.export(tmp_path / name), tab.toasts
        assert tab.toasts[-1] == f"Exported {name}"
        assert (tmp_path / name).stat().st_size > 0
    got = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    assert got["standards"] == ["ISO 354:2003"] and got["notes_folders"] == []
    assert isinstance(data.load(tmp_path / "a.h5"), BandResults)
    assert tomllib.loads((tmp_path / "setup.toml").read_text())["volume"] == 200.0
    back = load_setup(tmp_path / "setup.toml")
    assert [f.resolve() for f in back.empty_files] == [f.resolve() for f in tab.setup.empty_files]
    assert not tab.export(tmp_path / "a.xlsx")
    assert tab.toasts[-1].startswith("unsupported file extension '.xlsx'; use one of .png")
    assert not tab.export(tmp_path / "missing" / "a.csv")
    assert tab.toasts[-1].startswith("Export failed: ")


def test_set_folder_unknown_key(tab, tmp_path):
    with pytest.raises(KeyError):
        tab.set_folder("other", tmp_path)


def test_evaluating_title(room):
    seen = []

    def runner(work, done, failed):
        seen.append(tab.view.axes.get_title())
        seen.append(tab.evaluate_button.get_sensitive())

    tab = AbsorptionTab(runner, lambda _t: None, AppSettings)
    fill(tab, room)
    assert tab.evaluate()
    assert seen == [EVALUATING, False]


def test_table_overlay_toggle(tab):
    # the band table is an overlay behind a "Table" toggle, so the plot keeps the height
    assert tab.table_toggle.get_label() == "Table"
    assert not tab.table_frame.get_visible()
    tab.table_toggle.set_active(True)
    assert tab.table_frame.get_visible()
    assert isinstance(tab.table_frame.get_parent(), Gtk.Overlay)


def test_findings_popup_and_table_width(tab, room):
    # user, 2026-11-06: findings a pop-up like the Table, toggled next to it, one at a
    # time; the plot overlay is the only child below the bar, so it fills the right part
    fill(tab, room)
    tab.evaluate()
    bar = tab.table_toggle.get_parent()
    assert tab.findings_toggle.get_parent() is bar
    assert tab.findings_toggle.get_next_sibling() is tab.table_toggle
    assert isinstance(tab.findings_frame.get_parent(), Gtk.Overlay)
    assert bar.get_next_sibling() is tab.findings_frame.get_parent()
    assert bar.get_next_sibling().get_next_sibling() is None
    for frame in (tab.table_frame, tab.findings_frame):
        assert frame.has_css_class(OVERLAY_CLASS)
    tab.findings_toggle.set_active(True)
    assert tab.findings_frame.get_visible() and not tab.table_frame.get_visible()
    tab.table_toggle.set_active(True)
    assert tab.table_frame.get_visible() and not tab.findings_frame.get_visible()
    # the panel requests the table's full natural width (all columns, no scrolling)
    grid_width = tab.table.measure(Gtk.Orientation.HORIZONTAL, -1)[1]
    assert grid_width > 300
    assert tab.table_frame.measure(Gtk.Orientation.HORIZONTAL, -1)[1] >= grid_width
