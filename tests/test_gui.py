import json
import threading
import time

import numpy as np
import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no display available", allow_module_level=True)

from resonus_core.audio import SimulatedIO  # noqa: E402
from resonus_core.bands import band_set  # noqa: E402
from resonus_core.gui.host import StandaloneHost  # noqa: E402
from scipy.io import wavfile  # noqa: E402

import resonus_room.gui.plugin as plugin_module  # noqa: E402
from resonus_room.core.engine import open_device  # noqa: E402
from resonus_room.core.simulate import exponential_decay_ir  # noqa: E402

Adw.init()


def wait_for(cond, seconds=60):
    deadline = time.monotonic() + seconds
    ctx = GLib.MainContext.default()
    while not cond() and time.monotonic() < deadline:
        ctx.iteration(False)
        time.sleep(0.01)
    return cond()


@pytest.fixture(scope="module")
def app():
    application = Adw.Application(
        application_id="cz.resonus.RoomTest", flags=Gio.ApplicationFlags.NON_UNIQUE
    )
    application.register(None)
    return application


@pytest.fixture
def hosted(app, tmp_path, monkeypatch):
    monkeypatch.setattr(plugin_module, "settings_path", lambda: tmp_path / "settings.toml")
    plugin = plugin_module.Plugin()
    host = StandaloneHost(app, plugin, "test")
    plugin.settings = plugin.settings.replace(
        duration=0.5, tail=1.0, ir_length=1.0, sim_decay_time=0.4
    )
    yield plugin, app
    host.window.destroy()


def test_measure_fills_all_tabs(hosted):
    plugin, app = hosted
    app.activate_action("measure", None)
    assert wait_for(lambda: plugin.analysis is not None)
    assert plugin.tf_view.trace_ids() == ["tf:0"]
    assert plugin.ir_view.trace_ids()[:2] == ["ir", "decay"]
    assert "fit-T20" in plugin.ir_view.trace_ids()
    assert plugin.rt_view.trace_ids() == ["rt:0:EDT", "rt:0:T20", "rt:0:T30"]
    rows = plugin_module.table_rows(plugin.analysis)
    assert len(rows) == 6 and rows[0][0] == "125"
    assert len(rows[0]) == len(plugin_module.TABLE_COLUMNS)
    # band selector of the IR tab: broadband + 6 bands
    assert plugin._band_select.get_model().get_n_items() == 7
    plugin._band_select.set_selected(3)
    assert "decay" in plugin.ir_view.trace_ids()
    # switching to one-third octaves re-analyses the IR and saves the setting
    major = plugin.rt_view.axes.xaxis.get_major_locator().tick_values(1, 2)
    assert list(major) == [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]  # band centres
    plugin._bands.set_selected(1)
    assert plugin.settings.bands == "third" and len(plugin.analysis.band_results) == 18
    assert len(plugin.rt_view.axes.xaxis.get_minor_locator().tick_values(1, 2)) == 12
    assert plugin_module.AppSettings.load(plugin_module.settings_path()).bands == "third"
    plugin.clear()
    assert plugin.analysis is None and plugin.rt_view.trace_ids() == []
    assert plugin.tf_view.trace_ids() == [] and plugin.session.entries == []


def test_header_length_and_averages(hosted):
    # excitation length and averages next to Measure (D-11), saved with the settings
    plugin, _ = hosted
    plugin._length.set_value(0.7)
    plugin._repetitions.set_value(3)
    assert plugin.settings.duration == 0.7 and plugin.settings.repetitions == 3
    saved = plugin_module.AppSettings.load(plugin_module.settings_path())
    assert saved.duration == 0.7 and saved.repetitions == 3
    # a change from the preferences page updates the header controls
    plugin._on_settings(plugin.settings.replace(duration=1.5, repetitions=4))
    assert plugin._length.get_value() == 1.5 and plugin._repetitions.get_value() == 4


def test_open_wav(hosted, tmp_path):
    plugin, _ = hosted
    ir = exponential_decay_ir(48000, 1.0, np.full(6, 0.3), band_set(1, 125, 4000), seed=1)
    wavfile.write(tmp_path / "ir.wav", 48000, (0.5 * ir / np.abs(ir).max()).astype(np.float32))
    plugin.open_file(tmp_path / "ir.wav")
    assert wait_for(lambda: plugin.analysis is not None)
    # the transfer function is computed from the loaded IR (user, 2026-11-04)
    assert plugin.fr is not None and plugin.tf_view.trace_ids() == ["tf:0"]
    assert plugin.session.active_entry.title == "ir.wav"
    np.testing.assert_allclose(plugin.fr.values, np.fft.rfft(plugin.ir.data))
    t20 = plugin.analysis.values("T20")
    assert np.all(np.abs(t20[2:] - 0.3) < 0.06)


def test_table_rows_format(hosted):
    plugin, _ = hosted
    ir = exponential_decay_ir(48000, 1.0, np.full(6, 0.3), band_set(1, 125, 4000), seed=1)
    from resonus_room.core.analysis import analyze

    a = analyze(ir, 48000.0, band_set(1, 125, 4000))
    rows = plugin_module.table_rows(a)
    p = a.band_results[2]
    assert rows[2][0] == "500" and rows[2][2] == f"{p.t20:.2f}" and rows[2][-1] == " ".join(p.flags)


def test_preferences_save_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(plugin_module, "settings_path", lambda: tmp_path / "settings.toml")
    plugin = plugin_module.Plugin()
    plugin._on_settings(plugin.settings.replace(level=-20.0))
    assert (tmp_path / "settings.toml").exists()
    assert plugin_module.Plugin().settings.level == -20.0


def test_rt_tab_legend_toggles_and_table_overlay(hosted):
    # RT tab: plot is the main content; EDT/T20/T30 visibility via a legend bar;
    # the table is an overlay behind a toggle (user request, 2026-11-03)
    from matplotlib.backends.backend_gtk4 import NavigationToolbar2GTK4

    plugin, app = hosted
    for view in (plugin.tf_view, plugin.ir_view, plugin.rt_view):
        box = view.widget().get_parent()
        assert isinstance(box.get_first_child(), NavigationToolbar2GTK4)
    assert list(plugin._trace_toggles) == ["EDT", "T20", "T30"]
    assert all(b.get_active() for b in plugin._trace_toggles.values())
    assert not plugin._table_toggle.get_active()
    assert not plugin._table_frame.get_visible()
    assert isinstance(plugin._table_frame.get_parent(), Gtk.Overlay)

    app.activate_action("measure", None)
    assert wait_for(lambda: plugin.analysis is not None)
    plugin._trace_toggles["T20"].set_active(False)

    def state():
        lines = {ln.get_label().split(", ")[0]: ln for ln in plugin.rt_view.axes.get_lines()}
        legend = plugin.rt_view.axes.get_legend()
        texts = [t.get_text().split(", ")[0] for t in legend.get_texts()] if legend else []
        return {k: ln.get_visible() for k, ln in lines.items()}, texts, lines

    visible, texts, lines = state()
    assert visible == {"EDT": True, "T20": False, "T30": True}
    assert texts == ["EDT", "T30"]
    assert plugin.rt_view.trace_ids() == ["rt:0:EDT", "rt:0:T20", "rt:0:T30"]
    assert all(ln.get_marker() == "o" for ln in lines.values())
    # visibility survives a re-analysis (band change)
    plugin._bands.set_selected(1)
    visible, texts, _ = state()
    assert visible["T20"] is False and texts == ["EDT", "T30"]
    plugin._trace_toggles["T20"].set_active(True)
    assert state()[1] == ["EDT", "T20", "T30"]

    plugin._table_toggle.set_active(True)
    assert plugin._table_frame.get_visible()
    plugin._table_toggle.set_active(False)
    assert not plugin._table_frame.get_visible()


def test_export_by_extension(hosted, tmp_path):
    # one Export action; the file extension selects the format (user, 2026-11-04)
    plugin, app = hosted
    assert not plugin._export.get_sensitive()
    app.activate_action("measure", None)
    assert wait_for(lambda: plugin.analysis is not None)
    assert plugin._export.get_sensitive()
    plugin.export_to(tmp_path / "bands.csv")
    assert (tmp_path / "bands.csv").read_text(encoding="utf-8").startswith("band_hz")
    plugin.stack.set_visible_child_name("ir")
    plugin.export_to(tmp_path / "ir_tab.png")
    assert (tmp_path / "ir_tab.png").read_bytes().startswith(b"\x89PNG")
    # impulse response with the measurement metadata (D-16); the app opens it again
    settings = plugin.ir.metadata.extra["settings"]
    plugin.export_to(tmp_path / "m.ir.h5")
    plugin.clear()
    plugin.open_file(tmp_path / "m.ir.h5")
    assert wait_for(lambda: plugin.analysis is not None)
    # JSON keeps the values; tuples come back as lists
    assert plugin.ir.metadata.extra["settings"] == json.loads(json.dumps(settings))
    plugin.export_to(tmp_path / "x.txt")  # reported as a toast, no exception
    assert not (tmp_path / "x.txt").exists()
    plugin.clear()
    assert not plugin._export.get_sensitive()


def test_band_switch_in_rt_tab_and_analysis_settings_reanalyse(hosted):
    # octave / one-third-octave switch in the RT tab bar (user, 2026-11-04); analysis
    # settings changed in Preferences re-analyse the shown IR and keep the switch in sync
    plugin, app = hosted
    rt_page = plugin.stack.get_child_by_name("rt")
    w = plugin._bands
    while w is not None and w is not rt_page:
        w = w.get_parent()
    assert w is rt_page
    app.activate_action("measure", None)
    assert wait_for(lambda: plugin.analysis is not None)
    plugin._on_settings(plugin.settings.replace(bands="third"))  # as from Preferences
    assert plugin._bands.get_selected() == 1
    assert len(plugin.analysis.band_results) == 18
    before = plugin.analysis
    plugin._on_settings(plugin.settings.replace(decay_method="none"))
    assert plugin.analysis is not before  # analysis setting: re-analysed
    before = plugin.analysis
    plugin._on_settings(plugin.settings.replace(level=-20.0))
    assert plugin.analysis is before  # measurement setting: applies to the next run
    plugin._bands.set_selected(0)
    assert plugin.settings.bands == "octave" and len(plugin.analysis.band_results) == 6


def test_ir_tab_band_filtered_ir_and_toggles(hosted):
    # band selection shows the band-filtered IR; IR, decay and fits can be hidden
    # (user, 2026-11-04)
    from resonus_room.core.engine import band_impulse_responses

    plugin, app = hosted
    names = ["Impulse response", "Decay curve", "EDT fit", "T20 fit", "T30 fit"]
    assert list(plugin._ir_toggles) == names
    app.activate_action("measure", None)
    assert wait_for(lambda: plugin.analysis is not None)
    a, ir = plugin.analysis, plugin.ir
    line = plugin.ir_view._lines["ir"]
    assert len(line.get_xdata()) == ir.data.size and line.get_xdata()[0] == 0.0  # broadband
    plugin._band_select.set_selected(4)  # 1 kHz octave band
    yb = band_impulse_responses(ir, a, plugin.settings)[:, 3]
    x = np.asarray(line.get_xdata(), float)
    y = np.asarray(line.get_ydata(), float)
    assert x[0] == pytest.approx(a.onset / ir.sample_rate) and y.size == yb.size
    np.testing.assert_allclose(
        y, np.maximum(10 * np.log10(yb**2 / np.max(yb**2)), -200.0), atol=1e-9
    )
    assert "1k" in plugin.ir_view.axes.get_title()
    plugin._ir_toggles["Decay curve"].set_active(False)
    plugin._ir_toggles["T20 fit"].set_active(False)

    def visible():
        return {ln.get_label(): ln.get_visible() for ln in plugin.ir_view.axes.get_lines()}

    assert visible()["Decay curve"] is False and visible()["T20 fit"] is False
    assert visible()["Impulse response"] is True
    plugin._band_select.set_selected(2)  # visibility survives a band change
    assert visible()["Decay curve"] is False and visible()["T20 fit"] is False
    plugin._band_select.set_selected(0)
    assert len(plugin.ir_view._lines["ir"].get_xdata()) == ir.data.size
    assert "broadband" in plugin.ir_view.axes.get_title().lower()


def test_ir_tab_time_windows_toggle(hosted):
    # time-window markers in the IR tab, off by default (user, 2026-11-04)
    from resonus_room.plotting import time_windows

    plugin, app = hosted
    toggle = plugin._windows_toggle
    assert toggle.get_label() == "Time windows" and not toggle.get_active()
    app.activate_action("measure", None)
    assert wait_for(lambda: plugin.analysis is not None)
    a, ir = plugin.analysis, plugin.ir
    start = a.onset / ir.sample_rate

    def markers():
        texts = plugin._window_artists[1::2]
        lines = plugin._window_artists[0::2]
        return (
            [tx.get_text().strip() for tx in texts],
            [float(c.get_segments()[0][0, 0]) for c in lines],
            {art.get_visible() for art in plugin._window_artists},
        )

    labels, xs, shown = markers()
    assert labels == ["onset", "50 ms", "80 ms", "t1"] and shown == {False}
    toggle.set_active(True)
    expected = time_windows(start, a.broadband.curve, ir.sample_rate)
    labels, xs, shown = markers()
    assert shown == {True}
    np.testing.assert_allclose(xs, [t for _, t in expected])
    plugin._band_select.set_selected(4)  # t1 follows the band; markers stay shown
    expected = time_windows(start, a.band_results[3].curve, ir.sample_rate)
    labels, xs, shown = markers()
    assert shown == {True} and len(plugin._window_artists) == 8
    np.testing.assert_allclose(xs, [t for _, t in expected])
    n_collections = len(plugin.ir_view.axes.collections)
    assert n_collections == 4  # old markers removed on band change
    assert "onset" not in [ln.get_label() for ln in plugin.ir_view.axes.get_lines()]
    toggle.set_active(False)
    assert markers()[2] == {False}
    plugin.clear()
    assert plugin._window_artists == [] and not plugin.ir_view.axes.collections


FAKE_DEVICES = [
    plugin_module.DeviceInfo(0, "UR22", "ALSA", 2, 2, 48000.0),
    plugin_module.DeviceInfo(1, "system", "JACK", 4, 2, 48000.0),
]


def group_titles(page):
    titles, stack = [], [page.get_first_child()]
    while stack:
        w = stack.pop()
        if w is None:
            continue
        stack.append(w.get_next_sibling())
        if isinstance(w, Adw.PreferencesGroup):
            titles.append(w.get_title())
        else:
            stack.append(w.get_first_child())
    return titles


@pytest.fixture
def fake_devices(monkeypatch):
    monkeypatch.setattr(plugin_module, "list_devices", lambda: list(FAKE_DEVICES))
    monkeypatch.setattr(
        plugin_module, "default_device", lambda direction, devices=None: FAKE_DEVICES[0]
    )


def test_preferences_pages(hosted, fake_devices):
    # Measurement / Device / Analysis (D-17); every settings group is on a page
    from dataclasses import fields

    plugin, _app = hosted
    host = plugin._host
    pages = plugin.preferences(host)
    assert [p.get_title() for p in pages] == ["Measurement", "Device", "Analysis"]
    assert group_titles(pages[0]) == ["Signal", "Processing", "Simulation"]
    assert group_titles(pages[1]) == ["Device", "Inputs", "Outputs"]
    assert group_titles(pages[2]) == ["Analysis", "Plots"]  # Plots: plot mark (D-24)
    all_groups = {f.metadata["group"] for f in fields(plugin_module.AppSettings)}
    shown = set(plugin_module.MEASUREMENT_GROUPS + plugin_module.ANALYSIS_GROUPS)
    assert all_groups == shown | {"Device", "Channels"}


def test_preferences_pages_share_settings(hosted, fake_devices):
    plugin, _app = hosted
    pages = plugin.preferences(plugin._host)
    dg = plugin._device_groups
    assert dg is not None
    dg.device.input_row.set_selected(2)  # JACK system: 4 inputs
    assert plugin.settings.input_device == "JACK: system"
    dg.inputs.add_button.emit("activated")
    assert plugin.settings.input_channels == (1, 2)
    assert plugin.settings.input_labels == ("m1", "m2")
    # a change on the Analysis page keeps the device page's changes
    combo = next(r for r in _rows(pages[2]) if isinstance(r, Adw.ComboRow))
    combo.set_selected(1 - combo.get_selected())
    assert plugin.settings.input_channels == (1, 2)
    assert plugin.settings.input_device == "JACK: system"
    saved = plugin_module.AppSettings.load(plugin_module.settings_path())
    assert saved.input_channels == (1, 2)  # saved like every other setting


def _rows(page):
    found, stack = [], [page.get_first_child()]
    while stack:
        w = stack.pop()
        if w is None:
            continue
        if isinstance(w, Adw.SpinRow | Adw.ComboRow | Adw.SwitchRow):
            found.append(w)
        stack.extend([w.get_next_sibling(), w.get_first_child()])
    return found


def test_device_list_failure_is_reported(hosted, monkeypatch):
    plugin, _app = hosted
    toasts = []
    monkeypatch.setattr(plugin._host, "toast", toasts.append)

    def broken():
        raise OSError("PortAudio library not found")

    monkeypatch.setattr(plugin_module, "list_devices", broken)
    pages = plugin.preferences(plugin._host)
    assert [p.get_title() for p in pages] == ["Measurement", "Device", "Analysis"]
    assert toasts == ["audio devices not available: PortAudio library not found"]


def test_measurement_warnings_shown(hosted, monkeypatch):
    plugin, app = hosted
    toasts = []
    monkeypatch.setattr(plugin._host, "toast", toasts.append)
    plugin.settings = plugin.settings.replace(ir_length=1.5)  # > tail (1 s)
    app.activate_action("measure", None)
    assert wait_for(lambda: plugin.analysis is not None)
    assert toasts[0].startswith("IR length 1.5 s exceeds the silence after the sweep")
    assert toasts[-1] == "Measurement done"


def measure_and_wait(plugin, app, n_entries):
    app.activate_action("measure", None)
    assert wait_for(lambda: len(plugin.session.entries) == n_entries)
    assert wait_for(lambda: plugin._measure.get_sensitive())


def legend(view):
    lg = view.axes.get_legend()
    return [t.get_text() for t in lg.get_texts()] if lg else []


def test_results_list_channels_and_overlay(hosted):
    # every measurement adds its channels to the results list; one active, others
    # overlaid in the TF and RT tabs (D-19, replaces the channel selector of D-17)
    plugin, app = hosted
    assert plugin._results.button.get_parent() is not None  # in the header
    assert not plugin._results.button.get_sensitive()
    plugin.settings = plugin.settings.replace(input_channels=(1, 2), input_labels=("m1L", "m1R"))
    measure_and_wait(plugin, app, 2)
    s = plugin.session
    assert [e.title for e in s.entries] == ["Measurement 1 s1m1L", "Measurement 1 s1m1R"]
    assert [r.key for r in plugin._results.rows] == [0, 1]
    assert plugin._results.button.get_sensitive()
    assert s.active == 0 and plugin.ir is s.entries[0].ir and plugin.fr is s.entries[0].fr
    assert plugin.tf_view.trace_ids() == ["tf:0"]
    # make the second channel active from the list
    first = plugin.analysis
    plugin._results.rows[1].active.set_active(True)
    assert s.active == 1 and plugin.ir is s.entries[1].ir and plugin.analysis is not first
    assert plugin.ir.metadata.extra["channel"]["label"] == "s1m1R"
    assert plugin.tf_view.trace_ids() == ["tf:1"]
    assert plugin.rt_view.trace_ids() == ["rt:1:EDT", "rt:1:T20", "rt:1:T30"]
    # overlay the first channel: TF and RT get its traces in its colour, IR tab does not
    plugin._results.rows[0].overlay.set_active(True)
    assert s.overlay == {0}
    assert sorted(plugin.tf_view.trace_ids()) == ["tf:0", "tf:1"]
    assert sorted(plugin.rt_view.trace_ids()) == sorted(
        f"rt:{k}:{n}" for k in (0, 1) for n in ("EDT", "T20", "T30")
    )
    lines = {ln.get_label(): ln for ln in plugin.rt_view.axes.get_lines()}
    assert lines["T30, Measurement 1 s1m1L"].get_color() == s.get(0).color
    assert lines["T30, Measurement 1 s1m1R"].get_color() == s.get(1).color
    assert lines["EDT, Measurement 1 s1m1L"].get_linestyle() == ":"
    tf = {ln.get_label(): ln for ln in plugin.tf_view.axes.get_lines()}
    assert tf["Measurement 1 s1m1L"].get_color() == s.get(0).color
    handles = plugin.tf_view.axes.get_legend().legend_handles
    assert sorted(h.get_color() for h in handles) == sorted([s.get(1).color, s.get(0).color])
    assert "ir" in plugin.ir_view.trace_ids() and len(plugin.ir_view.trace_ids()) == 5
    # the EDT/T20/T30 check buttons filter all entries
    plugin._trace_toggles["T20"].set_active(False)
    assert sorted(legend(plugin.rt_view)) == sorted(
        f"{n}, Measurement 1 {lab}" for n in ("EDT", "T30") for lab in ("s1m1L", "s1m1R")
    )
    plugin._trace_toggles["T20"].set_active(True)
    # a second measurement is added, becomes active, the overlay stays
    plugin.settings = plugin.settings.replace(input_channels=(1,), input_labels=("m1",))
    measure_and_wait(plugin, app, 3)
    assert s.entries[2].title == "Measurement 2 s1m1" and s.active == 2 and s.overlay == {0}
    assert sorted(plugin.tf_view.trace_ids()) == ["tf:0", "tf:2"]
    # analysis settings re-analyse every shown entry
    plugin._bands.set_selected(1)
    assert len(s.get(0).analysis.band_results) == 18
    assert len(s.get(2).analysis.band_results) == 18
    assert s.get(1).analysis is None  # not shown: analysed when shown again
    # remove from the list; clear all
    plugin._results.rows[2].remove.emit("clicked")
    assert [e.key for e in s.entries] == [0, 1] and s.active == 0
    assert plugin.tf_view.trace_ids() == ["tf:0"]
    plugin._results.clear_button.emit("clicked")
    assert s.entries == [] and plugin.analysis is None
    assert plugin.tf_view.trace_ids() == [] and plugin.rt_view.trace_ids() == []
    assert plugin.ir_view.trace_ids() == [] and not plugin._export.get_sensitive()


def test_open_same_file_twice(hosted, tmp_path):
    plugin, _ = hosted
    ir = exponential_decay_ir(48000, 1.0, np.full(6, 0.3), band_set(1, 125, 4000), seed=1)
    wavfile.write(tmp_path / "ir.wav", 48000, (0.5 * ir / np.abs(ir).max()).astype(np.float32))
    for n in (1, 2):
        plugin.open_file(tmp_path / "ir.wav")
        assert wait_for(lambda n=n: len(plugin.session.entries) == n)
    assert [e.title for e in plugin.session.entries] == ["ir.wav", "ir.wav (2)"]


def test_channel_label_and_empty_channels(hosted):
    from resonus_core.data import ImpulseResponse, Metadata

    plugin, _app = hosted
    meta = Metadata("x", "0", extra={"channel": {"label": "s2m3"}})
    ir = ImpulseResponse(np.zeros(4), 48000.0, "1", meta)
    assert plugin_module.channel_label(ir, "fallback") == "s2m3"
    plain = ImpulseResponse(np.zeros(4), 48000.0, "1", Metadata("x", "0"))
    assert plugin_module.channel_label(plain, "room.wav") == "room.wav"
    odd = ImpulseResponse(np.zeros(4), 48000.0, "1", Metadata("x", "0", extra={"channel": 5}))
    assert plugin_module.channel_label(odd, "f") == "f"
    with pytest.raises(ValueError, match="no channels"):
        plugin.add_results("x", [], None)


def test_export_per_channel(hosted, tmp_path, monkeypatch):
    plugin, app = hosted
    toasts = []
    monkeypatch.setattr(plugin._host, "toast", toasts.append)
    plugin.settings = plugin.settings.replace(input_channels=(1, 2), input_labels=("m1L", "m1R"))
    app.activate_action("measure", None)
    assert wait_for(lambda: plugin.analysis is not None)
    plugin.export_to(tmp_path / "bh.ir.h5")
    assert sorted(q.name for q in tmp_path.glob("bh*")) == ["bh_s1m1L.ir.h5", "bh_s1m1R.ir.h5"]
    assert toasts[-1] == "Exported 2 files: bh_s1m1L.ir.h5, bh_s1m1R.ir.h5"
    plugin._results.rows[1].active.set_active(True)
    plugin.export_to(tmp_path / "bh.csv")  # analysis formats: the channel shown
    assert (tmp_path / "bh.csv").exists()
    assert toasts[-1] == "Exported bh.csv (channel s1m1R)"


def test_room_average_toggle(hosted, monkeypatch):
    # room average of the shown entries in the RT tab (D-19); Table 1 findings as toasts
    plugin, app = hosted
    toasts = []
    monkeypatch.setattr(plugin._host, "toast", toasts.append)
    toggle = plugin._average_toggle
    assert toggle.get_label() == "Room average" and not toggle.get_active()
    plugin.settings = plugin.settings.replace(
        input_channels=(1, 2), input_labels=("m1", "m2"), position_method="survey"
    )
    measure_and_wait(plugin, app, 2)
    s = plugin.session
    s.set_overlay(1, True)
    plugin.show_session()
    toggle.set_active(True)
    ids = plugin.rt_view.trace_ids()
    assert [i for i in ids if i.startswith("rt:avg:")] == ["rt:avg:EDT", "rt:avg:T20", "rt:avg:T30"]
    line = {ln.get_label(): ln for ln in plugin.rt_view.axes.get_lines()}["T20, Room average"]
    t20 = np.array([s.get(k).analysis.values("T20") for k in (0, 1)])
    np.testing.assert_allclose(line.get_ydata(), np.nanmean(t20, axis=0))
    assert line.get_color() == plugin_module.AVERAGE_COLOR
    assert not [t for t in toasts if t.startswith("ISO 3382-2")]  # survey: 2 combinations ok
    # the parameter buttons filter the average too
    plugin._trace_toggles["T20"].set_active(False)
    assert not line.get_visible()
    plugin._trace_toggles["T20"].set_active(True)
    # engineering: Table 1 findings, shown once (not again on a redraw)
    n = len(toasts)
    plugin._on_settings(plugin.settings.replace(position_method="engineering"))
    found = toasts[n:]
    assert (
        found[0] == "ISO 3382-2 Table 1: 2 source-microphone combinations, engineering requires 6"
    )
    plugin.show_session()
    assert toasts[n:] == found
    # dummy-head labels: no average, one message
    s.set_overlay(1, False)
    plugin.settings = plugin.settings.replace(input_labels=("m1L", "m1R"))
    measure_and_wait(plugin, app, 4)
    assert s.active == 2
    s.set_overlay(3, True)
    plugin.show_session()
    assert not [i for i in plugin.rt_view.trace_ids() if i.startswith("rt:avg:")]
    assert toasts[-1].startswith("Room average: not one source and one microphone position")
    toggle.set_active(False)
    s.set_active(0)
    plugin.show_session()
    assert not [i for i in plugin.rt_view.trace_ids() if i.startswith("rt:avg:")]


def test_rename_entry_for_room_average(hosted, tmp_path, monkeypatch):
    # an opened file gets an s#m# label in the results list and can then be averaged
    plugin, _ = hosted
    toasts = []
    monkeypatch.setattr(plugin._host, "toast", toasts.append)
    ir = exponential_decay_ir(48000, 1.0, np.full(6, 0.3), band_set(1, 125, 4000), seed=1)
    for k, name in enumerate(("a.wav", "b.wav")):
        x = (0.5 * ir / np.abs(ir).max()).astype(np.float32)
        wavfile.write(tmp_path / name, 48000, x)
        plugin.open_file(tmp_path / name)
        assert wait_for(lambda k=k: len(plugin.session.entries) == k + 1)
    plugin._results.rows[0].label.start_editing()
    plugin._results.rows[0].label.set_text("s1m1")
    plugin._results.rows[0].label.stop_editing(True)
    assert plugin.session.get(0).label == "s1m1"
    assert plugin._results.rows[0].label.get_text() == "s1m1"
    plugin._on_rename(1, "x/y")
    assert toasts[-1] == "Not renamed: label 'x/y' contains a path separator"
    assert plugin._results.rows[1].label.get_text() == "b.wav"  # list shows the old label
    plugin._on_rename(1, "s1m2")
    plugin.session.set_overlay(0, True)
    plugin.settings = plugin.settings.replace(position_method="survey")
    plugin._average_toggle.set_active(True)
    assert "rt:avg:T20" in plugin.rt_view.trace_ids()


def test_open_several_files_with_position_names(hosted, tmp_path, monkeypatch):
    # several IRs in one go (user request 2026-11-04); REW-style names give the
    # positions (" s1m 1.wav" -> s1m1, D-19), so they can be averaged directly
    plugin, _ = hosted
    toasts = []
    monkeypatch.setattr(plugin._host, "toast", toasts.append)
    bands = band_set(1, 125, 4000)
    paths = []
    for k, name in enumerate((" s1m 1.wav", " s1m 2.wav", "s2m 1.wav", "room.wav")):
        ir = exponential_decay_ir(48000, 1.0, np.full(6, 0.3), bands, seed=k)
        wavfile.write(tmp_path / name, 48000, (0.5 * ir / np.abs(ir).max()).astype(np.float32))
        paths.append(tmp_path / name)
    (tmp_path / "broken.wav").write_bytes(b"not a wav file")
    paths.insert(2, tmp_path / "broken.wav")
    plugin.open_files(paths)
    s = plugin.session
    assert wait_for(lambda: len(s.entries) == 4)
    assert wait_for(lambda: plugin._open.get_sensitive())
    assert [e.label for e in s.entries] == ["s1m1", "s1m2", "s2m1", "room.wav"]
    assert [e.name for e in s.entries] == [" s1m 1.wav", " s1m 2.wav", "s2m 1.wav", "room.wav"]
    assert s.active == s.entries[0].key  # the first opened file
    assert toasts[0].startswith("Cannot open broken.wav: ")
    assert toasts[-1] == "Opened 4 files"
    for e in s.entries[1:3]:
        s.set_overlay(e.key, True)
    plugin.settings = plugin.settings.replace(position_method="survey")
    plugin._average_toggle.set_active(True)
    assert "rt:avg:T20" in plugin.rt_view.trace_ids()
    assert not [t for t in toasts if t.startswith("Room average")]
    plugin.open_files([])  # nothing to do
    plugin.open_file(paths[0])
    assert wait_for(lambda: len(s.entries) == 5)
    assert s.entries[-1].name == " s1m 1.wav (2)" and s.entries[-1].label == "s1m1"
    assert toasts[-1] == "Opened  s1m 1.wav"


def test_open_one_good_one_broken(hosted, tmp_path, monkeypatch):
    plugin, _ = hosted
    toasts = []
    monkeypatch.setattr(plugin._host, "toast", toasts.append)
    ir = exponential_decay_ir(48000, 1.0, np.full(6, 0.3), band_set(1, 125, 4000), seed=1)
    wavfile.write(tmp_path / "s1m1.wav", 48000, (0.5 * ir / np.abs(ir).max()).astype(np.float32))
    (tmp_path / "zz.wav").write_bytes(b"broken")
    plugin.open_files([tmp_path / "s1m1.wav", tmp_path / "zz.wav"])
    assert wait_for(lambda: plugin._open.get_sensitive() and toasts)
    assert wait_for(lambda: len(toasts) == 2)
    assert toasts[0].startswith("Cannot open zz.wav: ") and toasts[1] == "Opened s1m1.wav"


def test_ribbon_layout_and_empty_plots(hosted):
    # D-20: the header bar holds only title and menu; empty plots get default axes
    from resonus_room.core.engine import settings_band_set
    from resonus_room.plotting import (
        EMPTY_FREQUENCY_LIMITS,
        EMPTY_LEVEL_LIMITS,
        EMPTY_MAGNITUDE_LIMITS,
        EMPTY_RT_LIMITS,
        EMPTY_TIME_LIMITS,
        PHASE_LIMITS,
    )

    plugin, app = hosted
    # header bar: only the brand (D-23); the controls are in the ribbon (D-20)
    # the brand is the last (only plugin) widget at the start of the header bar
    assert plugin._brand.get_next_sibling() is None
    assert plugin_module.RIBBON_GROUPS == ("Measure", "Device", "Files", "Results")

    def check_empty():
        assert tuple(plugin.tf_view.axes.get_xlim()) == EMPTY_FREQUENCY_LIMITS
        assert tuple(plugin.tf_view.axes.get_ylim()) == EMPTY_MAGNITUDE_LIMITS
        assert tuple(plugin.phase_view.axes.get_xlim()) == EMPTY_FREQUENCY_LIMITS
        assert tuple(plugin.phase_view.axes.get_ylim()) == PHASE_LIMITS
        assert tuple(plugin.ir_view.axes.get_xlim()) == EMPTY_TIME_LIMITS
        assert tuple(plugin.ir_view.axes.get_ylim()) == EMPTY_LEVEL_LIMITS
        bands = settings_band_set(plugin.settings)
        assert tuple(plugin.rt_view.axes.get_xlim()) == pytest.approx(
            (bands.lower[0], bands.upper[-1])
        )
        assert tuple(plugin.rt_view.axes.get_ylim()) == EMPTY_RT_LIMITS

    check_empty()
    app.activate_action("measure", None)
    assert wait_for(lambda: plugin.analysis is not None)
    assert plugin.phase_view.trace_ids() == ["ph:0"]
    assert plugin.phase_view.axes.get_legend() is None
    assert tuple(plugin.phase_view.axes.get_ylim()) == PHASE_LIMITS
    # magnitude and phase share the frequency axis
    plugin.tf_view.axes.set_xlim(100.0, 2000.0)
    assert tuple(plugin.phase_view.axes.get_xlim()) == pytest.approx((100.0, 2000.0))
    assert plugin.tf_view.axes.get_xlim()[1] != EMPTY_FREQUENCY_LIMITS[1]
    plugin.clear()
    assert plugin.phase_view.trace_ids() == []
    check_empty()
    # the empty RT plot follows the band setting
    plugin._on_settings(plugin.settings.replace(bands="third", f_min=200.0, f_max=2000.0))
    check_empty()


def test_ribbon_level(hosted):
    plugin, _ = hosted
    assert plugin._level.get_value() == plugin.settings.level
    plugin._level.set_value(-20.0)
    assert plugin.settings.level == -20.0
    assert plugin_module.AppSettings.load(plugin_module.settings_path()).level == -20.0
    plugin._on_settings(plugin.settings.replace(level=-6.5))
    assert plugin._level.get_value() == -6.5
    plugin._busy(True)
    assert not plugin._level.get_sensitive() and not plugin._device.get_sensitive()
    plugin._busy(False)
    assert plugin._level.get_sensitive() and plugin._device.get_sensitive()


def test_device_button_opens_device_page(hosted):
    plugin, _ = hosted
    plugin._device.emit("clicked")
    dialog = plugin._preferences_dialog
    assert dialog is not None
    assert dialog.get_visible_page().get_title() == "Device"
    dialog.force_close()


def test_level_test_meters(hosted):
    # D-21: Test plays 1 s of noise on each output in turn (D-21); one meter per input
    plugin, app = hosted
    assert plugin._meters.labels == ["m1"]
    plugin._on_settings(plugin.settings.replace(level=-20.0))
    app.activate_action("test", None)
    assert not plugin._test.get_sensitive()  # busy
    assert wait_for(lambda: plugin._test.get_sensitive())
    meters = plugin._meters
    # the simulated room has unit energy (D-21): -20 dB peak noise does not clip
    assert meters.current[0] is not None and not meters.clipped[0]
    assert meters.held_peak[0] == pytest.approx(meters.current[0].peak_db)
    assert -40.0 < meters.current[0].peak_db < -12.0
    # one meter per recorded input, the loopback reference last
    plugin._on_settings(
        plugin.settings.replace(
            input_channels=(1, 2),
            input_labels=("m1", "m2"),
            output_channels=(1, 2),
            output_labels=("s1", "s2"),
            reference_channel=3,
            level=0.0,
        )
    )
    assert plugin._meters.labels == ["m1", "m2", "ref"]
    assert plugin._meters.current == [None, None, None]
    # the reference wire from output 1 reaches full scale: its meter shows the clip
    plugin.start_test()
    assert wait_for(lambda: plugin._test.get_sensitive())
    assert plugin._meters.clipped[2]
    assert plugin._meters.current[2].peak_db == pytest.approx(0.0, abs=0.01)  # + sim noise floor


def test_phase_relative_to_ir_peak(hosted):
    from resonus_room.plotting import peak_delay, phase_degrees, trace_line

    plugin, app = hosted
    app.activate_action("measure", None)
    assert wait_for(lambda: plugin.analysis is not None)
    entry = plugin.session.entries[0]
    y = trace_line(plugin.phase_view, entry.title).get_ydata()
    keep = entry.fr.frequency > 0
    np.testing.assert_allclose(y, phase_degrees(entry.fr, peak_delay(entry.ir))[keep])


def test_tf_tab_one_toolbar_for_magnitude_and_phase(hosted):
    # D-22: the phase is a panel of the magnitude's figure: shared x axis, one toolbar
    from matplotlib.backends.backend_gtk4 import NavigationToolbar2GTK4

    plugin, _ = hosted
    tf, ph = plugin.tf_view, plugin.phase_view
    assert ph.figure is tf.figure and tf.panels() == [ph]
    assert ph.widget() is tf.widget()
    assert tf.axes.get_shared_x_axes().joined(tf.axes, ph.axes)
    gs = tf.axes.get_subplotspec().get_gridspec()
    assert list(gs.get_height_ratios()) == [2.0, 1.0]  # phase half the height (D-20)
    page = plugin.stack.get_child_by_name("tf")
    toolbars = []
    child = page.get_first_child()
    while child is not None:
        if isinstance(child, NavigationToolbar2GTK4):
            toolbars.append(child)
        child = child.get_next_sibling()
    assert len(toolbars) == 1
    assert ph.axes.get_xlabel() == "Frequency [Hz]" and tf.axes.get_xlabel() == ""


def test_stop_measurement(hosted, monkeypatch):
    # D-22: while measuring, the Measure button is a red Stop; stopping keeps no result
    plugin, app = hosted
    toasts = []
    monkeypatch.setattr(plugin._host, "toast", toasts.append)
    started = threading.Event()

    class SlowIO(SimulatedIO):
        def playrec(self, signal):
            started.set()
            self._aborted.wait(30)  # "plays" until stopped
            return super().playrec(signal)

    monkeypatch.setattr(plugin_module, "open_device", lambda s: SlowIO(s.sample_rate))
    n = len(plugin.session.entries)
    app.activate_action("measure", None)
    assert plugin._measure.get_label() == "Stop" and plugin._measure.get_sensitive()
    assert plugin._measure.has_css_class("destructive-action")
    assert not plugin._measure.has_css_class("suggested-action")
    assert not plugin._open.get_sensitive() and not plugin._test.get_sensitive()
    assert started.wait(10)
    app.activate_action("measure", None)  # Ctrl+M / the button again: stop
    assert wait_for(lambda: plugin._measure.get_label() == "Measure", 10)
    assert plugin._measure.get_sensitive() and plugin._measure.has_css_class("suggested-action")
    assert not plugin._measure.has_css_class("destructive-action")
    assert toasts[-1] == "Measurement stopped"
    assert len(plugin.session.entries) == n and plugin._measure_io is None
    # the next measurement runs normally
    monkeypatch.setattr(plugin_module, "open_device", open_device)
    measure_and_wait(plugin, app, n + 1)
    assert toasts[-1] == "Measurement done"


def test_measurement_device_error(hosted, monkeypatch):
    plugin, app = hosted
    toasts = []
    monkeypatch.setattr(plugin._host, "toast", toasts.append)

    def broken(_settings):
        raise ValueError("no such device")

    monkeypatch.setattr(plugin_module, "open_device", broken)
    app.activate_action("measure", None)
    assert toasts[-1] == "Measurement failed: no such device"
    assert plugin._measure.get_label() == "Measure" and plugin._measure_io is None


def test_header_brand_and_title(hosted):
    # D-23: "Re:sonus" left in Lexend SemiBold, "Room" centred; window title unchanged
    from gi.repository import PangoCairo

    plugin, _ = hosted
    brand = plugin._brand
    assert brand is not None and brand.get_label() == "Re:sonus"
    assert brand.has_css_class("resonus-brand")
    header = brand.get_ancestor(Adw.HeaderBar)
    assert isinstance(header, Adw.HeaderBar)
    title = header.get_title_widget()
    assert isinstance(title, Adw.WindowTitle) and title.get_title() == "Room"
    assert plugin_module.HEADER_TITLE == "Room"
    assert brand.get_root().get_title() == "Resonus Room"  # window list, About
    families = {f.get_name(): f for f in PangoCairo.FontMap.get_default().list_families()}
    assert "SemiBold" in [face.get_face_name() for face in families["Lexend"].list_faces()]
    assert "font-weight: 600" in plugin_module.APP_CSS


def test_plot_mark_setting(hosted):
    # D-24: the text in the bottom-right corner of every plot is a setting
    plugin, _ = hosted

    def marks():
        views = (plugin.tf_view, plugin.ir_view, plugin.rt_view)
        return [
            [
                (t.get_text(), t.get_visible())
                for t in v.figure.texts
                if t.get_gid() == "resonus-brand"
            ]
            for v in views
        ]

    assert plugin.settings.plot_mark == "Re:sonus"
    assert marks() == [[("Re:sonus", True)]] * 3
    plugin._on_settings(plugin.settings.replace(plot_mark="Acme Acoustics"))
    assert marks() == [[("Acme Acoustics", True)]] * 3
    plugin._on_settings(plugin.settings.replace(plot_mark=""))
    assert marks() == [[("", False)]] * 3


def test_plot_mark_from_saved_settings(tmp_path, monkeypatch):
    path = tmp_path / "settings.toml"
    plugin_module.AppSettings(plot_mark="Studio X").save(path)
    monkeypatch.setattr(plugin_module, "settings_path", lambda: path)
    plugin = plugin_module.Plugin()
    texts = [t.get_text() for t in plugin.rt_view.figure.texts if t.get_gid() == "resonus-brand"]
    assert texts == ["Studio X"]
