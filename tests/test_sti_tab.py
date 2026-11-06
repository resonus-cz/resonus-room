"""STI tab (D-31): inputs, status texts, cache, drawing. PROVISIONAL."""

import numpy as np
import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
from gi.repository import Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no display available", allow_module_level=True)

from resonus_core import data  # noqa: E402
from resonus_core.data import ImpulseResponse, Metadata, TimeSignal  # noqa: E402

from resonus_room.core.sti_inputs import NOISE_ROLE, SOURCE_ROLE  # noqa: E402
from resonus_room.gui import sti_tab  # noqa: E402
from resonus_room.gui.sti_tab import StiTab  # noqa: E402

FS = 32000.0
LEVELS = (80.0,) * 7


def meta(**extra):
    return Metadata("t", "0", extra=extra)


def room(unit="Pa/FS"):
    return ImpulseResponse(np.array([1e-3]), FS, unit, meta())


@pytest.fixture
def files(tmp_path):
    src = tmp_path / "src.h5"
    data.save(src, ImpulseResponse(np.array([1e-3]), FS, "Pa/FS", meta(role=SOURCE_ROLE)))
    noise = tmp_path / "noise.h5"
    rng = np.random.default_rng(0)
    data.save(
        noise,
        ImpulseResponse(0.01 * rng.standard_normal(int(9 * FS)), FS, "Pa", meta(role=NOISE_ROLE)),
    )
    other = tmp_path / "ir.h5"
    data.save(other, room())
    return src, noise, other


class Runner:
    """Synchronous run_in_thread; can hold the work back."""

    def __init__(self):
        self.calls = 0
        self.held = []
        self.hold = False

    def __call__(self, work, done, failed):
        self.calls += 1
        if self.hold:
            self.held.append((work, done, failed))
            return
        try:
            result = work()
        except Exception as exc:
            failed(exc)
        else:
            done(result)


@pytest.fixture
def tab():
    toasts = []
    runner = Runner()
    t = StiTab(runner, toasts.append, brand_text="mark")
    return t, runner, toasts


def test_widgets(tab):
    t, _runner, _toasts = tab
    assert isinstance(t.widget, Gtk.Box)
    assert t.source_button.get_label() == "Source calibration…"
    assert t.noise_button.get_label() == "Noise…"
    assert t.result_label.get_text() == ""
    assert t.view.axes.get_title() == sti_tab.NO_ENTRY
    assert t.view.axes.get_ylim() == (0.0, 1.0)
    assert t.source is None and t.noise is None and t.speech_levels == () and t.report is None


def test_messages_in_order(tab, files):
    t, runner, _toasts = tab
    src, noise, _other = files
    assert t.message() == sti_tab.NO_ENTRY
    t.show_entry("s1m1", room("1"))
    assert t.message() == sti_tab.NOT_CALIBRATED
    assert t.view.axes.get_title() == sti_tab.NOT_CALIBRATED
    t.show_entry("s1m1", room())
    assert t.message() == sti_tab.NO_SOURCE
    assert t.load_source(src)
    assert t.source_button.get_label() == "Source: src.h5" and t.source_path == src
    assert t.message() == sti_tab.NO_NOISE
    assert t.load_noise(noise)
    assert t.noise_button.get_label() == "Noise: noise.h5"
    assert t.message() == sti_tab.NO_LEVELS
    assert runner.calls == 0
    t.set_speech_levels(LEVELS)
    assert t.message() == ""
    assert runner.calls == 1


def test_draw_and_cache(tab, files):
    t, runner, _toasts = tab
    src, noise, _other = files
    ir = room()
    t.load_source(src)
    t.load_noise(noise)
    t.set_speech_levels(LEVELS)
    t.show_entry("s1m1", ir)
    assert runner.calls == 1
    assert t.report is not None and t.report.sti > 0.9  # about 30 dB S/N
    assert t.result_label.get_text() == f"STI {t.report.sti:.2f}"
    assert t.view.axes.get_title() == f"s1m1: STI {t.report.sti:.2f} (PROVISIONAL)"
    assert sti_tab.TRACE_ID in t.view.trace_ids()
    t.show_entry(None, None)
    assert t.report is None and t.view.axes.get_title() == sti_tab.NO_ENTRY
    assert sti_tab.TRACE_ID not in t.view.trace_ids()
    t.show_entry("s1m1", ir)  # cached: no new computation
    assert runner.calls == 1 and t.report is not None
    t.set_speech_levels(LEVELS)  # unchanged: nothing to do
    assert runner.calls == 1
    t.set_speech_levels((70.0,) * 7)
    assert runner.calls == 2


def test_wrong_files(tab, files):
    t, runner, toasts = tab
    src, noise, other = files
    assert not t.load_source(other)
    assert toasts[-1] == "ir.h5: not a source calibration"
    assert not t.load_noise(src)
    assert toasts[-1] == "src.h5: not a noise recording"
    assert not t.load_source(src.parent / "missing.h5")
    assert toasts[-1].startswith("missing.h5: ")
    assert t.source is None and t.noise is None
    assert t.source_button.get_label() == "Source calibration…"


def test_stale_result_is_not_drawn(tab, files):
    t, runner, _toasts = tab
    src, noise, _other = files
    t.load_source(src)
    t.load_noise(noise)
    t.set_speech_levels(LEVELS)
    runner.hold = True
    t.show_entry("a", room())
    assert t.view.axes.get_title() == "Computing STI…"
    t.show_entry(None, None)
    work, done, failed = runner.held.pop()
    done(work())
    assert t.report is None and t.view.axes.get_title() == sti_tab.NO_ENTRY


def test_failure_is_cached_and_shown(tab, files, tmp_path):
    t, runner, toasts = tab
    src, _noise, _other = files
    short = tmp_path / "short.h5"
    data.save(short, ImpulseResponse(np.zeros(int(FS)), FS, "Pa", meta(role=NOISE_ROLE)))
    t.load_source(src)
    t.load_noise(short)
    t.set_speech_levels(LEVELS)
    t.show_entry("a", room())
    assert toasts[-1].startswith("STI failed: noise recording too short")
    assert t.view.axes.get_title().startswith("noise recording too short")
    calls, n = runner.calls, len(toasts)
    first = t._ir  # noqa: SLF001
    t.show_entry(None, None)
    t.show_entry("a", first)  # cached error: shown again, not recomputed, no new toast
    assert runner.calls == calls and len(toasts) == n
    assert t.view.axes.get_title().startswith("noise recording too short")
    t.show_entry("a", room())  # another IR object with the same name: recomputed
    assert runner.calls == calls + 1 and len(toasts) == n + 1


def test_brand(tab):
    t, _runner, _toasts = tab
    t.set_brand("other")


def test_noise_as_time_signal(tab, files, tmp_path):
    # resonus-core 0.11.0 (Q-C9): noise recordings are TimeSignal; a TimeSignal is no
    # source calibration
    t, _runner, toasts = tab
    path = tmp_path / "noise_ts.h5"
    sig = TimeSignal(np.zeros(int(FS)), FS, "Pa", meta(role=NOISE_ROLE))
    data.save(path, sig)
    assert t.load_noise(path) and isinstance(t.noise, TimeSignal)
    assert t.noise_button.get_label() == "Noise: noise_ts.h5"
    assert not t.load_source(path)
    assert toasts[-1] == "noise_ts.h5: not a source calibration"
