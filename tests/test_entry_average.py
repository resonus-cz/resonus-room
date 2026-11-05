"""Room average of session entries and its plot (D-19)."""

from pathlib import Path

import numpy as np
import pytest
from resonus_core.bands import band_set
from resonus_core.data import ImpulseResponse, Metadata
from resonus_core.plot import MARKER_SIZE, MplPlotView

from resonus_room.core.engine import analyze_ir
from resonus_room.core.positions import Position, PositionSetup
from resonus_room.core.room_average import average_analyses, average_entries
from resonus_room.core.session import AVERAGE_COLOR, Session
from resonus_room.core.settings import AppSettings
from resonus_room.core.simulate import exponential_decay_ir
from resonus_room.plotting import (
    AVERAGE_LINE_WIDTH,
    AVERAGE_TITLE,
    show_average,
    trace_line,
)

FS = 16000.0
T = np.array([0.6, 0.55, 0.5, 0.45, 0.4, 0.35])
OCT = band_set(1, 125, 4000)


def ir(seed):
    x = exponential_decay_ir(FS, 1.5, T, OCT, delay=0.01, noise_rms=1e-5, seed=seed)
    return ImpulseResponse(x, FS, "1", Metadata("t", "0"))


def session(labels, settings=None):
    s = Session()
    s.add("Measurement 1", [(lab, ir(k), None) for k, lab in enumerate(labels)])
    for e in s.entries:
        e.analysis = analyze_ir(e.ir, settings or AppSettings())
    return s


def test_setting_default():
    # default by user decision D-19
    assert AppSettings().position_method == "engineering"
    with pytest.raises(ValueError):
        AppSettings(position_method="fast")


def test_average_entries_matches_average_analyses():
    s = session(["s1m1", "s1m2", "s2m1"])
    avg = average_entries(s.entries, "survey")
    setup = PositionSetup(
        "survey",
        (
            Position(Path("Measurement 1 s1m1"), "s1", "m1"),
            Position(Path("Measurement 1 s1m2"), "s1", "m2"),
            Position(Path("Measurement 1 s2m1"), "s2", "m1"),
        ),
    )
    ref = average_analyses(setup, [e.analysis for e in s.entries], OCT)
    assert avg.setup == setup and avg.bands == OCT
    for name in ref.mean:
        np.testing.assert_array_equal(avg.mean[name], ref.mean[name])
    assert avg.issues == ()
    eng = average_entries(s.entries, "engineering")
    assert eng.issues[0] == (
        "ISO 3382-2 Table 1: 3 source-microphone combinations, engineering requires 6"
    )


def test_average_entries_errors():
    with pytest.raises(ValueError, match="no entries"):
        average_entries([], "survey")
    s = session(["s1m1", "s1m1L", "s1m1R"])
    with pytest.raises(ValueError, match="unknown method 'fast'"):
        average_entries(s.entries, "fast")
    msg = (
        r"^not one source and one microphone position \(s#m#\): "
        r"'Measurement 1 s1m1L', 'Measurement 1 s1m1R'$"
    )
    with pytest.raises(ValueError, match=msg):
        average_entries(s.entries, "survey")
    s = session(["s1m1", "s1m2"])
    s.entries[1].analysis = None
    with pytest.raises(ValueError, match="^Measurement 1 s1m2: not analysed$"):
        average_entries(s.entries, "survey")
    s.entries[1].analysis = analyze_ir(s.entries[1].ir, AppSettings(bands="third"))
    with pytest.raises(ValueError, match="entries analysed with different bands"):
        average_entries(s.entries, "survey")


def test_show_average():
    avg = average_entries(session(["s1m1", "s1m2"]).entries, "survey")
    view = MplPlotView()
    for name in ("EDT", "T20", "T30"):
        show_average(view, avg, name, f"rt:avg:{name}", AVERAGE_COLOR)
    assert AVERAGE_TITLE == "Room average" and AVERAGE_LINE_WIDTH == 2.5
    line = trace_line(view, "T20, Room average")
    np.testing.assert_allclose(line.get_xdata(), OCT.nominal)
    np.testing.assert_allclose(line.get_ydata(), avg.mean["T20"])
    assert line.get_color() == AVERAGE_COLOR and line.get_linestyle() == "--"
    assert line.get_marker() == "s" and line.get_markersize() == MARKER_SIZE
    assert line.get_linewidth() == AVERAGE_LINE_WIDTH
    assert trace_line(view, "EDT, Room average").get_linestyle() == ":"
    assert view.trace_ids() == ["rt:avg:EDT", "rt:avg:T20", "rt:avg:T30"]
