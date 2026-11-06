"""Overlay helpers of the plots (D-19)."""

import numpy as np
import pytest
from resonus_core.bands import band_set
from resonus_core.data import FrequencyResponse, Metadata
from resonus_core.plot import MARKER_SIZE, MplPlotView

from resonus_room.core.analysis import analyze
from resonus_room.core.simulate import exponential_decay_ir
from resonus_room.plotting import (
    PARAMETER_LINESTYLES,
    RT_PARAMETERS,
    keep_traces,
    rt_label,
    setup_response_axes,
    setup_rt_axes,
    show_parameters,
    show_response_entry,
    show_rt_entry,
    trace_line,
    update_legend,
)

FS = 48000.0
OCT = band_set(1, 250, 2000)


@pytest.fixture(scope="module")
def analyses():
    out = []
    for seed, t in ((1, 1.0), (2, 0.6)):
        ir = exponential_decay_ir(FS, 2.0, [t] * 4, OCT, delay=0.01, seed=seed)
        out.append(analyze(ir, FS, OCT))
    return out


def legend_texts(view):
    legend = view.axes.get_legend()
    return [] if legend is None else [t.get_text() for t in legend.get_texts()]


def test_linestyles_and_label():
    assert PARAMETER_LINESTYLES == {
        "EDT": ":", "T20": "--", "T30": "-", "Ts": "-",
        "C50": "--", "C80": "-", "D50": "-",
        "IACC early": "-", "IACC late": "--", "IACC full": ":",
    }  # fmt: skip
    assert rt_label("T20", "Measurement 1 s1m1") == "T20, Measurement 1 s1m1"


def test_trace_line():
    view = MplPlotView()
    view.set_trace("a", [1, 2], [3, 4], label="A")
    line = trace_line(view, "A")
    assert list(line.get_ydata()) == [3, 4]
    with pytest.raises(ValueError, match="no trace labelled 'B'"):
        trace_line(view, "B")


def test_rt_entries_style_and_filter(analyses):
    view = MplPlotView()
    setup_rt_axes(view)
    colors = ("#3584e4", "#e01b24")
    for k, (a, color) in enumerate(zip(analyses, colors, strict=True)):
        for name in RT_PARAMETERS:
            show_rt_entry(view, a, name, f"rt:{k}:{name}", f"m{k}", color)
    assert len(view.trace_ids()) == 6
    line = trace_line(view, "T30, m1")
    assert line.get_color() == "#e01b24" and line.get_linestyle() == "-"
    assert line.get_marker() == "o" and line.get_markersize() == MARKER_SIZE
    assert trace_line(view, "EDT, m0").get_linestyle() == ":"
    assert trace_line(view, "T20, m0").get_linestyle() == "--"
    np.testing.assert_allclose(trace_line(view, "T20, m1").get_ydata(), analyses[1].values("T20"))
    # replacing keeps one line per id and applies the new colour
    show_rt_entry(view, analyses[0], "T20", "rt:0:T20", "m0", "#26a269")
    assert len(view.trace_ids()) == 6 and trace_line(view, "T20, m0").get_color() == "#26a269"
    view.set_trace("other", [1, 2], [0, 0], label="reference")
    show_parameters(view, {"T20"})
    visible = {ln.get_label() for ln in view.axes.get_lines() if ln.get_visible()}
    assert visible == {"T20, m0", "T20, m1", "reference"}
    assert sorted(legend_texts(view)) == ["T20, m0", "T20, m1", "reference"]
    show_parameters(view, PARAMETER_LINESTYLES)
    assert all(ln.get_visible() for ln in view.axes.get_lines())


def test_show_parameters_none_visible_removes_legend(analyses):
    view = MplPlotView()
    show_rt_entry(view, analyses[0], "T20", "t", "m0", "#3584e4")
    show_parameters(view, {"T20"})
    assert legend_texts(view) == ["T20, m0"]
    show_parameters(view, set())
    assert view.axes.get_legend() is None


def test_response_entry_and_keep_traces():
    view = MplPlotView()
    setup_response_axes(view)
    f = np.array([0.0, 100.0, 200.0])
    fr = FrequencyResponse(f, np.array([1.0, 0.5, 0.25], complex), "1", Metadata("t", "0"))
    show_response_entry(view, fr, "tf:0", "Measurement 1 s1m1", "#ff7800")
    line = trace_line(view, "Measurement 1 s1m1")
    assert line.get_color() == "#ff7800" and list(line.get_xdata()) == [100.0, 200.0]
    show_response_entry(view, fr, "tf:1", "bh.wav", "#9141ac")
    keep_traces(view, {"tf:1", "unknown"})
    assert view.trace_ids() == ["tf:1"]
    keep_traces(view, [])
    assert view.trace_ids() == []


def test_update_legend_uses_entry_colours():
    view = MplPlotView()
    f = np.array([100.0, 200.0])
    fr = FrequencyResponse(f, np.array([1.0, 0.5], complex), "1", Metadata("t", "0"))
    show_response_entry(view, fr, "tf:0", "a", "#ff7800")
    update_legend(view)
    handles = view.axes.get_legend().legend_handles
    assert [h.get_color() for h in handles] == ["#ff7800"]
    trace_line(view, "a").set_visible(False)
    update_legend(view)
    assert view.axes.get_legend() is None
