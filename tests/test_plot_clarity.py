"""Plots of the Clarity and IACC tabs (D-28)."""

import numpy as np
import pytest
from resonus_core.bands import band_set
from resonus_core.plot import MARKER_SIZE, MplPlotView

from resonus_room.core.analysis import analyze
from resonus_room.core.binaural import BinauralParameters, iacc_band_set
from resonus_room.core.simulate import exponential_decay_ir
from resonus_room.plotting import (
    CENTRE_TIME_PARAMETERS,
    CLARITY_PARAMETERS,
    DEFINITION_PARAMETERS,
    EMPTY_CLARITY_LIMITS,
    EMPTY_RT_LIMITS,
    EMPTY_TS_LIMITS,
    IACC_PARAMETERS,
    PARAMETER_SCALES,
    RATIO_LIMITS,
    RT_PARAMETERS,
    empty_band_axes,
    empty_iacc_axes,
    empty_rt_axes,
    setup_centre_time_axes,
    setup_clarity_axes,
    setup_definition_axes,
    setup_iacc_axes,
    show_iacc_entry,
    show_parameters,
    show_rt_entry,
    trace_line,
)

FS = 48000.0
OCT = band_set(1, 250, 2000)


def test_constants():
    assert RT_PARAMETERS == ("EDT", "T20", "T30")
    assert CENTRE_TIME_PARAMETERS == ("Ts",) and PARAMETER_SCALES == {"Ts": 1000.0}
    assert EMPTY_TS_LIMITS == (0.0, 300.0)
    assert CLARITY_PARAMETERS == ("C50", "C80") and DEFINITION_PARAMETERS == ("D50",)
    assert IACC_PARAMETERS == ("IACC early", "IACC late", "IACC full")
    assert EMPTY_CLARITY_LIMITS == (-10.0, 10.0) and RATIO_LIMITS == (0.0, 1.0)


def test_axes_setup():
    view = MplPlotView()
    setup_clarity_axes(view)
    ax = view.axes
    assert ax.get_ylabel() == "Clarity [dB]" and ax.get_xlabel() == "Frequency [Hz]"
    assert ax.get_title() == "Clarity" and ax.get_xscale() == "log"
    lower = view.add_panel((2, 1))
    setup_definition_axes(lower)
    assert lower.axes.get_ylabel() == "D50 [–]" and lower.axes.get_title() == "Definition"
    assert lower.axes.get_xlabel() == "Frequency [Hz]" and lower.axes.get_xscale() == "log"
    iacc_view = MplPlotView()
    setup_iacc_axes(iacc_view)
    assert iacc_view.axes.get_ylabel() == "IACC [–]"
    assert iacc_view.axes.get_title() == "Interaural cross-correlation"
    assert iacc_view.axes.get_xlabel() == "Frequency [Hz]"
    assert iacc_view.axes.get_xscale() == "log"


def test_empty_band_axes():
    view = MplPlotView()
    setup_clarity_axes(view)
    empty_band_axes(view, OCT, EMPTY_CLARITY_LIMITS)
    assert tuple(view.axes.get_ylim()) == EMPTY_CLARITY_LIMITS
    np.testing.assert_allclose(view.axes.get_xlim(), (OCT.lower[0], OCT.upper[-1]))
    np.testing.assert_allclose(view.axes.xaxis.get_majorticklocs(), OCT.nominal)
    rt = MplPlotView()
    empty_rt_axes(rt, OCT)
    assert tuple(rt.axes.get_ylim()) == EMPTY_RT_LIMITS


def test_empty_iacc_axes():
    view = MplPlotView()
    setup_iacc_axes(view)
    empty_iacc_axes(view)
    bands = iacc_band_set()
    assert tuple(view.axes.get_ylim()) == RATIO_LIMITS
    np.testing.assert_allclose(view.axes.get_xlim(), (bands.lower[0], bands.upper[-1]))
    np.testing.assert_allclose(view.axes.xaxis.get_majorticklocs(), bands.nominal)


def test_clarity_entries_in_two_panels():
    ir = exponential_decay_ir(FS, 2.0, [1.0] * 4, OCT, delay=0.01, seed=1)
    a = analyze(ir, FS, OCT)
    view = MplPlotView()
    setup_clarity_axes(view)
    lower = view.add_panel((2, 1))
    setup_definition_axes(lower)
    for name in CLARITY_PARAMETERS:
        show_rt_entry(view, a, name, f"cl:0:{name}", "m0", "#3584e4")
    show_rt_entry(lower, a, "D50", "d:0:D50", "m0", "#3584e4")
    np.testing.assert_allclose(trace_line(view, "C80, m0").get_ydata(), a.values("C80"))
    assert trace_line(view, "C50, m0").get_linestyle() == "--"
    assert trace_line(view, "C80, m0").get_linestyle() == "-"
    assert trace_line(lower, "D50, m0").get_linestyle() == "-"
    show_parameters(view, {"C80"})
    visible = {ln.get_label() for ln in view.axes.get_lines() if ln.get_visible()}
    assert visible == {"C80, m0"}


def params(n, seed=0):
    rng = np.random.default_rng(seed)
    early, late, full = (rng.uniform(0.1, 0.9, n) for _ in range(3))
    return BinauralParameters(early=early, late=late, full=full, onset=0)


def test_show_iacc_entry():
    bands = iacc_band_set()
    p = params(len(bands))
    view = MplPlotView()
    setup_iacc_axes(view)
    for name in IACC_PARAMETERS:
        show_iacc_entry(view, p, bands, name, f"iacc:1:{name}", "M1 s1m1", "#e01b24")
    assert len(view.trace_ids()) == 3
    for name, values, style in zip(
        IACC_PARAMETERS, (p.early, p.late, p.full), ("-", "--", ":"), strict=True
    ):
        line = trace_line(view, f"{name}, M1 s1m1")
        np.testing.assert_allclose(line.get_xdata(), bands.nominal)
        np.testing.assert_allclose(line.get_ydata(), values)
        assert line.get_color() == "#e01b24" and line.get_linestyle() == style
        assert line.get_marker() == "o" and line.get_markersize() == MARKER_SIZE
    # replacing keeps one trace per id
    show_iacc_entry(view, p, bands, "IACC late", "iacc:1:IACC late", "M1 s1m1", "#26a269")
    assert len(view.trace_ids()) == 3
    assert trace_line(view, "IACC late, M1 s1m1").get_color() == "#26a269"
    show_parameters(view, {"IACC early"})
    visible = {ln.get_label() for ln in view.axes.get_lines() if ln.get_visible()}
    assert visible == {"IACC early, M1 s1m1"}


def test_show_iacc_entry_errors():
    bands = iacc_band_set()
    view = MplPlotView()
    with pytest.raises(KeyError):
        show_iacc_entry(view, params(len(bands)), bands, "IACC", "x", "t", "#000000")
    with pytest.raises(ValueError, match="5 values for 6 bands"):
        show_iacc_entry(view, params(5), bands, "IACC full", "x", "t", "#000000")


def test_ts_panel_in_ms():
    ir = exponential_decay_ir(FS, 2.0, [1.0] * 4, OCT, delay=0.01, seed=1)
    a = analyze(ir, FS, OCT)
    view = MplPlotView()
    setup_clarity_axes(view)
    d50, ts = view.add_row((2, 1), 2)
    setup_centre_time_axes(ts)
    assert ts.axes.get_ylabel() == "Ts [ms]" and ts.axes.get_xscale() == "log"
    show_rt_entry(ts, a, "Ts", "ts:0:Ts", "m0", "#3584e4")
    line = trace_line(ts, "Ts, m0")
    np.testing.assert_allclose(line.get_ydata(), a.values("Ts") * 1000.0)
    assert line.get_linestyle() == "-"
    # T = 1 s exponential decay: Ts about T / 13.8 = 72 ms (closed form, no direct sound)
    assert np.all((line.get_ydata() > 55) & (line.get_ydata() < 95))
