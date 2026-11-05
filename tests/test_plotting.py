import numpy as np
import pytest
from resonus_core.bands import band_set
from resonus_core.data import BandResults, FrequencyResponse, ImpulseResponse, Metadata
from resonus_core.plot import MplPlotView

from resonus_room.core.absorption import Climate, absorption
from resonus_room.core.analysis import analyze
from resonus_room.core.simulate import exponential_decay_ir
from resonus_room.plotting import (
    LEVEL_FLOOR_DB,
    PREFERRED_QUANTITIES,
    default_quantity,
    ir_level,
    setup_absorption_axes,
    setup_band_axes,
    setup_ir_axes,
    setup_response_axes,
    setup_rt_axes,
    show_absorption,
    show_band_ir,
    show_band_results,
    show_decay,
    show_fit,
    show_ir,
    show_response,
    show_rt,
    show_time_windows,
    time_windows,
)

FS = 48000.0
META = Metadata("test", "0")
OCT = band_set(1, 250, 2000)


@pytest.fixture(scope="module")
def analysis():
    ir = exponential_decay_ir(FS, 2.0, [1.0, 0.9, 0.8, 0.7], OCT, delay=0.01, seed=1)
    return ir, analyze(ir, FS, OCT)


def xy(view, trace_id):
    line = view._lines[trace_id]
    return np.asarray(line.get_xdata(), float), np.asarray(line.get_ydata(), float)


def test_response_skips_zero_frequency():
    view = MplPlotView()
    setup_response_axes(view)
    fr = FrequencyResponse(np.array([0.0, 10.0, 100.0]), np.array([1, 0.1, 10], complex), "1", META)
    show_response(view, fr, "a", "A")
    x, y = xy(view, "a")
    np.testing.assert_allclose(x, [10, 100])
    np.testing.assert_allclose(y, [-20, 20])
    assert view.axes.get_xscale() == "log"


def test_ir_level():
    ir = ImpulseResponse(np.array([0.0, -2.0, 1.0, 0.2]), FS, "1", META)
    np.testing.assert_allclose(ir_level(ir), [LEVEL_FLOOR_DB, 0.0, -6.0206, -20.0], atol=1e-4)
    with pytest.raises(ValueError):
        ir_level(ImpulseResponse(np.zeros(3), FS, "1", META))


def test_ir_decay_and_fit_traces(analysis):
    ir, a = analysis
    view = MplPlotView()
    setup_ir_axes(view)
    show_ir(view, ImpulseResponse(ir, FS, "1", META), "ir", "IR")
    t0 = a.onset / FS
    p = a.band_results[1]
    show_decay(view, p.curve, t0, "decay", "500 Hz")
    show_fit(view, p.fits["T30"], t0, "fit", "T30")
    x, y = xy(view, "decay")
    assert x[0] == pytest.approx(t0) and y[0] == 0.0 and np.all(y >= LEVEL_FLOOR_DB)
    x, y = xy(view, "fit")
    f = p.fits["T30"]
    np.testing.assert_allclose(x, [f.t_start + t0, f.t_stop + t0])
    np.testing.assert_allclose(
        y, [f.intercept + f.slope * f.t_start, f.intercept + f.slope * f.t_stop]
    )
    assert view.trace_ids() == ["ir", "decay", "fit"]
    nan_fit = type(f)("T30", -5.0, -35.0, *[float("nan")] * 7, 0)
    show_fit(view, nan_fit, t0, "fit", "T30")
    assert "fit" not in view.trace_ids()


def test_rt_and_absorption(analysis, tmp_path):
    _, a = analysis
    view = MplPlotView()
    setup_rt_axes(view)
    for name in ("EDT", "T20", "T30"):
        show_rt(view, a, name, name, name)
    x, y = xy(view, "T20")
    np.testing.assert_allclose(x, [250, 500, 1000, 2000])
    np.testing.assert_allclose(y, a.values("T20"))
    view.save(str(tmp_path / "rt.png"))
    assert (tmp_path / "rt.png").stat().st_size > 1000
    c = Climate(20.0, 50.0)
    t1 = np.linspace(6.0, 3.0, 18)
    r = absorption(t1, 0.7 * t1, 200.0, c, c, specimen_area=10.0)
    view = MplPlotView()
    setup_absorption_axes(view)
    show_absorption(view, r, "alpha", "Specimen")
    x, y = xy(view, "alpha")
    np.testing.assert_allclose(x, r.frequency)
    np.testing.assert_allclose(y, r.alpha)


def bands(values):
    return BandResults(1, OCT.center, tuple(OCT.labels), values, dict.fromkeys(values, "s"), META)


def test_default_quantity():
    assert PREFERRED_QUANTITIES[0] == "T20"
    assert default_quantity(bands({"EDT": np.ones(4), "T30": np.ones(4)})) == "T30"
    assert default_quantity(bands({"C80": np.ones(4), "Ts": np.ones(4)})) == "C80"


def test_band_results_plot():
    view = MplPlotView()
    setup_band_axes(view, "T20", "s")
    assert view.axes.get_ylabel() == "T20 [s]" and view.axes.get_xscale() == "log"
    assert view.axes.get_xlabel() == "Frequency [Hz]" and view.axes.get_title() == "T20"
    r = bands({"T20": [1.0, np.nan, 0.8, 0.7]})
    show_band_results(view, r, "T20", "a", "pos 1")
    x, y = xy(view, "a")
    np.testing.assert_array_equal(x, OCT.center)
    np.testing.assert_array_equal(y, r.values["T20"])
    with pytest.raises(KeyError):
        show_band_results(view, r, "T30", "b", "x")


def test_band_ir_level_and_time():
    view = MplPlotView()
    show_band_ir(view, [0.0, 2.0, 1.0, 0.0], 10.0, 0.5, "ir", "Impulse response")
    x, y = xy(view, "ir")
    np.testing.assert_allclose(x, [0.5, 0.6, 0.7, 0.8])
    np.testing.assert_allclose(y, [LEVEL_FLOOR_DB, 0.0, 10 * np.log10(0.25), LEVEL_FLOOR_DB])
    with pytest.raises(ValueError):
        show_band_ir(view, np.zeros(4), 10.0, 0.0, "ir", "x")


def test_time_windows_positions():
    # ISO 3382-1:2009 A.3.4 (onset), A.2.3 (50/80 ms), 5.3.3 (t1); user request 2026-11-04
    from resonus_room.core.decay import DecayCurve

    curve = DecayCurve(np.zeros(3), np.zeros(3), 24000, 0.0, -80.0, 60.0)
    w = time_windows(0.01, curve, FS)
    assert [label for label, _ in w] == ["onset", "50 ms", "80 ms", "t1"]
    np.testing.assert_allclose([t for _, t in w], [0.01, 0.06, 0.09, 0.51])


def test_show_time_windows_not_traces(analysis):
    from matplotlib.collections import LineCollection

    ir, a = analysis
    view = MplPlotView()
    show_ir(view, ImpulseResponse(ir, FS, "1", META), "ir", "Impulse response")
    ylim = view.axes.get_ylim()
    artists = show_time_windows(view, [("onset", 0.01), ("t1", 0.5)])
    assert len(artists) == 4
    lines, texts = artists[0::2], artists[1::2]
    assert all(isinstance(c, LineCollection) for c in lines)
    assert [tx.get_text().strip() for tx in texts] == ["onset", "t1"]
    xs = [c.get_segments()[0][:, 0] for c in lines]
    np.testing.assert_allclose(xs, [[0.01, 0.01], [0.5, 0.5]])
    # the markers span the axes height and stay out of traces, legend and y limits
    np.testing.assert_allclose(lines[0].get_segments()[0][:, 1], [0.0, 1.0])
    assert [ln.get_label() for ln in view.axes.get_lines()] == ["Impulse response"]
    assert view.trace_ids() == ["ir"]
    assert view.axes.get_ylim() == pytest.approx(ylim)
    assert show_time_windows(view, []) == []
