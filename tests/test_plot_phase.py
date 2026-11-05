"""Phase plot, shared frequency axis and empty-plot axes (D-20)."""

import numpy as np
import pytest
from resonus_core.bands import band_set
from resonus_core.data import FrequencyResponse, Metadata
from resonus_core.plot import MplPlotView

from resonus_room.plotting import (
    EMPTY_FREQUENCY_LIMITS,
    EMPTY_MAGNITUDE_LIMITS,
    EMPTY_RT_LIMITS,
    PHASE_LIMITS,
    PHASE_TICKS,
    empty_limits,
    empty_rt_axes,
    hide_legend,
    phase_degrees,
    setup_phase_axes,
    setup_rt_axes,
    show_phase_entry,
    trace_line,
)


def fr_with_phase(theta):
    f = np.arange(len(theta), dtype=float) * 100.0
    return FrequencyResponse(f, 2.0 * np.exp(1j * np.asarray(theta)), "1", Metadata("t", "0"))


def test_constants():
    assert PHASE_LIMITS == (-180.0, 180.0)
    assert PHASE_TICKS == (-180.0, -90.0, 0.0, 90.0, 180.0)
    assert EMPTY_FREQUENCY_LIMITS == (20.0, 20000.0)


def test_setup_phase_axes():
    view = MplPlotView()
    setup_phase_axes(view)
    ax = view.axes
    assert ax.get_ylabel() == "Phase [°]" and ax.get_xlabel() == "Frequency [Hz]"
    assert ax.get_xscale() == "log"
    assert tuple(ax.get_ylim()) == PHASE_LIMITS
    assert tuple(ax.get_yticks()) == PHASE_TICKS


def test_rt_axis_label():
    view = MplPlotView()
    setup_rt_axes(view)
    assert view.axes.get_ylabel() == "Reverberation time [s]"


def test_phase_degrees_wrapped():
    theta = [0.0, np.pi / 2, -np.pi / 2, np.pi, 3 * np.pi / 2, -np.pi / 4]
    got = phase_degrees(fr_with_phase(theta))
    assert got.dtype == np.float64 and got.shape == (6,)
    np.testing.assert_allclose(got, [0.0, 90.0, -90.0, 180.0, -90.0, -45.0], atol=1e-9)
    assert np.all(got > -180.0) and np.all(got <= 180.0)


def test_show_phase_entry():
    view = MplPlotView()
    setup_phase_axes(view)
    fr = fr_with_phase([0.3, np.pi / 2, np.pi, -np.pi / 2])
    show_phase_entry(view, fr, "ph:0", "Measurement 1", "#123456")
    line = trace_line(view, "Measurement 1")
    np.testing.assert_allclose(line.get_xdata(), [100.0, 200.0, 300.0])  # 0 Hz omitted
    np.testing.assert_allclose(line.get_ydata(), [90.0, 180.0, -90.0])
    assert line.get_color() == "#123456"
    assert tuple(view.axes.get_ylim()) == PHASE_LIMITS  # not autoscaled to the data
    assert view.axes.get_legend() is None
    show_phase_entry(view, fr, "ph:0", "Measurement 1", "#654321")  # replace
    assert view.trace_ids() == ["ph:0"] and line.get_color() == "#654321"


def test_hide_legend():
    view = MplPlotView()
    hide_legend(view)  # no legend: nothing happens
    view.set_trace("a", [1, 2], [1, 2], label="a")
    assert view.axes.get_legend() is not None
    hide_legend(view)
    assert view.axes.get_legend() is None


def test_empty_limits_keep_autoscaling():
    view = MplPlotView()
    view.axes.set_xscale("log")
    empty_limits(view, EMPTY_FREQUENCY_LIMITS, EMPTY_MAGNITUDE_LIMITS)
    assert tuple(view.axes.get_xlim()) == EMPTY_FREQUENCY_LIMITS
    assert tuple(view.axes.get_ylim()) == EMPTY_MAGNITUDE_LIMITS
    view.figure.canvas.draw()  # drawing an empty plot keeps the limits
    assert tuple(view.axes.get_xlim()) == EMPTY_FREQUENCY_LIMITS
    view.set_trace("t", [100.0, 1000.0], [1.0, 3.0], label="t")
    lo, hi = view.axes.get_ylim()
    assert 0.0 < lo < 1.0 and 3.0 < hi < 4.0  # follows the data


def test_empty_rt_axes():
    view = MplPlotView()
    setup_rt_axes(view)
    bands = band_set(1, 125, 4000)
    empty_rt_axes(view, bands)
    assert tuple(view.axes.get_xlim()) == pytest.approx((bands.lower[0], bands.upper[-1]))
    assert tuple(view.axes.get_ylim()) == EMPTY_RT_LIMITS
    major = view.axes.xaxis.get_major_locator().tick_values(1, 2)
    assert list(major) == [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]


def delayed_impulse(k, n=256, fs=48000.0, sign=1.0):
    from resonus_core.data import ImpulseResponse

    from resonus_room.core.files import impulse_to_frequency_response

    h = np.zeros(n)
    h[k] = sign
    h[k + 3] = 0.3 * sign  # a reflection after the peak
    ir = ImpulseResponse(h, fs, "1", Metadata("t", "0"))
    return ir, impulse_to_frequency_response(ir)


def test_peak_delay():
    from resonus_room.plotting import peak_delay

    ir, _ = delayed_impulse(40)
    assert peak_delay(ir) == pytest.approx(40 / 48000.0)
    ir, _ = delayed_impulse(17, sign=-1.0)  # the peak of |h|, also when negative
    assert peak_delay(ir) == pytest.approx(17 / 48000.0)
    assert isinstance(peak_delay(ir), float)


def test_phase_delay_removed():
    from resonus_room.plotting import peak_delay

    ir, fr = delayed_impulse(40)
    raw = phase_degrees(fr)
    assert np.max(np.abs(raw)) > 170  # the delay wraps the raw phase
    got = phase_degrees(fr, peak_delay(ir))
    # what is left is the phase of 1 + 0.3 exp(-j w 3/fs): |angle| <= asin(0.3)
    want = np.degrees(np.angle(1 + 0.3 * np.exp(-2j * np.pi * fr.frequency * 3 / 48000.0)))
    np.testing.assert_allclose(got, want, atol=1e-9)
    assert np.max(np.abs(got)) <= np.degrees(np.arcsin(0.3)) + 1e-9
    np.testing.assert_allclose(phase_degrees(fr, 0.0), raw)  # default: no delay removed


def test_show_phase_entry_with_delay():
    from resonus_room.plotting import peak_delay

    ir, fr = delayed_impulse(40)
    view = MplPlotView()
    show_phase_entry(view, fr, "ph:0", "e", "#000000", delay=peak_delay(ir))
    y = trace_line(view, "e").get_ydata()
    np.testing.assert_allclose(y, phase_degrees(fr, peak_delay(ir))[1:])


def test_peak_delay_empty():
    from resonus_core.data import ImpulseResponse

    from resonus_room.plotting import peak_delay

    with pytest.raises(ValueError, match="empty"):
        peak_delay(ImpulseResponse(np.zeros(0), 48000.0, "1", Metadata("t", "0")))
