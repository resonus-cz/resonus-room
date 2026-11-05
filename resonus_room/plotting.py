"""Plots of Resonus Room results, through the renderer-independent PlotView.

Graph conventions: log frequency axes with nominal band frequencies (ISO 3382-1:2009 9.1,
ISO 3382-2:2008 9.1: points connected by straight lines), levels in dB with stated
reference, time in s.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any, Final

import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from numpy.typing import ArrayLike, NDArray
from resonus_core.bands import BandSet, band_index, frequency_label, nominal_frequency
from resonus_core.data import BandResults, FrequencyResponse, ImpulseResponse
from resonus_core.plot import MARKER_SIZE, MplPlotView, PlotView

from resonus_room.core.absorption import AbsorptionResult
from resonus_room.core.analysis import RoomAnalysis
from resonus_room.core.decay import DecayCurve
from resonus_room.core.energy import EARLY_50, EARLY_80
from resonus_room.core.fit import DecayFit
from resonus_room.core.room_average import RoomAverage

# Display floor of level plots in dB (design choice: keeps -inf out of the axes).
LEVEL_FLOOR_DB = -200.0

# Wrapped phase in degrees: numpy.angle returns (-pi, pi], shown on fixed axes (D-20).
PHASE_LIMITS: Final = (-180.0, 180.0)
PHASE_TICKS: Final = (-180.0, -90.0, 0.0, 90.0, 180.0)

# Axes of the empty plots before any result (design choices, D-20; no standard applies):
EMPTY_FREQUENCY_LIMITS: Final = (20.0, 20000.0)  # Hz, audio range
EMPTY_MAGNITUDE_LIMITS: Final = (-60.0, 20.0)  # dB re 1, uncalibrated transfer functions
EMPTY_TIME_LIMITS: Final = (0.0, 1.0)  # s
EMPTY_LEVEL_LIMITS: Final = (-100.0, 0.0)  # dB re the peak
EMPTY_RT_LIMITS: Final = (0.0, 3.0)  # s, ordinary rooms to large halls


def setup_response_axes(view: PlotView) -> None:
    """Configure axes for a magnitude response (transfer function tab)."""
    view.set_axes(
        xlabel="Frequency [Hz]",
        ylabel="Magnitude [dB re 1]",
        xscale="log",
        title="Transfer function",
    )


def show_response(view: PlotView, fr: FrequencyResponse, trace_id: str, label: str) -> None:
    """Add or replace a magnitude-response trace (``fr.magnitude_db()``, 0 Hz omitted)."""
    keep = fr.frequency > 0
    view.set_trace(trace_id, fr.frequency[keep], fr.magnitude_db()[keep], label=label)


def ir_level(ir: ImpulseResponse) -> NDArray[np.float64]:
    """Level ``10*lg(h**2 / max(h**2))`` in dB re the peak, floored at ``LEVEL_FLOOR_DB``.

    Raises
    ------
    ValueError
        If the impulse response is all zero.
    """
    e = ir.data**2
    peak = float(np.max(e))
    if peak == 0:
        raise ValueError("Impulse response is all zero")
    with np.errstate(divide="ignore"):
        return np.maximum(10 * np.log10(e / peak), LEVEL_FLOOR_DB)


def setup_ir_axes(view: PlotView) -> None:
    """Configure axes for impulse-response levels and decay curves (time in s, dB)."""
    view.set_axes(
        xlabel="Time [s]",
        ylabel="Level [dB]",
        title="Impulse response",
    )


def show_ir(view: PlotView, ir: ImpulseResponse, trace_id: str, label: str) -> None:
    """Add or replace the IR level trace: ``ir.time`` vs :func:`ir_level`."""
    view.set_trace(trace_id, ir.time, ir_level(ir), label=label)


def show_band_ir(
    view: PlotView,
    signal: ArrayLike,
    sample_rate: float,
    start_time: float,
    trace_id: str,
    label: str,
) -> None:
    """Add or replace the level of a band-filtered impulse response.

    ``y = np.asarray(signal, float)``; ``t = start_time + np.arange(y.size) / sample_rate``
    vs ``10*lg(y**2 / max(y**2))`` floored at ``LEVEL_FLOOR_DB`` (as :func:`ir_level`);
    ``start_time`` is the onset in s, so the trace lines up with the decay curve.

    Raises
    ------
    ValueError
        If ``signal`` is all zero.
    """
    y = np.asarray(signal, float)
    peak = np.max(y**2)
    if peak == 0:
        raise ValueError("Signal is all zero")
    with np.errstate(divide="ignore"):
        level = np.maximum(10 * np.log10(y**2 / peak), LEVEL_FLOOR_DB)
    t = start_time + np.arange(y.size) / sample_rate
    view.set_trace(trace_id, t, level, label=label)


# Style of the time-window markers in the impulse-response plot (design choice).
WINDOW_COLOR = "0.45"
WINDOW_LINESTYLE: Final = "--"


def time_windows(
    start_time: float, curve: DecayCurve, sample_rate: float
) -> list[tuple[str, float]]:
    """Return the time-window markers of one analysed response, in plot time.

    ``[("onset", start_time), ("50 ms", start_time + EARLY_50), ("80 ms", start_time +
    EARLY_80), ("t1", start_time + curve.truncation / sample_rate)]``, with ``EARLY_50``,
    ``EARLY_80`` from ``resonus_room.core.energy``.

    Parameters
    ----------
    start_time
        Onset in s from the first IR sample (``analysis.onset / sample_rate``).
    curve
        Decay curve of the shown band (its ``truncation`` is relative to the onset).
    sample_rate
        Sample rate in Hz.

    Returns
    -------
    list of (label, time in s)

    References
    ----------
    ISO 3382-1:2009 A.3.4 (onset, t = 0), A.2.3 (C50/C80/D50 limits of 50 and 80 ms),
    5.3.3 (truncation point t1).
    """
    return [
        ("onset", start_time),
        ("50 ms", start_time + EARLY_50),
        ("80 ms", start_time + EARLY_80),
        ("t1", start_time + curve.truncation / sample_rate),
    ]


def show_time_windows(view: MplPlotView, windows: Sequence[tuple[str, float]]) -> list[Any]:
    """Draw vertical time-window markers with their labels; they are not traces.

    For each ``(label, t)``: ``view.axes.vlines(t, 0, 1, transform=
    view.axes.get_xaxis_transform(), colors=WINDOW_COLOR, linestyles=WINDOW_LINESTYLE,
    linewidth=0.8)`` (a line collection spanning the axes height: neither in
    ``axes.get_lines()`` nor in the legend) and ``view.axes.text(t, 0.98, " " + label,
    transform=view.axes.get_xaxis_transform(), va="top", ha="left", fontsize="small",
    color=WINDOW_COLOR)``. Then ``view.axes.relim(visible_only=True)``,
    ``view.axes.autoscale_view()`` (``vlines`` adds y 0...1 in data units to the data
    limits; ``relim`` ignores collections) and ``view.refresh()``.

    Parameters
    ----------
    view
        Impulse-response plot.
    windows
        ``(label, time in s)`` pairs, e.g. from :func:`time_windows`.

    Returns
    -------
    list
        The artists, line collection then text per marker, so the caller can hide or
        remove them.
    """
    artists: list[Any] = []
    for label, t in windows:
        line = view.axes.vlines(
            t,
            0,
            1,
            transform=view.axes.get_xaxis_transform(),
            colors=WINDOW_COLOR,
            linestyles=WINDOW_LINESTYLE,
            linewidth=0.8,
        )
        text = view.axes.text(
            t,
            0.98,
            " " + label,
            transform=view.axes.get_xaxis_transform(),
            va="top",
            ha="left",
            fontsize="small",
            color=WINDOW_COLOR,
        )
        artists.extend([line, text])
    view.axes.relim(visible_only=True)
    view.axes.autoscale_view()
    view.refresh()
    return artists


def show_decay(
    view: PlotView, curve: DecayCurve, start_time: float, trace_id: str, label: str
) -> None:
    """Add or replace a decay-curve trace.

    Add or replace a decay-curve trace: ``curve.time + start_time`` vs ``curve.level``
    (floored at ``LEVEL_FLOOR_DB``); ``start_time`` is the onset in s.
    """
    view.set_trace(
        trace_id,
        curve.time + start_time,
        np.maximum(curve.level, LEVEL_FLOOR_DB),
        label=label,
    )


def show_fit(view: PlotView, fit: DecayFit, start_time: float, trace_id: str, label: str) -> None:
    """Add or replace the fitted line of ``fit`` between ``t_start`` and ``t_stop``.

    Two points ``t = [t_start, t_stop]``: ``x = t + start_time``,
    ``y = intercept + slope * t``. Nothing is drawn (an existing trace is removed) if the
    fit has no decay time.
    """
    if not np.isfinite(fit.decay_time):
        view.remove_trace(trace_id)
        return
    t = np.array([fit.t_start, fit.t_stop])
    view.set_trace(trace_id, t + start_time, fit.intercept + fit.slope * t, label=label)


def setup_rt_axes(view: PlotView) -> None:
    """Configure axes for decay times per band (log frequency, s)."""
    view.set_axes(
        xlabel="Frequency [Hz]",
        ylabel="Reverberation time [s]",
        xscale="log",
        title="Reverberation time",
    )


def show_rt(view: PlotView, analysis: RoomAnalysis, name: str, trace_id: str, label: str) -> None:
    """Add or replace the per-band values of one parameter.

    Add or replace the per-band values of ``name`` (``"EDT"``, ``"T20"``, ``"T30"``)
    against the nominal band frequencies; missing values are NaN gaps.
    """
    view.set_trace(trace_id, analysis.bands.nominal, analysis.values(name), label=label)


def setup_absorption_axes(view: PlotView) -> None:
    """Configure axes for absorption coefficients (log frequency, alpha_s)."""
    view.set_axes(
        xlabel="Frequency [Hz]",
        ylabel="Sound absorption coefficient [-]",
        xscale="log",
        title="Sound absorption (ISO 354)",
    )


def show_absorption(view: PlotView, result: AbsorptionResult, trace_id: str, label: str) -> None:
    """Add or replace ``alpha_s`` per band against the nominal frequencies."""
    view.set_trace(trace_id, result.frequency, result.alpha, label=label)


# Preferred quantity to plot when none is named (design choice): reverberation time first,
# then the ISO 354 absorption coefficient.
PREFERRED_QUANTITIES = ("T20", "alpha_s", "T30", "EDT")


def default_quantity(results: BandResults) -> str:
    """Return the quantity shown when none is named.

    The first of ``PREFERRED_QUANTITIES`` present in ``results.values``, else the first
    key of ``results.values``.
    """
    for q in PREFERRED_QUANTITIES:
        if q in results.values:
            return q
    return next(iter(results.values))


def setup_band_axes(view: PlotView, quantity: str, unit: str) -> None:
    """Configure axes for per-band values.

    ``xlabel="Frequency [Hz]"``, ``ylabel=f"{quantity} [{unit}]"``, ``xscale="log"``,
    ``title=quantity``.
    """
    view.set_axes(
        xlabel="Frequency [Hz]",
        ylabel=f"{quantity} [{unit}]",
        xscale="log",
        title=quantity,
    )


def show_band_results(
    view: PlotView, results: BandResults, quantity: str, trace_id: str, label: str
) -> None:
    """Add or replace one quantity of ``results`` against the exact band centres (log axis).

    Raises
    ------
    KeyError
        If ``quantity`` is not in ``results.values``.
    """
    if quantity not in results.values:
        raise KeyError(quantity)
    view.set_trace(trace_id, results.center, results.values[quantity], label=label)


# Overlay of session entries (D-19) --------------------------------------------------

# Design choice (user decision 2026-11-04, D-19): in the RT tab the colour identifies the
# entry and the line style the parameter.
PARAMETER_LINESTYLES: Final = {"EDT": ":", "T20": "--", "T30": "-"}
# Design choices (D-19): legend title and a wider line for the room average.
AVERAGE_TITLE = "Room average"
AVERAGE_LINE_WIDTH = 2.5


def trace_line(view: MplPlotView, label: str) -> Any:
    """Return the matplotlib line labelled ``label`` (first match in ``axes.get_lines()``).

    Raises
    ------
    ValueError
        ``f"no trace labelled {label!r}"`` if there is none.
    """
    for line in view.axes.get_lines():
        if line.get_label() == label:
            return line
    raise ValueError(f"no trace labelled {label!r}")


def rt_label(name: str, title: str) -> str:
    """Legend text of one parameter of one entry: ``f"{name}, {title}"``."""
    return f"{name}, {title}"


def show_rt_entry(
    view: MplPlotView,
    analysis: RoomAnalysis,
    name: str,
    trace_id: str,
    title: str,
    color: str,
) -> None:
    """Add or replace one parameter of one entry in the RT tab.

    ``show_rt(view, analysis, name, trace_id, rt_label(name, title))``; then on
    ``trace_line(view, rt_label(name, title))``: ``set_color(color)``,
    ``set_linestyle(PARAMETER_LINESTYLES[name])``, ``set_marker("o")``,
    ``set_markersize(MARKER_SIZE)`` (resonus-core ``plot.MARKER_SIZE``).
    """
    label = rt_label(name, title)
    show_rt(view, analysis, name, trace_id, label)
    line = trace_line(view, label)
    line.set_color(color)
    line.set_linestyle(PARAMETER_LINESTYLES[name])
    line.set_marker("o")
    line.set_markersize(MARKER_SIZE)


def show_response_entry(
    view: MplPlotView, fr: FrequencyResponse, trace_id: str, label: str, color: str
) -> None:
    """Add or replace a transfer function in the entry's colour.

    ``show_response(view, fr, trace_id, label)``, then
    ``trace_line(view, label).set_color(color)``.
    """
    show_response(view, fr, trace_id, label)
    trace_line(view, label).set_color(color)


def show_parameters(view: MplPlotView, visible: Collection[str]) -> None:
    """Show only the RT lines of the parameters in ``visible``.

    For every line of ``view.axes.get_lines()``: ``name = label.split(", ", 1)[0]``; if
    ``name`` is in ``PARAMETER_LINESTYLES``, ``line.set_visible(name in visible)`` (other
    lines are left alone). Legend of the visible lines (``view.axes.legend(handles=shown,
    loc="best")``, or remove the legend if none is visible), ``view.axes.relim(
    visible_only=True)``, ``view.axes.autoscale_view()``, ``view.refresh()``.
    """
    # Update visibility of parameter lines
    for line in view.axes.get_lines():
        name = str(line.get_label()).split(", ", 1)[0]
        if name in PARAMETER_LINESTYLES:
            line.set_visible(name in visible)
    # Collect all currently visible lines for the legend
    shown = [line for line in view.axes.get_lines() if line.get_visible()]
    if shown:
        view.axes.legend(handles=shown, loc="best")
    else:
        legend = view.axes.get_legend()
        if legend is not None:
            legend.remove()
    view.axes.relim(visible_only=True)
    view.axes.autoscale_view()
    view.refresh()


def keep_traces(view: PlotView, keep: Collection[str]) -> None:
    """Remove every trace whose id is not in ``keep`` (``view.remove_trace``)."""
    for tid in list(view.trace_ids()):
        if tid not in keep:
            view.remove_trace(tid)


def update_legend(view: MplPlotView) -> None:
    """Rebuild the legend from the visible lines (current colours); none if none is visible.

    Needed after a line's colour is changed: matplotlib copies the colours into the
    legend when it is drawn (``MplPlotView.set_trace`` draws it before the entry colour
    is applied).
    """
    shown = [ln for ln in view.axes.get_lines() if ln.get_visible()]
    if shown:
        view.axes.legend(handles=shown, loc="best")
    else:
        legend = view.axes.get_legend()
        if legend is not None:
            legend.remove()
    view.refresh()


def show_average(
    view: MplPlotView, average: RoomAverage, name: str, trace_id: str, color: str
) -> None:
    """Add or replace the room average of one parameter in the RT tab (D-19).

    ``view.set_trace(trace_id, average.bands.nominal, average.mean[name],
    label=rt_label(name, AVERAGE_TITLE))``; then on that line (``trace_line``):
    ``set_color(color)``, ``set_linestyle(PARAMETER_LINESTYLES[name])``,
    ``set_marker("s")`` (square, design choice: tells the average from the entries),
    ``set_markersize(MARKER_SIZE)``, ``set_linewidth(AVERAGE_LINE_WIDTH)``.
    """
    view.set_trace(
        trace_id,
        average.bands.nominal,
        average.mean[name],
        label=rt_label(name, AVERAGE_TITLE),
    )
    line = trace_line(view, rt_label(name, AVERAGE_TITLE))
    line.set_color(color)
    line.set_linestyle(PARAMETER_LINESTYLES[name])
    line.set_marker("s")
    line.set_markersize(MARKER_SIZE)
    line.set_linewidth(AVERAGE_LINE_WIDTH)


def band_ticks(view: MplPlotView, frequencies: ArrayLike, fraction: int) -> None:
    """Put the frequency ticks of a band plot at the band centres (not decades).

    ``idx = band_index(frequencies, fraction)`` (exact or nominal centres both map to
    their band); ``nominal = nominal_frequency(idx, fraction)``. Octave bands are the
    major ticks: for ``fraction == 1`` every band, for ``fraction == 3`` the bands with
    ``idx % 3 == 0`` (one-third-octave index 3k is the octave band k of the base-10
    system, 1 kHz at index 0). The other one-third-octave bands are minor ticks without
    labels.

    ``view.axes.xaxis``: ``set_major_locator(FixedLocator(major))``,
    ``set_major_formatter(FuncFormatter(lambda v, _pos: frequency_label(v)))``,
    ``set_minor_locator(FixedLocator(minor))``, ``set_minor_formatter(NullFormatter())``;
    then ``view.refresh()``. Nothing is changed for an empty ``frequencies``.

    Raises
    ------
    ValueError
        From :func:`resonus_core.bands.nominal_frequency` for a fraction other than 1
        or 3 or a band outside the nominal table, or from ``band_index`` for
        frequencies <= 0.
    """
    f = np.asarray(frequencies, dtype=float)
    if f.size == 0:
        return
    idx = band_index(f, fraction)
    nominal = nominal_frequency(idx, fraction)
    is_major = np.ones_like(idx, dtype=bool) if fraction == 1 else (idx % 3 == 0)
    major = nominal[is_major]
    minor = nominal[~is_major]
    view.axes.xaxis.set_major_locator(FixedLocator(major))
    view.axes.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: frequency_label(v)))
    view.axes.xaxis.set_minor_locator(FixedLocator(minor))
    view.axes.xaxis.set_minor_formatter(NullFormatter())
    view.refresh()


def setup_phase_axes(view: MplPlotView) -> None:
    """Configure axes for the phase of a transfer function (below the magnitude).

    ``view.set_axes(xlabel="Frequency [Hz]", ylabel="Phase [°]", xscale="log",
    ylim=PHASE_LIMITS)``, then ``view.axes.yaxis.set_major_locator(FixedLocator(
    PHASE_TICKS))`` and ``view.refresh()``.
    """
    view.set_axes(
        xlabel="Frequency [Hz]",
        ylabel="Phase [°]",
        xscale="log",
        ylim=PHASE_LIMITS,
    )
    view.axes.yaxis.set_major_locator(FixedLocator(PHASE_TICKS))
    view.refresh()


def peak_delay(ir: ImpulseResponse) -> float:
    """Return the time of the impulse response's peak in s, ``argmax(|ir.data|) / fs``.

    Reference time of the phase plot (user decision D-20, 2026-11-05: delay to the IR
    peak removed, like REW's IR-peak reference). Whole samples: the residual phase error
    is at most ``180 f / fs`` degrees (half a sample).

    Raises
    ------
    ValueError
        ``"impulse response is empty"`` for ``ir.data.size == 0``.
    """
    if ir.data.size == 0:
        raise ValueError("impulse response is empty")
    return float(np.argmax(np.abs(ir.data)) / ir.sample_rate)


def phase_degrees(fr: FrequencyResponse, delay: float = 0.0) -> NDArray[np.float64]:
    """Return the wrapped phase of ``fr.values`` in degrees, in (-180, 180], delay removed.

    ``np.degrees(np.angle(fr.values * np.exp(2j * np.pi * fr.frequency * delay)))``: a
    delay of ``delay`` s (:func:`peak_delay`) is removed, ``delay = 0`` gives the phase
    relative to the time origin of the impulse response.
    """
    shifted = fr.values * np.exp(2j * np.pi * fr.frequency * delay)
    return np.degrees(np.angle(shifted))


def hide_legend(view: MplPlotView) -> None:
    """Remove the legend of ``view`` if it has one (the phase plot shares the magnitude's)."""
    legend = view.axes.get_legend()
    if legend is not None:
        legend.remove()
    view.refresh()


def show_phase_entry(
    view: MplPlotView,
    fr: FrequencyResponse,
    trace_id: str,
    label: str,
    color: str,
    delay: float = 0.0,
) -> None:
    """Add or replace the phase of a transfer function in the entry's colour.

    ``keep = fr.frequency > 0``; ``view.set_trace(trace_id, fr.frequency[keep],
    phase_degrees(fr, delay)[keep], label=label)``; ``trace_line(view, label).set_color(color)``;
    ``view.axes.set_ylim(*PHASE_LIMITS)`` (set_trace autoscales the y axis);
    :func:`hide_legend`.
    """
    keep = fr.frequency > 0
    view.set_trace(trace_id, fr.frequency[keep], phase_degrees(fr, delay)[keep], label=label)
    trace_line(view, label).set_color(color)
    view.axes.set_ylim(*PHASE_LIMITS)
    hide_legend(view)


def empty_limits(view: MplPlotView, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    """Set the axes limits of an empty plot, keeping autoscaling on for the next data.

    ``view.axes.set_xlim(*xlim, auto=True)``, ``view.axes.set_ylim(*ylim, auto=True)``,
    ``view.refresh()``.
    """
    view.axes.set_xlim(*xlim, auto=True)
    view.axes.set_ylim(*ylim, auto=True)
    view.refresh()


def empty_rt_axes(view: MplPlotView, bands: BandSet) -> None:
    """Empty reverberation-time plot: the band range of the settings and their ticks.

    ``band_ticks(view, bands.nominal, bands.fraction)``, then ``empty_limits(view,
    (bands.lower[0], bands.upper[-1]), EMPTY_RT_LIMITS)``. Nothing for an empty band set.
    """
    if len(bands) == 0:
        return
    band_ticks(view, bands.nominal, bands.fraction)
    empty_limits(view, (float(bands.lower[0]), float(bands.upper[-1])), EMPTY_RT_LIMITS)
