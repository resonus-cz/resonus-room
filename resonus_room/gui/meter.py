"""Vertical input level meters of the level test, one per recorded input (GTK, D-21).

Each meter shows the peak level of the latest block as a bar coloured by its zone
(:func:`resonus_room.core.levels.meter_zone`), the highest peak so far as a line (peak
hold), the RMS level as a short tick, and a clipping indicator on top that stays lit
until the next test. A scale on the left marks ``METER_MARKS``. Colours are design
choices (GNOME palette).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from gi.repository import Gtk

from resonus_room.core.levels import METER_MARKS, InputLevel, meter_fraction, meter_zone

# Zone colours (design choice, GNOME palette): red 3, green 4, yellow 5, light 5.
ZONE_COLORS: Final = {
    "hot": (0.878, 0.106, 0.141),
    "good": (0.180, 0.761, 0.494),
    "low": (0.898, 0.647, 0.039),
    "quiet": (0.604, 0.600, 0.588),
}
CLIP_COLOR: Final = (0.878, 0.106, 0.141)
METER_WIDTH: Final = 12  # px
METER_HEIGHT: Final = 64  # px, three ribbon rows minus the label
CLIP_HEIGHT: Final = 6  # px
SCALE_WIDTH: Final = 30  # px


def _bar_top(fraction: float, height: int) -> float:
    """Y coordinate of a level on the meter (cairo: y grows downwards)."""
    top = CLIP_HEIGHT + 2
    return top + (1.0 - fraction) * (height - top)


class LevelMeters:
    """Meters for the inputs named by ``labels`` (order of the recording).

    Attributes
    ----------
    box : Gtk.Box
        Scale and meters, for the ribbon.
    labels : list of str
    current : list of InputLevel or None
        Levels of the latest block (None before the first one).
    held_peak : list of float
        Highest peak in dB since the last :meth:`reset` (``-inf`` at first).
    clipped : list of bool
        True once an input clipped since the last :meth:`reset`.
    """

    def __init__(self, labels: Sequence[str]) -> None:
        self.box = Gtk.Box(spacing=3, valign=Gtk.Align.CENTER)
        self.box.set_tooltip_text(
            "Input levels of the level test, dB re full scale: bar = peak, line = highest "
            "peak, tick = RMS; red box = clipped"
        )
        self._scale = Gtk.DrawingArea(content_width=SCALE_WIDTH, content_height=METER_HEIGHT)
        self._scale.set_draw_func(self._draw_scale)
        self._areas: list[Gtk.DrawingArea] = []
        self.labels: list[str] = []
        self.current: list[InputLevel | None] = []
        self.held_peak: list[float] = []
        self.clipped: list[bool] = []
        self.set_labels(labels)

    def set_labels(self, labels: Sequence[str]) -> None:
        """Rebuild the meters for ``labels``; all levels are reset."""
        child = self.box.get_first_child()
        while child is not None:
            following = child.get_next_sibling()
            self.box.remove(child)
            child = following
        # same column structure as a meter (empty caption), so the marks line up
        scale_column = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=1)
        scale_column.append(self._scale)
        blank = Gtk.Label(label=" ")
        blank.add_css_class("ribbon-caption")
        scale_column.append(blank)
        self.box.append(scale_column)
        self.labels = list(labels)
        self._areas = []
        for k, label in enumerate(self.labels):
            column = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=1)
            area = Gtk.DrawingArea(content_width=METER_WIDTH, content_height=METER_HEIGHT)
            area.set_draw_func(self._draw_meter, k)
            caption = Gtk.Label(label=label)
            caption.add_css_class("ribbon-caption")
            column.append(area)
            column.append(caption)
            self.box.append(column)
            self._areas.append(area)
        self.reset()

    def reset(self) -> None:
        """Clear levels, peak hold and clipping indicators (start of a test)."""
        n = len(self.labels)
        self.current = [None] * n
        self.held_peak = [float("-inf")] * n
        self.clipped = [False] * n
        self._redraw()

    def update(self, levels: Sequence[InputLevel]) -> None:
        """Show the levels of one block (or of the whole test): one per meter.

        ``current = levels``; ``held_peak`` and ``clipped`` accumulate. A list of another
        length (inputs changed meanwhile) is ignored.
        """
        if len(levels) != len(self.labels):
            return
        self.current = list(levels)
        for k, v in enumerate(levels):
            self.held_peak[k] = max(self.held_peak[k], v.peak_db)
            self.clipped[k] = self.clipped[k] or v.clipped
        self._redraw()

    def _redraw(self) -> None:
        for area in self._areas:
            area.queue_draw()

    def _draw_scale(self, _area: Gtk.DrawingArea, cr: Any, width: int, height: int) -> None:
        cr.set_source_rgba(0.5, 0.5, 0.5, 0.9)
        cr.set_font_size(8.0)
        for mark in METER_MARKS:
            y = _bar_top(meter_fraction(mark), height)
            text = f"{mark:g}"
            extents = cr.text_extents(text)
            # keep the text inside the area (the -100 mark is at the bottom edge)
            y_text = min(max(y + extents.height / 2, extents.height), height - 1)
            cr.move_to(width - extents.width - 2, y_text)
            cr.show_text(text)

    def _draw_meter(self, _area: Gtk.DrawingArea, cr: Any, width: int, height: int, k: int) -> None:
        top = CLIP_HEIGHT + 2
        # background of the bar
        cr.set_source_rgba(0.5, 0.5, 0.5, 0.18)
        cr.rectangle(0, top, width, height - top)
        cr.fill()
        # clipping indicator
        if self.clipped[k]:
            cr.set_source_rgb(*CLIP_COLOR)
            cr.rectangle(0, 0, width, CLIP_HEIGHT)
            cr.fill()
        else:
            cr.set_source_rgba(0.5, 0.5, 0.5, 0.5)
            cr.set_line_width(1.0)
            cr.rectangle(0.5, 0.5, width - 1, CLIP_HEIGHT - 1)
            cr.stroke()
        level = self.current[k]
        if level is not None:
            y = _bar_top(meter_fraction(level.peak_db), height)
            cr.set_source_rgb(*ZONE_COLORS[meter_zone(level.peak_db)])
            cr.rectangle(0, y, width, height - y)
            cr.fill()
            y_rms = _bar_top(meter_fraction(level.rms_db), height)
            cr.set_source_rgba(1.0, 1.0, 1.0, 0.9)
            cr.rectangle(width / 2, y_rms - 0.75, width / 2, 1.5)
            cr.fill()
        if self.held_peak[k] > float("-inf"):
            y = _bar_top(meter_fraction(self.held_peak[k]), height)
            cr.set_source_rgb(*ZONE_COLORS[meter_zone(self.held_peak[k])])
            cr.rectangle(0, y - 1, width, 2)
            cr.fill()
        # marks across the bar
        cr.set_source_rgba(0.0, 0.0, 0.0, 0.35)
        for mark in METER_MARKS[:-1]:
            y = _bar_top(meter_fraction(mark), height)
            cr.rectangle(0, y - 0.5, width, 1)
            cr.fill()
