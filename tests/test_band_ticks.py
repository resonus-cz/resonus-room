"""Frequency ticks at the band centres."""

import numpy as np
import pytest
from resonus_core.bands import band_set
from resonus_core.plot import MplPlotView

from resonus_room.plotting import band_ticks

OCTAVES = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]


def ticks(view):
    ax = view.axes.xaxis
    major = list(ax.get_major_locator().tick_values(1, 2))
    minor = list(ax.get_minor_locator().tick_values(1, 2))
    labels = [ax.get_major_formatter()(v, 0) for v in major]
    minor_labels = [ax.get_minor_formatter()(v, 0) for v in minor]
    return major, minor, labels, minor_labels


def test_octaves():
    view = MplPlotView()
    view.set_axes(xlabel="f", ylabel="T", xscale="log")
    band_ticks(view, OCTAVES, 1)
    major, minor, labels, _ = ticks(view)
    assert major == OCTAVES and minor == []
    assert labels == ["125", "250", "500", "1k", "2k", "4k"]


@pytest.mark.parametrize("use_centres", [False, True])
def test_third_octaves(use_centres):
    bands = band_set(3, 100, 5000)
    freq = bands.center if use_centres else bands.nominal
    view = MplPlotView()
    view.set_axes(xlabel="f", ylabel="T", xscale="log")
    band_ticks(view, freq, 3)
    major, minor, labels, minor_labels = ticks(view)
    assert major == OCTAVES
    assert minor == [100.0, 160.0, 200.0, 315.0, 400.0, 630.0, 800.0, 1250.0, 1600.0] + [
        2500.0,
        3150.0,
        5000.0,
    ]
    assert labels[-1] == "4k" and set(minor_labels) == {""}


def test_small_set_and_errors():
    view = MplPlotView()
    band_ticks(view, [31.5, 63.0], 1)
    assert ticks(view)[0] == [31.5, 63.0]
    before = ticks(view)
    band_ticks(view, [], 3)  # nothing to do
    assert ticks(view) == before
    with pytest.raises(ValueError):
        band_ticks(view, [1000.0], 6)
    with pytest.raises(ValueError):
        band_ticks(view, np.array([0.0]), 1)
