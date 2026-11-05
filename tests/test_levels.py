"""Level test: test noise, input levels, meter zones (D-21)."""

import math

import numpy as np
import pytest

from resonus_room.core.levels import (
    CLIP_THRESHOLD,
    METER_MARKS,
    METER_RANGE,
    TEST_DURATION,
    InputLevel,
    combined_levels,
    input_levels,
    level_noise,
    level_test,
    meter_fraction,
    meter_labels,
    meter_zone,
    pair_labels,
)
from resonus_room.core.settings import AppSettings

FS = 48000.0


def test_constants():
    assert TEST_DURATION == 1.0
    assert METER_MARKS == (-12.0, -42.0, -60.0, -100.0)
    assert METER_RANGE == (-100.0, 0.0)
    assert CLIP_THRESHOLD == 1.0 - 2.0**-15


def test_test_noise_peak_at_level():
    x = level_noise(-12.0, FS)
    assert x.shape == (48000,) and x.dtype == np.float64
    assert np.max(np.abs(x)) == pytest.approx(10 ** (-12 / 20), rel=1e-12)
    assert np.std(x) > 0.1 * np.max(np.abs(x))  # noise, not a single spike
    np.testing.assert_array_equal(x, level_noise(-12.0, FS))  # deterministic
    assert level_noise(0.0, 1000.0, 0.5).shape == (500,)
    assert np.max(np.abs(level_noise(0.0, 1000.0, 0.5))) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("args", "words"),
    [((0.5, FS), "level"), ((-6.0, FS, 0.0), "positive"), ((-6.0, 0.0), "positive")],
)
def test_test_noise_errors(args, words):
    with pytest.raises(ValueError, match=words):
        level_noise(*args)


def test_input_levels_closed_form():
    n = 4800
    t = np.arange(n) / FS
    sine = 0.5 * np.sin(2 * np.pi * 1000 * t)  # whole periods: rms = 0.5 / sqrt(2)
    rec = np.stack([sine, np.zeros(n), np.full(n, 0.25)], axis=1)
    lv = input_levels(rec)
    assert len(lv) == 3 and all(isinstance(v, InputLevel) for v in lv)
    assert lv[0].peak_db == pytest.approx(20 * math.log10(0.5), abs=1e-6)
    assert lv[0].rms_db == pytest.approx(20 * math.log10(0.5 / math.sqrt(2)), abs=1e-6)
    assert lv[1].peak_db == -math.inf and lv[1].rms_db == -math.inf
    assert lv[2].peak_db == pytest.approx(lv[2].rms_db) == pytest.approx(20 * math.log10(0.25))
    assert not any(v.clipped for v in lv)
    assert all(isinstance(v.peak_db, float) and isinstance(v.clipped, bool) for v in lv)


def test_input_levels_clipping_and_1d():
    x = np.zeros(10)
    x[3] = -CLIP_THRESHOLD
    (lv,) = input_levels(x)
    assert lv.clipped
    assert lv.peak_db == pytest.approx(20 * math.log10(CLIP_THRESHOLD))
    assert lv.rms_db == pytest.approx(20 * math.log10(CLIP_THRESHOLD / math.sqrt(10)))
    x[3] = 1.0 - 2.0**-14  # just below the threshold
    assert not input_levels(x)[0].clipped


@pytest.mark.parametrize("bad", [np.zeros((0, 2)), np.zeros((2, 2, 2)), np.zeros(0)])
def test_input_levels_errors(bad):
    with pytest.raises(ValueError, match="non-empty"):
        input_levels(bad)


@pytest.mark.parametrize(
    ("level", "zone"),
    [
        (0.0, "hot"), (-12.0, "hot"), (-12.1, "good"), (-42.0, "good"), (-42.1, "low"),
        (-60.0, "low"), (-60.1, "quiet"), (-120.0, "quiet"), (-math.inf, "quiet"),
        (math.nan, "quiet"),
    ],
)  # fmt: skip
def test_meter_zone(level, zone):
    assert meter_zone(level) == zone


@pytest.mark.parametrize(
    ("level", "fraction"),
    [(-100.0, 0.0), (0.0, 1.0), (-50.0, 0.5), (-12.0, 0.88), (-150.0, 0.0), (6.0, 1.0),
     (-math.inf, 0.0), (math.nan, 0.0)],
)  # fmt: skip
def test_meter_fraction(level, fraction):
    got = meter_fraction(level)
    assert isinstance(got, float) and got == pytest.approx(fraction)


def test_meter_labels():
    assert meter_labels(AppSettings()) == ["m1"]
    s = AppSettings(input_channels=(1, 2), input_labels=("m1", "m2"), reference_channel=3)
    assert meter_labels(s) == ["m1", "m2", "ref"]


class FakeIO:
    """One input hears output 1 at half amplitude and output 2 at double (clipped)."""

    sample_rate = 8000.0
    n_inputs = 1
    n_outputs = 2

    def __init__(self):
        self.played = []

    def playrec_blocks(self, signal, on_block, block_size=1024):
        x = np.asarray(signal)
        self.played.append(x)
        rec = np.clip(0.5 * x[:, :1] + 2.0 * x[:, 1:2], -1, 1)
        for i in range(0, len(rec), block_size):
            on_block(rec[i : i + block_size])
        return rec


def two_outputs(**kw):
    return AppSettings(output_channels=(1, 2), output_labels=("s1", "s2"), level=-6.0, **kw)


def test_level_test_one_output_after_the_other():
    io = FakeIO()
    seen = []
    results = level_test(two_outputs(), io, lambda out, lv: seen.append((out, lv)))
    # each run plays the noise on one output only
    assert len(io.played) == 2 and all(x.shape == (8000, 2) for x in io.played)
    assert np.all(io.played[0][:, 1] == 0) and np.all(io.played[1][:, 0] == 0)
    np.testing.assert_array_equal(io.played[0][:, 0], io.played[1][:, 1])
    assert np.max(np.abs(io.played[0][:, 0])) == pytest.approx(10 ** (-6 / 20))
    # results per output, in the order of the output labels
    assert [out for out, _ in results] == ["s1", "s2"]
    (_, s1), (_, s2) = results
    assert s1[0].peak_db == pytest.approx(-6.0 + 20 * math.log10(0.5)) and not s1[0].clipped
    assert s2[0].clipped
    # live blocks tagged with their output: ceil(8000 / 1024) = 8 each
    assert [out for out, _ in seen] == ["s1"] * 8 + ["s2"] * 8
    assert all(len(lv) == 1 and isinstance(lv[0], InputLevel) for _, lv in seen)
    assert level_test(two_outputs(), FakeIO()) == results  # on_block optional


def test_level_test_output_count_mismatch():
    with pytest.raises(ValueError, match="2 outputs, but 1 output labels"):
        level_test(AppSettings(), FakeIO())


def test_pair_labels():
    assert pair_labels(AppSettings()) == [["s1m1"]]
    s = two_outputs(input_channels=(1, 2), input_labels=("m1", "m2"), reference_channel=3)
    assert pair_labels(s) == [["s1m1", "s1m2", "s1ref"], ["s2m1", "s2m2", "s2ref"]]


def test_combined_levels():
    a = InputLevel(-20.0, -30.0, False)
    b = InputLevel(-10.0, -35.0, True)
    c = InputLevel(-50.0, -60.0, False)
    got = combined_levels([("s1", [a, c]), ("s2", [b, c])])
    assert got == [InputLevel(-10.0, -30.0, True), InputLevel(-50.0, -60.0, False)]
    with pytest.raises(ValueError, match="no results"):
        combined_levels([])
