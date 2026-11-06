"""STI inputs and the STI at a position (PROVISIONAL, D-31).

Tolerances from a private reference over three seeds (2026-11-05): noise only at
S/N 0 dB, STI within 0.0013 of 0.5 (test: 0.01); exponential decay T = 1 s without noise,
0.0129-0.0142 above Houtgast & Steeneken 1985 Eq. (1) via sti_from_mtf, a bias of the
random decay's realisation (test: 0.025).
"""

import dataclasses

import numpy as np
import pytest
from resonus_core.calibration import P0
from resonus_core.data import ImpulseResponse, Metadata, TimeSignal

from resonus_room.core.calibrate import OFFSET_KEY
from resonus_room.core.settings import AppSettings
from resonus_room.core.sti import MODULATION_FREQUENCIES, mtf_reverberation, sti_from_mtf
from resonus_room.core.sti_inputs import (
    DEFAULT_NOISE_DURATION,
    NOISE_ROLE,
    SOURCE_ROLE,
    STI_WINDOW_S,
    StiReport,
    noise_band_levels,
    record_noise,
    role,
    source_calibration,
    sti_at_position,
)

FS = 32000.0
META = Metadata("test", "0")


def ir(data, unit="Pa/FS", **extra):
    return ImpulseResponse(
        np.asarray(data, float), FS, unit, dataclasses.replace(META, extra=extra)
    )


def noise(data):
    return ir(data, "Pa", role=NOISE_ROLE)


class NoiseIO:
    sample_rate = FS
    n_inputs = 3
    n_outputs = 1

    def rec(self, n):
        self.n = n
        return np.tile([0.1, 0.2, 0.9], (n, 1))


def settings():
    return AppSettings(
        input_channels=(1, 2), input_labels=("m1", "m2"), reference_channel=3, sample_rate=48000
    )


def test_constants():
    assert STI_WINDOW_S == 8.0 and DEFAULT_NOISE_DURATION == 20.0
    assert NOISE_ROLE == "background-noise" and SOURCE_ROLE == "source-calibration"


def test_role():
    assert role(ir([1.0])) == ""
    assert role(noise([1.0])) == NOISE_ROLE


def test_record_noise():
    io = NoiseIO()
    out = record_noise(settings(), io, [94.0, 100.0], duration=0.5)
    assert io.n == 16000
    assert len(out) == 2
    a, b = out
    # a TimeSignal since resonus-core 0.11.0 (Q-C9), with the start time of the recording
    assert isinstance(a, TimeSignal) and a.start_utc == a.metadata.created
    assert a.unit == "Pa" and a.sample_rate == FS and a.data.shape == (16000,)
    np.testing.assert_allclose(a.data, 0.1 * P0 * 10 ** (94 / 20))
    np.testing.assert_allclose(b.data, 0.2 * P0 * 10**5)
    assert a.metadata.extra == {"role": NOISE_ROLE, "label": "m1", OFFSET_KEY: 94.0}
    assert b.metadata.extra["label"] == "m2"
    assert a.metadata.software == "resonus-room"
    assert a.metadata.description == "Background noise m1"
    assert a.metadata.created.endswith("+00:00") and len(a.metadata.created) == 25
    with pytest.raises(ValueError, match="duration must be > 0"):
        record_noise(settings(), NoiseIO(), [94.0, 100.0], duration=0.0)
    with pytest.raises(ValueError, match="input m2 is not calibrated"):
        record_noise(settings(), NoiseIO(), [94.0, None], duration=0.1)


def test_source_calibration():
    src = source_calibration(ir([0.0, 1.0], label="s1"))
    assert src.metadata.extra == {"label": "s1", "role": SOURCE_ROLE, "distance_m": 1.0}
    assert role(src) == SOURCE_ROLE
    assert source_calibration(ir([1.0]), 2.0).metadata.extra["distance_m"] == 2.0
    with pytest.raises(ValueError, match=r"unit 'Pa/FS'\), got '1'"):
        source_calibration(ir([1.0], "1"))
    with pytest.raises(ValueError, match="distance must be > 0"):
        source_calibration(ir([1.0]), 0.0)


def test_noise_band_levels_reads_time_signals_and_old_files():
    # noise files written before resonus-core 0.11.0 are ImpulseResponse with the role
    t = np.arange(int(FS)) / FS
    x = np.sin(2 * np.pi * 1000 * t)
    old = noise(x)
    new = TimeSignal(x, FS, "Pa", old.metadata)
    np.testing.assert_array_equal(noise_band_levels(new), noise_band_levels(old))


def test_noise_band_levels():
    t = np.arange(int(2 * FS)) / FS
    levels = noise_band_levels(noise(np.sin(2 * np.pi * 1000 * t)))
    assert levels.shape == (7,)
    assert levels[3] == pytest.approx(90.97, abs=0.1)  # 1 Pa amplitude: 20 lg(1/sqrt2/P0)
    assert levels[0] < levels[3] - 40
    with pytest.raises(ValueError, match="not a noise recording"):
        noise_band_levels(ir([1.0]))


def test_noise_only():
    rng = np.random.default_rng(3)
    n = noise(0.02 * rng.standard_normal(int(9 * FS)))
    ln = noise_band_levels(n)
    r = sti_at_position(ir([1.0]), source_calibration(ir([1.0])), n, ln, seed=4)
    assert isinstance(r, StiReport)
    assert r.sti == pytest.approx(0.5, abs=0.01)
    assert r.m.shape == (7, 14) and r.mti.shape == (7,)
    np.testing.assert_allclose(r.speech_levels_db, ln)
    np.testing.assert_allclose(r.noise_levels_db, ln)
    assert r.seed == 4
    assert r.sti == pytest.approx(
        float(np.dot([0.129, 0.143, 0.114, 0.114, 0.186, 0.171, 0.143], r.mti))
    )


def test_reverberation_without_noise():
    rng = np.random.default_rng(5)
    t = np.arange(int(1.5 * FS)) / FS
    h = rng.standard_normal(t.size) * np.exp(-3 * np.log(10) * t)  # T = 1 s
    room = ir(1e-3 * h)
    n = noise(np.zeros(int((1.5 + STI_WINDOW_S) * FS) + 1))
    r = sti_at_position(room, source_calibration(ir([2e-3])), n, np.full(7, 60.0), seed=6)
    expected = sti_from_mtf(np.tile(mtf_reverberation(MODULATION_FREQUENCIES, 1.0), (7, 1))).sti
    assert r.sti == pytest.approx(expected, abs=0.025)
    assert np.all(r.noise_levels_db == -np.inf)


def test_errors_in_order():
    src = source_calibration(ir([1.0]))
    n = noise(np.zeros(int(9 * FS)))
    lv = np.full(7, 60.0)
    with pytest.raises(ValueError, match="must be calibrated"):
        sti_at_position(ir([1.0], "1"), src, n, lv)
    with pytest.raises(ValueError, match="must be calibrated"):
        sti_at_position(src, src, n, lv)  # a source calibration is not an in-situ IR
    with pytest.raises(ValueError, match="not a source calibration"):
        sti_at_position(ir([1.0]), ir([1.0]), n, lv)
    with pytest.raises(ValueError, match="not a noise recording"):
        sti_at_position(ir([1.0]), src, ir([1.0]), lv)
    other = dataclasses.replace(n, sample_rate=48000.0)
    with pytest.raises(ValueError, match="sample rates differ"):
        sti_at_position(ir([1.0]), src, other, lv)
    with pytest.raises(ValueError, match="speech_levels_db must have 7 values"):
        sti_at_position(ir([1.0]), src, n, lv[:6])
    short = noise(np.zeros(int(4 * FS)))
    with pytest.raises(ValueError, match=r"noise recording too short: 4.0 s, need 8.0 s"):
        sti_at_position(ir([1.0]), src, short, lv)
