"""Room-side level calibration (D-31): profiles, offsets, capture, Pa."""

from datetime import UTC, datetime

import numpy as np
import pytest
from resonus_core.calibration import (
    P0,
    CalibrationError,
    CalibrationRecord,
    CalibrationStore,
    InputProfile,
)
from resonus_core.data import ImpulseResponse, Metadata
from resonus_core.devices import DeviceInfo

from resonus_room.core.calibrate import (
    CALIBRATED_IR_UNIT,
    OFFSET_KEY,
    SIMULATED_DEVICE_ID,
    calibrated_ir,
    calibrated_recording,
    capture_tone,
    input_offsets,
    input_profiles,
    store_path,
)
from resonus_room.core.settings import AppSettings

FS = 48000
DEV = DeviceInfo(0, "Scarlett 2i2", "ALSA", 2, 2, 48000.0)
OTHER = DeviceInfo(1, "Scarlett 2i2", "JACK", 2, 2, 48000.0)
T0 = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)


def settings(**kw):
    base = dict(
        input_device="ALSA: Scarlett 2i2",
        input_channels=(1, 2),
        input_labels=("m1", "m2"),
        sample_rate=FS,
        simulate=False,
    )
    base.update(kw)
    return AppSettings(**base)


def test_store_path(tmp_path, monkeypatch):
    assert store_path(tmp_path) == tmp_path / "resonus-room" / "calibrations.toml"
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert store_path() == tmp_path / "xdg" / "resonus-room" / "calibrations.toml"
    monkeypatch.setenv("XDG_DATA_HOME", "")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    expected = tmp_path / "home" / ".local" / "share" / "resonus-room" / "calibrations.toml"
    assert store_path() == expected
    assert not expected.exists()


def test_input_profiles():
    s = settings(reference_channel=0)
    profiles = input_profiles(s, [OTHER, DEV])
    assert profiles == [
        InputProfile("ALSA: Scarlett 2i2", "Scarlett 2i2", 0, FS),
        InputProfile("ALSA: Scarlett 2i2", "Scarlett 2i2", 1, FS),
    ]
    one = input_profiles(settings(input_channels=(2,), input_labels=("m1",)), [DEV])
    assert [p.channel_index for p in one] == [1]


def test_input_profiles_simulated_and_errors():
    sim = input_profiles(settings(simulate=True, input_device=""), [])
    assert [(p.device_id, p.channel_index) for p in sim] == [
        (SIMULATED_DEVICE_ID, 0),
        (SIMULATED_DEVICE_ID, 1),
    ]
    with pytest.raises(ValueError, match="choose the input device in Preferences"):
        input_profiles(settings(input_device=""), [DEV])
    with pytest.raises(ValueError, match="input device 'ALSA: Scarlett 2i2' not found"):
        input_profiles(settings(), [OTHER])


def test_input_offsets():
    s = settings()
    store = CalibrationStore.unsaved()
    assert input_offsets(s, [DEV], store) == [None, None]
    p1 = input_profiles(s, [DEV])[1]
    store.append(CalibrationRecord.manual(p1, 120.5, T0))
    store.append(CalibrationRecord.manual(p1, 121.0, T0))
    assert input_offsets(s, [DEV], store) == [None, 121.0]


class ToneIO:
    """Records a tone on the second input, silence-free noise on the first."""

    sample_rate = float(FS)
    n_inputs = 2
    n_outputs = 1

    def __init__(self, amplitude):
        self.amplitude = amplitude
        self.requested = None

    def rec(self, n):
        self.requested = n
        t = np.arange(n) / FS
        x = np.zeros((n, 2))
        x[:, 0] = 1e-3 * np.random.default_rng(0).standard_normal(n)
        x[:, 1] = self.amplitude * np.sin(2 * np.pi * 1000.0 * t)
        return x


def test_capture_tone():
    io = ToneIO(0.1)
    p = capture_tone(settings(), io, 1, 1000.0, 94.0, duration=4.0)
    assert io.requested == 4 * FS
    # L_tone = 20 lg(0.1 / sqrt 2) = -23.01 dB re FS -> C = 94 + 23.01
    assert p.measured_level_dbfs == pytest.approx(-23.010, abs=0.01)
    assert p.offset_db == pytest.approx(117.010, abs=0.01)
    p = capture_tone(settings(), ToneIO(0.1), 1, 1000.0, 94.0, 0.2, -0.1, duration=4.0)
    assert p.offset_db == pytest.approx(117.110, abs=0.01)
    with pytest.raises(CalibrationError):
        capture_tone(settings(), ToneIO(1.0), 1, 1000.0, 94.0, duration=4.0)  # clipped
    with pytest.raises(ValueError, match="input_index out of range"):
        capture_tone(settings(), ToneIO(0.1), 2, 1000.0, 94.0)


def test_calibrated_ir():
    meta = Metadata("resonus-room", "0", extra={"label": "s1m1"})
    ir = ImpulseResponse(np.array([0.0, 0.5, -0.25]), 48000.0, "1", meta)
    out = calibrated_ir(ir, 100.0)
    np.testing.assert_allclose(out.data, ir.data * P0 * 10**5)
    assert out.unit == CALIBRATED_IR_UNIT == "Pa/FS"
    assert out.sample_rate == 48000.0
    assert out.metadata.extra == {"label": "s1m1", OFFSET_KEY: 100.0}
    assert ir.metadata.extra == {"label": "s1m1"}  # not modified
    with pytest.raises(ValueError, match="impulse response unit 'Pa/FS'"):
        calibrated_ir(out, 100.0)


def test_calibrated_recording():
    rec = np.ones((4, 2))
    out = calibrated_recording(rec, [94.0, 100.0], ["m1", "m2"])
    assert out.shape == (4, 2) and out.dtype == np.float64
    np.testing.assert_allclose(out[:, 0], P0 * 10 ** (94 / 20))
    np.testing.assert_allclose(out[:, 1], P0 * 10**5)
    with pytest.raises(ValueError, match="input m2 is not calibrated"):
        calibrated_recording(rec, [94.0, None], ["m1", "m2"])
    with pytest.raises(ValueError, match="one offset and one label per column"):
        calibrated_recording(rec, [94.0], ["m1"])
    with pytest.raises(ValueError, match="recording must be 2-D"):
        calibrated_recording(np.ones(4), [94.0], ["m1"])
