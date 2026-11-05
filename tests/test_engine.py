from dataclasses import fields

import numpy as np
import pytest
from resonus_core.audio import SimulatedIO
from resonus_core.data import ImpulseResponse, Metadata
from resonus_core.measure import MeasurementResult, MeasurementSettings

from resonus_room import __version__
from resonus_room.core.analysis import RoomAnalysis
from resonus_room.core.engine import (
    analyze_ir,
    band_impulse_responses,
    measure,
    open_device,
    settings_band_set,
    simulated_room,
)
from resonus_room.core.settings import AppSettings


def test_settings_defaults_and_relations(tmp_path):
    s = AppSettings()
    assert isinstance(s, MeasurementSettings)
    assert (s.bands, s.f_min, s.f_max, s.filter_order) == ("octave", 100.0, 5000.0, 3)
    assert (s.tail, s.ir_length, s.decay_method, s.reverse_filter) == (3.0, 3.0, "correct", False)
    assert s.simulate is True
    with pytest.raises(ValueError, match="f_max"):
        AppSettings(f_min=1000.0, f_max=500.0)
    with pytest.raises(ValueError):
        AppSettings(bands="half")
    with pytest.raises(ValueError):
        AppSettings(decay_method="lundeby")
    s.save(tmp_path / "s.toml")
    assert AppSettings.load(tmp_path / "s.toml") == s


def test_settings_band_set():
    # ISO 3382-2:2008 5.1: 125 Hz-4 kHz octaves or 100 Hz-5 kHz one-third octaves.
    oct_ = settings_band_set(AppSettings())
    np.testing.assert_array_equal(oct_.nominal, [125, 250, 500, 1000, 2000, 4000])
    third = settings_band_set(AppSettings(bands="third"))
    assert len(third) == 18 and third.nominal[0] == 100 and third.nominal[-1] == 5000


def test_simulated_room():
    s = AppSettings(sim_decay_time=0.8)
    h = simulated_room(s)
    assert h.shape == (round((2 * 0.8 + 0.5) * 48000),)
    assert np.all(h[:240] == 0) and np.any(h[240:250] != 0)  # 5 ms delay
    np.testing.assert_array_equal(h, simulated_room(s))  # deterministic
    # unit energy (user decision D-21): white noise keeps its RMS level through the room
    assert np.sum(h**2) == pytest.approx(1.0, rel=1e-12)


def test_open_device():
    dev = open_device(AppSettings())
    assert isinstance(dev, SimulatedIO) and dev.sample_rate == 48000
    assert "simulated" in dev.describe().lower()


def test_measure_and_analyze_simulated_room():
    s = AppSettings(duration=1.0, sim_decay_time=1.0)
    r = measure(s, open_device(s))
    assert isinstance(r, MeasurementResult) and len(r.impulse_responses) == 1
    ir = r.impulse_responses[0]
    assert ir.metadata.software == "resonus-room" and ir.metadata.software_version == __version__
    assert ir.data.size == 3 * 48000
    a = analyze_ir(ir, s)
    assert isinstance(a, RoomAnalysis) and len(a.band_results) == 6
    # The simulated room has T = 1 s in every band: T20/T30 within 3 sigma (ISO 3382-2 A.4).
    assert np.all(np.abs(a.values("T20") - 1.0) < 3 * a.t20_std)
    assert np.all(np.abs(a.values("T30") - 1.0) < 3 * a.t30_std)
    assert np.all(a.values("PNR") > 60)
    assert a.settings == {"filter_order": 3, "reverse": False, "decay_method": "correct"}
    a3 = analyze_ir(ir, s.replace(bands="third", reverse_filter=True, filter_order=2))
    assert len(a3.band_results) == 18 and a3.settings["reverse"] is True


def test_sample_rate_mismatch():
    s = AppSettings()
    with pytest.raises(ValueError):
        measure(s, open_device(AppSettings(sample_rate=44100)))


def test_band_impulse_responses_match_analysis_filtering():
    from resonus_core.bands import filter_bands

    from resonus_room.core.simulate import exponential_decay_ir

    settings = AppSettings(bands="octave", f_min=125.0, f_max=4000.0)
    bands = settings_band_set(settings)
    x = exponential_decay_ir(48000, 1.0, [0.5] * len(bands), bands, delay=0.01, seed=4)
    x = np.concatenate([x, np.zeros(4800)])  # trailing zeros as in REW exports
    ir = ImpulseResponse(x, 48000.0, "1", Metadata("t", "0"))
    a = analyze_ir(ir, settings)
    yb = band_impulse_responses(ir, a, settings)
    trimmed = x[: np.flatnonzero(x)[-1] + 1]
    assert yb.shape == (trimmed.size - a.onset, len(bands))
    expected = filter_bands(trimmed[a.onset :], bands, 48000.0, 3, False)
    np.testing.assert_allclose(yb, expected)


def test_simulated_device_has_the_channel_layout():
    # resonus-core 0.5.0: io inputs = input_channels, then the reference (D-17)
    s = AppSettings(
        input_channels=(1, 2),
        input_labels=("m1L", "m1R"),
        output_channels=(1, 2),
        output_labels=("s1", "s2"),
        output_mode="sequential",
    )
    dev = open_device(s)
    assert isinstance(dev, SimulatedIO) and (dev.n_inputs, dev.n_outputs) == (2, 2)
    dev = open_device(s.replace(output_channels=(1,), output_labels=("s1",), reference_channel=3))
    assert (dev.n_inputs, dev.n_outputs) == (3, 1)
    groups = {f.name: f.metadata["group"] for f in fields(AppSettings)}
    assert groups["simulate"] == "Simulation"


def test_simulated_room_seed():
    s = AppSettings(sim_decay_time=0.3)
    a, b = simulated_room(s), simulated_room(s, seed=1)
    assert a.shape == b.shape and not np.allclose(a, b)
    np.testing.assert_array_equal(a, simulated_room(s, seed=0))


def test_measure_simulated_two_by_two_sequential():
    s = AppSettings(
        duration=0.5,
        tail=1.0,
        ir_length=1.0,
        sim_decay_time=0.5,
        input_channels=(1, 2),
        input_labels=("m1", "m2"),
        output_channels=(1, 2),
        output_labels=("s1", "s2"),
        output_mode="sequential",
    )
    r = measure(s, open_device(s))
    labels = [ir.metadata.extra["channel"]["label"] for ir in r.impulse_responses]
    assert labels == ["s1m1", "s1m2", "s2m1", "s2m2"]
    data = [ir.data for ir in r.impulse_responses]
    assert not np.allclose(data[0], data[1]) and not np.allclose(data[0], data[2])
    for ir in r.impulse_responses:  # every path is a room with T = 0.5 s
        a = analyze_ir(ir, s)  # within 3 sigma (ISO 3382-2 A.4), as for one channel
        assert np.all(np.abs(a.values("T20") - 0.5) < 3 * a.t20_std)


def test_measure_simulated_with_reference():
    s = AppSettings(duration=0.5, tail=1.0, ir_length=1.0, sim_decay_time=0.3, reference_channel=2)
    r = measure(s, open_device(s))
    assert len(r.impulse_responses) == 1
    # the loopback is a wire: the room measured through it equals the room measured
    # without it (other deconvolution: regularized with the recorded reference)
    ir = r.impulse_responses[0].data
    plain = s.replace(reference_channel=0)
    ir0 = measure(plain, open_device(plain)).impulse_responses[0].data
    assert np.argmax(np.abs(ir)) == np.argmax(np.abs(ir0))
    assert np.dot(ir, ir0) / np.sqrt(np.dot(ir, ir) * np.dot(ir0, ir0)) > 0.99


def test_simulated_reference_is_the_last_io_input():
    # reference on device channel 4, two microphones: io inputs m1, m2, reference
    s = AppSettings(input_channels=(1, 2), input_labels=("m1", "m2"), reference_channel=4)
    dev = open_device(s)
    assert dev.n_inputs == 3
    out = dev.playrec(np.r_[1.0, np.zeros(99)])
    np.testing.assert_allclose(out[:, 2], np.r_[1.0, np.zeros(99)], atol=1e-3)  # the wire
    assert not np.allclose(out[:, 0], out[:, 2], atol=1e-3)
