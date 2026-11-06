"""CLI of the calibrated STI chain (D-31): calibrate, record-noise, sti."""

import json

import numpy as np
import pytest
from resonus_core import data
from resonus_core.calibration import P0
from resonus_core.data import ImpulseResponse, Metadata, TimeSignal

from resonus_room import cli_sti
from resonus_room.cli import main
from resonus_room.cli_sti import speech_levels
from resonus_room.core.calibrate import CALIBRATED_IR_UNIT, OFFSET_KEY
from resonus_room.core.settings import AppSettings
from resonus_room.core.sti import vocal_effort_levels
from resonus_room.core.sti_inputs import NOISE_ROLE, SOURCE_ROLE, role

FS = 32000  # STI files
RATE = 48000  # settings and the fake device


@pytest.fixture
def setup(tmp_path):
    s = tmp_path / "s.toml"
    AppSettings(duration=0.5, tail=0.5, ir_length=0.5, sim_decay_time=0.3, sample_rate=RATE).save(s)
    return s, tmp_path / "cal.toml"


class ToneIO:
    sample_rate = float(RATE)
    n_inputs = 1
    n_outputs = 1

    def __init__(self, amplitude=0.1, tone=True):
        self.amplitude, self.tone = amplitude, tone

    def rec(self, n):
        t = np.arange(n) / RATE
        if self.tone:
            return (self.amplitude * np.sin(2 * np.pi * 1000.0 * t))[:, None]
        return np.full((n, 1), self.amplitude)

    def describe(self):
        return "test device"


def test_speech_levels():
    s = AppSettings()
    np.testing.assert_allclose(
        speech_levels("60, 61,62,63,58,52,47", s), [60, 61, 62, 63, 58, 52, 47]
    )
    own = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    s7 = AppSettings(sti_vocal_effort="custom", sti_speech_levels=own)
    np.testing.assert_allclose(speech_levels(None, s7), [1, 2, 3, 4, 5, 6, 7])
    assert speech_levels(None, s7).dtype == np.float64
    # default: the vocal-effort spectrum "normal" (Q-STI3, D-34)
    np.testing.assert_allclose(speech_levels(None, s), vocal_effort_levels("normal"))
    with pytest.raises(ValueError, match="sti_vocal_effort = custom"):
        speech_levels(None, s.replace(sti_vocal_effort="custom"))
    with pytest.raises(ValueError, match=r"--speech-levels: not a number: 'x'"):
        speech_levels("60,x", s)
    with pytest.raises(ValueError, match="give 7 values"):
        speech_levels("60,61", s)


def test_settings_speech_levels_need_seven():
    with pytest.raises(ValueError, match="sti_speech_levels"):
        AppSettings(sti_speech_levels=(60.0, 61.0))
    assert AppSettings().sti_speech_levels == ()
    assert AppSettings().show_sti is False


def test_calibrate_show_and_manual(setup, capsys):
    s, store = setup
    base = ["calibrate", "--settings", str(s), "--store", str(store)]
    assert main([*base, "--show"]) == 0
    assert capsys.readouterr().out == "m1: not calibrated\n"
    assert main([*base, "--input", "m1", "--manual", "120"]) == 0
    out = capsys.readouterr().out
    assert out == "m1: offset 120.00 dB\nnot saved: run again with --save to keep it\n"
    assert not store.exists()
    assert main([*base, "--input", "m1", "--manual", "120.5", "--save", "--note", "n"]) == 0
    assert capsys.readouterr().out == f"m1: offset 120.50 dB\nsaved to {store}\n"
    assert main([*base, "--show"]) == 0
    assert capsys.readouterr().out == "m1: 120.50 dB\n"
    assert main([*base, "--show", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"m1": 120.5}


def test_calibrate_with_tone(setup, capsys, monkeypatch):
    s, store = setup
    monkeypatch.setattr("resonus_room.core.engine.open_device", lambda settings: ToneIO())
    args = ["calibrate", "--settings", str(s), "--store", str(store), "--input", "m1"]
    assert main([*args, "--level", "94", "--duration", "4", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["input"] == "m1" and d["saved"] is False
    assert d["offset_db"] == pytest.approx(117.01, abs=0.01)  # 94 - 20 lg(0.1 / sqrt 2)
    assert main([*args, "--level", "94", "--duration", "4", "--save"]) == 0
    capsys.readouterr()
    assert main(["calibrate", "--settings", str(s), "--store", str(store), "--show"]) == 0
    assert capsys.readouterr().out == "m1: 117.01 dB\n"


def test_calibrate_errors(setup, capsys, monkeypatch):
    s, store = setup
    base = ["calibrate", "--settings", str(s), "--store", str(store)]
    assert main(base) == 1
    assert "--input is required" in capsys.readouterr().err
    assert main([*base, "--input", "m7", "--manual", "1"]) == 1
    assert "no input 'm7' in the settings" in capsys.readouterr().err
    assert main([*base, "--input", "m1"]) == 1
    assert "--level is required (or --manual)" in capsys.readouterr().err
    monkeypatch.setattr(
        "resonus_room.core.engine.open_device", lambda settings: ToneIO(1.0)
    )  # clipped
    assert main([*base, "--input", "m1", "--level", "94", "--duration", "2"]) == 1
    assert "error:" in capsys.readouterr().err


def test_record_noise(setup, capsys, monkeypatch, tmp_path):
    s, store = setup
    out = tmp_path / "noise"
    monkeypatch.setattr(
        "resonus_room.core.engine.open_device", lambda settings: ToneIO(0.1, tone=False)
    )
    args = ["record-noise", "--settings", str(s), "--store", str(store), "--out", str(out)]
    assert main([*args, "--duration", "1"]) == 1
    assert "input m1 is not calibrated" in capsys.readouterr().err
    main(["calibrate", "--settings", str(s), "--store", str(store), "--input", "m1",
          "--manual", "100", "--save"])  # fmt: skip
    capsys.readouterr()
    assert main([*args, "--duration", "1"]) == 0
    printed = capsys.readouterr().out
    n = data.load(out / "noise_m1.h5")
    assert isinstance(n, TimeSignal)  # resonus-core 0.11.0 (Q-C9)
    assert role(n) == NOISE_ROLE and n.unit == "Pa" and n.data.size == RATE
    np.testing.assert_allclose(n.data, 0.1 * P0 * 10**5)
    csv = (out / "noise.csv").read_text(encoding="utf-8").splitlines()
    assert csv[0] == "input,125,250,500,1000,2000,4000,8000"
    assert csv[1].startswith("m1,") and len(csv[1].split(",")) == 8
    assert printed.splitlines() == csv
    assert main([*args, "--duration", "1", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert list(d) == ["m1"] and len(d["m1"]) == 7


def test_measure_calibrated(setup, capsys, tmp_path):
    s, store = setup
    out = tmp_path / "run"
    cmd = ["measure", "--settings", str(s), "--simulate", "--out", str(out)]
    assert main([*cmd, "--calibrated", "--store", str(store)]) == 1
    assert (
        "input m1 is not calibrated: run resonus-room calibrate --input m1"
        in capsys.readouterr().err
    )
    main(["calibrate", "--settings", str(s), "--store", str(store), "--input", "m1",
          "--manual", "100", "--save"])  # fmt: skip
    capsys.readouterr()
    assert main([*cmd, "--calibrated", "--store", str(store), "--source-calibration"]) == 0
    ir = data.load(out / "impulse_response.h5")
    assert ir.unit == CALIBRATED_IR_UNIT and ir.metadata.extra[OFFSET_KEY] == 100.0
    assert role(ir) == ""
    src = data.load(out / "source_calibration.h5")
    assert role(src) == SOURCE_ROLE
    np.testing.assert_array_equal(src.data, ir.data)
    assert (out / "analysis.json").exists()
    with pytest.raises(SystemExit):
        main([*cmd, "--source-calibration"])  # needs --calibrated (argparse error)
    capsys.readouterr()


def _files(tmp_path):
    meta = Metadata("test", "0")
    room = ImpulseResponse(np.array([1e-3]), float(FS), "Pa/FS", meta)
    src = ImpulseResponse(
        np.array([1e-3]), float(FS), "Pa/FS", Metadata("t", "0", extra={"role": SOURCE_ROLE})
    )
    rng = np.random.default_rng(1)
    noise = ImpulseResponse(
        0.01 * rng.standard_normal(9 * FS),
        float(FS),
        "Pa",
        Metadata("t", "0", extra={"role": NOISE_ROLE}),
    )
    paths = []
    for name, obj in (("room", room), ("src", src), ("noise", noise)):
        p = tmp_path / f"{name}.h5"
        data.save(p, obj)
        paths.append(str(p))
    return paths


def test_sti_command(tmp_path, capsys):
    room, src, noise = _files(tmp_path)
    base = ["sti", room, "--source", src, "--noise", noise]
    levels = ["--speech-levels", "80,80,80,80,80,80,80"]
    # default settings: the "normal" vocal-effort spectrum (Q-STI3, D-34)
    assert main(base) == 0
    capsys.readouterr()
    assert main([*base, *levels]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "band_hz,MTI,speech_level_1m_dB,noise_level_dB" and lines[-1].startswith(
        "STI,"
    )
    assert float(lines[-1].split(",")[1]) > 0.9  # about 30 dB S/N: nearly 1
    assert main([*base, *levels, "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["STI"] > 0.9 and d["method"].startswith("PROVISIONAL")
    out = tmp_path / "sti"
    assert main([*base, *levels, "--out", str(out)]) == 0
    assert capsys.readouterr().out == ""
    for name in ("sti.json", "sti.csv", "sti.h5", "sti.png"):
        assert (out / name).exists(), name
    b = data.load(out / "sti.h5")
    assert b.metadata.description == "room" and b.metadata.extra["STI"] == pytest.approx(d["STI"])
    assert main(["sti", noise, "--source", src, "--noise", noise, *levels]) == 1
    assert "must be calibrated" in capsys.readouterr().err


def test_sti_command_rejects_other_files(tmp_path, capsys):
    room, src, noise = _files(tmp_path)
    from resonus_core.data import BandResults

    b = BandResults(
        1, np.array([1000.0]), ("1k",), {"x": np.array([1.0])}, {"x": "1"}, Metadata("t", "0")
    )
    p = tmp_path / "b.h5"
    data.save(p, b)
    assert (
        main(["sti", str(p), "--source", src, "--noise", noise, "--speech-levels", "1,1,1,1,1,1,1"])
        == 1
    )
    assert f"{p}: not an impulse response file" in capsys.readouterr().err


def test_devices_for_simulated():
    assert cli_sti.devices_for(AppSettings(simulate=True)) == []
