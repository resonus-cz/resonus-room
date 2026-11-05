import csv
import io
import json

import numpy as np
from resonus_core import data
from resonus_core.bands import band_set
from scipy.io import wavfile

from resonus_room.cli import main
from resonus_room.core.settings import AppSettings
from resonus_room.core.simulate import exponential_decay_ir


def small_settings(tmp_path):
    path = tmp_path / "s.toml"
    AppSettings(duration=0.5, tail=1.0, ir_length=1.0, sim_decay_time=0.4).save(path)
    return path


def test_measure_simulated(tmp_path, capsys):
    out = tmp_path / "run"
    s = small_settings(tmp_path)
    assert main(["measure", "--settings", str(s), "--simulate", "--out", str(out), "--json"]) == 0
    for name in (
        "impulse_response.h5",
        "frequency_response.h5",
        "settings.toml",
        "analysis.json",
        "analysis.csv",
        "reverberation.png",
        "transfer_function.png",
        "impulse_response.png",
    ):
        assert (out / name).exists(), name
    summary = json.loads(capsys.readouterr().out)
    assert "simulated" in summary["device"].lower()
    assert summary["bands_hz"] == [125, 250, 500, 1000, 2000, 4000]
    # 500 Hz and up: scatter of a 0.4 s decay is small there (ISO 3382-2 Eq. A.4)
    assert all(abs(t - 0.4) < 0.08 for t in summary["T20"][2:])
    full = json.loads((out / "analysis.json").read_text())
    assert full["software"] == "resonus-room" and "device" in full
    assert AppSettings.load(out / "settings.toml").duration == 0.5


def test_analyze_wav_and_h5(tmp_path, capsys):
    bs = band_set(3, 100, 5000)
    ir = exponential_decay_ir(16000, 1.5, np.full(18, 0.5), bs, delay=0.01, seed=3)
    wavfile.write(tmp_path / "ir.wav", 16000, (0.5 * ir / np.abs(ir).max()).astype(np.float32))
    assert main(["analyze", str(tmp_path / "ir.wav"), "--bands", "third"]) == 0
    rows = list(csv.reader(io.StringIO(capsys.readouterr().out)))
    assert rows[0][0] == "band_hz" and len(rows) == 19
    out = tmp_path / "a"
    assert main(["analyze", str(tmp_path / "ir.wav"), "--out", str(out), "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["bands"]["nominal_hz"] == [125, 250, 500, 1000, 2000, 4000]
    assert d["source"].endswith("ir.wav")
    assert (out / "analysis.csv").exists() and (out / "reverberation.png").exists()
    run = tmp_path / "run"
    main(["measure", "--settings", str(small_settings(tmp_path)), "--out", str(run)])
    capsys.readouterr()
    assert main(["analyze", str(run / "impulse_response.h5"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["sample_rate"] == 48000


def test_absorption(tmp_path, capsys):
    bs = band_set(3, 100, 5000)
    for kind, t60 in (("e", 1.2), ("s", 0.8)):
        for k in range(2):
            ir = exponential_decay_ir(16000, 3.0, np.full(18, t60), bs, delay=0.01, seed=k)
            wavfile.write(
                tmp_path / f"{kind}{k}.wav", 16000, (0.5 * ir / np.abs(ir).max()).astype(np.float32)
            )
    (tmp_path / "setup.toml").write_text(
        "volume = 200.0\nspecimen_area = 10.8\n"
        '[empty]\ntemperature = 20.0\nrelative_humidity = 50.0\nfiles = ["e0.wav", "e1.wav"]\n'
        '[specimen]\ntemperature = 20.0\nrelative_humidity = 50.0\nfiles = ["s0.wav", "s1.wav"]\n'
    )
    out = tmp_path / "out"
    assert main(["absorption", str(tmp_path / "setup.toml"), "--out", str(out), "--json"]) == 0
    captured = capsys.readouterr()
    d = json.loads(captured.out)
    assert d["standards"] == ["ISO 354:2003"] and len(d["alpha"]) == 18
    assert "warning: 6.1.4" in captured.err  # A1 of a 1.2 s room exceeds Table 1
    for name in ("absorption.json", "absorption.csv", "absorption.png"):
        assert (out / name).exists()
    br = data.load(out / "absorption.h5")
    assert isinstance(br, data.BandResults) and br.metadata.software == "resonus-room"
    np.testing.assert_allclose(br.values["alpha_s"], d["alpha"], atol=0.005)
    assert main(["absorption", str(tmp_path / "setup.toml")]) == 0
    assert capsys.readouterr().out.startswith("band_hz,")


def test_settings_and_plot(tmp_path):
    path = tmp_path / "s.toml"
    assert main(["settings", str(path)]) == 0
    assert AppSettings.load(path) == AppSettings()
    run = tmp_path / "run"
    assert main(["measure", "--settings", str(small_settings(tmp_path)), "--out", str(run)]) == 0
    for name in ("frequency_response.h5", "impulse_response.h5"):
        img = tmp_path / f"{name}.svg"
        assert main(["plot", str(run / name), "-o", str(img)]) == 0
        assert img.stat().st_size > 1000
    assert data.load(run / "impulse_response.h5").unit == "1"


def test_errors_give_exit_code_1(tmp_path, capsys):
    bad = tmp_path / "bad.toml"
    bad.write_text("level = 5.0\n")
    assert main(["measure", "--settings", str(bad), "--out", str(tmp_path / "x")]) == 1
    assert "maximum" in capsys.readouterr().err
    assert main(["plot", str(tmp_path / "missing.h5"), "-o", str(tmp_path / "p.png")]) == 1
    assert main(["analyze", str(tmp_path / "missing.wav")]) == 1
    assert main(["absorption", str(tmp_path / "missing.toml")]) == 1


def test_measure_length_and_averages(tmp_path):
    # excitation length and number of averages are chosen per measurement (D-11)
    out = tmp_path / "run"
    args = ["measure", "--settings", str(small_settings(tmp_path)), "--simulate"]
    assert main(args + ["--duration", "0.4", "--repetitions", "2", "--out", str(out)]) == 0
    saved = AppSettings.load(out / "settings.toml")
    assert saved.duration == 0.4 and saved.repetitions == 2
    assert main(args + ["--repetitions", "0", "--out", str(tmp_path / "bad")]) == 1


def test_analysis_h5_written(tmp_path):
    out = tmp_path / "run"
    args = ["measure", "--settings", str(small_settings(tmp_path)), "--simulate", "--out", str(out)]
    assert main(args) == 0
    br = data.load(out / "analysis.h5")
    assert isinstance(br, data.BandResults) and br.units["T20"] == "s"
    assert br.metadata.device != "" and br.metadata.software == "resonus-room"


def test_plot_band_results(tmp_path):
    run = tmp_path / "run"
    assert main(["measure", "--settings", str(small_settings(tmp_path)), "--out", str(run)]) == 0
    h5 = str(run / "analysis.h5")
    img = tmp_path / "rt.png"
    assert main(["plot", h5, "-o", str(img)]) == 0 and img.stat().st_size > 1000
    # several quantities with the same unit, several files overlaid
    img2 = tmp_path / "rt2.svg"
    args = ["plot", h5, h5, "--quantity", "EDT", "--quantity", "T30", "-o", str(img2)]
    assert main(args) == 0 and img2.stat().st_size > 1000
    # different units, unknown quantity, several IR files: errors
    assert main(["plot", h5, "--quantity", "T20", "--quantity", "C80", "-o", str(img)]) == 1
    assert main(["plot", h5, "--quantity", "XYZ", "-o", str(img)]) == 1
    ir = str(run / "impulse_response.h5")
    assert main(["plot", ir, ir, "-o", str(img)]) == 1


CHANNEL_FILES = (
    "impulse_response.h5",
    "frequency_response.h5",
    "analysis.json",
    "analysis.csv",
    "analysis.h5",
    "reverberation.png",
    "transfer_function.png",
    "impulse_response.png",
)


def test_measure_two_channels_one_folder_each(tmp_path, capsys):
    # one subfolder per channel, named by the s#m# label (D-17)
    path = tmp_path / "s.toml"
    AppSettings(
        duration=0.5,
        tail=1.0,
        ir_length=1.0,
        sim_decay_time=0.4,
        input_channels=(1, 2),
        input_labels=("m1L", "m1R"),
    ).save(path)
    out = tmp_path / "bh"
    assert (
        main(["measure", "--settings", str(path), "--simulate", "--out", str(out), "--json"]) == 0
    )
    assert (out / "settings.toml").exists() and not (out / "impulse_response.h5").exists()
    for label in ("s1m1L", "s1m1R"):
        for name in CHANNEL_FILES:
            assert (out / label / name).exists(), (label, name)
        ir = data.load(out / label / "impulse_response.h5")
        assert ir.metadata.extra["channel"]["label"] == label
    summary = json.loads(capsys.readouterr().out)
    assert sorted(summary["channels"]) == ["s1m1L", "s1m1R"]
    assert summary["channels"]["s1m1R"]["bands_hz"] == [125, 250, 500, 1000, 2000, 4000]
    assert "simulated" in summary["device"].lower()


def test_measure_inputs_option(tmp_path):
    base = ["measure", "--settings", str(small_settings(tmp_path)), "--simulate"]
    out = tmp_path / "a"
    assert main(base + ["--inputs", "1, 2", "--out", str(out)]) == 0
    assert (out / "s1m1" / "analysis.csv").exists() and (out / "s1m2" / "analysis.csv").exists()
    saved = AppSettings.load(out / "settings.toml")
    assert saved.input_channels == (1, 2) and saved.input_labels == ("m1", "m2")
    out = tmp_path / "b"
    labels = ["--input-labels", "m3L,m3R"]
    assert main(base + ["--inputs", "1,2", *labels, "--out", str(out)]) == 0
    assert (out / "s1m3L").is_dir() and (out / "s1m3R").is_dir()


def test_measure_inputs_errors(tmp_path, capsys):
    base = ["measure", "--settings", str(small_settings(tmp_path)), "--simulate"]
    assert main(base + ["--inputs", "1,x", "--out", str(tmp_path / "e1")]) == 1
    assert "not a channel number: 'x'" in capsys.readouterr().err
    assert (
        main(base + ["--inputs", "1,2", "--input-labels", "m1", "--out", str(tmp_path / "e2")]) == 1
    )
    assert main(base + ["--input-labels", "mic", "--out", str(tmp_path / "e3")]) == 1
    assert "input label 'mic'" in capsys.readouterr().err
    out = tmp_path / "ok"
    assert main(base + ["--input-labels", "m4", "--out", str(out)]) == 0  # one channel
    assert (out / "impulse_response.h5").exists()  # single-channel layout kept
    assert AppSettings.load(out / "settings.toml").input_labels == ("m4",)


def test_measure_warnings_on_stderr(tmp_path, capsys):
    path = tmp_path / "s.toml"
    AppSettings(duration=0.5, tail=1.0, ir_length=1.5, sim_decay_time=0.4).save(path)
    assert (
        main(["measure", "--settings", str(path), "--simulate", "--out", str(tmp_path / "w")]) == 0
    )
    assert "warning: IR length 1.5 s exceeds the silence after the sweep" in capsys.readouterr().err


def test_measure_inputs_empty_items_rejected(tmp_path, capsys):
    base = ["measure", "--settings", str(small_settings(tmp_path)), "--simulate"]
    assert main(base + ["--inputs", "1,,2", "--out", str(tmp_path / "e1")]) == 1
    assert "not a channel number: ''" in capsys.readouterr().err
    assert (
        main(base + ["--inputs", "1,2", "--input-labels", "m1,", "--out", str(tmp_path / "e2")])
        == 1
    )
    assert "empty" in capsys.readouterr().err


def test_plot_mark_in_exported_images(tmp_path):
    # D-24: the plot mark comes from the settings (default "Re:sonus", empty: none)
    run = tmp_path / "run"
    assert main(["measure", "--settings", str(small_settings(tmp_path)), "--out", str(run)]) == 0
    ir = str(run / "impulse_response.h5")
    default = tmp_path / "default.svg"
    assert main(["plot", ir, "-o", str(default)]) == 0
    assert "Re:sonus" in default.read_text()
    for mark, name in (("Acme Acoustics", "custom"), ("", "none")):
        s = tmp_path / f"{name}.toml"
        AppSettings(plot_mark=mark).save(s)
        img = tmp_path / f"{name}.svg"
        assert main(["plot", ir, "-o", str(img), "--settings", str(s)]) == 0
        svg = img.read_text()
        assert "Re:sonus" not in svg and (mark in svg if mark else True)


def test_plot_mark_default_and_roundtrip(tmp_path):
    assert AppSettings().plot_mark == "Re:sonus"
    path = tmp_path / "s.toml"
    AppSettings(plot_mark="Ing. Novák").save(path)
    assert AppSettings.load(path).plot_mark == "Ing. Novák"
