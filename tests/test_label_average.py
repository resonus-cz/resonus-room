"""Room average from channel labels and measure --out folders (D-19)."""

import numpy as np
import pytest
from resonus_core import data
from resonus_core.bands import band_set
from resonus_core.data import ImpulseResponse, Metadata

from resonus_room.cli import main
from resonus_room.core.engine import analyze_ir, settings_band_set
from resonus_room.core.positions import (
    CHANNEL_FILE,
    FILE_POSITION,
    FILE_SUFFIXES,
    POSITION_LABEL,
    Position,
    PositionSetup,
    label_from_filename,
    label_position,
    positions_from_folders,
)
from resonus_room.core.room_average import average_analyses, average_positions
from resonus_room.core.settings import AppSettings
from resonus_room.core.simulate import exponential_decay_ir

FS = 16000.0
T = np.array([0.6, 0.55, 0.5, 0.45, 0.4, 0.35])
OCT = band_set(1, 125, 4000)


@pytest.mark.parametrize(
    ("label", "expected"),
    [("s1m1", ("s1", "m1")), ("s2m13", ("s2", "m13")), ("s10m2", ("s10", "m2"))],
)
def test_label_position(label, expected):
    assert label_position(label) == expected


@pytest.mark.parametrize(
    "label", ["s1m1L", "s1m1R", "s1+s2m1", "m1", "s1", "ir.wav", "s0m1", "S1M1", " s1m1"]
)
def test_label_not_a_position(label):
    with pytest.raises(ValueError, match="is not one source and one microphone position"):
        label_position(label)
    assert POSITION_LABEL.fullmatch(label) is None


def run_folder(folder, labels, seed0=0):
    for k, label in enumerate(labels):
        ir = exponential_decay_ir(FS, 1.5, T, OCT, delay=0.01, noise_rms=1e-5, seed=seed0 + k)
        meta = Metadata("t", "0", extra={"channel": {"label": label}})
        (folder / label).mkdir(parents=True)
        data.save(folder / label / CHANNEL_FILE, ImpulseResponse(ir, FS, "1", meta))
    return folder


def test_positions_from_folders(tmp_path):
    a = run_folder(tmp_path / "run1", ["s1m2", "s1m1", "s1m3"])
    b = run_folder(tmp_path / "run2", ["s2m1"], seed0=10)
    (a / "settings.toml").write_text("")  # files are ignored
    (a / "notes").mkdir()  # subfolders without CHANNEL_FILE are ignored
    setup = positions_from_folders([a, b], "engineering")
    assert isinstance(setup, PositionSetup) and setup.method == "engineering"
    assert setup.positions == (
        Position(a / "s1m1" / CHANNEL_FILE, "s1", "m1"),
        Position(a / "s1m2" / CHANNEL_FILE, "s1", "m2"),
        Position(a / "s1m3" / CHANNEL_FILE, "s1", "m3"),
        Position(b / "s2m1" / CHANNEL_FILE, "s2", "m1"),
    )
    assert setup.volume is None and setup.reverberation_time is None
    assert CHANNEL_FILE == "impulse_response.h5"


def test_positions_from_folders_errors(tmp_path):
    a = run_folder(tmp_path / "bh", ["s1m1L", "s1m1R"])
    ok = run_folder(tmp_path / "ok", ["s1m1"])
    msg = r"^not one source and one microphone position \(s#m#\): 's1m1L', 's1m1R', 's2m1R'$"
    c = run_folder(tmp_path / "bh2", ["s2m1R"])
    with pytest.raises(ValueError, match=msg):
        positions_from_folders([a, ok, c], "survey")  # all folders in one message
    with pytest.raises(ValueError, match="unknown method 'fast'"):
        positions_from_folders([a], "fast")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no channel folders with impulse_response.h5"):
        positions_from_folders([empty], "survey")


def test_average_analyses_matches_average_positions(tmp_path):
    a = run_folder(tmp_path / "run1", ["s1m1", "s1m2", "s1m3"])
    setup = positions_from_folders([a], "survey")
    settings = AppSettings()
    ref = average_positions(setup, settings)
    analyses = [analyze_ir(data.load(p.file), settings) for p in setup.positions]
    avg = average_analyses(setup, analyses, settings_band_set(settings))
    for name in ref.mean:
        np.testing.assert_array_equal(avg.mean[name], ref.mean[name])
    np.testing.assert_array_equal(avg.sigma["T20"], ref.sigma["T20"])
    assert avg.issues == ref.issues == ()  # survey: 1 source, 3 microphones, octaves
    with pytest.raises(ValueError, match="one analysis per position"):
        average_analyses(setup, analyses[:2], settings_band_set(settings))


def test_cli_average_folders(tmp_path, capsys):
    a = run_folder(tmp_path / "run1", ["s1m1", "s1m2", "s1m3"])
    b = run_folder(tmp_path / "run2", ["s2m1", "s2m2", "s2m3"], seed0=10)
    out = tmp_path / "room"
    args = ["average", str(a), str(b), "--method", "engineering", "--out", str(out)]
    assert main(args) == 0
    assert "warning" not in capsys.readouterr().err  # 6 combinations, 2 sources, 3 mics
    for name in ("average.csv", "average.h5", "average.png"):
        assert (out / name).exists()
    assert data.load(out / "average.h5").metadata.description == "run1, run2"
    assert main(["average", str(a), "--method", "engineering"]) == 0
    io = capsys.readouterr()
    assert io.out.startswith("band_hz,N,")
    assert "warning: ISO 3382-2 Table 1: 1 source positions, engineering requires 2" in io.err


def test_cli_average_argument_errors(tmp_path, capsys):
    a = run_folder(tmp_path / "run1", ["s1m1"])
    assert main(["average", str(a)]) == 1  # folders need --method
    assert "--method is required" in capsys.readouterr().err
    setup = tmp_path / "positions.toml"
    setup.write_text('method = "survey"\n')
    assert main(["average", str(setup), "--method", "survey"]) == 1
    assert "--method applies to measurement folders only" in capsys.readouterr().err
    assert main(["average", str(setup), str(a)]) == 1
    assert "one setup file or measurement folders" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (" s1m 1.wav", "s1m1"),  # REW export names
        ("s2m 6.wav", "s2m6"),
        ("S2M3.WAV", "s2m3"),
        ("s1 m12.ir.h5", "s1m12"),
        ("s3m4.h5", "s3m4"),
        ("s1m1", "s1m1"),
        ("s10m2.Ir.H5", "s10m2"),
    ],
)
def test_label_from_filename(name, expected):
    assert label_from_filename(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "room_s1m1.wav",
        "s1m1L.wav",
        "s0m1.wav",
        "s1m0.wav",
        "s1m1_take2.wav",
        "ir.wav",
        "s1m1.txt",
        "s1.wav",
        "m1.wav",
        "",
    ],
)
def test_label_from_filename_none(name):
    assert label_from_filename(name) is None


def test_file_suffixes_longest_first():
    assert FILE_SUFFIXES == (".ir.h5", ".h5", ".wav")
    assert FILE_POSITION.fullmatch("s1 m 2")
