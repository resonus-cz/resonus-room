import csv
import io
from pathlib import Path

import numpy as np
import pytest
from resonus_core import data
from resonus_core.bands import band_set
from resonus_core.data import BandResults, Metadata
from scipy.io import wavfile

from resonus_room.cli import main
from resonus_room.core.positions import load_positions
from resonus_room.core.room_average import (
    AVERAGE_COLUMNS,
    AVERAGED,
    STANDARDS,
    RoomAverage,
    average_positions,
    average_to_band_results,
    average_to_csv,
)
from resonus_room.core.settings import AppSettings
from resonus_room.core.simulate import exponential_decay_ir
from resonus_room.core.uncertainty import bandwidth, reverberation_std

FS = 16000
T = np.array([0.6, 0.55, 0.5, 0.45, 0.4, 0.35])
OCT = band_set(1, 125, 4000)
REAL = Path(__file__).parents[1] / "refs" / "rew-test-data" / "resonus-room-test-data" / "test1"


def write_setup(folder, method="engineering", n_src=2, n_mic=3):
    lines = [f'method = "{method}"']
    k = 0
    for s in range(n_src):
        for m in range(n_mic):
            ir = exponential_decay_ir(FS, 1.5, T, OCT, delay=0.01, noise_rms=1e-5, seed=k)
            name = f"s{s}m{m}.wav"
            wavfile.write(folder / name, FS, (0.5 * ir / np.abs(ir).max()).astype(np.float32))
            lines += [
                "[[position]]",
                f'file = "{name}"',
                f'source = "S{s}"',
                f'microphone = "M{m}"',
            ]
            k += 1
    (folder / "positions.toml").write_text("\n".join(lines) + "\n")
    return folder / "positions.toml"


@pytest.fixture(scope="module")
def average(tmp_path_factory):
    folder = tmp_path_factory.mktemp("room")
    return average_positions(load_positions(write_setup(folder)), AppSettings())


def test_average(average):
    a = average
    assert isinstance(a, RoomAverage) and a.bands == OCT and len(a.analyses) == 6
    assert set(a.mean) == set(AVERAGED) == set(a.std) == set(a.count)
    t20 = np.array([x.values("T20") for x in a.analyses])
    np.testing.assert_allclose(a.mean["T20"], t20.mean(axis=0))
    np.testing.assert_allclose(a.std["T20"], t20.std(axis=0, ddof=1))
    assert a.count["T30"].tolist() == [6] * 6
    bw = bandwidth(OCT.center, 1)
    # set T recovered within 3 standard deviations of the mean, Eq. (A.4) with N = 6
    assert np.all(np.abs(a.mean["T20"] - T) <= 3 * reverberation_std(T, bw, 6, 20))
    np.testing.assert_allclose(a.sigma["T30"], reverberation_std(a.mean["T30"], bw, 6, 30))
    assert a.issues == ()  # engineering: 6 combinations, 2 sources, 3 microphones, 125-4k


def test_issues_for_precision(tmp_path):
    a = average_positions(
        load_positions(write_setup(tmp_path, "precision", 1, 2)), AppSettings(bands="octave")
    )
    assert a.issues == (
        "ISO 3382-2 Table 1: 2 source-microphone combinations, precision requires 12",
        "ISO 3382-2 Table 1: 1 source positions, precision requires 2",
        "ISO 3382-2 Table 1: 2 microphone positions, precision requires 3",
    )


def test_band_results_and_csv(average):
    br = average_to_band_results(average, Metadata("resonus-room", "0", description="room"))
    assert isinstance(br, BandResults) and br.labels == tuple(OCT.labels)
    keys = [k for n in AVERAGED for k in (n, f"{n}_std", f"{n}_n")] + ["T20_sigma", "T30_sigma"]
    assert list(br.values) == keys
    assert br.units["C80_std"] == "dB" and br.units["T20_n"] == "1" and br.units["T30_sigma"] == "s"
    np.testing.assert_array_equal(br.values["T30_n"], [6.0] * 6)
    assert br.metadata.standards == STANDARDS and br.metadata.extra["method"] == "engineering"
    assert br.metadata.extra["positions"][0] == {
        "file": "s0m0.wav",
        "source": "S0",
        "microphone": "M0",
    }
    rows = list(csv.reader(io.StringIO(average_to_csv(average))))
    assert tuple(rows[0]) == AVERAGE_COLUMNS and len(rows) == 7
    r = dict(zip(AVERAGE_COLUMNS, rows[1], strict=True))
    assert r["band_hz"] == "125" and r["N"] == "6"
    assert r["T20_s"] == f"{average.mean['T20'][0]:.3f}"
    assert r["C80_dB"] == f"{average.mean['C80'][0]:.2f}"


def test_cli_average(tmp_path, capsys):
    setup = write_setup(tmp_path, "precision", 1, 3)
    out = tmp_path / "out"
    assert main(["average", str(setup), "--out", str(out)]) == 0
    assert (
        "warning: ISO 3382-2 Table 1: 3 source-microphone combinations" in capsys.readouterr().err
    )
    for name in ("average.csv", "average.h5", "average.png"):
        assert (out / name).exists()
    assert isinstance(data.load(out / "average.h5"), BandResults)
    assert main(["average", str(setup), "--bands", "third"]) == 0
    text = capsys.readouterr().out
    assert text.startswith("band_hz,N,") and len(text.splitlines()) == 1 + 18
    assert main(["average", str(tmp_path / "missing.toml")]) == 1


@pytest.mark.skipif(not REAL.exists(), reason="REW sample data (refs/) not available")
def test_real_box_precision(tmp_path):
    # One box of the REW set (file names without a leading space; the user moves the two
    # boxes between box1/ and box2/): 2 sources x 6 microphones = 12 combinations.
    box = next(d for d in sorted(REAL.glob("box*")) if (d / "s1m 1.wav").exists())
    lines = ['method = "precision"']
    for s in (1, 2):
        for m in range(1, 7):
            wav = box / f"s{s}m {m}.wav"
            lines += ["[[position]]", f'file = "{wav}"']
            lines += [f'source = "S{s}"', f'microphone = "M{m}"']
    (tmp_path / "p.toml").write_text("\n".join(lines) + "\n")
    a = average_positions(load_positions(tmp_path / "p.toml"), AppSettings(bands="third"))
    assert a.issues == () and a.count["T20"][0] == 12
