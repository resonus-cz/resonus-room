"""IACC export (CSV, BandResults) and the CLI command binaural."""

import numpy as np
import pytest
from resonus_core import data
from resonus_core.bands import band_set
from resonus_core.data import BandResults, Metadata
from scipy.io import wavfile

from resonus_room.cli import main
from resonus_room.core.binaural import EARLY_LIMIT, TAU_MAX, BinauralParameters
from resonus_room.core.files import (
    BINAURAL_COLUMNS,
    BINAURAL_QUANTITIES,
    BINAURAL_STANDARDS,
    NOTES,
    binaural_to_band_results,
    binaural_to_csv,
)
from resonus_room.core.simulate import exponential_decay_ir

FS = 48000.0
BANDS = band_set(1, 125, 4000)


def params():
    n = len(BANDS)
    return BinauralParameters(
        early=np.linspace(0.9, 0.4, n),
        late=np.full(n, 0.1234),
        full=np.r_[np.nan, np.full(n - 1, 0.5)],
        onset=480,
    )


def test_binaural_csv():
    text = binaural_to_csv(params(), BANDS)
    lines = text.splitlines()
    assert lines[0] == ",".join(BINAURAL_COLUMNS)
    assert BINAURAL_COLUMNS == ("band_hz", "center_hz", "IACC_early", "IACC_late", "IACC_full")
    assert len(lines) == 1 + len(BANDS)
    assert lines[1] == f"125,{BANDS.center[0]:.2f},0.900,0.123,"  # NaN: empty field
    assert lines[-1].startswith("4000,") and lines[-1].endswith(",0.400,0.123,0.500")
    with pytest.raises(ValueError, match="one IACC value per band"):
        binaural_to_csv(params(), band_set(1, 125, 2000))


def test_binaural_band_results():
    meta = Metadata("resonus-room", "1.0", description="L / R")
    r = binaural_to_band_results(params(), BANDS, FS, meta)
    assert isinstance(r, BandResults) and r.fraction == 1
    assert list(r.values) == list(BINAURAL_QUANTITIES) == ["IACC_early", "IACC_late", "IACC_full"]
    assert r.units == {"IACC_early": "1", "IACC_late": "1", "IACC_full": "1"}
    np.testing.assert_array_equal(r.values["IACC_late"], np.full(len(BANDS), 0.1234))
    assert r.labels == tuple(BANDS.labels)
    assert r.metadata.standards == BINAURAL_STANDARDS
    assert r.metadata.description == "L / R"
    extra = r.metadata.extra
    assert extra["notes"] == list(NOTES) and extra["sample_rate"] == FS
    assert extra["onset_s"] == pytest.approx(480 / FS)
    assert extra["early_limit_s"] == EARLY_LIMIT and extra["tau_max_s"] == TAU_MAX
    with pytest.raises(ValueError, match="one IACC value per band"):
        binaural_to_band_results(params(), band_set(1, 250, 4000), FS, meta)


def ears(tmp_path, seed_right):
    bands = band_set(1, 125, 4000)
    t = np.full(len(bands), 0.6)
    left = exponential_decay_ir(FS, 1.0, t, bands, delay=0.01, seed=1)
    right = exponential_decay_ir(FS, 1.0, t, bands, delay=0.01, seed=seed_right)
    stereo = np.stack([left, right], axis=1)
    path = tmp_path / "brir.wav"
    wavfile.write(path, int(FS), (0.5 * stereo / np.abs(stereo).max()).astype(np.float32))
    return path


def test_cli_binaural_identical_ears(tmp_path, capsys):
    path = ears(tmp_path, seed_right=1)  # both channels the same response
    args = ["binaural", str(path), str(path), "--right-channel", "1"]
    assert main(args) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == ",".join(BINAURAL_COLUMNS) and len(lines) == 7
    for line in lines[1:]:
        assert line.split(",")[2:] == ["1.000", "1.000", "1.000"]


def test_cli_binaural_files_and_independent_ears(tmp_path):
    path = ears(tmp_path, seed_right=2)
    out = tmp_path / "iacc"
    args = ["binaural", str(path), str(path), "--right-channel", "1", "--out", str(out)]
    assert main(args) == 0
    assert (out / "binaural.csv").read_text().startswith("band_hz,center_hz,IACC_early")
    r = data.load(out / "binaural.h5")
    assert isinstance(r, BandResults) and r.metadata.description == "brir.wav / brir.wav"
    assert np.all(r.values["IACC_late"][2:] < 0.2)  # independent diffuse tails, 500 Hz up


def test_cli_binaural_sample_rate_mismatch(tmp_path, capsys):
    path = ears(tmp_path, seed_right=1)
    other = tmp_path / "r44.wav"
    _fs, x = wavfile.read(path)
    wavfile.write(other, 44100, x[:, 1])
    assert main(["binaural", str(path), str(other)]) == 1
    assert "sample rates differ" in capsys.readouterr().err
