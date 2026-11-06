"""ISO 354 from two folders (D-32).

Synthetic reverberation room as in the notebook (section 4): A1 = 6 m2, a 10.8 m2
specimen with a known alpha_s profile, T from ISO 354:2003 Eq. (5) at 20 degC (c = 343
m/s), 3 microphones x 2 sources. Private reference over three seed sets
(2026-11-06): |alpha - alpha_true| at most 0.047, at most 0.98 of (4 reported standard
uncertainties + 0.02); test: 4 standard uncertainties + 0.04.
"""

import dataclasses
import tomllib

import numpy as np
import pytest
from resonus_core import data
from resonus_core.air import power_attenuation_coefficient
from resonus_core.bands import band_set
from resonus_core.data import FrequencyResponse, ImpulseResponse, Metadata
from scipy.io import wavfile

from resonus_room.core.absorption import Climate
from resonus_room.core.iso354 import load_setup, run_absorption
from resonus_room.core.iso354_folders import (
    EMPTY_NAME,
    SPECIMEN_NAME,
    FolderPositions,
    folder_positions,
    position_counts,
    setup_from_folders,
    setup_to_toml,
)
from resonus_room.core.settings import AppSettings
from resonus_room.core.simulate import exponential_decay_ir

FS = 16000
META = Metadata("test", "0")
CLIMATE = Climate(20.0, 50.0)
VOLUME, AREA = 200.0, 10.8
BANDS = band_set(3, 100, 5000)
ALPHA_TRUE = np.linspace(0.1, 0.9, 18)


def t_of(a):
    m = power_attenuation_coefficient(BANDS.center, 20.0, 50.0)
    return 55.3 * VOLUME / (343.0 * (a + 4 * VOLUME * m))


def write_wav(path, t60=None, seed=0):
    t60 = np.full(18, 0.5) if t60 is None else t60
    h = exponential_decay_ir(FS, 2 * t60.max() + 0.5, t60, BANDS, delay=0.01, seed=seed)
    wavfile.write(path, FS, (0.5 * h / np.abs(h).max()).astype(np.float32))


def ir(n=64, **extra):
    x = np.zeros(n)
    x[0] = 1.0
    return ImpulseResponse(x, float(FS), "1", dataclasses.replace(META, extra=extra))


def test_top_level_wav_and_h5(tmp_path):
    write_wav(tmp_path / " s1m 2.wav")
    write_wav(tmp_path / "S2M1.WAV")
    data.save(tmp_path / "impulse_response.h5", ir())
    data.save(tmp_path / "source_calibration.h5", ir(role="source-calibration"))
    data.save(
        tmp_path / "frequency_response.h5",
        FrequencyResponse(np.array([1.0, 2.0, 3.0]), np.ones(3, complex), "1", META),
    )
    (tmp_path / "notes.txt").write_text("x")
    write_wav(tmp_path / ".hidden.wav")
    got = folder_positions(tmp_path)
    assert isinstance(got, FolderPositions)
    assert got.files == (
        tmp_path / " s1m 2.wav",
        tmp_path / "S2M1.WAV",
        tmp_path / "impulse_response.h5",
    )
    assert got.labels == ("s1m2", "s2m1", None)


def test_channel_subfolders(tmp_path):
    for name in ("s2m1", "s1m1", "left"):
        (tmp_path / name).mkdir()
        data.save(tmp_path / name / "impulse_response.h5", ir())
    (tmp_path / "empty_sub").mkdir()
    write_wav(tmp_path / "s3m1.wav")
    got = folder_positions(str(tmp_path))
    assert got.files == (
        tmp_path / "s3m1.wav",
        tmp_path / "left" / "impulse_response.h5",
        tmp_path / "s1m1" / "impulse_response.h5",
        tmp_path / "s2m1" / "impulse_response.h5",
    )
    assert got.labels == ("s3m1", None, "s1m1", "s2m1")


def test_folder_errors(tmp_path):
    with pytest.raises(ValueError, match="not a folder"):
        folder_positions(tmp_path / "missing")
    (tmp_path / "a.txt").write_text("x")
    with pytest.raises(ValueError, match=r"no impulse responses \(\.wav, \.h5 or"):
        folder_positions(tmp_path)
    write_wav(tmp_path / " s1m 1.wav")
    write_wav(tmp_path / "s1m 1.wav")
    write_wav(tmp_path / "s1m2.wav")
    write_wav(tmp_path / "S1M2.wav")
    with pytest.raises(
        ValueError,
        match=r"several files for position\(s\) s1m1, s1m2: one file per position "
        r"\(put different rooms in separate folders\)",
    ):
        folder_positions(tmp_path)


def test_broken_h5(tmp_path):
    (tmp_path / "bad.h5").write_bytes(b"not hdf5")
    with pytest.raises(ValueError, match=r"bad\.h5: "):
        folder_positions(tmp_path)


def test_position_counts():
    assert position_counts(["s1m1", "s1m2", "s2m1", "s2m3"]) == (3, 2)
    assert position_counts(["s1m1"]) == (1, 1)
    assert position_counts([]) is None
    assert position_counts(["s1m1", None]) is None


def make_room(tmp_path, n_mic=6, n_src=2, specimen_mics=None):
    folders = {}
    for kind, t60 in (("empty", t_of(6.0)), ("specimen", t_of(6.0 + ALPHA_TRUE * AREA))):
        folder = tmp_path / kind
        folder.mkdir()
        mics = n_mic if kind == "empty" or specimen_mics is None else specimen_mics
        k = 0
        for s in range(1, n_src + 1):
            for m in range(1, mics + 1):
                write_wav(folder / f"s{s}m{m}.wav", t60, seed=10 * k + len(kind))
                k += 1
        folders[kind] = folder
    return folders["empty"], folders["specimen"]


def test_setup_and_evaluation(tmp_path):
    empty, specimen = make_room(tmp_path, n_mic=3, n_src=2)
    setup, notes = setup_from_folders(
        empty, specimen, VOLUME, CLIMATE, Climate(21.0, 45.0, 100.0), specimen_area=AREA
    )
    assert notes == []
    assert setup.volume == VOLUME
    assert setup.empty_files == folder_positions(empty).files
    assert setup.specimen_files == folder_positions(specimen).files
    assert setup.empty_climate == CLIMATE
    assert setup.specimen_climate == Climate(21.0, 45.0, 100.0)
    assert (setup.specimen_area, setup.n_objects, setup.max_length) == (AREA, None, None)
    assert (setup.microphones, setup.sources) == (3, 2)
    setup = dataclasses.replace(setup, specimen_climate=CLIMATE)
    run = run_absorption(setup, AppSettings())
    res = run.result
    assert run.t1_positions.shape == (6, 18)
    assert np.all(np.abs(res.alpha - ALPHA_TRUE) <= 4 * res.alpha_std + 0.04)
    assert any("7.1.4" in issue for issue in res.issues)  # 6 decays < 12


def test_setup_notes_when_counts_differ_or_missing(tmp_path):
    empty, specimen = make_room(tmp_path, n_mic=3, n_src=2, specimen_mics=6)
    setup, notes = setup_from_folders(empty, specimen, VOLUME, CLIMATE, CLIMATE, n_objects=4)
    assert (setup.microphones, setup.sources) == (None, None)
    assert (setup.specimen_area, setup.n_objects) == (None, 4)
    assert notes == [
        f"{EMPTY_NAME}: 7.1.4: 6 decay curves < 12",
        f"positions differ between the conditions (only {EMPTY_NAME}: -; "
        f"only {SPECIMEN_NAME}: s1m4, s1m5, s1m6, s2m4, s2m5, s2m6)",
    ]
    write_wav(specimen / "extra.wav", t_of(6.0))
    _, notes = setup_from_folders(empty, specimen, VOLUME, CLIMATE, CLIMATE, n_objects=4)
    assert notes == [
        f"{EMPTY_NAME}: 7.1.4: 6 decay curves < 12",
        f"{SPECIMEN_NAME}: positions not checked (ISO 354:2003 7.1.4): 1 of 13 file(s) "
        "without an s#m# label",
    ]


def test_setup_needs_one_specimen_size(tmp_path):
    with pytest.raises(ValueError, match="give exactly one of specimen_area and n_objects"):
        setup_from_folders(tmp_path / "a", tmp_path / "b", VOLUME, CLIMATE, CLIMATE)
    with pytest.raises(ValueError, match="give exactly one of specimen_area and n_objects"):
        setup_from_folders(
            tmp_path / "a", tmp_path / "b", VOLUME, CLIMATE, CLIMATE, specimen_area=1.0, n_objects=2
        )


def test_setup_to_toml_round_trip(tmp_path):
    empty, specimen = make_room(tmp_path, n_mic=1, n_src=1)
    setup, _ = setup_from_folders(
        empty,
        specimen,
        210.5,
        Climate(20.1, 50.2, 101.3),
        Climate(19.9, 48.0),
        specimen_area=10.8,
        max_length=10.25,
    )
    out = tmp_path / "eval" / "setup.toml"
    out.parent.mkdir()
    setup_to_toml(setup, out)
    text = out.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert '"../empty/s1m1.wav"' in text
    cfg = tomllib.loads(text)
    assert list(cfg) == [
        "volume",
        "specimen_area",
        "max_length",
        "microphones",
        "sources",
        "empty",
        "specimen",
    ]
    assert list(cfg["empty"]) == ["temperature", "relative_humidity", "pressure", "files"]
    back = load_setup(out)
    resolved = [f.resolve() for f in back.empty_files + back.specimen_files]
    assert resolved == [f.resolve() for f in setup.empty_files + setup.specimen_files]
    assert dataclasses.replace(back, empty_files=(), specimen_files=()) == dataclasses.replace(
        setup, empty_files=(), specimen_files=()
    )


def test_setup_to_toml_objects_and_quotes(tmp_path):
    folder = tmp_path / 'room "a"'
    folder.mkdir()
    write_wav(folder / " s1m 1.wav")
    setup, _ = setup_from_folders(folder, folder, 100.0, CLIMATE, CLIMATE, n_objects=3)
    setup_to_toml(setup, tmp_path / "s.toml")
    back = load_setup(tmp_path / "s.toml")
    assert back.n_objects == 3 and back.specimen_area is None
    assert back.empty_files == (tmp_path / 'room "a"' / " s1m 1.wav",)


def test_matching_labels_and_differing_positions(tmp_path):
    # usual case: the same s#m# labels in both folders -> one check, no notes
    empty, specimen = make_room(tmp_path, n_mic=6, n_src=2)
    setup, notes = setup_from_folders(empty, specimen, VOLUME, CLIMATE, CLIMATE, specimen_area=AREA)
    assert notes == [] and (setup.microphones, setup.sources) == (6, 2)
    # same counts, other positions
    (specimen / "s2m6.wav").rename(specimen / "s3m1.wav")
    setup, notes = setup_from_folders(empty, specimen, VOLUME, CLIMATE, CLIMATE, specimen_area=AREA)
    assert (setup.microphones, setup.sources) == (None, None)
    assert notes == [
        f"positions differ between the conditions (only {EMPTY_NAME}: s2m6; "
        f"only {SPECIMEN_NAME}: s3m1)"
    ]
    (specimen / "s3m1.wav").unlink()
    _, notes = setup_from_folders(empty, specimen, VOLUME, CLIMATE, CLIMATE, specimen_area=AREA)
    assert notes == [
        f"positions differ between the conditions (only {EMPTY_NAME}: s2m6; "
        f"only {SPECIMEN_NAME}: -)"
    ]
