import numpy as np
import pytest
from resonus_core.air import power_attenuation_coefficient
from resonus_core.bands import band_set
from scipy.io import wavfile

from resonus_room.core.iso354 import (
    AbsorptionRun,
    AbsorptionSetup,
    load_setup,
    position_t20,
    run_absorption,
)
from resonus_room.core.settings import AppSettings
from resonus_room.core.simulate import exponential_decay_ir

FS = 16000  # one-third octaves up to 5 kHz (upper edge 5.6 kHz < 8 kHz)
V = 200.0
S_AREA = 10.8
BANDS = band_set(3, 100, 5000)
ALPHA = np.linspace(0.1, 0.9, 18)
A1 = 6.0
SPEED = 331 + 0.6 * 20.0


def t_for(area: np.ndarray) -> np.ndarray:
    # ISO 354 Eq. (5) solved for T at 20 degC, 50 % RH, 101.325 kPa.
    m = power_attenuation_coefficient(BANDS.center, 20.0, 50.0)
    return 55.3 * V / (SPEED * (area + 4 * V * m))


@pytest.fixture(scope="module")
def setup_file(tmp_path_factory):
    d = tmp_path_factory.mktemp("iso354")
    t1 = t_for(np.full(18, A1))
    t2 = t_for(A1 + ALPHA * S_AREA)
    names = {"empty": [], "specimen": []}
    for kind, t60 in (("empty", t1), ("specimen", t2)):
        (d / kind).mkdir()
        for k in range(4):
            ir = exponential_decay_ir(
                FS, 2 * t60.max() + 0.5, t60, BANDS, delay=0.01, seed=10 * k + len(kind)
            )
            name = f"{kind}/p{k}.wav"
            wavfile.write(d / name, FS, (0.5 * ir / np.max(np.abs(ir))).astype(np.float32))
            names[kind].append(name)
    text = f"""volume = {V}
specimen_area = {S_AREA}
max_length = 10.0
microphones = 2
sources = 2
[empty]
temperature = 20.0
relative_humidity = 50.0
files = {names["empty"]!r}
[specimen]
temperature = 20.0
relative_humidity = 50.0
pressure = 101.325
files = {names["specimen"]!r}
""".replace("'", '"')
    (d / "setup.toml").write_text(text)
    return d / "setup.toml"


def test_load_setup(setup_file):
    s = load_setup(setup_file)
    assert isinstance(s, AbsorptionSetup)
    assert s.volume == V and s.specimen_area == S_AREA and s.n_objects is None
    assert len(s.empty_files) == 4 and s.empty_files[0] == setup_file.parent / "empty/p0.wav"
    assert s.empty_climate.pressure == 101.325 and s.specimen_climate.temperature == 20.0
    assert (s.max_length, s.microphones, s.sources) == (10.0, 2, 2)


def test_load_setup_errors(tmp_path):
    base = '[empty]\ntemperature = 20.0\nrelative_humidity = 50.0\nfiles = ["a.wav"]\n'
    spec = base.replace("[empty]", "[specimen]")
    (tmp_path / "a.toml").write_text("volume = 200.0\n" + base + spec)
    with pytest.raises(ValueError, match="specimen_area"):
        load_setup(tmp_path / "a.toml")  # neither area nor objects
    (tmp_path / "b.toml").write_text("specimen_area = 10.0\n" + base + spec)
    with pytest.raises(ValueError, match="volume"):
        load_setup(tmp_path / "b.toml")
    (tmp_path / "c.toml").write_text(
        "volume = 200.0\nn_objects = 3\n" + base + spec.replace('"a.wav"', "")
    )
    with pytest.raises(ValueError, match="empty file list"):
        load_setup(tmp_path / "c.toml")
    with pytest.raises(OSError):
        load_setup(tmp_path / "missing.toml")


def test_run_recovers_alpha(setup_file):
    run = run_absorption(load_setup(setup_file), AppSettings())
    assert isinstance(run, AbsorptionRun)
    assert run.t1_positions.shape == run.t2_positions.shape == (4, 18)
    np.testing.assert_allclose(run.result.t1, run.t1_positions.mean(axis=0))
    r = run.result
    # Each band within 4 reported standard uncertainties (floor 0.02), no overall bias.
    err = r.alpha - ALPHA
    assert np.all(np.abs(err) < 4 * np.maximum(r.alpha_std, 0.02)), err
    assert abs(np.mean(err)) < 0.03
    np.testing.assert_allclose(r.a1, A1, rtol=0.1)
    # max_length 10 m < 1.9 * 200**(1/3) = 11.1 m: no 6.1.2 issue; 2 x 2 positions < 12
    assert "7.1.4: 4 decay curves < 12" in r.issues
    assert "7.1.4: 2 microphone positions < 3" in r.issues
    assert not any(i.startswith("6.1.2") for i in r.issues)


def test_position_t20_uses_iso354_bands(setup_file):
    s = load_setup(setup_file)
    t = position_t20(s.empty_files[:1], AppSettings(bands="octave", f_min=500.0))
    assert t.shape == (1, 18)
