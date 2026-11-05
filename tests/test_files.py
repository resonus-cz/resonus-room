import csv
import io
import json

import numpy as np
import pytest
from resonus_core import data
from resonus_core.bands import band_set
from resonus_core.data import FrequencyResponse, ImpulseResponse, Metadata
from scipy.io import wavfile

from resonus_room import __version__
from resonus_room.core.absorption import Climate, absorption
from resonus_room.core.analysis import analyze
from resonus_room.core.files import (
    ABSORPTION_COLUMNS,
    ABSORPTION_QUANTITIES,
    ANALYSIS_QUANTITIES,
    CSV_COLUMNS,
    NOTES,
    absorption_to_band_results,
    absorption_to_csv,
    absorption_to_dict,
    analysis_to_band_results,
    analysis_to_csv,
    analysis_to_dict,
    load_impulse_response,
)
from resonus_room.core.simulate import exponential_decay_ir

FS = 48000.0
OCT = band_set(1, 250, 2000)


@pytest.fixture(scope="module")
def result():
    ir = exponential_decay_ir(FS, 2.0, [1.2, 1.1, 1.0, 0.9], OCT, delay=0.005, seed=2)
    return analyze(ir, FS, OCT)


def test_load_wav_int16_and_multichannel(tmp_path):
    x = np.array([[0, 16384], [-32768, 8192], [32767, 0]], dtype=np.int16)
    wavfile.write(tmp_path / "a.wav", 44100, x)
    ir = load_impulse_response(tmp_path / "a.wav", channel=1)
    assert isinstance(ir, ImpulseResponse) and ir.sample_rate == 44100.0 and ir.unit == "1"
    np.testing.assert_allclose(ir.data, [0.5, 0.25, 0.0])
    np.testing.assert_allclose(
        load_impulse_response(tmp_path / "a.wav").data, [0, -1, 32767 / 32768]
    )
    assert ir.metadata.software == "resonus-room" and ir.metadata.description == "a.wav"
    with pytest.raises(ValueError):
        load_impulse_response(tmp_path / "a.wav", channel=2)


def test_load_wav_float_int32_uint8(tmp_path):
    wavfile.write(tmp_path / "f.wav", 48000, np.array([0.25, -0.5], dtype=np.float32))
    np.testing.assert_allclose(load_impulse_response(tmp_path / "f.wav").data, [0.25, -0.5])
    wavfile.write(tmp_path / "i.wav", 48000, np.array([2**30, -(2**31)], dtype=np.int32))
    np.testing.assert_allclose(load_impulse_response(tmp_path / "i.wav").data, [0.5, -1.0])
    wavfile.write(tmp_path / "u.wav", 8000, np.array([128, 192, 0], dtype=np.uint8))
    np.testing.assert_allclose(load_impulse_response(tmp_path / "u.wav").data, [0, 0.5, -1])
    with pytest.raises(ValueError):
        load_impulse_response(tmp_path / "f.wav", channel=1)


def test_load_h5(tmp_path):
    meta = Metadata("x", "1")
    data.save(tmp_path / "ir.h5", ImpulseResponse(np.array([1.0, 0.5]), 48000.0, "Pa", meta))
    ir = load_impulse_response(tmp_path / "ir.h5")
    np.testing.assert_allclose(ir.data, [1.0, 0.5])
    assert ir.unit == "Pa"
    fr = FrequencyResponse(np.array([1.0, 2.0]), np.array([1, 1j]), "1", meta)
    data.save(tmp_path / "fr.h5", fr)
    with pytest.raises(ValueError):
        load_impulse_response(tmp_path / "fr.h5")
    with pytest.raises(ValueError):
        load_impulse_response(tmp_path / "x.flac")
    with pytest.raises(OSError):
        load_impulse_response(tmp_path / "missing.wav")


def test_analysis_to_dict_is_json(result):
    d = analysis_to_dict(result)
    text = json.dumps(d, allow_nan=False)  # NaN must have become None
    back = json.loads(text)
    assert back["software"] == "resonus-room" and back["software_version"] == __version__
    assert back["notes"] == list(NOTES) and back["standards"] == list(result.standards)
    assert back["bands"]["nominal_hz"] == [250, 500, 1000, 2000]
    assert back["bands"]["fraction"] == 1
    assert back["onset_s"] == pytest.approx(result.onset / FS)
    for name in ("EDT", "T20", "T30", "C", "C50", "C80", "D50", "Ts", "PNR"):
        vals = [np.nan if v is None else v for v in back["values"][name]]
        np.testing.assert_allclose(vals, result.values(name))
    np.testing.assert_allclose(back["values"]["T20_std"], result.t20_std)
    assert back["fits"]["T30"]["n_points"] == [p.fits["T30"].n_points for p in result.band_results]
    assert back["fits"]["EDT"]["xi"][0] == pytest.approx(result.band_results[0].fits["EDT"].xi)
    assert back["flags"] == [list(p.flags) for p in result.band_results]
    assert back["broadband"]["T20"] == pytest.approx(result.broadband.t20)
    assert back["settings"]["decay_method"] == "correct"


def test_nan_becomes_none(result):
    ir = exponential_decay_ir(FS, 0.3, [0.05] * 4, OCT, noise_rms=0.05, seed=1)
    d = analysis_to_dict(analyze(ir, FS, OCT, onset=0))
    json.dumps(d, allow_nan=False)
    assert None in d["values"]["T30"]


def test_analysis_to_csv(result):
    rows = list(csv.reader(io.StringIO(analysis_to_csv(result))))
    assert tuple(rows[0]) == CSV_COLUMNS and len(rows) == 5
    r = dict(zip(CSV_COLUMNS, rows[2], strict=True))
    p = result.band_results[1]
    assert r["band_hz"] == "500" and r["center_hz"] == f"{result.bands.center[1]:.2f}"
    assert r["T20_s"] == f"{p.t20:.3f}" and r["C80_dB"] == f"{p.c80:.2f}"
    assert r["xi_T30"] == f"{p.fits['T30'].xi:.1f}" and r["flags"] == " ".join(p.flags)


def test_absorption_export():
    c = Climate(20.0, 50.0)
    t1 = np.linspace(6.0, 3.0, 18)
    res = absorption(t1, 0.7 * t1, 200.0, c, c, specimen_area=10.0)
    d = absorption_to_dict(res)
    json.dumps(d, allow_nan=False)
    assert d["standards"] == ["ISO 354:2003"] and d["frequency_hz"][0] == 100
    np.testing.assert_allclose(d["alpha"], np.round(res.alpha / 0.01) * 0.01)
    np.testing.assert_allclose(d["AT"], np.round(res.area / 0.1) * 0.1)
    assert d["A_obj"] == [None] * 18 and d["issues"] == list(res.issues)
    raw = absorption_to_dict(res, rounded=False)
    np.testing.assert_allclose(raw["alpha"], res.alpha)
    rows = list(csv.reader(io.StringIO(absorption_to_csv(res))))
    assert tuple(rows[0]) == ABSORPTION_COLUMNS and len(rows) == 19
    r = dict(zip(ABSORPTION_COLUMNS, rows[1], strict=True))
    assert r["band_hz"] == "100" and r["T1_s"] == "6.00" and r["A_obj_m2"] == ""
    assert r["alpha_s"] == f"{np.round(res.alpha[0] / 0.01) * 0.01:.2f}"


META = Metadata("resonus-room", __version__, description="pos 1", device="sim")


def test_analysis_band_results_roundtrip(result, tmp_path):
    br = analysis_to_band_results(result, META)
    assert br.fraction == 1 and br.labels == tuple(OCT.labels)
    np.testing.assert_array_equal(br.center, OCT.center)
    assert list(br.values) == list(ANALYSIS_QUANTITIES) and br.units == ANALYSIS_QUANTITIES
    np.testing.assert_array_equal(br.values["T20"], result.values("T20"))
    np.testing.assert_array_equal(br.values["T30_std"], result.t30_std)
    np.testing.assert_array_equal(
        br.values["xi_T30"], [p.fits["T30"].xi for p in result.band_results]
    )
    np.testing.assert_array_equal(
        br.values["r2_EDT"], [p.fits["EDT"].r2 for p in result.band_results]
    )
    assert br.flags == tuple(p.flags for p in result.band_results)
    assert br.metadata.standards == result.standards
    assert br.metadata.description == "pos 1" and br.metadata.device == "sim"
    assert br.metadata.extra["notes"] == list(NOTES)
    assert br.metadata.extra["onset_s"] == result.onset / FS
    data.save(tmp_path / "a.h5", br)
    back = data.load(tmp_path / "a.h5")
    np.testing.assert_array_equal(back.values["EDT"], br.values["EDT"])
    assert back.metadata.extra["settings"] == dict(result.settings)


def test_absorption_band_results():
    c = Climate(20.0, 50.0)
    t1 = np.linspace(6.0, 3.0, 18)
    res = absorption(t1, 0.7 * t1, 200.0, c, c, specimen_area=10.0)
    br = absorption_to_band_results(res, META)
    assert br.fraction == 3 and br.labels[0] == "100" and len(br.labels) == 18
    assert list(br.values) == list(ABSORPTION_QUANTITIES)
    np.testing.assert_array_equal(br.values["alpha_s"], res.alpha)  # unrounded
    np.testing.assert_array_equal(br.values["T2"], res.t2)
    assert np.all(np.isnan(br.values["A_obj"])) and br.flags == ()
    assert br.metadata.standards == ("ISO 354:2003",)
    assert br.metadata.extra["issues"] == list(res.issues)


def test_impulse_to_frequency_response_closed_form():
    from resonus_room.core.files import impulse_to_frequency_response

    fs, n, d = 48000.0, 1001, 7
    x = np.zeros(n)
    x[d] = 0.5
    meta = Metadata("resonus-room", "0", description="ext.wav")
    fr = impulse_to_frequency_response(ImpulseResponse(x, fs, "Pa/V", meta))
    assert fr.frequency.shape == fr.values.shape == (n // 2 + 1,)
    np.testing.assert_allclose(fr.frequency, np.arange(n // 2 + 1) * fs / n)
    # delayed scaled impulse: H(f) = 0.5 exp(-j 2 pi f d / fs)
    np.testing.assert_allclose(
        fr.values, 0.5 * np.exp(-2j * np.pi * fr.frequency * d / fs), atol=1e-12
    )
    assert fr.unit == "Pa/V" and fr.metadata == meta
