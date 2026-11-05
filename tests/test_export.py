import json
from pathlib import Path

import numpy as np
import pytest
from resonus_core import data
from resonus_core.bands import band_set
from resonus_core.data import BandResults, ImpulseResponse, Metadata
from resonus_core.plot import MplPlotView
from scipy.io import wavfile

from resonus_room.core.analysis import analyze
from resonus_room.core.export import (
    EXPORT_FORMATS,
    IR_FORMATS,
    IR_SUFFIX,
    PLOT_SUFFIXES,
    channel_paths,
    export_channels,
    export_results,
    export_suffix,
)
from resonus_room.core.files import analysis_to_csv
from resonus_room.core.simulate import exponential_decay_ir
from resonus_room.plotting import setup_rt_axes, show_rt

FS = 48000


@pytest.fixture(scope="module")
def results():
    bands = band_set(1, 125, 4000)
    x = exponential_decay_ir(FS, 1.5, np.full(6, 0.8), bands, delay=0.01, noise_rms=1e-6)
    analysis = analyze(x, FS, bands)
    meta = Metadata("resonus-room", "0", description="room 1", device="sim")
    ir = ImpulseResponse(x, FS, "1", meta)
    view = MplPlotView()
    setup_rt_axes(view)
    show_rt(view, analysis, "T20", "T20", "T20")
    return analysis, ir, view, meta


def test_formats():
    assert set(EXPORT_FORMATS) == {".png", ".svg", ".pdf", ".json", ".csv", ".ir.h5", ".h5", ".wav"}
    assert set(PLOT_SUFFIXES) <= set(EXPORT_FORMATS)


@pytest.mark.parametrize(
    ("suffix", "magic"), [(".png", b"\x89PNG"), (".pdf", b"%PDF"), (".svg", b"<?xml")]
)
def test_plot_formats(results, tmp_path, suffix, magic):
    out = export_results(tmp_path / f"plot{suffix}", *results)
    assert out == tmp_path / f"plot{suffix}"
    assert out.read_bytes().startswith(magic)


def test_extension_case_insensitive(results, tmp_path):
    out = export_results(tmp_path / "PLOT.PNG", *results)
    assert out.read_bytes().startswith(b"\x89PNG")


def test_json(results, tmp_path):
    analysis = results[0]
    d = json.loads(export_results(tmp_path / "a.json", *results).read_text(encoding="utf-8"))
    assert d["source"] == "room 1" and d["device"] == "sim"
    assert d["software"] == "resonus-room"
    assert d["values"]["T20"] == pytest.approx(list(analysis.values("T20")))


def test_csv(results, tmp_path):
    text = export_results(tmp_path / "a.csv", *results).read_text(encoding="utf-8")
    assert text == analysis_to_csv(results[0])


def test_h5_band_results(results, tmp_path):
    analysis = results[0]
    br = data.load(export_results(tmp_path / "a.h5", *results))
    assert isinstance(br, BandResults)
    np.testing.assert_allclose(br.values["T20"], analysis.values("T20"))
    assert br.metadata.description == "room 1"


def test_wav_impulse_response(results, tmp_path):
    ir = results[1]
    rate, y = wavfile.read(export_results(tmp_path / "ir.wav", *results))
    assert rate == FS and y.dtype == np.float32 and y.shape == ir.data.shape
    np.testing.assert_allclose(y, ir.data.astype(np.float32))


@pytest.mark.parametrize("name", ["a.txt", "noext"])
def test_unsupported_extension(results, tmp_path, name):
    with pytest.raises(ValueError, match="unsupported file extension"):
        export_results(tmp_path / name, *results)
    assert not (tmp_path / name).exists()


@pytest.mark.parametrize("name", ["room.ir.h5", "ROOM.IR.H5"])
def test_ir_h5_keeps_impulse_response_and_metadata(results, tmp_path, name):
    ir = results[1]
    assert IR_SUFFIX == ".ir.h5"
    back = data.load(export_results(tmp_path / name, *results))
    assert isinstance(back, ImpulseResponse)
    np.testing.assert_array_equal(back.data, ir.data)
    assert back.sample_rate == ir.sample_rate and back.unit == ir.unit
    assert back.metadata == ir.metadata  # the IR's own metadata, not the export's


def test_export_suffix():
    assert export_suffix("a/Room.IR.h5") == IR_SUFFIX
    assert export_suffix("room.h5") == ".h5" and export_suffix("x.WAV") == ".wav"
    assert IR_FORMATS == (IR_SUFFIX, ".wav")


def test_channel_paths(tmp_path):
    # one file per channel, named after the s#m# label (D-17)
    p = tmp_path / "room.ir.h5"
    assert channel_paths(p, ["s1m1"]) == [p]
    assert channel_paths(p, ["s1m1L", "s1m1R"]) == [
        tmp_path / "room_s1m1L.ir.h5",
        tmp_path / "room_s1m1R.ir.h5",
    ]
    assert channel_paths(tmp_path / "Hall.IR.H5", ["s1m1", "s2m1"])[1].name == "Hall_s2m1.IR.H5"
    assert channel_paths("pos.3.wav", ["s1m1", "s1+s2m2"]) == [
        Path("pos.3_s1m1.wav"),
        Path("pos.3_s1+s2m2.wav"),
    ]
    with pytest.raises(ValueError, match="no channels"):
        channel_paths(p, [])
    with pytest.raises(ValueError, match="duplicate channel label 's1m1'"):
        channel_paths(p, ["s1m1", "s1m1"])


def test_export_channels(tmp_path):
    irs = []
    for k, label in enumerate(("s1m1L", "s1m1R")):
        meta = Metadata("resonus-room", "0", extra={"channel": {"label": label}})
        irs.append((label, ImpulseResponse(np.arange(4.0) + k, 48000.0, "1", meta)))
    paths = export_channels(tmp_path / "bh.ir.h5", irs)
    assert [q.name for q in paths] == ["bh_s1m1L.ir.h5", "bh_s1m1R.ir.h5"]
    for q, (label, ir) in zip(paths, irs, strict=True):
        back = data.load(q)
        np.testing.assert_array_equal(back.data, ir.data)
        assert back.metadata.extra["channel"]["label"] == label  # own metadata kept
    wavs = export_channels(tmp_path / "bh.wav", irs)
    fs, x = wavfile.read(wavs[1])
    assert fs == 48000 and x.dtype == np.float32
    np.testing.assert_array_equal(x, np.arange(4.0, dtype=np.float32) + 1)
    with pytest.raises(ValueError, match="one file per channel"):
        export_channels(tmp_path / "bh.csv", irs)
    assert not (tmp_path / "bh.csv").exists()
