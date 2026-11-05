from pathlib import Path

import numpy as np
import pytest
from resonus_core.bands import band_set
from resonus_core.data import BandResults

from resonus_room.core.rew import REW_COLUMNS, RewRT60, parse_value, read_rew_rt60

SAMPLE = Path(__file__).parent / "data" / "rew_rt60_sample.txt"
REAL = Path(__file__).parents[1] / "refs" / "rew-test-data" / "resonus-room-test-data"


def test_parse_value():
    assert parse_value("1.25") == 1.25 and parse_value("-0.5") == -0.5
    for empty in ("", "NaN", "-", "nan"):
        assert np.isnan(parse_value(empty))
    with pytest.raises(ValueError):
        parse_value("abc")


def test_read_sample():
    r = read_rew_rt60(SAMPLE)
    assert isinstance(r, RewRT60) and r.filtering == "Zero Phase"
    assert r.header["Saved by"] == "REW V5.31.3"
    assert r.header["Measurement"] == "Synthetic sample pos 1"
    assert r.header["Start time"] == "-2.730666668333475 seconds"
    o, t = r.octave, r.third
    assert isinstance(o, BandResults) and isinstance(t, BandResults)
    assert o.fraction == 1 and o.labels == ("125", "250", "1k")
    np.testing.assert_allclose(o.center[[0, 2]], band_set(1, 125, 1000).center[[0, 3]])
    assert t.fraction == 3 and t.labels == ("100", "125", "160", "10k")
    assert list(o.values) == [name for name, _ in REW_COLUMNS.values()]
    assert o.units["T20"] == "s" and o.units["D50"] == "1"
    np.testing.assert_allclose(o.values["T20"], [1.2, 1.05, 0.95])
    np.testing.assert_allclose(o.values["r_T30"], [-0.999, -0.999, -0.997])
    np.testing.assert_allclose(t.values["D50"], [0.74, 0.426, 0.307, 0.967])
    np.testing.assert_allclose(t.values["Topt_end"], [-59, -28, -31, -47])
    assert r.broadband["T30"] == 1.073 and r.broadband["D50"] == pytest.approx(0.537)
    assert o.metadata.software == "REW" and o.metadata.software_version == "REW V5.31.3"
    assert o.metadata.description == "Synthetic sample pos 1"
    assert o.metadata.extra == {
        "dated": "01-Nov-2026 10:00:00",
        "filtering": "Zero Phase",
        "file": SAMPLE.name,
    }


def test_only_octaves(tmp_path):
    lines = SAMPLE.read_text().splitlines()
    keep = [ln for ln in lines if ", 1/3," not in ln]
    (tmp_path / "o.txt").write_text("\n".join(keep))
    r = read_rew_rt60(tmp_path / "o.txt")
    assert r.third is None and r.octave is not None and r.broadband == {}


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.replace("RT60 data saved by", "Something else"),
        lambda s: s.replace("Format is", "Columns are"),
        lambda s: s.replace("125, 1/1,", "125, 1/2,"),
        lambda s: s.replace("250, 1/1,", "260, 1/1,"),  # not a nominal frequency
        lambda s: s.replace("1000, 1/1,", "200, 1/1,"),  # 200 Hz is no octave band
        lambda s: s.replace("1.050, -0.999", "x.xx, -0.999"),
    ],
)
def test_malformed(tmp_path, change):
    (tmp_path / "bad.txt").write_text(change(SAMPLE.read_text()))
    with pytest.raises(ValueError):
        read_rew_rt60(tmp_path / "bad.txt")


@pytest.mark.skipif(not REAL.exists(), reason="REW sample data (refs/) not available")
def test_real_exports():
    files = sorted(REAL.glob("*/RT60_*.txt"))
    assert len(files) == 36
    for f in files:
        r = read_rew_rt60(f)
        assert r.octave is not None and r.octave.labels[0] == "63" and len(r.octave.labels) == 8
        assert r.third is not None and r.third.labels[0] == "50" and len(r.third.labels) == 24
        assert np.all(r.third.values["T20"] > 0) and r.filtering == "Zero Phase"
