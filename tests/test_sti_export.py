"""Export of the STI at a position (D-31)."""

import csv
import io
import json

import numpy as np
import pytest
from resonus_core.data import BandResults, Metadata

from resonus_room.core.files import NOTES
from resonus_room.core.sti import BAND_CENTERS, MODULATION_FREQUENCIES
from resonus_room.core.sti_export import (
    CSV_HEADER,
    METHOD_NOTE,
    STI_QUANTITIES,
    sti_to_band_results,
    sti_to_csv,
    sti_to_dict,
)
from resonus_room.core.sti_inputs import StiReport


def report():
    m = np.full((7, 14), 0.5)
    m[0, 0] = 0.25
    return StiReport(
        sti=0.51234,
        mti=np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]),
        m=m,
        speech_levels_db=np.array([60.0, 61.0, 62.0, 63.04, 58.0, 52.0, 47.0]),
        noise_levels_db=np.array([45.0, 40.0, 35.0, 30.0, 25.0, 20.0, -np.inf]),
        seed=3,
    )


def test_method_note_is_provisional():
    assert METHOD_NOTE.startswith("PROVISIONAL")
    assert "IEC 60268-16:2020" in METHOD_NOTE
    assert STI_QUANTITIES == {"MTI": "1", "speech_level_1m": "dB", "noise_level": "dB"}


def test_to_dict():
    d = sti_to_dict(report())
    assert list(d) == [
        "STI", "bands_hz", "MTI", "speech_level_1m_dB", "noise_level_dB",
        "modulation_frequencies_hz", "m", "seed", "method", "notes",
    ]  # fmt: skip
    assert d["STI"] == 0.51234
    assert d["bands_hz"] == [125, 250, 500, 1000, 2000, 4000, 8000]
    assert d["MTI"] == [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    assert d["noise_level_dB"][-1] is None and d["noise_level_dB"][0] == 45.0
    assert d["modulation_frequencies_hz"] == list(MODULATION_FREQUENCIES)
    assert len(d["m"]) == 7 and all(len(row) == 14 for row in d["m"])
    assert d["m"][0][0] == 0.25
    assert d["seed"] == 3 and d["method"] == METHOD_NOTE and d["notes"] == list(NOTES)
    json.dumps(d, allow_nan=False)  # JSON-ready
    assert all(type(x) is float for x in d["MTI"])


def test_to_csv():
    text = sti_to_csv(report())
    lines = text.split("\n")
    assert lines[0] == CSV_HEADER == "band_hz,MTI,speech_level_1m_dB,noise_level_dB"
    assert lines[1] == "125,0.400,60.0,45.0"
    assert lines[4] == "1000,0.550,63.0,30.0"
    assert lines[7] == "8000,0.700,47.0,"
    assert lines[8] == "STI,0.512,,"
    assert text.endswith("\n") and len(lines) == 10
    rows = list(csv.reader(io.StringIO(text)))
    assert all(len(r) == 4 for r in rows)


def test_to_band_results():
    meta = Metadata("resonus-room", "0.1", description="s1m1", extra={"position": "P1"})
    r = report()
    b = sti_to_band_results(r, meta)
    assert isinstance(b, BandResults)
    assert b.fraction == 1
    np.testing.assert_allclose(b.center, BAND_CENTERS)
    assert b.labels[0] == "125" and len(b.labels) == 7
    assert list(b.values) == ["MTI", "speech_level_1m", "noise_level"]
    np.testing.assert_allclose(b.values["MTI"], r.mti)
    assert np.isnan(b.values["noise_level"][-1]) and b.values["noise_level"][0] == 45.0
    assert b.units == STI_QUANTITIES
    assert b.metadata.description == "s1m1"
    assert b.metadata.extra == {
        "position": "P1", "STI": 0.51234, "seed": 3, "method": METHOD_NOTE, "notes": list(NOTES),
    }  # fmt: skip
    assert meta.extra == {"position": "P1"}  # not modified
    assert np.isinf(r.noise_levels_db[-1])  # report not modified


def test_band_results_round_trip(tmp_path):
    from resonus_core import data

    b = sti_to_band_results(report(), Metadata("resonus-room", "0.1"))
    data.save(tmp_path / "sti.h5", b)
    back = data.load(tmp_path / "sti.h5")
    assert back.metadata.extra["STI"] == pytest.approx(0.51234)
