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
from resonus_room.core.compare import (
    ABSOLUTE_TOLERANCES,
    COMPARISON_COLUMNS,
    RELATIVE_TOLERANCE,
    Comparison,
    compare,
    comparisons_to_csv,
    summary,
)
from resonus_room.core.simulate import exponential_decay_ir

META = Metadata("t", "0")
REAL = Path(__file__).parents[1] / "refs" / "rew-test-data" / "resonus-room-test-data" / "test1"


def br(fraction, lo, hi, values, units):
    b = band_set(fraction, lo, hi)
    return BandResults(fraction, b.center, tuple(b.labels), values, units, META)


def test_compare_matches_bands_and_computes_differences():
    ours = br(1, 125, 1000, {"T20": [1.0, 1.0, 1.0, 1.0], "C80": [1.0, 2.0, 3.0, 4.0]},
              {"T20": "s", "C80": "dB"})  # fmt: skip
    ref = br(1, 250, 2000, {"T20": [0.99, 0.97, np.nan, 5.0], "C80": [1.5, 3.0, 4.0, 0.0]},
             {"T20": "s", "C80": "dB"})  # fmt: skip
    assert RELATIVE_TOLERANCE == 0.02
    t20, c80 = compare(ours, ref)
    assert isinstance(t20, Comparison) and t20.quantity == "T20" and c80.quantity == "C80"
    assert t20.labels == ("250", "500", "1k")
    np.testing.assert_allclose(t20.ours, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(t20.difference[:2], [100 * (1 / 0.99 - 1), 100 * (1 / 0.97 - 1)])
    assert np.isnan(t20.difference[2]) and t20.difference_unit == "%"
    assert t20.within.tolist() == [True, False, False]
    np.testing.assert_allclose(c80.difference, [0.5, 0.0, 0.0])
    assert ABSOLUTE_TOLERANCES == {"C80": 0.5}
    assert c80.difference_unit == "dB" and c80.within.tolist() == [True, True, True]
    # quantity selection and order; missing quantities skipped
    assert [c.quantity for c in compare(ours, ref, ("C80", "T20", "EDT"))] == ["C80", "T20"]


def test_compare_errors():
    a = br(1, 125, 250, {"T20": [1.0, 1.0]}, {"T20": "s"})
    with pytest.raises(ValueError):
        compare(a, br(3, 125, 160, {"T20": [1.0, 1.0]}, {"T20": "s"}))
    with pytest.raises(ValueError):
        compare(a, br(1, 1000, 2000, {"T20": [1.0, 1.0]}, {"T20": "s"}))


def test_csv_and_summary():
    ours = br(1, 125, 250, {"T20": [1.0, np.nan], "D50": [0.5, 0.6]}, {"T20": "s", "D50": "1"})
    ref = br(1, 125, 250, {"T20": [1.01, 1.0], "D50": [0.4, 0.6]}, {"T20": "s", "D50": "1"})
    comps = compare(ours, ref)
    rows = list(csv.reader(io.StringIO(comparisons_to_csv(comps))))
    assert tuple(rows[0]) == COMPARISON_COLUMNS and len(rows) == 5
    assert rows[1] == ["125", "T20", "1", "1.01", f"{100 * (1 / 1.01 - 1):.4g}", "%", "yes"]
    assert rows[2] == ["250", "T20", "", "1", "", "%", "no"]
    assert rows[3] == ["125", "D50", "0.5", "0.4", "0.1", "1", ""]
    assert summary(comps) == ["T20: 1/2 bands within 2 %"]
    c80 = br(1, 125, 250, {"C80": [1.0, 2.0]}, {"C80": "dB"})
    ref80 = br(1, 125, 250, {"C80": [1.6, np.nan]}, {"C80": "dB"})
    (c,) = compare(c80, ref80)
    assert c.within.tolist() == [False, False]
    assert summary([c]) == ["C80: 0/2 bands within 0.5 dB"]


def rew_file(path, t):
    """REW-style RT60 export with the given T for 125...4000 Hz octaves."""
    head = (
        "RT60 data saved by REW V5.31.3\nMeasurement: synthetic\n\n"
        "Format is freq (Hz), BW (octaves), ...\n\n"
    )
    rows = [
        f"{f}, 1/1, {x:.3f}, -0.999, {x:.3f}, -0.999, {x:.3f}, -0.999, {x:.3f}, -0.999, "
        f"-5.000, -35.000, 0.000, Zero Phase, 0.00, 0.00, 50.0, 0.050"
        for f, x in zip((125, 250, 500, 1000, 2000, 4000), t, strict=True)
    ]
    path.write_text(head + "\n".join(rows) + "\n")


def test_cli_compare_rew_and_import(tmp_path, capsys):
    t = np.array([1.2, 1.1, 1.0, 0.9, 0.8, 0.7])
    ir = exponential_decay_ir(48000, 3.0, t, band_set(1, 125, 4000), delay=0.01, seed=4)
    wavfile.write(tmp_path / "ir.wav", 48000, (0.5 * ir / np.abs(ir).max()).astype(np.float32))
    rew_file(tmp_path / "RT60.txt", t)
    out = tmp_path / "cmp.csv"
    args = ["compare-rew", str(tmp_path / "ir.wav"), str(tmp_path / "RT60.txt")]
    assert main(args + ["--bands", "octave", "--out", str(out)]) == 0
    rows = list(csv.reader(io.StringIO(out.read_text())))
    assert tuple(rows[0]) == COMPARISON_COLUMNS and len(rows) == 1 + 7 * 6
    t20 = [r for r in rows[1:] if r[1] == "T20"]
    assert [r[0] for r in t20] == ["125", "250", "500", "1k", "2k", "4k"]
    assert all(abs(float(r[4])) < 6 for r in t20)
    err = capsys.readouterr().err
    assert "T20: " in err and "bands within 2 %" in err
    # without --out the CSV goes to stdout
    assert main(args + ["--bands", "octave"]) == 0
    assert capsys.readouterr().out.startswith("band,quantity,")
    # no one-third-octave rows in this export
    assert main(args + ["--bands", "third"]) == 1
    # import: one BandResults file per resolution present
    assert main(["import-rew", str(tmp_path / "RT60.txt"), "--out", str(tmp_path / "r")]) == 0
    imported = data.load(tmp_path / "r" / "rew_octave.h5")
    assert isinstance(imported, BandResults) and imported.metadata.software == "REW"
    assert not (tmp_path / "r" / "rew_third.h5").exists()


@pytest.mark.skipif(not REAL.exists(), reason="REW sample data (refs/) not available")
def test_real_rew_agreement(tmp_path):
    # Regression guard for the verification in docs/VERIFICATION.md (V10): REW V5.31.3,
    # 24 positions in two boxes, one-third octaves 400 Hz - 4 kHz.
    from resonus_core.bands import band_set as bs

    from resonus_room.core.analysis import analyze
    from resonus_room.core.files import analysis_to_band_results, load_impulse_response
    from resonus_room.core.rew import read_rew_rt60

    diffs = {"T20": [], "T30": []}
    for wav in sorted(REAL.glob("*.wav")):
        ir = load_impulse_response(wav)
        a = analyze(ir.data, ir.sample_rate, bs(3, 400, 4000))
        ref = read_rew_rt60(wav.with_name(f"RT60_{wav.stem}.txt")).third
        for c in compare(analysis_to_band_results(a, META), ref, ("T20", "T30")):
            diffs[c.quantity].append(c.difference)
    for q, d in diffs.items():
        d = np.abs(np.array(d))
        assert d.shape == (24, 11)
        assert np.median(d) < 1.0, q
        assert np.mean(d <= 100 * RELATIVE_TOLERANCE) >= 0.9, q
