"""CLI test-levels: the level test of the GUI's Test button (tasks 031c/032, D-21)."""

import json

from resonus_room.cli import main
from resonus_room.core.engine import open_device
from resonus_room.core.levels import level_test, meter_zone
from resonus_room.core.settings import AppSettings


def test_json_matches_level_test(capsys):
    assert main(["test-levels", "--simulate", "--level", "-20", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    s = AppSettings(simulate=True, level=-20.0)
    ((out, want),) = level_test(s, open_device(s))
    assert [(r["output"], r["input"], r["label"]) for r in rows] == [("s1", "m1", "s1m1")]
    assert out == "s1"
    assert rows[0]["peak_db"] == want[0].peak_db and rows[0]["rms_db"] == want[0].rms_db
    assert rows[0]["zone"] == meter_zone(want[0].peak_db)
    assert rows[0]["clipped"] is False


def two_by_two(tmp_path, reference=0):
    path = tmp_path / "settings.toml"
    AppSettings(
        simulate=True,
        input_channels=(1, 2),
        input_labels=("m1", "m2"),
        output_channels=(1, 2),
        output_labels=("s1", "s2"),
        reference_channel=reference,
    ).save(path)
    return path


def test_text_one_line_per_pair(capsys, tmp_path):
    assert main(["test-levels", "--settings", str(two_by_two(tmp_path)), "--level", "-20"]) == 0
    out = capsys.readouterr()
    lines = out.out.strip().splitlines()
    assert [line.split()[0] for line in lines] == ["s1m1", "s1m2", "s2m1", "s2m2"]
    assert all("peak" in line and "RMS" in line and "dB" in line for line in lines)
    assert "CLIPPED" not in out.out and out.err == ""


def test_clipping_reported(capsys, tmp_path):
    # the simulated loopback reference is a wire from output 1: at 0 dB it reaches full
    # scale (clipped); output 2 does not reach it. Room paths may clip too (noise crest
    # factor), so only the reference pairs are asserted.
    settings = str(two_by_two(tmp_path, reference=3))
    assert main(["test-levels", "--settings", settings, "--level", "0"]) == 0
    out = capsys.readouterr()
    clipped = [line.split()[0] for line in out.out.splitlines() if line.endswith("CLIPPED")]
    assert "s1ref" in clipped and "s2ref" not in clipped
    assert out.err == "".join(f"warning: {label} clipped\n" for label in clipped)
    assert main(["test-levels", "--settings", settings, "--level", "0", "--json"]) == 0
    out = capsys.readouterr()
    assert [r["label"] for r in json.loads(out.out) if r["clipped"]] == clipped
    assert out.err == "".join(f"warning: {label} clipped\n" for label in clipped)
