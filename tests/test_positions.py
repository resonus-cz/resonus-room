import math
from pathlib import Path

import pytest

from resonus_room.core.positions import (
    METHODS,
    POSITION_REQUIREMENTS,
    Position,
    PositionSetup,
    check_positions,
    load_positions,
    minimum_source_distance,
)

OCT = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]


def setup(method, pairs, **kw):
    pos = tuple(Position(Path(f"{s}{m}.wav"), s, m) for s, m in pairs)
    return PositionSetup(method, pos, **kw)


def grid(n_src, n_mic):
    return [(f"S{i}", f"M{j}") for i in range(n_src) for j in range(n_mic)]


def test_table_1_values():
    # ISO 3382-2:2008 Table 1 and footnote a
    assert POSITION_REQUIREMENTS == {
        "survey": (2, 1, 2),
        "engineering": (6, 2, 2),
        "precision": (12, 2, 3),
        "correction": (3, 1, 3),
    }
    assert METHODS == ("survey", "engineering", "precision", "correction")


def test_minimum_distance_eq_1():
    # d_min = 2 sqrt(V / (c T)); V = 200 m3, c = 343 m/s, T = 1 s -> 1.527 m
    assert minimum_source_distance(200.0, 343.0, 1.0) == pytest.approx(2 * math.sqrt(200 / 343))
    for bad in ((0.0, 343.0, 1.0), (200.0, -1.0, 1.0), (200.0, 343.0, 0.0)):
        with pytest.raises(ValueError):
            minimum_source_distance(*bad)


@pytest.mark.parametrize(
    ("method", "n_src", "n_mic"),
    [("survey", 1, 2), ("engineering", 2, 3), ("precision", 2, 6), ("correction", 1, 3)],
)
def test_minimum_layouts_pass(method, n_src, n_mic):
    assert check_positions(setup(method, grid(n_src, n_mic)), 1, OCT) == []


def test_too_few_positions():
    issues = check_positions(setup("precision", grid(1, 6)), 1, OCT)
    assert issues == [
        "ISO 3382-2 Table 1: 6 source-microphone combinations, precision requires 12",
        "ISO 3382-2 Table 1: 1 source positions, precision requires 2",
    ]
    # repeated pairs count once; microphones counted separately
    s = setup("engineering", [("S1", "M1"), ("S1", "M1"), ("S2", "M1")])
    assert check_positions(s, 1, OCT) == [
        "ISO 3382-2 Table 1: 2 source-microphone combinations, engineering requires 6",
        "ISO 3382-2 Table 1: 1 microphone positions, engineering requires 2",
    ]


def test_bands_and_range():
    s = setup("survey", grid(1, 2))
    assert check_positions(s, 3, [250.0, 2000.0]) == [
        "ISO 3382-2 4.3.2: survey measurements are made in octave bands only"
    ]
    assert check_positions(s, 1, [500.0, 1000.0]) == [
        "ISO 3382-2 5.1: the frequency range should cover at least 250-2000 Hz"
    ]
    e = setup("engineering", grid(2, 3))
    assert check_positions(e, 3, [100.0, 5000.0]) == []
    assert check_positions(e, 3, [125.0, 5000.0]) == [
        "ISO 3382-2 5.1: the frequency range should cover at least 100-5000 Hz"
    ]


def test_minimum_distance_check():
    pos = (
        Position(Path("a.wav"), "S1", "M1", 1.0),
        Position(Path("b.wav"), "S1", "M2", 3.0),
        Position(Path("c.wav"), "S1", "M3", None),
    )
    s = PositionSetup("correction", pos, volume=200.0, reverberation_time=1.0, temperature=20.0)
    d = 2 * math.sqrt(200 / (331 + 0.6 * 20))
    assert check_positions(s, 1, OCT) == [
        f"ISO 3382-2 4.3.1 Eq. (1): a.wav: distance 1 m is below d_min = {d:.2f} m"
    ]
    # without volume or expected T no distance check
    s2 = PositionSetup("correction", pos, volume=200.0)
    assert check_positions(s2, 1, OCT) == []


def test_load_positions(tmp_path):
    (tmp_path / "pos.toml").write_text(
        'method = "engineering"\nvolume = 120.5\nreverberation_time = 0.8\n'
        '[[position]]\nfile = "a.wav"\nsource = "S1"\nmicrophone = 1\ndistance = 2.5\n'
        '[[position]]\nfile = "sub/b.wav"\nsource = "S2"\nmicrophone = "M2"\n'
    )
    s = load_positions(tmp_path / "pos.toml")
    assert s.method == "engineering" and s.volume == 120.5 and s.reverberation_time == 0.8
    assert s.temperature == 20.0
    assert s.positions[0] == Position(tmp_path / "a.wav", "S1", "1", 2.5)
    assert s.positions[1] == Position(tmp_path / "sub" / "b.wav", "S2", "M2", None)


@pytest.mark.parametrize(
    "text",
    [
        '[[position]]\nfile = "a.wav"\nsource = "S1"\nmicrophone = "M1"\n',  # no method
        'method = "laboratory"\n[[position]]\nfile = "a"\nsource = "S"\nmicrophone = "M"\n',
        'method = "survey"\n',  # no positions
        'method = "survey"\nposition = []\n',
        'method = "survey"\n[[position]]\nfile = "a.wav"\nsource = "S1"\n',  # no microphone
    ],
)
def test_load_errors(tmp_path, text):
    (tmp_path / "p.toml").write_text(text)
    with pytest.raises(ValueError):
        load_positions(tmp_path / "p.toml")
