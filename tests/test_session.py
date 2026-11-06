"""Session list of results (D-19)."""

import numpy as np
import pytest
from resonus_core.data import ImpulseResponse, Metadata
from resonus_core.style import TOKENS, TRACE_COLORS

from resonus_room.core.session import (
    AVERAGE_COLOR,
    ENTRY_COLORS,
    Entry,
    Session,
)


def ir(k=0):
    return ImpulseResponse(np.arange(4.0) + k, 48000.0, "1", Metadata("t", "0"))


def chans(*labels):
    return [(lab, ir(i), None) for i, lab in enumerate(labels)]


def test_colors():
    assert ENTRY_COLORS == TRACE_COLORS and len(ENTRY_COLORS) == 7
    assert AVERAGE_COLOR == TOKENS["neutral_color"] == "#3d3846"
    assert AVERAGE_COLOR not in ENTRY_COLORS


def test_title():
    e = Entry(0, "Measurement 1", "s1m1", ir(), None, "#000000")
    assert e.title == "Measurement 1 s1m1"
    assert Entry(1, "bh.wav", "bh.wav", ir(), None, "#000000").title == "bh.wav"


def test_names():
    s = Session()
    assert s.next_measurement_name() == "Measurement 1"
    assert s.next_measurement_name() == "Measurement 2"
    assert s.unique_name("bh.wav") == "bh.wav"
    s.add("bh.wav", chans("bh.wav"))
    assert s.unique_name("bh.wav") == "bh.wav (2)"
    s.add("bh.wav", chans("bh.wav"))
    assert [e.name for e in s.entries] == ["bh.wav", "bh.wav (2)"]
    s.remove(s.entries[0].key)
    assert s.unique_name("bh.wav") == "bh.wav"
    assert s.unique_name("bh.wav (2)") == "bh.wav (2) (2)"


def test_add_keys_colors_active():
    s = Session()
    assert s.active is None and s.active_entry is None and s.shown() == []
    new = s.add("Measurement 1", chans("s1m1", "s1m2"))
    assert [e.key for e in new] == [0, 1] and [e.label for e in new] == ["s1m1", "s1m2"]
    assert all(isinstance(e, Entry) and e.name == "Measurement 1" for e in new)
    assert s.active == 0 and s.active_entry is new[0]
    np.testing.assert_array_equal(new[1].ir.data, ir(1).data)
    s.set_overlay(1, True)
    more = s.add("Measurement 2", chans("s1m1"))
    assert more[0].key == 2 and s.active == 2 and s.overlay == {1}
    assert [e.color for e in s.entries] == list(ENTRY_COLORS[:3])
    with pytest.raises(ValueError, match="no channels"):
        s.add("x", [])
    assert len(s.entries) == 3


def test_colors_cycle_and_stay():
    s = Session()
    for k in range(9):
        s.add(f"f{k}.wav", chans(f"f{k}.wav"))
    assert s.entries[7].color == ENTRY_COLORS[0] and s.entries[8].color == ENTRY_COLORS[1]
    s.remove(0)
    assert s.entries[0].key == 1 and s.entries[0].color == ENTRY_COLORS[1]
    s.add("g.wav", chans("g.wav"))
    assert s.entries[-1].key == 9  # keys are not reused


def test_active_overlay_shown():
    s = Session()
    s.add("Measurement 1", chans("s1m1", "s1m2", "s1m3"))
    s.set_overlay(2, True)
    s.set_overlay(1, True)
    s.set_overlay(0, True)  # active: ignored
    assert s.overlay == {1, 2}
    assert [e.key for e in s.shown()] == [0, 1, 2]
    s.set_active(2)
    assert s.overlay == {1} and [e.key for e in s.shown()] == [2, 1]
    s.set_overlay(1, False)
    assert [e.key for e in s.shown()] == [2]
    for call in (lambda: s.set_active(7), lambda: s.set_overlay(7, True), lambda: s.remove(7)):
        with pytest.raises(KeyError):
            call()
    with pytest.raises(KeyError):
        s.get(7)
    assert s.get(1).label == "s1m2" and s.active == 2


def test_remove_and_clear():
    s = Session()
    s.add("Measurement 1", chans("s1m1", "s1m2", "s1m3"))
    s.set_overlay(1, True)
    s.set_active(2)
    s.remove(2)  # active removed: the first remaining entry becomes active
    assert s.active == 0 and s.overlay == {1}
    s.set_active(1)
    assert s.overlay == set()
    s.set_overlay(0, True)
    s.remove(0)
    assert s.overlay == set() and s.active == 1
    s.remove(1)
    assert s.active is None and s.entries == [] and s.shown() == []
    s.add("a.wav", chans("a.wav", "b"))
    s.set_overlay(s.entries[1].key, True)
    s.clear()
    assert s.entries == [] and s.active is None and s.overlay == set()
    assert s.add("c.wav", chans("c.wav"))[0].key == 5


def test_group_and_invalidate():
    s = Session()
    s.add("Measurement 1", chans("s1m1", "s1m2"))
    s.add("bh.wav", chans("bh.wav"))
    assert [e.label for e in s.group("Measurement 1")] == ["s1m1", "s1m2"]
    assert s.group("nothing") == []
    for e in s.entries:
        e.analysis = object()
    s.invalidate()
    assert all(e.analysis is None for e in s.entries)


def test_rename():
    s = Session()
    s.add("Measurement 1", chans("s1m1", "s1m2"))
    s.add("ir.wav", chans("ir.wav"))
    s.rename(2, "  s2m1 ")
    assert s.get(2).label == "s2m1" and s.get(2).title == "ir.wav s2m1"
    np.testing.assert_array_equal(s.get(2).ir.data, ir(0).data)  # data unchanged
    s.rename(0, "s1m1")  # same label: allowed
    s.rename(2, "s1m1")  # same label in another measurement: allowed
    with pytest.raises(ValueError, match="duplicate label 's1m2' in Measurement 1"):
        s.rename(0, "s1m2")
    for bad, msg in (
        ("", "empty label"),
        ("   ", "empty label"),
        ("a/b", "path separator"),
        ("a\\b", "path separator"),
    ):
        with pytest.raises(ValueError, match=msg):
            s.rename(1, bad)
    assert [e.label for e in s.entries] == ["s1m1", "s1m2", "s1m1"]
    with pytest.raises(KeyError):
        s.rename(9, "x")
