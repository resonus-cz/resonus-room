"""Results list widget of the header (D-19)."""

import numpy as np
import pytest

gi = pytest.importorskip("gi")
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gtk  # noqa: E402

if not Gtk.init_check():
    pytest.skip("no display available", allow_module_level=True)

from resonus_core.data import ImpulseResponse, Metadata  # noqa: E402

from resonus_room.core.session import Session  # noqa: E402
from resonus_room.gui.results import SWATCH, ResultRow, ResultsList  # noqa: E402

Adw.init()


def session():
    s = Session()
    ir = ImpulseResponse(np.ones(4), 48000.0, "1", Metadata("t", "0"))
    s.add("Measurement 1", [("s1m1", ir, None), ("s1m2", ir, None)])
    s.add("bh.wav", [("bh.wav", ir, None)])
    return s


class Fixture:
    def __init__(self, s):
        self.session = s
        self.calls = []
        self.widget = ResultsList(
            s,
            on_active=lambda k: self.calls.append(("active", k)),
            on_overlay=lambda k, on: self.calls.append(("overlay", k, on)),
            on_remove=lambda k: self.calls.append(("remove", k)),
            on_clear=lambda: self.calls.append(("clear",)),
            on_rename=lambda k, text: self.calls.append(("rename", k, text)),
        )


def children(box):
    out, child = [], box.get_first_child()
    while child is not None:
        out.append(child)
        child = child.get_next_sibling()
    return out


def test_rows_reflect_session():
    s = session()
    s.set_overlay(0, True)
    f = Fixture(s)
    w = f.widget
    assert isinstance(w.button, Gtk.MenuButton) and w.button.get_icon_name() == "view-list-symbolic"
    assert w.button.get_popover() is not None
    assert [r.key for r in w.rows] == [0, 1, 2] and all(isinstance(r, ResultRow) for r in w.rows)
    assert [r.name.get_label() for r in w.rows] == ["Measurement 1", "Measurement 1", "bh.wav"]
    assert [r.name.get_visible() for r in w.rows] == [True, True, False]  # reads as the title
    assert [r.label.get_text() for r in w.rows] == ["s1m1", "s1m2", "bh.wav"]
    assert all(isinstance(r.label, Gtk.EditableLabel) and r.label.get_editable() for r in w.rows)
    assert [r.active.get_active() for r in w.rows] == [False, False, True]  # bh.wav added last
    assert [r.overlay.get_active() for r in w.rows] == [True, False, False]
    assert [r.overlay.get_sensitive() for r in w.rows] == [True, True, False]
    e = s.get(1)
    assert w.rows[1].swatch.get_label() == f'<span foreground="{e.color}">{SWATCH}</span>'
    assert w.rows[1].swatch.get_use_markup()
    r = w.rows[1]
    assert children(r.box) == [r.active, r.swatch, r.name, r.label, r.overlay, r.remove]
    assert r.remove.get_icon_name() == "window-close-symbolic"
    assert w.list.get_row_at_index(2) is not None and w.list.get_row_at_index(3) is None
    assert w.button.get_sensitive() and w.clear_button.get_sensitive()
    assert f.calls == []  # building calls nothing


def test_clicks_call_back_and_refresh_is_silent():
    s = session()
    f = Fixture(s)
    w = f.widget
    w.rows[0].active.set_active(True)
    assert f.calls == [("active", 0)]
    w.rows[1].overlay.set_active(True)
    w.rows[1].overlay.set_active(False)
    w.rows[0].remove.emit("clicked")
    w.clear_button.emit("clicked")
    assert f.calls[1:] == [("overlay", 1, True), ("overlay", 1, False), ("remove", 0), ("clear",)]
    f.calls.clear()
    s.set_active(1)
    s.set_overlay(2, True)
    w.refresh(s)
    assert f.calls == []
    assert [r.active.get_active() for r in w.rows] == [False, True, False]
    assert [r.overlay.get_active() for r in w.rows] == [False, False, True]


def test_refresh_inside_own_handler_and_empty():
    s = session()
    holder = {}

    def on_remove(key):
        s.remove(key)
        holder["w"].refresh(s)  # rebuild from inside the clicked button's handler

    w = ResultsList(s, lambda k: None, lambda k, on: None, on_remove, lambda: None)
    holder["w"] = w
    w.rows[2].remove.emit("clicked")
    assert [r.key for r in w.rows] == [0, 1]
    assert w.rows[0].active.get_active()  # first remaining entry became active
    s.clear()
    w.refresh(s)
    assert w.rows == [] and w.list.get_row_at_index(0) is None
    assert not w.button.get_sensitive() and not w.clear_button.get_sensitive()


def edit_label(row, text, commit=True):
    row.label.start_editing()
    row.label.set_text(text)
    row.label.stop_editing(commit)


def test_rename_calls_back():
    s = session()
    f = Fixture(s)
    w = f.widget
    edit_label(w.rows[2], "s2m1")
    assert f.calls == [("rename", 2, "s2m1")]
    edit_label(w.rows[0], "s1m1")  # unchanged text: nothing
    edit_label(w.rows[1], "other", commit=False)  # cancelled: text restored, nothing
    assert f.calls == [("rename", 2, "s2m1")] and w.rows[1].label.get_text() == "s1m2"
    s.rename(2, "s2m1")
    w.refresh(s)
    assert w.rows[2].name.get_visible() and w.rows[2].label.get_text() == "s2m1"
    assert f.calls == [("rename", 2, "s2m1")]


def test_labels_not_editable_without_callback():
    w = ResultsList(session(), lambda k: None, lambda k, on: None, lambda k: None, lambda: None)
    assert not any(r.label.get_editable() for r in w.rows)
