"""Results list of the header: active and overlaid session entries (GTK), D-19.

A menu button opens a popover with one row per session entry (radio button: active
entry; colour swatch, measurement name and the channel label, editable; check button:
overlay; cross: remove) and a "Clear all" button. The rules live in
:mod:`resonus_room.core.session`; this widget only shows a :class:`Session` and reports
the user's clicks through callbacks. The owner changes the session and calls
:meth:`ResultsList.refresh`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from gi.repository import Gtk

from resonus_room.core.session import Session

SWATCH = "●"  # filled circle in the entry's colour


@dataclass
class ResultRow:
    """Widgets of one entry.

    Attributes
    ----------
    key : int
        The entry's key.
    box : Gtk.Box
        Horizontal box appended to ``ResultsList.list``, children in this order:
        ``active``, ``swatch``, ``name``, ``label``, ``overlay``, ``remove``.
    active : Gtk.CheckButton
        Radio button (all rows in one group: ``set_group`` with the first row's);
        active iff the entry is the session's active entry; tooltip "Active".
    swatch : Gtk.Label
        Markup ``f'<span foreground="{entry.color}">{SWATCH}</span>'``.
    name : Gtk.Label
        ``entry.name``, ``xalign=0``; visible only if ``entry.name != entry.label`` (so
        the row reads like ``entry.title``).
    label : Gtk.EditableLabel
        ``text=entry.label``, ``hexpand=True``, tooltip "Channel label (click to
        rename)".
    overlay : Gtk.CheckButton
        Active iff the key is in ``session.overlay``; insensitive on the active entry;
        tooltip "Overlay".
    remove : Gtk.Button
        ``icon_name="window-close-symbolic"``, CSS class "flat", tooltip "Remove".
    """

    key: int
    box: Gtk.Box
    active: Gtk.CheckButton
    swatch: Gtk.Label
    name: Gtk.Label
    label: Gtk.EditableLabel
    overlay: Gtk.CheckButton
    remove: Gtk.Button


class ResultsList:
    """Header menu button with the list of session entries.

    Widgets: ``button`` (``Gtk.MenuButton(icon_name="view-list-symbolic")``, tooltip
    "Results: active and overlaid"), its popover holding a vertical box with ``list``
    (``Gtk.ListBox``, selection mode NONE) and ``clear_button`` (``Gtk.Button(label=
    "Clear all")``).

    :meth:`refresh` rebuilds the rows. Clicks call (never while refreshing):

    - radio of row ``key`` toggled on -> ``on_active(key)``;
    - overlay check toggled -> ``on_overlay(key, check.get_active())``;
    - remove button -> ``on_remove(key)``;
    - label edited (``notify::editing`` with ``get_editing()`` False) and its text differs
      from ``entry.label`` -> ``on_rename(key, label.get_text())`` (if ``on_rename`` is
      given); a cancelled edit restores the text and calls nothing;
    - ``clear_button`` -> ``on_clear()``.

    Parameters
    ----------
    session
        Shown at construction (:meth:`refresh`).
    on_active, on_overlay, on_remove, on_clear
        Callbacks.
    on_rename
        Called with the key and the new label text; None: labels are not editable
        (``label.set_editable(False)``).

    Attributes
    ----------
    button : Gtk.MenuButton
    list : Gtk.ListBox
    clear_button : Gtk.Button
    rows : list of ResultRow
        One per entry, in the order of ``session.entries``.
    """

    def __init__(
        self,
        session: Session,
        on_active: Callable[[int], None],
        on_overlay: Callable[[int, bool], None],
        on_remove: Callable[[int], None],
        on_clear: Callable[[], None],
        on_rename: Callable[[int, str], None] | None = None,
    ) -> None:
        self._on_active = on_active
        self._on_overlay = on_overlay
        self._on_remove = on_remove
        self._on_clear = on_clear
        self._on_rename = on_rename
        self._updating = False
        self._retired: list[ResultRow] = []
        self.rows: list[ResultRow] = []
        self.button = Gtk.MenuButton(icon_name="view-list-symbolic")
        self.button.set_tooltip_text("Results: active and overlaid")
        self.list = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE)
        self.clear_button = Gtk.Button(label="Clear all")
        self.clear_button.connect("clicked", lambda _b: self._clicked_clear())
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        box.append(self.list)
        box.append(self.clear_button)
        popover = Gtk.Popover()
        popover.set_child(box)
        self.button.set_popover(popover)
        self.refresh(session)

    def _toggled_active(self, key: int, button: Gtk.CheckButton) -> None:
        if not self._updating and button.get_active():
            self._on_active(key)

    def _toggled_overlay(self, key: int, button: Gtk.CheckButton) -> None:
        if not self._updating:
            self._on_overlay(key, button.get_active())

    def _clicked_remove(self, key: int) -> None:
        if not self._updating:
            self._on_remove(key)

    def _edited(self, key: int, old: str, label: Gtk.EditableLabel) -> None:
        if self._updating:
            return
        if label.get_editing():
            return
        if self._on_rename is None:
            return
        if label.get_text() != old:
            self._on_rename(key, label.get_text())

    def refresh(self, session: Session) -> None:
        """Rebuild the rows from ``session`` without calling any callback.

        Set ``self._updating = True`` (``try``/``finally``); keep the old rows in
        ``self._retired`` (refresh may run inside a handler of one of their widgets);
        ``self.list.remove_all()``; one :class:`ResultRow` per entry (see its attributes),
        each ``box`` appended to ``self.list``. ``button`` and ``clear_button`` are
        sensitive iff ``session.entries`` is not empty.
        """
        self._updating = True
        try:
            self._retired = self.rows
            self.rows = []
            self.list.remove_all()
            first_active = None
            for e in session.entries:
                active = Gtk.CheckButton(active=e.key == session.active)
                active.set_tooltip_text("Active")
                if first_active is None:
                    first_active = active
                else:
                    active.set_group(first_active)
                swatch = Gtk.Label()
                swatch.set_markup(f'<span foreground="{e.color}">{SWATCH}</span>')
                name = Gtk.Label(label=e.name, xalign=0, visible=e.name != e.label)
                label = Gtk.EditableLabel()
                label.set_text(e.label)
                label.set_hexpand(True)
                label.set_tooltip_text("Channel label (click to rename)")
                label.set_editable(self._on_rename is not None)
                overlay = Gtk.CheckButton(active=e.key in session.overlay)
                overlay.set_tooltip_text("Overlay")
                overlay.set_sensitive(e.key != session.active)
                remove = Gtk.Button(icon_name="window-close-symbolic")
                remove.add_css_class("flat")
                remove.set_tooltip_text("Remove")
                box = Gtk.Box(spacing=6)
                box.append(active)
                box.append(swatch)
                box.append(name)
                box.append(label)
                box.append(overlay)
                box.append(remove)
                self.list.append(box)
                self.rows.append(
                    ResultRow(e.key, box, active, swatch, name, label, overlay, remove)
                )
                active.connect("toggled", lambda b, k=e.key: self._toggled_active(k, b))
                overlay.connect("toggled", lambda b, k=e.key: self._toggled_overlay(k, b))
                remove.connect("clicked", lambda _b, k=e.key: self._clicked_remove(k))
                label.connect(
                    "notify::editing", lambda w, _p, k=e.key, old=e.label: self._edited(k, old, w)
                )
            self.button.set_sensitive(bool(session.entries))
            self.clear_button.set_sensitive(bool(session.entries))
        finally:
            self._updating = False

    def _clicked_clear(self) -> None:
        if not self._updating:
            self._on_clear()
