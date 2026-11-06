"""Session list of results: one entry per channel of every measurement or opened file.

User decision 2026-11-04 (D-19): every measurement or opened file adds its channels to
a list; one entry is *active* (impulse-response tab, band table, export of the non-IR
formats), any other entries can be ticked to *overlay* in the transfer-function and
reverberation-time tabs. Entries keep their colour when others are removed. No GTK
here, so the rules are tested without a display; ``gui/plugin.py`` builds the widgets.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from resonus_core.data import FrequencyResponse, ImpulseResponse
from resonus_core.style import TOKENS, TRACE_COLORS

from resonus_room.core.analysis import RoomAnalysis

# Design choice (D-19): the seven trace colours of the shared palette for the entries,
# in order of creation; the room average uses the neutral colour Dark 3 (resonus-core
# 0.10.0 moved it out of the trace colours into the token `neutral_color`, core D-T1).
ENTRY_COLORS = TRACE_COLORS
AVERAGE_COLOR = str(TOKENS["neutral_color"])
MEASUREMENT_NAME = "Measurement"
# Dummy-head ear label s#m#L / s#m#R (user decisions D-17, Q-M10): group 1 the position,
# group 2 the ear.
EAR_LABEL = re.compile(r"(s[1-9][0-9]*m[1-9][0-9]*)([LR])")


@dataclass
class Entry:
    """One channel of one measurement or file.

    Parameters
    ----------
    key
        Unique number, in order of creation (never reused within a session).
    name
        Name of the measurement or file (``"Measurement 2"``, ``"bh.wav"``), unique among
        the measurements and files of the session; shared by the channels of one run.
    label
        Channel label (``s#m#``) or, for a file without channel metadata, the file name.
    ir, fr
        Impulse response and transfer function (``fr`` None if not computed).
    color
        ``ENTRY_COLORS[key % len(ENTRY_COLORS)]``.
    analysis
        Cached analysis with the current settings, None if not computed (or invalidated).
    """

    key: int
    name: str
    label: str
    ir: ImpulseResponse
    fr: FrequencyResponse | None
    color: str
    analysis: RoomAnalysis | None = None

    @property
    def title(self) -> str:
        """Text in lists and legends: ``label`` if equal to ``name``, else ``f"{name} {label}"``."""
        return self.label if self.label == self.name else f"{self.name} {self.label}"


class Session:
    """Ordered list of entries with one active entry and a set of overlaid entries.

    Attributes
    ----------
    entries : list of Entry
        In order of creation.
    active : int or None
        Key of the active entry; None only when ``entries`` is empty.
    overlay : set of int
        Keys of the ticked entries; never contains ``active``.
    """

    def __init__(self) -> None:
        self.entries: list[Entry] = []
        self.active: int | None = None
        self.overlay: set[int] = set()
        self._next_key = 0
        self._measurements = 0

    def next_measurement_name(self) -> str:
        """Return ``f"{MEASUREMENT_NAME} {n}"``, n = 1, 2, ... (one more on every call)."""
        self._measurements += 1
        return f"{MEASUREMENT_NAME} {self._measurements}"

    def unique_name(self, name: str) -> str:
        """``name`` if no entry has it, else ``f"{name} ({n})"`` with the smallest n >= 2 unused.

        Example: opening ``bh.wav`` twice gives ``"bh.wav"`` and ``"bh.wav (2)"``.
        """
        existing = {e.name for e in self.entries}
        if name not in existing:
            return name
        n = 2
        while f"{name} ({n})" in existing:
            n += 1
        return f"{name} ({n})"

    def add(
        self, name: str, channels: Sequence[tuple[str, ImpulseResponse, FrequencyResponse | None]]
    ) -> list[Entry]:
        """Append one entry per ``(label, ir, fr)``, all with ``self.unique_name(name)``.

        Keys from the internal counter (``self._next_key``, then incremented), colour
        ``ENTRY_COLORS[key % len(ENTRY_COLORS)]``. The first new entry becomes active;
        the overlay set is kept.

        Returns
        -------
        list of Entry
            The new entries, in the order of ``channels``.

        Raises
        ------
        ValueError
            ``"no channels"`` if ``channels`` is empty (nothing is changed).
        """
        if not channels:
            raise ValueError("no channels")
        name = self.unique_name(name)
        new_entries: list[Entry] = []
        for label, ir, fr in channels:
            key = self._next_key
            self._next_key += 1
            color = ENTRY_COLORS[key % len(ENTRY_COLORS)]
            entry = Entry(key, name, label, ir, fr, color)
            new_entries.append(entry)
            self.entries.append(entry)
        self.set_active(new_entries[0].key)
        return new_entries

    def get(self, key: int) -> Entry:
        """Return the entry with ``key``.

        Raises
        ------
        KeyError
            If there is none.
        """
        for e in self.entries:
            if e.key == key:
                return e
        raise KeyError(key)

    @property
    def active_entry(self) -> Entry | None:
        """The active entry, or None."""
        if self.active is None:
            return None
        return self.get(self.active)

    def set_active(self, key: int) -> None:
        """Make ``key`` the active entry and remove it from ``overlay``.

        Raises
        ------
        KeyError
            For an unknown key (nothing is changed).
        """
        self.get(key)  # validate existence
        self.active = key
        self.overlay.discard(key)

    def set_overlay(self, key: int, on: bool) -> None:
        """Tick (``on``) or untick an entry; the active entry is ignored (always shown).

        Raises
        ------
        KeyError
            For an unknown key.
        """
        self.get(key)  # validate existence
        if key == self.active:
            return
        if on:
            self.overlay.add(key)
        else:
            self.overlay.discard(key)

    def remove(self, key: int) -> None:
        """Remove an entry (also from ``overlay``).

        If it was active, the first remaining entry becomes active (it leaves ``overlay``),
        or ``active = None`` when the list is empty.

        Raises
        ------
        KeyError
            For an unknown key.
        """
        self.get(key)  # validate existence
        self.entries = [e for e in self.entries if e.key != key]
        self.overlay.discard(key)
        if key == self.active:
            self.active = None
            if self.entries:
                self.set_active(self.entries[0].key)

    def rename(self, key: int, label: str) -> None:
        """Give an entry a new channel label (results list, D-19).

        ``label`` is stripped of surrounding white space. Only ``Entry.label`` changes:
        the impulse response and its metadata (the measurement record, e.g.
        ``extra["channel"]["label"]``) stay as measured; the room average and the export
        file names use the new label.

        Raises
        ------
        KeyError
            For an unknown key.
        ValueError
            ``"empty label"``; ``f"label {label!r} contains a path separator"`` if it
            contains a slash or a backslash (labels become parts of export file names);
            ``f"duplicate label {label!r} in {entry.name}"`` if another entry with the
            same ``name`` has this label. Nothing is changed on an error.
        """
        entry = self.get(key)
        label = label.strip()
        if not label:
            raise ValueError("empty label")
        if "/" in label or "\\" in label:
            raise ValueError(f"label {label!r} contains a path separator")
        if any(e.label == label for e in self.group(entry.name) if e.key != key):
            raise ValueError(f"duplicate label {label!r} in {entry.name}")
        entry.label = label

    def clear(self) -> None:
        """Remove all entries (``active = None``, ``overlay`` empty); keys are not reused."""
        self.entries.clear()
        self.active = None
        self.overlay.clear()

    def shown(self) -> list[Entry]:
        """Return the active entry first, then the overlaid ones in list order; [] if empty."""
        active = self.active_entry
        if active is None:
            return []
        overlay_entries = [e for e in self.entries if e.key in self.overlay]
        return [active] + overlay_entries

    def group(self, name: str) -> list[Entry]:
        """Return the entries of one measurement or file (same ``name``), in list order."""
        return [e for e in self.entries if e.name == name]

    def invalidate(self) -> None:
        """Drop the cached analyses (``analysis = None`` on every entry)."""
        for e in self.entries:
            e.analysis = None


def ear_partner(entry: Entry, entries: Sequence[Entry]) -> Entry | None:
    """Return the other ear of ``entry`` (IACC tab, D-28), or None.

    ``m = EAR_LABEL.fullmatch(entry.label)``; None if no match. ``other = m.group(1) +
    ("R" if m.group(2) == "L" else "L")``. The partner is the first ``e`` of ``entries``
    with ``e.key != entry.key`` and ``e.label == other`` and either ``e.name ==
    entry.name`` (two channels of one measurement) or ``e.name ==
    entry.name.replace(entry.label, other)`` (two opened files such as ``"s1m1L.wav"`` /
    ``"s1m1R.wav"`` labelled ``"s1m1L"`` / ``"s1m1R"``). None if there is no such entry.
    """
    m = EAR_LABEL.fullmatch(entry.label)
    if not m:
        return None
    base = m.group(1)
    side = m.group(2)
    other_side = "R" if side == "L" else "L"
    other_label = base + other_side
    for e in entries:
        if e.key == entry.key:
            continue
        if e.label != other_label:
            continue
        if e.name == entry.name or e.name == entry.name.replace(entry.label, other_label):
            return e
    return None


def ear_pairs(shown: Sequence[Entry], entries: Sequence[Entry]) -> list[tuple[Entry, Entry]]:
    """Ear pairs ``(left, right)`` of the shown entries, in the order of ``shown``.

    For each ``e`` in ``shown`` with ``p = ear_partner(e, entries)`` not None: the pair is
    ``(e, p)`` if ``e``'s label ends with ``"L"``, else ``(p, e)``; a pair already in the
    list (same two keys) is not added again. So the partner of a shown entry needs not be
    shown itself.
    """
    pairs: list[tuple[Entry, Entry]] = []
    seen: set[tuple[int, int]] = set()
    for e in shown:
        p = ear_partner(e, entries)
        if p is None:
            continue
        key_pair = (e.key, p.key)
        if key_pair in seen or (p.key, e.key) in seen:
            continue
        if e.label.endswith("L"):
            pair = (e, p)
        else:
            pair = (p, e)
        pairs.append(pair)
        seen.add((pair[0].key, pair[1].key))
    return pairs
