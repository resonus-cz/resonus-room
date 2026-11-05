"""Band-by-band comparison of our results with a reference tool (REW), docs/VERIFICATION.md.

Decay times (EDT, T20, T30) are compared relatively: ``100 (ours / reference - 1)`` in
percent, passing at ``RELATIVE_TOLERANCE`` (user decision 2026-11-03, docs/DECISIONS.md
D-11). Energy parameters (C50, C80, D50, Ts) are reported as differences ``ours -
reference`` in their unit; C80 passes at ``ABSOLUTE_TOLERANCES["C80"]`` (user decision,
D-14), the others have no pass/fail criterion.
"""

from __future__ import annotations

import csv  # noqa: F401
import io  # noqa: F401
import math  # noqa: F401
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from resonus_core.data import BandResults

# User decision D-11 (D-11): cross-tool comparisons of decay times pass at |dT|/T <= 2 %.
RELATIVE_TOLERANCE = 0.02
# User decision (D-14): C80 passes at |ours - reference| <= 0.5 dB.
ABSOLUTE_TOLERANCES = {"C80": 0.5}
TIME_QUANTITIES = ("EDT", "T20", "T30")
ENERGY_QUANTITIES = ("C50", "C80", "D50", "Ts")
COMPARISON_COLUMNS = (
    "band", "quantity", "ours", "reference", "difference", "difference_unit", "within",
)  # fmt: skip


@dataclass(frozen=True)
class Comparison:
    """Comparison of one quantity over the common bands.

    Parameters
    ----------
    quantity
        Quantity name, e.g. ``"T20"``.
    labels
        Labels of the compared bands (in the order of ``ours``).
    ours, reference
        Values per band.
    difference
        Relative difference in percent for decay times, ``ours - reference`` otherwise.
    difference_unit
        ``"%"`` for decay times, else the unit of the quantity in ``ours``.
    within
        Decay times: ``abs(ours / reference - 1) <= RELATIVE_TOLERANCE`` per band;
        quantities in ``ABSOLUTE_TOLERANCES``: ``abs(ours - reference) <= tolerance``
        (False where either value is NaN); None for the other quantities.
    """

    quantity: str
    labels: tuple[str, ...]
    ours: NDArray[np.float64]
    reference: NDArray[np.float64]
    difference: NDArray[np.float64]
    difference_unit: str
    within: NDArray[np.bool_] | None


def compare(
    ours: BandResults,
    reference: BandResults,
    quantities: tuple[str, ...] = TIME_QUANTITIES + ENERGY_QUANTITIES,
) -> list[Comparison]:
    """Compare two band results over their common bands.

    Bands are matched by label; the common bands keep the order of ``ours``. Quantities
    missing in either result are skipped. For each remaining quantity (in the order of
    ``quantities``) one :class:`Comparison` (see there for the difference and ``within``).

    Raises
    ------
    ValueError
        If the fractions differ or there is no common band.
    """
    if ours.fraction != reference.fraction:
        raise ValueError("fraction mismatch")
    ref_index = {label: i for i, label in enumerate(reference.labels)}
    common = [(i, ref_index[lab]) for i, lab in enumerate(ours.labels) if lab in ref_index]
    if not common:
        raise ValueError("no common bands")
    io_ = np.array([c[0] for c in common])
    ir_ = np.array([c[1] for c in common])
    labels = tuple(ours.labels[i] for i in io_)
    comps: list[Comparison] = []
    for q in quantities:
        if q in ours.values and q in reference.values:
            a = ours.values[q][io_]
            b = reference.values[q][ir_]
            if q in TIME_QUANTITIES:
                with np.errstate(invalid="ignore", divide="ignore"):
                    rel = a / b - 1
                difference = 100 * rel
                unit = "%"
                within = np.abs(rel) <= RELATIVE_TOLERANCE
            else:
                difference = a - b
                unit = ours.units[q]
                tol = ABSOLUTE_TOLERANCES.get(q)
                with np.errstate(invalid="ignore"):
                    within = None if tol is None else np.abs(difference) <= tol
            comps.append(Comparison(q, labels, a, b, difference, unit, within))
    return comps


def comparisons_to_csv(comparisons: list[Comparison]) -> str:
    """CSV text: header ``COMPARISON_COLUMNS``, one row per quantity and band.

    Numbers with 4 significant digits (``f"{x:.4g}"``), NaN as empty field; ``within``
    as ``"yes"``/``"no"``, empty for quantities without tolerance.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(COMPARISON_COLUMNS)

    def fmt(x: float) -> str:
        return "" if not math.isfinite(x) else f"{x:.4g}"

    for c in comparisons:
        for k in range(len(c.labels)):
            ours_val = fmt(c.ours[k])
            ref_val = fmt(c.reference[k])
            diff_val = fmt(c.difference[k])
            within_str = "" if c.within is None else ("yes" if c.within[k] else "no")
            writer.writerow(
                [
                    c.labels[k],
                    c.quantity,
                    ours_val,
                    ref_val,
                    diff_val,
                    c.difference_unit,
                    within_str,
                ]
            )
    return buf.getvalue()


def summary(comparisons: list[Comparison]) -> list[str]:
    """One line per comparison with a criterion: ``"{q}: {n_within}/{n} bands within {tol}"``.

    ``tol`` is ``RELATIVE_TOLERANCE`` in percent (``"2 %"``) for decay times, else the
    absolute tolerance with the unit (``"0.5 dB"``), numbers formatted with ``:g``.
    """
    lines: list[str] = []
    for c in comparisons:
        if c.within is not None:
            n_within = int(np.sum(c.within))
            n = c.within.size
            if c.quantity in ABSOLUTE_TOLERANCES:
                tol = f"{ABSOLUTE_TOLERANCES[c.quantity]:g} {c.difference_unit}"
            else:
                tol = f"{100 * RELATIVE_TOLERANCE:g} %"
            lines.append(f"{c.quantity}: {n_within}/{n} bands within {tol}")
    return lines
