"""Sound absorption in a reverberation room (ISO 354:2003).

Equivalent sound absorption areas from reverberation times (Sabine with air absorption,
8.1.2), absorption coefficient (8.1.3), area per object (8.1.4), rounding (8.3), the
octave values of 8.3 and the checks of clauses 6 and 7. Interface after sapyne
``experimental_absorption.py`` (https://github.com/vyhyb/sapyne, GPL-3.0, not copied;
docs/DECISIONS.md D-4).

References
----------
ISO 354:2003, Acoustics - Measurement of sound absorption in a reverberation room.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from resonus_core.air import power_attenuation_coefficient, speed_of_sound  # noqa: F401
from resonus_core.bands import band_set  # noqa: F401

# ISO 354:2003 Eqs. (5), (7), (8): A = 55.3 V / (c T) - 4 V m.
SABINE_CONSTANT = 55.3
# ISO 354:2003 clause 5: nominal one-third-octave centre frequencies of the measurement.
ISO354_FREQUENCIES = (
    100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0,
    1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0,
)  # fmt: skip
# ISO 354:2003 6.1.4 Table 1: maximum A1 in m2 for V = 200 m3, bands of ISO354_FREQUENCIES.
MAX_EMPTY_ROOM_AREA_200 = (
    6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 7.0, 7.5, 8.0, 9.5, 10.5, 12.0, 13.0, 14.0,
)  # fmt: skip
# ISO 354:2003 6.1.4: reference volume of Table 1 and of the scaling (V/200 m3)**(2/3).
REFERENCE_VOLUME = 200.0
# ISO 354:2003 6.1.1: minimum room volume in m3.
MIN_VOLUME = 150.0
# ISO 354:2003 6.1.2 Eq. (1): l_max < 1.9 V**(1/3).
MAX_LENGTH_FACTOR = 1.9
# ISO 354:2003 6.1.4: no dips or peaks differing by more than 15 % from the mean of the
# two adjacent bands.
SMOOTHNESS_LIMIT = 0.15
# ISO 354:2003 6.2.1.1: plane-absorber area 10...12 m2 (upper limit scaled for V > 200 m3).
SPECIMEN_AREA = (10.0, 12.0)
# ISO 354:2003 6.2.2.3: discrete objects change A by more than 1 m2 and at most 12 m2
# (both scaled for V > 200 m3).
OBJECT_AREA_CHANGE = (1.0, 12.0)
# ISO 354:2003 6.3.2: relative humidity 30...90 %, temperature at least 15 degC.
HUMIDITY_RANGE = (30.0, 90.0)
MIN_TEMPERATURE = 15.0
# ISO 354:2003 7.1.4: at least 12 decay curves (6 with simultaneous sources), 3 microphone
# positions and 2 source positions.
MIN_DECAYS = 12
MIN_DECAYS_SIMULTANEOUS = 6
MIN_MICROPHONES = 3
MIN_SOURCES = 2
# ISO 354:2003 8.3: A_T rounded to 0.1 m2, alpha_s to 0.01.
AREA_RESOLUTION = 0.1
ALPHA_RESOLUTION = 0.01


@dataclass(frozen=True)
class Climate:
    """Air conditions during one reverberation-time measurement.

    Parameters
    ----------
    temperature
        Air temperature in degC.
    relative_humidity
        Relative humidity in percent.
    pressure
        Atmospheric pressure in kPa.
    """

    temperature: float
    relative_humidity: float
    pressure: float = 101.325


@dataclass(frozen=True)
class AbsorptionResult:
    """ISO 354 result per one-third-octave band (``ISO354_FREQUENCIES``).

    Parameters
    ----------
    frequency
        Nominal centre frequencies in Hz.
    t1, t2
        Mean reverberation times without / with specimen in s.
    a1, a2
        Equivalent sound absorption areas of the room in m2, Eqs. (5), (7).
    area
        Equivalent sound absorption area of the specimen ``A_T`` in m2, Eq. (8).
    alpha
        Sound absorption coefficient ``alpha_s`` (plane absorber or array; NaN array if
        no specimen area was given), Eq. (9).
    area_per_object
        ``A_obj = A_T / n`` in m2 (NaN array if no object count was given), 8.1.4.
    alpha_std, area_std
        Standard uncertainties propagated from the T standard deviations (NaN arrays if
        none were given).
    issues
        Findings of the clause 6/7 checks (empty: none).
    """

    frequency: NDArray[np.float64]
    t1: NDArray[np.float64]
    t2: NDArray[np.float64]
    a1: NDArray[np.float64]
    a2: NDArray[np.float64]
    area: NDArray[np.float64]
    alpha: NDArray[np.float64]
    area_per_object: NDArray[np.float64]
    alpha_std: NDArray[np.float64]
    area_std: NDArray[np.float64]
    issues: tuple[str, ...]


def equivalent_absorption_area(
    decay_time: ArrayLike, volume: float, speed: float, m: ArrayLike = 0.0
) -> NDArray[np.float64]:
    """Equivalent sound absorption area ``A = 55.3 V / (c T) - 4 V m`` in m2.

    Parameters
    ----------
    decay_time
        Reverberation time ``T`` in s, > 0.
    volume
        Room volume ``V`` in m3, > 0.
    speed
        Speed of sound ``c`` in m/s, > 0.
    m
        Power attenuation coefficient in 1/m, >= 0.

    Raises
    ------
    ValueError
        If ``T``, ``V`` or ``c`` are not positive or ``m`` is negative.

    References
    ----------
    ISO 354:2003, 8.1.2.1 Eq. (5), 8.1.2.2 Eq. (7).
    """
    t = np.asarray(decay_time, dtype=float)
    mm = np.asarray(m, dtype=float)
    if np.any(t <= 0) or volume <= 0 or speed <= 0:
        raise ValueError("Decay time, volume and speed must be positive.")
    if np.any(mm < 0):
        raise ValueError("Power attenuation coefficient must be non-negative.")
    return SABINE_CONSTANT * volume / (speed * t) - 4 * volume * mm


def volume_factor(volume: float) -> float:
    """Scaling factor ``(V / 200 m3)**(2/3)`` of ISO 354:2003 6.1.4, 6.2.1.1, 6.2.2.3."""
    if volume <= 0:
        raise ValueError("Volume must be positive.")
    return float((volume / REFERENCE_VOLUME) ** (2 / 3))


def max_empty_room_area(volume: float) -> NDArray[np.float64]:
    """Maximum ``A1`` per band: Table 1 times ``volume_factor(volume)`` (ISO 354:2003 6.1.4)."""
    return np.asarray(MAX_EMPTY_ROOM_AREA_200) * volume_factor(volume)


def octave_values(third_octave: ArrayLike) -> NDArray[np.float64]:
    """Octave values 125 Hz...4 kHz as arithmetic means of three one-third-octave values.

    ``third_octave`` holds 18 values for ``ISO354_FREQUENCIES``; the octave value of band
    ``k`` (125, 250, ..., 4000 Hz) is the mean of thirds ``3k ... 3k+2`` (100/125/160 Hz
    for 125 Hz, ...). ISO 354:2003 8.3 (with reference to ISO 11654).

    Raises
    ------
    ValueError
        If not exactly 18 values are given.
    """
    arr = np.asarray(third_octave, dtype=float)
    if arr.shape != (18,):
        raise ValueError("Exactly 18 values required.")
    return arr.reshape(6, 3).mean(axis=1)


def round_to(values: ArrayLike, resolution: float) -> NDArray[np.float64]:
    """Round to a multiple of ``resolution`` (``np.round(v / r) * r``; ties to even).

    ISO 354:2003 8.3 requires rounding of A_T to 0.1 m2 and alpha_s to 0.01 without
    specifying ties.
    """
    if resolution <= 0:
        raise ValueError("Resolution must be positive.")
    return np.round(np.asarray(values, dtype=float) / resolution) * resolution


def check_room(volume: float, max_length: float) -> list[str]:
    """Check the reverberation room against ISO 354:2003 6.1.1 and 6.1.2.

    Issues: ``"6.1.1: volume ... < 150 m3"`` if ``volume < MIN_VOLUME``;
    ``"6.1.2: l_max ... >= 1.9 V^(1/3) = ..."`` if ``max_length >= 1.9 * volume**(1/3)``.
    Each issue string starts with the clause number.
    """
    issues = []
    if volume < MIN_VOLUME:
        issues.append(f"6.1.1: volume {volume:g} m3 < {MIN_VOLUME:g} m3")
    limit = MAX_LENGTH_FACTOR * volume ** (1 / 3)
    if max_length >= limit:
        issues.append(f"6.1.2: l_max {max_length:g} m >= 1.9 V^(1/3) = {limit:.2f} m")
    return issues


def check_empty_room(a1: ArrayLike, volume: float) -> list[str]:
    """Check ``A1`` (18 bands) against ISO 354:2003 6.1.4.

    - For each band with ``a1 > max_empty_room_area(volume)``:
      ``"6.1.4: A1 ... m2 > ... m2 at <f> Hz"``.
    - For each inner band ``i = 1...16``: with ``ref = (a1[i-1] + a1[i+1]) / 2``, if
      ``abs(a1[i] - ref) > SMOOTHNESS_LIMIT * ref``: ``"6.1.4: A1 at <f> Hz deviates ...
      from its neighbours"``.

    Raises
    ------
    ValueError
        If not exactly 18 values are given.
    """
    a = np.asarray(a1, dtype=float)
    if a.shape != (18,):
        raise ValueError("Exactly 18 values required.")
    issues = []
    for f, val, lim in zip(ISO354_FREQUENCIES, a, max_empty_room_area(volume), strict=True):
        if val > lim:
            issues.append(f"6.1.4: A1 {val:.2f} m2 > {lim:.2f} m2 at {f:g} Hz")
    for i in range(1, 17):
        ref = (a[i - 1] + a[i + 1]) / 2
        if abs(a[i] - ref) > SMOOTHNESS_LIMIT * ref:
            dev = 100 * (a[i] / ref - 1)
            issues.append(
                f"6.1.4: A1 at {ISO354_FREQUENCIES[i]:g} Hz deviates {dev:+.0f} % "
                "from its neighbours"
            )
    return issues


def check_specimen(
    volume: float, area: float | None = None, area_change: ArrayLike | None = None
) -> list[str]:
    """Check the specimen size against ISO 354:2003 6.2.1.1 (plane) and 6.2.2.3 (objects).

    - ``area`` (plane absorber, m2): issue ``"6.2.1.1: ..."`` unless
      ``10 <= area <= 12 * k`` with ``k = volume_factor(volume)`` if ``volume > 200`` else 1.
    - ``area_change`` (``A_T`` per band of discrete objects, m2): issue ``"6.2.2.3: ..."``
      for each band outside ``(1 * k, 12 * k]`` with the same ``k``.
    """
    k = volume_factor(volume) if volume > REFERENCE_VOLUME else 1.0
    issues = []
    if area is not None:
        lo, hi = SPECIMEN_AREA[0], SPECIMEN_AREA[1] * k
        if not (lo <= area <= hi):
            issues.append(f"6.2.1.1: specimen area {area:g} m2 outside {lo:g}...{hi:.2f} m2")
    if area_change is not None:
        lo, hi = OBJECT_AREA_CHANGE[0] * k, OBJECT_AREA_CHANGE[1] * k
        for f, val in zip(ISO354_FREQUENCIES, np.asarray(area_change, dtype=float), strict=True):
            if not (lo < val <= hi):
                issues.append(
                    f"6.2.2.3: A_T {val:.2f} m2 outside ({lo:.2f}, {hi:.2f}] m2 at {f:g} Hz"
                )
    return issues


def check_climate(climate: Climate) -> list[str]:
    """Check air conditions against ISO 354:2003 6.3.2 (RH 30...90 %, t >= 15 degC)."""
    issues = []
    lo, hi = HUMIDITY_RANGE
    if not (lo <= climate.relative_humidity <= hi):
        issues.append(
            f"6.3.2: relative humidity {climate.relative_humidity:g} % outside {lo:g}...{hi:g} %"
        )
    if climate.temperature < MIN_TEMPERATURE:
        issues.append(f"6.3.2: temperature {climate.temperature:g} degC < {MIN_TEMPERATURE:g} degC")
    return issues


def check_positions(n_microphones: int, n_sources: int, simultaneous: bool = False) -> list[str]:
    """Check the number of positions against ISO 354:2003 7.1.4.

    Issues (clause ``"7.1.4: ..."``) if ``n_microphones * n_sources`` is below
    ``MIN_DECAYS_SIMULTANEOUS if simultaneous else MIN_DECAYS``, if ``n_microphones <
    MIN_MICROPHONES``, or if ``n_sources < MIN_SOURCES``.
    """
    issues = []
    need = MIN_DECAYS_SIMULTANEOUS if simultaneous else MIN_DECAYS
    if n_microphones * n_sources < need:
        issues.append(f"7.1.4: {n_microphones * n_sources} decay curves < {need}")
    if n_microphones < MIN_MICROPHONES:
        issues.append(f"7.1.4: {n_microphones} microphone positions < {MIN_MICROPHONES}")
    if n_sources < MIN_SOURCES:
        issues.append(f"7.1.4: {n_sources} source positions < {MIN_SOURCES}")
    return issues


def absorption(
    t1: ArrayLike,
    t2: ArrayLike,
    volume: float,
    climate1: Climate,
    climate2: Climate,
    specimen_area: float | None = None,
    n_objects: int | None = None,
    t1_std: ArrayLike | None = None,
    t2_std: ArrayLike | None = None,
) -> AbsorptionResult:
    """ISO 354 evaluation for the 18 one-third-octave bands of ``ISO354_FREQUENCIES``.

    1. ``fm = band_set(3, 100, 5000).center`` (exact mid-band frequencies, ISO 9613-1:1993
       8.2.1); ``c_i = speed_of_sound(climate_i.temperature)``;
       ``m_i = power_attenuation_coefficient(fm, t_i, rh_i, p_i)`` (``m2`` from the
       conditions during the T2 measurement, D-9, confirmed by the user 2026-11-03).
    2. ``a1 = equivalent_absorption_area(t1, volume, c1, m1)``, ``a2`` likewise;
       ``area = a2 - a1`` (Eq. 8); ``alpha = area / specimen_area`` (Eq. 9) or NaN;
       ``area_per_object = area / n_objects`` (8.1.4) or NaN.
    3. Uncertainty (D-11, confirmed by the user; first-order propagation as in sapyne): if both
       ``t1_std`` and ``t2_std`` are given, ``s_ai = 55.3 V / c_i * t_std_i / t_i**2``,
       ``area_std = sqrt(s_a1**2 + s_a2**2)``, ``alpha_std = area_std / specimen_area``
       (NaN without area); else NaN arrays.
    4. ``issues = check_empty_room(a1, volume) + check_climate(climate1) +
       check_climate(climate2) + check_specimen(volume, specimen_area,
       area if n_objects is not None else None)``.

    Raises
    ------
    ValueError
        If ``t1``/``t2`` (and stds) do not have 18 values, ``specimen_area <= 0`` or
        ``n_objects < 1``, or on invalid inputs of the helper functions.

    References
    ----------
    ISO 354:2003, 8.1.2 Eqs. (5), (7), (8); 8.1.3 Eq. (9); 8.1.4.
    """
    t1a = np.asarray(t1, dtype=float)
    t2a = np.asarray(t2, dtype=float)
    if t1a.shape != (18,) or t2a.shape != (18,):
        raise ValueError("t1 and t2 must have 18 values.")
    if specimen_area is not None and specimen_area <= 0:
        raise ValueError("Specimen area must be positive.")
    if n_objects is not None and n_objects < 1:
        raise ValueError("Number of objects must be at least 1.")
    fm = band_set(3, 100, 5000).center
    c1 = float(speed_of_sound(climate1.temperature))
    c2 = float(speed_of_sound(climate2.temperature))
    m1 = power_attenuation_coefficient(
        fm, climate1.temperature, climate1.relative_humidity, climate1.pressure
    )
    m2 = power_attenuation_coefficient(
        fm, climate2.temperature, climate2.relative_humidity, climate2.pressure
    )
    a1 = equivalent_absorption_area(t1a, volume, c1, m1)
    a2 = equivalent_absorption_area(t2a, volume, c2, m2)
    area = a2 - a1
    nan = np.full(18, np.nan)
    alpha = area / specimen_area if specimen_area is not None else nan
    per_object = area / n_objects if n_objects is not None else nan
    area_std = alpha_std = nan
    if t1_std is not None and t2_std is not None:
        s1 = np.asarray(t1_std, dtype=float)
        s2 = np.asarray(t2_std, dtype=float)
        if s1.shape != (18,) or s2.shape != (18,):
            raise ValueError("t1_std and t2_std must have 18 values.")
        sa1 = SABINE_CONSTANT * volume / c1 * s1 / t1a**2
        sa2 = SABINE_CONSTANT * volume / c2 * s2 / t2a**2
        area_std = np.sqrt(sa1**2 + sa2**2)
        alpha_std = area_std / specimen_area if specimen_area is not None else nan
    issues = (
        check_empty_room(a1, volume)
        + check_climate(climate1)
        + check_climate(climate2)
        + check_specimen(volume, specimen_area, area if n_objects is not None else None)
    )
    return AbsorptionResult(
        frequency=np.asarray(ISO354_FREQUENCIES),
        t1=t1a,
        t2=t2a,
        a1=a1,
        a2=a2,
        area=area,
        alpha=alpha,
        area_per_object=per_object,
        alpha_std=alpha_std,
        area_std=area_std,
        issues=tuple(issues),
    )
