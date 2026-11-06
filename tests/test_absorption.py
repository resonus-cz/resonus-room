import numpy as np
import pytest
from resonus_core.air import power_attenuation_coefficient, speed_of_sound
from resonus_core.bands import band_set

from resonus_room.core.absorption import (
    ISO354_FREQUENCIES,
    MAX_EMPTY_ROOM_AREA_200,
    AbsorptionResult,
    Climate,
    absorption,
    check_climate,
    check_empty_room,
    check_positions,
    check_room,
    check_specimen,
    equivalent_absorption_area,
    max_empty_room_area,
    octave_values,
    round_to,
    speed_of_sound_reported,
    volume_factor,
)


def test_tables_transcribed_from_iso354():
    # Clause 5 and Table 1 (6.1.4); transcription also listed for user check (VERIFICATION V14).
    assert len(ISO354_FREQUENCIES) == 18 and ISO354_FREQUENCIES[0] == 100.0
    assert ISO354_FREQUENCIES[-1] == 5000.0
    assert MAX_EMPTY_ROOM_AREA_200[:10] == (6.5,) * 10
    assert MAX_EMPTY_ROOM_AREA_200[10:] == (7.0, 7.5, 8.0, 9.5, 10.5, 12.0, 13.0, 14.0)


def test_equivalent_absorption_area_eq5():
    # V = 200 m3, c = 343 m/s, T = 5 s, m = 0.001 1/m:
    # A = 55.3 * 200 / (343 * 5) - 4 * 200 * 0.001 = 6.44898 - 0.8
    a = equivalent_absorption_area(5.0, 200.0, 343.0, 0.001)
    assert a == pytest.approx(55.3 * 200 / (343 * 5) - 0.8, rel=1e-12)
    np.testing.assert_allclose(
        equivalent_absorption_area([2.0, 4.0], 100.0, 340.0), 55.3 * 100 / 340 / np.array([2, 4])
    )
    for bad in ((0.0, 200.0, 343.0), (5.0, 0.0, 343.0), (5.0, 200.0, 0.0)):
        with pytest.raises(ValueError):
            equivalent_absorption_area(*bad)
    with pytest.raises(ValueError):
        equivalent_absorption_area(5.0, 200.0, 343.0, -1e-3)


def test_volume_scaling_and_table1():
    assert volume_factor(200.0) == 1.0
    assert volume_factor(400.0) == pytest.approx(2 ** (2 / 3))
    np.testing.assert_allclose(
        max_empty_room_area(100.0), np.array(MAX_EMPTY_ROOM_AREA_200) * 0.5 ** (2 / 3)
    )
    with pytest.raises(ValueError):
        volume_factor(0.0)


def test_octave_values():
    v = np.arange(18.0)
    np.testing.assert_allclose(octave_values(v), [1, 4, 7, 10, 13, 16])
    with pytest.raises(ValueError):
        octave_values(np.arange(17.0))


def test_round_to():
    np.testing.assert_allclose(round_to([0.444, 0.456, 1.0049], 0.01), [0.44, 0.46, 1.0])
    np.testing.assert_allclose(round_to([3.14, 3.16, -0.26], 0.1), [3.1, 3.2, -0.3])
    with pytest.raises(ValueError):
        round_to(1.0, 0.0)


def test_check_room():
    assert check_room(200.0, 10.0) == []  # 1.9 * 200**(1/3) = 11.11 m
    issues = check_room(140.0, 11.0)
    assert len(issues) == 2 and issues[0].startswith("6.1.1") and issues[1].startswith("6.1.2")


def test_check_empty_room():
    ok = np.array(MAX_EMPTY_ROOM_AREA_200) * 0.9
    assert check_empty_room(ok, 200.0) == []
    high = ok.copy()
    high[17] = 15.0  # above 14.0 at 5 kHz (last band: no smoothness check)
    issues = check_empty_room(high, 200.0)
    assert issues == [issues[0]] and issues[0].startswith("6.1.4") and "5000 Hz" in issues[0]
    dip = np.full(18, 5.0)
    dip[5] = 4.0  # 20 % below the mean of its neighbours at 315 Hz; neighbours deviate 11 %
    issues = check_empty_room(dip, 200.0)
    assert len(issues) == 1 and issues[0].startswith("6.1.4") and "315 Hz" in issues[0]
    with pytest.raises(ValueError):
        check_empty_room(np.ones(17), 200.0)


def test_check_specimen():
    assert check_specimen(200.0, area=10.8) == []
    assert check_specimen(200.0, area=9.9)[0].startswith("6.2.1.1")
    assert check_specimen(200.0, area=12.5)[0].startswith("6.2.1.1")
    assert check_specimen(400.0, area=12.5) == []  # upper limit 12 * 2**(2/3) = 19.05
    assert check_specimen(400.0, area=9.9)[0].startswith("6.2.1.1")  # lower limit not scaled
    change = np.full(18, 5.0)
    assert check_specimen(200.0, area_change=change) == []
    change[0] = 0.9
    change[1] = 12.5
    issues = check_specimen(200.0, area_change=change)
    assert len(issues) == 2 and all(i.startswith("6.2.2.3") for i in issues)
    assert check_specimen(150.0, area_change=np.full(18, 1.0)) != []  # not more than 1 m2


def test_check_climate_and_positions():
    assert check_climate(Climate(20.0, 50.0)) == []
    assert check_climate(Climate(14.0, 50.0))[0].startswith("6.3.2")
    assert len(check_climate(Climate(14.0, 95.0))) == 2
    assert check_climate(Climate(20.0, 29.0))[0].startswith("6.3.2")
    assert check_positions(3, 4) == []
    assert check_positions(6, 1, simultaneous=True) == ["7.1.4: 1 source positions < 2"]
    assert len(check_positions(2, 2)) == 2
    assert check_positions(3, 2, simultaneous=True) == []


def test_absorption_closed_form():
    v = 210.0
    c1, c2 = Climate(20.0, 50.0, 100.0), Climate(22.0, 45.0, 100.5)
    t1 = np.linspace(6.0, 3.0, 18)
    t2 = np.linspace(4.0, 1.2, 18)
    r = absorption(t1, t2, v, c1, c2, specimen_area=10.8)
    assert isinstance(r, AbsorptionResult)
    fm = band_set(3, 100, 5000).center
    sp1, sp2 = 331 + 0.6 * 20.0, 331 + 0.6 * 22.0
    m1 = power_attenuation_coefficient(fm, 20.0, 50.0, 100.0)
    m2 = power_attenuation_coefficient(fm, 22.0, 45.0, 100.5)
    a1 = 55.3 * v / (sp1 * t1) - 4 * v * m1
    a2 = 55.3 * v / (sp2 * t2) - 4 * v * m2
    np.testing.assert_allclose(r.a1, a1, rtol=1e-12)
    np.testing.assert_allclose(r.a2, a2, rtol=1e-12)
    # Eq. (8) written out
    at = 55.3 * v * (1 / (sp2 * t2) - 1 / (sp1 * t1)) - 4 * v * (m2 - m1)
    np.testing.assert_allclose(r.area, at, rtol=1e-12)
    np.testing.assert_allclose(r.alpha, at / 10.8, rtol=1e-12)
    np.testing.assert_array_equal(r.frequency, ISO354_FREQUENCIES)
    assert np.all(np.isnan(r.area_per_object)) and np.all(np.isnan(r.alpha_std))
    assert speed_of_sound(20.0) == sp1


def test_absorption_objects_and_uncertainty():
    v = 200.0
    c = Climate(20.0, 50.0)
    t1 = np.full(18, 5.0)
    t2 = np.full(18, 4.0)
    s1 = np.full(18, 0.05)
    s2 = np.full(18, 0.04)
    r = absorption(t1, t2, v, c, c, n_objects=4, t1_std=s1, t2_std=s2)
    np.testing.assert_allclose(r.area_per_object, r.area / 4)
    assert np.all(np.isnan(r.alpha)) and np.all(np.isnan(r.alpha_std))
    sp = 343.0
    expected = np.hypot(55.3 * v / sp * 0.05 / 25, 55.3 * v / sp * 0.04 / 16)
    np.testing.assert_allclose(r.area_std, expected, rtol=1e-12)
    r = absorption(t1, t2, v, c, c, specimen_area=10.0, t1_std=s1, t2_std=s2)
    np.testing.assert_allclose(r.alpha_std, expected / 10.0, rtol=1e-12)


def test_absorption_issues_and_errors():
    c = Climate(20.0, 50.0)
    t1 = np.full(18, 1.0)  # A1 = 55.3*200/343 - 4Vm >= 20 m2 > Table 1 in every band
    r = absorption(t1, np.full(18, 0.9), 200.0, c, c)
    assert len(r.issues) == 18 and all(i.startswith("6.1.4") for i in r.issues)
    # T1 for a smooth A1 = 6 m2 including air absorption: T1 = 55.3 V / (c (A1 + 4 V m1))
    m1 = power_attenuation_coefficient(band_set(3, 100, 5000).center, 20.0, 50.0)
    t1 = 55.3 * 200 / (343.0 * (6.0 + 4 * 200 * m1))
    r = absorption(t1, 0.8 * t1, 200.0, c, Climate(16.0, 20.0), specimen_area=11.0)
    np.testing.assert_allclose(r.a1, 6.0, rtol=1e-12)
    assert r.issues == ("with specimen: 6.3.2: relative humidity 20 % outside 30...90 %",)
    # user, 2026-11-06 (D-35): temperatures outside 15...30 degC are findings, not errors;
    # c = 331 + 0.6 t extrapolated (ISO 354:2003 8.1.2.1 Eq. (6))
    r = absorption(t1, 0.8 * t1, 200.0, Climate(12.0, 50.0), Climate(32.0, 95.0), 11.0)
    assert r.issues == (
        "empty room: 6.3.2: temperature 12 degC < 15 degC",
        "empty room: 8.1.2.1: temperature 12 degC outside 15...30 degC of Eq. (6); "
        "c = 338.2 m/s extrapolated",
        "with specimen: 6.3.2: relative humidity 95 % outside 30...90 %",
        "with specimen: 8.1.2.1: temperature 32 degC outside 15...30 degC of Eq. (6); "
        "c = 350.2 m/s extrapolated",
    )
    assert np.all(np.isfinite(r.alpha))
    assert speed_of_sound_reported(12.0)[0] == pytest.approx(331.0 + 0.6 * 12.0)
    assert speed_of_sound_reported(20.0) == (float(speed_of_sound(20.0)), [])
    with pytest.raises(ValueError, match="no positive speed of sound"):
        speed_of_sound_reported(-600.0)
    with pytest.raises(ValueError):
        absorption(np.ones(17), np.ones(18), 200.0, c, c)
    with pytest.raises(ValueError):
        absorption(np.ones(18), np.ones(18), 200.0, c, c, specimen_area=0.0)
    with pytest.raises(ValueError):
        absorption(np.ones(18), np.ones(18), 200.0, c, c, n_objects=0)
    # c outside 15-30 degC: a finding, not an error since 2026-11-06 (D-35)
    r = absorption(np.ones(18), np.ones(18), 200.0, Climate(10.0, 50.0), c)
    assert any(i.startswith("empty room: 8.1.2.1") for i in r.issues)
