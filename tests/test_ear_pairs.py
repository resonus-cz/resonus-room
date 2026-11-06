"""Ear pairs and IACC of a pair for the IACC tab (D-28)."""

import numpy as np
import pytest
from resonus_core.bands import band_set
from resonus_core.data import ImpulseResponse, Metadata

from resonus_room.core.binaural import (
    BinauralParameters,
    binaural_parameters,
    iacc_band_set,
    pair_iacc,
)
from resonus_room.core.session import EAR_LABEL, Session, ear_pairs, ear_partner
from resonus_room.core.settings import AppSettings
from resonus_room.core.simulate import exponential_decay_ir

FS = 48000.0


def ir(data=None, fs=FS):
    data = np.arange(4.0) if data is None else data
    return ImpulseResponse(np.asarray(data, dtype=float), fs, "1", Metadata("t", "0"))


def chans(*labels):
    return [(lab, ir(), None) for lab in labels]


def keys(pairs):
    return [(a.key, b.key) for a, b in pairs]


def test_ear_label():
    assert EAR_LABEL.fullmatch("s1m1L").groups() == ("s1m1", "L")
    assert EAR_LABEL.fullmatch("s12m3R").groups() == ("s12m3", "R")
    for label in ("s1m1", "s1m1l", "s0m1L", "m1L", "s1m1L.wav", "s1m1LR"):
        assert EAR_LABEL.fullmatch(label) is None


def test_partner_within_measurement():
    s = Session()
    m1 = s.add("Measurement 1", chans("s1m1L", "s1m1R", "s1m2"))
    m2 = s.add("Measurement 2", chans("s1m1L", "s1m1R"))
    assert ear_partner(m1[0], s.entries) is m1[1]
    assert ear_partner(m1[1], s.entries) is m1[0]
    assert ear_partner(m2[1], s.entries) is m2[0]  # same name only, not Measurement 1
    assert ear_partner(m1[2], s.entries) is None  # no ear label


def test_partner_of_opened_files():
    s = Session()
    left = s.add("s1m1L.wav", chans("s1m1L"))[0]
    right = s.add("s1m1R.wav", chans("s1m1R"))[0]
    other = s.add("s2m1R.wav", chans("s2m1R"))[0]
    assert ear_partner(left, s.entries) is right
    assert ear_partner(right, s.entries) is left
    assert ear_partner(other, s.entries) is None


def test_files_without_ear_labels_do_not_pair():
    s = Session()
    a = s.add("s1m1L.wav", chans("s1m1L.wav"))[0]
    s.add("s1m1R.wav", chans("s1m1R.wav"))
    assert ear_partner(a, s.entries) is None


def test_partner_names_must_match():
    s = Session()
    left = s.add("left.wav", chans("s1m1L"))[0]
    s.add("right.wav", chans("s1m1R"))
    assert ear_partner(left, s.entries) is None


def test_pairs_order_and_dedup():
    s = Session()
    m1 = s.add("Measurement 1", chans("s1m1L", "s1m1R", "s1m2L", "s1m2R"))
    # shown: right ear first, then its left ear (same pair), then the second pair's left
    shown = [m1[1], m1[0], m1[2]]
    assert keys(ear_pairs(shown, s.entries)) == [(m1[0].key, m1[1].key), (m1[2].key, m1[3].key)]
    assert ear_pairs([], s.entries) == []
    lone = s.add("Measurement 2", chans("s1m1L"))
    assert ear_pairs(lone, s.entries) == []


def test_iacc_band_set():
    bands = iacc_band_set()
    assert bands.fraction == 1
    np.testing.assert_array_equal(bands.nominal, [125, 250, 500, 1000, 2000, 4000])


def test_pair_iacc_matches_binaural_parameters():
    octs = band_set(1, 125, 4000)
    left = exponential_decay_ir(FS, 1.0, [0.8] * 6, octs, delay=0.01, seed=1)
    right = exponential_decay_ir(FS, 1.0, [0.8] * 6, octs, delay=0.01, seed=2)
    settings = AppSettings()
    got = pair_iacc(ir(left), ir(right), settings)
    want = binaural_parameters(
        left, right, FS, octs, settings.filter_order, settings.reverse_filter
    )
    assert isinstance(got, BinauralParameters)
    assert got.onset == want.onset
    for name in ("early", "late", "full"):
        assert getattr(got, name).shape == (6,)
        np.testing.assert_allclose(getattr(got, name), getattr(want, name), rtol=1e-12)
    # identical ears: fully correlated in every band and interval
    same = pair_iacc(ir(left), ir(left), settings)
    for name in ("early", "late", "full"):
        np.testing.assert_allclose(getattr(same, name), 1.0, atol=1e-9)
    # uncorrelated noise decays: well below 1 in the late interval
    assert np.all(got.late < 0.6)


def test_pair_iacc_settings_used():
    octs = band_set(1, 125, 4000)
    left = exponential_decay_ir(FS, 1.0, [0.8] * 6, octs, delay=0.01, seed=3)
    right = exponential_decay_ir(FS, 1.0, [0.8] * 6, octs, delay=0.01, seed=4)
    settings = AppSettings(reverse_filter=True)
    got = pair_iacc(ir(left), ir(right), settings)
    want = binaural_parameters(left, right, FS, octs, settings.filter_order, True)
    np.testing.assert_allclose(got.full, want.full, rtol=1e-12)


def test_pair_iacc_errors():
    settings = AppSettings()
    x = np.zeros(48000)
    x[100] = 1.0
    with pytest.raises(ValueError, match=r"sample rates differ: 48000 Hz \(left\), 44100 Hz"):
        pair_iacc(ir(x), ir(x, 44100.0), settings)
    with pytest.raises(ValueError, match="ear responses differ in length: 48000 and 47999"):
        pair_iacc(ir(x), ir(x[:-1]), settings)


def test_show_iacc_setting(tmp_path):
    assert AppSettings().show_iacc is False
    path = tmp_path / "s.toml"
    AppSettings(show_iacc=True).save(path)
    assert AppSettings.load(path).show_iacc is True
