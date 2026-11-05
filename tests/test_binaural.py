import math

import numpy as np
import pytest
from resonus_core.bands import band_set

from resonus_room.core.binaural import (
    EARLY_LIMIT,
    IACC_JND,
    TAU_MAX,
    BinauralParameters,
    binaural_parameters,
    iacc,
    iacf,
    max_lag,
)
from resonus_room.core.simulate import exponential_decay_ir

FS = 48000.0


def test_constants():
    # ISO 3382-1:2009 B.2 Eq. (B.2) and B.4
    assert TAU_MAX == 1e-3 and EARLY_LIMIT == 0.080 and IACC_JND == 0.075


def test_max_lag_strict():
    assert max_lag(48000) == 47  # 48/48000 = 1 ms is excluded
    assert max_lag(44100) == 44  # 44/44100 = 0.998 ms
    assert max_lag(96000) == 95
    assert max_lag(1500) == 1
    with pytest.raises(ValueError):
        max_lag(0)


def test_iacf_identical_and_inverted():
    x = np.random.default_rng(1).standard_normal(4800)
    lags, v = iacf(x, x, FS)
    assert lags.shape == v.shape == (2 * 47 + 1,)
    assert lags[0] == pytest.approx(-47 / FS) and lags[-1] == pytest.approx(47 / FS)
    assert v[47] == pytest.approx(1.0)
    assert iacc(x, x, FS) == pytest.approx(1.0)
    lags, v = iacf(x, -x, FS)
    assert v[47] == pytest.approx(-1.0)
    assert iacc(x, -x, FS) == pytest.approx(1.0)  # absolute value in Eq. (B.2)
    assert iacc(x, 0.25 * x, FS) == pytest.approx(1.0)  # normalization of Eq. (B.1)


def test_iacf_lag_sign_and_range():
    rng = np.random.default_rng(2)
    x = np.concatenate([rng.standard_normal(4800), np.zeros(200)])
    d = 20
    r = np.concatenate([np.zeros(d), x[:-d]])  # right ear delayed by d samples
    lags, v = iacf(x, r, FS)
    k = int(np.argmax(np.abs(v)))
    assert lags[k] == pytest.approx(d / FS)  # positive lag: right ear lags
    assert v[k] == pytest.approx(1.0)
    # a delay beyond 1 ms is outside the lag range: white noise decorrelates
    r2 = np.concatenate([np.zeros(60), x[:-60]])
    assert iacc(x, r2, FS) < 0.1


def test_iacc_tone_quadrature_closed_form():
    # l = sin(w t), r = cos(w t), 250 Hz, 250 whole periods: IACF(tau) = -sin(w tau) up to
    # edge terms O(K/N); the maximum over |tau| < 1 ms is |sin(w K / fs)|
    f, n = 250.0, 48000
    t = np.arange(n) / FS
    left, right = np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)
    expected = abs(math.sin(2 * math.pi * f * 47 / FS))
    assert iacc(left, right, FS) == pytest.approx(expected, abs=2e-3)


def test_uncorrelated_noise():
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal((2, 96000))
    assert iacc(a, b, FS) < 0.05  # ~ 3.3 / sqrt(N) = 0.011 expected


@pytest.mark.parametrize(
    "args",
    [
        (np.ones(10), np.ones(11)),
        (np.zeros(10), np.ones(10)),
        (np.ones((2, 5)), np.ones((2, 5))),
        (np.array([]), np.array([])),
    ],
)
def test_iacf_errors(args):
    with pytest.raises(ValueError):
        iacf(*args, FS)


def decays(seed_l, seed_r):
    bands = band_set(1, 125, 4000)
    t = np.full(len(bands), 0.6)
    left = exponential_decay_ir(FS, 1.0, t, bands, delay=0.01, seed=seed_l)
    right = exponential_decay_ir(FS, 1.0, t, bands, delay=0.01, seed=seed_r)
    return bands, left, right


def test_binaural_identical_ears():
    bands, left, _ = decays(1, 1)
    p = binaural_parameters(left, left, FS)
    assert isinstance(p, BinauralParameters)
    for v in (p.early, p.late, p.full):
        assert v.shape == (1,) and v[0] == pytest.approx(1.0)
    pb = binaural_parameters(left, left, FS, bands)
    for v in (pb.early, pb.late, pb.full):
        assert v.shape == (len(bands),)
        np.testing.assert_allclose(v, 1.0, atol=1e-9)
    assert pb.onset == p.onset


def test_binaural_common_onset_earlier_ear():
    from resonus_core.window import onset as find_onset

    _, left, _ = decays(1, 1)
    d = 20  # 0.42 ms: right ear later; same response otherwise
    right = np.concatenate([np.zeros(d), left[:-d]])
    p = binaural_parameters(left, right, FS)
    assert p.onset == min(find_onset(left), find_onset(right)) == find_onset(left)
    assert p.full[0] == pytest.approx(1.0, abs=1e-3)
    assert p.early[0] > 0.95


def test_binaural_independent_ears_late_low():
    _, left, right = decays(1, 2)
    p = binaural_parameters(left, right, FS)
    assert p.late[0] < 0.1  # independent diffuse tails, ~0.2 s of noise


def test_binaural_explicit_onset_and_errors():
    _, left, right = decays(1, 2)
    p = binaural_parameters(left, right, FS, onset=100)
    assert p.onset == 100
    with pytest.raises(ValueError):
        binaural_parameters(left, right[:-1], FS)
    with pytest.raises(ValueError):
        binaural_parameters(left, right, FS, onset=left.size)
    with pytest.raises(ValueError):
        binaural_parameters(left[:3000], right[:3000], FS, onset=0)  # no late part
