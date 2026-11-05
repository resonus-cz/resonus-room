import numpy as np
import pytest
from resonus_core.bands import band_set

from resonus_room.core.analysis import PARAMETERS, STANDARDS, ParameterSet, RoomAnalysis, analyze
from resonus_room.core.decay import decay_curve
from resonus_room.core.energy import energy_parameters
from resonus_room.core.fit import EVALUATION_RANGES, fit_decay
from resonus_room.core.simulate import exponential_decay_ir
from resonus_room.core.uncertainty import bandwidth, reverberation_std

FS = 48000.0
OCT = band_set(1, 125, 4000)
T_OCT = np.array([2.0, 1.8, 1.6, 1.4, 1.2, 1.0])


def room(seed: int = 1, noise: float = 1e-4) -> np.ndarray:
    return exponential_decay_ir(FS, 3.0, T_OCT, OCT, delay=0.01, noise_rms=noise, seed=seed)


def test_structure_and_onset():
    a = analyze(room(), FS, OCT)
    assert isinstance(a, RoomAnalysis) and isinstance(a.broadband, ParameterSet)
    assert len(a.band_results) == 6 and a.bands == OCT
    assert abs(a.onset - 480) < 0.002 * FS  # 10 ms delay, ISO 3382-1 A.3.4 onset
    assert a.settings == {"filter_order": 3, "reverse": False, "decay_method": "correct"}
    assert a.standards == STANDARDS and PARAMETERS == ("EDT", "T20", "T30")
    for name in ("EDT", "T20", "T30", "C", "C50", "C80", "D50", "Ts", "PNR"):
        assert a.values(name).shape == (6,)
    with pytest.raises(KeyError):
        a.values("G")
    p = a.band_results[2]
    assert set(p.fits) == {"EDT", "T20", "T30"}
    assert (p.edt, p.t20, p.t30) == tuple(p.fits[n].decay_time for n in PARAMETERS)


def test_band_values_follow_their_definition():
    # Each band result equals the building blocks applied to the filtered response.
    from resonus_core.bands import filter_bands

    ir = room()
    a = analyze(ir, FS, OCT, order=2, reverse=True, method="truncate", onset=400)
    y = filter_bands(ir[400:], OCT, FS, 2, True)
    e = energy_parameters(ir, FS, 400, OCT, 2, True)
    for k, p in enumerate(a.band_results):
        curve = decay_curve(y[:, k], FS, "truncate")
        assert p.curve.truncation == curve.truncation
        fit = fit_decay(curve.time, curve.level, *EVALUATION_RANGES["T30"])
        assert p.t30 == pytest.approx(fit.decay_time, rel=1e-12)
        assert p.c80 == pytest.approx(e.c80[k], rel=1e-12)
        assert p.ts == pytest.approx(e.ts[k], rel=1e-12)
    bw = bandwidth(OCT.center, 1)
    np.testing.assert_allclose(a.t20_std, reverberation_std(a.values("T20"), bw, 1, 20))
    np.testing.assert_allclose(a.t30_std, reverberation_std(a.values("T30"), bw, 1, 30))


def test_recovers_set_decay_times_within_iso_uncertainty():
    # Single noise realization: deviations stay within 3 sigma of ISO 3382-2 Eq. (A.4)
    # (N = 1, n = 10 for the integrated impulse response method).
    a = analyze(room(), FS, OCT)
    for name, sigma in (("T20", a.t20_std), ("T30", a.t30_std)):
        assert np.all(np.abs(a.values(name) - T_OCT) < 3 * sigma), name


def test_no_bias_pooled_over_realizations():
    t20 = []
    t30 = []
    for seed in range(6):
        a = analyze(room(seed), FS, OCT)
        t20.append(a.values("T20"))
        t30.append(a.values("T30"))
    # 500 Hz...4 kHz: filter and method bias below 3 % (random part averaged out).
    np.testing.assert_allclose(np.mean(t20, 0)[2:], T_OCT[2:], rtol=0.03)
    np.testing.assert_allclose(np.mean(t30, 0)[2:], T_OCT[2:], rtol=0.03)


def test_flags_low_dynamic_range():
    a = analyze(room(noise=3e-2), FS, OCT)  # strong background: peak-to-noise ~ 40 dB
    pnr = a.values("PNR")
    assert np.all(pnr < 45)
    for p, r in zip(a.band_results, pnr, strict=True):
        # "range" (curve does not reach the stop level) replaces the other flags
        low30 = {"T30:dynamic_range", "T30:range"} & set(p.flags)
        low20 = {"T20:dynamic_range", "T20:range"} & set(p.flags)
        assert bool(low30) == (r < 45)
        assert bool(low20) == (r < 35)


def test_broadband_parameters():
    a = analyze(room(), FS, OCT)
    b = a.broadband
    assert 1.0 < b.t20 < 2.0
    assert "T20:bt" not in b.flags  # no bandwidth for broadband
    e = energy_parameters(room(), FS, a.onset)
    assert b.c80 == pytest.approx(float(e.c80[0]))


def test_errors():
    with pytest.raises(ValueError):
        analyze(np.zeros((10, 2)), FS, OCT)
    with pytest.raises(ValueError):
        analyze(room(), FS, OCT, onset=10**7)


def test_trailing_digital_silence_is_ignored():
    # REW WAV exports end with exact zeros (padding); they are not part of the measurement
    # and must not change the onset (background estimate) or any parameter.
    ir = room()
    padded = np.concatenate([ir, np.zeros(int(1.5 * FS))])
    a, b = analyze(ir, FS, OCT), analyze(padded, FS, OCT)
    assert a.onset == b.onset
    for name in ("EDT", "T20", "T30", "C80", "D50", "Ts"):
        np.testing.assert_array_equal(a.values(name), b.values(name))


def test_quantized_ir_with_zero_padding():
    # 24-bit integer IR with a small peak (as exported by REW, about 12800 LSB): the decay
    # falls below one LSB, leaving runs of exact zeros, and the file ends with zero padding.
    # The analysis must not fail, and quantization must barely change T20/T30.
    ir = room(seed=3, noise=1e-5)
    q = np.round(ir / np.abs(ir).max() * 12845) / 12845 * np.abs(ir).max()
    q = np.concatenate([q, np.zeros(int(1.5 * FS))])
    a, b = analyze(ir, FS, OCT), analyze(q, FS, OCT)
    assert abs(a.onset - b.onset) < 0.002 * FS
    for name in ("T20", "T30"):
        assert np.all(np.isfinite(b.values(name)))
        np.testing.assert_allclose(b.values(name), a.values(name), rtol=0.02)
