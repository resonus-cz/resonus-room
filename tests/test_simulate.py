import numpy as np
import pytest
from resonus_core.bands import band_set

from resonus_room.core.decay import schroeder
from resonus_room.core.fit import fit_decay
from resonus_room.core.simulate import exponential_decay_ir

FS = 48000.0


def test_shape_delay_and_band_limits():
    bs = band_set(1, 250, 1000)
    ir = exponential_decay_ir(FS, 1.0, [1.0, 1.0, 1.0], bs, delay=0.02)
    assert ir.shape == (48000,)
    assert np.all(ir[:960] == 0) and np.any(ir[960:970] != 0)
    spec = np.abs(np.fft.rfft(ir[960:] * 10 ** (3 * np.arange(ir.size - 960) / FS)))
    f = np.fft.rfftfreq(ir.size - 960, 1 / FS)
    inside = (f >= bs.lower[0]) & (f < bs.upper[-1])
    assert np.max(spec[~inside]) < 1e-9 * np.max(spec[inside])


def test_equal_decay_times_give_broadband_t():
    # Same T in every band: the IR is band-limited noise times one envelope, so the
    # broadband Schroeder curve decays with T (many degrees of freedom: small scatter).
    bs = band_set(3, 100, 5000)
    ir = exponential_decay_ir(FS, 3.0, np.full(len(bs), 1.2), bs, seed=4)
    e = schroeder(ir, FS)
    level = 10 * np.log10(e / e[0])
    t20 = fit_decay(np.arange(ir.size) / FS, level, -5.0, -25.0).decay_time
    assert t20 == pytest.approx(1.2, rel=0.03)


def test_additive_over_bands():
    # Same seed -> same white noise; band components depend only on their own mask and T.
    both = exponential_decay_ir(FS, 0.5, [0.8, 0.4], band_set(1, 500, 1000), seed=9)
    low = exponential_decay_ir(FS, 0.5, [0.8], band_set(1, 500, 500), seed=9)
    high = exponential_decay_ir(FS, 0.5, [0.4], band_set(1, 1000, 1000), seed=9)
    np.testing.assert_allclose(both, low + high, atol=1e-12)


def test_noise_and_seed():
    bs = band_set(1, 1000, 1000)
    a = exponential_decay_ir(FS, 1.0, [0.5], bs, noise_rms=1e-3, seed=2)
    b = exponential_decay_ir(FS, 1.0, [0.5], bs, noise_rms=1e-3, seed=2)
    np.testing.assert_array_equal(a, b)
    tail = a[-4800:]  # signal decayed by > 100 dB: only background noise left
    assert np.sqrt(np.mean(tail**2)) == pytest.approx(1e-3, rel=0.05)
    assert not np.array_equal(a, exponential_decay_ir(FS, 1.0, [0.5], bs, seed=3))


def test_invalid():
    bs = band_set(1, 1000, 2000)
    with pytest.raises(ValueError):
        exponential_decay_ir(FS, 1.0, [1.0], bs)  # one value per band
    with pytest.raises(ValueError):
        exponential_decay_ir(FS, 1.0, [1.0, 0.0], bs)
    with pytest.raises(ValueError):
        exponential_decay_ir(FS, 0.1, [1.0, 1.0], bs, delay=0.2)
    with pytest.raises(ValueError):
        exponential_decay_ir(FS, 1.0, [1.0, 1.0], bs, noise_rms=-1.0)
    with pytest.raises(ValueError):
        exponential_decay_ir(5000.0, 1.0, [1.0, 1.0], bs)  # 2 kHz band edge 2.83 kHz > 2.5 kHz
