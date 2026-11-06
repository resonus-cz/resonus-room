"""Speech spectra per vocal effort for STI (Q-STI3, D-34).

Reference values: ODEON application note "Calculation of Speech Transmission Index in
rooms" (J. H. Rindel, February 2014), Table 1 (ANSI S3.5 SPL at 1 m, 250 Hz-8 kHz) and
p. 3 (125 Hz = 250 Hz - 6 dB); p. 4 (ISO 9921:2003 Annex A: shouted, 82.3 dB(A), is
reduced by 0.4 (82.3 - 75) = 2.92 dB; the note rounds it to 2.9 dB).

Transcription check (2026-11-06): the A-weighted sum of each table row with the
octave A-weightings of IEC 61672-1:2013 Table 3 (ČSN EN 61672-1 ed. 2) gives 59.49,
66.48, 73.69 and 82.32 dB, matching the table's 59.5, 66.5, 73.7 and 82.3 dB.
"""

import math

import pytest

from resonus_room.core.settings import AppSettings
from resonus_room.core.sti import (
    CUSTOM_EFFORT,
    SPEECH_SPL_250_8K,
    SPEECH_SPL_A,
    VOCAL_EFFORTS,
    speech_level_reduction,
    vocal_effort_levels,
)
from resonus_room.core.sti_inputs import settings_speech_levels

# IEC 61672-1:2013 Table 3, A-weighting at the octave centres 250 Hz-8 kHz, dB.
A_WEIGHTING = (-8.6, -3.2, 0.0, 1.2, 1.0, -1.1)


@pytest.mark.parametrize("effort", list(SPEECH_SPL_250_8K))
def test_table_transcription(effort):
    levels = SPEECH_SPL_250_8K[effort]
    la = 10 * math.log10(
        sum(10 ** ((x + a) / 10) for x, a in zip(levels, A_WEIGHTING, strict=True))
    )
    assert abs(la - SPEECH_SPL_A[effort]) < 0.05


def test_efforts():
    assert VOCAL_EFFORTS == ("custom", "normal", "raised", "loud", "shouted")
    assert CUSTOM_EFFORT == "custom"


def test_reduction():
    assert speech_level_reduction(75.0) == 0.0
    assert speech_level_reduction(59.5) == 0.0
    assert speech_level_reduction(82.3) == pytest.approx(2.92)
    assert speech_level_reduction(85.0) == pytest.approx(4.0)


def test_levels():
    normal = vocal_effort_levels("normal")
    assert isinstance(normal, tuple) and len(normal) == 7
    assert all(isinstance(x, float) for x in normal)
    assert normal == pytest.approx((51.2, 57.2, 59.8, 53.5, 48.8, 43.8, 38.6))
    assert vocal_effort_levels("raised") == pytest.approx(
        (55.5, 61.5, 65.6, 62.3, 56.8, 51.3, 42.6)
    )
    assert vocal_effort_levels("loud") == pytest.approx((58.0, 64.0, 70.3, 70.6, 65.9, 59.9, 48.9))
    shouted = (59.0, 65.0, 74.7, 79.8, 75.8, 68.9, 58.2)
    assert vocal_effort_levels("shouted") == pytest.approx(tuple(x - 2.92 for x in shouted))


def test_unknown_effort():
    msg = r"unknown vocal effort 'whisper'; use one of normal, raised, loud, shouted"
    with pytest.raises(ValueError, match=msg):
        vocal_effort_levels("whisper")
    with pytest.raises(ValueError, match="unknown vocal effort 'custom'"):
        vocal_effort_levels(CUSTOM_EFFORT)


def test_settings():
    s = AppSettings()
    assert s.sti_vocal_effort == "normal"
    assert settings_speech_levels(s) == vocal_effort_levels("normal")
    raised = s.replace(sti_vocal_effort="raised")
    assert settings_speech_levels(raised) == vocal_effort_levels("raised")
    custom = s.replace(sti_vocal_effort="custom")
    assert settings_speech_levels(custom) == ()
    own = (60.0, 61.0, 62.0, 63.0, 58.0, 52.0, 47.0)
    assert settings_speech_levels(custom.replace(sti_speech_levels=own)) == own
    # own levels are kept but not used with a table effort
    assert settings_speech_levels(raised.replace(sti_speech_levels=own)) == vocal_effort_levels(
        "raised"
    )
    with pytest.raises(ValueError):
        AppSettings(sti_vocal_effort="whisper")
