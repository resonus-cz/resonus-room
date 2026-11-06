# Verification register

One row per feature or value that must be checked against a standard, a reference tool
(REW, Dirac, sapyne) or by the user. Status: **unverified** / **verified-by-test**
(automated test against an independent reference, named in "How") / **verified-by-REW**
(cross-tool comparison on real measurements, criterion in the row) / **verified-by-user**.
Code paths are filled in when the task merges.
Cross-tool comparisons (REW, Dirac) pass at a relative difference of at most 2 % per band
(user decision 2026-11-03, DECISIONS.md D-11).

| # | What | Where in code | How to check | Status |
|---|---|---|---|---|
| V1 | Base-10 exact band centre frequencies and edges | `resonus_core.bands::exact_center, band_edges` | ISO 9613-1:1993 Eq. (6) (thirds, resonus-core `test_bands.py`); octave/even-fraction formulas from python-acoustics; IEC 61260-1:2014 when available | verified-by-test (thirds); unverified vs IEC 61260-1 |
| V2 | Nominal band labels | `resonus_core.bands::NOMINAL_*` | thirds 50 Hz-10 kHz vs ISO 9613-1 Table 1 and ISO 354 cl. 5 (resonus-core `test_bands.py`); other labels (25-40 Hz, 12.5-20 kHz, octaves) from python-acoustics: check vs ISO 266 | verified-by-test (50 Hz-10 kHz thirds); rest unverified |
| V3 | Filter magnitude meets IEC 61260-1 class 1 limits | `resonus_core.bands::band_filter` (Butterworth order 3, PROVISIONAL) | IEC 61260-1 Table of acceptance limits (missing, Q-S1) | unverified |
| V4 | ISO 9613-1 attenuation coefficient | `resonus_core.air::attenuation_coefficient` | Table 1 (20 °C, 25 °C; 110 values, resonus-core `test_air.py`); sapyne `attenuation.py` same formulas (code review) | verified-by-test |
| V5 | Speed of sound c = 331 + 0.6 t | `resonus_core.air::speed_of_sound` | ISO 354:2003 Eq. (6) (resonus-core `test_air.py`) | verified-by-test |
| V6 | Schroeder backward integration; energy envelope precision | `core/decay.py::schroeder, energy_envelope` | closed-form geometric series; envelope over 300 dB (`test_decay_basic.py`) | verified-by-test |
| V7 | Truncation point t1 and tail correction C | `core/decay.py::truncation_point, tail_correction` | deterministic decays with known noise floor (`test_decay.py`: t1 within 2.5 %, C within 3 %); indirectly via V10 (REW) | verified-by-test (synthetic); indirectly vs REW (V10) |
| V8 | Least-squares fit, EDT/T20/T30 ranges | `core/fit.py::fit_decay` | ISO 3382-2 Annex C; hand-computed and closed-form cases (`test_fit.py`) | verified-by-test |
| V9 | ξ (B.3) and curvature C (B.4), warning limits | `core/fit.py` | hand-computed r² (`test_fit.py`) | verified-by-test |
| V10 | EDT/T20/T30 of real IRs | `core/analysis.py::analyze` | REW V5.31.3 on the same IRs: 24 positions in two small boxes (refs/rew-test-data/test1), `resonus-room compare-rew`, guard `tests/test_compare.py::test_real_rew_agreement`. One-third octaves 400 Hz-4 kHz, forward filtering: T20/T30 median abs. diff 0.4 %, 97 % of values within 2 %, 99 % within 5 %; EDT median 0.8 %, 80 % within 2 % (92 % with time-reversed filtering; REW filters zero-phase). 250 Hz-8 kHz: T20/T30 93 % within 2 %, worst cases at 10 kHz (T ~ 0.1 s). Below 125 Hz both tools give unusable values in these boxes (no proper decay). Dirac not checked. | verified-by-REW (400 Hz-4 kHz, 2 % criterion met for 97 %); outside that range not met |
| V11 | C50/C80/D50/Ts | `core/energy.py` | broadband: closed form (`test_energy.py`); bands vs REW V5.31.3 (V10 data, 400 Hz-4 kHz): C50/C80 systematically 0.2 dB lower (median), D50 0.01 lower, Ts 5 ms longer with forward filtering (0 ms with time reversal). Explained by the method: we split early/late energy before filtering (ISO 3382-1 A.3.4, D-8); REW presumably filters first. C80 criterion 0.5 dB (D-14): met for 81 % of values (400 Hz-4 kHz), 74 % (250 Hz-8 kHz); no criterion for C50/D50/Ts. | verified-by-test (broadband); vs REW: C80 criterion met for 81 % (400 Hz-4 kHz), systematic -0.2 dB offset explained |
| V12 | Uncertainty Eq. (2)-(3), (A.4), ISO 354 Eq. (10) | `core/uncertainty.py` | equations by hand (`test_uncertainty.py`); EDT use of Table A.1 D = 10 dB is PROVISIONAL (Q-D8) | verified-by-test |
| V13 | ISO 354 Eq. (5)-(9) | `core/absorption.py::absorption` | equations by hand (`test_absorption.py`); same formulas as sapyne `experimental_absorption.py` (code review) | verified-by-test |
| V14 | ISO 354 Table 1 (max A1) transcription | `core/absorption.py::MAX_EMPTY_ROOM_AREA_200` | user compares with the standard | unverified |
| V15 | αs of a real specimen | `core/iso354.py::run_absorption` | laboratory report / previous sapyne evaluation (Q-D6) | unverified |
| V16 | Synthetic room simulation reproduces set T | `core/simulate.py` | broadband T20 of equal-T IR; additivity (`test_simulate.py`) | verified-by-test |
| V17 | Level test and live meters on a real sound card | `core/levels.py`, resonus-core `SounddeviceIO.playrec_blocks` | user: Test with the interface, input gain turned up until the red clip box lights; meter peak vs the interface's own meter | verified by the user on real hardware (2026-11-05: "seems to work"); exact meter-vs-interface comparison not reported |
| V18 | Stop during a measurement on a real sound card | resonus-core `SounddeviceIO.abort` (`sounddevice.stop()`) | user: start a long measurement, press Stop: playback ends at once, toast "Measurement stopped", no result added; then measure again normally | unverified (mocked sounddevice and simulation tested) |
| V19 | Clarity and IACC tabs on real measurements | `gui/plugin.py` (D-28) | user: measure or open IRs; Clarity tab: C50/C80/D50 per band match the analysis CSV/JSON; Ts in the RT tab; with the IACC tab on, a dummy-head pair labelled s#m#L/s#m#R shows early/late/full IACC per octave (values within 0..1, early above late in typical rooms) | unverified (simulated and synthetic data tested) |
