# Questions, missing sources and decisions for the user

Collected during development (from 2026-11-03). Each item: what is
needed, why, which steps it blocks, and the default used meanwhile. PROVISIONAL defaults
are marked the same way in the code (search for `PROVISIONAL`).

## Sources (standards, papers)
Since 2026-11-05 all sources are in the maintainer's standards library (not distributed: standards are copyrighted). IEC
61260-1:2014 (+ Z1:2017) and ISO 266 are now there (`core/`), so Q-S1 and Q-S5 can be
closed by a verification task (band formulas, Annex E labels, class 1 limits).
- **Q-S1 IEC 61260-1:2014** (octave-band and fractional-octave-band filters). Needed for:
  confirmation of the base-10 band formulas, nominal labels (Annex E), and the class 1/2
  relative-attenuation limits to verify our Butterworth filters. Blocks: claiming IEC
  61260 conformance (ISO 3382-1 4.2.2.2, ISO 3382-2 4.2.2.1, ISO 354 7.3.3 require it).
  Default: Butterworth band-pass, order 3 (6th-order band-pass) on exact base-10 edges
  (D-3), formulas from python-acoustics (D-5). PROVISIONAL.
- **Q-S2 Lundeby et al. 1995** ("Uncertainties of measurements in room acoustics", Acustica
  81, 344-355) or your preferred truncation algorithm. Needed for: a published,
  iterative choice of the truncation point t1 and tail correction C (ISO 3382-1 5.3.3
  leaves the details open). Blocks: nothing (D-6 default works); improves low-SNR bands.
  Default: single-pass method of D-6. PROVISIONAL.
- **Q-S3 Editions in force.** Please confirm ISO 3382-1:2009, ISO 3382-2:2008, ISO 354:2003
  and ISO 9613-1:1993 are the editions your clients/authorities require (or Czech ČSN EN
  ISO adoptions with any national annex). Blocks: nothing; editions are recorded in code.
- **Q-S5 ISO 266:1997** (preferred frequencies) — optional; nominal labels are currently
  taken from ISO 354 clause 5, ISO 9613-1 Table 1 and python-acoustics (D-5).

## Licensing
- **Q-L2 python-acoustics (BSD-3-Clause)** band formulas are ported with attribution
  (`THIRD_PARTY.md`). Compatible with GPL-3.0; just confirm.

## Data needed (method decisions)
- **Q-D5 Dirac export files.** REW exports were provided (2026-11-03,
  refs/rew-test-data); a Dirac parameter export is still needed for a Dirac importer.
- **Q-D6 Reverberation-room data.** The REW set (two small boxes, 12 positions each,
  plus RT60-only exports) covers room-acoustic cross-checks. Still needed for ISO 354: one
  reverberation-room set (empty / with specimen) with known αs (VERIFICATION V15).

## M4 multichannel
Answered 2026-11-04: Q-M1 toggle sequential/simultaneous; Q-M2 one file per channel, a
multichannel kind later for the 4-microphone impedance tube; Q-M3 allowed with warning;
Q-M4 loopback optional; Q-M5 `s#m#`; Q-M6 binaural parameters now, not in the GUI.
Answered 2026-11-04 (second round): Q-M7 go ahead in resonus-core; Q-M8 rates 44.1, 48,
88.2, 96, 176.4, 192 kHz (no 32/384); Q-M9 IACC limits as proposed; Q-M10 `s1m1L` /
`s1m1R`; Q-M11 two Preferences pages as proposed.

## Overlay and room average (D-19)
Answered 2026-11-04: Q-O1 default method "engineering" for the GUI room-average check
(the method only selects the ISO 3382-2:2008 Table 1 / 4.3.2 / 5.1 findings, it changes
no computed value); Q-O2 dummy-head ears (`m1L`/`m1R`) and simultaneous sources
(`s1+s2m1`) stay refused for the room average.

## Measurement checks
- **Q-T1 Decay cut by the silence after the sweep.** Only IR length > silence is warned
  (resonus-core `measurement_warnings`); the measured decay is not checked against the
  silence. Add a warning when, in the highest band, the truncation point lies less than a
  margin before the end of the silence (margin: design choice, no standard clause known)?

## resonus-core candidates
- **Q-C4 `decay` + `fit`** (Schroeder integration, ISO 3382 fits): deferred until the
  building-acoustics app (user 2026-11-03; open item).

## Answered by the user (2026-11-03)
- Q-D1 interrupted noise: not needed (D-2 accepted).
- Q-D2 filter direction: forward default (ISO 3382-2:2008 7.3 NOTE: time reversal is
  for very short reverberation times), reversed as option (D-3). Filter type/order stays with Q-S1.
- Q-D3 m2 from the T2 conditions: ok (D-9).
- Q-D4 late integral over the whole IR: keep; a possible change is an open item (D-8).
- Q-D7 fit-quality thresholds: ISO values, no stricter rules (D-11).
- Q-D8 uncertainty reporting: ok (D-11).
- Q-D9 rounding ties: numpy default (D-11).
- Q-D10 frequency-dependent simulated room: not needed.
- Q-D11 excitation length and averages chosen per measurement (D-11).
- Q-V1 cross-tool tolerance: relative, 2 % (VERIFICATION.md).
- Q-C1...Q-C3 `BandResults`, `bands`, `air`: move to resonus-core (D-10).

## Answered by the user (2026-11-05)
- Q-G1 phase relative to the IR peak; Q-G2 keep "dBFS"; Q-G3 keep the accent frame (D-20).
- Streaming play/record goes into resonus-core (0.6.0, D-21).
- Q-G4 the level test plays on one output after the other; Q-G5 the simulated room is
  normalised to unit energy; Q-G4a the test noise keeps its peak at the level (D-21).

## Answered by the user (2026-11-04)
- Q-D12: impulse response exported as `name.ir.h5` (resonus HDF5 with metadata, format
  documented); WAV kept for other tools (D-16).
- Q-C5 plot helpers (navigation, scroll zoom, trace visibility, markers): move to
  resonus-core. Done: `resonus_core.plot`, resonus-core 0.4.0 (core D-P1, here D-15).

## Answered by the user (2026-11-03, second round)
- Q-0 project plan signed off.
- Q-L1 license: GPL-3.0 for now; recheck before publishing on GitHub (D-12; now AGPL-3.0-only, D-25).
- Q-L3 the ISO 354 copy in refs/ may be used.
- Q-S4 prediction models are not part of this suite.
- Q-D5 (REW part) sample data provided: RT60 exports, IRs (WAV) and frequency responses.
- Q-V2 C80 tolerance vs REW: 0.5 dB (D-14).
- Q-V3 forward filtering stays the default; REW exports were made with reversed filtering (D-14).
- GUI: no REW operations (D-14).
