# Questions, missing sources and decisions for the user

Collected during development (from 2026-11-03). Each item: what is
needed, why, which steps it blocks, and the default used meanwhile. PROVISIONAL defaults
are marked the same way in the code (search for `PROVISIONAL`).

## STI (provisional, 2026-11-05/06)
Provisional version on public sources (the standard comes later); work package: level
calibration in resonus-core, noise capture, source calibration at 1 m, calibrated
in-situ IR, compound signal (test signal * IR + noise), STI. Implemented 2026-11-06
(tasks 039-043c, D-31): CLI and GUI tab, all marked PROVISIONAL.
- **Q-STI1** IEC 60268-16: later (user); provisional sources meanwhile (Houtgast &
  Steeneken 1980/1985, Schroeder 1981, Steeneken & Houtgast 2002), to be collected.
- **Q-STI3** speech level: reference spectrum at a vocal effort chosen in the settings?
  **Answered** (user 2026-11-06): the Odeon application note spectra, D-34.
- **Q-STI4** auditory masking and the reception threshold included? (2026-11-06: planned as an STIr task: masking intensity from the band below and the reception threshold added to the noise term of each band, m' = m I/(I + I_am + I_rt); values needed from IEC 60268-16:2020 Annex A or Steeneken & Houtgast 1999, Q-STI1/Q-STI9.)
- **Q-STI5** per position, plus the room mean? **answered** yes (user 2026-11-06): STI per position plus the room mean.
- **Q-STI6** male weighting first, female later? **answered** yes (user 2026-11-06): male weighting first, female later.
- **Q-STI7** full STI test signal first, STIPA when the standard is available?
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
Q-M4 loopback optional; Q-M5 `s#m#`; Q-M6 binaural parameters now, not in the GUI
(superseded 2026-11-05: IACC tab, D-28).
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

## Table content (user to revise)
- **Q-T2 Revise the content of the band tables** (user, 2026-11-06: "I want to revise the
  content of the currently shown tables"). Open, waiting for the user's list. Tables as
  of 2026-11-06:
  - RT tab (`TABLE_COLUMNS`): Band, EDT, T20, T30, r² T20, ξ T20, ξ T30, C, PNR, Flags.
  - Clarity tab (`CLARITY_COLUMNS`): Band, C50, C80, D50, Ts.
  - Absorption tab (`AREA_COLUMNS` / `OBJECT_COLUMNS`): Band, T1, T2, A_T or A_obj, αs
    and its standard deviation.
  Layout (pop-up panels, sizes, background) is done (D-36); only the columns and their
  formats are open.

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

## Answered by the user (2026-11-06)
- **Q-C6** window layout helpers: "move them": `resonus_core.gui.layout` and
  `resonus_core.gui.overlays` (resonus-core 0.11.0); room and the app template
  import them (D-37).
- **Q-C9** plain time-signal kind: "add it": `resonus_core.data.TimeSignal` (0.11.0, task
  028); noise recordings are TimeSignal, older files still read (D-37).
- **Q-C10** Stop can hang the audio backend: "fix it": `playrec`/`rec` on own streams,
  `abort()` only sets the flag, finish timeout (0.11.0; D-37).
- **Q-STI4 source**: the user asked for a publication to start with (2026-11-06). The
  library holds Steeneken & Houtgast 2002 (validation of STIr), which defers the values to
  Steeneken & Houtgast 1999 (Speech Communication 28, 109-123) and IEC 60268-16;
  van Wijngaarden & Steeneken 1999 (Eurospeech, level-dependent masking) is a candidate.
  Recommended: IEC 60268-16:2020 Annex A. Still waiting for the source.
- **Q-STI10** (2026-11-06) provisional choices of D-31 **confirmed** ("Q-STI10 ok"),
  made while working unsupervised: (a) the app keeps its own calibration history
  (`~/.local/share/resonus-room/calibrations.toml`), not shared with resonus-meter;
  (b) calibration needs an explicitly chosen input device (not "default"); (c) a
  calibrated IR has unit Pa/FS and is comparable with the source calibration only with
  the same drive chain (output, amplifier gain, reference); (d) noise recordings and
  source calibrations are stored as `ImpulseResponse` files with a `role` (see Q-C9);
  (e) STI analysis: the whole in-situ IR as build-up, then 8 s, noise 20 s by default;
  (f) speech levels at 1 m are a required input (7 octave values) until Q-STI3. Until
  then, measure the source calibration with the same output level and amplifier gain
  as the in-situ IRs.

- **Q-A1** choices of D-32 **confirmed** ("Q-A1 ok"): which files of a folder are
  positions (top-level `.wav` / IR `.h5`, `s#m#/impulse_response.h5`); repeated labels in
  one folder refused; a note (not an error) when the two folders hold different
  positions; climate without defaults except pressure 101.325 kPa; inputs in the tab,
  not in Preferences.


## Answered by the user (2026-11-05)
- Q-C7 tag resonus-core 0.10.0: **release** (user). Tagged v0.10.0 (core 0309d7b); room
  pins it (D-30), the averages use `TOKENS["neutral_color"]`, `sti.P0` comes from core.
- Q-C8 time histories: implement `LevelHistory` as proposed (first
  written as "Q-S1", which is the IEC 61260 question).
- Q-MT1 (first written as "Q-M3"): the meter is "Re:sonus Meter" in bundles (done, meter bc3edfa) and on its
  audio stream (meter TODO).
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
