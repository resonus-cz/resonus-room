# Changelog

Semantic versioning: patch = fixes, minor = additions, major = breaking changes.

## 0.2.1 (2026-11-06)
- Requires resonus-core 0.11.1 (comments only). README screenshot of the current GUI.
  Comments and docs cleaned up; no change to the code.

## 0.2.0 (2026-11-06)
- **ISO 354 Absorption tab** (Preferences, Plots: "Absorption tab"): measurements of the
  empty room and with the specimen from two folders (positions from file names `s#m#`),
  room volume, specimen area or number of objects, climate per condition; αs or A_obj
  per one-third-octave band, band table, export as JSON, CSV, HDF5, plot or a setup file
  for `resonus-room absorption` (D-32). Compliance findings (ISO 354 6.1.4, 6.2.1.1,
  6.3.2, 7.1.4, temperature outside 15...30 °C) are shown and recorded, never block the
  evaluation (D-35).
- **Level calibration** of the measurement inputs with a calibrator or a manual offset
  (Preferences, Calibration; `resonus-room calibrate`); calibrated measurements give
  impulse responses in Pa/FS (D-31).
- **STI estimate, PROVISIONAL** (Preferences, Plots: "STI tab"): STI and MTI per octave at
  a position from a calibrated in-situ impulse response, a source calibration at 1 m and
  a background-noise recording; speech spectra per vocal effort (ODEON application note,
  ANSI S3.5). Based on public papers until IEC 60268-16 is available: no auditory masking
  or reception threshold yet. CLI `record-noise`, `sti`, `measure --calibrated`.
- Clarity tab with inner tabs Clarity, Definition and Centre time (D-29, D-33); optional
  IACC tab for ear pairs `s#m#L` / `s#m#R` (D-28); Ts no longer in the
  reverberation-time tab.
- Band tables and findings in pop-up panels over the plots, sized to the table, nearly
  opaque (D-36).
- Stop no longer freezes the GUI or hangs the audio backend (resonus-core 0.11.0);
  Measure and Stop have the same width.
- Noise recordings are stored as resonus-core `TimeSignal`; older noise files are still
  read.
- Requires resonus-core 0.11.0.

## 0.1.1 (2026-11-05)
- README: screenshot of the app. No change to the code.

## 0.1.0 (2026-11-05)
First public release.

- Measurement of impulse responses (sweep, noise, MLS) with one or more inputs and
  outputs, loopback reference, repetitions; level test with live input meters; Stop
  during a measurement.
- Analysis per ISO 3382-1/-2: EDT, T20, T30 with fit quality (ISO 3382-2:2008 Annex B),
  C50, C80, D50, Ts, IACC; room average over positions (ISO 3382-2 clause 8).
- ISO 354 absorption coefficient evaluation with air absorption (ISO 9613-1).
- REW RT60 import and band-by-band comparison.
- GTK 4 / libadwaita GUI (ribbon, transfer function with phase, impulse response,
  reverberation time), command line `resonus-room`, Jupyter notebook example.
- Results as HDF5, CSV, JSON and images; replaceable plot mark.
- Requires resonus-core 0.8.1. License AGPL-3.0-only.
