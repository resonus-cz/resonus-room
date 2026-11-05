# Changelog

Semantic versioning: patch = fixes, minor = additions, major = breaking changes.

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
