# Decision log

One entry per decision; never rewrite old entries, supersede them.
Entries marked PROVISIONAL were proposed during development and await the maintainer's
confirmation; each has a matching item in `docs/QUESTIONS.md`.

## D-1: Python only, no Rust
- Date / status: 2026-11-03, accepted (project default)
- Context: analysis of recorded impulse responses; audio I/O already in resonus-core.
- Decision and reason: Python; no hard real-time path in this app.

## D-2: Integrated impulse response method only
- Date / status: 2026-11-03, accepted by the user 2026-11-03 (Q-D1: no interrupted-noise method)
- Context: ISO 3382-1/-2 and ISO 354 allow interrupted noise or integrated IR; both have
  the same expectation value (ISO 3382-2 5.1).
- Decision and reason: integrated IR method (sweeps/MLS/noise via resonus-core). The
  interrupted-noise method needs a different recording/averaging chain and adds nothing
  for the requested features.
- Consequences: uncertainty per ISO 3382-2 A.3 uses n = 10.

## D-3: Band filters: Butterworth, forward by default, optional time reversal
- Date / status: 2026-11-03; direction accepted by the user 2026-11-03 (Q-D2: forward default,
  reversed as option); filter type/order PROVISIONAL (Q-S1)
- Context: IEC 61260-1 not available; ISO 3382-1/-2/354 require IEC 61260 filters.
- Options: (a) Butterworth band-pass on the exact base-10 band edges (resonus-core
  `band_sos`); (b) FFT brick-wall bands; (c) wait for the standard.
- Decision: (a), order configurable (default 3, i.e. 6th-order band-pass), forward
  (causal) filtering as required for decay analysis (resonus-core D-F1), optional
  time-reversed filtering (ISO 3382-2 7.3 NOTE: BT > 4 instead of BT > 16).
- Consequences: docstrings and results state that IEC 61260-1 class compliance is not
  verified. Verification task once the standard is available.

## D-4: sapyne reuse by re-implementation from the standard
- Date / status: 2026-11-03, PROVISIONAL (Q-L1)
- Context: sapyne is GPL-3.0; license of resonus-room undecided. sapyne's git history
  shows one author (vyhyb).
- Decision: implement air attenuation and ISO 354 directly from the standards in refs/
  (same interface ideas: `m` in 1/m, `A = 55.3 V/(cT) - 4 V m`), cite sapyne as origin
  and use it as an independent cross-check. No verbatim copy until Q-L1 is answered.
- Review findings on sapyne (for the user): `attenuation.py` matches ISO 9613-1:1993
  Eq. (3)-(5), (B.1)-(B.3) but prints its result and requires pandas; `rt_imports.py` uses
  `eval()` on file content (unsafe; the port must not).

## D-5: Band frequencies from base-10 formulas, ported from python-acoustics
- Date / status: 2026-11-03, accepted per user note ("bands can be taken from the
  python-acoustics package"), formulas cross-checked with ISO 9613-1:1993 Eq. (6)
- Decision: port `exact_center_frequency`, edges and the nominal-label tables of
  python-acoustics `acoustics.standards.iec_61260_1_2014` (BSD-3-Clause, commit
  99d7920) into `core/bands.py`, with attribution; do not add the package as a dependency
  (unmaintained, heavy dependencies). Nominal labels limited to 25 Hz-20 kHz (third
  octaves) and 31.5 Hz-16 kHz (octaves).
- Consequences: `THIRD_PARTY.md` holds the BSD-3 notice.

## D-6: Truncation point and tail correction (ISO 3382-1 5.3.3)
- Date / status: 2026-11-03, PROVISIONAL (Q-S2)
- Context: 5.3.3 defines t1 as the intersection of the background-noise level with "a
  sloping line through a representative part" of the squared IR (dB), and C as the
  energy of an exponential continuation with the rate between t0 (10 dB above the level
  at t1) and t1. It does not define the smoothing, the background estimate or the
  "representative part".
- Decision: smoothed energy envelope (moving average, default 10 ms); background = mean
  energy of the last 10 % of the envelope; regression line over the envelope from 5 dB
  below its maximum down to 10 dB above the background; t1 = intersection (clamped to the
  IR end); C from a regression over [t0, t1] of the envelope. Parameters exposed; single
  pass (no Lundeby iteration).
- Consequences: VERIFICATION rows comparing with REW/Dirac on real IRs.

## D-7: Room results file format app-local (JSON + CSV)
- Date / status: 2026-11-03, superseded by D-10 (Q-C1)
- Context: resonus-core data model has only ImpulseResponse/FrequencyResponse.
- Decision: `RoomAnalysis.to_dict()` -> JSON (full, with metadata, settings, standards)
  and a flat CSV band table; propose a `BandResults` type for resonus-core.

## D-8: Energy parameters with window-before-filtering (ISO 3382-1 A.3.4)
- Date / status: 2026-11-03, accepted by the user 2026-11-03 (Q-D4; a possible change is an open item)
- Decision: onset from the broadband IR (resonus-core `window.onset`, A.3.4); early and
  late parts split at onset + te before filtering, each filtered separately with
  zero padding so delayed filter energy is included; late integral runs to the end of
  the IR. Ts from the filtered IR starting at the onset.

## D-9: ISO 354 air correction m2 uses the conditions during the T2 measurement
- Date / status: 2026-11-03, accepted by the user 2026-11-03 (Q-D3)
- Context: ISO 354:2003 8.1.2.2 literally says m2 is calculated "using the climatic
  conditions that have been present in the empty reverberation room", copied from 8.1.2.1;
  Eq. (8) pairs m2 with c2 (temperature t2 during T2), so the conditions during the T2
  measurement are evidently meant.
- Decision: the API takes the conditions of each measurement (T, RH, p) explicitly;
  `m2` is computed from the T2 conditions. Flagged for the user.

## D-10: bands, air and a band-results type move to resonus-core
- Date / status: 2026-11-03, user decision (Q-C1...Q-C3)
- Decision: `core/bands.py` (base-10 band set + Butterworth filter bank), `core/air.py`
  (ISO 9613-1) and a new `BandResults` data type with HDF5 file format go to resonus-core
  (release 0.3.0); resonus-room imports them from there. `decay` + `fit` (Q-C4) stay
  app-local: building acoustics, their second user, is far on the schedule.
- Consequences: band-filter compliance (Q-S1) is now tracked in resonus-core; the app
  JSON/CSV exports stay as human-readable reports next to the `.h5` band results.

## D-11: Remaining small decisions (user, 2026-11-03)
- Rounding ties (ISO 354 8.3): numpy default (half to even) (Q-D9).
- Simulated room keeps one T for all bands (Q-D10).
- Excitation length and number of averages are chosen per measurement (CLI flags, GUI
  controls next to Measure), not derived from an expected T (Q-D11).
- REW/Dirac cross-check tolerance: relative, |ΔT|/T <= 2 % (Q-V1).
- Fit-quality warning limits stay at the ISO 3382-2 values, no stricter in-house rules
  (Q-D7). Uncertainty reporting as implemented (Q-D8).

## D-12: License GPL-3.0 (superseded by D-25: AGPL-3.0-only); plan signed off; scope (user, 2026-11-03)
- resonus-room is licensed GPL-3.0 (`LICENSE`); recheck before publishing on GitHub
  Consequence: sapyne (GPL-3.0) code may be reused; resonus-core (no license
  yet, same author) is a dependency, check its license before distribution.
- Project plan signed off (Q-0).
- Prediction models (Sabine/Eyring/Millington) are not part of this suite (Q-S4 closed).
- The ISO 354 copy in refs/ may be used (Q-L3 closed).

## D-13: Real-IR preprocessing: trailing zeros, exact-zero samples (2026-11-03)
- Context: REW WAV exports (refs/rew-test-data) are 24-bit with a peak of about 12800 LSB
  and end with about 1.7 s of exact zeros. The tail-based background estimate then read
  digital silence (onset 0.3 s too early, inside the harmonic-distortion part) and the
  truncation line fit hit -inf levels (crash).
- Decision: `analyze` removes trailing exact zeros before anything else (they are not
  measured data); the truncation line fit uses only samples with finite level. Not a
  normative choice (ISO 3382-1 5.3.3 leaves the background estimate open).

## D-14: REW comparison criteria, filter default, no REW in the GUI (user, 2026-11-03)
- C80 vs REW passes at |difference| <= 0.5 dB (Q-V2); C50, D50, Ts: differences reported
  without pass/fail.
- Forward filtering stays the default (Q-V3, confirms Q-D2). The user exported the REW
  results with reversed filtering (the export files state "Zero Phase"); this explains why
  our time-reversed option agrees better for EDT and Ts.
- REW results are handled in the CLI only (`import-rew`, `compare-rew`); the GUI has no
  REW operations (keeps it simple).

## D-15: Plot navigation helpers moved to resonus-core
- Date / status: 2026-11-04, accepted (user, Q-C5).
- Decision: `navigable`, `on_scroll`, `zoom_limits`, `set_trace_visible`, `style_points`
  live in `resonus_core.plot` from resonus-core 0.4.0 (core D-P1); the app
  pins v0.4.0 and keeps no copy. Their tests moved with them.

## D-16: One Export action, format from the file extension
- Date / status: 2026-11-04, user request (one button, extension selects the format, as
  in Inkscape); mapping proposed during development, PROVISIONAL until confirmed.
- Decision: `.png/.svg/.pdf` plot of the current tab; `.json` full analysis; `.csv` band
  table; `.h5` band results (as the CLI's analysis.h5, for the viewer); `.wav` impulse
  response as 32-bit float WAV, unscaled. Unknown extension: error toast, nothing written.
- Q-D12 (user, 2026-11-04): the impulse response is also exported as `name.ir.h5` in the
  resonus HDF5 format with its full metadata; structure documented in resonus-core
  docs/FILE_FORMAT.md. `.wav` stays for exchange with other tools.

## D-17: M4 multichannel measurement
- Date / status: 2026-11-04, accepted (user, Q-M1..Q-M11).
- Decision: separate input/output devices (warning if they differ); channel rows with
  labels; outputs sequential or simultaneous (toggle); loopback optional; one `.ir.h5`
  per channel named `s#m#` (`s1m1L`/`s1m1R` for a dummy head); sample rates 44.1-192 kHz
  limited to what the devices accept, default from the device; IACC per ISO 3382-1:2009
  Annex B in core analysis/CLI/notebook, not in the GUI; Preferences split into
  Measurement and Analysis pages. resonus-core gets list-valued settings, device pair and
  listing, multi-output measurement, `settings_page` group filter/shared state, and a
  device/channel preferences group.

## D-18: Preferences: device and channel groups on their own page
- Date / status: 2026-11-04, proposed during development (reversible; deviation from Q-M11).
- Context: Q-M11 approved two pages, Measurement (with the Device groups) and Analysis.
- Decision (reversible): three pages, Measurement (Signal, Processing,
  Simulation), Device (resonus-core `device_groups`: devices, sample rate, input/output
  channel rows, reference, output mode) and Analysis. Reason: `settings_page` adds its
  groups in one block and cannot insert the device groups between Signal and
  Processing; a page of its own needs no core change.
- Alternative: an insertion option in resonus-core `settings_page`, if the user wants
  the device groups inside the Measurement page.

## D-19: Overlay of channels and measurements; room average in the GUI and from labels
- Date / status: 2026-11-04, accepted (user: session list, colour per entry, IR tab
  active entry only, GUI average and CLI average from labels; Q-O1, Q-O2 on
  2026-11-04).
- Session list (`core/session.py`): every measurement or opened file adds one entry per
  channel; one entry is active (IR tab, band table, export of non-IR formats), any
  others can be ticked to overlay in the transfer-function and RT tabs. Entries can be
  removed; Measure/Open no longer clear the earlier results. Replaces the channel
  device drop-down.
- Colours: entry `k` (in order of creation, stable when others are removed) gets
  `TRACE_COLORS[k % 7]` of resonus-core `style` (the first seven); the eighth colour
  (Dark 3) is reserved for the room average. RT tab: parameter by line style, EDT
  dotted, T20 dashed, T30 solid; the EDT/T20/T30 check buttons filter all entries.
- Room average in the GUI: mean of the shown entries (active + ticked) with the code of
  the CLI `average` (ISO 3382-2:2008 clause 8 a), Eq. (A.4)); Table 1 findings as toast.
  Positions from the entry labels `s#m#` (s1m2: source s1, microphone m2).
- Q-O1 (user): the measurement method for the Table 1 check is a setting
  `position_method`, default "engineering". It selects the findings of
  `check_positions` only (Table 1 minimum positions, 4.3.2 octave bands for survey, 5.1
  frequency range); means and sigma (Eq. (A.4) with the actual N) do not depend on it.
  The Table 1 row "decays in each position" applies to the interrupted noise method only.
- Q-O2 (user): labels that are not one source and one microphone position
  (dummy-head ears `m1L`/`m1R`, simultaneous sources `s1+s2m1`, file names) are
  rejected for averaging with a message; no guessing of positions.
- Opened files without channel metadata (user, 2026-11-04): a file name that is only
  source and microphone in the `s#m#` convention, case and white space ignored, after
  removing `.ir.h5`/`.h5`/`.wav` (" s1m 1.wav", "S2M3.WAV") gives the label `s1m1`;
  other names keep the file name. Open accepts several files at once. The room average
  stays over the shown entries (active + ticked).
- Entries can be renamed in the results list (channel label only; the IR metadata, i.e.
  the measurement record, keeps the measured label; room average and export file names
  use the new label).
- CLI: `resonus-room average` also accepts a `measure --out` folder with one subfolder
  per channel label; `--method` is then required (no default for the CLI).

## D-20: Ribbon layout, magnitude + phase, empty-plot axes (user request 2026-11-05)
- Context: the header bar was crowded. The user accepts a departure from the GNOME HIG.
- Header bar: title and main menu only (`header_widgets` returns `[]`). Below it a
  minimal ribbon (`RIBBON_GROUPS`): **Measure** (Measure, excitation length, repetitions,
  playback level, Device button, busy spinner), **Files** (Open, Export), **Results**
  (results list, D-19). Under the ribbon a tab-like `Adw.ViewSwitcher` (Transfer function |
  Impulse response | Reverberation time); the tab content sits in a frame with a slight
  accent border (`APP_CSS`, class `resonus-panel`; to be judged by the user).
- Playback level: the existing setting `level` (resonus-core `MeasurementSettings`: peak
  re amplitude 1 for sweeps/MLS, RMS re 1 for noise) in the ribbon, labelled "dBFS" as
  requested; the tooltip states peak/RMS and the 3.01 dB offset of AES17 dBFS for noise.
- Device button: opens the preferences on the Device page (app-local; the host has no
  "open at page" call, a candidate for resonus-core when the template is derived).
- Transfer function: magnitude over phase at height ratio 2:1, one figure with a shared
  x axis and one navigation toolbar (D-22; first two figures with a toolbar each and
  `link_x`). Phase wrapped to (-180, 180]
  degrees, relative to the time origin of the IR (no delay removed, Q-G1).
- Empty plots (design choices, no standard): TF 20 Hz-20 kHz, -60..20 dB re 1; phase
  +-180 degrees; IR 0-1 s, -100..0 dB; RT the band range of the settings, 0-3 s.
  Autoscaling stays on, so the first result rescales the axes.
- RT axis label "Reverberation time [s]".
- User answers 2026-11-05: Q-G1 phase relative to the IR peak (`peak_delay`, whole
  samples; residual error at most 180 f/fs degrees); Q-G2 keep "dBFS"; Q-G3 keep the
  accent frame.

## D-21: Level test with live input meters; ribbon columns (user requests 2026-11-05)
- Ribbon (supersedes the D-20 rows): Measure = Measure button over the full height,
  beside it Length / Repetitions / Level in one column; Device = Device (1 row) above
  Test (2 rows), beside them the input meters; Files = Open over Export; Results.
- Test: `core.levels.level_test` plays `TEST_DURATION` = 1 s of white noise with its
  **peak** at the level (user decision Q-G4a, 2026-11-05: the same peak as a sweep at
  that level, never clips the output; its RMS is about 12 dB lower) on **each output in turn** (user decision Q-G4, 2026-11-05; order of the
  output labels, whatever the output mode), records all inputs (inputs, then the
  loopback reference "ref") and returns peak, RMS and clipping per output/input pair
  (labels `s1m1`, `s2ref`, ...). The meters show per input the highest peak/RMS over the
  outputs (`combined_levels`), clipped if any output clipped it. The same function is
  the CLI command `test-levels` and is called from the notebook.
- Live meters: resonus-core 0.6.0 `AudioIO.playrec_blocks` (user decision 2026-11-05:
  streaming call in resonus-core; Python callback, sufficient for a meter: the callback
  only computes two numbers per block, no hard real-time need, so no Rust). Each block's
  levels go to the GUI with `GLib.idle_add`.
- Meter: scale -100...0 dB re full scale, marks -12, -42, -60, -100 dB (user); zones
  (design choice): hot >= -12, good -42...-12, low -60...-42, quiet below; bar = peak of
  the latest block, line = highest peak, tick = RMS, red box = clipped since the test
  started (|x| >= 1 - 2**-15, the top code of a 16-bit converter; design choice).
- The simulated room is normalised to unit energy, sum h**2 = 1 (user decision Q-G5,
  2026-11-05; before: energy gain about +34 dB, so simulated tests clipped above about
  -34 dB). T and all other level-independent results are unchanged; absolute simulated
  levels are lower by that gain.

## D-22: Stop button; one toolbar for magnitude and phase (user requests 2026-11-05)
- Stop: while a measurement runs, the Measure button shows a solid red "Stop" (Ctrl+M
  toggles too). It calls `abort()` on the measurement's device (resonus-core 0.7.0,
  `AudioIO.abort`, user decision: in resonus-core, not an app-local `sounddevice.stop()`);
  the running play/record call raises `MeasurementAborted`, the measurement is discarded
  (no partial result), toast "Measurement stopped". The device is opened on the main
  loop before the thread starts, so Stop has an object to abort; an aborted device is
  used up (the next measurement opens a new one).
- TF tab: the phase is a panel of the magnitude's figure (resonus-core 0.7.0
  `MplPlotView.add_panel((2, 1))`, user decision: in resonus-core): one canvas, one
  toolbar (pan/zoom/home act on both, x shared), the magnitude's x tick labels hidden;
  the phase axes are no longer squeezed by a second toolbar (user: "phase slightly too
  narrow"). Export of the TF plot now contains the phase too. `plotting.link_x` removed.
- The solid red of Stop is app CSS (`APP_CSS`, libadwaita's `--destructive-bg-color`):
  libadwaita draws `destructive-action` as a pale tint that looked insensitive next to the
  solid blue Measure (design choice).

## D-23: Header bar brand and title (user request 2026-11-05)
- Left: "Re:sonus" (`style.TOKENS["brand_text"]`) in Lexend SemiBold (600), the weight of
  the plot mark (Re:sonus style decision D1; the user's "MediumBold" read as SemiBold, between
  the bundled Medium and Bold), 1.2 times the header font (user: "slightly larger"), its
  text starting at the left edge of the Measure button (CSS `margin-left: 5px`, measured).
  Centre: "Room" (`HEADER_TITLE`, an `Adw.WindowTitle` set
  on the host's header bar once the brand label is packed into it). The window title stays
  "Resonus Room" (window list, About dialog).
- GTK gets the bundled Lexend files through `PangoCairo.FontMap.add_font_file` (Pango
  >= 1.56); without it the brand falls back to the default font.
- Candidate for the app template / resonus-core host (brand slot in the header
  bar), with the Device-page call of D-21.

## D-24: User-replaceable plot mark (user request 2026-11-05)
- Setting `plot_mark` (preferences: Analysis page, group "Plots"; settings TOML; CLI
  `--settings`, now also for `plot`; notebook `MplPlotView(brand_text=settings.plot_mark)`):
  the text in the bottom-right corner of every plot and exported image. Default
  resonus-core's "Re:sonus" (`style.TOKENS["brand_text"]`), empty: no mark.
- In resonus-core 0.8.0 (user decision): `add_brand(fig, text)`, `MplPlotView(brand_text=)`,
  `MplPlotView.set_brand(text)`, so every Re:sonus app can offer it. The GUI
  changes the mark of the open plots at once.
- The header-bar "Re:sonus" (D-23) stays fixed: the setting is for plots only (user).

## D-25: License AGPL-3.0-only for resonus-room and resonus-core (user, 2026-11-05)
- Before publishing both repos on GitHub. User goals: own and third-party commercial use
  allowed; selling allowed as long as the code stays available. GPL-3.0 already keeps
  derivatives open when distributed; AGPL-3.0 section 13 also covers modified versions
  offered as a network service. `-only` (not `-or-later`): the user keeps control of the
  version; as sole copyright holder the user can relicense (external contributions
  would need a CLA to keep that option).
- AGPL is OSI-approved and FSF-free: GitHub, PyPI and Flathub show it as open source.
- Supersedes the GPL-3.0 of D-12. Compatible: sapyne (GPL-3.0, same author, nothing
  copied), python-acoustics (BSD-3-Clause, attribution in THIRD_PARTY.md), dependencies
  per `pip-licenses` (VERIFICATION / THIRD_PARTY.md). Not legal advice.

## D-26: Application id cz.resonus.Room (user, 2026-11-05)
- The user owns resonus.cz: app ids are `cz.resonus.<Name>` (Re:sonus architecture decision
  D7, also in the project template), independent of the code hosting. resonus-room:
  `cz.resonus.Room` (was `io.github.vyhyb.ResonusRoom`, never released). Settings are
  stored by folder name (`~/.config/resonus-room`), so existing settings stay valid.
- Flathub verification: token at `https://resonus.cz/.well-known/org.flathub.VerifiedApps.txt`
  (needs a web page on the domain).

## D-27: Publishing: private repo -> public subset -> GitHub (user, 2026-11-05)
- GitHub organisation `resonus-cz` (the name `resonus` is taken by another user). The
  private repos (Gitea) keep everything; `scripts/export_public.sh <tag> <public-clone>`
  copies only `scripts/public_files.txt` of the tag into the public repo (fresh history,
  one "Release" commit per version), after `scripts/check.sh` passed; a Gitea
  `<name>-public` repo push-mirrors to GitHub.
- The export rewrites only metadata: local (file URL) dependency URLs ->
  `https://github.com/resonus-cz/<repo>@<tag>`, and a public `.gitignore`. It refuses to
  export local paths or internal files and runs the tests on the exported tree.
- Public content (user decisions 2026-11-05): code, tests, examples, LICENSE, README,
  CHANGELOG, THIRD_PARTY, CONTRIBUTING (+ PR template), pyproject, lock file, check/lint
  scripts, docs FILE_FORMAT and DECISIONS (resonus-room also QUESTIONS, COMPLIANCE,
  VERIFICATION); those docs were reworded once to be publishable, and references in the
  code point only to them. The REW sample data stays private (its tests skip).
- Pull requests: issue first, CLA before integration (keeps relicensing possible); a PR
  is applied to the private repo (`git am`, authorship kept), released, credited with
  `export_public.sh --co-author` and the changelog, and closed with a link. Never merged
  on GitHub (the push mirror would overwrite it).
- The export refuses workflow traces and the copyright-holder placeholder of the README.
