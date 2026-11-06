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
  started (|x| >= 1 - 2**-15, the top code of a 16-bit converter; design choice; the
  one clipping value of the whole stack, user 2026-11-05: resonus-core calibration and
  resonus-meter use it too).
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

## D-28: Energy parameters and IACC in the GUI (user, 2026-11-05)
Replaces the earlier "binaural parameters not in the GUI" (Q-M6).
- **Ts** (superseded by D-29: Clarity tab) in the reverberation-time tab, a check button next to EDT/T20/T30 (a time in s on
  the same axis; ylabel "Time [s]"), dash-dot line; a Ts column in the band table.
- **Clarity tab**: C50 and C80 (dB) above D50 (ratio, ISO 3382-1:2009 Eq. (A.11), not a
  percentage) in one figure with a shared frequency axis; check buttons C50 / C80 / D50
  as in the RT tab; overlaid entries, room average (ISO 3382-2:2008 clause 8) and band
  table as in the RT tab.
- **IACC tab**, shown only with the setting `show_iacc` (group Plots, off by default):
  early / late / full IACC (ISO 3382-1:2009 B.4) of each ear pair, check buttons per
  interval; always octave bands 125 Hz-4 kHz (B.4), whatever the band switch.
- **Ear pairs by label** (D-17 convention): `s#m#L` with `s#m#R` of the same
  measurement, or of two opened files whose names differ only in that label
  (`s1m1L.wav` / `s1m1R.wav`, after labelling them). No pair, no IACC.
- Line styles (design choice): C50 dashed, C80 and D50 solid; IACC early solid, late
  dashed, full dotted.

## D-29: Clarity tab layout; Ts with the clarity measures (user, 2026-11-05)
Replaces the Ts placement of D-28.
- Ts is the centre time (ISO 3382-1:2009 A.2.3, Eq. (A.13)), grouped in Table A.1 with C80
  and D50 under "perceived clarity of sound"; it moves from the reverberation-time tab to
  the Clarity tab.
- Layout: C50/C80 (dB) as the larger plot on top; D50 (ratio) and Ts (ms, as in Table A.1)
  side by side below; one figure with one toolbar (resonus-core 0.9.0
  `MplPlotView.add_row`). Check buttons C50 / C80 / D50 / Ts; the Clarity table gains a
  "Ts [ms]" column, the RT table loses its Ts column.
- Ts is computed and stored in s (CSV, JSON, HDF5, room-average CSV unchanged); only the
  plot and the GUI table show ms. Empty Ts plot 0-300 ms (design choice, around the
  typical range 60-260 ms of Table A.1).

## D-30: resonus-core 0.10.0 (user, 2026-11-05; Q-C7)
- Core main is released as v0.10.0 (calibration D-C1, style tokens with the dark variant
  D-T1, meter-bundle reader D-S1, `LevelHistory`); room pins the tag.
- Colours (core D-T1): the entries use all seven trace colours (trace 4 is now Orange 4);
  the room average uses the neutral token `TOKENS["neutral_color"]` (Dark 3, unchanged
  value), which is no longer a trace colour.
- `sti.P0` is `resonus_core.calibration.P0` (ISO 1683:2015 Table 1); STI items 2-4
  (noise capture, source calibration, calibrated in-situ IR) can now use core
  calibration.

## D-31: Calibration and STI inputs in the app (2026-11-05; confirmed by the user 2026-11-06, Q-STI10)
Working unsupervised ("do everything you can without my input"); each point is a design
choice, listed in QUESTIONS.md Q-STI10.
- **Calibration history:** the app's own file `$XDG_DATA_HOME/resonus-room/calibrations.toml`
  (resonus-core format and method, D-C1). Not shared with resonus-meter: the meter
  identifies devices by cpal names, room by PortAudio keys `"host API: name"`, so records
  would not match anyway.
- **Profiles:** one per measurement input (not the reference); the input device must be
  chosen explicitly (the default device can change); the simulated device has the id
  `simulated`.
- **Units:** a measured IR has unit `"1"` (input FS per output FS, or per reference FS);
  calibrated with the input's offset it is `"Pa/FS"`. Source calibration and in-situ IR
  are comparable only with the same drive chain (output, amplifier gain, reference).
- **Files of the STI inputs:** resonus-core has no time-signal kind, so the noise recording
  (unit `"Pa"`) and the source calibration (unit `"Pa/FS"`) are `ImpulseResponse` files
  with `metadata.extra["role"]` = `background-noise` / `source-calibration`. A core kind
  for plain signals would be cleaner: candidate for the user's core curation (Q-C9).
- **STI analysis lengths:** the whole in-situ IR as build-up, then 8 s of analysis (at
  least five periods of 0.63 Hz); noise recordings default to 20 s.
- **Speech levels:** a required user input until Q-STI3 (no default from the filed
  sources).

## D-32: ISO 354 absorption from two folders, GUI tab (2026-11-06; confirmed by the user the same day, Q-A1)
User request 2026-11-06: "load irs from two selected folders, with and without sample,
fill in the other inputs and evaluate. Should go in another tab, toggled in the settings."
Provisional choices (QUESTIONS.md Q-A1):
- **Folder contents** (`core/iso354_folders.py`): every impulse response in the folder is
  one position: top-level `.wav` (first channel; REW exports) and resonus `.h5` files that
  hold an `ImpulseResponse` without a `role` (so `measure --out` folders work), plus
  `<label>/impulse_response.h5` channel subfolders. Hidden files are skipped.
- **Positions:** labels `s#m#` from file or subfolder names; two files with the same label
  in one folder are refused (the REW set test1 holds two boxes, " s1m 1" and "s1m 1",
  in one folder: averaging them would mix rooms). With labels for every file the counts
  of ISO 354:2003 7.1.4 are checked, per condition when the two folders differ. The
  usual case is the same labels in both folders (same positions with and without the
  specimen; user 2026-11-06): one check for both. Fully labelled folders with different
  position sets get a note naming the labels found in only one of them (design choice,
  not a clause; the means T1 and T2 are still taken over each folder, 8.1.1).
- **Reproducibility:** the evaluated setup can be saved as the setup TOML of
  `resonus-room absorption`, so a GUI result can be recomputed from the command line.
- **GUI** (tab "Absorption", setting `show_absorption`, off by default): the inputs of one
  evaluation live in the tab, not in Preferences: volume, specimen area or number of
  objects, optional largest room dimension (6.1.2 check), temperature and relative
  humidity per condition without defaults (they must be measured), pressure default
  101.325 kPa (as `load_setup`). The analysis settings (filter direction, onset, ...)
  are those of Preferences; bands are fixed to one-third octaves 100 Hz-5 kHz
  (ISO 354:2003 clause 5, `position_t20`).

## D-33: Clarity tab with inner tabs; Measure/Stop one width (user, 2026-11-06)
User: "In the clarity and definition tab, the plots are a bit too wide. There is also too
much on a single frame. Instead of the visibility selection make a second level of tabs
inside the frame, the room average and other selectors can stay next to it."
- The Clarity tab has inner tabs (`Adw.InlineViewSwitcher`) **Clarity** (C50, C80),
  **Definition** (D50) and **Centre time** (Ts), each one plot with its own toolbar and
  title, replacing the C50/C80/D50/Ts check buttons and the shared figure of D-29. Room
  average, band switch and Table stay on the right of the bar. Export of a plot format
  writes the visible inner tab. C50 and C80 share one plot (same quantity and unit,
  ISO 3382-1:2009 Table A.1); separate inner tabs for them are a one-line change.
- Measure and Stop: both labels in one homogeneous `Gtk.Stack` inside the button, so the
  button keeps the width of the longer label (user, same day).

## D-34: STI speech levels per vocal effort (user, 2026-11-06; Q-STI3)
User: "the Odeon application note lists spectra for different vocal effort. I would use
those". Source: J. H. Rindel, ODEON Application Note "Calculation of Speech Transmission
Index in rooms", February 2014, Table 1 (ANSI S3.5 speech SPL at 1 m, 250 Hz-8 kHz).
- 125 Hz: 250 Hz level - 6 dB, the rule of the note's text p. 3 (user choice; the note's
  sound power rows would give about 4 dB less).
- Shouted (82.3 dB(A)) lowered by 0.4 (82.3 - 75) = 2.92 dB in every band: ISO 9921:2003
  Annex A as quoted by the note p. 4 (user choice; ISO 9921 itself not in the library).
- Setting `sti_vocal_effort` (custom, normal, raised, loud, shouted); default "normal"
  (design choice: the note's first row, BB93 student-to-student); "custom" uses
  `sti_speech_levels`. Transcription checked: the A-weighted sums of the rows with IEC
  61672-1:2013 Table 3 reproduce the table's A-weighted column within 0.05 dB.
- Q-STI5 (per position plus room mean) and Q-STI6 (male weighting first) answered yes by
  the user the same day; Q-STI4 (masking, reception threshold) needs IEC 60268-16:2020
  Annex A or Steeneken & Houtgast 1999 (Q-STI1, Q-STI9).

## D-35: ISO 354 compliance findings never block the evaluation (user, 2026-11-06)

"I would like to calculate the absorption no matter what standard compliancy is met - the
issues should be displayed and recorded, but should not block the analysis."
- `run_absorption` already collects the 6.1.4, 6.2.1.1, 7.1.4 and other checks into
  `AbsorptionResult.issues` without raising; only unusable input (no folder, no IR, a
  repeated label, a non-number in the form) stops an evaluation.
- GUI: the plot and table are always drawn; the findings are in a pop-up panel (D-36)
  behind a "Findings (N)" toggle, so they never take the plot's height (the cause of the
  report: 15 wrapped lines below the plot squeezed it on the user's 1 m3 box data).
- Temperature outside 15...30 degC (user, 2026-11-06: "I get errors about the
  temperature range. That and RH should also go to findings"): resonus-core
  `speed_of_sound` raises there (ISO 354:2003 8.1.2.1 NOTE gives Eq. (6) for that range);
  the app extrapolates c = 331 + 0.6 t and adds the finding "8.1.2.1: ... extrapolated"
  (`speed_of_sound_reported`). RH outside 30...90 % was already a 6.3.2 finding. Climate
  findings now start with "empty room: " or "with specimen: ". ISO 9613-1:1993 clause 7
  gives accuracy estimates, not limits; the core air attenuation only refuses RH outside
  0...100 % and p <= 0 (physically invalid, still errors).
- Recorded: JSON (`issues`, `notes_folders`), HDF5 extra metadata (`issues`, `notes`,
  `notes_folders`), CLI warnings on stderr. The CSV stays the plain band table.

## D-36: Pop-up panels for tables and findings (user, 2026-11-06)

"change findings to a pop-up similar to the Table and toggled next to it ... Finetune the
Table size ... for all such tables through the app, make the background less
transparent (also apply to the template if applicable)."
- `gui/overlays.py`: `overlay_panel` (hidden frame at the top right of the plot's
  `Gtk.Overlay`, CSS class `resonus-overlay`: 95 % view background, accent border,
  shadow), `band_table`, `findings_panel`, `fill_table`, `exclusive` (one panel at a
  time). The scroller takes the content's natural width (the overlay limits it to the plot) and height up to 480 px (layout
  choice), so a table shows all its columns; before, the absorption table had no width
  and showed about one column.
- Used by the RT, Clarity and Absorption tables and the Absorption findings ("Findings
  (N)" next to "Table"); the plot fills the right part of the Absorption tab.
- Also in the app template (skeleton `gui/overlays.py`, CSS in `layout.py`). A second
  app-local copy: part of the resonus-core layout proposal Q-C6.
- Table content to be revised by the user: Q-T2.

## D-37: resonus-core 0.11.0 (user, 2026-11-06: Q-C6 "move them", Q-C9 "add it", Q-C10 "fix it")

- Stop (Q-C10): `SounddeviceIO.playrec`/`rec` run on their own streams
  whose callback ends at the next block after `abort()`; `abort()` no longer calls
  `sounddevice.stop()` from another thread; a stream that does not finish within the
  signal length + 10 s raises instead of hanging. Room still calls `abort()` off the main
  loop (044c).
- `TimeSignal` (Q-C9): noise recordings (`record_noise`, Record noise…,
  `resonus-room record-noise`) are written as `TimeSignal` with the role key and
  `start_utc`; `noise_band_levels`, `sti_at_position`, the STI tab and `resonus-room sti`
  also read the older `ImpulseResponse` noise files (`NoiseRecording`).
- Layout (Q-C6): `resonus_core.gui.layout` (CSS, brand, ribbon, tab panel)
  and `resonus_core.gui.overlays` (pop-up panels, D-36) replace room's own copies
  (`plugin.py` helpers, `gui/overlays.py`); the ribbon now uses `ribbon(groups)`, so its
  groups share one vertical size group (captions on one line, as in the template).
- The app template is versioned; this app records its compliance in a private file.
