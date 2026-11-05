# Files written by resonus-room

The HDF5 files use the resonus format of resonus-core (container, root attributes,
`metadata` JSON, the three result kinds): **resonus-core docs/FILE_FORMAT.md**. This page
lists what resonus-room puts into them.

| File | Written by | Kind | Content |
|---|---|---|---|
| `name.ir.h5` | GUI Export (D-16, Q-D12) | ImpulseResponse | the impulse response with its own metadata, unchanged |
| `name_<label>.ir.h5`, `name_<label>.wav` | GUI Export with several channels (Q-M2, D-17) | ImpulseResponse / WAV | one file per channel, `<label>` = `s#m#` (e.g. `bh_s1m1L.ir.h5`); analysis and plot formats export the channel shown |
| `impulse_response.h5` | CLI `measure --out` | ImpulseResponse | same as `name.ir.h5` |
| `frequency_response.h5` | CLI `measure --out` | FrequencyResponse | transfer function of the same measurement |
| `name.h5` (GUI), `analysis.h5` (CLI) | GUI Export, CLI `measure`/`analyze` | BandResults | per-band results, quantities `files.ANALYSIS_QUANTITIES` |
| `absorption.h5` | CLI `absorption` | BandResults | ISO 354 results, `files.ABSORPTION_QUANTITIES` |
| `binaural.h5`, `binaural.csv` | CLI `binaural --out` | BandResults / text | IACC early/late/full per octave band (ISO 3382-1:2009 Annex B), `files.BINAURAL_QUANTITIES`; metadata extra: `onset_s`, `early_limit_s`, `tau_max_s`; without `--out` the CSV goes to stdout |
| `average.h5` | CLI `average` | BandResults | room average over positions (`room_average.py`) |
| `.json`, `.csv` | GUI Export, CLI | text | full analysis / band table (`files.py`) |
| `.wav` | GUI Export | 32-bit float WAV | impulse response for other tools; no metadata |

The app opens `.ir.h5`, `impulse_response.h5` and `.wav` impulse responses (Open, Ctrl+O;
CLI `analyze`).

## Metadata of a measured impulse response (`.ir.h5`)

| Key | Content |
|---|---|
| `software`, `software_version` | `"resonus-room"` and the app version |
| `created` | time of the measurement, UTC |
| `device` | audio device as described by the backend (name, sample rate, channels; for the simulated room also its parameters) |
| `description` | empty for a GUI measurement; the file name for an opened WAV |
| `standards` | empty (the impulse response itself is not computed to a standard) |
| `extra.excitation` | excitation kind |
| `extra.channel` | `input`, `input_label`, `outputs`, `output_labels`, `label` (`s#m#`, e.g. `s1m1L`; `s1+s2m1` when several outputs played at once); resonus-core 0.5.0, its docs/FILE_FORMAT.md |
| `extra.settings` | all settings **at the time of the measurement** (table below) |

An impulse response opened from a WAV file has only `software`, `software_version`,
`created` (time of opening) and `description` (file name); exporting it as `.ir.h5`
keeps exactly that, no measurement settings are invented.

The analysis settings in `extra.settings` are those active during the measurement. If the
bands or the analysis method are changed afterwards in the GUI, the exported band results
(`.h5`, `.json`) use the new values; the `.ir.h5` still records the measurement-time
settings.

### `extra.settings` (`core/settings.py`, `AppSettings`)

| Key | Unit | Meaning |
|---|---|---|
| `excitation` | - | `exp_sweep`, `sync_sweep`, `linear_sweep`, `white_noise`, `pink_noise`, `mls` |
| `sample_rate` | Hz | sample rate |
| `f_start`, `f_stop` | Hz | sweep start and stop frequency |
| `duration` | s | sweep length, or the period of noise/MLS |
| `level` | dB re 1 | peak level (sweeps, MLS) or RMS level (noise), digital full scale |
| `fade_in`, `fade_out` | s | sweep fades |
| `tail` | s | silence after each sweep |
| `repetitions` | - | number of averages |
| `ir_length` | s | length of the stored impulse response |
| `reference_channel` | - | device input channel with the loopback reference; 0 = none |
| `method` | - | deconvolution: `auto`, `inverse`, `regularized`, `novak` |
| `input_device`, `output_device` | - | device key `"<host API>: <name>"`; empty = default device (files before resonus-core 0.5.0 have `device`, a PortAudio name) |
| `input_channels`, `input_labels` | - | device input channels (1-based) and their labels `m#` / `m#L` / `m#R` |
| `output_channels`, `output_labels` | - | device output channels and their labels `s#` |
| `output_mode` | - | `simultaneous` (all outputs at once) or `sequential` (one after another) |
| `simulate` | - | `true` if the built-in simulated room was used |
| `bands` | - | `octave` or `third` |
| `f_min`, `f_max` | Hz | lowest and highest band analysed |
| `filter_order` | - | Butterworth band filter order (PROVISIONAL, IEC 61260-1 not verified) |
| `reverse_filter` | - | time-reversed band filtering (ISO 3382-2:2008 7.3 NOTE) |
| `decay_method` | - | noise compensation: `none`, `truncate`, `correct` (ISO 3382-1:2009 5.3.3) |
| `position_method` | - | method of the GUI room-average check, ISO 3382-2:2008 Table 1: `survey`, `engineering`, `precision`, `correction` (default `engineering`, Q-O1) |
| `sim_decay_time` | s | reverberation time of the simulated room |
| `sim_noise` | dB re 1 | background noise level of the simulated room |

## Not recorded (yet)

Room name, source and microphone position, temperature and humidity, operator,
equipment serial numbers and calibration are not part of the metadata. The samples are
uncalibrated (`unit = "1"`); level calibration is deferred in resonus-core (open item).
