"""AppSettings: channel labels s#m#, migration of the former device field."""

from dataclasses import fields

import pytest
from resonus_core.devices import DeviceInfo

from resonus_room.core.settings import AppSettings, load_settings, migrate_settings

DEVICES = [
    DeviceInfo(0, "UR22", "ALSA", 2, 2, 44100.0),
    DeviceInfo(1, "UR22", "JACK", 2, 2, 48000.0),
    DeviceInfo(2, "HDMI", "ALSA", 0, 8, 48000.0),
]


def test_defaults_and_groups():
    s = AppSettings()
    assert s.input_labels == ("m1",) and s.output_labels == ("s1",)
    assert s.input_device == s.output_device == ""
    assert "device" not in s.to_dict()
    g = {f.name: f.metadata.get("group") for f in fields(AppSettings)}
    assert g["ir_length"] == "Processing"  # measurement-time setting (D-17)
    assert g["bands"] == "Analysis"


@pytest.mark.parametrize(
    "labels",
    [("m1",), ("m1", "m2"), ("m1L", "m1R"), ("m12",), ("m1L", "m1R", "m2")],
)
def test_valid_input_labels(labels):
    AppSettings(input_channels=tuple(range(1, len(labels) + 1)), input_labels=labels)


@pytest.mark.parametrize("label", ["m0", "m", "M1", "mic1", "m1l", "m1LR", "s1", "m01"])
def test_invalid_input_labels(label):
    with pytest.raises(ValueError, match="input label"):
        AppSettings(input_labels=(label,))


@pytest.mark.parametrize("label", ["s0", "s", "S1", "s1L", "m1", "src1", "s01"])
def test_invalid_output_labels(label):
    with pytest.raises(ValueError, match="output label"):
        AppSettings(output_labels=(label,))


def test_valid_output_labels():
    AppSettings(output_channels=(1, 2), output_labels=("s1", "s2"))


def no_devices():
    raise AssertionError("devices() must not be called")


def test_migrate_nothing_to_do():
    data = {"repetitions": 2}
    assert migrate_settings(data, no_devices) == ({"repetitions": 2}, [])
    assert migrate_settings({"device": ""}, no_devices) == ({}, [])


def test_migrate_unique_name():
    data = {"device": "HDMI", "repetitions": 2}
    new, messages = migrate_settings(data, lambda: DEVICES)
    assert new == {"repetitions": 2, "input_device": "ALSA: HDMI", "output_device": "ALSA: HDMI"}
    assert messages == [] and "device" in data  # input not modified


def test_migrate_keeps_explicit_new_keys():
    new, _ = migrate_settings({"device": "HDMI", "input_device": "JACK: UR22"}, lambda: DEVICES)
    assert new == {"input_device": "JACK: UR22", "output_device": "ALSA: HDMI"}


@pytest.mark.parametrize(
    ("name", "words"), [("UR22", "several host APIs"), ("Scarlett", "not found")]
)
def test_migrate_ambiguous_or_missing_uses_default(name, words):
    new, messages = migrate_settings({"device": name}, lambda: DEVICES)
    assert new == {}
    assert len(messages) == 1 and words in messages[0] and repr(name) in messages[0]
    assert "default device" in messages[0]


def test_load_settings(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text('device = "HDMI"\nrepetitions = 3\n', encoding="utf-8")
    s, messages = load_settings(path, lambda: DEVICES)
    assert s == AppSettings(repetitions=3, input_device="ALSA: HDMI", output_device="ALSA: HDMI")
    assert messages == []
    AppSettings(input_channels=(1, 2), input_labels=("m1L", "m1R")).save(path)
    s, messages = load_settings(path, no_devices)
    assert s.input_channels == (1, 2) and s.input_labels == ("m1L", "m1R") and not messages
    path.write_text("bogus = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_settings(path, no_devices)
