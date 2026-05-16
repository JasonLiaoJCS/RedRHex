"""Tests for PresetStore — independently runnable, no Isaac Sim required."""
from __future__ import annotations

import json
import pytest

from tools.training_panel.training_panel.presets import BUILT_IN_PRESETS, PresetStore


@pytest.fixture()
def store(tmp_path):
    return PresetStore(tmp_path / "reward_presets.json")


def test_built_in_presets_are_loaded(store):
    presets = store.list_presets()
    ids = [p["id"] for p in presets]
    assert "baseline" in ids
    assert "speed-focus" in ids
    assert "stability-focus" in ids


def test_get_preset_returns_correct_preset(store):
    p = store.get_preset("baseline")
    assert p is not None
    assert p["name"] == "Baseline"
    assert p["built_in"] is True


def test_get_nonexistent_preset_returns_none(store):
    assert store.get_preset("does-not-exist") is None


def test_create_preset_stores_values(store):
    p = store.create_preset("My Preset", "Test description", {"rew_scale_forward_vel": 5.0})
    assert p["id"]
    assert p["values"]["rew_scale_forward_vel"] == 5.0
    assert p["built_in"] is False
    # Persists across reload
    p2 = store.get_preset(p["id"])
    assert p2 is not None
    assert p2["values"]["rew_scale_forward_vel"] == 5.0


def test_create_preset_slugifies_name(store):
    p = store.create_preset("My Cool Preset!", "desc", {})
    assert p["id"] == "my-cool-preset"


def test_create_preset_generates_unique_ids(store):
    p1 = store.create_preset("Dup", "desc", {})
    p2 = store.create_preset("Dup", "desc", {})
    assert p1["id"] != p2["id"]


def test_update_preset_changes_values(store):
    p = store.create_preset("Edit Me", "original desc", {"rew_scale_alive": 0.5})
    updated = store.update_preset(p["id"], values={"rew_scale_alive": 1.0}, name="Renamed", description="updated desc")
    assert updated["name"] == "Renamed"
    assert updated["values"]["rew_scale_alive"] == 1.0
    assert updated["description"] == "updated desc"


def test_update_built_in_preset_raises(store):
    with pytest.raises(ValueError, match="Cannot modify built-in"):
        store.update_preset("baseline", values={"rew_scale_alive": 9.9})


def test_delete_built_in_preset_raises(store):
    with pytest.raises(ValueError, match="Cannot delete built-in"):
        store.delete_preset("baseline")


def test_delete_custom_preset(store):
    p = store.create_preset("Temp", "temp", {})
    deleted = store.delete_preset(p["id"])
    assert deleted is True
    assert store.get_preset(p["id"]) is None


def test_delete_nonexistent_returns_false(store):
    assert store.delete_preset("ghost") is False


def test_active_preset_defaults_to_baseline(store):
    assert store.get_active_preset_id() == "baseline"


def test_set_active_preset_persists(store, tmp_path):
    p = store.create_preset("Active Test", "desc", {})
    store.set_active_preset(p["id"])
    assert store.get_active_preset_id() == p["id"]
    # Reload from file to confirm persistence
    store2 = PresetStore(tmp_path / "reward_presets.json")
    assert store2.get_active_preset_id() == p["id"]


def test_set_active_preset_to_nonexistent_raises(store):
    with pytest.raises(KeyError):
        store.set_active_preset("nonexistent-preset")


def test_active_preset_resets_to_baseline_after_deletion(store):
    p = store.create_preset("Will Delete", "desc", {})
    store.set_active_preset(p["id"])
    store.delete_preset(p["id"])
    assert store.get_active_preset_id() == "baseline"


def test_store_merges_built_ins_on_reload(tmp_path):
    """If reward_presets.json exists but is missing a built-in, it's added on load."""
    preset_file = tmp_path / "reward_presets.json"
    preset_file.write_text(json.dumps({
        "active_preset_id": "baseline",
        "presets": [{"id": "baseline", "name": "Baseline", "built_in": True, "values": {}, "description": ""}],
    }))
    store = PresetStore(preset_file)
    ids = [p["id"] for p in store.list_presets()]
    assert "speed-focus" in ids
    assert "stability-focus" in ids
