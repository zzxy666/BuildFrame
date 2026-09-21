import importlib

import pytest


def test_failed_config_replace_preserves_existing_file(tmp_path, monkeypatch):
    module = importlib.import_module("labelCloud.control.config_manager")
    path = tmp_path / "config.ini"
    path.write_text("[FILE]\npointcloud_folder = original\n")
    monkeypatch.setattr(module.ConfigManager, "PATH_TO_CONFIG", path)
    manager = module.ConfigManager()
    manager.config.set("FILE", "pointcloud_folder", "updated")

    def fail(*args):
        raise PermissionError("simulated replace failure")

    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(PermissionError):
        manager.write_into_file()
    assert "original" in path.read_text()
    assert list(tmp_path.iterdir()) == [path]


def test_config_success_writes_complete_file(tmp_path, monkeypatch):
    module = importlib.import_module("labelCloud.control.config_manager")
    path = tmp_path / "config.ini"
    path.write_text("[FILE]\npointcloud_folder = original\n")
    monkeypatch.setattr(module.ConfigManager, "PATH_TO_CONFIG", path)
    manager = module.ConfigManager()
    manager.config.set("FILE", "pointcloud_folder", "updated")
    manager.write_into_file()
    assert "updated" in path.read_text()
    assert list(tmp_path.iterdir()) == [path]
