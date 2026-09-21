"""Check dialog lifecycle in its own Qt process, separate from QCoreApp tests."""

import os
import subprocess
import sys
import textwrap


def test_filter_settings_cancel_save_failure_and_live_language_switch():
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            from types import SimpleNamespace
            from PyQt5 import QtWidgets
            from labelCloud.control.config_manager import ExtendedConfigParser
            from labelCloud.view import ground_filter_dialog as ui
            from labelCloud.view.i18n import language_manager

            app = QtWidgets.QApplication([])
            clean_settings = ExtendedConfigParser()
            clean_settings.read_dict({"UNRELATED": {"value": "untouched"}})
            ui.config = clean_settings
            initial_writes = []
            ui.config_manager = SimpleNamespace(write_into_file=lambda: initial_writes.append(True))
            dialog = ui.GroundFilterDialog()
            assert dialog.resolution_spin.value() == 1.0
            assert dialog.units_spin.value() == 1.0
            assert dialog.time_step_spin.value() == 0.65
            assert dialog.threshold_spin.value() == 0.5
            assert dialog.terrain_combo.currentData() == 3
            assert dialog.slope_checkbox.isChecked()
            assert dialog.iterations_spin.value() == 500
            assert not clean_settings.has_section("GROUND_FILTER")
            dialog.save()
            assert initial_writes == [True]
            assert clean_settings.get("UNRELATED", "value") == "untouched"
            assert clean_settings.getfloat("GROUND_FILTER", "resolution") == 1.0
            assert clean_settings.getfloat("GROUND_FILTER", "units_per_meter") == 1.0

            settings = ExtendedConfigParser()
            settings.read_dict({
                "GROUND_FILTER": {
                    "resolution": "2.0", "threshold": "0.25",
                    "rigidness": "2", "slope_smooth": "False",
                    "iterations": "250", "time_step": "0.5",
                    "units_per_meter": "1000",
                    "custom_option": "preserve-me",
                },
                "UNRELATED": {"value": "untouched"},
            })
            ui.config = settings
            original = {section: dict(settings[section]) for section in settings.sections()}
            writes = []
            ui.config_manager = SimpleNamespace(write_into_file=lambda: writes.append(True))

            dialog = ui.GroundFilterDialog()
            dialog.resolution_spin.setValue(3.5)
            dialog.units_spin.setValue(100)
            dialog.time_step_spin.setValue(0.1)
            dialog.reset()
            assert dialog.units_spin.value() == 1.0
            assert dialog.time_step_spin.value() == 0.65
            dialog.reject()
            assert writes == []
            assert original == {section: dict(settings[section]) for section in settings.sections()}

            dialog = ui.GroundFilterDialog()
            assert dialog.units_spin.value() == 1000
            assert dialog.time_step_spin.value() == 0.5
            dialog.terrain_combo.setCurrentIndex(dialog.terrain_combo.findData(1))
            dialog.resolution_spin.setValue(0.25)
            dialog.units_spin.setValue(100)
            dialog.time_step_spin.setValue(0.1)
            language_manager.set_language("en_US")
            app.processEvents()
            assert dialog.windowTitle() == "Ground Filter Settings"
            assert dialog.terrain_combo.currentData() == 1
            assert dialog.terrain_combo.currentText() == "Steep / Rugged Terrain"
            assert dialog.resolution_spin.value() == 0.25
            assert dialog.units_spin.value() == 100
            assert dialog.time_step_spin.value() == 0.1
            assert dialog.units_label.text() == "Source Units per Metre"
            language_manager.set_language("zh_CN")
            app.processEvents()
            assert dialog.windowTitle() == "地面滤波设置"
            assert dialog.terrain_combo.currentData() == 1
            dialog.save()
            assert dialog.result() == QtWidgets.QDialog.Accepted
            assert writes == [True]
            assert settings.getfloat("GROUND_FILTER", "resolution") == 0.25
            assert settings.getint("GROUND_FILTER", "rigidness") == 1
            assert settings.getfloat("GROUND_FILTER", "time_step") == 0.1
            assert settings.getfloat("GROUND_FILTER", "units_per_meter") == 100
            assert settings.get("UNRELATED", "value") == "untouched"

            saved = {section: dict(settings[section]) for section in settings.sections()}
            def fail_write():
                raise PermissionError("read-only configuration")
            warnings = []
            ui.config_manager.write_into_file = fail_write
            ui.QtWidgets.QMessageBox.warning = lambda *args: warnings.append(args)
            dialog = ui.GroundFilterDialog()
            dialog.show()
            dialog.threshold_spin.setValue(0.9)
            dialog.save()
            assert dialog.isVisible()
            assert dialog.result() != QtWidgets.QDialog.Accepted
            assert len(warnings) == 1
            assert saved == {section: dict(settings[section]) for section in settings.sections()}
            dialog.reject()
            app.processEvents()
        """)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "BUILDFRAME_LOG_PATH": "NUL"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
