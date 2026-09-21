import logging
from importlib import resources
from pathlib import Path

from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5.QtWidgets import QDialog, QDialogButtonBox

from ..control.config_manager import config, config_manager
from .i18n import (
    LANGUAGE_CHINESE,
    LANGUAGE_ENGLISH,
    language_manager,
    normalize_language,
    tr,
)


_SETTINGS_UI, _ = uic.loadUiType(
    str(
        resources.files("labelCloud.resources.interfaces").joinpath(
            "settings_interface.ui"
        )
    )
)


class SettingsDialog(QDialog, _SETTINGS_UI):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.parent_gui = parent
        self.setupUi(self)
        self.reset_button.setMinimumWidth(165)
        self.buttonBox.button(QDialogButtonBox.Save).setMinimumWidth(130)
        self.buttonBox.button(QDialogButtonBox.Cancel).setMinimumWidth(130)
        self.fill_with_current_settings()
        self._retranslate_custom_widgets()

        self.buttonBox.accepted.connect(self.save)
        self.buttonBox.rejected.connect(self.cancel)
        self.reset_button.clicked.connect(self.reset)

    def _retranslate_custom_widgets(self) -> None:
        selected_language = self.comboBox_language.currentData()
        if selected_language is None:
            selected_language = normalize_language(
                config.get("USER_INTERFACE", "language", fallback=LANGUAGE_CHINESE)
            )
        self.comboBox_language.blockSignals(True)
        self.comboBox_language.clear()
        self.comboBox_language.addItem(tr("中文"), LANGUAGE_CHINESE)
        self.comboBox_language.addItem(tr("英文"), LANGUAGE_ENGLISH)
        index = self.comboBox_language.findData(selected_language)
        self.comboBox_language.setCurrentIndex(max(index, 0))
        self.comboBox_language.blockSignals(False)
        self.buttonBox.button(QDialogButtonBox.Save).setText(tr("保存"))
        self.buttonBox.button(QDialogButtonBox.Cancel).setText(tr("取消"))

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.LanguageChange and hasattr(
            self, "comboBox_language"
        ):
            self.retranslateUi(self)
            self._retranslate_custom_widgets()
            QtCore.QTimer.singleShot(0, self._retranslate_custom_widgets)

    def fill_with_current_settings(self) -> None:
        # File
        self.lineEdit_pointcloudfolder.setText(config.get("FILE", "pointcloud_folder"))
        self.lineEdit_labelfolder.setText(config.get("FILE", "label_folder"))

        # Pointcloud
        self.doubleSpinBox_pointsize.setValue(config.getfloat("POINTCLOUD", "POINT_SIZE"))
        self.lineEdit_pointcolor.setText(config["POINTCLOUD"]["colorless_color"])
        self.checkBox_colorizecolorless.setChecked(config.getboolean("POINTCLOUD", "colorless_colorize"))
        self.doubleSpinBox_standardtranslation.setValue(config.getfloat("POINTCLOUD", "std_translation"))
        self.doubleSpinBox_standardzoom.setValue(config.getfloat("POINTCLOUD", "std_zoom"))

        # User Interface（只保留有用的）
        self.checkBox_showfloor.setChecked(config.getboolean("USER_INTERFACE", "show_floor"))
        self.lineEdit_backgroundcolor.setText(config["USER_INTERFACE"]["background_color"])
        language = normalize_language(
            config.get("USER_INTERFACE", "language", fallback=LANGUAGE_CHINESE)
        )
        if self.comboBox_language.count() == 0:
            self._retranslate_custom_widgets()
        index = self.comboBox_language.findData(language)
        self.comboBox_language.setCurrentIndex(max(index, 0))

    def save(self) -> None:
        # File
        config["FILE"]["pointcloud_folder"] = self.lineEdit_pointcloudfolder.text()
        config["FILE"]["label_folder"] = self.lineEdit_labelfolder.text()

        # Pointcloud
        config["POINTCLOUD"]["point_size"] = str(self.doubleSpinBox_pointsize.value())
        config["POINTCLOUD"]["colorless_color"] = self.lineEdit_pointcolor.text()
        config["POINTCLOUD"]["colorless_colorize"] = str(self.checkBox_colorizecolorless.isChecked())
        config["POINTCLOUD"]["std_translation"] = str(self.doubleSpinBox_standardtranslation.value())
        config["POINTCLOUD"]["std_zoom"] = str(self.doubleSpinBox_standardzoom.value())

        # User Interface
        config["USER_INTERFACE"]["show_floor"] = str(self.checkBox_showfloor.isChecked())
        config["USER_INTERFACE"]["background_color"] = self.lineEdit_backgroundcolor.text()
        language = normalize_language(self.comboBox_language.currentData())
        config["USER_INTERFACE"]["language"] = language

        config_manager.write_into_file()
        language_manager.set_language(language)
        if self.parent_gui is not None:
            self.parent_gui.set_checkbox_states()
        logging.info("设置已保存。")

        if self.parent_gui is not None:
            self.parent_gui.gl_widget.initializeGL()

    def reset(self) -> None:
        config_manager.reset_to_default()
        self.fill_with_current_settings()
        logging.info("设置已重置为默认")

    def cancel(self) -> None:  # 修正拼写：chancel → cancel
        logging.info("设置对话框已取消")
