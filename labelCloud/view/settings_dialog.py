import logging
from pathlib import Path

import pkg_resources
from PyQt5 import uic
from PyQt5.QtWidgets import QDialog

from ..control.config_manager import config, config_manager


class SettingsDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.parent_gui = parent
        uic.loadUi(
            pkg_resources.resource_filename(
                "labelCloud.resources.interfaces", "settings_interface.ui"  
            ),
            self,
        )
        self.fill_with_current_settings()

        self.buttonBox.accepted.connect(self.save)
        self.buttonBox.rejected.connect(self.cancel)
        self.reset_button.clicked.connect(self.reset)

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

        config_manager.write_into_file()
        self.parent_gui.set_checkbox_states()  # 刷新界面（如地板显示）
        logging.info("设置已保存！")

        # 刷新背景色（立即生效）
        self.parent_gui.gl_widget.initializeGL()

    def reset(self) -> None:
        config_manager.reset_to_default()
        self.fill_with_current_settings()
        logging.info("设置已重置为默认")

    def cancel(self) -> None:  # 修正拼写：chancel → cancel
        logging.info("设置对话框已取消")