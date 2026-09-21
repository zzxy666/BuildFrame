"""Ground/non-ground filtering options, edited without changing live settings."""

import logging

from PyQt5 import QtCore, QtWidgets

from PointCloudFilter.csf_filter import CSFOptions

from ..control.config_manager import config, config_manager
from .i18n import tr


class GroundFilterDialog(QtWidgets.QDialog):
    """Edit CSF options; accept only after validated settings are saved."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(490)
        self.resize(680, 660)
        self._invalid_config = False

        layout = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        content = QtWidgets.QWidget(scroll)
        content_layout = QtWidgets.QVBoxLayout(content)
        self.intro_label = self._wrapped_label(content)
        content_layout.addWidget(self.intro_label)

        form = QtWidgets.QGridLayout()
        form.setColumnStretch(0, 1)
        form.setColumnStretch(1, 2)
        form.setVerticalSpacing(12)
        self.units_label = self._wrapped_label(content)
        self.units_spin = self._distance_spin(content)
        self.units_spin.setSingleStep(1.0)
        form.addWidget(self.units_label, 0, 0)
        form.addWidget(self.units_spin, 0, 1)
        self.resolution_label = self._wrapped_label(content)
        self.resolution_spin = self._distance_spin(content)
        form.addWidget(self.resolution_label, 1, 0)
        form.addWidget(self.resolution_spin, 1, 1)
        self.threshold_label = self._wrapped_label(content)
        self.threshold_spin = self._distance_spin(content)
        form.addWidget(self.threshold_label, 2, 0)
        form.addWidget(self.threshold_spin, 2, 1)
        self.terrain_label = self._wrapped_label(content)
        self.terrain_combo = QtWidgets.QComboBox(content)
        self.terrain_combo.addItem("", 3)
        self.terrain_combo.addItem("", 2)
        self.terrain_combo.addItem("", 1)
        form.addWidget(self.terrain_label, 3, 0)
        form.addWidget(self.terrain_combo, 3, 1)
        self.slope_checkbox = QtWidgets.QCheckBox(content)
        form.addWidget(self.slope_checkbox, 4, 0, 1, 2)
        self.iterations_label = self._wrapped_label(content)
        self.iterations_spin = QtWidgets.QSpinBox(content)
        self.iterations_spin.setRange(1, 10000)
        self.iterations_spin.setKeyboardTracking(False)
        form.addWidget(self.iterations_label, 5, 0)
        form.addWidget(self.iterations_spin, 5, 1)
        self.time_step_label = self._wrapped_label(content)
        self.time_step_spin = QtWidgets.QDoubleSpinBox(content)
        self.time_step_spin.setDecimals(3)
        self.time_step_spin.setRange(0.01, 1.0)
        self.time_step_spin.setSingleStep(0.05)
        self.time_step_spin.setKeyboardTracking(False)
        form.addWidget(self.time_step_label, 6, 0)
        form.addWidget(self.time_step_spin, 6, 1)
        content_layout.addLayout(form)

        self.help_label = self._wrapped_label(content)
        self.help_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        content_layout.addWidget(self.help_label)
        self.invalid_config_label = self._wrapped_label(content)
        content_layout.addWidget(self.invalid_config_label)
        content_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll)

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel,
            parent=self,
        )
        self.defaults_button = self.button_box.addButton(
            "", QtWidgets.QDialogButtonBox.ResetRole
        )
        layout.addWidget(self.button_box)
        self.button_box.accepted.connect(self.save)
        self.button_box.rejected.connect(self.reject)
        self.defaults_button.clicked.connect(self.reset)

        try:
            options = CSFOptions.from_config(config)
            options.validate()
        except (ValueError, TypeError):
            logging.warning("Invalid ground-filter settings; showing CSF defaults.")
            self._invalid_config = True
            options = CSFOptions()
        self._fill(options)
        self._retranslate_ui()

    @staticmethod
    def _wrapped_label(parent):
        label = QtWidgets.QLabel(parent)
        label.setWordWrap(True)
        label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        return label

    @staticmethod
    def _distance_spin(parent):
        spin = QtWidgets.QDoubleSpinBox(parent)
        spin.setDecimals(6)
        spin.setRange(0.000001, 1000000000.0)
        spin.setSingleStep(0.1)
        spin.setKeyboardTracking(False)
        spin.setMinimumWidth(170)
        return spin

    def _fill(self, options: CSFOptions) -> None:
        self.units_spin.setValue(options.units_per_meter)
        self.resolution_spin.setValue(options.resolution)
        self.threshold_spin.setValue(options.threshold)
        self.terrain_combo.setCurrentIndex(
            self.terrain_combo.findData(options.rigidness)
        )
        self.slope_checkbox.setChecked(options.slope_smooth)
        self.iterations_spin.setValue(options.iterations)
        self.time_step_spin.setValue(options.time_step)

    def reset(self) -> None:
        """Reset only the pending form; Cancel still leaves config unchanged."""
        self._fill(CSFOptions())

    def save(self) -> None:
        for spin in (
            self.units_spin,
            self.resolution_spin,
            self.threshold_spin,
            self.iterations_spin,
            self.time_step_spin,
        ):
            spin.interpretText()
        try:
            options = CSFOptions(
                units_per_meter=self.units_spin.value(),
                resolution=self.resolution_spin.value(),
                threshold=self.threshold_spin.value(),
                rigidness=self.terrain_combo.currentData(),
                slope_smooth=self.slope_checkbox.isChecked(),
                iterations=self.iterations_spin.value(),
                time_step=self.time_step_spin.value(),
            )
            options.validate()
        except (ValueError, TypeError) as exc:
            QtWidgets.QMessageBox.warning(
                self, tr("滤波参数无效"), tr("请检查滤波参数：\n{error}", error=str(exc))
            )
            return

        previous = (
            dict(config["GROUND_FILTER"])
            if config.has_section("GROUND_FILTER")
            else None
        )
        try:
            options.to_config(config)
            config_manager.write_into_file()
        except Exception:
            config.remove_section("GROUND_FILTER")
            if previous is not None:
                config["GROUND_FILTER"] = previous
            logging.exception("Could not save ground-filter settings.")
            QtWidgets.QMessageBox.warning(
                self,
                tr("保存失败"),
                tr("无法保存地面滤波设置，请检查配置文件的写入权限和磁盘空间。"),
            )
            return
        self.accept()

    def _retranslate_ui(self) -> None:
        self.setWindowTitle(tr("地面滤波设置"))
        self.intro_label.setText(
            tr("使用 CSF 布料模拟滤波分离地面和非地面点。保存后，下次执行地面滤波时使用新参数。")
        )
        self.units_label.setText(tr("单位换算系数（原始单位 / 米）"))
        self.units_spin.setToolTip(
            tr("1 米对应多少个原始坐标单位：米填 1，厘米填 100，毫米填 1000。若采用未知的自定义比例，请先核实实际尺寸，程序不会自动猜测。")
        )
        self.resolution_label.setText(tr("布料分辨率（米）"))
        self.resolution_spin.setToolTip(
            tr("以米输入布料网格间距。减小可保留更多地形细节，但会增加内存和计算时间；大范围点云宜适当增大。")
        )
        self.threshold_label.setText(tr("分类距离阈值（米）"))
        self.threshold_spin.setToolTip(
            tr("以米输入点到拟合地面的距离阈值。增大会将更多点归为地面，也可能误分低矮物体。")
        )
        self.terrain_label.setText(tr("地形预设"))
        for index, source in enumerate(("城市 / 平坦地形", "缓坡地形", "陡坡 / 起伏地形")):
            self.terrain_combo.setItemText(index, tr(source))
        self.terrain_combo.setToolTip(
            tr("仅调整布料硬度：平坦为 3、缓坡为 2、陡坡为 1。仅切换地形不够，还需结合密度和分类预览调整分辨率、距离阈值，必要时调整时间步长。")
        )
        self.slope_checkbox.setText(tr("启用坡面后处理"))
        self.slope_checkbox.setToolTip(tr("补充识别陡坡附近的地面点；建议结合分类预览判断是否启用。"))
        self.iterations_label.setText(tr("最大迭代数"))
        self.iterations_spin.setToolTip(tr("布料模拟的最大迭代次数；通常从 500 开始。"))
        self.time_step_label.setText(tr("模拟时间步长"))
        self.time_step_spin.setToolTip(
            tr("布料模拟的时间步长，默认 0.65。调整后需重新检查分类结果，并结合分辨率和迭代次数判断。")
        )
        self.help_label.setText(
            tr(
                "先确认坐标单位：Z 为高程，XY 对应地面平面，三个轴使用一致的长度单位。"
                "单位换算系数：米填 1，厘米填 100，毫米填 1000；未知比例须先核实实际尺寸，程序不会自动猜测。\n\n"
                "分辨率和阈值始终以米输入，默认 1.0 / 0.5 只是调参起点。仅切换地形不够，"
                "需结合密度和预览调整分辨率、阈值，陡坡可能还需调整时间步长。"
                "缺少地面、室内、多层结构等场景不能保证正确分离，请检查分类预览。"
            )
        )
        self.invalid_config_label.setText(tr("当前滤波配置无效，已在表单中显示默认值；保存前不会修改配置。"))
        self.invalid_config_label.setVisible(self._invalid_config)
        self.defaults_button.setText(tr("恢复默认设置"))
        self.button_box.button(QtWidgets.QDialogButtonBox.Save).setText(tr("保存"))
        self.button_box.button(QtWidgets.QDialogButtonBox.Cancel).setText(tr("取消"))
        for button in self.button_box.buttons():
            button.setMinimumWidth(max(90, button.fontMetrics().horizontalAdvance(button.text()) + 28))

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.LanguageChange and hasattr(self, "button_box"):
            self._retranslate_ui()
            QtCore.QTimer.singleShot(0, self._retranslate_ui)
