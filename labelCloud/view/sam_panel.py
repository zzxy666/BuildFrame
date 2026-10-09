"""SAM 配置和紧凑控制区；此模块不导入 torch。"""
from dataclasses import asdict
from PyQt5 import QtCore, QtWidgets
from ..model.segmentation.sam2_backend import SAMConfig


class SAMPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self);layout.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        self.settings = QtWidgets.QPushButton("SAM 设置")
        self.status = QtWidgets.QLabel("SAM: IDLE")
        self.status.setWordWrap(True)
        self.status.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        row.addWidget(self.settings);row.addWidget(self.status, 1);layout.addLayout(row)
        self.controls = QtWidgets.QWidget();grid = QtWidgets.QGridLayout(self.controls);grid.setContentsMargins(0, 0, 0, 0)
        self.reset = QtWidgets.QPushButton("重置提示 [R]")
        self.region = QtWidgets.QPushButton("重置 SAM 区域")
        self.accept = QtWidgets.QPushButton("接受 Mask → LiDAR [Enter]")
        self.cancel = QtWidgets.QPushButton("取消 [Esc]")
        self.next_mask = QtWidgets.QPushButton("下一 Mask")
        self.cpu = QtWidgets.QPushButton("改用 CPU 重试")
        for i, widget in enumerate((self.reset, self.region, self.accept, self.cancel, self.next_mask, self.cpu)):
            grid.addWidget(widget, i//2, i%2)
        layout.addWidget(self.controls);self.controls.hide();self.cpu.hide()
        self.accept.setEnabled(False)


def load_config():
    settings = QtCore.QSettings("BuildFrame", "SAM21")
    default = SAMConfig()
    try:
        return SAMConfig(str(settings.value("checkpoint", default.checkpoint)), str(settings.value("model_config", default.model_config)), str(settings.value("device", default.device)), int(settings.value("patch_size", default.patch_size)))
    except (ValueError, TypeError): return default


def configure(parent, config, reason=""):
    dialog = QtWidgets.QDialog(parent);dialog.setWindowTitle("SAM 2.1 可选后端设置")
    layout = QtWidgets.QFormLayout(dialog)
    if reason:
        label = QtWidgets.QLabel(reason);label.setWordWrap(True);layout.addRow(label)
    path = QtWidgets.QLineEdit(config.checkpoint);browse = QtWidgets.QPushButton("选择 .pt")
    row = QtWidgets.QHBoxLayout();row.addWidget(path);row.addWidget(browse)
    def choose():
        value, _ = QtWidgets.QFileDialog.getOpenFileName(dialog, "选择 SAM 2.1 权重", "", "Checkpoint (*.pt *.pth);;All files (*)")
        if value: path.setText(value)
    browse.clicked.connect(choose)
    layout.addRow("Checkpoint", row)
    model = QtWidgets.QLineEdit(config.model_config);layout.addRow("Model config", model)
    model.setToolTip("例如 configs/sam2.1/sam2.1_hiera_t.yaml；必须与 tiny/small/base_plus/large 权重匹配")
    device = QtWidgets.QComboBox();device.addItems(["auto", "cpu", "cuda"]);device.setCurrentText(config.device);layout.addRow("Device", device)
    size = QtWidgets.QComboBox();size.addItems(["1024", "1536", "2048"]);size.setCurrentText(str(config.patch_size));layout.addRow("Patch 像素上限", size)
    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject);layout.addRow(buttons)
    if dialog.exec_() != dialog.Accepted: return None
    result = SAMConfig(path.text().strip(), model.text().strip(), device.currentText(), int(size.currentText()))
    return result


def save_config(config):
    settings = QtCore.QSettings("BuildFrame", "SAM21")
    for key, value in asdict(config).items(): settings.setValue(key, value)
