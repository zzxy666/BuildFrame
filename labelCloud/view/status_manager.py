from typing import Optional

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt
#from ..definitions import Context, Mode


from PyQt5.QtWidgets import QLabel

class StatusManager:
    def __init__(self, status_bar):
        # 模式标签（可选，如果你想保留左边的粗体模式名）
        self.mode_label = QLabel("导航模式")
        self.mode_label.setStyleSheet("font-weight: bold; font-size: 14px; min-width: 200px;")
        self.mode_label.setAlignment(Qt.AlignCenter)
        status_bar.addWidget(self.mode_label, stretch=0)

        # 提示消息标签
        self.message_label = QLabel("")
        self.message_label.setStyleSheet("font-size: 14px;")
        self.message_label.setAlignment(Qt.AlignLeft)
        status_bar.addWidget(self.message_label, stretch=1)

    def set_mode(self, mode_text: str) -> None:
        """设置左边的模式名，例如 '点模式' 或 '线模式'"""
        self.mode_label.setText(mode_text)

    def set_message(self, message: str) -> None:
        """设置右边的操作提示，直接覆盖"""
        self.message_label.setText(message)

    def clear_message(self) -> None:
        """清空右边提示"""
        self.message_label.setText("")

    def update_status(self, message: str, mode_text: str = None) -> None:
        """统一接口：同时设置模式和提示（最常用）"""
        if mode_text is not None:
            self.set_mode(mode_text)
        self.set_message(message)