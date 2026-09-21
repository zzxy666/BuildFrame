from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel

from .i18n import tr


class StatusManager:
    def __init__(self, status_bar):
        self._mode_source = "导航模式"
        self._mode_values = {}
        self._message_source = ""
        self._message_values = {}
        self.mode_label = QLabel(tr(self._mode_source))
        self.mode_label.setStyleSheet("font-weight: bold; font-size: 14px; min-width: 200px;")
        self.mode_label.setAlignment(Qt.AlignCenter)
        status_bar.addWidget(self.mode_label, stretch=0)

        self.message_label = QLabel("")
        self.message_label.setStyleSheet("font-size: 14px;")
        self.message_label.setAlignment(Qt.AlignLeft)
        status_bar.addWidget(self.message_label, stretch=1)

    def set_mode(self, mode_text: str, **values) -> None:
        self._mode_source = mode_text
        self._mode_values = values
        self.mode_label.setText(tr(mode_text, **values))

    def set_message(self, message: str, **values) -> None:
        self._message_source = message
        self._message_values = values
        self.message_label.setText(tr(message, **values))

    def clear_message(self) -> None:
        self._message_source = ""
        self._message_values = {}
        self.message_label.setText("")

    def update_status(self, message: str, mode_text: str = None, **values) -> None:
        if mode_text is not None:
            self.set_mode(mode_text)
        self.set_message(message, **values)

    def retranslate(self) -> None:
        self.mode_label.setText(tr(self._mode_source, **self._mode_values))
        self.message_label.setText(
            tr(self._message_source, **self._message_values)
            if self._message_source
            else ""
        )
