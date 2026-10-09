"""统一下拉列表实际绘制，避免默认菜单 delegate 的白字白底问题。"""
from PyQt5 import QtCore, QtWidgets


POPUP_STYLE = """
QListView {
    color: #202020;
    background-color: #FFFFFF;
    selection-color: #FFFFFF;
    selection-background-color: #2468C4;
}
QListView::item {
    color: #202020;
    background-color: #FFFFFF;
    min-height: 24px;
}
QListView::item:hover, QListView::item:selected {
    color: #FFFFFF;
    background-color: #2468C4;
}
QListView::item:disabled {
    color: #808080;
    background-color: #F2F2F2;
}
"""


def configure_dropdown(widget):
    # 复用主窗口现有事件过滤器，不额外安装全局过滤器。
    if widget.property('buildframeReadablePopup'):
        return
    widget.setProperty('buildframeReadablePopup', True)
    view = QtWidgets.QListView(widget)
    widget.setView(view)
    view.setItemDelegate(QtWidgets.QStyledItemDelegate(view))
    view.setStyleSheet(POPUP_STYLE)
