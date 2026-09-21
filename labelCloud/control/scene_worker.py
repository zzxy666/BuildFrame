"""后台仅操作 NumPy/磁盘；Qt/OpenGL 更新始终留在主线程。"""
import traceback
from PyQt5 import QtCore, QtWidgets


class TaskSignals(QtCore.QObject):
    finished = QtCore.pyqtSignal(object, object)


class SceneTask(QtCore.QRunnable):
    def __init__(self, work):
        super().__init__()
        self.work = work
        self.signals = TaskSignals()

    def run(self):
        try:
            self.signals.finished.emit(self.work(), None)
        except Exception as error:
            self.signals.finished.emit(None, (error, traceback.format_exc()))


class SceneProgressDialog(QtWidgets.QProgressDialog):
    def reject(self):
        pass  # 不能用 Esc 关闭进度对话框后同时编辑正在保存的数据。


def run_scene_task(parent, text, work):
    """兼容原有同步导航接口：后台加载，主线程事件循环继续响应和绘制。"""
    loop = QtCore.QEventLoop()
    focus = QtWidgets.QApplication.focusWidget()
    progress = SceneProgressDialog(text, "", 0, 0, parent)
    progress.setCancelButton(None)
    progress.setWindowModality(QtCore.Qt.ApplicationModal)
    progress.setMinimumDuration(0)
    result = []
    task = SceneTask(work)
    def finish(value, error):
        result.extend((value, error))
        loop.quit()
    task.signals.finished.connect(finish)
    progress.show()
    QtCore.QThreadPool.globalInstance().start(task)
    loop.exec_()
    progress.close()
    progress.deleteLater()
    if parent.isVisible(): parent.activateWindow()
    if focus is not None: focus.setFocus()
    QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)
    if result[1]:
        raise result[1][0]
    return result[0]
