"""完整 Qt 主窗口交互测试；无显示器时仅替换 OpenGL 绘制，不替换业务控制器。"""
import os
import subprocess
import sys


def test_roof_plane_mode_selection_shortcuts_and_navigation(tmp_path):
    script = r'''
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np
import laspy
from PyQt5 import QtCore, QtGui, QtWidgets, QtTest
from labelCloud.control.config_manager import config
from labelCloud.control.controller import Controller
from labelCloud.model.point_cloud import PointCloud
from labelCloud.view.viewer import GLWidget
from labelCloud.view.gui import GUI

root = Path(sys.argv[1])
for name in ('a', 'b'):
    las = laspy.LasData(laspy.LasHeader(point_format=7, version='1.4'))
    las.x = [-.5, .5, -.5, .5]
    las.y = [-.5, -.5, .5, .5]
    las.z = [0., 0., 0., 0.]
    las.red = [1000, 2000, 3000, 4000]
    las.write(root / (name + '.las'))
config.set('FILE', 'pointcloud_folder', str(root))
app = QtWidgets.QApplication([])
errors = []
QtWidgets.QMessageBox.critical = lambda *a: errors.append(a[-1])
with patch.object(PointCloud, 'create_buffers'), patch.object(PointCloud, 'release_buffers'), \
     patch.object(GLWidget, 'initializeGL'), patch.object(GLWidget, 'paintGL'), \
     patch.object(GLWidget, 'resizeGL'), patch.object(GLWidget, 'updateGL'):
    owner = Controller()
    view = GUI(owner)
    app.installEventFilter(view)
    view.show()
    view.activateWindow()
    app.processEvents()
    roof = owner.roof_plane_controller
    assert not roof.active
    owner.roof_drawing_manager.vertices.append((.1, .2, .3))
    roof.mode.setCurrentIndex(1)
    assert roof.active and roof.model is not None
    assert len(owner.roof_drawing_manager.vertices) == 1
    assert not view.button_add_vertices.isEnabled()
    assert view.end_pt_control.isEnabled()
    assert view.button_view_top.isEnabled()
    cloud=owner.pcd_manager.pointcloud
    cloud.set_rotations(20,30,40)
    translation=cloud.get_translation()
    QtTest.QTest.mouseClick(view.button_view_top,QtCore.Qt.LeftButton)
    assert cloud.get_rotations()==(0,0,0)
    assert cloud.get_translation()==translation
    QtTest.QTest.keyClick(view.gl_widget,QtCore.Qt.Key_2,QtCore.Qt.AltModifier)
    assert cloud.get_rotations()==(270,0,0)
    assert cloud.get_translation()==translation
    roof.panel.tool.setCurrentIndex(0)
    roof.task=object()
    press=QtGui.QMouseEvent(QtCore.QEvent.MouseButtonPress,QtCore.QPointF(10,10),
                           QtCore.Qt.LeftButton,QtCore.Qt.LeftButton,QtCore.Qt.NoModifier)
    assert not roof.handle_mouse(press)  # 后台计算不阻断浏览模式的左键旋转。
    roof.task=None
    widget = view.gl_widget
    def wait_task():
        deadline=time.monotonic()+10
        while roof.task and time.monotonic()<deadline:
            app.processEvents(); time.sleep(.001)
        assert roof.task is None

    widget.modelview = np.eye(4)
    widget.projection = np.eye(4)
    roof.panel.tool.setCurrentIndex(1)
    # 选左半边，透视投影下包含两个点。
    width, height = widget.width(), widget.height()
    begin = QtCore.QPoint(int(width*.1), int(height*.1))
    end = QtCore.QPoint(int(width*.4), int(height*.9))
    before = owner.pcd_manager.pointcloud.get_rotations()
    QtTest.QTest.mousePress(widget, QtCore.Qt.LeftButton, pos=begin)
    QtTest.QTest.mouseMove(widget, end)
    QtTest.QTest.mouseRelease(widget, QtCore.Qt.LeftButton, pos=end)
    wait_task()
    assert roof.model.selection.sum() == 2, roof.model.selection
    assert owner.pcd_manager.pointcloud.get_rotations() == before
    np.testing.assert_array_equal(roof.cloud.overlays[-1][0], np.flatnonzero(roof.model.selection))
    selected_before = roof.model.selection.copy()
    # 工具快捷键可从列表触发；不能丢失已选点，也不触发上一文件。
    roof.panel.list.setFocus()
    for key, name in [(QtCore.Qt.Key_V, 'navigate'), (QtCore.Qt.Key_L, 'lasso'),
                      (QtCore.Qt.Key_B, 'rectangle')]:
        QtTest.QTest.keyClick(roof.panel.list, key)
        assert roof.panel.tool.currentData() == name
        np.testing.assert_array_equal(roof.model.selection, selected_before)
        assert owner.pcd_manager.current_id == 0
    # 中键拖动确实改变相机，但不改变当前工具、选择和标签。
    before = owner.pcd_manager.pointcloud.get_rotations()
    QtTest.QTest.mousePress(widget, QtCore.Qt.MiddleButton, pos=begin)
    move = QtGui.QMouseEvent(QtCore.QEvent.MouseMove, QtCore.QPointF(end),
                            QtCore.Qt.NoButton, QtCore.Qt.MiddleButton, QtCore.Qt.NoModifier)
    app.sendEvent(widget, move)
    QtTest.QTest.mouseRelease(widget, QtCore.Qt.MiddleButton, pos=end)
    assert owner.pcd_manager.pointcloud.get_rotations() != before
    assert roof.panel.tool.currentData() == 'rectangle'
    assert roof.orbit_position is None
    np.testing.assert_array_equal(roof.model.selection, selected_before)
    assert not roof.model.labels.any()
    # Qt 真正派发快捷键，验证焦点在视图/列表时均可用。
    widget.setFocus()
    roof.assign(3)
    assert (roof.model.labels == 3).sum() == 2
    roof.panel.list.setFocus()
    QtTest.QTest.keyClick(roof.panel.list, QtCore.Qt.Key_Z, QtCore.Qt.ControlModifier)
    assert not roof.model.labels.any()
    assert roof.model.selection.sum() == 2
    QtTest.QTest.keyClick(roof.panel.list, QtCore.Qt.Key_N)
    assert roof.model.current == 4
    roof.assign()
    assert (roof.model.labels == 4).sum() == 2
    # 套索选右半边。
    roof.panel.tool.setCurrentIndex(2)
    roof.gesture = [(width*.6,height*.1),(width*.9,height*.1),
                    (width*.9,height*.9),(width*.6,height*.9)]
    roof.gesture_modifiers = QtCore.Qt.NoModifier
    roof.finish_selection((width*.6,height*.1))
    wait_task()
    assert roof.model.selection.sum() == 2
    roof.assign(2)
    roof.panel.selection_scope.setCurrentIndex(2)  # 主动进入纠错范围，允许合并/删除。
    with patch.object(QtWidgets.QInputDialog, 'getText', return_value=('4', True)):
        roof.merge()
    assert np.all(roof.model.labels == 4)
    roof.delete()
    assert not roof.model.labels.any()
    roof.undo()
    assert np.all(roof.model.labels == 4)
    QtTest.QTest.keyClick(roof.panel.list, QtCore.Qt.Key_Escape)
    assert not roof.model.selection.any()
    assert roof.highlight is None
    roof.panel.display.setCurrentIndex(1)
    np.testing.assert_allclose(roof.cloud.display_colors[:,0], np.array([1000,2000,3000,4000])/65535)
    roof.mode.setCurrentIndex(0)
    assert all(not shortcut.isEnabled() for shortcut in roof.panel.shortcuts)
    assert roof.cloud.display_colors is None
    assert view.button_add_vertices.isEnabled()
    assert len(owner.roof_drawing_manager.vertices) == 1
    roof.mode.setCurrentIndex(1)
    assert np.all(roof.model.labels == 4)
    # 保存失败不得跳到下一文件。
    with patch.object(roof.model, 'save', side_effect=OSError('test disk full')):
        owner.next_pcd()
    assert owner.pcd_manager.current_id == 0
    assert errors and roof.model.dirty
    errors.clear()
    owner.next_pcd()
    assert owner.pcd_manager.current_id == 1
    assert (root/'a.planar'/'plane_id.npy').exists()
    owner.prev_pcd()
    assert np.all(roof.model.labels == 4)
    assert len(owner.roof_drawing_manager.vertices) == 1
    # 输出 GT 不进入下一轮原始文件队列。
    owner.pcd_manager.read_pointcloud_folder()
    assert [p.name for p in owner.pcd_manager.pcds] == ['a.las','b.las']
    view.timer.stop()
    view.hide()
    assert not errors, errors
print('Roof Plane UI integration OK')
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "BUILDFRAME_LOG_PATH": os.devnull,
             "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"},
        capture_output=True, text=True, encoding="utf-8", timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr
