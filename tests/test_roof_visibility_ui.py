"""验证真实 Qt 快捷键、选区过滤、显示切换及跨文件重置。"""
import os
import subprocess
import sys


def test_visibility_protection_and_navigation(tmp_path):
    script = r'''
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np
import laspy
from PyQt5 import QtCore, QtWidgets, QtTest
from labelCloud.control.config_manager import config
from labelCloud.control.controller import Controller
from labelCloud.model.point_cloud import PointCloud
from labelCloud.view.viewer import GLWidget
from labelCloud.view.gui import GUI

root = Path(sys.argv[1])
for name in ('a', 'b'):
    las = laspy.LasData(laspy.LasHeader(point_format=7, version='1.4'))
    las.x=[-.5,.5,-.5,.5]; las.y=[-.5,-.5,.5,.5]; las.z=[0,0,0,0]
    las.add_extra_dim(laspy.ExtraBytesParams(name='plane_id', type=np.uint32))
    las.plane_id=[1,2,0,0]
    las.write(root/(name+'.las'))
config.set('FILE','pointcloud_folder',str(root))
app=QtWidgets.QApplication([])
with patch.object(PointCloud,'create_buffers'), patch.object(PointCloud,'release_buffers'), \
     patch.object(GLWidget,'initializeGL'), patch.object(GLWidget,'paintGL'), \
     patch.object(GLWidget,'resizeGL'), patch.object(GLWidget,'updateGL'):
    owner=Controller(); view=GUI(owner); app.installEventFilter(view)
    view.show(); view.activateWindow(); app.processEvents()
    roof=owner.roof_plane_controller; roof.mode.setCurrentIndex(1)
    widget=view.gl_widget; model=roof.model
    def wait_task():
        deadline=time.monotonic()+10
        while roof.task and time.monotonic()<deadline:
            app.processEvents(); time.sleep(.001)
        assert roof.task is None

    widget.modelview=np.eye(4); widget.projection=np.eye(4)
    def select_all(tool=1):
        roof.panel.tool.setCurrentIndex(tool)
        w,h=widget.width(),widget.height()
        roof.gesture=[(0,0),(w,0),(w,h),(0,h)]
        roof.gesture_modifiers=QtCore.Qt.NoModifier
        roof.finish_selection((w,h) if tool==1 else (0,0))
        wait_task()
    select_all()
    np.testing.assert_array_equal(model.selection,[False,False,True,True])
    QtTest.QTest.keyClick(widget,QtCore.Qt.Key_H)
    np.testing.assert_array_equal(model.hidden_mask,[False,False,True,True])
    np.testing.assert_array_equal(roof.cloud.display_indices,[0,1])
    assert not model.dirty
    select_all(2)
    assert not model.selection.any()
    roof.panel.show_hidden.setChecked(True)
    np.testing.assert_array_equal(roof.cloud.display_indices,[2,3])
    roof.panel.selection_scope.setCurrentIndex(2)
    select_all()
    assert not model.selection.any()  # 仅显示隐藏点也不能选择。
    QtTest.QTest.keyClick(widget,QtCore.Qt.Key_H,QtCore.Qt.ShiftModifier)
    assert not model.hidden_mask.any() and not roof.panel.show_hidden.isChecked()
    QtTest.QTest.keyClick(widget,QtCore.Qt.Key_Z,QtCore.Qt.ControlModifier)
    assert model.hidden_mask.sum()==2
    QtTest.QTest.keyClick(widget,QtCore.Qt.Key_Z,QtCore.Qt.ControlModifier)
    assert not model.hidden_mask.any()
    # 当前 Plane 的纠错只能影响原当前 Plane。
    roof.panel.selection_scope.setCurrentIndex(1)
    model.current=1
    select_all(2)
    np.testing.assert_array_equal(model.selection,[True,False,False,False])
    roof.assign(3)
    np.testing.assert_array_equal(model.labels,[3,2,0,0])
    roof.panel.selection_scope.setCurrentIndex(0)
    model.selection[:]=True  # 赋值入口必须再次过滤过期选区。
    roof.assign(4)
    np.testing.assert_array_equal(model.labels,[3,2,4,4])
    roof.panel.selection_scope.setCurrentIndex(2)
    model.selection[:]=False; model.selection[1]=True
    roof.isolate_selected()
    np.testing.assert_array_equal(roof.cloud.display_indices,[1])
    before=model.hidden_mask.copy()
    roof.mode.setCurrentIndex(0)
    assert roof.cloud.display_indices is None
    roof.mode.setCurrentIndex(1)
    np.testing.assert_array_equal(model.hidden_mask,before)
    assert roof.save(force=True)
    roof.model.export()
    saved=laspy.read(root/'a_gt.las')
    assert len(saved.points)==4
    np.testing.assert_array_equal(saved.plane_id,[3,2,4,4])
    owner.next_pcd()
    assert not roof.model.hidden_mask.any()
    assert roof.panel.selection_scope.currentData()=='unlabelled'
    assert not roof.panel.show_hidden.isChecked()
    owner.prev_pcd()
    assert not roof.model.hidden_mask.any()
    np.testing.assert_array_equal(roof.model.labels,[3,2,4,4])
    assert roof.model.hidden_count==0
    view.timer.stop(); view.hide()
print('Visibility and protection UI OK')
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "BUILDFRAME_LOG_PATH": os.devnull,
             "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"},
        capture_output=True, text=True, encoding="utf-8", timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr
