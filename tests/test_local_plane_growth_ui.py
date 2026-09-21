"""候选预览与正式标签隔离：真实 Qt 快捷键、保存和撤销。"""
import os
import subprocess
import sys


def test_preview_confirm_cancel_and_preservation(tmp_path):
    script = r'''
import sys
import time
from pathlib import Path
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
x,y = np.meshgrid(np.arange(15)*.1, np.arange(15)*.1)
patch_xyz = np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
xyz = np.vstack((patch_xyz,patch_xyz+[5,0,0]))
las = laspy.LasData(laspy.LasHeader(point_format=7,version='1.4'))
las.x=xyz[:,0]; las.y=xyz[:,1]; las.z=xyz[:,2]
las.intensity=np.arange(len(xyz))
las.classification=np.repeat(6,len(xyz))
las.key_point=np.arange(len(xyz))%2
las.add_extra_dim(laspy.ExtraBytesParams(name='plane_id',type=np.uint32))
las.plane_id=np.zeros(len(xyz),dtype=np.uint32)
las.plane_id[0]=7
las.write(root/'roof.las')
config.set('FILE','pointcloud_folder',str(root))
app=QtWidgets.QApplication([])
with patch.object(PointCloud,'create_buffers'), patch.object(PointCloud,'release_buffers'), \
     patch.object(GLWidget,'initializeGL'), patch.object(GLWidget,'paintGL'), \
     patch.object(GLWidget,'resizeGL'), patch.object(GLWidget,'updateGL'):
    owner=Controller(); view=GUI(owner); app.installEventFilter(view)
    view.show(); view.activateWindow(); app.processEvents()
    roof=owner.roof_plane_controller; roof.mode.setCurrentIndex(1)
    roof.new_plane()  # 已有 Plane 7，因此新建为 8。
    target=roof.model.current
    def wait_task():
        deadline=time.monotonic()+10
        while roof.task and time.monotonic()<deadline:
            app.processEvents(); time.sleep(.001)
        assert roof.task is None

    roof.panel.coordinate_units.setCurrentIndex(1)  # 无 CRS，明确选择米。
    seeds=(xyz[:,0]>=.2)&(xyz[:,0]<=.5)&(xyz[:,1]>=.2)&(xyz[:,1]<=.5)
    roof.model.selection=seeds.copy()
    labels=roof.model.labels.copy(); history=len(roof.model.history)
    # 编辑 normal_k 时，数字必须进参数框，不能触发 plane_id 赋值。
    roof.panel.growth_settings_toggle.setChecked(True)
    edit=roof.panel.normal_k.lineEdit(); edit.setFocus(); edit.selectAll()
    QtTest.QTest.keyClicks(edit,'20')
    QtTest.QTest.keyClick(edit,QtCore.Qt.Key_Return)
    assert roof.panel.normal_k.value()==20
    np.testing.assert_array_equal(roof.model.labels,labels)
    view.gl_widget.setFocus()
    QtTest.QTest.keyClick(view.gl_widget,QtCore.Qt.Key_E)
    wait_task()
    assert roof.candidate is not None
    assert len(roof.candidate)==224
    assert (roof.candidate<225).all()
    assert 0 not in roof.candidate  # 保护其他 Plane。
    candidate=roof.candidate.copy()
    np.testing.assert_array_equal(roof.model.labels,labels)
    assert len(roof.model.history)==history
    np.testing.assert_array_equal(roof.cloud.overlays[-1][0],candidate)
    # 数字赋值与新建操作都不能绕过候选确认。
    QtTest.QTest.keyClick(view.gl_widget,QtCore.Qt.Key_2)
    roof.new_plane()
    assert roof.model.current==target
    np.testing.assert_array_equal(roof.model.labels,labels)
    # 保存只能写已确认标签。
    assert owner.save(force_plane=True)
    np.testing.assert_array_equal(np.load(root/'roof.planar'/'plane_id.npy'),labels)
    QtTest.QTest.keyClick(view.gl_widget,QtCore.Qt.Key_Escape)
    assert roof.candidate is None
    np.testing.assert_array_equal(roof.model.selection,seeds)
    np.testing.assert_array_equal(roof.model.labels,labels)
    roof.expand_local()
    wait_task()
    QtTest.QTest.keyClick(view.gl_widget,QtCore.Qt.Key_Return)
    assert roof.candidate is None
    assert np.all(roof.model.labels[candidate]==target)
    assert roof.model.labels[0]==7
    assert len(roof.model.history)==history+1
    assert owner.save(force_plane=True)
    roof.model.export()
    saved=laspy.read(root/'roof_gt.las')
    for field in las.points.array.dtype.names:
        if field!='plane_id':
            np.testing.assert_array_equal(saved.points.array[field],las.points.array[field])
    QtTest.QTest.keyClick(view.gl_widget,QtCore.Qt.Key_Z,QtCore.Qt.ControlModifier)
    np.testing.assert_array_equal(roof.model.labels,labels)
    assert roof.model.dirty
    # 模式切换丢弃未确认候选，不能悄悄写入。
    roof.model.selection=seeds.copy(); roof.expand_local()
    wait_task()
    roof.mode.setCurrentIndex(0)
    assert roof.candidate is None
    np.testing.assert_array_equal(roof.model.labels,labels)
    roof.mode.setCurrentIndex(1)
    roof.model.selection[:]=False; roof.model.selection[:2]=True
    roof.expand_local()
    wait_task()
    assert roof.candidate is None
    assert '6' in roof.panel.expansion_info.text()
    np.testing.assert_array_equal(roof.model.labels,labels)
    view.timer.stop(); view.hide()
print('Local expansion UI integration OK')
'''
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "BUILDFRAME_LOG_PATH": os.devnull,
             "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"},
        capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
