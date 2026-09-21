import os
import subprocess
import sys


def test_async_selection_roi_autosave_bookmarks_and_large_ids(tmp_path):
    script=r'''
import sys,time,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import laspy
from PyQt5 import QtCore,QtWidgets,QtTest
from labelCloud.control.config_manager import config
from labelCloud.control.controller import Controller
from labelCloud.model.point_cloud import PointCloud
from labelCloud.view.viewer import GLWidget
from labelCloud.view.gui import GUI
root=Path(sys.argv[1])
las=laspy.LasData(laspy.LasHeader(point_format=7,version='1.4'))
x,y=np.meshgrid(np.linspace(-.8,.8,20),np.linspace(-.8,.8,20))
las.x=x.ravel();las.y=y.ravel();las.z=np.zeros(x.size)
las.red=np.arange(x.size,dtype=np.uint16)
las.write(root/'scene.las')
config.set('FILE','pointcloud_folder',str(root))
app=QtWidgets.QApplication([])
with patch.object(PointCloud,'create_buffers'),patch.object(PointCloud,'release_buffers'), \
     patch.object(GLWidget,'initializeGL'),patch.object(GLWidget,'paintGL'), \
     patch.object(GLWidget,'resizeGL'),patch.object(GLWidget,'updateGL'):
    owner=Controller();view=GUI(owner);app.installEventFilter(view)
    view.show();view.activateWindow();app.processEvents()
    roof=owner.roof_plane_controller;roof.mode.setCurrentIndex(1)
    for _ in range(3): app.processEvents()
    widget=view.gl_widget;widget.modelview=np.eye(4);widget.projection=np.eye(4)
    roof.panel.coordinate_units.setCurrentIndex(1)
    def wait_task():
        limit=time.monotonic()+10
        while roof.task and time.monotonic()<limit:
            app.processEvents();time.sleep(.001)
        assert roof.task is None
    def select_box():
        roof.panel.tool.setCurrentIndex(1)
        roof.gesture=[(0,0)]
        roof.gesture_modifiers=QtCore.Qt.NoModifier
        roof.finish_selection((widget.width(),widget.height()))
        wait_task()
    base=roof.cloud.display_colors
    with patch.object(roof.model,'colors',side_effect=AssertionError('full recolor')):
        select_box()
        select_box()
    assert roof.projection_cache.builds==1
    assert roof.cloud.display_colors is base
    assert len(roof.selected_ids)==400
    # 数字已不再直接赋值；A 给当前大 ID 赋值。
    QtTest.QTest.keyClick(widget,QtCore.Qt.Key_3)
    assert not roof.model.labels.any()
    roof.model.max_plane_id=1578
    QtTest.QTest.keyClick(widget,QtCore.Qt.Key_N)
    assert roof.model.current==1579
    QtTest.QTest.keyClick(widget,QtCore.Qt.Key_A)
    assert np.all(roof.model.labels==1579)
    # 自动保存只创建 sidecar，不生成 GT。
    limit=time.monotonic()+5
    while roof.model.dirty and time.monotonic()<limit:
        app.processEvents();time.sleep(.005)
    assert not roof.model.dirty
    assert not (root/'scene_gt.las').exists()
    np.testing.assert_array_equal(np.load(root/'scene.planar'/'plane_id.npy'),roof.model.labels)
    roof.panel.plane_search.setText('1579');roof.jump_plane()
    assert roof.panel.table_model.rowCount()==1 and roof.model.current==1579
    roof.panel.plane_search.clear()
    roof.panel.selection_scope.setCurrentIndex(2)
    roof.apply_roi([[-.8,-.8,0],[0,0,0]])
    select_box()
    assert 0<len(roof.selected_ids)<400
    changed=roof.selected_ids.copy()
    roof.assign(2000)
    assert np.all(roof.model.labels[changed]==2000)
    assert np.all(roof.model.labels[~roof.workspace.roi_mask]==1579)
    roof.add_bookmark();camera=roof.capture_camera()
    roof.cloud.trans_x+=20
    roof.step_bookmark(-1)
    np.testing.assert_allclose(roof.cloud.get_translation(),camera['translation'])
    roof.panel.grid_size.setValue(1)
    roof.make_review_grid();wait_task()
    roof.next_unreviewed();roof.mark_reviewed()
    assert len(roof.workspace.grid['reviewed'])==1
    assert roof.save(force=True)
    assert json.loads((root/'scene.planar'/'bookmarks.json').read_text())['data']
    assert len(json.loads((root/'scene.planar'/'reviewed_grid.json').read_text())['data']['reviewed'])==1
    # 最终导出独立执行；先恢复隐藏点并保留全部点。
    roof.clear_work_roi();roof.panel.selection_scope.setCurrentIndex(2)
    select_box();roof.hide_selected()
    destination=root/'final.laz'
    with patch.object(QtWidgets.QFileDialog,'getSaveFileName',return_value=(str(destination),'LAZ')):
        roof.export_final();wait_task()
    assert not roof.model.hidden_mask.any()
    saved=laspy.read(destination,laz_backend=laspy.LazBackend.Laszip)
    assert len(saved.points)==400
    np.testing.assert_array_equal(saved.plane_id,roof.model.labels)
    roof.autosave_timer.stop();roof.camera_timer.stop();view.timer.stop();view.hide()
print('Large scene UI OK')
'''
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path)],
        env={**os.environ,'QT_QPA_PLATFORM':'offscreen','PYTHONIOENCODING':'utf-8'},
        capture_output=True,text=True,encoding='utf-8',timeout=50)
    assert result.returncode==0,result.stdout+result.stderr
