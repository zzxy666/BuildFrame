import os
import subprocess
import sys
import pytest
pytest.importorskip("rasterio")
pytest.importorskip("pyproj")


def test_rgb_polygon_roundtrip_and_restore(tmp_path):
    script=r"""
import sys,time,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import laspy,rasterio
from rasterio.transform import from_origin
from pyproj import CRS
from PyQt5 import QtCore,QtWidgets,QtTest
from labelCloud.control.config_manager import config
from labelCloud.control.controller import Controller
from labelCloud.model.point_cloud import PointCloud
from labelCloud.model.scene_planes import RoofPlanes
from labelCloud.view.viewer import GLWidget
from labelCloud.view.gui import GUI
root=Path(sys.argv[1])
las=laspy.LasData(laspy.LasHeader(point_format=7,version='1.4'))
las.header.add_crs(CRS.from_epsg(3857))
cols=np.array([2.5,3.5,4.5,5.5,8.5,4.5]);rows=cols.copy()
las.x=100+cols;las.y=200-rows;las.z=[10,10,10,10,10,3]
las.intensity=np.arange(6);las.classification=np.repeat(6,6)
las.add_extra_dim(laspy.ExtraBytesParams(name='plane_id',type=np.uint32))
las.plane_id=[7,0,0,0,0,0];las.write(root/'scene.las')
with rasterio.open(root/'rgb.tif','w',driver='GTiff',height=10,width=10,count=3,dtype='uint8',crs=3857,transform=from_origin(100,200,1,1)) as dst:
    dst.write(np.full((3,10,10),150,np.uint8))
config.set('FILE','pointcloud_folder',str(root))
app=QtWidgets.QApplication([])
with patch.object(PointCloud,'create_buffers'),patch.object(PointCloud,'release_buffers'),patch.object(GLWidget,'initializeGL'),patch.object(GLWidget,'paintGL'),patch.object(GLWidget,'resizeGL'),patch.object(GLWidget,'updateGL'):
    owner=Controller();view=GUI(owner);app.installEventFilter(view)
    view.show();view.activateWindow();app.processEvents()
    roof=owner.roof_plane_controller;roof.mode.setCurrentIndex(1)
    def wait():
        end=time.monotonic()+15
        while time.monotonic()<end:
            app.processEvents();time.sleep(.005)
            if roof.task is None and roof.rgb.overlay_task is None and roof.rgb.panel.viewer.read_task is None:
                app.processEvents();return
        raise AssertionError('timeout')
    rgb=roof.rgb;rgb.open_file(str(root/'rgb.tif'));wait()
    assert rgb.mapper is not None and rgb.panel.isVisible()
    roof.model.set_hidden_ids(np.array([1]),True)
    roof.new_plane();target=roof.model.current
    before=roof.model.labels.copy()
    vertices=np.array([[1,1],[6,1],[6,6],[1,6]],float)
    rgb.panel.tool.setCurrentIndex(1)
    viewer=rgb.panel.viewer;viewer.vertices=vertices.tolist();viewer.finish_polygon()
    assert rgb.panel.map_button.isEnabled()
    QtTest.QTest.keyClick(viewer,QtCore.Qt.Key_Z,QtCore.Qt.ControlModifier)
    assert rgb.polygon is None
    QtTest.QTest.keyClick(viewer,QtCore.Qt.Key_Y,QtCore.Qt.ControlModifier)
    assert rgb.polygon is not None
    rgb.map_polygon();wait()
    np.testing.assert_array_equal(roof.candidate,[2,3,5])
    np.testing.assert_array_equal(roof.model.labels,before)
    roof.confirm_expansion();wait()
    assert np.all(roof.model.labels[[2,3,5]]==target)
    assert roof.model.labels[0]==7 and roof.model.labels[1]==0
    assert len(roof.model.session_metadata['rgb_annotations'])==1
    roof.undo();wait()
    np.testing.assert_array_equal(roof.model.labels,before)
    assert not roof.model.session_metadata['rgb_annotations']
    roof.redo();wait()
    assert np.all(roof.model.labels[[2,3,5]]==target)
    rgb.query_pixel([4.5,4.5]);wait()
    assert roof.model.current==target and roof.highlight==target
    rgb.locate_point(2)
    assert not viewer.cross.path().isEmpty()
    assert roof.save(force=True)
    annotations=json.loads((root/'scene.planar'/'rgb_annotations.json').read_text())['annotations']
    assert annotations[0]['world_polygon'][0]==[101.,199.]
    loaded=RoofPlanes(root/'scene.las')
    np.testing.assert_array_equal(loaded.labels,roof.model.labels)
    assert loaded.session_metadata['rgb_registration']['rgb_file']==str((root/'rgb.tif').resolve())
    roof.model.export(root/'export.las');export=laspy.read(root/'export.las')
    for field in las.points.array.dtype.names:
        if field!='plane_id': np.testing.assert_array_equal(export.points.array[field],las.points.array[field])
    roof.on_pointcloud_changed();wait();QtTest.QTest.qWait(50);wait()
    assert roof.rgb.mapper is not None
    assert not roof.model.hidden_mask.any()
    roof.mode.setCurrentIndex(0)
    assert not rgb.panel.isVisible() and view.gl_widget.isVisible()
    roof.mode.setCurrentIndex(1);wait()
    assert rgb.panel.isVisible()
    view.grab().save(str(root/'dual-view.png'))
    roof.autosave_timer.stop();roof.camera_timer.stop();view.timer.stop();view.hide()
print('RGB P0 roundtrip OK')
"""
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path)],env={**os.environ,'QT_QPA_PLATFORM':'offscreen','PYTHONIOENCODING':'utf-8'},capture_output=True,text=True,encoding='utf-8',timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
