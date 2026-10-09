"""隔离进程的真实 Qt + mock 分割集成验证；不会加载 SAM 权重。"""
import json
from pathlib import Path
import sys
import threading
import time
from unittest.mock import patch
import numpy as np
import laspy
import rasterio
from rasterio.transform import from_origin
from pyproj import CRS
from PyQt5 import QtCore, QtWidgets, QtTest
from labelCloud.control.config_manager import config
from labelCloud.control.controller import Controller
from labelCloud.model.point_cloud import PointCloud
from labelCloud.model.segmentation.base_backend import BaseSegmentationBackend, Prediction
from labelCloud.model.segmentation.engine import SegmentationEngine
from labelCloud.model.segmentation.patch import SAMState
from labelCloud.view.viewer import GLWidget
from labelCloud.view.gui import GUI

root = Path(sys.argv[1])
las = laspy.LasData(laspy.LasHeader(point_format=7, version='1.4'))
las.header.add_crs(CRS.from_epsg(3857))
cols = np.array([2.5,3.5,4.5,5.5,8.5,4.5]);rows=cols.copy()
las.x=100+cols;las.y=200-rows;las.z=[10,10,10,10,10,3]
las.intensity=np.arange(6);las.classification=np.repeat(6,6)
las.add_extra_dim(laspy.ExtraBytesParams(name='plane_id',type=np.uint32))
las.plane_id=[7,0,0,0,0,0];las.write(root/'scene.las')
with rasterio.open(root/'rgb.tif','w',driver='GTiff',height=10,width=10,count=3,dtype='uint8',crs=3857,transform=from_origin(100,200,1,1)) as dst:
    dst.write(np.full((3,10,10),150,np.uint8))


class MockBackend(BaseSegmentationBackend):
    device='cpu'
    def __init__(self, config):
        super().__init__();self.encodes=0;self.decodes=0;self.block=False;self.started=threading.Event();self.release=threading.Event();self.raise_error=None
    def clear_image(self): self.reset_prompts()
    def set_image(self,image): self.encodes+=1;self.shape=image.shape[:2]
    def predict(self):
        self.decodes+=1
        if self.block:
            self.block=False;self.started.set();assert self.release.wait(5)
        if self.raise_error:
            error=self.raise_error;self.raise_error=None;raise RuntimeError(error)
        masks=np.zeros((3,*self.shape),bool)
        masks[1,1:9,1:9]=True
        if any(k==0 for _,k in self.points): masks[1,0,0]=True
        return Prediction(masks,np.array([.1,.9,.2]))


config.set('FILE','pointcloud_folder',str(root))
app=QtWidgets.QApplication([])
with patch('labelCloud.control.sam_controller.availability',return_value=(False,'test: SAM unavailable')),patch.object(PointCloud,'create_buffers'),patch.object(PointCloud,'release_buffers'),patch.object(GLWidget,'initializeGL'),patch.object(GLWidget,'paintGL'),patch.object(GLWidget,'resizeGL'),patch.object(GLWidget,'updateGL'):
    owner=Controller();view=GUI(owner);app.installEventFilter(view)
    view.show();view.activateWindow();app.processEvents()
    roof=owner.roof_plane_controller;roof.mode.setCurrentIndex(1)
    rgb=roof.rgb;sam=rgb.sam;viewer=rgb.panel.viewer
    def wait():
        end=time.monotonic()+15
        while time.monotonic()<end:
            app.processEvents();time.sleep(.005)
            if roof.task is None and sam.task is None and sam.pending is None and rgb.overlay_task is None and viewer.read_task is None:
                app.processEvents();return
        raise AssertionError('timeout')
    rgb.open_file(str(root/'rgb.tif'));wait()
    assert 'torch' not in sys.modules and 'sam2' not in sys.modules
    assert not rgb.panel.tool.model().item(2).isEnabled()
    # 测试专用注入；产品没有伪装 SAM 结果的 mock 模式。
    sam.engine=SegmentationEngine(MockBackend);sam.available=True;sam.reason='';sam.update_availability()
    roof.new_plane();target=roof.model.current
    roof.model.set_hidden_ids(np.array([1]),True)
    roof.workspace.roi_mask=np.array([True,True,True,True,False,True])
    rgb.panel.tool.setCurrentIndex(2);wait()
    backend=sam.engine.backend
    before=roof.model.labels.copy()
    point=viewer.mapFromScene(QtCore.QPointF(4.5,4.5))
    QtTest.QTest.mouseClick(viewer.viewport(),QtCore.Qt.LeftButton,pos=point);wait()
    assert sam.state==SAMState.MASK_PREVIEW and backend.encodes==1
    view.setWindowTitle('BuildFrame P1 — MOCK backend UI test (not real SAM)')
    view.grab().save(str(root/'sam-mock-ui.png'))
    assert sam.prompts.points[0][1]==1
    np.testing.assert_array_equal(roof.model.labels,before)
    # 连续请求：旧结果必须被丢弃，同时不能重编码同一 patch。
    backend.block=True
    sam.add_point((4.5,4.5),True)
    assert backend.started.wait(5)
    sam.add_point((2.5,2.5),False)
    assert sam.prediction is None
    backend.release.set();wait()
    assert sam.prediction.masks[sam.mask_index,0,0]
    assert backend.encodes==1 and backend.decodes==3
    QtTest.QTest.keyClick(viewer,QtCore.Qt.Key_Backspace);wait()
    assert not sam.prediction.masks[sam.mask_index,0,0]
    assert backend.encodes==1
    # 平移、缩放和移动鼠标不触发推理。
    decodes=backend.decodes;viewer.scale(1.1,1.1);viewer.centerOn(5,5)
    QtTest.QTest.mouseMove(viewer.viewport(),point);wait()
    assert backend.encodes==1 and backend.decodes==decodes
    # Box 单独使用，随后加负点；R 保留 embedding。
    QtTest.QTest.keyClick(viewer,QtCore.Qt.Key_R);wait()
    assert not sam.prompts.events and sam.patch.encoded
    rgb.panel.tool.setCurrentIndex(3)
    p1=viewer.mapFromScene(QtCore.QPointF(1,1));p2=viewer.mapFromScene(QtCore.QPointF(7,7))
    QtTest.QTest.mousePress(viewer.viewport(),QtCore.Qt.LeftButton,pos=p1)
    QtTest.QTest.mouseMove(viewer.viewport(),p2)
    QtTest.QTest.mouseRelease(viewer.viewport(),QtCore.Qt.LeftButton,pos=p2);wait()
    assert sam.prompts.box is not None and backend.box is not None and backend.encodes==1
    QtTest.QTest.mouseClick(viewer.viewport(),QtCore.Qt.RightButton,pos=point);wait()
    assert sam.prompts.points[-1][1]==0
    # 未接受 mask 切 Plane：No 保留旧目标，Yes 丢弃。
    with patch.object(QtWidgets.QMessageBox,'question',return_value=QtWidgets.QMessageBox.No):
        roof.model.current=7;roof.refresh();wait()
    assert roof.model.current==target and sam.prediction is not None
    with patch.object(QtWidgets.QMessageBox,'question',return_value=QtWidgets.QMessageBox.Yes):
        roof.model.current=7;roof.refresh();wait()
    assert sam.prediction is None and not sam.prompts.events
    roof.model.current=target;roof.refresh()
    sam.add_point((4.5,4.5),True);wait()
    assert backend.encodes==1
    # 第一次 Enter 只能进入候选，已有 Plane、隐藏点、ROI 外点都排除。
    viewer.setFocus();QtTest.QTest.keyClick(viewer,QtCore.Qt.Key_Return);wait()
    assert sam.state==SAMState.LIDAR_PREVIEW
    assert rgb.gt_pending is not None and roof.model.dual_gt.ensure(target)['rgb_gt_status']=='MISSING'
    np.testing.assert_array_equal(roof.candidate,[2,3,5])
    np.testing.assert_array_equal(roof.model.labels,before)
    roof.preview_edits.edit(np.array([3]),False);roof.adjust_edge_level(0)
    roof.undo();assert 3 in roof.candidate
    roof.redo();assert 3 not in roof.candidate
    # 第二次 Enter 才写 GT，随后差分撤销/重做和 metadata 同步。
    view.gl_widget.setFocus();QtTest.QTest.keyClick(view.gl_widget,QtCore.Qt.Key_Return);wait()
    assert sam.state==SAMState.CONFIRMED
    assert roof.model.labels.tolist()==[7,0,target,0,0,target]
    assert roof.model.dual_gt.ensure(target)['rgb_gt_status']=='CONFIRMED'
    annotation=roof.model.session_metadata['rgb_annotations'][-1]
    assert annotation['annotation_source']=='rgb_sam_lidar_corrected'
    assert annotation['candidate_count']==3 and annotation['final_confirmed_count']==2
    assert annotation['removed_in_lidar']==1
    mask_path=roof.model.session.folder/annotation['mask_path']
    with np.load(mask_path,allow_pickle=False) as data:
        assert data['mask'].shape==(10,10) and int(data['col0'])==0
    roof.undo();wait();np.testing.assert_array_equal(roof.model.labels,before)
    assert not roof.model.session_metadata['rgb_annotations']
    assert roof.model.dual_gt.ensure(target)['rgb_gt_status']=='MISSING'
    roof.redo();wait();assert roof.model.labels[2]==target
    assert roof.save(force=True)
    saved=json.loads((root/'scene.planar'/'rgb_annotations.json').read_text(encoding='utf-8'))
    assert saved['annotations'][-1]['mask_path']==annotation['mask_path']
    roof.model.export(root/'export.las');export=laspy.read(root/'export.las')
    for field in las.points.array.dtype.names:
        if field!='plane_id': np.testing.assert_array_equal(export.points.array[field],las.points.array[field])
    # OOM 仅禁用本次 mask，CPU 重试和 Polygon 都可继续。
    sam.add_point((5,5),True);wait()
    backend.raise_error='CUDA out of memory'
    sam.add_point((5,5),False);wait()
    assert 'Error' in sam.panel.status.text() and not sam.panel.accept.isEnabled()
    assert sam.panel.cpu.isVisible()
    sam.retry_cpu();wait()
    assert sam.config.device=='cpu' and sam.prediction is not None
    sam.reset_region();wait();assert sam.engine.patch_key is None and sam.patch is None
    # 换图只清 embedding，不重新加载模型。
    backend=sam.engine.backend
    rgb.open_file(str(root/'rgb.tif'));wait()
    assert sam.engine.backend is backend
    sam.add_point((4,4),True);wait();assert backend.encodes==2
    # 显式二维补标无需再次映射/覆盖 LiDAR；此处仍为 mock backend。
    labels=roof.model.labels.copy();previous=roof.model.dual_gt.ensure(target)['fragments']
    assert not sam.prediction.masks.any()  # 默认保护已标注像素，SAM 预览也不能占用。
    roof.dual.replace.setChecked(True);wait()
    assert sam.prediction.masks.any()
    roof.dual.confirm_rgb();wait()
    assert len(roof.model.dual_gt.ensure(target)['fragments'])==1
    assert roof.model.dual_gt.ensure(target)['fragments']!=previous
    np.testing.assert_array_equal(roof.model.labels,labels)
    roof.undo();wait();assert roof.model.dual_gt.ensure(target)['fragments']==previous
    np.testing.assert_array_equal(roof.model.labels,labels)
    sam.reset_prompts();rgb.panel.tool.setCurrentIndex(1)
    viewer.vertices=[[1,1],[6,1],[6,6],[1,6]];viewer.finish_polygon()
    assert rgb.panel.map_button.isEnabled();rgb.map_polygon();wait()
    assert roof.candidate is not None;roof.cancel_expansion();wait()
    roof.mode.setCurrentIndex(0);assert not rgb.panel.isVisible() and view.gl_widget.isVisible()
    roof.mode.setCurrentIndex(1);wait()
    roof.autosave_timer.stop();roof.camera_timer.stop();view.timer.stop();viewer.detail_timer.stop();rgb.overlay_timer.stop();view.hide()
print('P1 mock Qt integration OK (real SAM NOT validated)')
