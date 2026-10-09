"""真实 Qt/P0 Polygon 入口；无需 SAM 或 torch，OpenGL 绘制用 mock。"""
import json
from pathlib import Path
import sys
import time
import threading
from unittest.mock import patch
import laspy
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pyproj import CRS
from PyQt5 import QtCore,QtWidgets,QtTest
from labelCloud.control.config_manager import config
from labelCloud.control.controller import Controller
from labelCloud.control.geometry_controller import GeometryRefiner
from labelCloud.model.point_cloud import PointCloud
from labelCloud.model.geometry_refiner import plane_geometry
from labelCloud.model.coordinate_mapper import PixelMask
from labelCloud.view.viewer import GLWidget
from labelCloud.view.gui import GUI

root=Path(sys.argv[1]);x,y=np.meshgrid(np.arange(25)*.2,np.arange(25)*.2)
roof_xyz=np.column_stack((x.ravel()+100,y.ravel()+100,10+.2*x.ravel()+.1*y.ravel()))
outliers=roof_xyz[:125].copy();outliers[:,2]+=3
xyz=np.vstack((roof_xyz,outliers));n=len(xyz)
las=laspy.LasData(laspy.LasHeader(point_format=7,version='1.4'));las.header.add_crs(CRS.from_epsg(3857))
las.x,las.y,las.z=xyz.T;las.intensity=np.arange(n);las.classification=np.repeat(6,n)
las.add_extra_dim(laspy.ExtraBytesParams(name='plane_id',type=np.uint32));las.plane_id=np.zeros(n,np.uint32);las.plane_id[0]=7
las.write(root/'scene.las')
with rasterio.open(root/'rgb.tif','w',driver='GTiff',height=100,width=100,count=3,dtype='uint8',crs=3857,transform=from_origin(99,106,.1,.1)) as dst: dst.write(np.full((3,100,100),150,np.uint8))
config.set('FILE','pointcloud_folder',str(root));app=QtWidgets.QApplication([])
with patch.object(PointCloud,'create_buffers'),patch.object(PointCloud,'release_buffers'),patch.object(GLWidget,'initializeGL'),patch.object(GLWidget,'paintGL'),patch.object(GLWidget,'resizeGL'),patch.object(GLWidget,'updateGL'):
    owner=Controller();view=GUI(owner);app.installEventFilter(view);view.show();view.activateWindow();app.processEvents()
    r=owner.roof_plane_controller;r.mode.setCurrentIndex(1);rgb=r.rgb;g=r.geometry
    def wait():
        end=time.monotonic()+20
        while time.monotonic()<end:
            app.processEvents();time.sleep(.005)
            if r.task is None and g.task is None and rgb.sam.task is None and rgb.overlay_task is None and r.dual.overlay_task is None and rgb.panel.viewer.read_task is None:
                app.processEvents();return
        raise AssertionError('timeout')
    rgb.open_file(str(root/'rgb.tif'));wait();r.new_plane();target=r.model.current
    r.model.set_hidden_ids([1],True);r.workspace.roi_mask=np.ones(n,bool);r.workspace.roi_mask[2]=False
    original_labels=r.model.labels.copy()
    app.processEvents();rgb_height=rgb.panel.viewer.height()
    rgb.rgb_tools_action.setChecked(True);app.processEvents()
    assert not rgb.panel.controls.isVisible() and rgb.panel.viewer.height()>rgb_height
    assert rgb.rgb_tools_action.text()=='显示 RGB 工具'
    rgb.rgb_tools_action.setChecked(False);app.processEvents()
    assert rgb.panel.controls.isVisible() and rgb.sam.panel.parent() is rgb.panel.controls

    # 收起左右容器，双视图确实获得更多宽度；顶部动作可恢复。
    app.processEvents();width_before=rgb.splitter.width()
    for container,action,label in rgb.sidebars: action.setChecked(True)
    app.processEvents()
    assert all(not container.isVisible() for container,_,_ in rgb.sidebars)
    assert rgb.splitter.width()>width_before
    for container,action,label in rgb.sidebars: action.setChecked(False)
    app.processEvents()
    assert all(container.isVisible() for container,_,_ in rgb.sidebars)

    # 小 Polygon 候选可以按 E 向外扩展；二维 mask 保持原样，预览仍可撤销。
    rgb.panel.tool.setCurrentIndex(1)
    viewer=rgb.panel.viewer;viewer.vertices=[[20,40],[30,40]];viewer.draw_polygon()
    viewer.cursor_point=QtCore.QPointF(30,50);viewer.draw_guide()
    assert len(viewer.vertices)==2 and viewer.guide_path.elementCount()==3
    assert viewer.show_cursor
    viewer.vertices=[[20,40],[30,40],[30,50],[20,50]];viewer.finish_polygon()
    assert len(viewer.vertices)==4 and viewer.guide_path.isEmpty()
    rgb.map_polygon();wait();small=r.candidate.copy();pending=rgb.gt_pending['fragment'].copy()
    assert len(small)>=6 and r.panel.expand_button.isEnabled()
    r.expand_local();wait();expanded=r.candidate.copy()
    assert len(expanded)>len(small)
    assert not np.isin([0,1,2],expanded).any()
    assert rgb.gt_pending['fragment']==pending
    np.testing.assert_array_equal(r.model.labels,original_labels)
    r.undo();wait();np.testing.assert_array_equal(r.candidate,small)
    r.redo();wait();np.testing.assert_array_equal(r.candidate,expanded)
    assert len(r.preview_edge_levels)==11 and r.preview_edge_thresholds[10][0]>0
    assert r.panel.edge_level_controls.isVisible()
    assert all(shortcut.isEnabled() for shortcut in r.panel.edge_level_shortcuts)
    with patch.object(owner,'save') as save,patch.object(owner,'next_pcd') as nxt,patch.object(owner,'prev_pcd') as prev:
        view.gl_widget.setFocus();app.processEvents()
        r.adjust_edge_level(-10)
        QtTest.QTest.keyClick(view.gl_widget,QtCore.Qt.Key_Right);app.processEvents()
        assert r.edge_level==1
        QtTest.QTest.keyClick(view.gl_widget,QtCore.Qt.Key_Left);app.processEvents()
        assert r.edge_level==0
        # 快捷键暂时失效时，旧键盘入口也不能保存/切文件。
        for shortcut in r.panel.edge_level_shortcuts: shortcut.setEnabled(False)
        QtTest.QTest.keyClick(view.gl_widget,QtCore.Qt.Key_Right);app.processEvents()
        assert r.edge_level==1
        assert not save.called and not nxt.called and not prev.called
        r.panel.set_preview(True)
    r.adjust_edge_level(-10)
    expanded=r.candidate.copy()

    # 固定补边结果，验证 RGB 入口调档、人工排除与混合撤销，而非依赖点云噪声。
    saved_level=r.edge_level
    def staged_grow(*args, **kwargs):
        assert kwargs['preview_levels'] is True
        r.spatial.last_strict=expanded.copy()
        r.spatial.edge_levels=[np.arange(625,625+i,dtype=np.int64) for i in range(11)]
        r.spatial.edge_thresholds=[(i*.1,i*.01) for i in range(11)]
        return expanded
    with patch.object(r.spatial,'grow',side_effect=staged_grow):
        r.expand_local();wait()
    r.adjust_edge_level(-10);np.testing.assert_array_equal(r.candidate,expanded)
    r.adjust_edge_level(10);assert np.isin(np.arange(625,635),r.candidate).all()
    assert not np.isin([0,1,2],r.candidate).any()
    r.preview_edits.edit(np.array([625]),False);r.adjust_edge_level(0)
    r.adjust_edge_level(-10);r.adjust_edge_level(10);assert 625 not in r.candidate
    r.undo();assert 625 in r.candidate
    r.undo();np.testing.assert_array_equal(r.candidate,expanded);assert r.edge_level==saved_level
    r.redo();r.adjust_edge_level(10);assert 625 in r.candidate
    assert rgb.gt_pending['fragment']==pending
    np.testing.assert_array_equal(r.model.labels,original_labels)
    r.cancel_expansion();wait()
    rgb.panel.tool.setCurrentIndex(1);rgb.panel.viewer.vertices=[[0,0],[99,0],[99,99],[0,99]];rgb.panel.viewer.finish_polygon()
    rgb.map_polygon();wait();original=r.candidate.copy()
    assert rgb.gt_pending is not None
    assert r.model.dual_gt.ensure(target)['rgb_gt_status']=='MISSING'
    assert len(original)==n-3 and g.run.isEnabled()
    assert g.parameter_fields[1].isEnabled()
    g.parameter_fields[1].setValue(.16);assert r.panel.plane_distance.value()==.16
    g.parameter_fields[1].setValue(.15)
    # LOD 只影响渲染，输入仍取全部原始索引和原始 LAS 坐标。
    r.cloud.lod_stride=8;g.refine();wait()
    assert g.last.success and len(r.candidate)==622
    assert np.max(r.candidate)<625 and not np.isin([0,1,2],r.candidate).any()
    np.testing.assert_array_equal(r.model.labels,original_labels)
    filtered=r.candidate.copy();r.undo();np.testing.assert_array_equal(r.candidate,original)
    r.redo();np.testing.assert_array_equal(r.candidate,filtered)
    g.show_result(False);np.testing.assert_array_equal(r.candidate,original)
    g.show_result(True);np.testing.assert_array_equal(r.candidate,filtered)
    # 失败不清空候选。
    g.minimum.setValue(10000);before=r.candidate.copy();g.refine();wait()
    np.testing.assert_array_equal(r.candidate,before);g.minimum.setValue(20)
    # 旧任务不能覆盖后台执行期间发生的预览修改。
    started=threading.Event();release=threading.Event();real=GeometryRefiner.refine_plane
    def slow(self,*args,**kwargs): started.set();assert release.wait(5);return real(self,*args,**kwargs)
    with patch.object(GeometryRefiner,'refine_plane',slow):
        g.refine();assert started.wait(5)
        r.preview_edits.edit(np.array([10]),False);r.adjust_edge_level(0);before=r.candidate.copy()
        release.set();wait();np.testing.assert_array_equal(r.candidate,before)
    # 人工保留一个偏离平面的点，最终方程必须重新基于这些 GT 点拟合。
    r.preview_edits.edit(np.array([625]),True);r.adjust_edge_level(0)
    final=r.candidate.copy();expected=plane_geometry(xyz[final],1,target)
    view.gl_widget.setFocus();QtTest.QTest.keyClick(view.gl_widget,QtCore.Qt.Key_Return);wait()
    assert r.candidate is None
    actual=r.model.session_metadata['plane_geometry'][str(target)]
    assert actual['point_count']==len(final)==r.model.plane_counts[target]
    np.testing.assert_allclose(actual['normal'],expected['normal'],atol=1e-9)
    assert actual['rms_m']>g.roof.model.header.scales[2]
    annotation=r.model.session_metadata['rgb_annotations'][-1]
    assert annotation['annotation_source']=='rgb_polygon_geometry_refined_lidar_corrected'
    assert annotation['manual_added']==1 and annotation['manual_removed']==1
    gt=r.model.dual_gt.ensure(target)
    assert gt['rgb_gt_status']=='CONFIRMED'
    fragment=gt['fragments'][0]
    assert r.model.dual_gt.store.read(fragment).sum()==99*99
    with patch.object(rgb.panel.viewer,'fit_bounds',wraps=rgb.panel.viewer.fit_bounds) as fit:
        row=r.panel.table_model.ids.index(target)
        r.select_plane(r.panel.table_model.index(row,0));app.processEvents()
        assert fit.called
        assert (r.cloud.rot_x,r.cloud.rot_y,r.cloud.rot_z)==(0.,0.,0.)
        local=r.cloud.points[r.model.plane_indices(target)]
        center=(local.min(axis=0)+local.max(axis=0))/2
        np.testing.assert_allclose(r.cloud.orbit_pivot,center,atol=1e-5)
        assert r.cloud.trans_z < -center[2]
        labels=r.model.labels.copy();r.focus_plane(0)
        np.testing.assert_array_equal(r.model.labels,labels)

    r.undo();wait();np.testing.assert_array_equal(r.model.labels,original_labels)
    assert not r.model.session_metadata['plane_geometry']
    assert r.model.dual_gt.ensure(target)['rgb_gt_status']=='MISSING'
    np.testing.assert_array_equal(r.candidate,final)
    assert len(rgb.panel.viewer.vertices)==4 and rgb.polygon is not None
    assert rgb.gt_pending['fragment']==fragment
    r.redo();wait();assert r.model.session_metadata['plane_geometry'][str(target)]['point_count']==len(final)
    assert r.model.dual_gt.ensure(target)['fragments']==[fragment]
    # 再次撤销后新建目标，不重画、不重映射；点击已有 Plane 也能改回。
    r.undo();wait();draft_points=r.candidate.copy();draft_vertices=list(rgb.panel.viewer.vertices)
    r.new_plane();new_target=r.expansion_target
    assert new_target!=target and rgb.gt_pending['plane_id']==new_target
    assert rgb.gt_pending['plane_uid']==r.model.dual_gt.ensure(new_target)['plane_uid']
    np.testing.assert_array_equal(r.candidate,draft_points)
    assert rgb.panel.viewer.vertices==draft_vertices
    r.retarget_preview(target)
    r.confirm_expansion();wait();assert r.candidate is None
    assert r.model.dual_gt.ensure(target)['fragments']==[fragment]
    r.dual.refresh_overlay();wait()
    assert not r.dual.overlay.pixmap().isNull()
    r.model.assign(target,ids=[626]);assert r.model.session_metadata['plane_geometry'][str(target)]['geometry_dirty']
    assert r.save(force=True);wait()
    assert not r._saving and r.candidate is None
    assert r.dual.export_button.isEnabled(), '保存完成后必须恢复双 GT 导出按钮'

    saved=json.loads((root/'scene.planar'/'plane_geometry.json').read_text())['planes'][str(target)]
    assert saved['point_count']==len(final)+1 and not saved['geometry_dirty']
    r.model.export(root/'export.las');out=laspy.read(root/'export.las')
    for field in las.points.array.dtype.names:
        if field!='plane_id':np.testing.assert_array_equal(out.points.array[field],las.points.array[field])
    # SAM 来源使用完全相同的 P0 mask 入口；不需要安装 SAM。
    r.model.assign(0,'all',np.arange(3,n));r.new_plane()
    rgb.map_mask(lambda:PixelMask(np.ones((100,100),bool),0,0),{'annotation_source':'rgb_sam'});wait()
    assert r.candidate is None  # 旧 RGB GT 独立于已清零的 LiDAR，不能重复占用。
    next_target=r.model.current;r.model.delete(target);r.model.current=next_target;r.refresh()
    rgb.map_mask(lambda:PixelMask(np.ones((100,100),bool),0,0),{'annotation_source':'rgb_sam'});wait()
    g.refine();wait();np.testing.assert_array_equal(r.candidate,filtered)
    before=r.model.labels.copy();r.clear_selection();wait();assert r.candidate is None
    np.testing.assert_array_equal(r.model.labels,before)
    r.model.selection[3:]=True;g.manual_candidate();wait();g.refine();wait()
    np.testing.assert_array_equal(r.candidate,filtered);r.cancel_expansion();wait()
    # 给已有 LiDAR Plane 单独补二维 GT，Undo 不触碰点标签。
    r.model.current=7;r.refresh();labels=r.model.labels.copy()
    rgb.panel.tool.setCurrentIndex(1);rgb.panel.viewer.vertices=[[0,0],[10,0],[10,10],[0,10]];rgb.panel.viewer.finish_polygon()
    r.dual.confirm_rgb();wait()
    assert r.model.dual_gt.ensure(7)['rgb_gt_status']=='CONFIRMED'
    np.testing.assert_array_equal(r.model.labels,labels)
    r.undo();wait();assert r.model.dual_gt.ensure(7)['rgb_gt_status']=='MISSING'
    np.testing.assert_array_equal(r.model.labels,labels)
    r.redo();wait();assert r.model.dual_gt.ensure(7)['rgb_gt_status']=='CONFIRMED'
    # 校验和导出分别启动后台任务，Esc 必须关联各自最新的取消事件。
    from labelCloud.model.dual_gt_export import DualGTExporter
    events=[]
    def validation_task(self,strict):
        assert self.cancel is r.cancel_task;events.append(self.cancel)
        return dict(ok=True,lidar_planes=1,rgb_planes=1,complete_planes=1,missing_rgb_planes=[],missing_lidar_planes=[],
            needs_review_planes=[],cross_modal_review_planes=[],dirty_geometry=0,conflicts=[],errors=[])
    def export_task(self,path,strict):
        assert self.cancel is r.cancel_task and self.cancel is not events[0];events.append(self.cancel)
        return path
    with patch.object(DualGTExporter,'validate',validation_task),patch.object(DualGTExporter,'export',export_task), \
            patch.object(QtWidgets.QInputDialog,'getItem',return_value=('Strict Dual-GT',True)), \
            patch.object(QtWidgets.QMessageBox,'exec_',return_value=QtWidgets.QMessageBox.Yes), \
            patch.object(QtWidgets.QFileDialog,'getExistingDirectory',return_value=str(root)):
        r.dual.export_dialog();wait()
    assert len(events)==2
    r.mode.setCurrentIndex(0);assert view.gl_widget.isVisible();r.mode.setCurrentIndex(1);wait()
    r.autosave_timer.stop();r.camera_timer.stop();view.timer.stop();view.hide()
print('P2 Qt Polygon / SAM-mask / manual safety and metadata OK')
