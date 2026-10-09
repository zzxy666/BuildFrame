"""RGB 只提供候选和辅助几何；唯一点标签仍由 RoofPlanes 管理。"""
from datetime import datetime, timezone
from pathlib import Path
import time
import numpy as np
from PyQt5 import QtCore, QtWidgets
from .scene_worker import SceneTask
from ..model.preview_edits import PreviewEdits
from ..view.orthophoto_view import OrthophotoPanel


class RGBPlaneController:
    def __init__(self,roof):
        self.roof=roof;self.model=None;self.photo=None;self.mapper=None
        self.polygon=None;self.overlay_key=None;self.overlay_task=None
        self.overlay_candidate_ref=None
        self.gt_pending=None
        self.panel=OrthophotoPanel(roof.view)
        self.splitter=QtWidgets.QSplitter(QtCore.Qt.Horizontal,roof.view)
        roof.view.verticalLayout_2.replaceWidget(roof.view.gl_widget,self.splitter)
        self.splitter.addWidget(self.panel);self.splitter.addWidget(roof.view.gl_widget)
        self.splitter.setStretchFactor(0,1);self.splitter.setStretchFactor(1,1)
        self.toolbar=roof.view.addToolBar("RGB / LiDAR")
        self.toolbar.setMovable(False)
        open_action=self.toolbar.addAction("打开正射影像")
        open_action.triggered.connect(self.open_dialog)
        self.layout_mode=QtWidgets.QComboBox()
        self.layout_mode.addItems(["只看 LiDAR","RGB + LiDAR","只看 RGB"])
        self.toolbar.addWidget(self.layout_mode)
        self.layout_mode.currentIndexChanged.connect(self.update_layout)
        # 将现有两侧布局装进容器，隐藏整个容器才能释放宽度。
        self.sidebars=[]
        for index,label in ((0,'左栏'),(2,'右栏')):
            item=roof.view.horizontalLayout.takeAt(index)
            container=QtWidgets.QWidget(roof.view)
            layout=item.layout();layout.setParent(None);container.setLayout(layout)
            roof.view.horizontalLayout.insertWidget(index,container)
            action=self.toolbar.addAction('隐藏'+label);action.setCheckable(True)
            action.toggled.connect(self.update_layout)
            self.sidebars.append((container,action,label))
        self.rgb_tools_action=self.toolbar.addAction('隐藏 RGB 工具')
        self.rgb_tools_action.setCheckable(True)
        self.rgb_tools_action.toggled.connect(self.toggle_rgb_tools)
        p=self.panel
        p.open_button.clicked.connect(self.open_dialog)
        p.fit_button.clicked.connect(p.viewer.fit_image)
        p.roi_button.clicked.connect(self.fit_roi)
        p.tool.currentIndexChanged.connect(lambda index:p.viewer.set_polygon_mode(index==1))
        p.viewer.polygon_ready.connect(self.polygon_ready)
        p.viewer.pixel_clicked.connect(self.query_pixel)
        p.viewer.activity.connect(self.activity)
        p.viewer.undo_requested.connect(roof.undo)
        p.viewer.redo_requested.connect(roof.redo)
        p.map_button.clicked.connect(self.map_polygon)
        p.overlay_mode.currentIndexChanged.connect(lambda _:self.refresh_overlay(force=True))
        p.opacity.valueChanged.connect(lambda value:p.viewer.overlay.setOpacity(value/100))
        p.viewer.overlay.setOpacity(.7)
        self.overlay_timer=QtCore.QTimer(p);self.overlay_timer.setSingleShot(True)
        self.overlay_timer.setInterval(80);self.overlay_timer.timeout.connect(self.refresh_overlay)
        from .sam_controller import SAMController
        self.sam=SAMController(self)
        self.update_layout()

    def toggle_rgb_tools(self, hidden):
        self.panel.controls.setVisible(not hidden)
        self.rgb_tools_action.setText('显示 RGB 工具' if hidden else '隐藏 RGB 工具')
        # 布局更新后补读更大视窗所需的影像，不重置视角或标注草稿。
        QtCore.QTimer.singleShot(0,self.panel.viewer.request_detail)

    def activity(self): self.roof._last_interaction=time.monotonic()

    def update_layout(self,*_):
        active=self.roof.active
        self.toolbar.setVisible(active)
        self.panel.setVisible(active and self.layout_mode.currentIndex()!=0)
        self.roof.view.gl_widget.setVisible(not active or self.layout_mode.currentIndex()!=2)
        for container,action,label in self.sidebars:
            container.setVisible(not active or not action.isChecked())
            action.setText(('显示' if action.isChecked() else '隐藏')+label)
        self.panel.viewer.request_detail()

    def on_model(self):
        if self.model is self.roof.model: return
        self.model=self.roof.model
        old=self.photo;self.photo=self.mapper=None;self.polygon=None;self.overlay_key=None
        self.panel.viewer.photo=None;self.panel.viewer.generation+=1
        self.panel.viewer.clear_polygon()
        for item in (self.panel.viewer.base,self.panel.viewer.detail,self.panel.viewer.overlay):
            from PyQt5 import QtGui
            item.setPixmap(QtGui.QPixmap())
        self.panel.map_button.setEnabled(False)
        if old is not None:
            # 阅读窗口与关闭通过 Orthophoto 锁串行化，避免关闭正在读取的句柄。
            task=SceneTask(old.close);task.signals.finished.connect(lambda *_:None)
            QtCore.QThreadPool.globalInstance().start(task)
        if self.model:
            registration=self.model.session_metadata.get("rgb_registration",{})
            path=registration.get("rgb_file")
            if path:
                model=self.model
                QtCore.QTimer.singleShot(0,lambda:self.open_file(path,restore=True) if self.roof.model is model and self.roof.active else None)

    def open_dialog(self):
        if self.roof.model is None or self.roof.preview_pending(): return
        path,_=QtWidgets.QFileDialog.getOpenFileName(self.roof.view,"打开带地理参考的正射影像","","GeoTIFF (*.tif *.tiff *.TIF *.TIFF)")
        if path: self.open_file(path)

    def open_file(self,path,restore=False):
        roof=self.roof
        if roof.model is None or roof.preview_pending(): return
        model=roof.model
        def work():
            from ..model.orthophoto import Orthophoto
            from ..model.coordinate_mapper import CoordinateMapper
            photo=Orthophoto(path)
            try:
                overview=photo.read_window()
                previous=model.session_metadata.get("rgb_registration",{})
                if restore and previous.get("fingerprint")!=photo.fingerprint:
                    raise ValueError("正射影像文件已变化，请重新打开并检查配准")
                try:
                    lidar_crs=model.header.parse_crs()
                    mapper=CoordinateMapper(photo.transform,photo.width,photo.height,lidar_crs,photo.crs)
                    model.session.verify_source()
                    mapper.build_las(model.path,roof.cancel_task)
                    model.session.verify_source()
                    if len(mapper.valid)!=len(model.labels): raise ValueError("原始点数不匹配")
                    error="" if mapper.valid.any() else "无投影点落在影像内，请检查影像范围和 CRS"
                except ValueError as exc:
                    mapper=None;error=str(exc)
                return photo,mapper,overview,error
            except Exception:
                photo.close();raise
        def done(result):
            photo,mapper,overview,error=result
            if roof.model is not model: photo.close();return
            old=self.photo
            self.photo,self.mapper=photo,mapper
            self.polygon=None;self.overlay_key=None
            self.panel.viewer.set_photo(photo,overview)
            if old is not None: old.close()
            self.layout_mode.setCurrentIndex(1);self.splitter.setSizes([600,600])
            lidar=model.header.parse_crs()
            def name(crs):
                if crs is None: return "缺失"
                from pyproj import CRS
                value=CRS(crs);authority=value.to_authority()
                return ':'.join(authority) if authority else value.name
            status=error or "CRS 映射就绪（仍需目视检查影像配准）"
            text=f"LiDAR CRS: {name(lidar)}\nRGB CRS: {name(photo.crs)}\nRegistration: {status}"
            self.panel.info.setText(text);roof.view.status_manager.set_message(text.replace('\n',' | '))
            self.panel.map_button.setEnabled(False)
            model.session_metadata['rgb_registration']={"rgb_file":str(photo.path),"rgb_crs":str(photo.crs),
                "lidar_crs":str(lidar),"geotransform":list(photo.transform)[:6],"width":photo.width,"height":photo.height,
                "nodata":None if photo.nodata is None else str(photo.nodata),"pixel_size":list(photo.resolution),"bounds":list(photo.bounds),
                "dx":0.,"dy":0.,"status":status,"fingerprint":photo.fingerprint}
            if not restore: model.touch()
            self.fit_roi();roof.refresh();self.refresh_overlay(force=True)
        roof.start_task("正在读取 GeoTIFF 并建立原始点到像素缓存…",work,done)

    def fit_roi(self):
        if self.mapper is None: return
        roof=self.roof
        bounds=roof.workspace.roi or roof.workspace.bounds
        low,high=np.asarray(bounds,float)
        points=np.array([[low[0],low[1],0],[low[0],high[1],0],[high[0],low[1],0],[high[0],high[1],0]])
        world=roof.cloud.to_world_coordinates(points)
        col,row=self.mapper.lidar_xy_to_pixel(world[:,0],world[:,1])
        self.panel.viewer.fit_bounds((min(col)-20,min(row)-20,max(col)+20,max(row)+20))

    def polygon_ready(self,vertices):
        self.activity();self.polygon=vertices
        self.panel.map_button.setEnabled(vertices is not None and self.mapper is not None and self.roof.candidate is None)
        if vertices is not None: self.panel.info.setText("Polygon 已闭合；点击“映射到 LiDAR”进入候选预览，尚未修改标签。")

    def map_polygon(self):
        roof=self.roof;model=roof.model;mapper=self.mapper
        if model is None or mapper is None or self.polygon is None or roof.preview_pending(): return
        vertices=self.polygon.copy()
        if model.current==0: roof.new_plane()
        x,y=mapper.pixel_to_world(vertices[:,0],vertices[:,1])
        metadata={"annotation_source":"rgb_polygon", "pixel_polygon":vertices.tolist(),
            "world_polygon":np.column_stack((x,y)).tolist(), "world_crs":str(self.photo.crs)}
        self.map_mask(lambda:mapper.polygon_mask(vertices),metadata)

    def map_mask(self,mask_factory,metadata,save_mask=False,valid=lambda:True,on_preview=lambda:None):
        """Polygon / SAM 共用 mask 入口；原始索引和安全过滤保持不变。"""
        roof=self.roof;model=roof.model;mapper=self.mapper;photo=self.photo
        if model is None or mapper is None or roof.preview_pending(): return False
        target=model.current
        replace_gt=roof.dual.replace.isChecked()
        def work():
            start=time.perf_counter()
            from ..model.dual_gt import rgb_identity
            mask,blocked=model.dual_gt.unlabelled_mask(mask_factory(),rgb_identity(photo.path),target,replace_gt)
            ids=mapper.pixel_mask_to_candidate_point_ids(mask,roof.cancel_task)
            allowed=(model.labels[ids]==0)&~model.hidden_mask[ids]
            roi=roof.roi_mask()
            if roi is not None: allowed &= roi[ids]
            mask_path=None
            if save_mask and allowed.any():
                import uuid
                folder=model.session.folder/'rgb_masks';folder.mkdir(parents=True,exist_ok=True)
                name=f'plane_{target}_sam_{uuid.uuid4().hex}.npz'
                mask_path=folder/name
                np.savez_compressed(mask_path,mask=mask.data,col0=mask.col,row0=mask.row)
            import logging
            logging.debug("RGB mask to LiDAR=%.1f ms candidate points=%s",(time.perf_counter()-start)*1000,int(allowed.sum()))
            fragment=None
            if allowed.any():
                from ..model.dual_gt import rgb_identity
                fragment=model.dual_gt.store.write(mask,rgb_identity(photo.path),metadata["annotation_source"],existing=mask_path)
            return ids[allowed],int(mask.data.sum()),mask_path,fragment,blocked
        def done(result):
            ids,pixels,mask_path,fragment,blocked=result
            if roof.model is not model or self.photo is not photo or not valid() or model.current!=target:
                if mask_path is not None: mask_path.unlink(missing_ok=True)
                return
            if not len(ids):
                self.panel.info.setText(f"RGB mask {pixels} pixels（已排除 {blocked} 个已标注像素）；无可用未标注 LiDAR 点（检查隐藏、ROI 和已有标签）")
                return
            roof.rgb_annotation_pending={**metadata,"plane_id":target,
                "rgb_file":str(photo.path),"rgb_fingerprint":photo.fingerprint,
                "protected_pixels_excluded":blocked,"mask_pixels":pixels,"projected_candidates":len(ids),"candidate_count":len(ids),
                "timestamp":datetime.now(timezone.utc).isoformat()}
            if mask_path is not None:
                roof.rgb_annotation_pending["mask_path"]=str(mask_path.relative_to(model.session.folder)).replace('\\','/')
            self.gt_pending=dict(fragment=fragment,plane_uid=model.dual_gt.ensure(target)["plane_uid"],plane_id=target,replace=replace_gt)
            on_preview()
            roof.candidate=ids;roof.expansion_target=target;roof.highlight=None
            roof.strict_candidate=ids.copy();roof.preview_edits=PreviewEdits()
            roof.preview_edge_levels=[np.empty(0,np.int64) for _ in range(11)]
            roof.preview_edge_thresholds=[(0.,0.)]*11;roof.edge_level=0
            roof.preview_fit_info=f"RGB mask pixels: {pixels} | 已保护像素: {blocked} | LiDAR candidates: {len(ids)}\n"
            roof.panel.set_preview(True);roof.panel.edge_level_controls.hide()
            for shortcut in roof.panel.edge_level_shortcuts: shortcut.setEnabled(False)
            self.panel.viewer.locked=True;self.panel.map_button.setEnabled(False)
            self.layout_mode.setCurrentIndex(1)
            roof.adjust_edge_level(0);roof.view.gl_widget.setFocus()
            self.panel.info.setText("已映射为候选。可先点击“几何过滤”，再在 LiDAR 用 Shift/Alt 修正；Enter 确认或 Esc 取消。")
        started=roof.start_task("正在将 RGB mask 映射为 LiDAR 候选…",work,done)
        if started: self.sync()
        return started

    def sync(self):
        self.update_layout()
        self.on_model()
        self.sam.sync()
        self.panel.viewer.locked=self.roof.candidate is not None or bool(self.roof.task)
        self.panel.map_button.setEnabled(not self.sam.active and self.mapper is not None and self.polygon is not None and self.roof.candidate is None and not self.roof.task)
        if self.photo is not None: self.overlay_timer.start()

    def refresh_overlay(self,force=False):
        # 左侧只显示原图或正式 RGB 像素 GT，不再绘制 LiDAR 投影。
        self.panel.viewer.overlay.hide()
        self.panel.viewer.base.show();self.panel.viewer.detail.show()
        if hasattr(self.roof,"dual"): self.roof.dual.refresh_overlay()

    def query_pixel(self,point):
        if self.mapper is None or self.roof.preview_pending(): return
        col,row=np.floor(point).astype(int)
        ids=self.mapper.pixel_point_ids(col,row)
        ids=ids[~self.roof.model.hidden_mask[ids]]
        values,counts=np.unique(self.roof.model.labels[ids],return_counts=True)
        planes=values[values>0]
        x,y=self.mapper.pixel_to_world(col+.5,row+.5)
        self.panel.info.setText(f"Pixel row={row}, col={col} | World XY={x:.3f}, {y:.3f}\nLiDAR points={len(ids)} | Plane IDs={planes.tolist()}")
        if not len(planes): return
        pid=int(planes[0])
        if len(planes)>1:
            text,ok=QtWidgets.QInputDialog.getItem(self.roof.view,"同一像素有多个 Plane","选择要查询的 Plane：",list(map(str,planes)),0,False)
            if not ok: return
            pid=int(text)
        self.roof.panel.plane_search.clear()
        self.roof.model.current=pid;self.roof.highlight=pid;self.roof.refresh()
        row_index=self.roof.panel.table_model.ids.index(pid)
        self.roof.panel.list.scrollTo(self.roof.panel.table_model.index(row_index,0),QtWidgets.QAbstractItemView.PositionAtCenter)

    def locate_point(self,index):
        if self.mapper is not None and self.mapper.valid[index]:
            col,row=self.mapper.lidar_to_pixel([index])[0]
            self.panel.viewer.locate(float(col),float(row));self.refresh_overlay(force=True)
