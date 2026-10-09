"""复用原鼠标、模式切换、隐藏交互；大场景任务与显示更新在此协调。"""
import logging
import threading
import time
import numpy as np
from PyQt5 import QtCore, QtWidgets
from .roof_plane_controller import LegacyRoofPlaneController
from .scene_worker import SceneTask, run_scene_task
from ..model.roof_planes import RoofPlanes, project_points
from ..model.screen_projection import ScreenProjectionCache
from ..model.scene_spatial import SceneSpatialIndex, coordinate_scale
from ..model.scene_workspace import SceneWorkspace


class RoofPlaneController(LegacyRoofPlaneController):
    def __init__(self, owner):
        super().__init__(owner)
        self.task = None
        self.cancel_task = threading.Event()
        self.projection_cache = ScreenProjectionCache()
        self.workspace = None
        self.spatial = None
        self.camera_key = None
        self.color_key = None
        self.hidden_key = None
        self.highlight_key = None
        self.highlight_ids = np.empty(0, np.int64)
        self.selected_ids = np.empty(0, np.int64)
        self._saving = False
        self._autosave_task = None
        self._autosave_error = None
        self._save_scheduled = None
        self._last_interaction = time.monotonic()
        self.rgb_colors = None
        self.rgb_annotation_pending = None

    def set_view(self, view):
        super().set_view(view)
        self.autosave_timer = QtCore.QTimer(view)
        self.autosave_timer.setSingleShot(True)
        self.autosave_timer.setInterval(5000)
        self.autosave_timer.timeout.connect(self.autosave)
        self.camera_timer = QtCore.QTimer(view)
        self.camera_timer.setSingleShot(True)
        self.camera_timer.setInterval(250)
        self.camera_timer.timeout.connect(self.camera_idle)
        from .rgb_plane_controller import RGBPlaneController
        self.rgb = RGBPlaneController(self)
        from .geometry_controller import GeometryController
        self.geometry = GeometryController(self)
        from .dual_gt_controller import DualGTController
        self.dual = DualGTController(self)

    def change_mode(self, index):
        if self.task or self._saving:
            self.mode.blockSignals(True); self.mode.setCurrentIndex(1 if self.active else 0); self.mode.blockSignals(False)
            self.view.status_manager.set_message("后台任务进行中，可按 Esc 取消后切换模式")
            return
        self.color_key = self.hidden_key = self.highlight_key = None
        super().change_mode(index)
        if hasattr(self,"rgb"): self.rgb.update_layout()
        if not self.active and self.cloud is not None:
            self.cloud.set_overlays([]); self.cloud.lod_stride = 1
            self.camera_timer.stop()

    def on_pointcloud_changed(self):
        self.projection_cache = ScreenProjectionCache()
        self.workspace = self.spatial = None
        self.camera_key = self.color_key = self.hidden_key = self.highlight_key = None
        self.selected_ids = np.empty(0,np.int64)
        super().on_pointcloud_changed()
        if hasattr(self,"rgb"): self.rgb.sync()

    def load_current(self):
        cloud = self.owner.pcd_manager.pointcloud
        self.panel.set_available(False)
        if cloud is None or cloud.path.suffix.lower() not in (".las", ".laz"):
            self.panel.message.setText("Roof Plane 需要 LAS/LAZ 点云")
            return
        try:
            if self.cloud is not cloud or self.model is None:
                def load():
                    model = RoofPlanes(cloud.path)
                    if len(model.labels) != len(cloud.points): raise ValueError("标签与原始点数不一致")
                    workspace = SceneWorkspace(cloud.points, model.session)
                    return model, workspace, model.colors()
                model, workspace, colors = run_scene_task(self.view,"正在载入场景标注…",load)
                self.model, self.cloud, self.workspace = model, cloud, workspace
                self.spatial = SceneSpatialIndex(cloud.points)
                self.projection_cache = ScreenProjectionCache()
                self.color_key = (id(model), 0, model.labels_revision)
                cloud.set_display_colors(colors)
                self.highlight = None; self.hidden_key = self.highlight_key = None
                self.rgb_colors = cloud.original_colors
                for control in (self.panel.selection_scope,self.panel.show_hidden): control.blockSignals(True)
                self.panel.selection_scope.setCurrentIndex(0); self.panel.show_hidden.setChecked(False)
                for control in (self.panel.selection_scope,self.panel.show_hidden): control.blockSignals(False)
                camera = model.session_metadata.get("camera")
                if camera: self.apply_camera(camera)
            self.panel.message.setText(f"{cloud.path.name}\n0 = Unassigned / Background；工作标注保存到 .planar")
            self.panel.set_available(True)
            self.refresh()
        except Exception as exc:
            logging.exception("Scene load failed")
            self.model=None
            self.panel.message.setText(f"载入失败：{exc}")

    def roi_mask(self):
        return self.workspace.roi_mask if self.workspace else None

    def selection_allowed(self, ids=None):
        allowed=self.model.selectable(self.panel.selection_scope.currentData(),ids)
        roi=self.roi_mask()
        if roi is not None: allowed &= roi if ids is None else roi[ids]
        if self.panel.show_hidden.isChecked(): allowed[:]=False
        return allowed

    def refresh(self):
        if self.model is None: return
        ids=np.flatnonzero(self.model.selection)
        rejected=ids[~self.selection_allowed(ids)]
        self.model.selection[rejected]=False
        self.selected_ids=ids[self.selection_allowed(ids)]
        self.panel.refresh(self.model)
        self.state_label.setText(f"{self.panel.selection_scope.currentText()} | 隐藏 {self.model.hidden_count} | "
                                f"Plane {self.model.current} | {'未保存' if self.model.dirty else '已保存'}")
        self.refresh_colors()
        if hasattr(self,"rgb"): self.rgb.sync()
        if hasattr(self,"geometry"): self.geometry.sync()
        if hasattr(self,"dual"): self.dual.sync()
        save_key=(id(self.model),self.model.revision)
        if self.model.dirty and not self._saving and save_key != self._save_scheduled:
            self._save_scheduled=save_key
            self.autosave_timer.start()

    def refresh_colors(self, *_):
        if not self.active or self.model is None or self.cloud is None: return
        start=time.perf_counter()
        mode=self.panel.display.currentIndex()
        key=(id(self.model),mode,self.model.labels_revision)
        if self.color_key != key:
            if self.color_key is None or self.color_key[:2] != key[:2]:
                if mode == 0:
                    colors=run_scene_task(self.view,"正在切换场景颜色…", self.model.colors)
                    self.cloud.set_display_colors(colors)
                else:
                    if self.rgb_colors is None:
                        value=0. if "red" in self.model.header.point_format.dimension_names else .65
                        self.rgb_colors=np.full(self.cloud.points.shape,value,np.float32)
                    colors = (run_scene_task(self.view,"正在切换场景颜色…",self.mixed_colors)
                              if mode == 2 else self.rgb_colors)
                    self.cloud.set_display_colors(colors)
            elif mode in (0, 2):
                ids=self.model.changed_ids
                colors=self.mixed_colors(ids) if mode == 2 else self.model.colors(ids)
                self.cloud.update_display_colors(ids,colors)
            self.color_key=key
        hidden_key=(id(self.model),self.model.hidden_revision,self.panel.show_hidden.isChecked())
        if hidden_key != self.hidden_key:
            if self.panel.show_hidden.isChecked(): self.cloud.set_display_mask(self.model.hidden_mask)
            elif self.model.hidden_count: self.cloud.set_display_mask(~self.model.hidden_mask)
            else: self.cloud.set_display_mask(None)
            self.hidden_key=hidden_key
        highlight_key=(id(self.model),self.highlight,self.model.labels_revision,self.model.hidden_revision,
                       self.panel.show_hidden.isChecked())
        if highlight_key != self.highlight_key:
            ids=np.empty(0,np.int64) if self.highlight is None else self.model.plane_indices(self.highlight)
            mask=self.model.hidden_mask[ids]
            self.highlight_ids=ids[mask if self.panel.show_hidden.isChecked() else ~mask]
            self.highlight_key=highlight_key
        layers=[(self.highlight_ids,(0.,.95,1.)),(self.selected_ids,(0.,1.,1.))]
        if self.candidate is not None:
            layers=[]  # 预览只显示有效候选，避免原种子高亮遮盖手工移除结果。
            layers.append((getattr(self,"yellow_candidate",self.candidate),(1.,.9,.05)))
            edge = getattr(self,"edge_candidate",np.empty(0,np.int64))
            if len(edge): layers.append((edge,(1.,.45,.05)))
        self.cloud.set_overlays(layers)
        if self.panel.debug_perf.isChecked(): logging.info("Overlay update selected=%d ms=%.2f",len(self.selected_ids),(time.perf_counter()-start)*1000)
        self.view.gl_widget.update()

    def mixed_colors(self, ids=None):
        # 独立颜色副本，避免标注或撤销污染原始 RGB；分块限制大场景临时内存。
        colors = self.rgb_colors.copy() if ids is None else self.rgb_colors[ids].copy()
        for start in range(0, len(colors), 250000):
            stop = min(start + 250000, len(colors))
            source = np.arange(start, stop) if ids is None else ids[start:stop]
            labelled = self.model.labels[source] > 0
            if labelled.any():
                colors[start:stop][labelled] = self.model.colors(source[labelled])
        return colors

    def start_task(self, text, work, done):
        if self.task or self._saving: return False
        self.cancel_task=threading.Event()
        self.view.status_manager.set_message(text)
        self.panel.tools.setEnabled(False)
        task=SceneTask(work)
        self.task=task
        def finish(value,error):
            self.task=None
            self.panel.tools.setEnabled(True)
            if hasattr(self,"rgb"): self.rgb.sync()
            self.view.gl_widget.setFocus()
            if error:
                if not isinstance(error[0],InterruptedError):
                    logging.error(error[1]); self.panel.expansion_info.setText(str(error[0]))
                self.view.status_manager.set_message(str(error[0]))
                return
            if self.cancel_task.is_set():
                self.view.status_manager.set_message("任务已取消，标签未改变")
                return
            done(value)
        task.signals.finished.connect(finish)
        QtCore.QThreadPool.globalInstance().start(task)
        return True

    def preview_pending(self):
        if self.task or self._saving:
            self.view.status_manager.set_message("后台计算中，请等待或按 Esc 取消")
            return True
        return super().preview_pending()

    def handle_mouse(self,event):
        if self.active and event.type() in (QtCore.QEvent.MouseButtonPress,QtCore.QEvent.MouseButtonRelease,
                                            QtCore.QEvent.MouseMove,QtCore.QEvent.Wheel):
            self._last_interaction=time.monotonic()
        if (self.active and self.candidate is not None and not self.task
                and self.orbit_position is None and self.panel.tool.currentData() in ("rectangle","lasso")):
            kind=event.type()
            if kind==QtCore.QEvent.MouseButtonPress and event.button()==QtCore.Qt.LeftButton:
                if event.modifiers() & (QtCore.Qt.ShiftModifier|QtCore.Qt.AltModifier):
                    self.gesture=[(event.x(),event.y())]; self.gesture_modifiers=event.modifiers()
                    self.view.gl_widget.setFocus()
                else: self.view.status_manager.set_message("预览修正：按住 Shift 添加，Alt 移除")
                return True
            if self.gesture and kind==QtCore.QEvent.MouseMove:
                point=(event.x(),event.y())
                if self.panel.tool.currentData()=="rectangle": self.gesture=[self.gesture[0],point]
                elif np.linalg.norm(np.asarray(point)-self.gesture[-1])>=2: self.gesture.append(point)
                self.view.gl_widget.update(); return True
            if self.gesture and kind==QtCore.QEvent.MouseButtonRelease and event.button()==QtCore.Qt.LeftButton:
                self.finish_selection((event.x(),event.y())); return True
        if (self.active and self.model is not None and self.panel.tool.currentData()=="query"
                and self.orbit_position is None):
            kind = event.type()
            if kind in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseButtonRelease, QtCore.QEvent.MouseButtonDblClick) and event.button()==QtCore.Qt.LeftButton:
                self.view.gl_widget.setFocus()
                if kind==QtCore.QEvent.MouseButtonRelease:
                    self.query_plane((event.x(),event.y()))
                return True
            if kind==QtCore.QEvent.MouseMove and event.buttons() & QtCore.Qt.LeftButton:
                return True
        if self.active and self.task:
            if event.type() in (QtCore.QEvent.MouseButtonPress,QtCore.QEvent.MouseButtonRelease):
                if event.button()==QtCore.Qt.LeftButton and self.panel.tool.currentData()!="navigate": return True
            if event.type()==QtCore.QEvent.MouseMove and event.buttons()&QtCore.Qt.LeftButton and self.panel.tool.currentData()!="navigate": return True
        return super().handle_mouse(event)

    def query_plane(self, point):
        if self.model is None or self.preview_pending(): return
        if self.panel.show_hidden.isChecked():
            self.view.status_manager.set_message("请先退出仅显示隐藏点，再查询 Plane"); return
        widget=self.view.gl_widget
        if widget.modelview is None or widget.projection is None: return
        model=self.model; points=self.cloud.points
        mv=np.array(widget.modelview); pr=np.array(widget.projection)
        width,height=widget.width(),widget.height()
        def work():
            self.projection_cache.project(points,mv,pr,width,height,self.cancel_task)
            return self.projection_cache.pick(point,model.hidden_mask,cancel=self.cancel_task)
        def done(index):
            if self.model is not model: return
            if index is None or model.hidden_mask[index]:
                self.view.status_manager.set_message("未命中可见点，请靠近点点击"); return
            pid=int(model.labels[index])
            if pid==0:
                self.view.status_manager.set_message("该点未标注（Plane 0）"); return
            # 查询不受选择保护限制，也不改变选区或标签。
            self.panel.plane_search.clear()
            model.current=pid; self.highlight=pid; self.refresh()
            row=self.panel.table_model.ids.index(pid)
            self.panel.list.scrollTo(self.panel.table_model.index(row,0),QtWidgets.QAbstractItemView.PositionAtCenter)
            self.view.status_manager.set_message(f"查询：Plane {pid}，{model.plane_counts.get(pid,0)} points")
            if hasattr(self,"rgb"): self.rgb.locate_point(index)
        self.start_task("正在查询 Plane…",work,done)

    def clear_selection(self):
        if hasattr(self,"geometry"): self.geometry.cancel.set()
        if self.task:
            self.cancel_task.set(); return
        super().clear_selection()

    def undo(self):
        if self.task:
            self.cancel_task.set(); return
        if self.candidate is not None:
            self.preview_edits.undo(); self.adjust_edge_level(0); return
        record=self.model.history[-1] if self.model is not None and self.model.history else None
        super().undo()
        if record is not None and record.get('rgb_draft') is not None:
            self.restore_rgb_draft(record['rgb_draft'])

    def redo(self):
        if self.candidate is not None and not self.task:
            token=getattr(self,'_draft_redo',None)
            if (token is not None and token[0] is self.model and token[1] is self.candidate
                    and self.model.redo_history and token[2] is self.model.redo_history[-1]):
                self.cancel_expansion()
                self.model.redo();self.refresh();return
            self.preview_edits.redo(); self.adjust_edge_level(0); return
        if self.model is None or self.preview_pending(): return
        if not self.model.redo():
            self.view.status_manager.set_message("没有可重做的操作"); return
        self.panel.show_hidden.blockSignals(True)
        self.panel.show_hidden.setChecked(False)
        self.panel.show_hidden.blockSignals(False)
        self.highlight=None
        self.refresh()
        self.view.status_manager.set_message("已重做 [Ctrl+Y]")

    def finish_selection(self,point):
        if self.panel.tool.currentData()=="rectangle": self.gesture=[self.gesture[0],point]
        else: self.gesture.append(point)
        polygon=self.polygon(); modifiers=self.gesture_modifiers
        self.cancel_gesture()
        if len(polygon)<3 or self.panel.show_hidden.isChecked(): return
        widget=self.view.gl_widget
        if widget.modelview is None or widget.projection is None: return
        mv=np.array(widget.modelview); pr=np.array(widget.projection); w,h=widget.width(),widget.height()
        scope=self.panel.selection_scope.currentData(); current=self.model.current
        editing_preview=self.candidate is not None
        if editing_preview:
            if not modifiers & (QtCore.Qt.ShiftModifier|QtCore.Qt.AltModifier): return
            scope="unlabelled"
        roi=self.roi_mask(); model=self.model; points=self.cloud.points
        debug=self.panel.debug_perf.isChecked()
        def work():
            self.projection_cache.project(points,mv,pr,w,h,self.cancel_task)
            return self.projection_cache.select(polygon,model.labels,model.hidden_mask,scope,current,roi,self.cancel_task,debug)
        def done(ids):
            if self.model is not model: return
            if editing_preview:
                if self.candidate is None: return
                allowed=model.selectable("unlabelled",ids)
                if roi is not None: allowed &= roi[ids]
                ids=ids[allowed]
                adding=not bool(modifiers & QtCore.Qt.AltModifier)
                if not adding: ids=np.intersect1d(ids,self.candidate)
                self.preview_edits.edit(ids,adding)
                self.adjust_edge_level(0)
                return
            ids=ids[self.selection_allowed(ids)]
            if modifiers&QtCore.Qt.AltModifier: model.selection[ids]=False
            elif modifiers&QtCore.Qt.ShiftModifier: model.selection[ids]=True
            else:
                model.selection[:]=False; model.selection[ids]=True
            self.highlight=None; self.refresh()
            self.view.status_manager.set_message(f"已选择 {len(self.selected_ids)} points")
        self.start_task("正在选择…",work,done)

    def camera_observed(self,mv,pr,width,height):
        if self.cloud is None: return
        key=ScreenProjectionCache.camera_key(self.cloud.points,mv,pr,width,height)
        if key!=self.camera_key:
            self._last_interaction=time.monotonic()
            self.camera_key=key
            if self.panel.interactive_lod.isChecked(): self.cloud.lod_stride=8
            self.camera_timer.start()

    def camera_idle(self):
        if not self.active or self.cloud is None: return
        self.cloud.lod_stride=1; self.view.gl_widget.update()
        # 相机停止时预热缓存；若正在选择/扩展，下一次选择按矩阵键自动失效。
        if self.task or self._saving or self.gesture or self.model is None or self.candidate is not None: return
        widget=self.view.gl_widget
        if widget.modelview is None or widget.projection is None: return
        mv=np.array(widget.modelview); pr=np.array(widget.projection); w,h=widget.width(),widget.height()
        key=ScreenProjectionCache.camera_key(self.cloud.points,mv,pr,w,h)
        if self.projection_cache.key==key: return
        # 预热也使用受控任务，避免两个后台线程同时写缓存。
        self.start_task("正在更新屏幕投影缓存…",
                        lambda:self.projection_cache.project(self.cloud.points,mv,pr,w,h,self.cancel_task),
                        lambda _:self.view.status_manager.set_message("视角缓存已更新"))

    def assign(self,plane_id=None):
        if self.preview_pending() or self.model is None: return
        ids=np.flatnonzero(self.model.selection)
        ids=ids[self.selection_allowed(ids)]
        self.model.assign(self.model.current if plane_id is None else plane_id,
                          self.panel.selection_scope.currentData(),ids)
        self.highlight=None; self.refresh()

    def new_plane(self):
        try:
            if self.candidate is not None:
                if self.task or self._saving or self.geometry.task is not None: return
                if self.rgb.gt_pending and self.rgb.gt_pending.get('replace'):
                    self.view.status_manager.set_message('替换 GT 草稿请先取消，按普通追加模式映射后再更换 Plane');return
                self.retarget_preview(self.model.new_plane());return
            super().new_plane()
        except ValueError as exc:
            self.view.status_manager.set_message(str(exc))

    def retarget_preview(self, pid):
        """只更换候选的归属，不重新映射、不改点索引和 RGB mask。"""
        if self.task or self._saving or self.geometry.task is not None: return
        if pid <= 0 or pid not in self.model.plane_ids:
            self.view.status_manager.set_message('请选择非零 Plane，或新建 Plane');return
        # 替换当前 GT 的草稿可能包含旧 Plane 像素，不能直接转给其他 Plane。
        pending=self.rgb.gt_pending
        if pending and pending.get('replace') and pid!=self.expansion_target:
            self.view.status_manager.set_message('替换 GT 草稿不能直接换目标；请取消后按普通追加模式映射');return
        self.model.current=pid;self.expansion_target=pid;self.highlight=None
        if pending is not None:
            self.rgb.gt_pending={**pending,'plane_id':pid,'plane_uid':self.model.dual_gt.ensure(pid)['plane_uid']}
        if self.rgb_annotation_pending is not None:
            self.rgb_annotation_pending={**self.rgb_annotation_pending,'plane_id':pid}
        self.geometry.confirm_ready=False
        self._draft_redo=None
        self.adjust_edge_level(0)
        self.view.status_manager.set_message(f'候选目标已改为 Plane {pid}；Polygon 和候选点均保留')

    def select_plane(self, item):
        pid=int(item.data(QtCore.Qt.UserRole))
        if self.candidate is not None:
            self.retarget_preview(pid)
        else:
            super().select_plane(item)
        if self.model is not None and self.model.current==pid:
            self.focus_plane(pid)

    def focus_plane(self, pid):
        """联动定位只调整相机，不改标签、工作区或隐藏状态。"""
        if pid<=0 or self.model is None or self.task: return
        ids=self.model.plane_indices(pid)
        bounds=None
        if len(ids):
            points=self.cloud.points[ids]
            low=points.min(axis=0).astype(float);high=points.max(axis=0).astype(float)
            center=(low+high)/2
            viewport=self.view.gl_widget
            aspect=max(viewport.width(),1)/max(viewport.height(),1)
            span=max(float(high[1]-low[1]),float(high[0]-low[0])/aspect,.1)
            distance=span*.6/np.tan(np.radians(22.5))+float(high[2]-low[2])*.5
            self.cloud.orbit_pivot=center
            self.cloud.rot_x=self.cloud.rot_y=self.cloud.rot_z=0.
            self.cloud.trans_x=-center[0];self.cloud.trans_y=-center[1]
            self.cloud.trans_z=-center[2]-distance
            viewport.update()
            if self.rgb.mapper is not None:
                corners=np.array([[low[0],low[1],0],[low[0],high[1],0],
                                  [high[0],low[1],0],[high[0],high[1],0]])
                world=self.cloud.to_world_coordinates(corners)
                col,row=self.rgb.mapper.lidar_xy_to_pixel(world[:,0],world[:,1])
                bounds=(min(col),min(row),max(col),max(row))
        if self.rgb.photo is not None:
            # 按实际有标签像素定位，不能直接用 SAM 的整块 patch 边界。
            record=self.model.dual_gt.state['active'].get(str(pid),{})
            boxes=[]
            for fragment in record.get('fragments',[]):
                if str(self.rgb.photo.path)!=fragment['source']['path']: continue
                try:
                    mask=self.model.dual_gt.store.read(fragment)
                    rows=np.flatnonzero(mask.any(axis=1));cols=np.flatnonzero(mask.any(axis=0))
                    if len(rows) and len(cols):
                        boxes.append((fragment['col0']+cols[0],fragment['row0']+rows[0],
                                      fragment['col0']+cols[-1]+1,fragment['row0']+rows[-1]+1))
                except (OSError,ValueError,KeyError): continue
            if boxes:
                boxes=np.asarray(boxes);bounds=(boxes[:,0].min(),boxes[:,1].min(),boxes[:,2].max(),boxes[:,3].max())
            if bounds is not None and np.isfinite(bounds).all():
                left,top,right,bottom=bounds;padding=max(right-left,bottom-top,10)*.12
                self.rgb.panel.viewer.fit_bounds((left-padding,top-padding,right+padding,bottom+padding))
        self._last_interaction=time.monotonic()

    def capture_rgb_draft(self):
        if self.rgb.gt_pending is None: return None
        import copy
        names=('candidate','strict_candidate','preview_edge_levels','preview_edge_thresholds',
               'preview_fit_info','edge_level','expansion_target','preview_edits')
        return dict(photo=self.rgb.photo, values={name:getattr(self,name) for name in names},
                    vertices=copy.deepcopy(self.rgb.panel.viewer.vertices),polygon=copy.deepcopy(self.rgb.polygon),
                    pending=copy.deepcopy(self.rgb.gt_pending),annotation=copy.deepcopy(self.rgb_annotation_pending),
                    sam={name:copy.deepcopy(getattr(self.rgb.sam,name)) if name=='prompts' else getattr(self.rgb.sam,name)
                         for name in ('patch','prediction','mask_index','target','prompts','state')},
                    geometry={name:getattr(self.geometry,name) for name in ('last','original','refined','settings')})

    def restore_rgb_draft(self, draft):
        # 跨影像的草稿不放回当前视图，避免像素坐标错位。
        if draft['photo'] is not self.rgb.photo: return
        import copy
        for name,value in draft['values'].items(): setattr(self,name,value)
        self.rgb.gt_pending=copy.deepcopy(draft['pending'])
        self.rgb_annotation_pending=copy.deepcopy(draft['annotation'])
        self.rgb.polygon=copy.deepcopy(draft['polygon'])
        self.rgb.panel.viewer.vertices=copy.deepcopy(draft['vertices'])
        self.rgb.panel.viewer.draw_polygon()
        for name,value in draft['sam'].items():
            setattr(self.rgb.sam,name,copy.deepcopy(value) if name=='prompts' else value)
        self.rgb.sam.draw()
        if self.rgb.sam.active: self.rgb.sam.enter_candidate()
        for name,value in draft['geometry'].items(): setattr(self.geometry,name,value)
        self.geometry.confirm_ready=False
        self.model.current=self.expansion_target
        self.panel.set_preview(True);self.adjust_edge_level(0);self.rgb.sync()
        self._draft_redo=(self.model,self.candidate,self.model.redo_history[-1])
        self.view.status_manager.set_message('已撤销 GT 并恢复草稿；可按 N 新建 Plane，或点击列表换目标，再 Enter 确认')

    def update_hidden(self,action):
        if not self.active or self.model is None or self.preview_pending(): return
        ids=np.flatnonzero(self.model.selection); ids=ids[self.selection_allowed(ids)]
        if action!="restore" and not len(ids):
            self.view.status_manager.set_message("请先选择点"); return
        if action=="restore": self.model.set_hidden_ids(np.flatnonzero(self.model.hidden_mask),False)
        elif action=="hide": self.model.set_hidden_ids(ids,True)
        else:
            # 只创建一次操作用临时掩膜，撤销仅保留变化索引。
            mask=~self.model.hidden_mask; mask[ids]=False
            self.model.set_hidden_ids(np.flatnonzero(mask),True)
        self.panel.show_hidden.blockSignals(True); self.panel.show_hidden.setChecked(False); self.panel.show_hidden.blockSignals(False)
        self.refresh()

    def expand_local(self):
        if self.candidate is not None:
            self.grow_candidate();return
        if self.model is None or self.preview_pending(): return
        model=self.model; roi=self.roi_mask()
        seeds=np.flatnonzero(model.selection); seeds=seeds[self.selection_allowed(seeds)]
        options=self.panel.growth_options(); units=self.panel.coordinate_units.currentData()
        try: scale=coordinate_scale(model.header,units)*self.cloud.applied_scale
        except Exception as exc:
            self.panel.expansion_info.setText(str(exc)); return
        current=model.current; debug=self.panel.debug_perf.isChecked()
        def work():
            return self.spatial.grow(model.labels,model.hidden_mask,seeds,current,scale,options,roi,
                                     model.hidden_revision,self.workspace.roi_revision,self.cancel_task,debug,preview_levels=True)
        def done(ids):
            fit_info = (f"RANSAC 种子内点 {self.spatial.last_seed_inliers}/{self.spatial.last_seed_total}，排除 {self.spatial.last_seed_total-self.spatial.last_seed_inliers} 点。\n" if options.robust_fit else "")
            if not len(ids):
                self.panel.expansion_info.setText(fit_info+"没有可扩展的未标注点"); return
            self.candidate=ids; self.expansion_target=current; self.highlight=None
            self.strict_candidate=self.spatial.last_strict
            from ..model.preview_edits import PreviewEdits
            self.preview_edits=PreviewEdits()
            self.preview_edge_levels=self.spatial.edge_levels
            self.preview_edge_thresholds=self.spatial.edge_thresholds
            self.preview_fit_info=fit_info
            self.edge_level=3 if options.edge_completion else 0
            self.panel.set_preview(True)
            self.adjust_edge_level(0)
        self.start_task("正在局部扩展…（首次建立空间索引）",work,done)

    def grow_candidate(self):
        """复用局部扩展；只追加预览点，不改变二维 mask 和已提交标签。"""
        if self.model is None or self.task or self._saving: return
        if hasattr(self,'geometry') and self.geometry.task is not None:
            self.panel.expansion_info.setText('请等待几何过滤完成后再扩展');return
        model=self.model;before=self.candidate.copy();candidate_ref=self.candidate
        target=self.expansion_target;roi=self.roi_mask()
        allowed=model.selectable('unlabelled',before)
        if roi is not None: allowed &= roi[before]
        seeds=before[allowed]
        if len(seeds)<6:
            self.panel.expansion_info.setText('局部扩展至少需要 6 个同一平面的候选点');return
        debug=self.panel.debug_perf.isChecked()
        options=self.panel.growth_options()
        try: scale=coordinate_scale(model.header,self.panel.coordinate_units.currentData())*self.cloud.applied_scale
        except Exception as exc: self.panel.expansion_info.setText(str(exc));return
        def work():
            return self.spatial.grow(model.labels,model.hidden_mask,seeds,target,scale,options,roi,
                model.hidden_revision,self.workspace.roi_revision,self.cancel_task,debug,preview_levels=True)
        def done(ids):
            if self.model is not model or self.candidate is not candidate_ref: return
            # 基础候选和可调边缘分开保存，不能把高档补边永久并入主体。
            fields=('strict_candidate','preview_edge_levels','preview_edge_thresholds',
                    'preview_fit_info','edge_level')
            edits=self.preview_edits
            def snapshot():
                return dict(values={name:getattr(self,name) for name in fields},
                            added=edits.added.copy(),removed=edits.removed.copy())
            def restore(state):
                for name,value in state['values'].items(): setattr(self,name,value)
                edits.added=state['added'].copy();edits.removed=state['removed'].copy()
            previous=snapshot()
            def eligible(values):
                allowed=model.selectable('unlabelled',values)
                if roi is not None: allowed &= roi[values]
                return values[allowed]
            self.strict_candidate=np.union1d(eligible(before),eligible(self.spatial.last_strict))
            self.preview_edge_levels=[eligible(level) for level in self.spatial.edge_levels]
            self.preview_edge_thresholds=list(self.spatial.edge_thresholds)
            self.preview_fit_info='RGB/候选局部扩展；二维 Mask 保持不变。\n'
            self.edge_level=3 if options.edge_completion else 0
            edits.record_state(previous,snapshot(),restore)
            self.panel.set_preview(True)
            self.view.gl_widget.setFocus()
            self.adjust_edge_level(0)
            self.view.status_manager.set_message('候选已扩展：←/→ 调整 0～10 级补边；Enter 确认 / Ctrl+Z 撤销。RGB 区域不变。')
        self.start_task('正在从当前候选局部扩展；多屋面混合时请先几何过滤…',work,done)

    def adjust_edge_level(self, delta):
        self._last_interaction=time.monotonic()
        if self.task or self.model is None or self.candidate is None: return
        self.edge_level=max(0,min(10,self.edge_level+delta))
        self.yellow_candidate,self.edge_candidate=self.preview_edits.compose(
            self.strict_candidate,self.preview_edge_levels[self.edge_level])
        self.candidate=np.union1d(self.yellow_candidate,self.edge_candidate)
        radius,distance=self.preview_edge_thresholds[self.edge_level]
        self.panel.expansion_info.setText(self.preview_fit_info+
            f"边缘补选 {self.edge_level}/10 档 | 补选 {len(self.edge_candidate)} 点\n"
            f"邻距 {radius:.3f} m / 面距 {distance:.3f} m\n"
            f"黄色候选 {len(self.yellow_candidate)} 点，橙色补选 → Plane {self.expansion_target}\n"
            f"手工添加 {len(self.preview_edits.added)} / 移除 {len(self.preview_edits.removed)} 点\n"
            "Shift 圈选添加 / Alt 圈选移除；Ctrl+Z/Y 撤销/重做修正\n"
            "←/→ 调档，Enter 确认 / Esc 取消")
        self.refresh()

    def cancel_expansion(self):
        if hasattr(self,"geometry"): self.geometry.reset()
        if hasattr(self,"rgb"):
            self.rgb.sam.finish_candidate()
            self.rgb.gt_pending=None
        self.rgb_annotation_pending=None
        self.preview_edits=None
        self.yellow_candidate=np.empty(0,np.int64)
        self.strict_candidate=np.empty(0,np.int64)
        self.edge_candidate=np.empty(0,np.int64)
        self.preview_edge_levels=[]
        self.preview_edge_thresholds=[]
        super().cancel_expansion()

    def confirm_expansion(self):
        if self.task or self.model is None or self.candidate is None: return
        if hasattr(self,"dual") and not self.dual.validate_pending(): return
        if hasattr(self,"geometry") and self.geometry.prepare_confirm(): return
        draft=self.capture_rgb_draft()
        ids=self.candidate
        allowed=self.model.selectable("unlabelled",ids)
        roi=self.roi_mask()
        if roi is not None: allowed &= roi[ids]
        changed=self.model.assign(self.expansion_target,"unlabelled",ids[allowed])
        if changed and draft is not None:
            record=self.model.history[-1]
            record['rgb_draft']=draft
            previous_bytes=record['bytes'];record['bytes']=self.model._record_bytes(record)
            self.model.history_bytes+=record['bytes']-previous_bytes
        if changed and hasattr(self,"geometry"): self.geometry.attach_confirmed()
        if changed and self.rgb_annotation_pending is not None:
            annotation=dict(self.rgb_annotation_pending)
            annotation["confirmed_points"]=int(allowed.sum())
            annotation["manual_added"]=len(self.preview_edits.added)
            annotation["manual_removed"]=len(self.preview_edits.removed)
            if hasattr(self,"geometry") and self.geometry.last is not None:
                annotation["manual_added"]=len(np.setdiff1d(ids[allowed],self.geometry.refined))
                annotation["manual_removed"]=len(np.setdiff1d(self.geometry.refined,ids[allowed]))
            if annotation["manual_added"] or annotation["manual_removed"]:
                annotation["annotation_source"]+="_lidar_corrected"
            annotation["final_confirmed_count"]=annotation["confirmed_points"]
            annotation["added_in_lidar"]=annotation["manual_added"]
            annotation["removed_in_lidar"]=annotation["manual_removed"]
            self.model.attach_rgb_annotation(annotation)
            if self.rgb.gt_pending is not None:
                self.model.dual_gt.confirm(self.expansion_target,self.rgb.gt_pending["fragment"],replace=self.rgb.gt_pending.get("replace",False),attached=True)
            self.rgb.sam.finish_candidate(confirmed=True)
            self.rgb.panel.viewer.clear_polygon()
            self.rgb.polygon=None
        self.model._trim_history()
        self.cancel_expansion()
        self.view.status_manager.set_message("扩展已确认，可撤销并恢复 RGB 草稿")

    def delete(self):
        if self.model is None or self.preview_pending(): return
        if self.panel.selection_scope.currentData()=="unlabelled":
            self.view.status_manager.set_message("请切换到当前 Plane 或所有可见点后编辑"); return
        self.model.delete(self.model.current,self.operation_mask())
        self.highlight=None; self.refresh()

    def operation_mask(self):
        mask=~self.model.hidden_mask
        if self.roi_mask() is not None: mask &= self.roi_mask()
        if self.panel.show_hidden.isChecked(): mask[:]=False
        return mask

    def merge(self):
        if self.model is None or self.preview_pending(): return
        if self.panel.selection_scope.currentData()=="unlabelled":
            self.view.status_manager.set_message("请先切换到纠错选择范围"); return
        text,ok=QtWidgets.QInputDialog.getText(self.view,"合并 Plane","目标 Plane ID：")
        if not ok: return
        try:
            self.model.merge(self.model.current,int(text),self.operation_mask())
            self.highlight=self.model.current; self.refresh()
        except ValueError as exc: self.view.status_manager.set_message(str(exc))

    def jump_plane(self):
        if self.model is None or self.preview_pending(): return
        text=self.panel.plane_search.text().strip()
        try:
            pid=int(text)
            if pid not in self.model.plane_ids: raise ValueError()
        except ValueError:
            self.view.status_manager.set_message("请输入现有 Plane ID"); return
        self.model.current=pid; self.highlight=pid; self.refresh()

    def step_plane(self,direction):
        if self.model is None or self.preview_pending(): return
        ids=sorted(self.model.plane_ids); at=ids.index(self.model.current)
        self.model.current=ids[(at+direction)%len(ids)]; self.highlight=self.model.current; self.refresh()

    def search_planes(self,text):
        if self.model is not None: self.panel.table_model.refresh(self.model,text.strip())

    def capture_camera(self):
        cloud=self.cloud
        camera=dict(rotation=list(cloud.get_rotations()),translation=[cloud.trans_x,cloud.trans_y,cloud.trans_z],
                    pivot=cloud.orbit_pivot.tolist())
        mv=self.view.gl_widget.modelview
        if mv is not None:
            inverse=np.linalg.inv(mv)
            camera.update(position=inverse[3,:3].tolist(),focal_point=cloud.orbit_pivot.tolist(),view_up=inverse[1,:3].tolist())
        return camera

    def apply_camera(self,camera):
        self.cloud.rot_x,self.cloud.rot_y,self.cloud.rot_z=camera["rotation"]
        self.cloud.trans_x,self.cloud.trans_y,self.cloud.trans_z=camera["translation"]
        self.cloud.orbit_pivot=np.asarray(camera["pivot"],float)
        self.view.gl_widget.update()

    def save(self,force=False):
        if self.model is None: return True
        if self._saving: return False
        if self._autosave_task is not None:
            # 显式保存/切文件/退出先等待旧快照落盘，再保存最新版本。
            self._saving=True
            loop=QtCore.QEventLoop()
            self._autosave_task.signals.finished.connect(lambda *_:loop.quit())
            loop.exec_()
            self._saving=False
        if self.task:
            self.view.status_manager.set_message("请等待后台任务完成，或按 Esc 取消后再保存/切换")
            return False
        self.autosave_timer.stop()
        self._saving=True
        try:
            self.model.session_metadata.update(camera=self.capture_camera(),roi=self.workspace.roi,
                                               render_origin=self.cloud.original_center.tolist(),
                                               render_scale=self.cloud.applied_scale)
            self.state_label.setText("正在保存…")
            def work():
                path=self.model.save(); self.workspace.save(); return path
            path=run_scene_task(self.view,"正在保存工作标注…",work)
            self.view.status_manager.set_message(f"工作标注已保存：{path}")
            return True
        except Exception as exc:
            logging.exception("Session save failed")
            self.model.touch()
            QtWidgets.QMessageBox.critical(self.view,"保存失败",str(exc)); return False
        finally:
            # 先解除保存状态再刷新，否则双 GT 导出等按钮会停留在禁用状态。
            self._saving=False
            self.refresh()

    def autosave(self):
        if self.model is None or not self.model.dirty: return
        if (self.task or self._saving or self._autosave_task is not None or self.gesture
                or self.orbit_position is not None or self.candidate is not None
                or QtWidgets.QApplication.mouseButtons()!=QtCore.Qt.NoButton
                or time.monotonic()-self._last_interaction<5):
            self.autosave_timer.start(); return
        import copy
        from ..model.scene_session import atomic_json
        model=self.model; session=model.session; revision=model.revision
        metadata=copy.deepcopy(model.session_metadata)
        metadata.update(max_plane_id=model.max_plane_id,current_plane=model.current,
                        empty_planes=[pid for pid in model.plane_ids if not model.plane_counts.get(pid)],
                        camera=self.capture_camera(),roi=copy.deepcopy(self.workspace.roi),
                        render_origin=self.cloud.original_center.tolist(),render_scale=self.cloud.applied_scale)
        geometry_jobs=model.geometry_jobs(metadata)
        snapshot=session.snapshot(model.labels,metadata)
        workspace=copy.deepcopy((self.workspace.grid,self.workspace.bookmarks))
        self._autosave_error=None
        self.state_label.setText("正在后台保存…（可继续操作）")
        def work():
            model.fit_geometry_jobs(snapshot[3],geometry_jobs)
            session.write_snapshot(snapshot)
            for name,data in zip(("reviewed_grid.json","bookmarks.json"),workspace):
                atomic_json(session.folder/name,{"source":session.fingerprint,"data":data})
        def done(_,error):
            self._autosave_task=None
            if error:
                self._autosave_error=str(error[0])
                self.state_label.setText("自动保存失败 · 修改尚未保存")
                logging.error(error[1])
                self.view.status_manager.set_message("自动保存失败，修改仍保留；请按 Ctrl+S 重试："+str(error[0]))
                self.autosave_timer.start()
                return
            current_geometry=dict(model.session_metadata.get('plane_geometry',{}))
            for key,_,previous in geometry_jobs:
                if current_geometry.get(key)==previous:
                    updated=snapshot[3].get('plane_geometry',{}).get(key)
                    if updated is None: current_geometry.pop(key,None)
                    else: current_geometry[key]=updated
            if geometry_jobs: model.session_metadata['plane_geometry']=current_geometry
            del session.pending[:snapshot[4]]
            model.saved_revision=revision
            if self.model is model:
                self.refresh()
                self.view.status_manager.set_message("已自动保存；仍有新修改待保存" if model.dirty else "工作标注已自动保存")
                if model.dirty: self.autosave_timer.start()
        task=SceneTask(work)
        task.signals.finished.connect(done)
        self._autosave_task=task
        QtCore.QThreadPool.globalInstance().start(task)

    def export_final(self):
        if self.model is None or self.preview_pending(): return
        path,_=QtWidgets.QFileDialog.getSaveFileName(self.view,"导出最终 LAS/LAZ",str(self.model.output_path),"LAS (*.las);;LAZ (*.laz)")
        if not path: return
        if not self.save(force=True): return
        self.restore_hidden()
        self.start_task("正在导出全部原始点…",lambda:self.model.export(path,self.cancel_task),
                        lambda result:self.view.status_manager.set_message(f"最终点云已导出：{result}"))

    def set_work_roi(self,selection=False):
        if self.model is None or self.preview_pending(): return
        points=self.cloud.points; widget=self.view.gl_widget
        ids=np.flatnonzero(self.model.selection) if selection else None
        if selection and not len(ids):
            self.view.status_manager.set_message("请先框选工作区内的点"); return
        mv=np.array(widget.modelview); pr=np.array(widget.projection); w,h=widget.width(),widget.height()
        def work():
            chosen=ids
            if chosen is None:
                self.projection_cache.project(points,mv,pr,w,h,self.cancel_task)
                chosen=np.flatnonzero(self.projection_cache.valid)
            if not len(chosen): raise ValueError("视域内没有点")
            bounds=[points[chosen].min(axis=0).astype(float).tolist(),points[chosen].max(axis=0).astype(float).tolist()]
            # XY 工作区保留整列高度，避免从俯视图遗漏下方点。
            bounds[0][2]=self.workspace.bounds[0][2]; bounds[1][2]=self.workspace.bounds[1][2]
            return bounds
        self.start_task("正在设置工作区…",work,self.apply_roi)

    def apply_roi(self,bounds):
        self._saving=True
        try:
            run_scene_task(self.view,"正在更新工作区索引…",lambda:self.workspace.set_roi(bounds))
        finally:
            self._saving=False
        self.model.selection[:]=False
        self.model.touch(); self.refresh()

    def clear_work_roi(self):
        if self.model is None or self.preview_pending(): return
        self.apply_roi(None)

    def draw_roi_boundary(self):
        if not self.workspace or self.workspace.roi is None or not self.panel.roi_boundary.isChecked(): return
        widget=self.view.gl_widget
        if widget.modelview is None: return
        low,high=np.asarray(self.workspace.roi)
        z=(low[2]+high[2])/2
        vertices=np.asarray([[low[0],low[1],z],[high[0],low[1],z],[high[0],high[1],z],[low[0],high[1],z]])
        xy,_=project_points(vertices,widget.modelview,widget.projection,widget.width(),widget.height())
        widget.draw_plane_selection(xy)

    def make_review_grid(self):
        if self.model is None or self.preview_pending(): return
        try: scale=coordinate_scale(self.model.header,self.panel.coordinate_units.currentData())*self.cloud.applied_scale
        except Exception as exc:
            self.view.status_manager.set_message(str(exc)); return
        size=self.panel.grid_size.value()
        def work():
            import copy
            workspace=copy.copy(self.workspace)
            workspace.create_grid(size,scale)
            return workspace.grid
        def done(grid):
            self.workspace.grid=grid
            self.workspace_changed()
        self.start_task("正在建立检查网格…",work,done)

    def workspace_changed(self):
        self.model.touch(); self.refresh()
        grid=self.workspace.grid
        if grid: self.panel.grid_status.setText(f"已检查 {len(grid['reviewed'])} / {len(grid['cells'])} 区域")

    def mark_reviewed(self):
        if self.model is None or self.preview_pending(): return
        try:
            count=self.workspace.mark_reviewed(); self.workspace_changed()
            self.view.status_manager.set_message(f"新增 {count} 个已检查网格；仅标记完整落在工作区内的格子")
        except ValueError as exc: self.view.status_manager.set_message(str(exc))

    def next_unreviewed(self):
        if self.model is None or self.preview_pending(): return
        try:
            bounds=self.workspace.next_unreviewed()
            if bounds is None:
                self.view.status_manager.set_message("所有网格已检查"); return
            self.apply_roi(bounds)
            low,high=np.asarray(bounds); center=(low+high)/2
            self.cloud.orbit_pivot=center
            self.cloud.rot_x=self.cloud.rot_y=self.cloud.rot_z=0.
            self.cloud.trans_x=-center[0]; self.cloud.trans_y=-center[1]
            self.cloud.trans_z=-center[2]-max(high[0]-low[0],high[1]-low[1])*1.5
            self.view.gl_widget.update()
        except ValueError as exc: self.view.status_manager.set_message(str(exc))

    def add_bookmark(self):
        if self.model is None or self.preview_pending(): return
        self.workspace.bookmarks.append(dict(camera=self.capture_camera(),plane_id=self.model.current,roi=self.workspace.roi))
        self.workspace.bookmark_index=len(self.workspace.bookmarks)-1
        self.workspace_changed()
        self.view.status_manager.set_message(f"已添加书签 {len(self.workspace.bookmarks)}")

    def step_bookmark(self,direction):
        if self.model is None or self.preview_pending() or not self.workspace.bookmarks: return
        workspace=self.workspace
        workspace.bookmark_index=(workspace.bookmark_index+direction)%len(workspace.bookmarks)
        entry=workspace.bookmarks[workspace.bookmark_index]
        self.apply_camera(entry['camera']); self.apply_roi(entry.get('roi'))
        if entry['plane_id'] in self.model.plane_ids: self.model.current=entry['plane_id']
        self.refresh()
        self.view.status_manager.set_message(f"书签 {workspace.bookmark_index+1} / {len(workspace.bookmarks)}")

    def delete_bookmark(self):
        if self.model is None or self.preview_pending() or not self.workspace.bookmarks: return
        self.workspace.bookmarks.pop(self.workspace.bookmark_index)
        self.workspace.bookmark_index=min(self.workspace.bookmark_index,len(self.workspace.bookmarks)-1)
        self.workspace_changed()
