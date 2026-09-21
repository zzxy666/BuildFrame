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
        self.rgb_colors = None

    def set_view(self, view):
        super().set_view(view)
        self.autosave_timer = QtCore.QTimer(view)
        self.autosave_timer.setSingleShot(True)
        self.autosave_timer.setInterval(2000)
        self.autosave_timer.timeout.connect(self.autosave)
        self.camera_timer = QtCore.QTimer(view)
        self.camera_timer.setSingleShot(True)
        self.camera_timer.setInterval(250)
        self.camera_timer.timeout.connect(self.camera_idle)

    def change_mode(self, index):
        if self.task or self._saving:
            self.mode.blockSignals(True); self.mode.setCurrentIndex(1 if self.active else 0); self.mode.blockSignals(False)
            self.view.status_manager.set_message("后台任务进行中，可按 Esc 取消后切换模式")
            return
        self.color_key = self.hidden_key = self.highlight_key = None
        super().change_mode(index)
        if not self.active and self.cloud is not None:
            self.cloud.set_overlays([]); self.cloud.lod_stride = 1
            self.camera_timer.stop()

    def on_pointcloud_changed(self):
        self.projection_cache = ScreenProjectionCache()
        self.workspace = self.spatial = None
        self.camera_key = self.color_key = self.hidden_key = self.highlight_key = None
        self.selected_ids = np.empty(0,np.int64)
        super().on_pointcloud_changed()

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
        if self.model.dirty and not self._saving:
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
                    self.cloud.set_display_colors(self.rgb_colors)
            elif mode == 0:
                ids=self.model.changed_ids
                self.cloud.update_display_colors(ids,self.model.colors(ids))
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
        if self.candidate is not None: layers.append((self.candidate,(1.,.9,.05)))
        self.cloud.set_overlays(layers)
        if self.panel.debug_perf.isChecked(): logging.info("Overlay update selected=%d ms=%.2f",len(self.selected_ids),(time.perf_counter()-start)*1000)
        self.view.gl_widget.update()

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
        if self.active and self.task:
            if event.type() in (QtCore.QEvent.MouseButtonPress,QtCore.QEvent.MouseButtonRelease):
                if event.button()==QtCore.Qt.LeftButton and self.panel.tool.currentData()!="navigate": return True
            if event.type()==QtCore.QEvent.MouseMove and event.buttons()&QtCore.Qt.LeftButton and self.panel.tool.currentData()!="navigate": return True
        return super().handle_mouse(event)

    def clear_selection(self):
        if self.task:
            self.cancel_task.set(); return
        super().clear_selection()

    def undo(self):
        if self.task:
            self.cancel_task.set(); return
        super().undo()

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
        roi=self.roi_mask(); model=self.model; points=self.cloud.points
        debug=self.panel.debug_perf.isChecked()
        def work():
            self.projection_cache.project(points,mv,pr,w,h,self.cancel_task)
            return self.projection_cache.select(polygon,model.labels,model.hidden_mask,scope,current,roi,self.cancel_task,debug)
        def done(ids):
            if self.model is not model: return
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
            super().new_plane()
        except ValueError as exc:
            self.view.status_manager.set_message(str(exc))

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
                                     model.hidden_revision,self.workspace.roi_revision,self.cancel_task,debug)
        def done(ids):
            if not len(ids):
                self.panel.expansion_info.setText("没有可扩展的未标注点"); return
            self.candidate=ids; self.expansion_target=current; self.highlight=None
            self.panel.set_preview(True)
            self.panel.expansion_info.setText(f"候选 {len(ids)} 点（边缘补选 {self.spatial.last_edge_count}）→ Plane {current}，Enter 确认 / Esc 取消")
            self.refresh()
        self.start_task("正在局部扩展…（首次建立空间索引）",work,done)

    def confirm_expansion(self):
        if self.task or self.model is None or self.candidate is None: return
        ids=self.candidate
        allowed=self.model.selectable("unlabelled",ids)
        roi=self.roi_mask()
        if roi is not None: allowed &= roi[ids]
        self.model.assign(self.expansion_target,"unlabelled",ids[allowed])
        self.cancel_expansion()
        self.view.status_manager.set_message("扩展已确认，可撤销")

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
            self.refresh()
            return True
        except Exception as exc:
            logging.exception("Session save failed")
            self.model.touch()
            QtWidgets.QMessageBox.critical(self.view,"保存失败",str(exc)); return False
        finally: self._saving=False

    def autosave(self):
        if self.model is None or not self.model.dirty: return
        if self.task:
            self.autosave_timer.start(); return
        self.save()

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
