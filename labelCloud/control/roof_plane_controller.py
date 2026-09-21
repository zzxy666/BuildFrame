"""Roof Plane 交互；Wireframe 仍交由原控制器处理。"""
import logging

import numpy as np
from PyQt5 import QtCore, QtWidgets

from ..model.roof_planes import RoofPlanes, polygon_mask, project_points
from ..model.local_plane_growth import grow_plane, metric_coordinates
from ..view.roof_plane_panel import RoofPlanePanel, layout_widgets


class LegacyRoofPlaneController:
    def __init__(self, owner):
        self.owner = owner
        self.active = False
        self.model = None
        self.cloud = None
        self.highlight = None
        self.gesture = []
        self.orbit_position = None
        self.candidate = None
        self.expansion_target = None
        self.view = None

    def set_view(self, view):
        self.view = view
        self.wire_shortcuts = view.findChildren(QtWidgets.QShortcut)
        self.wire_stretches = [view.horizontalLayout.stretch(i) for i in range(view.horizontalLayout.count())]
        self.wire_right = list(layout_widgets(view.verticalLayout_3))
        self.wire_spacers = []
        for i in range(view.verticalLayout_3.count()):
            spacer = view.verticalLayout_3.itemAt(i).spacerItem()
            if spacer is not None:
                self.wire_spacers.append((spacer, spacer.sizeHint(), spacer.sizePolicy()))
        self.wire_buttons = [getattr(view, name) for name in (
            "button_add_vertices", "button_add_lines", "connect_two_vertices",
            "start_pt_control", "button_point_cloud_filtering")]
        self.panel = RoofPlanePanel(self, view)
        self.state_label = QtWidgets.QLabel()
        view.statusBar().addPermanentWidget(self.state_label)
        self.state_label.hide()
        # 优先占满右栏，避免原 Wireframe 底部占位挤压功能区。
        view.verticalLayout_3.insertWidget(0, self.panel, 1)
        self.panel.hide()
        toolbar = view.addToolBar("标注模式")
        toolbar.setMovable(False)
        toolbar.addWidget(QtWidgets.QLabel("标注模式： "))
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Wireframe", "Roof Plane"])
        self.mode.currentIndexChanged.connect(self.change_mode)
        toolbar.addWidget(self.mode)
        view.gl_widget.roof_plane_controller = self
        view.gl_widget.setFocusPolicy(QtCore.Qt.StrongFocus)

    def change_mode(self, index):
        self.cancel_expansion()
        self.active = index == 1
        self.cancel_gesture()
        self.owner.activate_roof_drawing_mode(None)
        self.owner.activate_connect_mode(False)
        for button in (self.view.button_add_vertices, self.view.button_add_lines, self.view.connect_two_vertices):
            button.setChecked(False)
            button.setStyleSheet("")
        for widget in self.wire_right:
            widget.setVisible(not self.active)
        for widget in self.wire_buttons:
            widget.setEnabled(not self.active)
        for shortcut in self.wire_shortcuts:
            shortcut.setEnabled(not self.active)
        # Roof Plane 收起原占位；切回 Wireframe 恢复尺寸和策略。
        for spacer, size, policy in self.wire_spacers:
            if self.active:
                spacer.changeSize(0, 0, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            else:
                spacer.changeSize(size.width(), size.height(), policy.horizontalPolicy(), policy.verticalPolicy())
        self.view.verticalLayout_3.invalidate()
        self.panel.setVisible(self.active)
        self.state_label.setVisible(self.active)
        for i, stretch in enumerate(self.wire_stretches):
            self.view.horizontalLayout.setStretch(i, (1 if i == 1 else 0) if self.active else stretch)
        if self.active:
            self.owner.filter_controller.cancel(announce=False)
            self.view.button_point_cloud_filtering.setEnabled(False)
            self.load_current()
        else:
            self.panel.set_available(False)
            if self.cloud is not None:
                self.cloud.set_display_colors(None)
                self.cloud.set_display_mask(None)
        self.view.status_manager.set_mode("Roof Plane" if self.active else "Wireframe")
        self.view.gl_widget.update()

    def on_pointcloud_changed(self):
        self.cancel_expansion()
        self.model = None
        self.cloud = None
        self.highlight = None
        self.cancel_gesture()
        if self.active:
            self.load_current()

    def load_current(self):
        cloud = self.owner.pcd_manager.pointcloud
        self.panel.set_available(False)
        if cloud is None or cloud.path.suffix.lower() not in (".las", ".laz"):
            self.panel.message.setText("Roof Plane 模式需要 LAS/LAZ 点云。请使用原有文件夹入口加载。")
            return
        try:
            if self.cloud is not cloud or self.model is None:
                model = RoofPlanes(cloud.path)
                if len(model.labels) != len(cloud.points):
                    raise ValueError("点数与原始 LAS 不一致，无法安全对应标签")
                self.model, self.cloud = model, cloud
                self.highlight = None
                for control in (self.panel.selection_scope, self.panel.show_hidden):
                    control.blockSignals(True)
                self.panel.selection_scope.setCurrentIndex(0)
                self.panel.show_hidden.setChecked(False)
                for control in (self.panel.selection_scope, self.panel.show_hidden):
                    control.blockSignals(False)
            self.panel.set_available(True)
            self.panel.message.setText(f"输出：{self.model.output_path.name}\n0 = 非屋顶/未标注；局部扩展需手工确认。")
            self.refresh()
        except Exception as exc:
            logging.exception("Failed to load Roof Plane labels")
            self.model = None
            self.panel.message.setText(f"无法加载标注：{exc}")

    def refresh(self):
        if self.model is not None:
            self.model.selection &= self.selection_allowed()
            self.panel.refresh(self.model)
            self.state_label.setText(
                f"选择模式：{self.panel.selection_scope.currentText()} | "
                f"隐藏 {self.model.hidden_mask.sum()} points | 当前 Plane {self.model.current}")
            self.refresh_colors()

    def selection_allowed(self):
        allowed = self.model.selectable(self.panel.selection_scope.currentData())
        if self.panel.show_hidden.isChecked():
            allowed[:] = False  # 隐藏点检查视图只读，不能绕过隐藏限制。
        return allowed

    def change_selection_scope(self, *_):
        self.cancel_gesture()
        if self.model is not None:
            self.model.selection[:] = False
            self.refresh()

    def change_hidden_view(self, *_):
        self.cancel_gesture()
        if self.model is not None:
            self.model.selection[:] = False
            self.refresh()

    def hide_selected(self):
        self.update_hidden("hide")

    def restore_hidden(self):
        self.update_hidden("restore")

    def isolate_selected(self):
        self.update_hidden("isolate")

    def update_hidden(self, action):
        if not self.active or self.model is None or self.preview_pending():
            return
        chosen = self.model.selection & self.selection_allowed()
        if action != "restore" and not chosen.any():
            self.view.status_manager.set_message("请先选择要隐藏或单独显示的点。")
            return
        mask = self.model.hidden_mask.copy()
        if action == "restore":
            mask[:] = False
        elif action == "hide":
            mask |= chosen
        else:
            mask |= ~chosen
        self.model.set_hidden(mask)
        self.panel.show_hidden.blockSignals(True)
        self.panel.show_hidden.setChecked(False)
        self.panel.show_hidden.blockSignals(False)
        self.cancel_gesture()
        self.refresh()

    def refresh_colors(self, *_):
        if not self.active or self.model is None or self.cloud is None:
            return
        if self.panel.display.currentIndex() == 0:
            colors = self.model.colors()
        else:
            source = self.model.source
            if "red" in list(source.point_format.dimension_names):
                colors = np.column_stack((source.red, source.green, source.blue)).astype(np.float32) / 65535.0
            else:
                colors = np.full((len(self.model.labels), 3), 0.65, dtype=np.float32)
        if self.highlight is not None:
            colors[self.model.labels == self.highlight] = (0.0, 0.95, 1.0)
        colors[self.model.selection] = (1.0, 0.9, 0.05)
        if self.candidate is not None:
            colors[self.candidate] = (1.0, 0.9, 0.05)
        self.cloud.set_display_colors(colors)
        visible = self.model.hidden_mask if self.panel.show_hidden.isChecked() else ~self.model.hidden_mask
        self.cloud.set_display_mask(visible)
        self.view.gl_widget.update()

    def select_plane(self, item):
        if self.preview_pending():
            self.refresh()
            return
        if self.model is not None:
            self.model.current = int(item.data(QtCore.Qt.UserRole))
            self.highlight = self.model.current
            self.refresh()

    def new_plane(self):
        if self.preview_pending():
            return
        if self.model is not None:
            self.model.new_plane()
            self.highlight = None
            self.refresh()

    def assign(self, plane_id=None):
        if self.preview_pending():
            return
        if self.model is not None:
            self.model.selection &= self.selection_allowed()
            if not self.model.assign(self.model.current if plane_id is None else plane_id,
                                     scope=self.panel.selection_scope.currentData()):
                self.view.status_manager.set_message("请先用矩形或套索选择点。")
            self.highlight = None
            self.refresh()

    def delete(self):
        if self.preview_pending():
            return
        if self.panel.selection_scope.currentData() == "unlabelled":
            self.view.status_manager.set_message("删除已标注 Plane 前，请主动切换到当前 Plane 或所有可见点。")
            return
        if self.model is not None:
            self.model.delete(self.model.current)
            self.highlight = None
            self.refresh()

    def merge(self):
        if self.preview_pending():
            return
        if self.panel.selection_scope.currentData() == "unlabelled":
            self.view.status_manager.set_message("合并已标注 Plane 前，请主动切换到当前 Plane 或所有可见点。")
            return
        if self.model is None:
            return
        source = self.model.current
        targets = sorted(self.model.plane_ids - {0, source})
        if source == 0 or not targets:
            self.view.status_manager.set_message("请先选中一个 Roof Plane，并确保另有一个可合并的 Plane。")
            return
        target, ok = QtWidgets.QInputDialog.getItem(self.view, "合并 Plane", f"将 Plane {source} 合并到：", [str(i) for i in targets], 0, False)
        if ok:
            self.model.merge(source, int(target))
            self.highlight = self.model.current
            self.refresh()

    def undo(self):
        if self.candidate is not None:
            self.cancel_expansion()
            return
        if self.model is not None:
            self.model.undo()
            self.panel.show_hidden.blockSignals(True)
            self.panel.show_hidden.setChecked(False)
            self.panel.show_hidden.blockSignals(False)
            self.highlight = None
            self.refresh()

    def clear_selection(self):
        if self.candidate is not None:
            self.cancel_expansion()
            return
        self.cancel_gesture()
        if self.model is not None:
            self.model.selection[:] = False
            self.highlight = None
            self.refresh()

    def cancel_gesture(self, *_):
        self.gesture = []
        self.orbit_position = None
        if self.view is not None:
            self.view.gl_widget.update()

    def set_tool(self, name):
        """快捷切换只取消未完成的选框，保留已选中的点。"""
        if not self.active or self.model is None:
            return
        index = self.panel.tool.findData(name)
        if index >= 0:
            self.cancel_gesture()
            self.panel.tool.setCurrentIndex(index)
            self.view.gl_widget.setFocus()

    def handle_mouse(self, event):
        """返回 True 时阻止选框拖动进入原有视角旋转/顶点拾取逻辑。"""
        if not self.active:
            return False
        kind = event.type()
        # 中键复用原有旋转控制器，不改变当前选择工具和已选点。
        if kind == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.MiddleButton:
            self.cancel_gesture()
            self.orbit_position = event.pos()
            self.view.gl_widget.setFocus()
            return True
        if self.orbit_position is not None:
            if kind == QtCore.QEvent.MouseMove:
                if not event.buttons() & QtCore.Qt.MiddleButton:
                    self.orbit_position = None
                    return True
                dx = (self.orbit_position.x() - event.x()) / 5
                dy = (self.orbit_position.y() - event.y()) / 5
                if self.owner.pcd_manager.pointcloud is not None:
                    self.owner.navigation_controller.rotate(dx, dy)
                self.orbit_position = event.pos()
                self.view.gl_widget.update()
                return True
            if kind == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.MiddleButton:
                self.orbit_position = None
                self.owner.last_cursor_pos = event.pos()
                return True
        if kind == QtCore.QEvent.MouseButtonDblClick:
            return True
        # 预览期间允许浏览，但不能用新的左键选择静默替换候选。
        if self.candidate is not None and kind in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.MouseButtonRelease):
            if event.button() == QtCore.Qt.LeftButton and self.panel.tool.currentData() != "navigate":
                self.preview_pending()
                return True
        if self.candidate is not None and kind == QtCore.QEvent.MouseMove and event.buttons() & QtCore.Qt.LeftButton:
            if self.panel.tool.currentData() != "navigate":
                return True
        selecting = self.model is not None and self.panel.tool.currentData() != "navigate"
        if selecting and kind == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
            self.gesture = [(event.x(), event.y())]
            self.gesture_modifiers = event.modifiers()
            self.view.gl_widget.setFocus()
            return True
        if self.gesture and kind == QtCore.QEvent.MouseMove:
            point = (event.x(), event.y())
            if self.panel.tool.currentData() == "rectangle":
                self.gesture = [self.gesture[0], point]
            elif np.linalg.norm(np.asarray(point) - self.gesture[-1]) >= 2:
                self.gesture.append(point)
            self.view.gl_widget.update()
            return True
        if self.gesture and kind == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
            self.finish_selection((event.x(), event.y()))
            return True
        return False

    def polygon(self):
        if self.panel.tool.currentData() == "rectangle" and len(self.gesture) >= 2:
            (x0, y0), (x1, y1) = self.gesture[0], self.gesture[-1]
            return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        return self.gesture

    def finish_selection(self, point):
        if self.panel.tool.currentData() == "rectangle":
            self.gesture = [self.gesture[0], point]
        else:
            self.gesture.append(point)
        widget = self.view.gl_widget
        polygon = self.polygon()
        if widget.modelview is not None and widget.projection is not None and len(polygon) >= 3:
            screen, visible = project_points(self.cloud.points, widget.modelview, widget.projection, widget.width(), widget.height())
            chosen = polygon_mask(screen, polygon) & visible & self.selection_allowed()
            if self.gesture_modifiers & QtCore.Qt.AltModifier:
                self.model.selection &= ~chosen
            elif self.gesture_modifiers & QtCore.Qt.ShiftModifier:
                self.model.selection |= chosen
            else:
                self.model.selection = chosen
            self.highlight = None
        self.cancel_gesture()
        self.refresh()

    def preview_pending(self):
        if self.candidate is None:
            return False
        self.view.status_manager.set_message("请先按 Enter 确认扩展，或 Esc 取消；候选尚未写入标签。")
        return True

    def expand_local(self):
        if not self.active or self.model is None or self.preview_pending():
            return
        self.cancel_gesture()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.model.selection &= self.selection_allowed()
            xyz, note = metric_coordinates(self.model.source, self.panel.coordinate_units.currentData())
            candidate = grow_plane(xyz, self.model.labels, self.model.selection,
                                   self.model.current, self.panel.growth_options(),
                                   available=~self.model.hidden_mask)
            if not candidate.any():
                raise ValueError("没有可扩展的未标注点。请检查种子、邻接半径和当前 Plane。")
            self.candidate = candidate
            self.expansion_target = self.model.current
            self.highlight = None
            self.panel.set_preview(True)
            self.panel.expansion_info.setText(f"候选 {candidate.sum()} 点 → Plane {self.expansion_target}，尚未修改标签。\n{note}")
            self.refresh()
            self.view.gl_widget.setFocus()
        except Exception as exc:
            logging.exception("Local plane expansion failed")
            self.panel.expansion_info.setText(str(exc))
            self.view.status_manager.set_message(f"局部扩展失败：{exc}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def cancel_expansion(self):
        """取消只清除候选，原种子选择与标签均保留。"""
        self.candidate = None
        self.expansion_target = None
        if self.view is not None and hasattr(self, "panel"):
            self.panel.set_preview(False)
            self.panel.expansion_info.setText("先选当前 Plane，再圈选至少 6 个相邻种子点。")
            self.refresh()

    def confirm_expansion(self):
        if not self.active or self.model is None or self.candidate is None:
            return
        # 再次限制到 0，确认过程中也绝不能覆盖已有标签；复用赋值撤销记录。
        self.model.selection = self.candidate & (self.model.labels == 0) & ~self.model.hidden_mask
        self.model.assign(self.expansion_target)
        self.cancel_expansion()
        self.view.status_manager.set_message("局部扩展已确认，Ctrl+Z 可撤销。")

    def save(self, force=False):
        if self.model is None or not (self.model.dirty or force):
            return True
        try:
            path = self.model.save()
            self.view.status_manager.set_message(f"Roof Plane 已保存：{path}")
            self.refresh()
            return True
        except Exception as exc:
            logging.exception("Failed to save Roof Plane labels")
            QtWidgets.QMessageBox.critical(self.view, "保存失败", f"标注仍保留在内存中，已停止切换/关闭。\n{exc}")
            return False

from .scene_roof_controller import RoofPlaneController
