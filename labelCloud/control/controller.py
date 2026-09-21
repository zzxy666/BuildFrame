import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QPoint
from PyQt5.QtCore import Qt as Keys

from .pcd_manager import PointCloudManger
from .annotation_controller import RoofAnnotationController
from .config_manager import config
from .filter_controller import PointCloudFilterController
from .navigation_controller import NavigationController
from .roof_plane_controller import RoofPlaneController

if TYPE_CHECKING:
    from ..view.gui import GUI


class Controller:
    MOVEMENT_THRESHOLD = 0.1

    def __init__(self) -> None:
        """Initializes all controllers and managers."""
        self.view: "GUI"
        self.pcd_manager = PointCloudManger()
        #self.bbox_controller = BoundingBoxController()

        # Drawing states
        #self.drawing_mode = DrawingManager(self.bbox_controller)
        #self.align_mode = AlignMode(self.pcd_manager)
        self.annotation_controller = RoofAnnotationController(self)
        self.roof_drawing_manager = self.annotation_controller.manager
        self.navigation_controller = NavigationController(self.pcd_manager)


        # Control states
        self.curr_cursor_pos: Optional[QPoint] = None  # updated by mouse movement
        self.last_cursor_pos: Optional[QPoint] = None  # updated by mouse click
        self.ctrl_pressed = False
        self.scroll_mode = False  # to enable the side-pulling

        self.mouse_press_pos = None 
        self.roof_drawing_active = False #屋顶绘制模式的开关
        self.roof_drawing_mode = None  # 可取 'point', 'line', 或 None

        self.filter_controller = PointCloudFilterController(self)
        self.roof_plane_controller = RoofPlaneController(self)



        # Correction states
        self.side_mode = False
        self.selected_side: Optional[str] = None

    def startup(self, view: "GUI") -> None:
        """Sets the view in all controllers and dependent modules; Loads labels from file."""
        self.view = view
        self.annotation_controller.set_view(self.view)
        self.filter_controller.set_view(self.view)
        self.navigation_controller.set_view(self.view)
        self.roof_plane_controller.set_view(self.view)
        self.view.gl_widget.roof_drawing_manager = self.roof_drawing_manager
        #self.bbox_controller.set_view(self.view)
        self.pcd_manager.set_view(self.view)
        #self.drawing_mode.set_view(self.view)
        #self.align_mode.set_view(self.view)
        #self.view.gl_widget.set_bbox_controller(self.bbox_controller)
        #self.bbox_controller.pcd_manager = self.pcd_manager

        # Read labels from folders
        self.pcd_manager.read_pointcloud_folder()
        self.next_pcd(save=False)

    def loop_gui(self) -> None:
        """Function collection called during each event loop iteration."""
        if self.roof_plane_controller.active:
            return  # Roof Plane 按交互事件重绘，空闲时不反复绘制千万点。
        if self.pcd_manager.pointcloud is not None:
            self.set_crosshair()
        #self.set_selected_side()
        self.view.gl_widget.updateGL()

    # POINT CLOUD METHODS
    def next_pcd(self, save: bool = True) -> None:
        if save and not self.save():
            return
        if self.pcd_manager.pcds_left():
            #previous_bboxes = self.bbox_controller.bboxes
            self.pcd_manager.get_next_pcd()
            self.filter_controller.reset_for_pointcloud()
            self.reset()
            self.reset_roof_drawing()
            self.load_roof_annotation()
            self.roof_plane_controller.on_pointcloud_changed()
            self.view.button_point_cloud_filtering.setEnabled(not self.roof_plane_controller.active)
            # self.bbox_controller.set_bboxes(self.pcd_manager.get_labels_from_file())

            # if not self.bbox_controller.bboxes and config.getboolean(
            #     "LABEL", "propagate_labels"
            # ):
            #     self.bbox_controller.set_bboxes(previous_bboxes)
            # self.bbox_controller.set_active_bbox(0)
        else:
            self.view.update_progress(len(self.pcd_manager.pcds))
            self.view.button_next_pcd.setEnabled(False)

    def prev_pcd(self) -> None:
        if not self.save():
            return
        if self.pcd_manager.current_id > 0:
            self.pcd_manager.get_prev_pcd()
            self.filter_controller.reset_for_pointcloud()
            self.reset()
            self.reset_roof_drawing()
            self.load_roof_annotation()
            self.roof_plane_controller.on_pointcloud_changed()
            self.view.button_point_cloud_filtering.setEnabled(not self.roof_plane_controller.active)
            #self.bbox_controller.set_bboxes(self.pcd_manager.get_labels_from_file())
            #self.bbox_controller.set_active_bbox(0)

    def custom_pcd(self, custom: int) -> None:
        if not self.save():
            return
        self.pcd_manager.get_custom_pcd(custom)
        self.filter_controller.reset_for_pointcloud()
        self.reset()
        self.reset_roof_drawing()
        self.load_roof_annotation()
        self.roof_plane_controller.on_pointcloud_changed()
        self.view.button_point_cloud_filtering.setEnabled(not self.roof_plane_controller.active)
        #self.bbox_controller.set_bboxes(self.pcd_manager.get_labels_from_file())

    # CONTROL METHODS
    def save(self, force_plane=False) -> bool:
        """两种标注分别保存；失败时禁止导航继续丢弃当前状态。"""
        return self.annotation_controller.save() and self.roof_plane_controller.save(force=force_plane)

    def reset(self) -> None:
        """Resets the controllers and bounding boxes from the current screen."""
        #self.bbox_controller.reset()
        #self.drawing_mode.reset()
        #self.align_mode.reset()

    # CORRECTION METHODS
    def set_crosshair(self) -> None:
        """Sets the crosshair position in the glWidget to the current cursor position."""
        if self.curr_cursor_pos:
            #self.view.gl_widget.crosshair_col = Colors.GREEN.value
            self.view.gl_widget.crosshair_pos = (
                self.curr_cursor_pos.x(),
                self.curr_cursor_pos.y(),
            )

    # def set_selected_side(self) -> None:
    #     """Sets the currently hovered bounding box side in the glWidget."""
    #     if (
    #         (not self.side_mode)
    #         and self.curr_cursor_pos
    #         #and self.bbox_controller.has_active_bbox()
    #         and (not self.scroll_mode)
    #     ):
    #         _, self.selected_side = oglhelper.get_intersected_sides(
    #             self.curr_cursor_pos.x(),
    #             self.curr_cursor_pos.y(),
    #             #self.bbox_controller.get_active_bbox(),  # type: ignore
    #             self.view.gl_widget.modelview,
    #             self.view.gl_widget.projection,
    #         )
    #     if (
    #         self.selected_side
    #         and (not self.ctrl_pressed)
    #         #and self.bbox_controller.has_active_bbox()
    #     ):
    #         self.view.gl_widget.crosshair_col = Colors.RED.value
    #         # side_vertices = self.bbox_controller.get_active_bbox().get_vertices()  # type: ignore
    #         # self.view.gl_widget.selected_side_vertices = side_vertices[
    #         #     BBOX_SIDES[self.selected_side]
    #         # ]
    #         self.view.status_manager.set_message(
    #             "Scroll to change the bounding box dimension.",
    #             context=Context.SIDE_HOVERED,
    #         )
    #     else:
    #         self.view.gl_widget.selected_side_vertices = np.array([])
    #         self.view.status_manager.clear_message(Context.SIDE_HOVERED)

    # EVENT PROCESSING
    def mouse_clicked(self, a0: QtGui.QMouseEvent) -> None:
        """Triggers actions when the user clicks the mouse."""
        if self.roof_plane_controller.active:
            return
        print("Mouse clicked at:", a0.pos())
        self.last_cursor_pos = a0.pos()


        if a0.button() != Keys.LeftButton:
            if self.roof_drawing_mode and (a0.button() & Keys.RightButton):
                self.roof_drawing_manager.close_polygon_manually()
                self.view.gl_widget.updateGL()
            return


        world_coords = self.view.gl_widget.get_world_coords(a0.x(), a0.y(), correction=True)
        if world_coords is None:
            return

        x, y, z = world_coords
        mgr = self.roof_drawing_manager

        if mgr.connect_mode:
            if mgr.try_handle_left_click((x, y, z)):
                # 已处理连接逻辑，直接返回，不执行下面的添加点行为
                self.update_point_list()  # 连接成功后刷新右侧列表
                self.view.gl_widget.updateGL()
                return
        if self.roof_drawing_mode in ("point", "line"):
            mgr.register_point(x, y, z)
            self.update_point_list()
            self.view.gl_widget.updateGL()
            return

        # 如果都不是屋顶模式，这里可以保留原来的视角旋转等逻辑（如果有的话）
        # 当前代码中似乎没有，继续向下执行其他操作即可


        # if (
        #     self.drawing_mode.is_active()
        #     and (a0.buttons() & Keys.LeftButton)
        #     and (not self.ctrl_pressed)
        # ):
        #     self.drawing_mode.register_point(a0.x(), a0.y(), correction=True)

        # elif self.align_mode.is_active and (not self.ctrl_pressed):
        #     self.align_mode.register_point(
        #         self.view.gl_widget.get_world_coords(a0.x(), a0.y(), correction=False)
        #     )

        # elif self.selected_side:
        #     self.side_mode = True

    # 鼠标按下事件
    def mouse_pressed(self, event: QtGui.QMouseEvent):
        self.mouse_press_pos = event.pos()
        self.last_cursor_pos = event.pos()
        if (
            event.button() == Keys.LeftButton
            and self.roof_drawing_mode is None
            and not self.roof_drawing_manager.connect_mode
            and config.getboolean(
                "USER_INTERFACE", "auto_pick_rotation_center", fallback=True
            )
        ):
            pivot = self.view.gl_widget.pick_surface_point(event.x(), event.y())
            self.navigation_controller.set_orbit_pivot(pivot)
    # 鼠标释放事件
    def mouse_released(self, event: QtGui.QMouseEvent):
        if self.mouse_press_pos is None:
            return

        dx = abs(event.x() - self.mouse_press_pos.x())
        dy = abs(event.y() - self.mouse_press_pos.y())

        if dx < 5 and dy < 5:  # 鼠标移动小于5像素，认为是点击
            self.mouse_clicked(event)
        else:
            print("Mouse drag detected — ignore click.")

    def mouse_double_clicked(self, event: QtGui.QMouseEvent) -> None:
        """Explicitly set the orbit pivot, following CloudCompare-style UX."""
        if event.button() != Keys.LeftButton or self.roof_drawing_mode is not None:
            return
        pivot = self.view.gl_widget.pick_surface_point(event.x(), event.y())
        self.navigation_controller.set_orbit_pivot(pivot)

    def mouse_move_event(self, a0: QtGui.QMouseEvent) -> None:
        """Triggers actions when the user moves the mouse."""

        if self.roof_drawing_mode:
            world_pos = self.view.gl_widget.get_world_coords(a0.x(), a0.y(), correction=True)
            self.roof_drawing_manager.update_preview(world_pos)
        
        if not self.roof_plane_controller.active:
            self.view.gl_widget.updateGL()

        self.curr_cursor_pos = a0.pos()  # Updates the current mouse cursor position

        # Methods that use absolute cursor position
        # if self.drawing_mode.is_active() and (not self.ctrl_pressed):
        #     self.drawing_mode.register_point(
        #         a0.x(), a0.y(), correction=True, is_temporary=True
        #     )

        # elif self.align_mode.is_active and (not self.ctrl_pressed):
        #     self.align_mode.register_tmp_point(
        #         self.view.gl_widget.get_world_coords(a0.x(), a0.y(), correction=False)
        #     )

        if self.last_cursor_pos:
            dx = (
                self.last_cursor_pos.x() - a0.x()
            ) / 5  # Calculate relative movement from last click position
            dy = (self.last_cursor_pos.y() - a0.y()) / 5

            if (
                self.ctrl_pressed
                #and (not self.drawing_mode.is_active())
                #and (not self.align_mode.is_active)
            ):
                pass
                # if a0.buttons() & Keys.LeftButton:  # bbox rotation
                #     self.bbox_controller.rotate_with_mouse(-dx, -dy)
                # elif a0.buttons() & Keys.RightButton:  # bbox translation
                #     new_center = self.view.gl_widget.get_world_coords(
                #         a0.x(), a0.y(), correction=True
                #     )
                #     self.bbox_controller.set_center(*new_center)  # absolute positioning
            else:
                if a0.buttons() & Keys.LeftButton:  # pcd rotation
                    self.navigation_controller.rotate(dx, dy)
                elif a0.buttons() & Keys.RightButton:  # pcd translation
                    self.navigation_controller.translate(dx, dy)

            # Reset scroll locks of "side scrolling" for significant cursor movements
            if dx > Controller.MOVEMENT_THRESHOLD or dy > Controller.MOVEMENT_THRESHOLD:
                if self.side_mode:
                    self.side_mode = False
                else:
                    self.scroll_mode = False
        self.last_cursor_pos = a0.pos()
        if self.roof_plane_controller.active:
            self.view.gl_widget.update()  # 相机变换完成后按事件绘制，不依赖空闲计时器。

        

    def mouse_scroll_event(self, a0: QtGui.QWheelEvent) -> None:
        """Triggers actions when the user scrolls the mouse wheel."""
        if self.selected_side:
            self.side_mode = True

        # if (
        #     self.drawing_mode.is_active()
        #     and (not self.ctrl_pressed)
        #     and self.drawing_mode.drawing_strategy is not None
        # ):
        #     self.drawing_mode.drawing_strategy.register_scrolling(a0.angleDelta().y())
        # elif self.side_mode and self.bbox_controller.has_active_bbox():
        #     self.bbox_controller.get_active_bbox().change_side(  # type: ignore
        #         self.selected_side, -a0.angleDelta().y() / 4000  # type: ignore
        #     )  # ToDo implement method
        else:
            focus_point = None
            if config.getboolean(
                "USER_INTERFACE", "zoom_to_cursor", fallback=True
            ):
                focus_point = self.view.gl_widget.pick_surface_point(
                    a0.position().x(), a0.position().y()
                )
            self.navigation_controller.zoom(a0.angleDelta().y(), focus_point)
            self.scroll_mode = True
            if self.roof_plane_controller.active:
                self.view.gl_widget.update()

    def key_press_event(self, a0: QtGui.QKeyEvent) -> None:
        """Triggers actions when the user presses a key."""
        if a0.key() == Keys.Key_Escape:
            if self.roof_drawing_mode:
                self.roof_drawing_manager.cancel_current_polygon()
                self.view.gl_widget.updateGL()
                return
        # Reset position to intial value
        # if a0.key() == Keys.Key_Control:
        #     self.ctrl_pressed = True
        #     self.view.status_manager.set_message(
        #         "Hold right mouse button to translate or left mouse button to rotate "
        #         "the bounding box.",
        #         context=Context.CONTROL_PRESSED,
        #     )
        # Reset point cloud pose to intial rotation and translation
        elif a0.key() in [Keys.Key_P, Keys.Key_Home]:
            self.pcd_manager.reset_transformations()
            self.navigation_controller.reset_orbit_pivot()
            logging.info("Reseted position to default.")

        # elif a0.key() == Keys.Key_Delete:  # Delete active bbox
        #     self.bbox_controller.delete_current_bbox()

        # Save labels to file
        elif a0.key() == Keys.Key_S and self.ctrl_pressed:
            self.save()

        # elif a0.key() == Keys.Key_Escape:
        #     if self.drawing_mode.is_active():
        #         self.drawing_mode.reset()
        #         logging.info("Resetted drawn points!")
            # elif self.align_mode.is_active:
            #     self.align_mode.reset()
            #     logging.info("Resetted selected points!")

        # BBOX MANIPULATION
        # elif a0.key() == Keys.Key_Z:
        #     # z rotate counterclockwise
        #     self.bbox_controller.rotate_around_z()
        # elif a0.key() == Keys.Key_X:
        #     # z rotate clockwise
        #     self.bbox_controller.rotate_around_z(clockwise=True)
        # elif a0.key() == Keys.Key_C:
        #     # y rotate counterclockwise
        #     self.bbox_controller.rotate_around_y()
        # elif a0.key() == Keys.Key_V:
        #     # y rotate clockwise
        #     self.bbox_controller.rotate_around_y(clockwise=True)
        # elif a0.key() == Keys.Key_B:
        #     # x rotate counterclockwise
        #     self.bbox_controller.rotate_around_x()
        # elif a0.key() == Keys.Key_N:
        #     # x rotate clockwise
        #     self.bbox_controller.rotate_around_x(clockwise=True)
        # elif a0.key() == Keys.Key_W:
        #     # move backward
        #     self.bbox_controller.translate_along_y()
        # elif a0.key() == Keys.Key_S:
        #     # move forward
        #     self.bbox_controller.translate_along_y(forward=True)
        # elif a0.key() == Keys.Key_A:
        #     # move left
        #     self.bbox_controller.translate_along_x(left=True)
        # elif a0.key() == Keys.Key_D:
        #     # move right
        #     self.bbox_controller.translate_along_x()
        # elif a0.key() == Keys.Key_Q:
        #     # move up
        #     self.bbox_controller.translate_along_z()
        # elif a0.key() == Keys.Key_E:
        #     # move down
        #     self.bbox_controller.translate_along_z(down=True)

        # BBOX Scaling
        # elif a0.key() == Keys.Key_I:
        #     # increase length
        #     self.bbox_controller.scale_along_length()
        # elif a0.key() == Keys.Key_O:
        #     # decrease length
        #     self.bbox_controller.scale_along_length(decrease=True)
        # elif a0.key() == Keys.Key_K:
        #     # increase width
        #     self.bbox_controller.scale_along_width()
        # elif a0.key() == Keys.Key_L:
        #     # decrease width
        #     self.bbox_controller.scale_along_width(decrease=True)
        # elif a0.key() == Keys.Key_Comma:
        #     # increase height
        #     self.bbox_controller.scale_along_height()
        # elif a0.key() == Keys.Key_Period:
        #     # decrease height
        #     self.bbox_controller.scale_along_height(decrease=True)

        elif a0.key() in [Keys.Key_R, Keys.Key_Left]:
            # load previous sample
            self.prev_pcd()
        elif a0.key() in [Keys.Key_F, Keys.Key_Right]:
            # load next sample
            self.next_pcd()
        # elif a0.key() in [Keys.Key_T, Keys.Key_Up]:
        #     # select previous bbox
        #     self.select_relative_bbox(-1)
        # elif a0.key() in [Keys.Key_G, Keys.Key_Down]:
        #     # select previous bbox
        #     self.select_relative_bbox(1)
        elif a0.key() == Keys.Key_Y:
            # change bbox class to previous available class
            self.select_relative_class(-1)
        elif a0.key() == Keys.Key_H:
            # change bbox class to next available class
            self.select_relative_class(1)
        # elif a0.key() in list(range(49, 58)):
        #     # select bboxes with 1-9 digit keys
        #     self.bbox_controller.set_active_bbox(int(a0.key()) - 49)

    def select_relative_class(self, step: int):
        if step == 0:
            return
        # curr_class = self.bbox_controller.get_active_bbox().get_classname()  # type: ignore
        # new_class = LabelConfig().get_relative_class(curr_class, step)
        # self.bbox_controller.get_active_bbox().set_classname(new_class)  # type: ignore
        # self.bbox_controller.update_all()  # updates UI in SelectBox

    # def select_relative_bbox(self, step: int):
    #     if step == 0:
    #         return
    #     #max_id = len(self.bbox_controller.bboxes) - 1
    #     curr_id = self.bbox_controller.active_bbox_id
    #     new_id = curr_id + step
    #     #corner_case_id = 0 if step > 0 else max_id
    #     #new_id = new_id if new_id in range(max_id + 1) else corner_case_id
    #     self.bbox_controller.set_active_bbox(new_id)

    # def key_release_event(self, a0: QtGui.QKeyEvent) -> None:
    #     """Triggers actions when the user releases a key."""
    #     if a0.key() == Keys.Key_Control:
    #         self.ctrl_pressed = False
    #         self.view.status_manager.clear_message(Context.CONTROL_PRESSED)

    def crop_pointcloud_inside_active_bbox(self) -> None:
        #bbox = self.bbox_controller.get_active_bbox()
        #assert bbox is not None
        assert self.pcd_manager.pointcloud is not None
        #points_inside = bbox.is_inside(self.pcd_manager.pointcloud.points)
        #pointcloud = self.pcd_manager.pointcloud.get_filtered_pointcloud(points_inside)
        # if pointcloud is None:
        #     logging.warning("No points found inside the box. Ignored.")
        #     return
        # self.view.save_point_cloud_as(pointcloud)

    def activate_roof_point_mode(self):
        """切换屋顶绘制模式的开关"""
        self.roof_drawing_active = not self.roof_drawing_active
        state = "ON" if self.roof_drawing_active else "OFF"
        print(f"[RoofDrawing] Mode switched {state}")

        # 更新按钮外观
        if hasattr(self.view, "button_add_vertices"):
            self.view.button_add_vertices.setStyleSheet(
                "background-color: lightgreen;" if self.roof_drawing_active else ""
            )


    def activate_roof_line_mode(self):
        """激活屋顶线标注模式"""
        self.roof_drawing_manager.set_mode("line") 
        #self.deactivate_other_modes()

    def is_roof_drawing_active(self):
        """检查是否处于屋顶标注模式"""
        return hasattr(self, 'roof_drawing_manager') and self.roof_drawing_manager.mode in ["point", "line"]
    
    def activate_roof_drawing_mode(self, mode: str):
        """用户点击按钮时调用"""
        if self.roof_drawing_mode == mode:
            # 再次点击同一按钮 → 关闭
            self.roof_drawing_mode = None
            self.roof_drawing_manager.reset_temp_state()
            print(f"[RoofDrawing] {mode} mode OFF")
        else:
            self.roof_drawing_mode = mode
            self.roof_drawing_manager.set_mode(mode)
            print(f"[RoofDrawing] {mode} mode ON")

    
    def on_vertex_item_clicked(self, item):
        """右侧列表点击点（Vertex）时：高亮为红色 + 显示坐标"""
        mgr = self.roof_drawing_manager

        mgr.active_line_index = None
        mgr.active_line_start_idx = None
        mgr.active_line_end_idx = None

        # 重置所有点为普通状态（避免多个红色）
        mgr.active_vertex_index = None

        # 找到被点击的点索引
        text = item.text()
        if not text.startswith("Vertex"):
            return

        try:
            # 从 "Vertex3" 提取数字（从1开始）
            index = int(text.replace("Vertex", "")) - 1
            if 0 <= index < len(mgr.vertices):
                mgr.active_vertex_index = index

                # 计算并显示真实世界坐标（考虑点云的缩放和平移）
                x, y, z = self.pcd_manager.pointcloud.to_world_coordinates(
                    mgr.vertices[index]
                )

                self.view.show_vertex_coordinates((x, y, z))

                print(f"选中点 {text}: ({x:.3f}, {y:.3f}, {z:.3f})")
        except:
            pass

        # 强制刷新 OpenGL 画面，让选中点变红
        if self.view and self.view.gl_widget:
            self.view.gl_widget.updateGL()


    def on_line_item_clicked(self, item):
        """右侧列表点击线（Edge）时：高亮为红色（如果你想支持线高亮）"""
        mgr = self.roof_drawing_manager

        # 先取消点的选中
        mgr.active_vertex_index = None

        # 重置线的选中
        mgr.active_line_index = None

        text = item.text()
        if not text.startswith("Edge"):
            return

        try:
            index = int(text.replace("Edge", "")) - 1
            if 0 <= index < len(mgr.lines):
                mgr.active_line_index = index

                line_start_pt = mgr.lines[index]["coord"][0]   # 起点坐标元组
                line_end_pt = mgr.lines[index]["coord"][1]     # 终点坐标元组

                mgr.active_line_start_idx = None
                mgr.active_line_end_idx = None

                for i, pt in enumerate(mgr.vertices):
                    # 使用 np.isclose 容差比较（避免浮点误差）
                    if np.all(np.isclose(pt, line_start_pt, atol=1e-5)):
                        mgr.active_line_start_idx = i
                    if np.all(np.isclose(pt, line_end_pt, atol=1e-5)):
                        mgr.active_line_end_idx = i

                # 显示线的两个端点坐标
                (x1, y1, z1), (x2, y2, z2) = mgr.lines[index]["coord"]

                wx1, wy1, wz1 = self.pcd_manager.pointcloud.to_world_coordinates(
                    (x1, y1, z1)
                )
                wx2, wy2, wz2 = self.pcd_manager.pointcloud.to_world_coordinates(
                    (x2, y2, z2)
                )

                self.view.show_line_coordinates(
                    (wx1, wy1, wz1), (wx2, wy2, wz2)
                )

                print(f"选中线 {text}: 从 ({wx1:.3f},{wy1:.3f},{wz1:.3f}) 到 ({wx2:.3f},{wy2:.3f},{wz2:.3f})")
        except:
            pass

        # 刷新画面（如果你在 draw_roof_annotations 中也支持线高亮）
        if self.view and self.view.gl_widget:
            self.view.gl_widget.updateGL()

    def on_label_item_clicked(self, item):
        """统一处理列表点击：区分点、线、闭合提示"""
        text = item.text()

        if text.startswith("Vertex"):
            self.on_vertex_item_clicked(item)
        elif text.startswith("Edge"):
            self.on_line_item_clicked(item)

    def update_point_list(self):
        self.view.label_list.blockSignals(True)
        self.view.label_list.clear()

        mgr = self.roof_drawing_manager

        # --- 先显示所有已确认的顶点（无论点模式还是线模式闭合后） ---
        for i, coord in enumerate(mgr.vertices):
            name = f"Vertex{i+1}"
            item = QtWidgets.QListWidgetItem(name)
            # 可选：存储坐标到item data，便于后续点击选中
            item.setData(QtCore.Qt.UserRole, coord)
            self.view.label_list.addItem(item)

        # --- 再显示所有已确认的线段 ---
        for i, line in enumerate(mgr.lines):
            name = line.get("name", f"Edge{i+1}")
            item = QtWidgets.QListWidgetItem(name)
            # 可选：存储线信息
            item.setData(QtCore.Qt.UserRole, line)
            self.view.label_list.addItem(item)

        # # --- 如果已闭合，显示提示 ---
        # if mgr.closed:
        #     item = QtWidgets.QListWidgetItem("✓ Polygon Closed")
        #     item.setForeground(QtGui.QColor("green"))
        #     self.view.label_list.addItem(item)

        self.view.label_list.blockSignals(False)
        try:
            self.view.label_list.itemClicked.disconnect(self.select_vertex_or_line)
        except:
            pass
        self.view.label_list.itemClicked.connect(self.select_vertex_or_line)

    def delete_selected_label(self):
        """删除右侧列表中选中的点或线"""
        mgr = self.roof_drawing_manager

        # 获取当前选中项（如果没有选中，返回）
        selected_items = self.view.label_list.selectedItems()
        if not selected_items:
            print("[RoofDrawing] 请先在右侧列表中选中一个点或线")
            return

        item = selected_items[0]
        text = item.text()

        try:
            index = int(text.replace("Vertex", "").replace("Edge", "")) - 1
        except ValueError:
            return

        deleted = False
        if text.startswith("Vertex"):
            removed_edges = mgr.delete_vertex(index)
            deleted = removed_edges is not None
            if deleted:
                print(
                    f"[RoofDrawing] 已删除点 Vertex{index + 1} "
                    f"和相关 {removed_edges} 条线"
                )
        elif text.startswith("Edge"):
            deleted = mgr.delete_edge(index)

        if deleted:
            # 刷新右侧列表和画面
            self.update_point_list()
            if self.view and self.view.gl_widget:
                self.view.gl_widget.updateGL()

            # 自动保存
            self.save()

    def filter_pointcloud(self):
        self.filter_controller.toggle()

    def reset_roof_drawing(self):
        """清空当前屋顶标注（切换点云时调用）"""
        if hasattr(self, 'roof_drawing_manager'):
            self.annotation_controller.reset()

            # 同时清空右侧列表
            self.update_point_list()

            # 强制刷新画面
            if self.view and self.view.gl_widget:
                self.view.gl_widget.update()
    
    def load_roof_annotation(self):
        """加载当前点云对应的屋顶标注文件"""
        if not hasattr(self, 'roof_drawing_manager'):
            return

        if not self.annotation_controller.load():
            self.update_point_list()
            return

        self.update_point_list()
        if self.view and self.view.gl_widget:
            self.view.gl_widget.update()

    def move_active_vertex(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0):
        """移动当前选中的顶点（核心方法）"""
        mgr = self.roof_drawing_manager
        if mgr.active_vertex_index is None:
            print("[RoofDrawing] 请先在右侧列表中选中一个顶点进行微调")
            return

        idx = mgr.active_vertex_index
        if idx >= len(mgr.vertices):
            return

        new_pos_tuple = mgr.move_vertex(idx, (dx, dy, dz))

        print(f"[RoofDrawing] 顶点 {idx+1} 已微调 → {new_pos_tuple}")

        # 刷新界面并自动保存
        self.update_point_list()
        self.view.gl_widget.updateGL()
        self.save()

    # 六个方向的便捷方法（步长可自行调整，推荐 0.02~0.1）
    def vertex_forward(self):
        self.move_active_vertex(dy=0.05)

    def vertex_backward(self):
        self.move_active_vertex(dy=-0.05)

    def vertex_left(self):
        self.move_active_vertex(dx=-0.05)

    def vertex_right(self):
        self.move_active_vertex(dx=0.05)

    def vertex_up(self):
        self.move_active_vertex(dz=0.05)

    def vertex_down(self):
        self.move_active_vertex(dz=-0.05)

    def select_vertex_or_line(self, item):
        """右侧列表点击时选中点或线"""
        mgr = self.roof_drawing_manager
        text = item.text()

        # 先清空旧选中
        mgr.active_vertex_index = None
        mgr.active_line_index = None
        mgr.active_line_start_idx = None
        mgr.active_line_end_idx = None

        if text.startswith("Vertex"):
            try:
                idx = int(text.replace("Vertex", "")) - 1
                if 0 <= idx < len(mgr.vertices):
                    mgr.active_vertex_index = idx
                    print(f"[RoofDrawing] 已选中顶点 Vertex{idx+1}")
            except:
                pass
        elif text.startswith("Edge"):
                try:
                    idx = int(text.replace("Edge", "")) - 1
                    if 0 <= idx < len(mgr.lines):
                        mgr.active_line_index = idx

                        # === 关键修复：设置起点和终点的索引 ===
                        line = mgr.lines[idx]
                        start_pt = line["coord"][0]
                        end_pt = line["coord"][1]

                        # 查找这两个坐标在 vertices 中的索引
                        for i, v in enumerate(mgr.vertices):
                            if np.allclose(v, start_pt, atol=1e-6):
                                mgr.active_line_start_idx = i
                            if np.allclose(v, end_pt, atol=1e-6):
                                mgr.active_line_end_idx = i

                        print(f"已选中线 Edge{idx+1}，起点 Vertex{mgr.active_line_start_idx+1 if mgr.active_line_start_idx is not None else '?'}，"
                            f"终点 Vertex{mgr.active_line_end_idx+1 if mgr.active_line_end_idx is not None else '?'}")
                except Exception as e:
                    print("选中线时出错:", e)

        # 刷新高亮
        self.view.gl_widget.updateGL()

    def set_standard_view(self, view_name: str):
        if self.roof_plane_controller.active and self.pcd_manager.pointcloud is not None:
            # 大场景只切换观察方向，保留当前局部旋转中心和缩放。
            rotation = self.navigation_controller.STANDARD_ROTATIONS[view_name]
            self.pcd_manager.pointcloud.set_rotations(*rotation)
            self.roof_plane_controller.cancel_gesture()
            self.view.gl_widget.update()
            return
        self.navigation_controller.set_standard_view(view_name)
    
    def activate_connect_mode(self, enabled: bool):
        """激活或关闭连接两点模式"""
        self.roof_drawing_manager.toggle_connect_mode(enabled)

        # 开启时自动关闭点/线模式
        if enabled and self.roof_drawing_mode is not None:
            self.activate_roof_drawing_mode(None)
