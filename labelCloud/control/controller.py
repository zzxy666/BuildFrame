import logging
from typing import Optional

import numpy as np
from PyQt5 import QtGui
from PyQt5.QtCore import QPoint
from PyQt5.QtCore import Qt as Keys

#from ..definitions import LabelingMode
#from ..io.labels.config import LabelConfig
from ..utils import oglhelper
from ..view.gui import GUI
#from .alignmode import AlignMode
#from .bbox_controller import BoundingBoxController
from .config_manager import config
#from .drawing_manager import DrawingManager
from .pcd_manager import PointCloudManger
from .roof_drawing_manager import RoofDrawingManager
from PyQt5 import QtGui,QtWidgets,QtCore
from PointCloudFilter.ground_filter import GroundFilter



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
        self.roof_drawing_manager = RoofDrawingManager(self)


        # Control states
        self.curr_cursor_pos: Optional[QPoint] = None  # updated by mouse movement
        self.last_cursor_pos: Optional[QPoint] = None  # updated by mouse click
        self.ctrl_pressed = False
        self.scroll_mode = False  # to enable the side-pulling

        self.mouse_press_pos = None 
        self.roof_drawing_active = False #屋顶绘制模式的开关
        self.roof_drawing_mode = None  # 可取 'point', 'line', 或 None

        self.ground_filter = GroundFilter()
        self.is_filtered = False



        # Correction states
        self.side_mode = False
        self.selected_side: Optional[str] = None

    def startup(self, view: "GUI") -> None:
        """Sets the view in all controllers and dependent modules; Loads labels from file."""
        self.view = view
        self.roof_drawing_manager.set_view(self.view)
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
        self.set_crosshair()
        #self.set_selected_side()
        self.view.gl_widget.updateGL()

    # POINT CLOUD METHODS
    def next_pcd(self, save: bool = True) -> None:
        if save:
            self.save()
        if self.pcd_manager.pcds_left():
            #previous_bboxes = self.bbox_controller.bboxes
            self.pcd_manager.get_next_pcd()
            self.reset()
            self.reset_roof_drawing()
            self.load_roof_annotation()
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
        self.save()
        if self.pcd_manager.current_id > 0:
            self.pcd_manager.get_prev_pcd()
            self.reset()
            self.reset_roof_drawing()
            self.load_roof_annotation()
            #self.bbox_controller.set_bboxes(self.pcd_manager.get_labels_from_file())
            #self.bbox_controller.set_active_bbox(0)

    def custom_pcd(self, custom: int) -> None:
        self.save()
        self.pcd_manager.get_custom_pcd(custom)
        self.reset()
        self.reset_roof_drawing()
        self.load_roof_annotation()
        #self.bbox_controller.set_bboxes(self.pcd_manager.get_labels_from_file())

    # CONTROL METHODS
    def save(self) -> None:
        """Saves all bounding boxes and optionally segmentation labels in the label file."""
        #self.pcd_manager.save_labels_into_file(self.bbox_controller.bboxes)

        # if LabelConfig().type == LabelingMode.SEMANTIC_SEGMENTATION:
        #     assert self.pcd_manager.pointcloud is not None
        #     self.pcd_manager.pointcloud.save_segmentation_labels()
        """只保存屋顶标注（不再保存 Bbox）"""
        if not self.pcd_manager.pcds or self.pcd_manager.current_id < 0:
            return
        if (self.roof_drawing_manager.vertices or 
            self.roof_drawing_manager.lines or 
            self.roof_drawing_manager.temp_points):
            
            pcd_path = self.pcd_manager.pcd_path
            roof_obj_path = pcd_path.with_suffix('.roof.obj')
            roof_obj_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                self.roof_drawing_manager.save_to_obj(str(roof_obj_path))
                print(f"[RoofDrawing] 屋顶标注已自动保存到: {roof_obj_path}")
            except Exception as e:
                print(f"[RoofDrawing] 保存屋顶标注失败: {e}")

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

    # def mouse_double_clicked(self, a0: QtGui.QMouseEvent) -> None:
    #     """Triggers actions when the user double clicks the mouse."""
    #     self.bbox_controller.select_bbox_by_ray(a0.x(), a0.y())

    def mouse_move_event(self, a0: QtGui.QMouseEvent) -> None:
        """Triggers actions when the user moves the mouse."""

        if self.roof_drawing_mode:
            world_pos = self.view.gl_widget.get_world_coords(a0.x(), a0.y(), correction=True)
            self.roof_drawing_manager.update_preview(world_pos)
        
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
                    self.pcd_manager.rotate_around_x(dy)
                    self.pcd_manager.rotate_around_z(dx)
                elif a0.buttons() & Keys.RightButton:  # pcd translation
                    self.pcd_manager.translate_along_x(dx)
                    self.pcd_manager.translate_along_y(dy)

            # Reset scroll locks of "side scrolling" for significant cursor movements
            if dx > Controller.MOVEMENT_THRESHOLD or dy > Controller.MOVEMENT_THRESHOLD:
                if self.side_mode:
                    self.side_mode = False
                else:
                    self.scroll_mode = False
        self.last_cursor_pos = a0.pos()

        

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
            self.pcd_manager.zoom_into(a0.angleDelta().y())
            self.scroll_mode = True

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
                nx, ny, nz = mgr.vertices[index]
                center = self.pcd_manager.pointcloud.last_center
                scale = self.pcd_manager.pointcloud.last_scale

                x = nx * scale + center[0]
                y = ny * scale + center[1]
                z = nz * scale + center[2]

                # 更新上方 Current Vertex 的坐标框
                self.view.current_pt_x.setText(f"{x:.3f}")
                self.view.current_pt_y.setText(f"{y:.3f}")
                self.view.current_pt_z.setText(f"{z:.3f}")

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

                center = self.pcd_manager.pointcloud.last_center
                scale = self.pcd_manager.pointcloud.last_scale

                wx1 = x1 * scale + center[0]
                wy1 = y1 * scale + center[1]
                wz1 = z1 * scale + center[2]
                wx2 = x2 * scale + center[0]
                wy2 = y2 * scale + center[1]
                wz2 = z2 * scale + center[2]

                self.view.start_pt_x.setText(f"{wx1:.3f}")
                self.view.end_pt_x.setText(f"{wx2:.3f}")
                self.view.start_pt_y.setText(f"{wy1:.3f}")
                self.view.end_pt_y.setText(f"{wy2:.3f}")
                self.view.start_pt_z.setText(f"{wz1:.3f}")
                self.view.end_pt_z.setText(f"{wz2:.3f}")

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

        deleted = False

        if text.startswith("Vertex"):
            # --- 删除点 ---
            try:
                index = int(text.replace("Vertex", "")) - 1
                if 0 <= index < len(mgr.vertices):
                    deleted_pt = mgr.vertices.pop(index)
                    mgr.vertex_info.pop(index)

                    # 删除所有包含这个点的线段
                    lines_to_remove = []
                    for i, line in enumerate(mgr.lines):
                        if np.all(np.isclose(line["coord"][0], deleted_pt, atol=1e-5)) or \
                           np.all(np.isclose(line["coord"][1], deleted_pt, atol=1e-5)):
                            lines_to_remove.append(i)

                    # 从后往前删除，避免索引错乱
                    for i in sorted(lines_to_remove, reverse=True):
                        del mgr.lines[i]

                    print(f"[RoofDrawing] 已删除点 Vertex{index+1} 和相关 {len(lines_to_remove)} 条线")
                    deleted = True
            except Exception as e:
                print("删除点失败:", e)

        elif text.startswith("Edge"):
            # --- 删除线 ---
            try:
                index = int(text.replace("Edge", "")) - 1
                if 0 <= index < len(mgr.lines):
                    del mgr.lines[index]
                    print(f"[RoofDrawing] 已删除线 Edge{index+1}")
                    deleted = True
            except Exception as e:
                print("删除线失败:", e)

        if deleted:
            # 重置选中状态
            mgr.active_vertex_index = None
            mgr.active_line_index = None
            mgr.active_line_start_idx = None
            mgr.active_line_end_idx = None

            # 重新编号（可选：让名字从1开始连续）
            for i, info in enumerate(mgr.vertex_info):
                info["name"] = f"Vertex{i+1}"
            for i, line in enumerate(mgr.lines):
                line["name"] = f"Edge{i+1}"

            # 刷新右侧列表和画面
            self.update_point_list()
            if self.view and self.view.gl_widget:
                self.view.gl_widget.updateGL()

            # 自动保存
            self.save()

    def filter_pointcloud(self):
        pc = self.pcd_manager.pointcloud

        # --- 如果已滤波 → 还原 ---
        if self.is_filtered:
            pc.points = pc.backup_points.copy()
            pc.colors = pc.backup_colors.copy()

            del pc.backup_points
            del pc.backup_colors

            self.is_filtered = False
            self.view.button_point_cloud_filtering.setText("Filter")

            pc.create_buffers()
            self.view.gl_widget.update()
            return

        # --- 未滤波 → 开始滤波 ---
        points = pc.points
        if points is None or len(points) == 0:
            logging.error("No points loaded.")
            return

        # 保存原始点云数据
        pc.backup_points = pc.points.copy()
        pc.backup_colors = pc.colors.copy()

        # 调用 GroundFilter
        ground_labels = self.ground_filter.process(points)

        # 设置颜色：绿色=地面 红色=非地面
        colors = np.zeros_like(points)
        colors[ground_labels == 2] = [0.0, 1.0, 0.0]  # 绿色 ground
        colors[ground_labels == 1] = [1.0, 0.0, 0.0]  # 红色 non-ground

        pc.colors = colors

        # 更新 OpenGL buffer
        pc.create_buffers()

        self.is_filtered = True
        self.view.button_point_cloud_filtering.setText("Restore")

        # 刷新
        self.view.gl_widget.update()

    def reset_roof_drawing(self):
        """清空当前屋顶标注（切换点云时调用）"""
        if hasattr(self, 'roof_drawing_manager'):
            self.roof_drawing_manager.vertices.clear()
            self.roof_drawing_manager.vertex_info.clear()
            self.roof_drawing_manager.lines.clear()
            self.roof_drawing_manager.temp_points.clear()
            self.roof_drawing_manager.preview_point = None
            self.roof_drawing_manager.preview_line = None
            self.roof_drawing_manager.closed = False
            self.roof_drawing_manager.active_vertex_index = None
            self.roof_drawing_manager.active_line_index = None
            self.roof_drawing_manager.reset_temp_state()
            self.roof_drawing_manager.mode = "point"  # 可以默认回到点模式

            # 同时清空右侧列表
            self.update_point_list()

            # 强制刷新画面
            if self.view and self.view.gl_widget:
                self.view.gl_widget.update()
    
    def load_roof_annotation(self):
        """加载当前点云对应的屋顶标注文件"""
        if not hasattr(self, 'roof_drawing_manager'):
            return

        # 约定：屋顶标注文件名为 原文件名 + .roof.obj
        # 例如：123.pcd → 123.roof.obj
        pcd_path = self.pcd_manager.pcd_path
        obj_path = pcd_path.with_suffix('.roof.obj')  # 替换后缀

        if obj_path.exists():
            self.roof_drawing_manager.load_from_obj(str(obj_path))
            self.update_point_list()
            if self.view and self.view.gl_widget:
                self.view.gl_widget.update()
        else:
            # 没有标注文件 → 清空
            self.reset_roof_drawing()

    def move_active_vertex(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0):
        """移动当前选中的顶点（核心方法）"""
        mgr = self.roof_drawing_manager
        if mgr.active_vertex_index is None:
            print("[RoofDrawing] 请先在右侧列表中选中一个顶点进行微调")
            return

        idx = mgr.active_vertex_index
        if idx >= len(mgr.vertices):
            return

        old_pos = np.array(mgr.vertices[idx])
        new_pos_np = old_pos + np.array([dx, dy, dz])

        # === 关键修改：微调时不再吸附到点云点，直接使用偏移后的坐标 ===
        new_pos_tuple = tuple(new_pos_np)

        # 更新 vertices 和 vertex_info
        mgr.vertices[idx] = new_pos_tuple
        mgr.vertex_info[idx]["coord"] = new_pos_tuple

        # 同步更新所有关联的线段
        for line in mgr.lines:
            c1, c2 = line["coord"]
            if np.allclose(c1, old_pos, atol=1e-6):
                line["coord"] = (new_pos_tuple, c2)
            if np.allclose(c2, old_pos, atol=1e-6):
                line["coord"] = (c1, new_pos_tuple)

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
        """切换到标准正交视角，并自动调整相机距离以完整显示点云"""
        if not self.pcd_manager.pointcloud:
            return

        pc = self.pcd_manager.pointcloud
        
        # 1. 先重置旋转，但不要重置平移（避免拉太远）
        pc.rot_x, pc.rot_y, pc.rot_z = 0, 0, 0
        
        # 2. 设置目标旋转角度
        if view_name == "top":      # 上视图（最常用，屋顶平面）
            pc.set_rot_x(0)
            pc.set_rot_y(0)
            pc.set_rot_z(0)
        
        elif view_name == "bottom": # 下视图（翻转，避免倒置）
            pc.set_rot_x(0)
            pc.set_rot_y(180)
            pc.set_rot_z(180)   # 改成180而不是-90，更稳定
        
        elif view_name == "front":  # 前视图（正Y方向）
            pc.set_rot_x(-90)
            pc.set_rot_y(0)
            pc.set_rot_z(0)
        
        elif view_name == "back":   # 后视图
            pc.set_rot_x(-90)
            pc.set_rot_y(0)
            pc.set_rot_z(180)
        
        elif view_name == "left":   # 左视图
            pc.set_rot_x(-90)
            pc.set_rot_y(0)
            pc.set_rot_z(90)
        
        elif view_name == "right":  # 右视图
            pc.set_rot_x(-90)
            pc.set_rot_y(0)
            pc.set_rot_z(-90)   # 保持-90（取模后270，但OpenGL处理正常）
        
        # 3. 关键修复：重新计算并设置合适的相机距离（zoom）
        # 计算点云对角线长度作为参考距离
        extents = np.linalg.norm(pc.pcd_maxs - pc.pcd_mins)
        zoom_distance = -extents * 2.0  # 乘2~3倍系数，确保完整显示（可根据需要调大）
        
        # 只重设Z距离（拉近相机），保留X/Y偏移（如果用户手动平移过）
        pc.set_trans_z(zoom_distance)
        
        # 可选：如果想完全居中，也可以重设X/Y
        # pc.set_trans_x(0)
        # pc.set_trans_y(0)
        
        # 4. 刷新显示
        self.view.gl_widget.updateGL()
    
    def activate_connect_mode(self, enabled: bool):
        """激活或关闭连接两点模式"""
        self.roof_drawing_manager.toggle_connect_mode(enabled)

        # 开启时自动关闭点/线模式
        if enabled and self.roof_drawing_mode is not None:
            self.activate_roof_drawing_mode(None)