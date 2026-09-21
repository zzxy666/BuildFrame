import logging
from contextlib import contextmanager
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import OpenGL.GL as GL
from OpenGL import GLU
from PyQt5 import QtGui, QtOpenGL

#from ..control.alignmode import AlignMode
#from ..control.bbox_controller import BoundingBoxController
from ..control.config_manager import config
#from ..control.drawing_manager import DrawingManager
from ..control.pcd_manager import PointCloudManger
from ..definitions.types import Color4f, Point2D
from ..utils import oglhelper


def calculate_clipping_planes(focus_distance: float, extent: float):
    """Calculate stable perspective clipping planes for the current zoom."""
    extent = max(float(extent), 1e-6)
    focus = max(float(focus_distance), extent * 1e-4, 1e-4)
    near = max(1e-5, min(focus * 0.05, extent * 1e-3))
    far = max(focus + extent * 4.0, near * 1000.0)
    return near, far


@contextmanager
def ignore_depth_mask():
    GL.glDepthMask(GL.GL_FALSE)
    try:
        yield
    finally:
        GL.glDepthMask(GL.GL_TRUE)


# Main widget for presenting the point cloud
class GLWidget(QtOpenGL.QGLWidget):
    NEAR_PLANE = config.getfloat("USER_INTERFACE", "near_plane")
    FAR_PLANE = config.getfloat("USER_INTERFACE", "far_plane")

    def __init__(self, parent=None) -> None:
        QtOpenGL.QGLWidget.__init__(self, parent)
        self.setMouseTracking(
            True
        )  # mouseMoveEvent is called also without button pressed

        self.modelview: Optional[npt.NDArray] = None
        self.projection: Optional[npt.NDArray] = None
        self.DEVICE_PIXEL_RATIO: float = (
            self.devicePixelRatioF()
        )  # 1 = normal; 2 = retina display
        oglhelper.DEVICE_PIXEL_RATIO = (
            self.DEVICE_PIXEL_RATIO
        )  # set for helper functions

        self.pcd_manager: PointCloudManger = None  # type: ignore
        #self.bbox_controller: BoundingBoxController = None  # type: ignore

        # Objects to be drawn
        self.crosshair_pos: Point2D = (0, 0)
        self.crosshair_col: Color4f = (0, 1, 0, 1)
        self.selected_side_vertices: npt.NDArray = np.array([])
        self.drawing_mode: DrawingManager = None  # type: ignore
        #self.align_mode: Union[AlignMode, None] = None

    def set_pointcloud_controller(self, pcd_manager: PointCloudManger) -> None:
        self.pcd_manager = pcd_manager

    # def set_bbox_controller(self, bbox_controller: BoundingBoxController) -> None:
    #     self.bbox_controller = bbox_controller

    # QGLWIDGET METHODS

    def initializeGL(self) -> None:
        from OpenGL.GLUT import glutInit
        self.glut_available = False
        try:
            glutInit()
            self.glut_available = True
        except Exception:
            logging.warning(
                "GLUT is unavailable; 3D text labels will be disabled.",
                exc_info=True,
            )
        bg_color = [
            int(fl_color)
            for fl_color in config.getlist("USER_INTERFACE", "BACKGROUND_COLOR")
        ]  # floats to ints
        self.qglClearColor(QtGui.QColor(*bg_color))  # screen background color
        GL.glEnable(GL.GL_DEPTH_TEST)  # for visualization of depth
        GL.glEnable(GL.GL_BLEND)  # enable transparency
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        logging.info("Intialized widget.")

        # Must be written again, due to buffer clearing
        if self.pcd_manager.pointcloud is not None:
            self.pcd_manager.pointcloud.create_buffers()

    def resizeGL(self, width, height) -> None:
        logging.info("Resized widget.")
        GL.glViewport(0, 0, width, height)
        self._update_projection(width, height)

    def _update_projection(self, width=None, height=None) -> None:
        """Continuously adapt clipping planes to the current zoom depth."""
        viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
        width = int(width if width is not None else viewport[2])
        height = int(height if height is not None else viewport[3])
        if width <= 0 or height <= 0:
            return
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        aspect = width / float(height)
        pointcloud = self.pcd_manager.pointcloud
        if pointcloud is not None:
            extent = max(
                float(np.linalg.norm(pointcloud.pcd_maxs - pointcloud.pcd_mins)),
                1e-6,
            )
            near, far = calculate_clipping_planes(
                pointcloud.focus_distance(), extent
            )
        else:
            near, far = 0.01, 1000.0
        GLU.gluPerspective(45.0, aspect, near, far)
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def paintGL(self) -> None:
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self.pcd_manager.pointcloud is None:
            return
        self._update_projection()
        GL.glPushMatrix()  # push the current matrix to the current stack

        # Draw point cloud
        self.pcd_manager.pointcloud.draw_pointcloud()  # type: ignore

        # Get actual matrices for click unprojection
        self.modelview = GL.glGetDoublev(GL.GL_MODELVIEW_MATRIX)
        self.projection = GL.glGetDoublev(GL.GL_PROJECTION_MATRIX)
        plane_control = getattr(self, "roof_plane_controller", None)
        if plane_control and plane_control.active:
            plane_control.camera_observed(self.modelview, self.projection, self.width(), self.height())

        with ignore_depth_mask():  # Do not write decoration and preview elements in depth buffer
            if config.getboolean("USER_INTERFACE", "show_floor"):
                oglhelper.draw_xy_plane(self.pcd_manager.pointcloud)  # type: ignore

            # Draw crosshair/ cursor in 3D world
            if self.crosshair_pos and not (plane_control and plane_control.active):
                cx, cy, cz = self.get_world_coords(*self.crosshair_pos, correction=True)
                oglhelper.draw_crosshair(cx, cy, cz, color=self.crosshair_col)

            # if self.drawing_mode.has_preview():
            #     self.drawing_mode.draw_preview()

            """if self.align_mode is not None:
                if self.align_mode.is_active:
                    self.align_mode.draw_preview()"""

            # Highlight selected side with filled rectangle
            if len(self.selected_side_vertices) == 4:
                oglhelper.draw_rectangles(
                    self.selected_side_vertices, color=(0, 1, 0, 0.3)
                )

        plane_mode = getattr(self, "roof_plane_controller", None)
        if hasattr(self, 'roof_drawing_manager') and self.roof_drawing_manager and not (plane_mode and plane_mode.active):
            mgr = self.roof_drawing_manager
            self.draw_roof_annotations()

            if mgr.connect_mode:
                    GL.glDisable(GL.GL_DEPTH_TEST)

                    # 高亮已选中的第一个顶点（紫色大点）
                    if mgr.connect_first_vertex_idx is not None and mgr.vertices:
                        GL.glPointSize(16.0)
                        GL.glBegin(GL.GL_POINTS)
                        GL.glColor3f(0.9, 0.3, 1.0)  # 紫色
                        vx, vy, vz = mgr.vertices[mgr.connect_first_vertex_idx]
                        GL.glVertex3f(vx, vy, vz)
                        GL.glEnd()

                    # 预览线：从已选起点到鼠标当前悬停的最近顶点
                    if mgr.preview_point and mgr.connect_first_vertex_idx is not None and mgr.vertices:
                        distances = [np.linalg.norm(np.array(mgr.preview_point) - np.array(v)) for v in mgr.vertices]
                        nearest_idx = int(np.argmin(distances))
                        min_dist = distances[nearest_idx]
                        if min_dist < 1.5 and nearest_idx != mgr.connect_first_vertex_idx:  # 阈值可调
                            GL.glLineWidth(3.0)
                            GL.glColor3f(0.6, 0.8, 1.0)  # 浅蓝色
                            GL.glBegin(GL.GL_LINES)
                            sx, sy, sz = mgr.vertices[mgr.connect_first_vertex_idx]
                            ex, ey, ez = mgr.vertices[nearest_idx]
                            GL.glVertex3f(sx, sy, sz)
                            GL.glVertex3f(ex, ey, ez)
                            GL.glEnd()

                    GL.glEnable(GL.GL_DEPTH_TEST)

            # 绘制顶点和线段的序号标签
            try:
                from ..utils.oglhelper import draw_text
            except ImportError:
                
                draw_text = None

            if self.glut_available and draw_text and (mgr.vertices or mgr.lines):
                GL.glDisable(GL.GL_DEPTH_TEST)  # 文字始终在最上层

                # 动态计算偏移量（根据点云高度自适应，避免太小或太大）
                if mgr.vertices:
                    zs = [v[2] for v in mgr.vertices]
                    height_range = max(zs) - min(zs) if len(zs) > 1 else 10.0
                    offset = max(0.3, height_range * 0.06)  # 取高度范围的6%
                else:
                    offset = 0.5

                # 1. 顶点序号（绿色文字）
                for i, v in enumerate(mgr.vertices):
                    label_pos = (v[0], v[1], v[2] + offset)
                    draw_text(label_pos, f"Vertex{i+1}", color=(0.0, 1.0, 0.0, 1.0))

                # 2. 线段序号（白色文字，放中点）
                for i, line in enumerate(mgr.lines):
                    p1, p2 = line["coord"]
                    mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2)
                    label_pos = (mid[0], mid[1], mid[2] + offset)
                    draw_text(label_pos, f"Edge{i+1}", color=(1.0, 1.0, 1.0, 1.0))

                GL.glEnable(GL.GL_DEPTH_TEST)

        GL.glPopMatrix()  # restore the previous modelview matrix
        if plane_mode and plane_mode.active:
            plane_mode.draw_roi_boundary()
            self.draw_plane_selection(plane_mode.polygon())

    def draw_plane_selection(self, polygon):
        """选框使用逻辑像素，与 Qt 鼠标位置保持一致（含高 DPI）。"""
        if len(polygon) < 2:
            return
        GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_CURRENT_BIT | GL.GL_LINE_BIT)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPushMatrix()
        GL.glLoadIdentity()
        GL.glOrtho(0, self.width(), self.height(), 0, -1, 1)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        GL.glLoadIdentity()
        GL.glColor3f(1.0, 0.85, 0.05)
        GL.glLineWidth(2)
        GL.glBegin(GL.GL_LINE_LOOP)
        for x, y in polygon:
            GL.glVertex2f(x, y)
        GL.glEnd()
        GL.glPopMatrix()
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPopMatrix()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopAttrib()

    # Translates the 2D cursor position from screen plane into 3D world space coordinates
    def get_world_coords(
        self,
        x: float,
        y: float,
        z: Optional[float] = None,
        correction: bool = False,
        require_surface: bool = False,
    ) -> Optional[Tuple[float, float, float]]:
        x *= self.DEVICE_PIXEL_RATIO  # For fixing mac retina bug
        y *= self.DEVICE_PIXEL_RATIO

        # Stored projection matrices are taken from loop
        viewport = GL.glGetIntegerv(GL.GL_VIEWPORT)
        real_y = viewport[3] - y  # adjust for down-facing y positions

        if z is None:
            buffer_size = 21
            center = buffer_size // 2 + 1
            depths = GL.glReadPixels(
                x - center + 1,
                real_y - center + 1,
                buffer_size,
                buffer_size,
                GL.GL_DEPTH_COMPONENT,
                GL.GL_FLOAT,
            )
            z = depths[center][center]  # Read selected pixel from depth buffer

            if require_surface:
                selected = depths[circular_mask(len(depths), center, 10)]
                valid_depths = selected[
                    np.isfinite(selected) & (selected > 0) & (selected < 1)
                ]
                if len(valid_depths) == 0:
                    return None
                z = float(np.min(valid_depths))

            elif z == 1:
                z = depth_smoothing(depths, center)
            elif correction:
                z = depth_min(depths, center)

        mod_x, mod_y, mod_z = GLU.gluUnProject(
            x, real_y, z, self.modelview, self.projection, viewport
        )
        return mod_x, mod_y, mod_z

    def pick_surface_point(self, x: float, y: float):
        """Pick visible point-cloud geometry near a screen position."""
        if self.modelview is None or self.projection is None:
            return None
        return self.get_world_coords(x, y, correction=True, require_surface=True)

    def draw_roof_annotations(self):
        mgr = getattr(self, "roof_drawing_manager", None)
        if not mgr:
            return

        # 1. 绘制所有已确认的顶点（绿色小球，活跃点红色）
        if mgr.vertices:
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glPointSize(8.0)
            GL.glBegin(GL.GL_POINTS)

            for i, (x, y, z) in enumerate(mgr.vertices):
                # 优先级：单独选中的点 > 线的终点 > 线的起点 > 普通点
                if i == mgr.active_vertex_index:  # 单独点击点选中
                    GL.glColor3f(1.0, 0.0, 0.0)   # 红色
                elif i == mgr.active_line_end_idx:   # 当前选中线的终点
                    GL.glColor3f(1.0, 0.0, 0.0)   # 红色
                elif i == mgr.active_line_start_idx: # 当前选中线的起点
                    GL.glColor3f(0.0, 0.0, 1.0)   # 蓝色
                else:
                    GL.glColor3f(0.0, 1.0, 0.0)   # 绿色（普通点）

                GL.glVertex3f(float(x), float(y), float(z))

            GL.glEnd()
            GL.glEnable(GL.GL_DEPTH_TEST)

        # 2. 绘制所有已确认的线（蓝色粗线）
        if mgr.lines:
            GL.glDisable(GL.GL_DEPTH_TEST)

            for i, line in enumerate(mgr.lines):
                (x1, y1, z1), (x2, y2, z2) = line["coord"]

                if i == getattr(mgr, "active_line_index", None):
                    GL.glColor3f(1.0, 0.5, 0.0)  # 橙色高亮
                    GL.glLineWidth(6.0)
                else:
                    GL.glColor3f(0.0, 0.5, 1.0)  # 深蓝色
                    GL.glLineWidth(3.0)

                GL.glBegin(GL.GL_LINES)
                GL.glVertex3f(float(x1), float(y1), float(z1))
                GL.glVertex3f(float(x2), float(y2), float(z2))
                GL.glEnd()

            GL.glEnable(GL.GL_DEPTH_TEST)

        # 3. 绘制鼠标实时预览（黄色）
        if mgr.preview_point:
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glPointSize(10.0)
            GL.glBegin(GL.GL_POINTS)
            GL.glColor3f(1.0, 1.0, 0.0)  # 黄色预览点
            x, y, z = mgr.preview_point
            GL.glVertex3f(float(x), float(y), float(z))
            GL.glEnd()
            GL.glEnable(GL.GL_DEPTH_TEST)

        if mgr.preview_line:
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glLineWidth(4.0)
            GL.glColor3f(1.0, 0.7, 0.0)  # 橙黄色预览线
            GL.glBegin(GL.GL_LINES)
            (x1, y1, z1), (x2, y2, z2) = mgr.preview_line
            GL.glVertex3f(float(x1), float(y1), float(z1))
            GL.glVertex3f(float(x2), float(y2), float(z2))
            GL.glEnd()
            GL.glEnable(GL.GL_DEPTH_TEST)

        if mgr.connect_mode:
            GL.glDisable(GL.GL_DEPTH_TEST)

            # 高亮已选中的第一个顶点（紫色大点）
            if mgr.connect_first_vertex_idx is not None:
                GL.glPointSize(16.0)
                GL.glBegin(GL.GL_POINTS)
                GL.glColor3f(0.9, 0.3, 1.0)  # 紫色
                x, y, z = mgr.vertices[mgr.connect_first_vertex_idx]
                GL.glVertex3f(float(x), float(y), float(z))
                GL.glEnd()

            # 预览线：从已选起点到当前鼠标悬停的最近顶点（浅蓝色）
            if mgr.preview_point and mgr.connect_first_vertex_idx is not None:
                # 计算鼠标当前悬停的最近顶点
                if mgr.vertices:
                    distances = [np.linalg.norm(np.array(mgr.preview_point) - np.array(v)) for v in mgr.vertices]
                    nearest_idx = int(np.argmin(distances))
                    if distances[nearest_idx] < 1.0 and nearest_idx != mgr.connect_first_vertex_idx:  # 距离合理且不是同一个点
                        GL.glLineWidth(3.0)
                        GL.glColor3f(0.6, 0.8, 1.0)  # 浅蓝色
                        GL.glBegin(GL.GL_LINES)
                        sx, sy, sz = mgr.vertices[mgr.connect_first_vertex_idx]
                        ex, ey, ez = mgr.vertices[nearest_idx]
                        GL.glVertex3f(float(sx), float(sy), float(sz))
                        GL.glVertex3f(float(ex), float(ey), float(ez))
                        GL.glEnd()

            GL.glEnable(GL.GL_DEPTH_TEST)

# Creates a circular mask with radius around center
def circular_mask(arr_length, center, radius) -> np.ndarray:
    dx = np.arange(arr_length)
    return (dx[np.newaxis, :] - center) ** 2 + (
        dx[:, np.newaxis] - center
    ) ** 2 < radius**2


# Returns the minimum (closest) depth for a specified radius around the center
def depth_min(depths, center, r=4) -> float:
    selected_depths = depths[circular_mask(len(depths), center, r)]
    filtered_depths = selected_depths[(0 < selected_depths) & (selected_depths < 1)]
    if 0 in depths:  # Check if cursor is at widget border
        return 1
    elif len(filtered_depths) > 0:
        return np.min(filtered_depths)
    else:
        return 0.5


# Returns the mean depth for a specified radius around the center
def depth_smoothing(depths, center, r=15) -> float:
    selected_depths = depths[circular_mask(len(depths), center, r)]
    if 0 in depths:  # Check if cursor is at widget border
        return 1
    elif np.isnan(
        selected_depths[selected_depths < 1]
    ).all():  # prevent mean of empty slice
        return 1
    return np.nanmedian(selected_depths[selected_depths < 1])
