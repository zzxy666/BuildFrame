import ctypes
import logging
from pathlib import Path
from typing import List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt
import OpenGL.GL as GL
from PyQt5.QtWidgets import QMessageBox

#from labelCloud.io.labels.config import LabelConfig

from ..control.config_manager import config
from ..definitions import Point3D, Rotations3D, Translation3D
from ..io.pointclouds import BasePointCloudHandler
#from ..io.segmentations import BaseSegmentationHandler
from ..utils.color import colorize_points_with_height
from ..utils.logger import end_section, green, print_column, red, start_section, yellow
from . import Perspective

# Get size of float (4 bytes) for VBOs
SIZE_OF_FLOAT = ctypes.sizeof(ctypes.c_float)


def calculate_init_translation(
    center: Tuple[float, float, float], mins: npt.NDArray, maxs: npt.NDArray
) -> Point3D:
    """Calculates the initial translation (x, y, z) of the point cloud. Considers ...

    - the point cloud center
    - the point cloud extents
    - the far plane setting (caps zoom)
    """
    zoom = min(  # type: ignore
        np.linalg.norm(maxs - mins),
        config.getfloat("USER_INTERFACE", "far_plane") * 0.9,
    )
    return tuple(-np.add(center, [0, 0, zoom]))  # type: ignore


def consecutive(data: npt.NDArray[np.int64], stepsize=1) -> List[npt.NDArray[np.int64]]:
    """Split an 1-d array of integers to a list of 1-d array where the elements are consecutive"""
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


class PointCloud(object):
    def __init__(
        self,
        path: Path,
        points: npt.NDArray[np.float32],
        colors: Optional[np.ndarray] = None,
        segmentation_labels: Optional[npt.NDArray[np.int8]] = None,
        init_translation: Optional[Tuple[float, float, float]] = None,
        init_rotation: Optional[Tuple[float, float, float]] = None,
        original_center: Optional[npt.NDArray] = None,
        applied_scale: float = 1.0,
        write_buffer: bool = True,
    ) -> None:
        start_section(f"Loading {path.name}")
        self.path = path
        self.points = points
        self.colors = colors if type(colors) == np.ndarray and len(colors) > 0 else None
        self.original_colors = self.colors
        self.original_center = np.asarray(
            original_center if original_center is not None else np.zeros(3),
            dtype=np.float64,
        )
        self.applied_scale = float(applied_scale)
        if self.applied_scale <= 0:
            raise ValueError("applied_scale must be greater than zero")

        self.labels = None
        self.display_colors = None
        self.display_colors_dirty = False
        self.display_indices = None
        self.overlays = []
        self.color_updates = []
        self.lod_stride = 1
        # if LabelConfig().type == LabelingMode.SEMANTIC_SEGMENTATION:
        #     self.labels = segmentation_labels
        #     self.validate_segmentation_label()
        #     self.mix_ratio = config.getfloat("POINTCLOUD", "label_color_mix_ratio")

        self.vbo = None
        self.center: Point3D = tuple(np.sum(points[:, i]) / len(points) for i in range(3))  # type: ignore
        self.pcd_mins: npt.NDArray[np.float32] = np.amin(points, axis=0)
        self.pcd_maxs: npt.NDArray[np.float32] = np.amax(points, axis=0)
        self.default_orbit_pivot = np.add(
            self.pcd_mins, np.subtract(self.pcd_maxs, self.pcd_mins) / 2
        ).astype(np.float64)
        self.orbit_pivot = self.default_orbit_pivot.copy()
        self.init_translation: Point3D = init_translation or calculate_init_translation(
            self.center, self.pcd_mins, self.pcd_maxs
        )
        self.init_rotation: Rotations3D = init_rotation or tuple([0, 0, 0])  # type: ignore

        # Point cloud transformations
        self.trans_x, self.trans_y, self.trans_z = self.init_translation
        self.rot_x, self.rot_y, self.rot_z = self.init_rotation


        if self.colorless:
            # if no color in point cloud, either color with height or color with a single color
            if config.getboolean("POINTCLOUD", "COLORLESS_COLORIZE"):
                self.colors = colorize_points_with_height(
                    self.points, self.pcd_mins[2], self.pcd_maxs[2]
                )
                logging.info(
                    "Generated colors for colorless point cloud based on height."
                )
            else:
                colorless_color = np.array(
                    config.getlist("POINTCLOUD", "COLORLESS_COLOR")
                )
                self.colors = (np.ones_like(self.points) * colorless_color).astype(
                    np.float32
                )
                logging.info(
                    "Generated colors for colorless point cloud based on `colorless_color`."
                )
        if write_buffer:
            self.create_buffers()

        logging.info(green(f"Successfully loaded point cloud from {path}!"))
        self.print_details()
        end_section()

    @property
    def point_size(self) -> float:
        return config.getfloat("POINTCLOUD", "point_size")

    def create_buffers(self) -> None:
        """Create 3 different buffers holding points, colors and label colors information"""
        self.release_buffers()
        self.colors = cast(npt.NDArray[np.float32], self.colors)
        (
            self.position_vbo,
            self.color_vbo,
            self.label_vbo,
        ) = GL.glGenBuffers(3)
        for data, vbo in [
            (self.points, self.position_vbo),
            (self.colors, self.color_vbo),
            (self.label_colors, self.label_vbo),
        ]:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_DYNAMIC_DRAW)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        self.display_colors_dirty = self.display_colors is not None

    def set_display_colors(self, colors) -> None:
        """只改变显示缓冲；原始 RGB 保留在 self.colors 和 LAS 中。"""
        self.display_colors = None if colors is None else np.ascontiguousarray(colors, dtype=np.float32)
        self.display_colors_dirty = True

    def update_display_colors(self, ids, colors):
        if self.display_colors is not None and len(ids):
            self.display_colors[ids] = colors
            self.color_updates.extend(np.unique(np.asarray(ids)//250000).tolist())

    def set_overlays(self, layers):
        self.overlays = [(np.ascontiguousarray(ids, dtype=np.uint32), color)
                         for ids, color in layers if len(ids)]

    def set_display_mask(self, visible=None) -> None:
        """只筛选绘制索引，不裁剪/重排原始 XYZ、RGB 或标签数组。"""
        if visible is None:
            self.display_indices = None
            return
        visible = np.asarray(visible, dtype=bool)
        if visible.shape != (len(self.points),):
            raise ValueError("显示掩膜与原始点数不一致")
        self.display_indices = np.ascontiguousarray(np.flatnonzero(visible), dtype=np.uint32)

    def release_buffers(self) -> None:
        buffer_ids = [
            getattr(self, name, None)
            for name in ("position_vbo", "color_vbo", "label_vbo")
        ]
        existing = [buffer_id for buffer_id in buffer_ids if buffer_id]
        if existing:
            try:
                GL.glDeleteBuffers(len(existing), existing)
            except Exception:
                logging.debug("Could not release OpenGL buffers", exc_info=True)
        self.position_vbo = self.color_vbo = self.label_vbo = None

    @property
    def label_colors(self) -> npt.NDArray[np.float32]:
        """blend the points with label color map"""
        self.colors = cast(npt.NDArray[np.float32], self.colors)
        if self.labels is not None:
            pass
            #colors = LabelConfig().color_map[LabelConfig().class_order[self.labels]]
            #return colors * self.mix_ratio + self.colors * (1 - self.mix_ratio)
        else:
            return self.colors

    # def save_segmentation_labels(self, extension=".bin") -> None:
    #     label_path = (
    #         config.getpath("FILE", "segmentation_folder")
    #         / f"{self.path.stem}{extension}"
    #     )
    #     seg_handler: BaseSegmentationHandler = BaseSegmentationHandler.get_handler(
    #         label_path.suffix
    #     )()
    #     assert self.labels is not None
    #     self.validate_segmentation_label()
    #     seg_handler.overwrite_labels(label_path=label_path, labels=self.labels)
    #     logging.info(f"Writing segmentation labels to {label_path}")

    @classmethod
    def from_file(
        cls,
        path: Path,
        perspective: Optional[Perspective] = None,
        write_buffer: bool = True,
    ) -> "PointCloud":
        init_translation, init_rotation = (None, None)
        if perspective:
            init_translation = perspective.translation
            init_rotation = perspective.rotation

        points, colors = BasePointCloudHandler.get_handler(
            path.suffix
        ).read_point_cloud(path=path)
        
        if points is None or len(points) == 0:
            raise ValueError(f"Point cloud contains no points: {path}")

        # Keep a per-cloud reversible transform. Rendering and annotations use
        # local coordinates; file exports can convert them back to world space.
        points = np.asarray(points, dtype=np.float64)
        center = points.mean(axis=0, dtype=np.float64)
        scale = 1.0
        points -= center

        bbox_size = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        if bbox_size > 1e4:
            scale = bbox_size / 100.0
            points /= scale
            logging.info(f"Auto-scaled point cloud by factor 1/{scale:.2f}")

        points = points.astype(np.float32)

        logging.info(f"Centered point cloud at origin, original center = {center}")



        labels = None
        # if LabelConfig().type == LabelingMode.SEMANTIC_SEGMENTATION:
        #     label_path = (
        #         config.getpath("FILE", "segmentation_folder") / f"{path.stem}.bin"
        #     )
        #     logging.info(f"Loading segmentation labels from {label_path}.")
        #     seg_handler = BaseSegmentationHandler.get_handler(label_path.suffix)()
        #     labels = seg_handler.read_or_create_labels(
        #         label_path=label_path, num_points=points.shape[0]
        #     )

        return cls(
            path,
            points,
            colors,
            labels,
            init_translation,
            init_rotation,
            center,
            scale,
            write_buffer,
        )

    def to_world_coordinates(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Convert rendering/local coordinates to source-file coordinates."""
        local = np.asarray(points, dtype=np.float64)
        return local * self.applied_scale + self.original_center

    def to_local_coordinates(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Convert source-file coordinates to rendering/local coordinates."""
        world = np.asarray(points, dtype=np.float64)
        return (world - self.original_center) / self.applied_scale

    @property
    def last_center(self) -> npt.NDArray[np.float64]:
        """Backward-compatible alias for older UI code."""
        return self.original_center

    @property
    def last_scale(self) -> float:
        """Backward-compatible alias for older UI code."""
        return self.applied_scale

    # def validate_segmentation_label(self) -> None:
    #     unique_label_ids = set(np.unique(self.labels))  # type: ignore
    #     unique_class_ids = set(c.id for c in LabelConfig().classes)
    #     if not unique_class_ids.issuperset(unique_label_ids):
    #         msg = QMessageBox()
    #         msg.setWindowTitle("Invalid segmentation label")
    #         msg.setText(
    #             f"Segmentation labels {unique_label_ids} of `{self.path}` don't match with the label config {unique_class_ids}."
    #         )
    #         labels_to_replace = unique_label_ids.difference(unique_class_ids)
    #         msg.setInformativeText(
    #             f"""
    #             Do you want to overwrite 
    #             the undefined labels {labels_to_replace} with 
    #             default label `{LabelConfig().get_default_class_name()}` of id `{LabelConfig().default}`?
    #             """
    #         )
    #         msg.setIcon(QMessageBox.Critical)
    #         msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)

    #         msg.accepted.connect(self.replace_missing_labels_with_default)
    #         msg.exec_()

    # def replace_missing_labels_with_default(self):
    #     unique_label_ids = set(np.unique(self.labels))
    #     unique_class_ids = set(c.id for c in LabelConfig().classes)
    #     labels_to_replace = list(unique_label_ids.difference(unique_class_ids))
    #     self.labels[np.isin(self.labels, labels_to_replace)] = LabelConfig().default

    def to_file(self, path: Optional[Path] = None) -> None:
        if not path:
            path = self.path
        BasePointCloudHandler.get_handler(path.suffix).write_point_cloud(
            path=path, pointcloud=self
        )

    @property
    def colorless(self) -> bool:
        return self.colors is None

    @property
    def color_with_label(self) -> bool:
        return config.getboolean("POINTCLOUD", "color_with_label")

    @property
    def has_label(self) -> bool:
        return self.labels is not None

    def update_selected_points_in_label_vbo(
        self, points_inside: npt.NDArray[np.bool_]
    ) -> None:
        """Send the selected updated label colors to label vbo. This function
        assumes the `self.label_colors[points_inside]` have been altered.
        This function only partially updates the label vbo to minimise the
        data sent to gpu. It leverages `glBufferSubData` method to perform
        partial update and `consecutive` method to find consecutive indexes
        so they can be updated in one single `glBufferSubData` call.
        """
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.label_vbo)
        inside_idx = np.where(points_inside)[0]
        if inside_idx.shape[0] == 0:
            logging.warning("No points are found inside the selected boxes.")
            return
        logging.debug(f"Update {len(inside_idx)} point colors in label VBO.")
        # find contiguous points so they can be updated together in one glBufferSubData call
        arrays = consecutive(inside_idx)
        label_color = self.label_colors
        stride = label_color.shape[1] * SIZE_OF_FLOAT
        for arr in arrays:
            colors: npt.NDArray[np.float32] = label_color[arr]
            # partially update label_vbo from positions arr[0] to arr[-1]
            GL.glBufferSubData(
                GL.GL_ARRAY_BUFFER,
                offset=arr[0] * stride,
                size=colors.nbytes,
                data=colors,
            )

    # GETTERS AND SETTERS
    def get_no_of_points(self) -> int:
        return len(self.points)

    def get_no_of_colors(self) -> int:
        return len(self.colors) if self.colors else 0

    def get_rotations(self) -> Rotations3D:
        return self.rot_x, self.rot_y, self.rot_z

    def get_translation(self) -> Translation3D:
        return self.trans_x, self.trans_y, self.trans_z

    def get_mins_maxs(self) -> Tuple[npt.NDArray, npt.NDArray]:
        return self.pcd_mins, self.pcd_maxs

    def get_min_max_height(self) -> Tuple[float, float]:
        return self.pcd_mins[2], self.pcd_maxs[2]

    def set_rot_x(self, angle) -> None:
        self.rot_x = angle % 360

    def set_rot_y(self, angle) -> None:
        self.rot_y = angle % 360

    def set_rot_z(self, angle) -> None:
        self.rot_z = angle % 360

    def set_rotations(self, x: float, y: float, z: float) -> None:
        self.rot_x = x % 360
        self.rot_y = y % 360
        self.rot_z = z % 360

    def set_trans_x(self, val) -> None:
        self.trans_x = val

    def set_trans_y(self, val) -> None:
        self.trans_y = val

    def set_trans_z(self, val) -> None:
        self.trans_z = val

    def set_translations(self, x: float, y: float, z: float) -> None:
        self.trans_x = x
        self.trans_y = y
        self.trans_z = z

    def _rotation_matrix(self) -> npt.NDArray[np.float64]:
        """Return the matrix matching the legacy OpenGL X/Y/Z call order."""
        x, y, z = np.deg2rad([self.rot_x, self.rot_y, self.rot_z])
        rx = np.array(
            [[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]]
        )
        ry = np.array(
            [[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]]
        )
        rz = np.array(
            [[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]]
        )
        return rx @ ry @ rz

    def set_orbit_pivot(self, pivot: npt.ArrayLike) -> None:
        """Change the orbit pivot without moving the currently rendered scene."""
        new_pivot = np.asarray(pivot, dtype=np.float64)
        if new_pivot.shape != (3,) or not np.isfinite(new_pivot).all():
            raise ValueError("orbit pivot must contain three finite coordinates")
        old_pivot = self.orbit_pivot
        rotation = self._rotation_matrix()
        compensation = (np.eye(3) - rotation) @ (old_pivot - new_pivot)
        self.trans_x += float(compensation[0])
        self.trans_y += float(compensation[1])
        self.trans_z += float(compensation[2])
        self.orbit_pivot = new_pivot

    def reset_orbit_pivot(self) -> None:
        self.set_orbit_pivot(self.default_orbit_pivot)

    def focus_distance(self) -> float:
        """Distance from the camera plane to the current orbit/zoom pivot."""
        return max(0.0, -float(self.trans_z + self.orbit_pivot[2]))

    def dolly(self, wheel_delta: float) -> float:
        """Exponentially zoom without crossing the current focus point."""
        extent = max(float(np.linalg.norm(self.pcd_maxs - self.pcd_mins)), 1e-6)
        minimum = max(extent * 1e-4, 1e-4)
        maximum = max(extent * 1e3, minimum)
        current = max(self.focus_distance(), minimum)
        wheel_steps = float(wheel_delta) / 120.0
        target = float(np.clip(current * (0.82**wheel_steps), minimum, maximum))
        self.trans_z = -target - float(self.orbit_pivot[2])
        return target

    def set_gl_background(self) -> None:
        GL.glTranslate(
            self.trans_x, self.trans_y, self.trans_z
        )  # third, pcd translation

        GL.glTranslate(*self.orbit_pivot)

        GL.glRotate(self.rot_x, 1.0, 0.0, 0.0)
        GL.glRotate(self.rot_y, 0.0, 1.0, 0.0)  # second, pcd rotation
        GL.glRotate(self.rot_z, 0.0, 0.0, 1.0)

        GL.glTranslate(*(-self.orbit_pivot))
        GL.glPointSize(self.point_size)

    def draw_pointcloud(self) -> None:
        if getattr(self, "position_vbo", None) is None:
            self.create_buffers()  # 只在当前 OpenGL 上下文中创建，导航/后台加载不创建 VBO。
        self.set_gl_background()
        if self.display_colors_dirty:
            colors = self.colors if self.display_colors is None else self.display_colors
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.color_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, colors.nbytes, colors, GL.GL_DYNAMIC_DRAW)
            self.display_colors_dirty = False
            self.color_updates.clear()
        elif self.color_updates and self.display_colors is not None:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.color_vbo)
            for block in sorted(set(self.color_updates)):
                first = block*250000
                data = self.display_colors[first:first+250000]
                GL.glBufferSubData(GL.GL_ARRAY_BUFFER, first*3*SIZE_OF_FLOAT, data.nbytes, data)
            self.color_updates.clear()
        stride = 3 * SIZE_OF_FLOAT

        # Bind position buffer
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.position_vbo)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glVertexPointer(3, GL.GL_FLOAT, stride, None)

        # Bind color buffer
        if self.color_with_label and self.display_colors is None:
            color_vbo = self.label_vbo
        else:
            color_vbo = self.color_vbo
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, color_vbo)
        GL.glEnableClientState(GL.GL_COLOR_ARRAY)
        GL.glColorPointer(3, GL.GL_FLOAT, stride, None)
        if self.display_indices is None:
            step = self.lod_stride
            if step > 1:
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.position_vbo)
                GL.glVertexPointer(3, GL.GL_FLOAT, stride*step, None)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, color_vbo)
                GL.glColorPointer(3, GL.GL_FLOAT, stride*step, None)
            GL.glDrawArrays(GL.GL_POINTS, 0, (self.get_no_of_points()+step-1)//step)
        else:
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
            ids = self.display_indices[::self.lod_stride]
            GL.glDrawElements(GL.GL_POINTS, len(ids), GL.GL_UNSIGNED_INT, ids)

        if self.overlays:
            # 独立高亮层复用原始位置 VBO；只提交选中点的 original_point_id。
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.position_vbo)
            GL.glVertexPointer(3, GL.GL_FLOAT, stride, None)
            GL.glDisableClientState(GL.GL_COLOR_ARRAY)
            GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_CURRENT_BIT | GL.GL_POINT_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glDisable(GL.GL_DEPTH_TEST); GL.glDepthMask(False)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
            GL.glPointSize(self.point_size+1)
            for ids, color in self.overlays:
                GL.glColor3f(*color)
                GL.glDrawElements(GL.GL_POINTS, len(ids), GL.GL_UNSIGNED_INT, ids)
            GL.glPopAttrib()

        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glDisableClientState(GL.GL_COLOR_ARRAY)
        # Release the buffer binding
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def reset_perspective(self) -> None:
        self.trans_x, self.trans_y, self.trans_z = self.init_translation
        self.rot_x, self.rot_y, self.rot_z = self.init_rotation
        self.orbit_pivot = self.default_orbit_pivot.copy()

    def get_filtered_pointcloud(
        self, indicies: npt.NDArray[np.bool_]
    ) -> Optional["PointCloud"]:
        assert self.points is not None
        assert self.colors is not None
        points = self.points[indicies]
        if points.shape[0] == 0:
            return None
        colors = self.colors[indicies]
        labels = self.labels[indicies] if self.labels is not None else None
        path = self.path.parent / (self.path.stem + "_cropped" + self.path.suffix)
        return PointCloud(
            path=path,
            points=points,
            colors=colors,
            segmentation_labels=labels,
            original_center=self.original_center,
            applied_scale=self.applied_scale,
            write_buffer=False,
        )

    def print_details(self) -> None:
        print_column(
            [
                "Number of Points:",
                (
                    green(len(self.points))
                    if len(self.points) > 0
                    else red(len(self.points))
                ),
            ]
        )
        print_column(
            [
                "Number of Colors:",
                (
                    yellow("None")
                    if self.colorless
                    else (
                        green(len(self.colors))  # type: ignore
                        if len(self.colors) == len(self.points)  # type: ignore
                        else red(len(self.colors))  # type: ignore
                    )
                ),
            ]
        )
        print_column(["Point Cloud Center:", str(np.round(self.center, 2))])
        print_column(["Point Cloud Minimums:", str(np.round(self.pcd_mins, 2))])
        print_column(["Point Cloud Maximums:", str(np.round(self.pcd_maxs, 2))])
        print_column(
            ["Initial Translation:", str(np.round(self.init_translation, 2))], last=True
        )
