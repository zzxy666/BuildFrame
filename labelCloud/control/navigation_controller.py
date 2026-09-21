import numpy as np


class NavigationController:
    """Point-cloud camera navigation independent of annotation state."""

    STANDARD_ROTATIONS = {
        "top": (0, 0, 0),
        "bottom": (0, 180, 180),
        "front": (-90, 0, 0),
        "back": (-90, 0, 180),
        "left": (-90, 0, 90),
        "right": (-90, 0, -90),
    }

    def __init__(self, pcd_manager) -> None:
        self.pcd_manager = pcd_manager
        self.view = None

    def set_view(self, view) -> None:
        self.view = view

    def rotate(self, dx: float, dy: float) -> None:
        self.pcd_manager.rotate_around_x(dy)
        self.pcd_manager.rotate_around_z(dx)

    def translate(self, dx: float, dy: float) -> None:
        self.pcd_manager.translate_along_x(dx)
        self.pcd_manager.translate_along_y(dy)

    def zoom(self, delta: float, focus_point=None) -> None:
        if focus_point is not None:
            pointcloud = self.pcd_manager.pointcloud
            if pointcloud is not None:
                pointcloud.set_orbit_pivot(focus_point)
        self.pcd_manager.zoom_into(delta)

    def set_orbit_pivot(self, point) -> bool:
        pointcloud = self.pcd_manager.pointcloud
        if pointcloud is None or point is None:
            return False
        pointcloud.set_orbit_pivot(point)
        if self.view is not None:
            world = pointcloud.to_world_coordinates(point)
            self.view.status_manager.update_status(
                "旋转中心：{x:.3f}, {y:.3f}, {z:.3f}",
                mode_text="局部旋转",
                x=world[0],
                y=world[1],
                z=world[2],
            )
            self.view.gl_widget.updateGL()
        return True

    def reset_orbit_pivot(self) -> None:
        pointcloud = self.pcd_manager.pointcloud
        if pointcloud is not None:
            pointcloud.reset_orbit_pivot()

    def set_standard_view(self, view_name: str) -> None:
        pointcloud = self.pcd_manager.pointcloud
        if pointcloud is None:
            return
        try:
            rotation = self.STANDARD_ROTATIONS[view_name]
        except KeyError as error:
            raise ValueError(f"Unknown standard view: {view_name}") from error
        pointcloud.set_rotations(*rotation)
        extent = float(np.linalg.norm(pointcloud.pcd_maxs - pointcloud.pcd_mins))
        pointcloud.set_trans_z(-max(extent * 2.0, 1.0))
        if self.view is not None:
            self.view.gl_widget.updateGL()
