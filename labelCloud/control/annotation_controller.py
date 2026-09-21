import logging

from PyQt5 import QtWidgets

from ..view.i18n import tr
from .roof_drawing_manager import RoofDrawingManager


class RoofAnnotationController:
    """Own roof annotation state and its persistence lifecycle."""

    def __init__(self, owner) -> None:
        self.owner = owner
        self.manager = RoofDrawingManager(owner)
        self.view = None

    def set_view(self, view) -> None:
        self.view = view
        self.manager.set_view(view)

    def save(self) -> bool:
        pcd_manager = self.owner.pcd_manager
        if not pcd_manager.pcds or pcd_manager.current_id < 0:
            return True
        path = pcd_manager.pcd_path.with_suffix(".roof.json")
        if not (self.manager.vertices or self.manager.lines or path.exists()):
            return True
        try:
            self.manager.save_project(path, pcd_manager.pointcloud)
            logging.info("Saved roof annotation to %s", path)
            return True
        except Exception:
            logging.exception("Failed to save roof annotation to %s", path)
            QtWidgets.QMessageBox.critical(
                self.view,
                tr("保存失败"),
                tr(
                    "无法保存屋顶标注：\n{path}\n\n请检查目录权限和磁盘空间。",
                    path=path,
                ),
            )
            return False

    def load(self) -> bool:
        pcd_manager = self.owner.pcd_manager
        path = pcd_manager.pcd_path.with_suffix(".roof.json")
        legacy_path = pcd_manager.pcd_path.with_suffix(".roof.obj")
        try:
            if path.exists():
                self.manager.load_project(path, pcd_manager.pointcloud)
            elif legacy_path.exists():
                self.manager.load_from_obj(legacy_path)
            else:
                self.manager.clear()
                return False
            return True
        except Exception:
            logging.exception("Failed to load roof annotation for %s", path)
            QtWidgets.QMessageBox.critical(
                self.view,
                tr("加载标注失败"),
                tr("无法读取屋顶标注：\n{path}", path=path),
            )
            self.manager.clear()
            return False

    def reset(self) -> None:
        self.manager.clear()
