"""轻量 Plane 表：行是 ID，点数来自增量字典，不创建逐行 QWidget。"""
from PyQt5 import QtCore


class PlaneTableModel(QtCore.QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = None
        self.ids = []
        self.revision = -1
        self.query = ""

    def rowCount(self, parent=QtCore.QModelIndex()):
        return 0 if parent.isValid() else len(self.ids)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return 2

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self.ids): return None
        pid = self.ids[index.row()]
        if role == QtCore.Qt.UserRole: return pid
        if role == QtCore.Qt.DisplayRole:
            if index.column() == 0: return "0 Unassigned / Background" if pid == 0 else f"Plane {pid}"
            return self.scene.plane_counts.get(pid, 0)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return ("Plane", "points")[section]

    def refresh(self, scene, query=None):
        if query is None: query = self.query
        if scene is self.scene and scene.table_revision == self.revision and query == self.query: return
        ids = [pid for pid in sorted(scene.plane_ids) if query in str(pid)]
        changed = ids != self.ids or scene is not self.scene
        if changed: self.beginResetModel()
        self.ids=ids; self.scene=scene; self.query=query; self.revision=scene.table_revision
        if changed: self.endResetModel()
        elif ids: self.dataChanged.emit(self.index(0,1),self.index(len(ids)-1,1))
