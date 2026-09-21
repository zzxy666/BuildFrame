"""大场景平面状态：uint32、增量统计、差分撤销、侧车保存。"""
from collections import deque
from pathlib import Path
import laspy
import numpy as np
from .scene_session import SceneSession, las_backend


class RoofPlanes:
    UNDO_LIMIT = 128 * 1024 * 1024

    def __init__(self, path):
        self.path = Path(path)
        self.output_path = self.path.with_name(self.path.stem + "_gt.las")
        self.session = SceneSession(path)
        self.header = self.session.header
        values = self.session.load()
        if values is None:
            values = np.zeros(self.header.point_count, dtype=np.uint32)
            # 兼容历史 GT，但仅首次导入；已建立侧车后以侧车为准。
            source = self.output_path if self.output_path.exists() else self.path
            with laspy.open(source, laz_backend=las_backend()) as reader:
                if reader.header.point_count != len(values):
                    raise ValueError("已有输出与原始点云不匹配")
                original = laspy.open(self.path, laz_backend=las_backend()) if source != self.path else None
                try:
                    offset = 0
                    has_labels = any(d in reader.header.point_format.dimension_names for d in ("plane_id", "planeid", "PlaneID", "PlaneId"))
                    chunks = reader.chunk_iterator(250000) if has_labels or original else ()
                    for chunk in chunks:
                        if original:
                            ref = original.read_points(len(chunk))
                            if not (np.array_equal(reader.header.scales, original.header.scales) and
                                    np.array_equal(reader.header.offsets, original.header.offsets) and
                                    all(np.array_equal(chunk[d], ref[d]) for d in ("X", "Y", "Z"))):
                                raise ValueError("已有输出与原始点序/坐标不匹配")
                        field = next((d for d in ("plane_id", "planeid", "PlaneID", "PlaneId")
                                      if d in chunk.point_format.dimension_names), None)
                        if field:
                            part = np.asarray(chunk[field])
                            if (not np.isfinite(part).all() or np.any(part < 0) or
                                    np.any(part > np.iinfo(np.uint32).max) or np.any(part != np.floor(part))):
                                raise ValueError("plane_id 必须是 uint32 范围内的整数")
                            values[offset:offset+len(chunk)] = part
                        offset += len(chunk)
                finally:
                    if original: original.close()
        self.labels = values
        ids, counts = np.unique(values, return_counts=True)  # 每次加载只初始化一次。
        self.plane_counts = dict(zip(map(int, ids), map(int, counts)))
        self.plane_counts.setdefault(0, 0)
        meta = self.session.metadata
        self.plane_ids = set(self.plane_counts) | set(map(int, meta.get("empty_planes", [])))
        self.max_plane_id = max(max(self.plane_ids), int(meta.get("max_plane_id", 0)))
        self.current = int(meta.get("current_plane", 0))
        if self.current not in self.plane_ids: self.current = 0
        self.selection = np.zeros(len(values), dtype=bool)
        self.hidden_mask = np.zeros(len(values), dtype=bool)
        self.hidden_count = 0
        self.hidden_revision = 0
        self.labels_revision = 0
        self.table_revision = 0
        self.revision = 0
        self.saved_revision = 0
        self.history = deque()
        self.history_bytes = 0
        self.undo_discarded = 0
        self.changed_ids = np.empty(0, dtype=np.int64)
        self._plane_indices = {}
        self.session_metadata = {k: v for k, v in meta.items() if k not in ("source", "version")}

    @property
    def source(self):
        # 仅为旧外部调用保留，交互/扩展/保存不使用全量 LasData。
        from .roof_planes import read_las
        return read_las(self.path)

    @property
    def dirty(self):
        return self.revision != self.saved_revision or bool(self.session.pending)

    def touch(self):
        self.revision += 1

    def _remember(self, kind, ids, values, added=(), removed=()):
        ids = np.asarray(ids, dtype=np.int64)
        record = dict(kind=kind, ids=ids.copy(), values=np.asarray(values).copy(),
                      current=self.current, added=tuple(added), removed=tuple(removed))
        if kind == "hidden": record["selection"] = np.flatnonzero(self.selection)
        record["bytes"] = sum(v.nbytes for v in record.values() if isinstance(v, np.ndarray)) + 256
        self.history.append(record); self.history_bytes += record["bytes"]
        while self.history_bytes > self.UNDO_LIMIT and self.history:
            self.history_bytes -= self.history.popleft()["bytes"]
            self.undo_discarded += 1

    def _change(self, ids, values):
        old, counts = np.unique(self.labels[ids], return_counts=True)
        for pid, n in zip(old, counts):
            pid = int(pid)
            self.plane_counts[pid] = self.plane_counts.get(pid, 0) - int(n)
            self._plane_indices.pop(pid, None)
        self.labels[ids] = values
        new, counts = np.unique(self.labels[ids], return_counts=True)
        for pid, n in zip(new, counts):
            pid = int(pid)
            self.plane_counts[pid] = self.plane_counts.get(pid, 0) + int(n)
            self._plane_indices.pop(pid, None)
        self.session.queue(ids, self.labels[ids])
        self.changed_ids = np.asarray(ids, dtype=np.int64)
        self.labels_revision += 1; self.table_revision += 1; self.touch()

    def new_plane(self):
        if self.max_plane_id >= np.iinfo(np.uint32).max:
            raise ValueError("Plane ID 已超出 uint32 范围")
        self.max_plane_id += 1
        pid = self.max_plane_id
        self._remember("labels", [], np.empty(0, np.uint32), added=(pid,))
        self.current = pid; self.plane_ids.add(pid); self.plane_counts[pid] = 0
        self.table_revision += 1; self.touch()
        return pid

    def selectable(self, scope="unlabelled", ids=None):
        hidden = self.hidden_mask if ids is None else self.hidden_mask[ids]
        labels = self.labels if ids is None else self.labels[ids]
        allowed = ~hidden
        if scope == "unlabelled": allowed &= labels == 0
        elif scope == "current": allowed &= labels == self.current
        elif scope != "all": raise ValueError("未知选择范围")
        return allowed

    def set_hidden_ids(self, ids, hidden):
        ids = np.asarray(ids, dtype=np.int64)
        ids = ids[self.hidden_mask[ids] != hidden]
        if not len(ids): return False
        self._remember("hidden", ids, self.hidden_mask[ids])
        self.hidden_mask[ids] = hidden
        self.hidden_count += len(ids) * (1 if hidden else -1)
        if hidden: self.selection[ids] = False
        self.hidden_revision += 1
        return True

    def set_hidden(self, mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.hidden_mask.shape: raise ValueError("隐藏掩膜点数错误")
        ids = np.flatnonzero(mask != self.hidden_mask)
        if not len(ids): return False
        self._remember("hidden", ids, self.hidden_mask[ids])
        self.hidden_count += int(mask[ids].sum()) - int(self.hidden_mask[ids].sum())
        self.hidden_mask[ids] = mask[ids]
        self.selection[ids[mask[ids]]] = False
        self.hidden_revision += 1
        return True

    def assign(self, plane_id, scope="unlabelled", ids=None):
        pid = int(plane_id)
        if not 0 <= pid <= np.iinfo(np.uint32).max: raise ValueError("无效 Plane ID")
        ids = np.flatnonzero(self.selection) if ids is None else np.asarray(ids, dtype=np.int64)
        ids = ids[self.selectable(scope, ids)]
        if not len(ids): return False
        changed = ids[self.labels[ids] != pid]
        added = () if pid in self.plane_ids else (pid,)
        if len(changed) or added:
            self._remember("labels", changed, self.labels[changed], added=added)
            self.plane_ids.add(pid)
            self.max_plane_id = max(self.max_plane_id, pid)
            self._change(changed, pid)
        self.current = pid
        self.selection[:] = False
        return True

    def plane_indices(self, pid):
        if pid not in self._plane_indices:
            self._plane_indices[pid] = np.flatnonzero(self.labels == pid)
            if len(self._plane_indices) > 8:
                del self._plane_indices[next(iter(self._plane_indices))]
        return self._plane_indices[pid]

    def delete(self, plane_id, allowed=None):
        if plane_id == 0 or plane_id not in self.plane_ids: return
        ids = self.plane_indices(plane_id)
        if allowed is not None: ids = ids[allowed[ids]]
        removed = (plane_id,) if len(ids) == self.plane_counts.get(plane_id, 0) else ()
        self._remember("labels", ids, self.labels[ids], removed=removed)
        self._change(ids, 0)
        self.plane_ids.difference_update(removed); self.current = 0

    def merge(self, source, target, allowed=None):
        if source == target or 0 in (source, target) or not {source, target} <= self.plane_ids:
            raise ValueError("请选择两个不同的 Roof Plane（ID 大于 0）")
        ids = self.plane_indices(source)
        if allowed is not None: ids = ids[allowed[ids]]
        removed = (source,) if len(ids) == self.plane_counts.get(source, 0) else ()
        self._remember("labels", ids, self.labels[ids], removed=removed)
        self._change(ids, target)
        self.plane_ids.difference_update(removed); self.current = target

    def undo(self):
        if not self.history: return False
        record = self.history.pop(); self.history_bytes -= record["bytes"]
        ids = record["ids"]
        self.selection[:] = False
        if record["kind"] == "hidden":
            self.hidden_count += int(record["values"].sum()) - int(self.hidden_mask[ids].sum())
            self.hidden_mask[ids] = record["values"]
            self.selection[record["selection"]] = True
            self.hidden_revision += 1
        else:
            self._change(ids, record["values"])
            self.plane_ids.difference_update(record["added"])
            self.plane_ids.update(record["removed"])
            self.current = record["current"]
            self.selection[ids] = True
        self.selection[self.hidden_mask] = False
        return True

    def colors(self, ids=None):
        from .roof_planes import plane_color
        values = self.labels if ids is None else self.labels[ids]
        unique, inverse = np.unique(values, return_inverse=True)
        return np.asarray([plane_color(int(i)) for i in unique], np.float32)[inverse]

    def counts(self):
        return [(pid, self.plane_counts.get(pid, 0)) for pid in sorted(self.plane_ids)]

    def save(self):
        self.session_metadata.update(max_plane_id=self.max_plane_id, current_plane=self.current,
                                     empty_planes=[pid for pid in self.plane_ids if not self.plane_counts.get(pid)])
        self.session.save(self.labels, self.session_metadata)
        self.saved_revision = self.revision
        return self.session.folder / "plane_id.npy"

    def export(self, destination=None, cancel=None):
        return self.session.export(self.labels, destination or self.output_path, cancel)
