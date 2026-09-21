"""工作区、检查网格、书签；全部坐标为当前场景的 local render coordinates。"""
import json
import numpy as np
from .scene_session import atomic_json


class SceneWorkspace:
    def __init__(self, points, session):
        self.points = points
        self.session = session
        self.roi = None
        self.roi_mask = None
        self.roi_revision = 0
        self.bounds = (points.min(axis=0).tolist(), points.max(axis=0).tolist())
        self.grid = None
        self.bookmarks = []
        self.bookmark_index = -1
        self.show_boundary = True
        for name, attr in (("reviewed_grid.json", "grid"), ("bookmarks.json", "bookmarks")):
            path = session.folder / name
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("source") != session.fingerprint:
                    raise ValueError("工作区 metadata 与源点云不匹配")
                setattr(self, attr, data["data"])
        self.set_roi(session.metadata.get("roi"))

    def set_roi(self, bounds):
        if bounds is None:
            self.roi = self.roi_mask = None
        else:
            low, high = np.asarray(bounds, float)
            if low.shape != (3,) or not np.isfinite([low, high]).all() or np.any(low > high):
                raise ValueError("工作区边界无效")
            self.roi = [low.tolist(), high.tolist()]
            mask = np.empty(len(self.points), bool)
            for first in range(0, len(mask), 250000):
                xyz = self.points[first:first+250000]
                mask[first:first+len(xyz)] = ((xyz >= low) & (xyz <= high)).all(axis=1)
            self.roi_mask = mask
        self.roi_revision += 1

    def create_grid(self, size_m, scale):
        if self.grid is not None: return
        step = (float(size_m)/np.asarray(scale[:2])).tolist()
        if not np.isfinite(step).all() or min(step)<=0: raise ValueError("无效网格大小")
        origin = self.bounds[0][:2]
        cells = set()
        for first in range(0,len(self.points),250000):
            xy = self.points[first:first+250000,:2].astype(float)
            indices = np.floor((xy-origin)/step).astype(np.int64)
            cells.update(map(tuple, np.unique(indices,axis=0).tolist()))
        self.grid = {"size_m":size_m,"step":step,"origin":origin,
                     "cells":[list(v) for v in sorted(cells)],"reviewed":[]}

    def cell_bounds(self, cell):
        low = np.asarray(self.bounds[0],float); high = np.asarray(self.bounds[1],float)
        low[:2] = np.asarray(self.grid["origin"])+np.asarray(cell)*self.grid["step"]
        high[:2] = low[:2]+self.grid["step"]
        return [low.tolist(), high.tolist()]

    def mark_reviewed(self):
        if self.roi is None: raise ValueError("请先设置工作区")
        if self.grid is None: raise ValueError("请先创建检查网格")
        done = set(map(tuple,self.grid["reviewed"]))
        low, high = np.asarray(self.roi)
        before = len(done)
        for cell in self.grid["cells"]:
            a,b = np.asarray(self.cell_bounds(cell))
            if (a[:2]>=low[:2]-1e-6).all() and (b[:2]<=high[:2]+1e-6).all(): done.add(tuple(cell))
        self.grid["reviewed"] = [list(v) for v in sorted(done)]
        return len(done)-before

    def next_unreviewed(self):
        if self.grid is None: raise ValueError("请先创建检查网格")
        done = set(map(tuple,self.grid["reviewed"]))
        for cell in self.grid["cells"]:
            if tuple(cell) not in done: return self.cell_bounds(cell)
        return None

    def save(self):
        self.session.folder.mkdir(parents=True,exist_ok=True)
        for name,data in (("reviewed_grid.json", self.grid), ("bookmarks.json", self.bookmarks)):
            atomic_json(self.session.folder/name, {"source":self.session.fingerprint,"data":data})
