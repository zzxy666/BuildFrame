"""批量屏幕投影缓存；Python 循环仅遍历分块和套索边，不遍历点。"""
import logging
import time
import numpy as np
from .roof_planes import polygon_mask


class ScreenProjectionCache:
    CHUNK = 250000

    def __init__(self):
        self.key = None
        self.screen = self.depth = self.valid = None
        self.builds = 0

    @staticmethod
    def camera_key(points, modelview, projection, width, height):
        return (id(points), points.shape, width, height,
                np.asarray(modelview, np.float64).tobytes(), np.asarray(projection, np.float64).tobytes())

    def project(self, points, modelview, projection, width, height, cancel=None):
        key = self.camera_key(points, modelview, projection, width, height)
        if key == self.key: return
        screen = np.empty((len(points), 2), np.float32)
        depth = np.empty(len(points), np.float32)
        valid = np.empty(len(points), bool)
        matrix = np.asarray(modelview) @ np.asarray(projection)
        for start in range(0, len(points), self.CHUNK):
            if cancel is not None and cancel.is_set(): raise InterruptedError("投影已取消")
            stop = min(start+self.CHUNK, len(points))
            xyz = points[start:stop]
            clip = xyz @ matrix[:3, :] + matrix[3, :]
            good = (clip[:, 3] > 1e-12) & np.isfinite(clip).all(axis=1)
            ndc = clip[:, :3] / np.where(good, clip[:, 3], 1)[:, None]
            valid[start:stop] = good & (np.abs(ndc) <= 1+1e-9).all(axis=1)
            screen[start:stop, 0] = (ndc[:, 0]+1)*width/2
            screen[start:stop, 1] = (1-ndc[:, 1])*height/2
            depth[start:stop] = ndc[:, 2]
        self.screen, self.depth, self.valid, self.key = screen, depth, valid, key
        self.builds += 1

    def select(self, polygon, labels, hidden, scope, current, roi=None, cancel=None, debug=False):
        start = time.perf_counter()
        polygon = np.asarray(polygon, float)
        if len(polygon) < 3: return np.empty(0, np.int64)
        low, high = polygon.min(axis=0), polygon.max(axis=0)
        parts = []; bbox_count = 0; filter_seconds = 0; polygon_seconds = 0; available_count = 0
        for first in range(0, len(labels), self.CHUNK):
            if cancel is not None and cancel.is_set(): raise InterruptedError("选择已取消")
            last = min(first+self.CHUNK, len(labels))
            t = time.perf_counter()
            xy = self.screen[first:last]
            allowed = self.valid[first:last] & ~hidden[first:last]
            if roi is not None: allowed &= roi[first:last]
            if scope == "unlabelled": allowed &= labels[first:last] == 0
            elif scope == "current": allowed &= labels[first:last] == current
            available_count += int(allowed.sum())
            allowed &= ((xy >= low) & (xy <= high)).all(axis=1)
            ids = np.flatnonzero(allowed)
            bbox_count += len(ids)
            filter_seconds += time.perf_counter()-t
            t = time.perf_counter()
            if len(ids): parts.append(ids[polygon_mask(xy[ids], polygon)] + first)
            polygon_seconds += time.perf_counter()-t
        result = np.concatenate(parts) if parts else np.empty(0, np.int64)
        if debug:
            logging.info("Lasso total=%d visible/allowed=%d bbox=%d selected=%d bbox_ms=%.1f polygon_ms=%.1f total_ms=%.1f",
                         len(labels), available_count, bbox_count, len(result), filter_seconds*1000,
                         polygon_seconds*1000, (time.perf_counter()-start)*1000)
        return result
