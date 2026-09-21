"""逐点 Roof Plane 标注；不依赖 GUI，不修改原始 LAS 点记录。"""

import colorsys
import copy
import os
from pathlib import Path
import tempfile

import laspy
import numpy as np


def read_las(path):
    backend = laspy.LazBackend.Laszip if laspy.LazBackend.Laszip.is_available() else None
    return laspy.read(path, laz_backend=backend)


def plane_color(plane_id):
    if plane_id == 0:
        return (0.48, 0.48, 0.48)
    return colorsys.hsv_to_rgb((plane_id * 0.61803398875) % 1, 0.72, 0.95)


def project_points(points, modelview, projection, width, height):
    """OpenGL 矩阵转屏幕逻辑像素；屏蔽相机后方和裁剪范围外的点。"""
    homogeneous = np.column_stack((points, np.ones(len(points))))
    clip = homogeneous @ np.asarray(modelview) @ np.asarray(projection)
    valid = (clip[:, 3] > 1e-12) & np.isfinite(clip).all(axis=1)
    ndc = np.zeros((len(points), 3))
    ndc[valid] = clip[valid, :3] / clip[valid, 3, None]
    valid &= (np.abs(ndc) <= 1 + 1e-9).all(axis=1)
    screen = np.column_stack(((ndc[:, 0] + 1) * width / 2,
                              (1 - ndc[:, 1]) * height / 2))
    return screen, valid


def polygon_mask(screen, polygon):
    """射线法批量判断套索内部，边界点也选中；允许凹多边形。"""
    polygon = np.asarray(polygon, dtype=float)
    result = np.zeros(len(screen), dtype=bool)
    if len(polygon) < 3:
        return result
    bounds = ((screen >= polygon.min(axis=0)) & (screen <= polygon.max(axis=0))).all(axis=1)
    ids = np.flatnonzero(bounds)
    x, y = screen[ids].T
    inside = np.zeros(len(ids), dtype=bool)
    boundary = inside.copy()
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        dx, dy = b - a
        cross = (x - a[0]) * dy - (y - a[1]) * dx
        boundary |= ((np.abs(cross) <= 1e-7 * max(1, abs(dx) + abs(dy))) &
                     (x >= min(a[0], b[0])) & (x <= max(a[0], b[0])) &
                     (y >= min(a[1], b[1])) & (y <= max(a[1], b[1])))
        if dy != 0:
            inside ^= ((a[1] > y) != (b[1] > y)) & (x < dx * (y - a[1]) / dy + a[0])
    result[ids] = inside | boundary
    return result


from .scene_planes import RoofPlanes
