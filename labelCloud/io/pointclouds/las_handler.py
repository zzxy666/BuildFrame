import numpy as np
import laspy
from pathlib import Path
import logging

from . import BasePointCloudHandler

class LASHandler(BasePointCloudHandler):
    """
    Point cloud handler for .las and .laz files.
    Only loads XYZ (and optional RGB if exists).
    """

    EXTENSIONS = {".las", ".laz"}

    def read_point_cloud(self, path: Path):
        logging.info(f"Reading LAS file: {path}")

        # 与 Roof Plane 保存共用解压器选择，支持新版压缩的 LAZ。
        from ...model.scene_session import las_backend
        # 分块解压，不在内存中同时保留整份 LAS 点记录和多份 XYZ/RGB。
        with laspy.open(path, laz_backend=las_backend()) as reader:
            points = np.empty((reader.header.point_count, 3), np.float64)
            has_rgb = "red" in reader.header.point_format.dimension_names
            colors = np.empty(points.shape, np.float32) if has_rgb else None
            offset = 0
            for chunk in reader.chunk_iterator(250000):
                end = offset + len(chunk)
                for column, dim in enumerate(("x", "y", "z")):
                    points[offset:end, column] = chunk[dim]
                if has_rgb:
                    for column, dim in enumerate(("red", "green", "blue")):
                        colors[offset:end, column] = chunk[dim] / 65535.
                offset = end
            if has_rgb and not np.any(colors):
                colors = None

        return points, colors

    def write_point_cloud(self, path: Path, pointcloud):
        """
        Optional: If you need saving .las, implement it.
        Otherwise leave it empty.
        """
        raise NotImplementedError("Writing LAS is not supported yet.")
