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

        las = laspy.read(path)

        # XYZ
        points = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)

        # Colors 
        colors = None
        if hasattr(las, "red"):
            red = las.red / 65535.0
            green = las.green / 65535.0
            blue = las.blue / 65535.0
            colors_temp = np.vstack([red, green, blue]).T.astype(np.float32)

            # 如果颜色全为0（或全相同且为0），认为无有效颜色
            if np.all(colors_temp == 0):
                colors = None  # 强制触发无颜色逻辑 → 高程伪彩
                logging.info("LAS文件检测到无效RGB（全黑），将使用高程伪彩显示")
            else:
                colors = colors_temp
        else:
            logging.info("LAS文件无RGB通道，将使用高程伪彩显示")

        return points, colors

    def write_point_cloud(self, path: Path, pointcloud):
        """
        Optional: If you need saving .las, implement it.
        Otherwise leave it empty.
        """
        raise NotImplementedError("Writing LAS is not supported yet.")
