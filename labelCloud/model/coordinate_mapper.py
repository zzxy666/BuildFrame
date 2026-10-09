"""正射像素坐标统一入口；缓存索引始终对应原始 LAS 点序。"""
from dataclasses import dataclass
import numpy as np


@dataclass
class PixelMask:
    data: np.ndarray
    col: int
    row: int


class CoordinateMapper:
    CHUNK = 250000

    def __init__(self, transform, width, height, lidar_crs, rgb_crs):
        from pyproj import CRS, Transformer
        if lidar_crs is None or rgb_crs is None:
            raise ValueError("LiDAR 或 GeoTIFF 缺少 CRS，已禁止映射；请先为源数据提供正确 CRS。")
        self.lidar_crs, self.rgb_crs = CRS(lidar_crs), CRS(rgb_crs)
        # 只转换水平坐标，避免复合垂直基准触发无关的高程转换。
        def horizontal(crs):
            if crs.is_compound:
                return crs.sub_crs_list[0].to_2d()
            return crs.to_2d()
        self.forward = Transformer.from_crs(horizontal(self.lidar_crs), horizontal(self.rgb_crs), always_xy=True)
        self.backward = Transformer.from_crs(horizontal(self.rgb_crs), horizontal(self.lidar_crs), always_xy=True)
        self.transform, self.inverse = transform, ~transform
        self.width, self.height = int(width), int(height)
        self.rows = self.cols = self.valid = self.pixels = None
        self.sorted_ids = self.sorted_keys = None

    def world_to_pixel(self, x, y):
        """输入 RGB CRS 世界坐标，输出连续像素坐标；整数为像素边界。"""
        return self.inverse * (np.asarray(x), np.asarray(y))

    def pixel_to_world(self, col, row):
        """像素中心请传 col+0.5、row+0.5，不隐式添加半像素。"""
        return self.transform * (np.asarray(col), np.asarray(row))

    def lidar_xy_to_pixel(self, x, y):
        return self.world_to_pixel(*self.forward.transform(x, y))

    def pixel_to_lidar_xy(self, col, row):
        return self.backward.transform(*self.pixel_to_world(col, row))

    def build(self, count, chunks, cancel=None):
        self.rows = np.full(count, -1, np.int32)
        self.cols = np.full(count, -1, np.int32)
        self.pixels = np.empty((count, 2), np.float32)
        self.valid = np.zeros(count, bool)
        start = 0
        for x, y in chunks:
            if cancel is not None and cancel.is_set(): raise InterruptedError("映射已取消")
            col, row = self.lidar_xy_to_pixel(x, y)
            stop = start+len(col)
            if stop > count: raise ValueError("LiDAR 点数发生变化")
            self.pixels[start:stop] = np.column_stack((col,row))
            good = np.isfinite(col)&np.isfinite(row)&(col>=0)&(row>=0)&(col<self.width)&(row<self.height)
            self.valid[start:stop] = good
            ids = np.flatnonzero(good)+start
            self.cols[ids] = np.floor(col[good]).astype(np.int32)
            self.rows[ids] = np.floor(row[good]).astype(np.int32)
            start = stop
        if start != count: raise ValueError("LiDAR 原始点数与标注点数不匹配")
        valid_ids = np.flatnonzero(self.valid)
        keys = self.rows[valid_ids].astype(np.int64)*self.width+self.cols[valid_ids]
        order = np.argsort(keys, kind="stable")
        self.sorted_ids, self.sorted_keys = valid_ids[order], keys[order]

    def build_las(self, path, cancel=None):
        import laspy
        from .scene_session import las_backend
        with laspy.open(path, laz_backend=las_backend()) as reader:
            self.build(reader.header.point_count,
                       ((np.asarray(chunk.x),np.asarray(chunk.y)) for chunk in reader.chunk_iterator(self.CHUNK)), cancel)

    def lidar_to_pixel(self, point_ids):
        return self.pixels[np.asarray(point_ids, np.int64)]

    def pixel_point_ids(self, col, row):
        if not (0 <= col < self.width and 0 <= row < self.height): return np.empty(0,np.int64)
        key = int(row)*self.width+int(col)
        lo, hi = np.searchsorted(self.sorted_keys, [key,key], side="left")
        hi = np.searchsorted(self.sorted_keys,key,side="right")
        return self.sorted_ids[lo:hi]

    def polygon_mask(self, vertices):
        from affine import Affine
        from rasterio.features import rasterize
        vertices = np.asarray(vertices,float)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices)<3 or not np.isfinite(vertices).all():
            raise ValueError("Polygon 至少需要 3 个有效顶点")
        low = np.maximum(np.floor(vertices.min(axis=0)).astype(int),0)
        high = np.minimum(np.ceil(vertices.max(axis=0)).astype(int),[self.width,self.height])
        width,height = high-low
        if width <= 0 or height <= 0: raise ValueError("Polygon 不在影像范围内")
        if int(width)*int(height)>64000000: raise ValueError("Polygon 超过 6400 万像素，请缩小到屋面或局部区域")
        ring=vertices.tolist()+[vertices[0].tolist()]
        data=rasterize([({"type":"Polygon","coordinates":[ring]},1)],out_shape=(int(height),int(width)),
                       transform=Affine.translation(*low),dtype="uint8")>0
        if not data.any(): raise ValueError("Polygon 未覆盖任何像素中心")
        return PixelMask(data,int(low[0]),int(low[1]))

    def pixel_mask_to_candidate_point_ids(self, mask, cancel=None):
        if isinstance(mask,np.ndarray): mask=PixelMask(mask,0,0)
        parts=[]; height,width=mask.data.shape
        for start in range(0,len(self.valid),self.CHUNK):
            if cancel is not None and cancel.is_set(): raise InterruptedError("映射已取消")
            stop=min(start+self.CHUNK,len(self.valid))
            row=self.rows[start:stop].astype(np.int64)-mask.row
            col=self.cols[start:stop].astype(np.int64)-mask.col
            ids=np.flatnonzero(self.valid[start:stop]&(row>=0)&(col>=0)&(row<height)&(col<width))
            parts.append(ids[mask.data[row[ids],col[ids]]]+start)
        return np.concatenate(parts) if parts else np.empty(0,np.int64)
