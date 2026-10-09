"""按 original_point_id 读取原始 LAS/LAZ 精度；与 OpenGL/LOD 完全无关。"""
from pathlib import Path
import threading
import laspy
import numpy as np
from .scene_session import las_backend


class OriginalPointReader:
    def __init__(self,path):
        self.path=Path(path);self.lock=threading.Lock();self.cache={}
        stat=self.path.stat();self.fingerprint=(stat.st_size,stat.st_mtime_ns)

    def read(self,ids):
        ids=np.asarray(ids,np.int64)
        with self.lock:
            stat=self.path.stat()
            if (stat.st_size,stat.st_mtime_ns)!=self.fingerprint:
                raise ValueError('原始 LAS/LAZ 已改变，请重新加载，避免原始点 ID 错位')
            missing=np.asarray(sorted(set(map(int,ids))-self.cache.keys()),np.int64)
            if len(missing):
                if len(self.cache)+len(missing)>200000: self.cache.clear();missing=np.unique(ids)
                with laspy.open(self.path,laz_backend=las_backend()) as reader:
                    if missing[0]<0 or missing[-1]>=reader.header.point_count: raise ValueError('原始点 ID 越界')
                    start=0
                    while start<len(missing):
                        stop=start+1
                        while stop<len(missing) and missing[stop]-missing[stop-1]<=32 and missing[stop]-missing[start]<8192: stop+=1
                        group=missing[start:stop];reader.seek(int(group[0]))
                        part=reader.read_points(int(group[-1]-group[0]+1))
                        local=group-group[0]
                        xyz=np.column_stack((np.asarray(part.x)[local],np.asarray(part.y)[local],np.asarray(part.z)[local]))
                        self.cache.update(zip(map(int,group),xyz));start=stop
            return np.asarray([self.cache[int(i)] for i in ids],np.float64).reshape(-1,3)
