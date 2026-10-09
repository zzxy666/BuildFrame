"""保留完整地理参考，按窗口/概览读取，不加载整幅原始分辨率影像。"""
from pathlib import Path
import threading
import numpy as np


class Orthophoto:
    def __init__(self,path):
        import rasterio
        self.path=Path(path).resolve()
        self.dataset=rasterio.open(self.path)
        self.lock=threading.Lock()
        ds=self.dataset
        self.width,self.height=ds.width,ds.height
        self.crs,self.transform=ds.crs,ds.transform
        self.bounds,self.resolution,self.nodata=ds.bounds,ds.res,ds.nodata
        stat=self.path.stat()
        self.fingerprint={"size":stat.st_size,"mtime_ns":stat.st_mtime_ns}

    def read_window(self,bounds=None,max_size=1600):
        from rasterio.windows import Window
        from rasterio.enums import Resampling, ColorInterp
        if bounds is None: bounds=(0,0,self.width,self.height)
        left,top,right,bottom=bounds
        left=max(0,int(np.floor(left)));top=max(0,int(np.floor(top)))
        right=min(self.width,int(np.ceil(right)));bottom=min(self.height,int(np.ceil(bottom)))
        if right<=left or bottom<=top: raise ValueError("视窗不在影像内")
        factor=max((right-left)/max_size,(bottom-top)/max_size,1)
        w=max(1,int(np.ceil((right-left)/factor)));h=max(1,int(np.ceil((bottom-top)/factor)))
        with self.lock:
            ds=self.dataset
            colors=list(ds.colorinterp)
            if all(c in colors for c in (ColorInterp.red,ColorInterp.green,ColorInterp.blue)):
                bands=[colors.index(c)+1 for c in (ColorInterp.red,ColorInterp.green,ColorInterp.blue)]
            else: bands=[1,2,3] if ds.count>=3 else [1,1,1]
            data=ds.read(bands,window=Window(left,top,right-left,bottom-top),out_shape=(3,h,w),
                         masked=True,resampling=Resampling.bilinear)
        valid=~np.ma.getmaskarray(data).any(axis=0)
        raw=data.filled(0)
        if raw.dtype==np.uint8: rgb=raw
        elif np.issubdtype(raw.dtype,np.integer):
            rgb=np.clip(raw.astype(float)*255/np.iinfo(raw.dtype).max,0,255).astype(np.uint8)
        else: rgb=np.clip(raw*255,0,255).astype(np.uint8)
        rgba=np.empty((h,w,4),np.uint8)
        rgba[:,:,:3]=np.moveaxis(rgb,0,-1);rgba[:,:,3]=valid*255
        return rgba,(left,top,right,bottom)

    def close(self):
        with self.lock: self.dataset.close()
