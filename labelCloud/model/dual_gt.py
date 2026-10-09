"""双 GT 元数据与不可变局部 mask；不从点投影生成二维 GT。"""
from pathlib import Path
from uuid import uuid4
import hashlib
import os
import numpy as np


def rgb_identity(path):
    import rasterio
    path = Path(path).resolve()
    stat = path.stat()
    with path.open('rb') as stream:
        first = stream.read(65536)
        stream.seek(max(0, stat.st_size - 65536))
        digest = hashlib.sha256(first + stream.read()).hexdigest()
    with rasterio.open(path) as ds:
        return dict(path=str(path), fingerprint=dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                    sample_sha256=digest), width=ds.width, height=ds.height,
                    crs=ds.crs.to_wkt() if ds.crs else None, transform=list(ds.transform)[:6])


class RGBGTStore:
    def __init__(self, folder):
        self.folder = Path(folder)

    def path(self, fragment):
        path = (self.folder / fragment['path']).resolve()
        if not path.is_relative_to(self.folder.resolve()) or path.suffix != '.npz':
            raise ValueError('RGB mask 路径不安全')
        return path

    def write(self, mask, source, annotation_source, existing=None):
        data = np.asarray(mask.data)
        col, row = int(mask.col), int(mask.row)
        if col != mask.col or row != mask.row: raise ValueError('RGB mask offset 必须为整数像素')
        if data.dtype != bool or data.ndim != 2 or not data.size or not data.any():
            raise ValueError('RGB mask 必须是非空二维 bool 区域')
        h, w = data.shape
        if min(col, row) < 0 or col+w > source['width'] or row+h > source['height']:
            raise ValueError('RGB fragment 超出原图边界')
        import rasterio
        from rasterio.windows import Window
        # 保留原始 proposal 文件；有效性由源图掩膜约束，不将 nodata 当作 GT。
        with rasterio.open(source['path']) as ds:
            valid_pixels = int(np.count_nonzero(data & np.all(ds.read_masks(window=Window(col,row,w,h))>0,axis=0)))
        if not valid_pixels: raise ValueError('RGB mask 没有有效源图像素（全部为 nodata）')
        uid = uuid4().hex
        if existing is None:
            folder = self.folder / 'rgb_gt'; folder.mkdir(parents=True, exist_ok=True)
            path = folder / (uid + '.npz'); temporary = path.with_suffix('.tmp')
            try:
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, mask=data, col0=col, row0=row)
                    stream.flush(); os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            path = Path(existing)
        result = dict(uuid=uid, path=path.relative_to(self.folder).as_posix(), col0=col, row0=row,
                      width=w, height=h, source=source, annotation_source=annotation_source, valid_pixels=valid_pixels)
        # 小型局部 mask 只算一次校验值，后续载入可检测文件损坏/替换。
        result['sha256'] = hashlib.sha256(data.tobytes()).hexdigest()
        self.read(result)
        return result

    def read(self, fragment):
        with np.load(self.path(fragment), allow_pickle=False) as archive:
            data = archive['mask']
            if (data.dtype != bool or data.shape != (fragment['height'], fragment['width']) or
                    int(archive['col0']) != fragment['col0'] or int(archive['row0']) != fragment['row0']):
                raise ValueError('RGB mask shape/window 不一致')
        if hashlib.sha256(data.tobytes()).hexdigest() != fragment['sha256']:
            raise ValueError('RGB mask 内容校验失败')
        return data


class DualGTRegistry:
    """采用替换式元数据；Undo 只保存引用快照，不复制点云或 mask。"""
    def __init__(self, model):
        self.model = model
        self.store = RGBGTStore(model.session.folder)
        if 'dual_gt' not in model.session_metadata:
            model.session_metadata['dual_gt'] = dict(schema_version=1, active={}, archived={})
        for pid in model.plane_ids - {0}:
            self.ensure(pid)

    @property
    def state(self):
        return self.model.session_metadata['dual_gt']

    def replace(self, active, archived=None):
        self.model.session_metadata['dual_gt'] = dict(schema_version=1, active=active,
            archived=self.state['archived'] if archived is None else archived)

    def ensure(self, pid):
        key = str(int(pid))
        if int(pid) == 0: return None
        if key not in self.state['active']:
            active = dict(self.state['active'])
            active[key] = dict(plane_id=int(pid), plane_uid=uuid4().hex, fragments=[],
                              rgb_gt_status='MISSING', cross_modal_status='UNCHECKED',
                              lidar_gt_status='CONFIRMED' if self.model.plane_counts.get(int(pid),0) else 'MISSING')
            self.replace(active)
        return self.state['active'][key]

    def labels_changed(self, ids):
        for pid in ids - {0}:
            self.ensure(pid)
        active = dict(self.state['active'])
        for pid in ids - {0}:
            record=active[str(pid)]
            active[str(pid)] = dict(record,
                lidar_gt_status='CONFIRMED' if self.model.plane_counts.get(pid,0) else 'MISSING',
                cross_modal_status='NEEDS_REVIEW' if record['fragments'] else record['cross_modal_status'])
        self.replace(active)

    def archive(self, pid):
        active = dict(self.state['active']); record = active.pop(str(pid), None)
        if record is None: return
        archived = dict(self.state['archived']); archived[record['plane_uid']] = record
        self.replace(active, archived)

    def merge(self, source, target, complete):
        a, b = self.ensure(target), self.ensure(source)
        if complete:
            fragments = a['fragments'] + b['fragments']
            self.archive(source)
            active = dict(self.state['active'])
            active[str(target)] = dict(a, fragments=fragments,
                rgb_gt_status='NEEDS_REVIEW' if fragments else 'MISSING', cross_modal_status='NEEDS_REVIEW')
            self.replace(active)
        else:
            # 局部 merge 不能知道应转移哪些二维像素；保留原 mask，双方等待复核。
            active = dict(self.state['active'])
            for record in (a, b):
                active[str(record['plane_id'])] = dict(record,
                    rgb_gt_status='NEEDS_REVIEW' if record['fragments'] else 'MISSING', cross_modal_status='NEEDS_REVIEW')
            self.replace(active)

    def unlabelled_mask(self, mask, source, plane_id=0, allow_current=False, state=None):
        """只在候选局部窗口排除已有像素；不改原 mask、旧 GT 或点标签。"""
        from .coordinate_mapper import PixelMask
        data=np.asarray(mask.data,dtype=bool).copy()
        row,col=int(mask.row),int(mask.col);height,width=data.shape
        before=int(data.sum())
        state=self.state if state is None else state
        for record in state['active'].values():
            if allow_current and record['plane_id']==plane_id: continue
            if record['rgb_gt_status'] not in ('CONFIRMED','NEEDS_REVIEW'): continue
            for fragment in record['fragments']:
                if fragment['source']!=source:
                    raise ValueError('已有 RGB GT 的来源与当前影像不一致，不能安全判断像素占用')
                c,r=fragment['col0'],fragment['row0']
                left,top=max(col,c),max(row,r)
                right,bottom=min(col+width,c+fragment['width']),min(row+height,r+fragment['height'])
                if right<=left or bottom<=top: continue
                occupied=self.store.read(fragment)
                data[top-row:bottom-row,left-col:right-col] &= ~occupied[top-r:bottom-r,left-c:right-c]
        return PixelMask(data,col,row),before-int(data.sum())

    def confirm(self, pid, fragment=None, replace=False, attached=False, annotation=None):
        if pid not in self.model.plane_ids or pid == 0: raise ValueError('请选择有效 Plane')
        record = self.ensure(pid)
        fragments = ([] if replace else record['fragments']) + ([] if fragment is None else [fragment])
        if not fragments: raise ValueError('当前 Plane 没有可确认的 RGB mask')
        if not attached: self.model._remember('rgb_gt', [], np.empty(0, np.uint32))
        active = dict(self.state['active'])
        active[str(pid)] = dict(record, fragments=fragments, rgb_gt_status='CONFIRMED', cross_modal_status='CONFIRMED')
        self.replace(active); self.model.touch()
        if annotation is not None: self.model.attach_rgb_annotation(annotation)

    def summaries(self):
        result = []
        for key, record in self.state['active'].items():
            count = self.model.plane_counts.get(int(key), 0)
            result.append(dict(record, lidar_point_count=count,
                               lidar_gt_status='CONFIRMED' if count else 'MISSING'))
        return result
