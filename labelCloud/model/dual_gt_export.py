"""双 GT 校验与分块导出；最终目录只在全部写入成功后出现。"""
from collections import OrderedDict, Counter
from datetime import datetime, timezone
from pathlib import Path
import logging
import os
import shutil
import tempfile
import time
import numpy as np
from .dual_gt import rgb_identity
from .scene_session import atomic_json


class DualGTValidationError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__(validation_text(report))


def validation_text(report):
    return '\n'.join([
        f"LiDAR Planes: {report['lidar_planes']} | RGB confirmed: {report['rgb_planes']} | Complete: {report['complete_planes']}",
        f"Missing RGB: {report['missing_rgb_planes']}", f"Missing LiDAR: {report['missing_lidar_planes']}",
        f"RGB Needs Review: {report['needs_review_planes']}",
        f"Cross-modal review（警告）: {report['cross_modal_review_planes']}",
        f"Dirty PlaneGeometry: {report['dirty_geometry']}",
        *[f"RGB GT Conflict: Plane {a} ↔ Plane {b}; overlap pixels: {n}" for a, b, n in report['conflicts']],
        *report['errors']])


class MaskBlocks:
    """局部 mask 有界 LRU；只分配当前 512×512 输出块。"""
    def __init__(self, store, records):
        self.store, self.records = store, records
        self.cache = OrderedDict(); self.bytes = 0

    def mask(self, fragment):
        key = fragment['uuid']
        if key in self.cache:
            self.cache.move_to_end(key); return self.cache[key]
        value = self.store.read(fragment)
        while self.cache and self.bytes + value.nbytes > 64*1024*1024:
            _, old = self.cache.popitem(last=False); self.bytes -= old.nbytes
        if value.nbytes <= 64*1024*1024:
            self.cache[key] = value; self.bytes += value.nbytes
        return value

    def planes(self, window, confirmed_only=False, records=None):
        x, y, w, h = map(int, (window.col_off, window.row_off, window.width, window.height))
        for record in self.records if records is None else records:
            if confirmed_only and record['rgb_gt_status'] != 'CONFIRMED': continue
            union = None
            for fragment in record['fragments']:
                c, r = fragment['col0'], fragment['row0']
                left, top = max(x, c), max(y, r)
                right, bottom = min(x+w, c+fragment['width']), min(y+h, r+fragment['height'])
                if right <= left or bottom <= top: continue
                if union is None: union = np.zeros((h, w), bool)
                union[top-y:bottom-y, left-x:right-x] |= self.mask(fragment)[top-r:bottom-r, left-c:right-c]
            if union is not None: yield record['plane_id'], union


class DualGTExporter:
    def __init__(self, model, rgb_path, cancel=None):
        self.model, self.rgb_path, self.cancel = model, Path(rgb_path), cancel

    def check_cancel(self):
        if self.cancel is not None and self.cancel.is_set(): raise InterruptedError('双 GT 导出已取消')

    @staticmethod
    def windows(ds):
        from rasterio.windows import Window
        for row in range(0, ds.height, 512):
            for col in range(0, ds.width, 512):
                yield Window(col, row, min(512, ds.width-col), min(512, ds.height-row))

    def validate(self, strict=True):
        import rasterio
        start = time.perf_counter(); model = self.model; model.session.verify_source()
        records = model.dual_gt.summaries()
        report = dict(errors=[], conflicts=[], missing_rgb_planes=[], missing_lidar_planes=[],
            needs_review_planes=[], cross_modal_review_planes=[], lidar_planes=0, rgb_planes=0,
            complete_planes=0, rgb_pixel_counts={}, dirty_geometry=sum(bool(g.get('geometry_dirty')) for g in model.session_metadata.get('plane_geometry',{}).values()))
        identity = rgb_identity(self.rgb_path)
        report['source_identity'] = identity
        if identity['crs'] is None: report['errors'].append('RGB CRS 缺失')
        used_uids = set(); seen_fragments = {}
        active = model.dual_gt.state['active']; archived = model.dual_gt.state['archived']
        for key, record in active.items():
            pid, uid = record['plane_id'], record['plane_uid']
            if str(pid) != key or not 0 < pid <= np.iinfo(np.uint32).max or pid not in model.plane_ids:
                report['errors'].append(f'非法 active Plane ID: {key}')
            if uid in used_uids or uid in archived: report['errors'].append(f'Plane {pid}: deleted/duplicate plane_uid')
            used_uids.add(uid)
            if record['rgb_gt_status'] not in ('MISSING','PENDING','CONFIRMED','NEEDS_REVIEW'):
                report['errors'].append(f'Plane {pid}: RGB status 非法')
            for f in record['fragments']:
                try:
                    if f['source'] != identity: raise ValueError('GeoTIFF fingerprint/grid/source 不匹配')
                    c, r, w, h = (f[k] for k in ('col0','row0','width','height'))
                    if any(type(v) is not int for v in (c,r,w,h)) or min(c,r)<0 or min(w,h)<1 or c+w>identity['width'] or r+h>identity['height']:
                        raise ValueError('fragment 越界或 window 非法')
                    if f['uuid'] in seen_fragments and seen_fragments[f['uuid']] != f: raise ValueError('fragment UUID 内容不一致')
                    seen_fragments[f['uuid']] = f
                    model.dual_gt.store.read(f)
                except (OSError, ValueError, KeyError) as exc:
                    report['errors'].append(f'Plane {pid}: {exc}')
        pixels = Counter(); conflicts = Counter()
        if not report['errors']:
            blocks = MaskBlocks(model.dual_gt.store, records)
            confirmed = {r['plane_id'] for r in records if r['rgb_gt_status']=='CONFIRMED'}
            by_id = {r['plane_id']:r for r in records}
            with rasterio.open(self.rgb_path) as ds:
                for window in self.windows(ds):
                    self.check_cancel()
                    valid = np.all(ds.read_masks(window=window) > 0, axis=0)
                    occupied = np.zeros(valid.shape,bool);previous=[]
                    for pid, mask in blocks.planes(window):
                        mask &= valid
                        if pid in confirmed: pixels[pid] += int(mask.sum())
                        if np.any(occupied[mask]):
                            # 冲突时重取先前局部块，不同时保留每个 Plane 的块数组。
                            for other, other_mask in blocks.planes(window,records=previous):
                                n = int(np.count_nonzero(mask & other_mask))
                                if n: conflicts[tuple(sorted((pid,other)))] += n
                        occupied |= mask;previous.append(by_id[pid])
        lidar_ids = {pid for pid, n in model.plane_counts.items() if pid>0 and n>0}
        rgb_ids = {pid for pid, n in pixels.items() if n>0}
        # 没有点也没有 mask 的空 Plane 不是导出对象。
        relevant = lidar_ids | {r['plane_id'] for r in records if r['fragments']}
        report.update(lidar_planes=len(lidar_ids), rgb_planes=len(rgb_ids), complete_planes=len(lidar_ids & rgb_ids),
            missing_rgb_planes=sorted(relevant-rgb_ids), missing_lidar_planes=sorted(relevant-lidar_ids),
            needs_review_planes=sorted(r['plane_id'] for r in records if r['rgb_gt_status']=='NEEDS_REVIEW'),
            cross_modal_review_planes=sorted(r['plane_id'] for r in records if r['cross_modal_status']=='NEEDS_REVIEW'),
            conflicts=[(a,b,n) for (a,b),n in sorted(conflicts.items())], rgb_pixel_counts=dict(pixels))
        report['complete_dual_gt'] = not (report['missing_rgb_planes'] or report['missing_lidar_planes'] or report['needs_review_planes'])
        for pid in lidar_ids:
            if str(pid) not in active: report['errors'].append(f'Plane {pid}: 缺少 active plane_uid')
        report['ok'] = not (report['errors'] or report['conflicts']) and (not strict or report['complete_dual_gt'])
        logging.debug('Dual GT overlap/validation %.1f ms; LiDAR points=%s planes=%s; RGB planes=%s fragments=%s pixels=%s',
            (time.perf_counter()-start)*1000, len(model.labels), len(lidar_ids), len(rgb_ids), len(seen_fragments), sum(pixels.values()))
        return report

    def export(self, destination, strict=True):
        import rasterio
        from rasterio.enums import Resampling
        from .roof_planes import plane_color
        start = time.perf_counter(); destination = Path(destination).resolve()
        if destination.exists(): raise ValueError('目标目录已存在，请选择新的双 GT 目录')
        report = self.validate(strict)
        if not report['ok']: raise DualGTValidationError(report)
        # 使用 P2 的最终点拟合与原有工作保存，不从 RGB 猜测几何。
        self.model.save()
        identity = rgb_identity(self.rgb_path)
        if identity != report['source_identity']: raise ValueError('校验后 RGB 来源发生变化')
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix='.'+destination.name+'-', dir=destination.parent)).resolve()
        try:
            for name in ('lidar','rgb','metadata'): (temporary/name).mkdir()
            scene = self.model.path.stem
            records = self.model.dual_gt.summaries()
            blocks = MaskBlocks(self.model.dual_gt.store, records)
            raster_start = time.perf_counter()
            label_path = temporary/'rgb'/f'{scene}_plane_id.tif'
            valid_path = temporary/'rgb'/f'{scene}_valid_mask.tif'
            with rasterio.open(self.rgb_path) as ds:
                profile = dict(driver='GTiff', width=ds.width, height=ds.height, crs=ds.crs, transform=ds.transform,
                    count=1, dtype='uint32', tiled=True, blockxsize=512, blockysize=512, compress='DEFLATE', BIGTIFF='IF_SAFER', nodata=0)
                with rasterio.open(label_path, 'w', **profile) as labels, rasterio.open(valid_path, 'w', **dict(profile,dtype='uint8')) as valid:
                    for window in self.windows(ds):
                        self.check_cancel()
                        out = np.zeros((int(window.height),int(window.width)), np.uint32)
                        for pid, mask in blocks.planes(window, confirmed_only=True): out[mask] = pid
                        out[~np.all(ds.read_masks(window=window)>0, axis=0)] = 0
                        labels.write(out,1,window=window); valid.write((out>0).astype(np.uint8),1,window=window)
            # 仅预览使用小图与颜色，正式 ID 永远保持 uint32 原网格。
            with rasterio.open(label_path) as ds:
                factor = max(ds.width/1600,ds.height/1600,1)
                tiny = ds.read(1,out_shape=(max(1,int(ds.height/factor)),max(1,int(ds.width/factor))),resampling=Resampling.nearest)
                ids, inverse = np.unique(tiny,return_inverse=True)
                colors = (np.asarray([plane_color(int(pid)) for pid in ids])*255).astype(np.uint8)
                image = colors[inverse].reshape((*tiny.shape,3))
            self._write_preview(temporary/'rgb'/f'{scene}_plane_preview.png',image)
            logging.debug('Dual GT raster write %.1f ms',(time.perf_counter()-raster_start)*1000)
            laz_start = time.perf_counter()
            self.model.session.export(self.model.labels,temporary/'lidar'/f'{scene}_gt.laz',self.cancel)
            logging.debug('Dual GT LAZ export %.1f ms',(time.perf_counter()-laz_start)*1000)
            meta_start = time.perf_counter()
            geometry = self.model.session_metadata.get('plane_geometry',{})
            plane_summaries = [dict(r,rgb_pixel_count=report['rgb_pixel_counts'].get(r['plane_id'],0),
                rgb_fragments=r['fragments'], annotation_source=sorted({f['annotation_source'] for f in r['fragments']}),
                geometry=geometry.get(str(r['plane_id']))) for r in records if r['fragments'] or r['lidar_point_count']]
            from importlib.metadata import version, PackageNotFoundError
            try: software_version = version('buildframe-pointcloud')
            except PackageNotFoundError: software_version = 'source-checkout'
            manifest = dict(schema_version=1, scene=dict(lidar_source=str(self.model.path.resolve()),lidar_fingerprint=self.model.session.fingerprint,
                rgb_source=str(self.rgb_path.resolve()),rgb_fingerprint=identity['fingerprint']),
                rgb_grid=dict(identity,dtype='uint32',valid_mask_dtype='uint8'),
                lidar=dict(point_count=len(self.model.labels),plane_id_dtype='uint32'),
                label_semantics=dict(lidar_zero='unassigned_or_non_plane',rgb_zero='no confirmed roof-plane label',
                    valid_mask={'0':'ignore / unknown','1':'confirmed roof-plane label'},preview_png='visualization only; not GT'),
                export=dict(timestamp=datetime.now(timezone.utc).isoformat(),mode='strict' if strict else 'partial',software_version=software_version),
                complete_dual_gt=bool(strict and report['complete_dual_gt']), planes=plane_summaries,
                missing_rgb_planes=report['missing_rgb_planes'],missing_lidar_planes=report['missing_lidar_planes'],
                needs_review_planes=report['needs_review_planes'],cross_modal_review_planes=report['cross_modal_review_planes'])
            atomic_json(temporary/'metadata'/'plane_geometry.json',dict(source=self.model.session.fingerprint,planes=geometry))
            atomic_json(temporary/'metadata'/'dual_gt_manifest.json',manifest)
            self._verify_outputs(temporary,scene,report)
            self.model.session.verify_source();self.check_cancel()
            if rgb_identity(self.rgb_path) != identity: raise ValueError('导出期间 RGB 文件发生变化')
            logging.debug('Dual GT metadata %.1f ms',(time.perf_counter()-meta_start)*1000)
            os.rename(temporary,destination)
        finally:
            # 仅删除本函数在目标父目录中新建的临时目录。
            if temporary.exists() and temporary.parent == destination.parent and temporary.name.startswith('.'+destination.name+'-'):
                shutil.rmtree(temporary)
        logging.debug('Dual GT total %.1f ms',(time.perf_counter()-start)*1000)
        return destination

    def _verify_outputs(self, folder, scene, report):
        """发布目录前重新打开检查网格、ID 统计、valid 和点云头。"""
        import rasterio
        import laspy
        from .scene_session import las_backend
        counts=Counter()
        with rasterio.open(self.rgb_path) as src, rasterio.open(folder/'rgb'/f'{scene}_plane_id.tif') as labels, rasterio.open(folder/'rgb'/f'{scene}_valid_mask.tif') as valid:
            for ds,dtype in ((labels,'uint32'),(valid,'uint8')):
                if ds.shape!=src.shape or ds.transform!=src.transform or ds.crs!=src.crs or ds.dtypes!=(dtype,):
                    raise ValueError('导出 Raster 网格/类型校验失败')
            for window in self.windows(src):
                self.check_cancel();data=labels.read(1,window=window)
                if not np.array_equal(valid.read(1,window=window),(data>0).astype(np.uint8)):
                    raise ValueError('导出 valid_mask 校验失败')
                ids,numbers=np.unique(data,return_counts=True)
                counts.update({int(pid):int(n) for pid,n in zip(ids,numbers) if pid>0})
        if dict(counts)!={pid:n for pid,n in report['rgb_pixel_counts'].items() if n>0}:
            raise ValueError('导出 RGB 像素统计校验失败')
        with laspy.open(folder/'lidar'/f'{scene}_gt.laz',laz_backend=las_backend()) as reader:
            if reader.header.point_count!=len(self.model.labels) or reader.header.point_format.dimension_by_name('plane_id').dtype!=np.dtype('uint32'):
                raise ValueError('导出 LiDAR 点数/plane_id 类型校验失败')

    @staticmethod
    def _write_preview(path,image):
        # PNG 不带坐标，不把它当作 label raster；QImage 不依赖 Pillow。
        from PyQt5.QtGui import QImage
        data = np.ascontiguousarray(image)
        result = QImage(data.data,data.shape[1],data.shape[0],data.strides[0],QImage.Format_RGB888)
        if not result.save(str(path),'PNG'): raise OSError('PNG 预览写入失败')
