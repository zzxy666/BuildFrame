"""来源无关的局部候选几何过滤；输入输出始终为原始点 ID，不写标签。"""
from dataclasses import dataclass, field
from collections import deque
import logging
import time
import numpy as np
from scipy.spatial import cKDTree
from .local_plane_growth import GrowthOptions, ransac_seed_mask
from .plane_math import least_squares_plane


@dataclass(frozen=True)
class RefineSettings:
    distance_m: float = .15
    neighbor_radius_m: float = .6
    normal_angle_deg: float = 10.
    use_normals: bool = True
    use_connectivity: bool = True
    component_mode: str = 'main'
    min_points: int = 20
    min_component_points: int = 10
    min_inlier_ratio: float = .3
    ransac_trials: int = 300
    random_seed: int = 0


@dataclass
class PlaneRefineResult:
    input_point_ids: np.ndarray
    ransac_inlier_ids: np.ndarray = field(default_factory=lambda:np.empty(0,np.int64))
    distance_filtered_ids: np.ndarray = field(default_factory=lambda:np.empty(0,np.int64))
    normal_filtered_ids: np.ndarray = field(default_factory=lambda:np.empty(0,np.int64))
    connectivity_filtered_ids: np.ndarray = field(default_factory=lambda:np.empty(0,np.int64))
    final_point_ids: np.ndarray = field(default_factory=lambda:np.empty(0,np.int64))
    plane_normal: object = None
    plane_d: object = None
    centroid: object = None
    input_count: int = 0
    ransac_inlier_count: int = 0
    final_count: int = 0
    inlier_ratio: float = 0.
    rms_distance: float = 0.
    median_distance: float = 0.
    p95_distance: float = 0.
    success: bool = False
    warning: str = ''
    failure_reason: str = ''
    timings_ms: dict = field(default_factory=dict)


def plane_geometry(xyz_native,scale,plane_id=0,method='svd_after_manual_confirm'):
    """方程在原生 XYZ 中；拟合/残差按物理距离，兼容 XY/Z 单位不同。"""
    xyz=np.asarray(xyz_native,np.float64);scale=np.broadcast_to(np.asarray(scale,float),(3,))
    if not np.isfinite(scale).all() or (scale<=0).any(): raise ValueError('无效单位换算')
    origin=xyz.mean(axis=0)
    metric=(xyz-origin)*scale
    normal,center=least_squares_plane(metric)
    residual=np.abs((metric-center)@normal)
    native_normal=normal*scale;native_normal/=np.linalg.norm(native_normal)
    centroid=origin+center/scale
    return dict(plane_id=int(plane_id),normal=native_normal.tolist(),d=float(-native_normal@centroid),
                centroid=centroid.tolist(),point_count=len(xyz),rms_m=float(np.sqrt(np.mean(residual**2))),
                median_m=float(np.median(residual)),p95_m=float(np.quantile(residual,.95)),
                fit_method=method,geometry_dirty=False,unit_to_m=scale.tolist(),
                coordinate_frame='original LAS native XYZ',warning='')


class GeometryRefiner:
    def __init__(self,point_reader,unit_to_m,normals_reader=None):
        self.point_reader=point_reader
        self.scale=np.broadcast_to(np.asarray(unit_to_m,float),(3,)).copy()
        self.normals_reader=normals_reader

    def refine_plane(self,candidate_point_ids,settings=RefineSettings(),seed_point_ids=None,cancel=None):
        if len(candidate_point_ids)>200000:
            return PlaneRefineResult(np.asarray(candidate_point_ids,np.int64),input_count=len(candidate_point_ids),
                failure_reason='候选超过 20 万点；请缩小到单个屋面，禁止对整场景运行 RANSAC')
        ids=np.unique(np.asarray(candidate_point_ids,np.int64))
        result=PlaneRefineResult(ids,input_count=len(ids));start=time.perf_counter();warnings=[]
        def check():
            if cancel is not None and cancel.is_set(): raise InterruptedError('几何过滤已取消')
        def mark(name,t): result.timings_ms[name]=(time.perf_counter()-t)*1000
        try:
            check()
            if len(ids)<settings.min_points or settings.min_points<3: raise ValueError('候选点不足，请扩大选择或降低最少候选点数')
            values=[settings.distance_m,settings.neighbor_radius_m,*self.scale]
            if not np.isfinite(values).all() or min(values)<=0: raise ValueError('距离和坐标单位必须为正数')
            if not 0<settings.normal_angle_deg<90 or settings.min_component_points<1 or settings.component_mode not in ('main','coplanar'):
                raise ValueError('无效法向角或连通性设置')
            xyz=np.asarray(self.point_reader(ids),np.float64)
            if xyz.shape!=(len(ids),3) or not np.isfinite(xyz).all(): raise ValueError('候选含 NaN/Inf 或点数不匹配')
            # 统一为 XY 原生单位；Z 如有不同单位只换算一次。平面计算不扫描全场。
            origin=xyz.mean(axis=0);ratio=self.scale/self.scale[0]
            points=(xyz-origin)*ratio
            distance=settings.distance_m/self.scale[0];radius=settings.neighbor_radius_m/self.scale[0]
            least_squares_plane(points)
            t=time.perf_counter()
            options=GrowthOptions(plane_distance=distance,ransac_distance=distance,
                ransac_iterations=settings.ransac_trials,ransac_min_ratio=settings.min_inlier_ratio)
            try:
                inliers=ransac_seed_mask(points,options,cancel,minimum=settings.min_points,
                    random_seed=settings.random_seed,require_majority=False)
            except ValueError as exc:
                raise ValueError('RANSAC 支持不足；候选可能包含多个平面，请缩小 mask 或人工选择。 '+str(exc)) from exc
            normal,center=least_squares_plane(points[inliers])
            result.ransac_inlier_ids=ids[inliers];result.ransac_inlier_count=int(inliers.sum())
            result.inlier_ratio=float(inliers.mean());mark('ransac',t);check()
            if result.inlier_ratio<.65: warnings.append('候选可能包含多个平面；当前仅拟合主要平面')
            t=time.perf_counter()
            residual=np.abs((points-center)@normal)
            keep=residual<=distance;result.distance_filtered_ids=ids[keep];mark('distance',t)
            t=time.perf_counter()
            if settings.use_normals:
                normals=None if self.normals_reader is None else self.normals_reader(ids)
                if normals is None: warnings.append('当前没有可用法向量，已跳过法向过滤')
                else:
                    normals=np.asarray(normals,float)
                    if normals.shape!=(len(ids),3): raise ValueError('法向缓存索引不匹配')
                    lengths=np.linalg.norm(normals,axis=1)
                    valid=np.isfinite(normals).all(axis=1)&(lengths>1e-10)
                    cos=np.zeros(len(ids));cos[valid]=np.abs(normals[valid]@normal)/lengths[valid]
                    keep &= ~valid | (cos>=np.cos(np.deg2rad(settings.normal_angle_deg)))
                    if not valid.all(): warnings.append(f'{int((~valid).sum())} 点无有效法向，保留距离判断')
            result.normal_filtered_ids=ids[keep];mark('normal',t);check()
            t=time.perf_counter();local=np.flatnonzero(keep)
            if settings.use_connectivity and len(local):
                tree=cKDTree(points[local]);seen=np.zeros(len(local),bool);components=[]
                for first in range(len(local)):
                    if seen[first]: continue
                    seen[first]=True;queue=deque([first]);component=[first]
                    while queue:
                        check()
                        batch=[queue.popleft() for _ in range(min(32,len(queue)))]
                        for group in tree.query_ball_point(points[local[batch]],radius):
                            group=np.asarray(group,np.int64);added=group[~seen[group]]
                            seen[added]=True;queue.extend(added.tolist());component.extend(added.tolist())
                        if seen.all(): break
                    part=local[np.asarray(component,np.int64)]
                    if len(part)>=settings.min_component_points: components.append(part)
                    if seen.all(): break
                if not components: raise ValueError('没有达到最小点数的连通分量；可调整邻接半径或最小分量点数')
                if settings.component_mode=='coplanar': local=np.concatenate(components)
                else:
                    seeds=np.asarray(seed_point_ids if seed_point_ids is not None else [],np.int64)
                    def rank(part): return (int(np.isin(ids[part],seeds).sum()),int(inliers[part].sum()),len(part),-int(ids[part].min()))
                    local=max(components,key=rank)
            result.connectivity_filtered_ids=np.sort(ids[local]);result.final_point_ids=result.connectivity_filtered_ids.copy()
            result.final_count=len(local);mark('connectivity',t)
            if len(local)<3: raise ValueError('过滤后点数不足，保留原候选')
            # 预览方程仍是 RANSAC 全内点精拟合结果；最终确认会重新拟合最终 GT。
            n=normal*ratio;n/=np.linalg.norm(n);centroid=origin+center/ratio
            result.plane_normal=n;result.plane_d=float(-n@centroid);result.centroid=centroid
            residual_m=residual[local]*self.scale[0]
            result.rms_distance=float(np.sqrt(np.mean(residual_m**2)))
            result.median_distance=float(np.median(residual_m));result.p95_distance=float(np.quantile(residual_m,.95))
            result.success=True;result.warning='；'.join(warnings)
        except ValueError as exc:
            result.failure_reason=str(exc)
        mark('total',start)
        logging.debug('Geometry Refine input=%s final=%s timings_ms=%s',len(ids),result.final_count,result.timings_ms)
        return result
