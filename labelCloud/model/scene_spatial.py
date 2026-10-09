"""一个场景一个空间索引；只计算当前 frontier 所需的局部法向。"""
import logging
import time
import numpy as np
from scipy.spatial import cKDTree
from .local_plane_growth import ransac_seed_mask


def coordinate_scale(header, units="auto"):
    if units in ("m", "ft", "ftUS"):
        return np.repeat({"m": 1., "ft": .3048, "ftUS": 1200/3937}[units], 3)
    crs = header.parse_crs()
    if crs is None or not crs.is_projected:
        raise ValueError("请在扩展参数中指定原始 XYZ 单位（米/英尺）")
    axes = crs.axis_info
    return np.asarray([axes[0].unit_conversion_factor, axes[1].unit_conversion_factor,
                       axes[2].unit_conversion_factor if len(axes)>2 else axes[0].unit_conversion_factor])


class SceneSpatialIndex:
    def __init__(self, render_points):
        self.points = render_points
        self.tree = None
        self.builds = 0
        self.normals = {}
        self.normal_key = None

    def ensure_tree(self):
        if self.tree is None:
            self.tree = cKDTree(self.points)
            self.builds += 1

    def grow(self, labels, hidden, seeds, target, scale, options, roi=None,
             hidden_revision=0, roi_revision=0, cancel=None, debug=False, preview_levels=False):
        start = time.perf_counter()
        seeds = np.asarray(seeds, np.int64)
        if len(seeds) < 6: raise ValueError("至少选择 6 个相邻种子点")
        if target <= 0: raise ValueError("请先选择或新建 Plane")
        if np.any(hidden[seeds]) or np.any((labels[seeds]!=0)&(labels[seeds]!=target)):
            raise ValueError("种子包含隐藏点或其他 Plane")
        if roi is not None and not roi[seeds].all(): raise ValueError("种子必须位于工作区")
        if options.normal_k < 6 or not 0 < options.normal_angle < 90 or options.neighbor_radius <= 0 or options.plane_distance <= 0:
            raise ValueError("无效扩展参数")
        scale = np.asarray(scale, float)
        if not np.isfinite(scale).all() or np.any(scale <= 0): raise ValueError("无效坐标单位")
        samples = self.points[seeds].astype(float)*scale
        self.last_seed_total = len(seeds)
        rejected = np.empty(0, np.int64)
        if options.robust_fit:
            inliers = ransac_seed_mask(samples, options, cancel)
            rejected = seeds[~inliers]
            seeds = seeds[inliers]; samples = samples[inliers]
        self.last_seed_inliers = len(seeds)
        center = samples.mean(axis=0); delta = samples-center
        eig, axes = np.linalg.eigh(delta.T@delta/len(samples)); normal = axes[:, 0]
        if eig[1] <= 1e-10 or eig[1]/max(eig[2],1e-12)<.01 or eig[0]/eig[1]>.2:
            raise ValueError("种子接近直线或不够平整，请重新选择有面积的屋面种子")
        residual = np.abs(delta@normal)
        if np.quantile(residual,.95)>options.plane_distance or residual.max()>2*options.plane_distance:
            raise ValueError("种子无法稳定拟合同一平面")
        # 小种子树只做连续性验证；全场索引始终复用。
        seed_tree = cKDTree(samples)
        seen = {0}; frontier = [0]
        while frontier:
            nearby = seed_tree.query_ball_point(samples[frontier], options.neighbor_radius)
            added = set(i for group in nearby for i in group) - seen
            seen.update(added); frontier = list(added)
        if len(seen)!=len(seeds): raise ValueError("种子区域不连续")
        self.ensure_tree()
        key = (tuple(scale), options.normal_k, options.neighbor_radius, hidden_revision, roi_revision)
        if key != self.normal_key:
            self.normals.clear(); self.normal_key = key
        if len(self.normals)>100000: self.normals.clear()
        seen = set(map(int,seeds)); frontier = seeds
        seen.update(map(int, rejected))
        result = [seeds[labels[seeds]==0]]
        radius = options.neighbor_radius/scale.min()
        query_seconds = normal_seconds = plane_seconds = 0.
        while len(frontier):
            if cancel is not None and cancel.is_set(): raise InterruptedError("扩展已取消")
            next_frontier = []
            for start_at in range(0,len(frontier),512):
                if cancel is not None and cancel.is_set(): raise InterruptedError("扩展已取消")
                batch = frontier[start_at:start_at+512]
                t=time.perf_counter()
                groups=self.tree.query_ball_point(self.points[batch],radius)
                candidates=[]
                for point_id, group in zip(batch,groups):
                    ids=np.asarray(group,np.int64)
                    # 树在渲染坐标中，精确邻接距离统一按米判断。
                    d=(self.points[ids].astype(float)-self.points[point_id])*scale
                    ids=ids[np.einsum('ij,ij->i',d,d)<=options.neighbor_radius**2]
                    candidates.extend(ids.tolist())
                ids=np.asarray(sorted(set(candidates)-seen),np.int64)
                seen.update(map(int,ids))
                query_seconds+=time.perf_counter()-t
                if not len(ids): continue
                t=time.perf_counter()
                allowed=(labels[ids]==0)&~hidden[ids]
                if roi is not None: allowed &= roi[ids]
                ids=ids[allowed]
                ids=ids[np.abs((self.points[ids].astype(float)*scale-center)@normal)<=options.plane_distance]
                plane_seconds+=time.perf_counter()-t
                if not len(ids): continue
                t=time.perf_counter()
                missing=np.asarray([i for i in ids if int(i) not in self.normals],np.int64)
                # 法向仅按需计算，按小批量矩阵求特征向量。
                for off in range(0,len(missing),1024):
                    part=missing[off:off+1024]
                    k=min(max(options.normal_k*4,24),len(labels))
                    distances, neighbors=self.tree.query(self.points[part],k=k,distance_upper_bound=radius*2)
                    valid=neighbors<len(labels)
                    neighbors=np.minimum(neighbors,len(labels)-1)
                    valid &= ~hidden[neighbors]
                    if roi is not None: valid &= roi[neighbors]
                    local=(self.points[neighbors].astype(float)-self.points[part,None,:])*scale
                    dist2=np.einsum('nki,nki->nk',local,local)
                    valid &= dist2 <= (2*options.neighbor_radius)**2
                    order=np.argsort(np.where(valid,dist2,np.inf),axis=1)[:,:options.normal_k]
                    local=np.take_along_axis(local,order[...,None],axis=1)
                    valid=np.take_along_axis(valid,order,axis=1)
                    count=valid.sum(axis=1)
                    means=(local*valid[...,None]).sum(axis=1)/np.maximum(count[:,None],1)
                    diff=(local-means[:,None,:])*valid[...,None]
                    val,vec=np.linalg.eigh(np.einsum('nki,nkj->nij',diff,diff))
                    stable=(count>=6)&(val[:,1]>1e-10)&(val[:,0]<=.2*val[:,1])
                    for i,n,ok in zip(part,vec[:,:,0],stable):
                        self.normals[int(i)]=n if ok else np.zeros(3)
                normals=np.asarray([self.normals[int(i)] for i in ids])
                ids=ids[np.abs(normals@normal)>=np.cos(np.radians(options.normal_angle))]
                normal_seconds+=time.perf_counter()-t
                if len(ids): next_frontier.append(ids); result.append(ids)
            frontier=np.concatenate(next_frontier) if next_frontier else np.empty(0,np.int64)
        result=np.unique(np.concatenate(result))
        self.last_strict = result.copy()
        self.edge_levels = [np.empty(0, np.int64) for _ in range(11)]
        self.edge_thresholds = [(0., 0.)]
        self.last_edge_count = 0
        if (options.edge_completion or preview_levels) and len(result):
            # 只补一圈：法向在屋脊/檐口常混合，用严格区域的多点支撑替代法向条件。
            # 补选点不能作为下一轮 frontier；高档仅放宽固定主体周边的阈值。
            factors = (1/3, 2/3, 1., 1.5, 2.) if preview_levels else (1.,)*5
            self.edge_thresholds += [(min(options.edge_radius*f, options.neighbor_radius),
                                      min(options.edge_distance*f, options.plane_distance)) for f in factors]
            if preview_levels:
                # 6～10 档放宽原封顶值，仍只依据固定主体补一圈。
                radius5, distance5 = self.edge_thresholds[5]
                self.edge_thresholds += [(radius5+(2*options.neighbor_radius-radius5)*step/5,
                                          distance5+(2*options.plane_distance-distance5)*step/5)
                                         for step in range(1,6)]
            else:
                self.edge_thresholds += [self.edge_thresholds[-1]]*5
            radius_m, distance_m = self.edge_thresholds[-1]
            if radius_m <= 0 or distance_m <= 0:
                raise ValueError("边缘补选阈值必须大于 0")
            strict = np.zeros(len(labels), bool)
            strict[result] = True; strict[seeds] = True
            border_ids = set(seen)
            if preview_levels:
                support = np.flatnonzero(strict)
                for off in range(0,len(support),512):
                    if cancel is not None and cancel.is_set(): raise InterruptedError("扩展已取消")
                    groups = self.tree.query_ball_point(self.points[support[off:off+512]],radius_m/scale.min())
                    for group in groups: border_ids.update(group)
            border = np.asarray(sorted(border_ids), np.int64)
            allowed = ~strict[border] & ~hidden[border] & (labels[border] == 0)
            if len(rejected): allowed &= ~np.isin(border, rejected)
            if roi is not None: allowed &= roi[border]
            border = border[allowed]
            border = border[np.abs((self.points[border].astype(float)*scale-center)@normal) <= distance_m]
            added = []; support_distances = []; plane_distances = []
            for off in range(0, len(border), 512):
                if cancel is not None and cancel.is_set(): raise InterruptedError("扩展已取消")
                batch = border[off:off+512]
                groups = self.tree.query_ball_point(self.points[batch], radius_m/scale.min())
                for point_id, group in zip(batch, groups):
                    ids = np.asarray(group, np.int64)
                    ids = ids[strict[ids]]
                    delta = (self.points[ids].astype(float)-self.points[point_id])*scale
                    distances = np.einsum('ij,ij->i',delta,delta)
                    if len(distances) >= 3:
                        third = float(np.sqrt(np.partition(distances,2)[2]))
                        if third <= radius_m:
                            added.append(point_id); support_distances.append(third)
                            plane_distances.append(abs((self.points[point_id].astype(float)*scale-center)@normal))
            # 缓存各点到第三近支持点和种子平面的距离，调档不重建树或法向。
            added = np.asarray(added,np.int64)
            support_distances = np.asarray(support_distances)
            plane_distances = np.asarray(plane_distances)
            for level,(radius_limit,distance_limit) in enumerate(self.edge_thresholds[1:],1):
                self.edge_levels[level] = added[(support_distances<=radius_limit)&(plane_distances<=distance_limit)]
            selected = self.edge_levels[3] if options.edge_completion else self.edge_levels[0]
            self.last_edge_count = len(selected)
            if len(selected): result = np.unique(np.concatenate((result,selected)))
        if debug:
            logging.info("Growth selected=%d tree_builds=%d query_ms=%.1f plane_ms=%.1f normal_ms=%.1f total_ms=%.1f",
                         len(result),self.builds,query_seconds*1000,plane_seconds*1000,normal_seconds*1000,(time.perf_counter()-start)*1000)
        return result
