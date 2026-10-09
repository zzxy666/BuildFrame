"""种子驱动的局部扩展：只计算候选索引，不修改任何 LAS 或标签。"""
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from .plane_math import least_squares_plane


@dataclass(frozen=True)
class GrowthOptions:
    normal_k: int = 20
    neighbor_radius: float = 0.6
    plane_distance: float = 0.15
    normal_angle: float = 10.0
    edge_completion: bool = False
    edge_radius: float = 0.25
    edge_distance: float = 0.08
    robust_fit: bool = False
    ransac_distance: float = 0.08
    ransac_iterations: int = 300
    ransac_min_ratio: float = 0.6


def ransac_seed_mask(samples, options, cancel=None, *, minimum=6, random_seed=0, require_majority=True):
    """三点采样找主要平面，再用 PCA 精拟合；仅筛选种子，不修改标签。"""
    threshold = min(options.ransac_distance, options.plane_distance)
    if not np.isfinite(threshold) or threshold <= 0 or options.ransac_iterations < 1:
        raise ValueError("无效 RANSAC 距离或迭代次数")
    if not (0.5 if require_majority else 0) < options.ransac_min_ratio <= 1:
        raise ValueError("RANSAC 最低内点比例必须大于 50% 且不超过 100%")
    required = max(minimum, int(np.ceil(len(samples)*options.ransac_min_ratio)))
    rng = np.random.default_rng(random_seed)  # 同样输入得到可复现的预览。
    best = np.zeros(len(samples), bool); best_error = np.inf
    for _ in range(options.ransac_iterations):
        if cancel is not None and cancel.is_set(): raise InterruptedError("拟合已取消")
        a, b, c = samples[rng.choice(len(samples), 3, replace=False)]
        normal = np.cross(b-a, c-a); length = np.linalg.norm(normal)
        if length < 1e-10: continue
        residual = np.abs((samples-a)@(normal/length))
        mask = residual <= threshold
        count = int(mask.sum()); error = float(residual[mask].mean())
        if count > best.sum() or (count == best.sum() and error < best_error):
            best, best_error = mask, error
    if best.sum() < required:
        raise ValueError("RANSAC 种子支持不足，请重新选择同一屋面的种子")
    # 单调剔除不满足精拟合平面的点，避免把离群种子重新加入。
    for _ in range(10):
        normal, center = least_squares_plane(samples[best])
        refined = best & (np.abs((samples-center)@normal) <= threshold)
        if refined.sum() < required:
            raise ValueError("RANSAC 精拟合后种子支持不足")
        if np.array_equal(refined, best): return best
        best = refined
    raise ValueError("RANSAC 拟合未稳定，请重新选择种子")


def metric_coordinates(las, units="auto"):
    """使用原始精度、局部原点和米单位计算，绝不修改原 LAS 的 XYZ。"""
    factors = {"m": 1.0, "ft": 0.3048, "ftUS": 1200 / 3937}
    note = ""
    if units in factors:
        scale = np.repeat(factors[units], 3)
        note = f"XYZ 单位由用户指定：{units}"
    elif units == "auto":
        try:
            crs = las.header.parse_crs()
        except ImportError as exc:
            raise ValueError("无法读取坐标单位，请在扩展参数中明确选择米或英尺。") from exc
        if crs is None or not crs.is_projected:
            raise ValueError("LAS 缺少投影坐标系。请在扩展参数中明确选择 XYZ 单位；经纬度数据需先投影。")
        axes = crs.axis_info
        scale = np.array([axes[0].unit_conversion_factor, axes[1].unit_conversion_factor,
                          axes[2].unit_conversion_factor if len(axes) >= 3 else axes[0].unit_conversion_factor])
        note = f"XY：{axes[0].unit_name}"
        note += f"；Z：{axes[2].unit_name}" if len(axes) >= 3 else "；无垂直单位，假设 Z 与 XY 相同"
    else:
        raise ValueError("未知坐标单位")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("无效坐标单位换算系数")
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("点云为空或含无效坐标")
    return (xyz - xyz.mean(axis=0)) * scale, note


def grow_plane(xyz, labels, seeds, target, options=GrowthOptions(), available=None):
    """PCA 固定种子平面 + 法向一致性 + 半径邻接 BFS，返回布尔候选掩膜。"""
    if (options.normal_k < 6 or not 0 < options.normal_angle < 90 or
            not all(np.isfinite(v) and v > 0 for v in (options.neighbor_radius, options.plane_distance))):
        raise ValueError("normal_k 至少为 6，距离必须为正数，夹角须在 0–90° 之间")
    xyz = np.asarray(xyz, dtype=np.float64)
    labels = np.asarray(labels)
    seeds = np.asarray(seeds, dtype=bool)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or labels.shape != (len(xyz),) or seeds.shape != labels.shape:
        raise ValueError("点云、种子与标签的点数不一致")
    if available is not None:
        # 隐藏点完全排除在法向邻域和生长树之外；结果映射回原始索引。
        available = np.asarray(available, dtype=bool)
        if available.shape != labels.shape:
            raise ValueError("可见掩膜与原始点数不一致")
        indices = np.flatnonzero(available)
        result = np.zeros(len(xyz), dtype=bool)
        result[indices] = grow_plane(xyz[indices], labels[indices], seeds[indices], target, options)
        return result
    if not np.isfinite(xyz).all() or target <= 0:
        raise ValueError("请先选择或新建一个 ID 大于 0 的 Roof Plane")
    ids = np.flatnonzero(seeds)
    if len(ids) < 6:
        raise ValueError("至少选择 6 个种子点，请在同一屋面内部选择一块有面积的区域。")
    if np.any((labels[ids] != 0) & (labels[ids] != target)):
        raise ValueError("种子包含其他 Plane 的点。请重新选择未标注点或当前 Plane 的点。")
    samples = xyz[ids]
    center = samples.mean(axis=0)
    values, vectors = np.linalg.eigh((samples - center).T @ (samples - center) / len(samples))
    if values[1] <= 1e-10 or values[1] / max(values[2], 1e-12) < 0.01 or values[0] / values[1] > 0.2:
        raise ValueError("种子过于集中、接近直线或不够平整，请重新选择一块稳定的屋面区域。")
    normal = vectors[:, 0]
    residual = np.abs((samples - center) @ normal)
    if np.quantile(residual, .95) > options.plane_distance or residual.max() > 2 * options.plane_distance:
        raise ValueError("种子无法稳定拟合同一平面，请避开屋脊、地面及混合屋面重新选择。")
    # 种子必须自身连续，避免一次框入两片相隔很远的共面区域。
    seed_tree = cKDTree(samples)
    visited = np.zeros(len(ids), dtype=bool)
    visited[0] = True
    queue = deque([0])
    while queue:
        for index in seed_tree.query_ball_point(samples[queue.popleft()], options.neighbor_radius):
            if not visited[index]:
                visited[index] = True
                queue.append(index)
    if not visited.all():
        raise ValueError("种子区域不连续，请圈选相邻的一小块点，或适当增大邻接半径。")

    tree = cKDTree(xyz)
    permitted = (labels == 0) & (np.abs((xyz - center) @ normal) <= options.plane_distance)
    eligible = np.zeros(len(xyz), dtype=bool)
    candidates = np.flatnonzero(permitted)
    # 分批 PCA；限制法向邻域到两倍生长半径，避免隔空估计另一栋建筑的法向。
    k = min(options.normal_k, len(xyz))
    for start in range(0, len(candidates), 2048):
        batch = candidates[start:start + 2048]
        distances, neighbors = tree.query(xyz[batch], k=k)
        valid = distances <= options.neighbor_radius * 2
        counts = valid.sum(axis=1)
        neighborhood = xyz[neighbors]
        means = (neighborhood * valid[..., None]).sum(axis=1) / np.maximum(counts[:, None], 1)
        delta = (neighborhood - means[:, None, :]) * valid[..., None]
        eig, axes = np.linalg.eigh(np.einsum("nki,nkj->nij", delta, delta))
        stable = (counts >= 6) & (eig[:, 1] > 1e-10) & (eig[:, 0] <= .2 * eig[:, 1])
        eligible[batch] = stable & (np.abs(axes[:, :, 0] @ normal) >= np.cos(np.radians(options.normal_angle)))
    # 手选种子已通过整体拟合，作为可靠的初始区域；新点须满足全部约束。
    reached = seeds.copy()
    queue = deque(ids.tolist())
    while queue:
        index = queue.popleft()
        neighbors = np.asarray(tree.query_ball_point(xyz[index], options.neighbor_radius), dtype=int)
        added = neighbors[eligible[neighbors] & ~reached[neighbors]]
        reached[added] = True
        queue.extend(added.tolist())
    # 手选的当前 Plane 种子可作为起点，生长新增点始终只能是 0。
    return reached & (labels == 0)
