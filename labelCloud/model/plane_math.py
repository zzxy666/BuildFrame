"""Shared stable least-squares plane fit."""
import numpy as np


def least_squares_plane(xyz):
    """中心化 PCA（等价最小二乘 SVD）；允许竖直面，拒绝共线/无效输入。"""
    xyz=np.asarray(xyz,np.float64)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<3 or not np.isfinite(xyz).all():
        raise ValueError('至少需要三个有限坐标点')
    center=xyz.mean(axis=0);delta=xyz-center
    eig,vec=np.linalg.eigh(delta.T@delta/len(xyz))
    if eig[1]<=1e-14 or eig[1]<=max(eig[2],1e-14)*1e-8:
        raise ValueError('点集接近直线或数值退化，无法拟合平面')
    normal=vec[:,0]
    if normal[2]<0: normal=-normal
    return normal,center
