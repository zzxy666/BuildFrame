import threading
import numpy as np
import pytest
from labelCloud.model.local_plane_growth import GrowthOptions, ransac_seed_mask
from labelCloud.model.scene_spatial import SceneSpatialIndex


def test_robust_growth_rejects_outliers_without_changing_labels():
    x,y=np.meshgrid(np.arange(12)*.1,np.arange(12)*.1)
    roof=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
    points=np.vstack((roof,[[.3,.3,1],[.4,.3,1],[.3,.4,1]]))
    seeds=np.r_[np.flatnonzero((roof[:,0]<=.5)&(roof[:,1]<=.5)),np.arange(144,147)]
    labels=np.zeros(len(points),np.uint32);labels[100]=2
    hidden=np.zeros(len(points),bool);hidden[101]=True
    spatial=SceneSpatialIndex(points)
    with pytest.raises(ValueError):
        spatial.grow(labels,hidden,seeds,1,np.ones(3),GrowthOptions())
    before=labels.copy()
    options=GrowthOptions(robust_fit=True,edge_completion=True)
    result=spatial.grow(labels,hidden,seeds,1,np.ones(3),options)
    assert len(result)>50
    assert not np.isin([100,101,144,145,146],result).any()
    assert spatial.last_seed_inliers==len(seeds)-3
    np.testing.assert_array_equal(labels,before)
    feet=SceneSpatialIndex(points/.3048)
    np.testing.assert_array_equal(result,feet.grow(labels,hidden,seeds,1,np.repeat(.3048,3),options))


def test_ransac_rejects_weak_support_and_checks_cancel():
    samples=np.random.default_rng(3).normal(size=(100,3))
    with pytest.raises(ValueError):
        ransac_seed_mask(samples,GrowthOptions(robust_fit=True,ransac_distance=.001))
    cancel=threading.Event();cancel.set()
    with pytest.raises(InterruptedError):
        ransac_seed_mask(samples,GrowthOptions(robust_fit=True),cancel)


def test_disconnected_inliers_still_rejected():
    x,y=np.meshgrid(np.arange(3)*.1,np.arange(3)*.1)
    patch=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
    points=np.vstack((patch,patch+[10,0,0]))
    with pytest.raises(ValueError):
        SceneSpatialIndex(points).grow(np.zeros(18,np.uint32),np.zeros(18,bool),np.arange(18),1,np.ones(3),GrowthOptions(robust_fit=True))
