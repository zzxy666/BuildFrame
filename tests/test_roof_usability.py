import numpy as np
from labelCloud.model.scene_spatial import SceneSpatialIndex
from labelCloud.model.local_plane_growth import GrowthOptions
from test_local_plane_growth import grid, seed_patch


def test_edge_completion_is_single_ring_and_respects_protection():
    xyz=grid().astype(np.float32)
    spatial=SceneSpatialIndex(xyz)
    labels=np.zeros(len(xyz),np.uint32)
    hidden=np.zeros(len(xyz),bool)
    seeds=np.flatnonzero(seed_patch(xyz))
    options=GrowthOptions(edge_completion=True,edge_radius=.16,edge_distance=.05)
    spatial.normal_key=((1.,1.,1.),20,.6,0,0)
    # 模拟边缘受相邻坡面污染的法向；仅贴着严格区域的第一圈允许补回。
    spatial.normals={i:np.array([0,0,1.] if p[0]<.85 else [1.,0,0]) for i,p in enumerate(xyz)}
    protected=np.flatnonzero((xyz[:,0]>.85)&(xyz[:,0]<.95))[4]
    labels[protected]=7
    result=spatial.grow(labels,hidden,seeds,1,np.ones(3),options)
    assert spatial.last_edge_count>0
    assert protected not in result
    assert np.any(xyz[result,0]>.85)
    assert not np.any(xyz[result,0]>.95)
    assert spatial.builds==1
    options=GrowthOptions(edge_completion=False)
    strict=spatial.grow(labels,hidden,seeds,1,np.ones(3),options)
    assert not np.any(xyz[strict,0]>.85)


def test_preview_levels_nested_protected_and_baseline():
    xyz=grid().astype(np.float32);spatial=SceneSpatialIndex(xyz)
    labels=np.zeros(len(xyz),np.uint32);hidden=np.zeros(len(xyz),bool)
    seeds=np.flatnonzero(seed_patch(xyz))
    options=GrowthOptions(edge_completion=True,edge_radius=.16,edge_distance=.05)
    spatial.normal_key=((1.,1.,1.),20,.6,0,0)
    spatial.normals={i:np.array([0,0,1.] if p[0]<.85 else [1.,0,0]) for i,p in enumerate(xyz)}
    protected=np.flatnonzero(xyz[:,0]>.85)[0];labels[protected]=7
    baseline=spatial.grow(labels,hidden,seeds,1,np.ones(3),options)
    preview=spatial.grow(labels,hidden,seeds,1,np.ones(3),options,preview_levels=True)
    np.testing.assert_array_equal(baseline,preview)
    assert len(spatial.edge_levels)==11 and len(spatial.edge_levels[0])==0
    for a,b in zip(spatial.edge_levels,spatial.edge_levels[1:]): assert np.isin(a,b).all()
    assert len(spatial.edge_levels[5])>len(spatial.edge_levels[1])
    for level in spatial.edge_levels:
        assert protected not in level
        assert not np.isin(level,spatial.last_strict).any()
    assert spatial.builds==1


def test_high_levels_extend_beyond_old_radius_and_preserve_masks():
    xyz=grid().astype(np.float32);spatial=SceneSpatialIndex(xyz)
    labels=np.zeros(len(xyz),np.uint32);hidden=np.zeros(len(xyz),bool)
    seeds=np.flatnonzero(seed_patch(xyz))
    options=GrowthOptions(edge_completion=True,edge_radius=.12,edge_distance=.05,neighbor_radius=.3)
    spatial.normal_key=((1.,1.,1.),20,.3,0,0)
    spatial.normals={i:np.array([0,0,1.] if p[0]<.65 else [1.,0,0]) for i,p in enumerate(xyz)}
    labels[-1]=7;hidden[-2]=True
    roi=np.ones(len(xyz),bool);roi[-3]=False
    spatial.grow(labels,hidden,seeds,1,np.ones(3),options,roi=roi,preview_levels=True)
    assert len(spatial.edge_levels[10])>len(spatial.edge_levels[5])
    assert np.max(xyz[spatial.edge_levels[10],0])>np.max(xyz[spatial.edge_levels[5],0])
    assert not np.isin([len(xyz)-1,len(xyz)-2,len(xyz)-3],spatial.edge_levels[10]).any()
    np.testing.assert_allclose(spatial.edge_thresholds[10],[.6,.3])
