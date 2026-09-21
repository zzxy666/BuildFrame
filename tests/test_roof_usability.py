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
