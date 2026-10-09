import json
import threading
from unittest.mock import patch
import laspy
import numpy as np
import pytest
from test_roof_planes import source_file
from labelCloud.model.roof_planes import RoofPlanes, project_points
from labelCloud.model.screen_projection import ScreenProjectionCache
from labelCloud.model.scene_spatial import SceneSpatialIndex
from labelCloud.model.local_plane_growth import GrowthOptions
from labelCloud.model.scene_workspace import SceneWorkspace


def test_existing_id_counts_undo_and_sidecar(tmp_path):
    path=tmp_path/'roof.las'; source_file(path)
    model=RoofPlanes(path)
    model.selection[:4]=True; model.assign(1000000)
    assert model.labels.dtype==np.uint32
    assert model.new_plane()==1
    model.undo()
    assert model.new_plane()==1
    model.delete(1000000)
    assert model.plane_counts[0]==12
    model.undo()
    assert model.plane_counts[1000000]==4
    original=path.read_bytes()
    saved=model.save()
    assert saved.name=='plane_id.npy' and not model.output_path.exists()
    assert not model.dirty
    reloaded=RoofPlanes(path)
    np.testing.assert_array_equal(reloaded.labels,model.labels)
    assert reloaded.new_plane()==2
    assert path.read_bytes()==original
    # 选区刷新读取缓存统计，不对全部点调用 unique。
    with patch('numpy.unique',side_effect=AssertionError('full scan')):
        assert (1000000,4) in model.counts()


@pytest.mark.parametrize('suffix',['.las','.laz'])
def test_chunk_export_preserves_every_field_and_evlr(tmp_path,suffix):
    path=tmp_path/'roof.las'; source=source_file(path)
    from laspy.vlrs.vlrlist import VLRList
    source.evlrs=VLRList([laspy.VLR(user_id='custom',record_id=900,record_data=b'EVLR data')])
    source.write(path)
    model=RoofPlanes(path)
    model.selection[[1,5,7]]=True; model.assign(4294967295)
    assert model.new_plane() == 1
    model.set_hidden(np.ones(12,bool))
    destination=model.export(tmp_path/('export'+suffix))
    saved=laspy.read(destination,laz_backend=laspy.LazBackend.Laszip)
    for name in source.points.array.dtype.names:
        np.testing.assert_array_equal(source.points.array[name],saved.points.array[name])
    np.testing.assert_array_equal(saved.plane_id,model.labels)
    assert saved.evlrs[0].record_data_bytes()==b'EVLR data'
    assert 'hidden_mask' not in saved.point_format.dimension_names
    with pytest.raises(ValueError): model.export(path)
    old=destination.read_bytes()
    cancel=threading.Event();cancel.set()
    with pytest.raises(InterruptedError): model.export(destination,cancel)
    assert destination.read_bytes()==old


def test_journal_recovery_and_source_validation(tmp_path):
    path=tmp_path/'roof.las'; source_file(path)
    model=RoofPlanes(path); model.save()
    np.savez(model.session.folder/'pending.npz',ids=np.array([2,7]),values=np.array([9000,9000],np.uint32))
    recovered=RoofPlanes(path)
    assert recovered.labels[2]==9000 and recovered.labels[7]==9000
    assert recovered.max_plane_id==9000
    with path.open('ab') as stream: stream.write(b'x')
    with pytest.raises(ValueError,match='不匹配'): RoofPlanes(path)


def test_projection_cache_bbox_filters_and_invalidation():
    rng=np.random.default_rng(6)
    points=rng.uniform(-2,2,(4000,3)).astype(np.float32)
    cache=ScreenProjectionCache(); cache.CHUNK=300
    matrix=np.eye(4)
    cache.project(points,matrix,matrix,600,400)
    xy,valid=project_points(points,matrix,matrix,600,400)
    np.testing.assert_allclose(cache.screen,xy,atol=.0001)
    np.testing.assert_array_equal(cache.valid,valid)
    cache.project(points,matrix,matrix,600,400)
    assert cache.builds==1
    labels=np.zeros(len(points),np.uint32); labels[::3]=3
    hidden=np.zeros(len(points),bool); hidden[::7]=True
    roi=points[:,2]>.2
    polygon=[[150,100],[450,100],[450,300],[150,300]]
    result=cache.select(polygon,labels,hidden,'unlabelled',0,roi)
    expected=np.flatnonzero(valid&~hidden&(labels==0)&roi&(xy[:,0]>=150)&(xy[:,0]<=450)&(xy[:,1]>=100)&(xy[:,1]<=300))
    np.testing.assert_array_equal(result,expected)
    moved=matrix.copy(); moved[3,0]=.3
    cache.project(points,moved,matrix,600,400); assert cache.builds==2
    cache.project(points,moved,matrix,800,400); assert cache.builds==3
    cancel=threading.Event(); cancel.set()
    with pytest.raises(InterruptedError): cache.project(points,matrix,matrix,600,400,cancel)


def test_reusable_tree_normals_hidden_roi_and_feet():
    x,y=np.meshgrid(np.arange(21)*.1,np.arange(15)*.1)
    xyz=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
    xyz=np.vstack((xyz,xyz+[10,0,0]))
    # 渲染坐标使用 US foot，所有算法阈值仍为米。
    factor=1200/3937
    spatial=SceneSpatialIndex((xyz/factor).astype(np.float32))
    labels=np.zeros(len(xyz),np.uint32); hidden=np.zeros(len(xyz),bool)
    seeds=np.flatnonzero((xyz[:,0]>=.1)&(xyz[:,0]<=.4)&(xyz[:,1]>=.2)&(xyz[:,1]<=.5))
    options=GrowthOptions(neighbor_radius=.25)
    result=spatial.grow(labels,hidden,seeds,1,np.repeat(factor,3),options)
    assert len(result)>200 and not (result>=315).any()
    count=len(spatial.normals)
    result2=spatial.grow(labels,hidden,seeds,1,np.repeat(factor,3),options)
    assert spatial.builds==1 and len(spatial.normals)==count
    np.testing.assert_array_equal(result,result2)
    hidden[(xyz[:,0]>=.7)&(xyz[:,0]<=1.2)]=True
    result=spatial.grow(labels,hidden,seeds,1,np.repeat(factor,3),options,hidden_revision=1)
    assert not hidden[result].any() and not (xyz[result,0]>1.2).any()
    roi=xyz[:,0]<.6
    result=spatial.grow(labels,hidden,seeds,1,np.repeat(factor,3),options,roi,1,1)
    assert roi[result].all() and spatial.builds==1


def test_workspace_grid_bookmarks_and_memory_bound(tmp_path):
    path=tmp_path/'roof.las'; source_file(path)
    model=RoofPlanes(path)
    xyz=np.column_stack((np.arange(12),np.arange(12),np.zeros(12))).astype(np.float32)
    work=SceneWorkspace(xyz,model.session)
    work.create_grid(5,np.ones(3))
    bounds=work.next_unreviewed(); work.set_roi(bounds)
    assert work.mark_reviewed()==1
    work.bookmarks.append({'camera':{'position':[0,0,1]},'plane_id':0,'roi':bounds})
    model.session_metadata['roi']=bounds
    model.save(); work.save()
    other=RoofPlanes(path); restored=SceneWorkspace(xyz,other.session)
    assert len(restored.grid['reviewed'])==1 and restored.bookmarks==work.bookmarks
    np.testing.assert_array_equal(work.roi_mask,restored.roi_mask)
    model.UNDO_LIMIT=800
    for i in range(20): model.new_plane()
    assert model.history_bytes<=800


def test_deleted_highest_id_reused_after_reload_and_undo(tmp_path):
    path = tmp_path / 'roof.las'; source_file(path)
    model = RoofPlanes(path)
    for pid in range(1, 9):
        assert model.new_plane() == pid
    model.selection[:4] = True; model.assign(8)
    model.delete(8)
    model.save()
    reloaded = RoofPlanes(path)
    assert reloaded.new_plane() == 8
    assert model.new_plane() == 8
    model.selection[4:8] = True; model.assign(8)
    model.undo(); model.undo(); model.undo()
    assert np.all(model.labels[:4] == 8)
    assert not model.labels[4:].any()
    assert model.new_plane() == 9
    model.delete(8)
    assert model.new_plane() == 8


def test_deleted_uint32_max_does_not_block_new_plane(tmp_path):
    path = tmp_path / 'roof.las'; source_file(path)
    model = RoofPlanes(path)
    model.selection[:2] = True; model.assign(int(np.iinfo(np.uint32).max))
    model.delete(model.current)
    assert model.new_plane() == 1


def test_middle_gap_reuse_and_undo_after_reload(tmp_path):
    path=tmp_path/'roof.las'; source_file(path)
    model=RoofPlanes(path)
    for pid in range(1,72): assert model.new_plane()==pid
    model.selection[:4]=True; model.assign(69)
    model.delete(69); model.save()
    reloaded=RoofPlanes(path)
    assert reloaded.new_plane()==69
    assert model.new_plane()==69
    model.selection[4:8]=True; model.assign(69)
    model.undo(); model.undo(); model.undo()
    assert np.all(model.labels[:4]==69)
    assert not model.labels[4:].any()
    assert model.new_plane()==72
    model.delete(20); model.delete(69)
    assert model.new_plane()==20
    assert model.new_plane()==69
