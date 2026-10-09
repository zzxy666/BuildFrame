import json
import numpy as np
import laspy
import pytest
import rasterio
from affine import Affine
from pyproj import CRS
from labelCloud.model.scene_planes import RoofPlanes
from labelCloud.model.coordinate_mapper import PixelMask, CoordinateMapper
from labelCloud.model.dual_gt import rgb_identity
from labelCloud.model.dual_gt_export import DualGTExporter, DualGTValidationError, MaskBlocks
from labelCloud.model.geometry_refiner import plane_geometry


@pytest.fixture
def scene(tmp_path):
    path=tmp_path/'scene.las';rgb=tmp_path/'rgb.tif'
    las=laspy.LasData(laspy.LasHeader(point_format=7,version='1.4'))
    las.header.add_crs(CRS.from_epsg(3857));x,y=np.meshgrid(np.arange(10),np.arange(10))
    las.x=x.ravel()+100;las.y=y.ravel()+100;las.z=(x+y).ravel()*.2
    las.intensity=np.arange(100)*10;las.classification=np.arange(100)%7
    las.red=np.arange(100)*100;las.green=np.arange(100)*200;las.blue=np.arange(100)*300
    las.return_number=np.arange(100)%3+1;las.number_of_returns=np.full(100,3)
    las.gps_time=np.arange(100)+1000000.125
    las.add_extra_dim(laspy.ExtraBytesParams(name='original_custom',type=np.int32));las.original_custom=np.arange(100)-50
    las.write(path)
    transform=Affine(0.1,0.02,99,0.01,-0.1,112)
    with rasterio.open(rgb,'w',driver='GTiff',width=40,height=30,count=3,dtype='uint8',crs=3857,transform=transform,nodata=0) as ds:
        data=np.full((3,30,40),120,np.uint8);data[:,0,0]=0;ds.write(data)
    return RoofPlanes(path),rgb,las


def fragment(model,rgb,col=2,row=3,width=4,height=5,source='rgb_polygon'):
    return model.dual_gt.store.write(PixelMask(np.ones((height,width),bool),col,row),rgb_identity(rgb),source)


def confirm(model,rgb,pid=1,ids=None,**kwargs):
    model.assign(pid,ids=np.arange(20) if ids is None else ids)
    f=fragment(model,rgb,**kwargs);model.dual_gt.confirm(pid,f)
    return f


def test_lidar_first_rgb_only_undo_reload_and_independence(scene):
    m,rgb,_=scene;m.assign(1,ids=np.arange(20));labels=m.labels.copy()
    assert not DualGTExporter(m,rgb).validate()['ok']
    f=fragment(m,rgb);m.dual_gt.confirm(1,f)
    assert DualGTExporter(m,rgb).validate()['ok']
    np.testing.assert_array_equal(m.labels,labels)
    m.undo();np.testing.assert_array_equal(m.labels,labels)
    assert m.dual_gt.ensure(1)['rgb_gt_status']=='MISSING'
    m.redo();m.save();loaded=RoofPlanes(m.path)
    assert loaded.dual_gt.state==m.dual_gt.state
    np.testing.assert_array_equal(loaded.dual_gt.store.read(f),np.ones((5,4),bool))
    assert (m.session.folder/'dual_gt.json').exists()
    old=m.dual_gt.ensure(1)['fragments'];m.assign(1,ids=[20,21])
    assert m.dual_gt.ensure(1)['fragments']==old
    assert m.dual_gt.ensure(1)['cross_modal_status']=='NEEDS_REVIEW'
    assert DualGTExporter(m,rgb).validate()['ok']
    labels=m.labels.copy();m.dual_gt.confirm(1,fragment(m,rgb,col=15),replace=True)
    np.testing.assert_array_equal(m.labels,labels);m.undo();assert m.dual_gt.ensure(1)['fragments']==old


def test_dual_commit_one_undo_four_states(scene):
    m,rgb,_=scene;pid=m.new_plane();before=m.labels.copy();state=m.dual_gt.state
    f=fragment(m,rgb);m.assign(pid,ids=np.arange(20))
    g=plane_geometry(m.point_reader.read(np.arange(20)),1,pid)
    m.attach_plane_geometry(g);m.attach_rgb_annotation({'plane_id':pid,'annotation_source':'rgb_polygon'})
    m.dual_gt.confirm(pid,f,attached=True)
    m.undo();np.testing.assert_array_equal(m.labels,before);assert m.dual_gt.state==state
    assert not m.session_metadata['plane_geometry'] and not m.session_metadata['rgb_annotations']
    m.redo();assert m.plane_counts[pid]==20 and m.dual_gt.ensure(pid)['rgb_gt_status']=='CONFIRMED'
    assert m.session_metadata['plane_geometry'][str(pid)]==g
    assert len(m.session_metadata['rgb_annotations'])==1


def test_polygon_native_grid_union_edge_and_sam_reference(scene):
    m,rgb,_=scene
    with rasterio.open(rgb) as ds: mapper=CoordinateMapper(ds.transform,ds.width,ds.height,3857,3857)
    mask=mapper.polygon_mask(np.array([[36,26],[40,26],[40,30],[36,30]]))
    m.assign(1,ids=np.arange(20));f=m.dual_gt.store.write(mask,rgb_identity(rgb),'rgb_polygon');m.dual_gt.confirm(1,f)
    folder=m.session.folder/'rgb_masks';folder.mkdir()
    path=folder/'sam.npz';data=np.ones((3,3),bool);np.savez_compressed(path,mask=data,col0=35,row0=25)
    f2=m.dual_gt.store.write(PixelMask(data,35,25),rgb_identity(rgb),'rgb_sam',existing=path)
    m.dual_gt.confirm(1,f2);m.save();loaded=RoofPlanes(m.path)
    assert len(list((m.session.folder/'rgb_gt').glob('*.npz')))==1
    result=DualGTExporter(loaded,rgb).validate();assert result['ok']
    assert result['rgb_pixel_counts'][1]==21 # 16+9-4
    assert loaded.dual_gt.store.path(f2)==path


def test_delete_reuse_merge_and_partial_merge(scene):
    m,rgb,_=scene;confirm(m,rgb);confirm(m,rgb,2,np.arange(20,40),col=10)
    uid=m.dual_gt.ensure(1)['plane_uid'];old=m.dual_gt.state
    m.merge(2,1);r=m.dual_gt.ensure(1)
    assert r['rgb_gt_status']=='NEEDS_REVIEW' and len(r['fragments'])==2
    assert '2' not in m.dual_gt.state['active'];assert not DualGTExporter(m,rgb).validate()['ok']
    m.undo();assert m.dual_gt.state==old;m.redo();m.dual_gt.confirm(1)
    assert DualGTExporter(m,rgb).validate()['ok']
    m.delete(1);assert '1' not in m.dual_gt.state['active']
    assert uid in m.dual_gt.state['archived'];assert m.new_plane()==1
    assert m.dual_gt.ensure(1)['plane_uid']!=uid and m.dual_gt.ensure(1)['rgb_gt_status']=='MISSING'
    m.undo();m.undo();assert m.dual_gt.ensure(1)['plane_uid']==uid
    m.undo();m.undo() # undo RGB review then merge
    allowed=np.zeros(100,bool);allowed[20:25]=True;m.merge(2,1,allowed)
    assert m.dual_gt.ensure(2)['rgb_gt_status']=='NEEDS_REVIEW'
    assert len(m.dual_gt.ensure(1)['fragments'])==1


@pytest.mark.parametrize('pid',[1,255,256,65535,65536,100000,100001])
def test_export_exact_raster_all_las_fields_nodata_large_ids(scene,tmp_path,pid):
    m,rgb,las=scene;confirm(m,rgb,pid,col=0,row=0)
    m.set_hidden_ids(np.arange(10),True)
    m.attach_plane_geometry(plane_geometry(m.point_reader.read(np.arange(20)),1,pid))
    m.assign(pid,ids=np.arange(20,30));assert m.session_metadata['plane_geometry'][str(pid)]['geometry_dirty']
    target=tmp_path/'out';DualGTExporter(m,rgb).export(target)
    with rasterio.open(rgb) as src,rasterio.open(target/'rgb/scene_plane_id.tif') as gt,rasterio.open(target/'rgb/scene_valid_mask.tif') as valid:
        assert gt.transform==src.transform and gt.crs==src.crs and gt.shape==src.shape and gt.dtypes==('uint32',)
        expected=np.zeros(src.shape,np.uint32);expected[:5,:4]=pid;expected[0,0]=0
        np.testing.assert_array_equal(gt.read(1),expected);np.testing.assert_array_equal(valid.read(1),expected>0)
        assert gt.block_shapes==[(512,512)] and valid.dtypes==('uint8',)
    out=laspy.read(target/'lidar/scene_gt.laz')
    assert len(out.points)==100
    for name in las.points.array.dtype.names: np.testing.assert_array_equal(out.points.array[name],las.points.array[name])
    np.testing.assert_array_equal(out.plane_id,m.labels);assert out.plane_id.dtype==np.uint32
    manifest=json.loads((target/'metadata/dual_gt_manifest.json').read_text())
    assert manifest['complete_dual_gt'] and manifest['planes'][0]['rgb_pixel_count']==19
    assert manifest['planes'][0]['lidar_point_count']==manifest['planes'][0]['geometry']['point_count']==30
    assert (target/'rgb/scene_plane_preview.png').exists()


def test_overlap_report_blocks_both_modes(scene,tmp_path):
    m,rgb,_=scene;confirm(m,rgb);confirm(m,rgb,2,np.arange(20,40),col=4)
    exporter=DualGTExporter(m,rgb);report=exporter.validate()
    assert report['conflicts']==[(1,2,10)]
    for strict in (True,False):
        with pytest.raises(DualGTValidationError,match='overlap pixels: 10'): exporter.export(tmp_path/'bad',strict)
        assert not (tmp_path/'bad').exists()


def test_partial_missing_modalities_and_review(scene,tmp_path):
    m,rgb,_=scene;m.assign(1,ids=np.arange(10));pid=m.new_plane();m.dual_gt.confirm(pid,fragment(m,rgb))
    report=DualGTExporter(m,rgb).validate();assert report['missing_rgb_planes']==[1] and report['missing_lidar_planes']==[2]
    DualGTExporter(m,rgb).export(tmp_path/'partial',False)
    manifest=json.loads((tmp_path/'partial/metadata/dual_gt_manifest.json').read_text())
    assert not manifest['complete_dual_gt'] and manifest['missing_rgb_planes']==[1] and manifest['missing_lidar_planes']==[2]


@pytest.mark.parametrize('damage',['missing','shape','checksum','bounds','source','uid','grid'])
def test_invalid_masks_block(scene,tmp_path,damage):
    m,rgb,_=scene;f=confirm(m,rgb);record=m.dual_gt.ensure(1)
    if damage=='missing': m.dual_gt.store.path(f).unlink()
    elif damage=='shape': np.savez(m.dual_gt.store.path(f),mask=np.ones((2,2),bool),col0=2,row0=3)
    elif damage=='checksum':
        data=np.ones((5,4),bool);data[0,0]=False;np.savez(m.dual_gt.store.path(f),mask=data,col0=2,row0=3)
    elif damage=='bounds': f['col0']=99
    elif damage=='source': f['source']=dict(f['source'],path='different.tif')
    elif damage=='grid': f['source']=dict(f['source'],transform=[1,0,0,0,-1,0])
    else: m.dual_gt.state['archived'][record['plane_uid']]=record
    report=DualGTExporter(m,rgb).validate();assert report['errors'] and not report['ok']
    with pytest.raises(DualGTValidationError): DualGTExporter(m,rgb).export(tmp_path/'bad',False)
    assert not (tmp_path/'bad').exists()


@pytest.mark.parametrize('stage',['laz','preview','metadata'])
def test_atomic_failures(scene,tmp_path,monkeypatch,stage):
    m,rgb,_=scene;confirm(m,rgb);exporter=DualGTExporter(m,rgb)
    def fail(*args,**kwargs): raise OSError('injected failure')
    if stage=='laz': monkeypatch.setattr(m.session,'export',fail)
    elif stage=='preview': monkeypatch.setattr(exporter,'_write_preview',fail)
    else: monkeypatch.setattr('labelCloud.model.dual_gt_export.atomic_json',fail)
    with pytest.raises(OSError,match='injected'): exporter.export(tmp_path/'final')
    assert not (tmp_path/'final').exists() and not list(tmp_path.glob('.final-*'))
    assert m.path.exists() and rgb.exists()


def test_legacy_history_not_promoted(scene):
    m,rgb,_=scene;m.assign(1,ids=np.arange(20));m.attach_rgb_annotation({'annotation_source':'rgb_polygon','plane_id':1})
    fragment(m,rgb);m.save();loaded=RoofPlanes(m.path)
    assert loaded.dual_gt.ensure(1)['rgb_gt_status']=='MISSING'
    assert not DualGTExporter(loaded,rgb).validate()['ok']


def test_window_seams_and_all_nodata_rejected(scene,tmp_path):
    m,rgb,_=scene
    with rasterio.open(rgb,'w',driver='GTiff',width=1040,height=1040,count=3,dtype='uint8',crs=3857,
            transform=Affine(.1,0,99,0,-.1,112),nodata=0) as ds:
        data=np.full((3,1040,1040),120,np.uint8);data[:,1024,1024]=0;ds.write(data)
    confirm(m,rgb,col=510,row=510,width=530,height=530)
    with pytest.raises(ValueError,match='nodata'): fragment(m,rgb,col=1024,row=1024,width=1,height=1)
    DualGTExporter(m,rgb).export(tmp_path/'windows')
    with rasterio.open(tmp_path/'windows/rgb/scene_plane_id.tif') as ds:
        out=ds.read(1);assert out[511,512]==out[512,511]==out[1039,1039]==1
        assert out[1024,1024]==0 and np.count_nonzero(out)==530*530-1


def test_partial_excludes_unreviewed_rgb_union(scene,tmp_path):
    m,rgb,_=scene;confirm(m,rgb);confirm(m,rgb,2,np.arange(20,40),col=10);m.merge(2,1)
    DualGTExporter(m,rgb).export(tmp_path/'partial',False)
    with rasterio.open(tmp_path/'partial/rgb/scene_plane_id.tif') as ds: assert not ds.read(1).any()
    manifest=json.loads((tmp_path/'partial/metadata/dual_gt_manifest.json').read_text())
    assert manifest['needs_review_planes']==[1] and not manifest['complete_dual_gt']
    assert manifest['planes'][0]['rgb_gt_status']=='NEEDS_REVIEW'


@pytest.mark.parametrize('stage',['raster','verification'])
def test_atomic_raster_and_verification_errors(scene,tmp_path,monkeypatch,stage):
    m,rgb,_=scene;confirm(m,rgb);exporter=DualGTExporter(m,rgb)
    def fail(*args,**kwargs): raise OSError('write/verify failed')
    if stage=='verification': monkeypatch.setattr(exporter,'_verify_outputs',fail)
    else:
        original=rasterio.open
        def open_checked(path,*args,**kwargs):
            if args and args[0]=='w' and str(path).endswith('_plane_id.tif'): fail()
            return original(path,*args,**kwargs)
        monkeypatch.setattr(rasterio,'open',open_checked)
    with pytest.raises(OSError): exporter.export(tmp_path/'final')
    assert not (tmp_path/'final').exists() and not list(tmp_path.glob('.final-*'))


def test_cancel_and_existing_directory_never_overwritten(scene,tmp_path):
    import threading
    m,rgb,_=scene;confirm(m,rgb);event=threading.Event();event.set()
    with pytest.raises(InterruptedError): DualGTExporter(m,rgb,event).export(tmp_path/'cancelled')
    assert not (tmp_path/'cancelled').exists()
    directory=tmp_path/'existing';directory.mkdir();(directory/'user.txt').write_text('preserve')
    with pytest.raises(ValueError,match='已存在'): DualGTExporter(m,rgb).export(directory)
    assert (directory/'user.txt').read_text()=='preserve'


def test_no_whole_point_scan_for_validation_statistics(scene,monkeypatch):
    m,rgb,_=scene;confirm(m,rgb)
    def fail(*args): raise AssertionError('must not collect full plane indices to count dirty geometry')
    monkeypatch.setattr(m,'geometry_jobs',fail)
    assert DualGTExporter(m,rgb).validate()['ok']


def test_triple_overlap_counts_every_pair_once(scene):
    m,rgb,_=scene
    for i in range(1,4): confirm(m,rgb,i,np.arange((i-1)*20,i*20))
    # 同一 Plane 自身重叠 fragment 不重复计数。
    m.dual_gt.confirm(1,fragment(m,rgb))
    report=DualGTExporter(m,rgb).validate()
    assert report['conflicts']==[(1,2,20),(1,3,20),(2,3,20)]
    assert report['rgb_pixel_counts']=={1:20,2:20,3:20}


def test_pixel_protection_offsets_replace_and_undo(scene):
    m,rgb,_=scene
    first=confirm(m,rgb,1,col=2,row=3,width=4,height=5)
    confirm(m,rgb,2,np.arange(20,40),col=4,row=3,width=4,height=5)
    source=rgb_identity(rgb);candidate=PixelMask(np.ones((5,8),bool),0,3)
    before=m.labels.copy();state=m.dual_gt.state
    free,blocked=m.dual_gt.unlabelled_mask(candidate,source,1)
    assert blocked==30 and free.data.sum()==10
    assert candidate.data.all() and m.dual_gt.state is state
    np.testing.assert_array_equal(m.labels,before)
    free,blocked=m.dual_gt.unlabelled_mask(candidate,source,1,True)
    assert blocked==20 and free.data.sum()==20
    # 替换当前 Plane 时仍排除另一 Plane；已有重叠可这样手工修复。
    f=m.dual_gt.store.write(free,source,'rgb_polygon');m.dual_gt.confirm(1,f,replace=True)
    assert not DualGTExporter(m,rgb).validate()['conflicts']
    m.undo();assert m.dual_gt.ensure(1)['fragments']==[first]
    assert DualGTExporter(m,rgb).validate()['conflicts']==[(1,2,10)]


def test_pixel_protection_archive_review_and_source(scene):
    m,rgb,_=scene;confirm(m,rgb);source=rgb_identity(rgb)
    mask=PixelMask(np.ones((5,4),bool),2,3)
    free,blocked=m.dual_gt.unlabelled_mask(mask,source,3);assert blocked==20 and not free.data.any()
    m.dual_gt.state['active']['1']['rgb_gt_status']='NEEDS_REVIEW'
    assert m.dual_gt.unlabelled_mask(mask,source,3)[1]==20
    with pytest.raises(ValueError,match='来源'): m.dual_gt.unlabelled_mask(mask,dict(source,path='other.tif'),3)
    m.delete(1);assert m.dual_gt.unlabelled_mask(mask,source,3)[1]==0
    m.undo();assert m.dual_gt.unlabelled_mask(mask,source,3)[1]==20
