from dataclasses import replace
import threading
import numpy as np
import pytest
from labelCloud.model.geometry_refiner import GeometryRefiner, RefineSettings, plane_geometry


def plane(n=25,offset=(0,0,0)):
    x,y=np.meshgrid(np.arange(n)*.2,np.arange(n)*.2)
    xyz=np.column_stack((x.ravel(),y.ravel(),(.2*x+.1*y).ravel()))
    return xyz+offset


def refine(xyz,settings=None,seeds=None,normals=None,scale=1):
    return GeometryRefiner(lambda ids:xyz[ids],scale,None if normals is None else lambda ids:normals[ids]).refine_plane(np.arange(len(xyz)),settings or RefineSettings(),seeds)


def test_noisy_plane_and_random_outliers():
    rng=np.random.default_rng(4);clean=plane()+rng.normal(0,.005,(625,3))
    xyz=np.vstack((clean,rng.uniform([-2,-2,2],[6,6,8],(156,3))))
    result=refine(xyz)
    assert result.success and result.final_count==625 and result.inlier_ratio>.79
    assert np.dot(result.plane_normal,[-.2,-.1,1])>1
    assert result.rms_distance<.01 and result.plane_normal[2]>0
    assert '法向' in result.warning


def test_tree_high_points_not_highest_z():
    rng=np.random.default_rng(7);roof=plane()
    tree=roof[:150]+[0,0,3]+rng.normal(0,.2,(150,3))
    result=refine(np.vstack((roof,tree)))
    assert result.success and np.max(result.final_point_ids)<len(roof)


def test_two_slopes_warning_and_low_ratio_failure():
    a=plane(20);b=plane(19);b[:,2]=-b[:,0]*.6+6
    xyz=np.vstack((a,b));result=refine(xyz)
    assert result.success and '多个平面' in result.warning
    result=refine(xyz,replace(RefineSettings(),min_inlier_ratio=.85))
    assert not result.success and '多个平面' in result.failure_reason


def test_connectivity_seed_priority_and_coplanar_option():
    xyz=np.vstack((plane(20),plane(12,(20,0,4))))
    result=refine(xyz)
    assert result.success and result.final_count==400
    seeded=refine(xyz,seeds=[401,402])
    assert seeded.final_count==144 and seeded.final_point_ids.min()==400
    all_parts=refine(xyz,replace(RefineSettings(),component_mode='coplanar'))
    assert all_parts.final_count==544


def test_normal_sign_and_invalid_normals_are_optional():
    xyz=plane();normal=np.array([-.2,-.1,1]);normal/=np.linalg.norm(normal)
    normals=np.repeat(normal[None],len(xyz),axis=0);normals[::2]*=-1
    result=refine(xyz,normals=normals);assert result.final_count==len(xyz)
    normals[:20]=[1,0,0];normals[20:40]=0
    result=refine(xyz,normals=normals);assert result.final_count==len(xyz)-20
    assert '无有效法向' in result.warning
    result=refine(xyz,replace(RefineSettings(),use_normals=False),normals=normals)
    assert result.final_count==len(xyz)


@pytest.mark.parametrize('factor',[1.,.3048,1200/3937])
def test_metre_international_foot_survey_foot(factor):
    rng=np.random.default_rng(9);xyz=np.vstack((plane(),plane(8,(0,0,.3))))
    xyz+=rng.normal(0,.002,xyz.shape)
    reference=refine(xyz)
    native=xyz/factor+[2590000,468000,350]
    result=refine(native,scale=factor)
    np.testing.assert_array_equal(result.final_point_ids,reference.final_point_ids)
    assert result.rms_distance==pytest.approx(reference.rms_distance,abs=1e-8)
    geom=plane_geometry(native[result.final_point_ids],factor)
    assert np.linalg.norm(geom['normal'])==pytest.approx(1)
    assert geom['rms_m']==pytest.approx(reference.rms_distance,abs=1e-8)
    residual=np.abs(native[result.final_point_ids]@geom['normal']+geom['d'])*factor
    assert np.sqrt(np.mean(residual**2))==pytest.approx(geom['rms_m'],abs=1e-8)


def test_mixed_vertical_units():
    xyz=plane();scale=np.array([1200/3937,1200/3937,1])
    native=xyz/scale
    a=refine(xyz);b=refine(native,scale=scale)
    np.testing.assert_array_equal(a.final_point_ids,b.final_point_ids)
    geom=plane_geometry(native,scale)
    assert np.max(np.abs(native@geom['normal']+geom['d']))<1e-10


def test_vertical_plane_and_degenerate_failures():
    xyz=plane();xyz=xyz[:,[2,1,0]];xyz[:,0]=2
    assert refine(xyz).success
    for xyz in (np.zeros((3,3)),np.column_stack([np.arange(30)]*3),np.full((30,3),np.nan),np.full((30,3),np.inf)):
        assert not refine(xyz).success


def test_ids_are_original_and_cancellation():
    xyz=np.vstack((plane(),plane(10,(30,0,6))))
    ids=np.arange(625,len(xyz));reader=lambda part:xyz[part]
    engine=GeometryRefiner(reader,1)
    result=engine.refine_plane(ids)
    np.testing.assert_array_equal(result.final_point_ids,ids)
    cancel=threading.Event();cancel.set()
    with pytest.raises(InterruptedError): engine.refine_plane(ids,cancel=cancel)


def test_repeatable_and_no_full_scene_reader():
    requested=[];xyz=plane()
    def reader(ids): requested.append(ids.copy());return xyz[ids-10000000]
    ids=np.arange(len(xyz))+10000000;engine=GeometryRefiner(reader,1)
    a=engine.refine_plane(ids);b=engine.refine_plane(ids)
    np.testing.assert_array_equal(a.final_point_ids,b.final_point_ids)
    assert all(np.array_equal(x,ids) for x in requested)


def test_reject_scene_sized_candidate_before_reading():
    def forbidden(ids): raise AssertionError('must not read a scene-sized candidate')
    result=GeometryRefiner(forbidden,1).refine_plane(np.arange(200001))
    assert not result.success and '20 万' in result.failure_reason


@pytest.mark.parametrize('crs,factor',[(3857,1.),(6565,1200/3937),('+proj=utm +zone=18 +datum=WGS84 +units=ft +type=crs',.3048)])
def test_crs_unit_reader(crs,factor):
    pyproj=pytest.importorskip('pyproj')
    from types import SimpleNamespace
    from labelCloud.model.scene_spatial import coordinate_scale
    header=SimpleNamespace(parse_crs=lambda:pyproj.CRS(crs))
    np.testing.assert_allclose(coordinate_scale(header),[factor]*3,rtol=1e-12)
