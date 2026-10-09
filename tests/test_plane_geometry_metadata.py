import json
import laspy
import numpy as np
from labelCloud.model.scene_planes import RoofPlanes
from labelCloud.model.geometry_refiner import plane_geometry


def test_geometry_lifecycle_and_exact_reader(tmp_path):
    las=laspy.LasData(laspy.LasHeader(point_format=7,version='1.4'))
    x,y=np.meshgrid(np.arange(10),np.arange(10));las.x=x.ravel()+2000000;las.y=y.ravel()+450000;las.z=(x+y).ravel()*.2
    source=tmp_path/'scene.las';las.write(source);model=RoofPlanes(source)
    ids=np.arange(100);raw=np.column_stack((las.x,las.y,las.z))
    np.testing.assert_array_equal(model.point_reader.read(ids[::-1]),raw[::-1])
    model.assign(1,ids=ids[:80]);geom=plane_geometry(raw[:80],1,1);model.attach_plane_geometry(geom)
    model.undo();assert not model.session_metadata['plane_geometry']
    model.redo();assert model.session_metadata['plane_geometry']['1']['point_count']==80
    model.assign(1,ids=ids[80:90]);dirty=model.session_metadata['plane_geometry']['1']
    assert dirty['geometry_dirty'] and 'normal' not in dirty
    model.save();value=model.session_metadata['plane_geometry']['1']
    assert not value['geometry_dirty'] and value['point_count']==90
    model.assign(2,ids=ids[90:]);model.merge(2,1)
    assert model.session_metadata['plane_geometry']['1']['geometry_dirty']
    assert '2' not in model.session_metadata['plane_geometry']
    model.save();assert model.session_metadata['plane_geometry']['1']['point_count']==100
    model.assign(0,'all',ids[:10]);model.save()
    assert model.session_metadata['plane_geometry']['1']['point_count']==90
    model.delete(1);assert '1' not in model.session_metadata['plane_geometry']
    model.undo();assert model.session_metadata['plane_geometry']['1']['point_count']==90
    model.redo();model.save()
    assert json.loads((source.with_suffix('.planar')/'plane_geometry.json').read_text())['planes']=={}
    np.testing.assert_array_equal(laspy.read(source).X,las.X)


def test_preview_replace_is_one_undo():
    from labelCloud.model.preview_edits import PreviewEdits
    edits=PreviewEdits();base=np.arange(10)
    edits.replace(base,base[:5]);assert len(edits.history)==1
    np.testing.assert_array_equal(edits.compose(base,[])[0],base[:5])
    edits.undo();np.testing.assert_array_equal(edits.compose(base,[])[0],base)
    edits.redo();np.testing.assert_array_equal(edits.compose(base,[])[0],base[:5])
