import numpy as np
import pytest
pytest.importorskip("rasterio")
pytest.importorskip("pyproj")
from affine import Affine
from labelCloud.model.coordinate_mapper import CoordinateMapper


def test_rotated_affine_centers_and_multiple_points():
    affine=Affine.translation(1000,2000)*Affine.rotation(17)*Affine.scale(.2,-.3)
    m=CoordinateMapper(affine,20,20,'EPSG:3857','EPSG:3857')
    c=np.array([1.5,1.5,8.5,-1.]);r=np.array([2.5,2.5,9.5,0.])
    x,y=m.pixel_to_world(c,r)
    np.testing.assert_allclose(m.world_to_pixel(x,y),[c,r],atol=1e-10)
    m.build(4,[(x,y)])
    np.testing.assert_array_equal(m.pixel_point_ids(1,2),[0,1])
    mask=m.polygon_mask([[1,2],[2,2],[2,3],[1,3]])
    np.testing.assert_array_equal(m.pixel_mask_to_candidate_point_ids(mask),[0,1])
    assert not m.valid[3]


def test_crs_units_and_axis_order():
    from pyproj import Transformer
    x,y=2200000.,250000.
    forward=Transformer.from_crs(6565,3857,always_xy=True)
    wx,wy=forward.transform(x,y)
    m=CoordinateMapper(Affine(1,0,wx-10.5,0,-1,wy+20.5),100,100,6565,3857)
    c,r=m.lidar_xy_to_pixel(x,y)
    np.testing.assert_allclose([c,r],[10.5,20.5],atol=1e-6)
    np.testing.assert_allclose(m.pixel_to_lidar_xy(c,r),[x,y],atol=1e-5)
    for lidar,rgb in [(None,3857),(6565,None)]:
        with pytest.raises(ValueError): CoordinateMapper(Affine.identity(),10,10,lidar,rgb)


def test_large_polygon_rejected():
    m=CoordinateMapper(Affine.identity(),100000,100000,3857,3857)
    with pytest.raises(ValueError): m.polygon_mask([[0,0],[99999,0],[99999,99999],[0,99999]])
