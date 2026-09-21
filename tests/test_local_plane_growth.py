import laspy
import numpy as np
import pytest
from pyproj import CRS

from labelCloud.model.local_plane_growth import GrowthOptions, grow_plane, metric_coordinates


def grid():
    x, y = np.meshgrid(np.arange(15) * .1, np.arange(15) * .1)
    return np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))


def seed_patch(points):
    return ((points[:, 0] >= .2) & (points[:, 0] <= .5) &
            (points[:, 1] >= .2) & (points[:, 1] <= .5))


def test_contiguous_growth_does_not_jump_to_remote_coplanar_roof():
    patch = grid()
    xyz = np.vstack((patch, patch + [5, 0, 0], patch + [0, 0, 1]))
    labels = np.zeros(len(xyz), dtype=np.uint32)
    seeds = seed_patch(xyz) & (xyz[:, 2] == 0)
    before = labels.copy()
    candidate = grow_plane(xyz, labels, seeds, 1)
    assert candidate[:225].sum() == 225
    assert not candidate[225:].any()
    np.testing.assert_array_equal(labels, before)


def test_existing_plane_is_protected_and_cannot_bridge_region():
    xyz = grid()
    labels = np.zeros(len(xyz), dtype=np.uint32)
    labels[(xyz[:, 0] >= .6) & (xyz[:, 0] <= .9)] = 9
    result = grow_plane(xyz, labels, seed_patch(xyz), 1,
                        GrowthOptions(neighbor_radius=.25))
    assert result.any()
    assert not result[labels == 9].any()
    assert not result[xyz[:, 0] >= 1].any()
    with pytest.raises(ValueError, match="其他 Plane"):
        grow_plane(xyz, labels, np.ones(len(xyz), dtype=bool), 1)


def test_same_plane_seeds_can_extend_into_unassigned_points():
    xyz = grid()
    seeds = seed_patch(xyz)
    labels = np.zeros(len(xyz), dtype=np.uint32)
    labels[seeds] = 2
    result = grow_plane(xyz, labels, seeds, 2)
    assert result.sum() == len(xyz) - seeds.sum()
    assert not result[seeds].any()


def test_angle_constraint_rejects_adjoining_sloped_surface():
    flat = grid()
    sloped = grid() + [1.5, 0, 0]
    sloped[:, 2] = (sloped[:, 0] - 1.5) * np.tan(np.radians(35))
    xyz = np.vstack((flat, sloped))
    # 放宽距离，仍应由法向约束阻止扩展到相邻斜面。
    result = grow_plane(xyz, np.zeros(len(xyz)), seed_patch(xyz), 1,
                        GrowthOptions(plane_distance=2, neighbor_radius=.25))
    assert result[:225].sum() > 100
    assert not result[225 + np.flatnonzero(sloped[:, 0] > 2)].any()


@pytest.mark.parametrize("kind", ["few", "line", "disconnected", "noisy"])
def test_bad_seeds_are_rejected(kind):
    xyz = grid()
    seeds = np.zeros(len(xyz), dtype=bool)
    if kind == "few":
        seeds[:3] = True
    elif kind == "line":
        seeds[:12] = True
    elif kind == "disconnected":
        xyz = np.vstack((grid(), grid() + [5, 0, 0]))
        seeds = np.ones(len(xyz), dtype=bool)
    else:
        xyz = np.random.default_rng(42).normal(size=(80, 3))
        seeds = np.ones(len(xyz), dtype=bool)
    with pytest.raises(ValueError):
        grow_plane(xyz, np.zeros(len(xyz)), seeds, 1)


def test_feet_and_meters_give_same_region_and_keep_raw_coordinates():
    xyz = grid()
    foot = laspy.LasData(laspy.LasHeader(point_format=7, version="1.4"))
    foot.header.scales = np.repeat(1e-7, 3)
    foot.header.offsets = [2500000, 400000, 300]
    foot.header.add_crs(CRS.from_epsg(6565))
    for i, dim in enumerate(("x", "y", "z")):
        setattr(foot, dim, xyz[:, i] / (1200 / 3937) + foot.header.offsets[i])
    before = foot.points.array.copy()
    metric, note = metric_coordinates(foot)
    np.testing.assert_allclose(metric, xyz - xyz.mean(axis=0), atol=2e-7)
    result = grow_plane(metric, np.zeros(len(xyz)), seed_patch(xyz), 1)
    expected = grow_plane(xyz, np.zeros(len(xyz)), seed_patch(xyz), 1)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(foot.points.array, before)
    assert "Z" in note


def test_unknown_units_require_explicit_choice():
    las = laspy.LasData(laspy.LasHeader(point_format=7, version="1.4"))
    las.x = [0, 1]; las.y = [0, 1]; las.z = [0, 1]
    with pytest.raises(ValueError, match="单位"):
        metric_coordinates(las)
    meters, _ = metric_coordinates(las, "m")
    feet, _ = metric_coordinates(las, "ft")
    np.testing.assert_allclose(feet, meters * .3048)
