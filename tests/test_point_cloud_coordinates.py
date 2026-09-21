from pathlib import Path

import numpy as np
import pytest

from labelCloud.io.pointclouds import BasePointCloudHandler
from labelCloud.model.point_cloud import PointCloud


class _Handler:
    def __init__(self, points):
        self.points = points

    def read_point_cloud(self, path):
        return self.points.copy(), None


def test_ply_reader_preserves_small_differences_at_large_world_offsets(tmp_path):
    import open3d as o3d
    from labelCloud.io.pointclouds.open3d import Open3DHandler

    points = np.array([[1000000.01, 2000000.02, 300.001],
                       [1000000.03, 2000000.04, 300.002]], dtype=np.float64)
    path = tmp_path / "precise.ply"
    source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    assert o3d.io.write_point_cloud(str(path), source)
    read, _ = Open3DHandler().read_point_cloud(path)
    np.testing.assert_array_equal(read, points)
    cloud = PointCloud.from_file(path, write_buffer=False)
    np.testing.assert_allclose(cloud.to_world_coordinates(cloud.points), points, rtol=0, atol=1e-8)


def test_coordinate_round_trip_without_scaling(monkeypatch):
    points = np.array([[10, 20, 30], [20, 40, 50]], dtype=np.float32)
    monkeypatch.setattr(
        BasePointCloudHandler, "get_handler", lambda extension: _Handler(points)
    )

    cloud = PointCloud.from_file(Path("small.ply"), write_buffer=False)

    assert cloud.applied_scale == 1.0
    np.testing.assert_allclose(
        cloud.to_world_coordinates(cloud.points), points, rtol=0, atol=1e-5
    )


def test_coordinate_round_trip_with_large_source_coordinates(monkeypatch):
    points = np.array(
        [[1_000_000, 2_000_000, 10], [1_020_000, 2_000_000, 30]],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        BasePointCloudHandler, "get_handler", lambda extension: _Handler(points)
    )

    cloud = PointCloud.from_file(Path("large.las"), write_buffer=False)

    assert cloud.applied_scale == pytest.approx(
        np.linalg.norm([20_000.0, 0.0, 20.0]) / 100.0
    )
    np.testing.assert_allclose(
        cloud.to_world_coordinates(cloud.points), points, rtol=0, atol=1e-3
    )


def test_changing_orbit_pivot_does_not_move_current_view(monkeypatch):
    points = np.array(
        [[-2, -1, 0], [4, 3, 2], [1, 0, 1]], dtype=np.float32
    )
    monkeypatch.setattr(
        BasePointCloudHandler, "get_handler", lambda extension: _Handler(points)
    )
    cloud = PointCloud.from_file(Path("pivot.ply"), write_buffer=False)
    cloud.set_rotations(25, 10, 40)
    sample = np.array([0.3, -0.2, 0.5])

    def rendered(point):
        translation = np.array(cloud.get_translation())
        pivot = cloud.orbit_pivot
        return translation + pivot + cloud._rotation_matrix() @ (point - pivot)

    before = rendered(sample)
    cloud.set_orbit_pivot(np.array([1.0, 0.5, -0.25]))

    np.testing.assert_allclose(rendered(sample), before, atol=1e-10)


def test_orbit_keeps_selected_pivot_fixed(monkeypatch):
    points = np.array([[-1, -1, 0], [1, 1, 1]], dtype=np.float32)
    monkeypatch.setattr(
        BasePointCloudHandler, "get_handler", lambda extension: _Handler(points)
    )
    cloud = PointCloud.from_file(Path("pivot.ply"), write_buffer=False)
    pivot = np.array([0.5, 0.25, 0.1])
    cloud.set_orbit_pivot(pivot)
    position_before = np.array(cloud.get_translation()) + pivot

    cloud.set_rotations(80, 0, 120)
    position_after = np.array(cloud.get_translation()) + pivot

    np.testing.assert_allclose(position_after, position_before)


def test_repeated_zoom_never_crosses_focus_point(monkeypatch):
    points = np.array([[-10, -10, -2], [10, 10, 2]], dtype=np.float32)
    monkeypatch.setattr(
        BasePointCloudHandler, "get_handler", lambda extension: _Handler(points)
    )
    cloud = PointCloud.from_file(Path("zoom.ply"), write_buffer=False)

    initial = cloud.focus_distance()
    for _ in range(200):
        cloud.dolly(120)

    closest = cloud.focus_distance()
    assert 0 < closest < initial
    cloud.dolly(-120)
    assert cloud.focus_distance() > closest
