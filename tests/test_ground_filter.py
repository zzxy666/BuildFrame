import numpy as np
import pytest

from PointCloudFilter.ground_filter import GroundFilter


def test_empty_point_cloud_returns_empty_labels():
    labels = GroundFilter().process(np.empty((0, 3), dtype=np.float32))
    assert labels.shape == (0,)


def test_single_xy_cell_does_not_crash():
    points = np.array([[1, 1, 0], [1, 1, 0.05], [1, 1, 10]], dtype=float)
    labels = GroundFilter(z_reserve=0.5).process(points)
    assert labels.tolist() == [2, 2, 1]


def test_sparse_grid_does_not_use_virtual_minus_one_index():
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [50, 50, 20]], dtype=np.float64
    )
    labels = GroundFilter(pixel_size=1.0, z_reserve=0.5).process(points)
    assert labels[-1] == 1


def test_fewer_ground_cells_than_neighbor_count_is_supported():
    points = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0.01]], dtype=float)
    labels = GroundFilter(pixel_size=1.0, k_neighbors=5).process(points)
    assert len(labels) == len(points)


def test_invalid_coordinates_are_rejected():
    with pytest.raises(ValueError, match="NaN"):
        GroundFilter().process(np.array([[0, 0, np.nan]]))
