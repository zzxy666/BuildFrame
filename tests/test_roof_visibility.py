import laspy
import numpy as np

from labelCloud.model.roof_planes import RoofPlanes
from labelCloud.model.local_plane_growth import grow_plane, GrowthOptions
from test_roof_planes import source_file


def test_hidden_undo_interleaved_with_labels_and_full_save(tmp_path):
    path = tmp_path / "roof.las"
    source = source_file(path)
    model = RoofPlanes(path)
    model.selection[[1, 3]] = True
    model.set_hidden(model.selection.copy())
    assert not model.dirty
    assert not model.selection.any()
    model.selection[[1, 2]] = True  # 隐藏点即使残留在选区也不能赋值。
    model.assign(1)
    assert model.labels[1] == 0 and model.labels[2] == 1
    model.set_hidden(np.zeros(12, dtype=bool))
    model.undo()  # 撤销恢复
    assert model.hidden_mask[[1, 3]].all()
    model.undo()  # 撤销赋值
    assert not model.labels.any()
    model.undo()  # 撤销隐藏，恢复原选区
    assert not model.hidden_mask.any()
    assert model.selection[[1, 3]].all()
    model.set_hidden(np.ones(12, dtype=bool))
    model.save()
    model.export()
    saved = laspy.read(model.output_path)
    assert len(saved.points) == len(source.points)
    for name in source.points.array.dtype.names:
        np.testing.assert_array_equal(source.points.array[name], saved.points.array[name])
    assert not np.asarray(saved.plane_id).any()
    assert "hidden_mask" not in list(saved.point_format.dimension_names)
    assert not RoofPlanes(path).hidden_mask.any()


def test_default_protection_and_explicit_correction(tmp_path):
    path = tmp_path / "roof.las"
    source_file(path)
    model = RoofPlanes(path)
    model.selection[:3] = True
    model.assign(1)
    model.selection[:5] = True
    model.assign(2)
    np.testing.assert_array_equal(model.labels[:5], [1, 1, 1, 2, 2])
    model.current = 1
    model.selection[:] = True
    model.assign(3, scope="current")
    np.testing.assert_array_equal(model.labels[:5], [3, 3, 3, 2, 2])
    model.selection[:5] = True
    model.assign(0, scope="all")
    assert not model.labels.any()


def test_hidden_and_labelled_barriers_cannot_bridge_growth():
    x, y = np.meshgrid(np.arange(21)*.1, np.arange(15)*.1)
    xyz = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    seeds = (xyz[:, 0] >= .1) & (xyz[:, 0] <= .4) & (xyz[:, 1] >= .2) & (xyz[:, 1] <= .5)
    barrier = (xyz[:, 0] >= .7) & (xyz[:, 0] <= 1.2)
    labels = np.zeros(len(xyz), dtype=np.uint32)
    result = grow_plane(xyz, labels, seeds, 1, GrowthOptions(neighbor_radius=.25), available=~barrier)
    assert result.any()
    assert not result[barrier | (xyz[:, 0] > 1.2)].any()
    labels[barrier] = 1  # 即使是当前 Plane，未手选的标签点也不作生长桥梁。
    result = grow_plane(xyz, labels, seeds, 1, GrowthOptions(neighbor_radius=.25))
    assert not result[barrier | (xyz[:, 0] > 1.2)].any()


def test_display_mask_preserves_original_indices_and_arrays(tmp_path):
    from unittest.mock import patch
    from labelCloud.model.point_cloud import PointCloud
    xyz = np.arange(30, dtype=np.float32).reshape(10, 3)
    colors = np.ones_like(xyz)
    cloud = PointCloud(tmp_path / "roof.las", xyz, colors, write_buffer=False)
    cloud.position_vbo, cloud.color_vbo, cloud.label_vbo = 1, 2, 3
    mask = np.zeros(10, dtype=bool)
    mask[[2, 7, 9]] = True
    cloud.set_display_mask(mask)
    assert cloud.points is xyz and cloud.colors is colors
    np.testing.assert_array_equal(cloud.display_indices, [2, 7, 9])
    with patch('labelCloud.model.point_cloud.GL') as gl:
        cloud.draw_pointcloud()
        args = gl.glDrawElements.call_args.args
        assert args[1] == 3
        np.testing.assert_array_equal(args[3], [2, 7, 9])
        gl.glDrawArrays.assert_not_called()
        cloud.set_display_mask(None)
        cloud.draw_pointcloud()
        gl.glDrawArrays.assert_called_once()
