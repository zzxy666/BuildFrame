import numpy as np
import laspy
import pytest

from labelCloud.model.roof_planes import RoofPlanes, polygon_mask, project_points


def source_file(path, count=12):
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = [.001, .002, .003]
    header.offsets = [2500000, 400000, 100]
    header.vlrs.append(laspy.VLR(user_id="test", record_id=19, record_data=b"keep me"))
    cloud = laspy.LasData(header)
    cloud.X = np.arange(count, dtype=np.int32)
    cloud.Y = np.arange(count, dtype=np.int32) * 2
    cloud.Z = np.arange(count, dtype=np.int32) * 3
    cloud.intensity = np.arange(count) + 100
    cloud.classification = np.arange(count) % 7
    cloud.return_number = np.arange(count) % 3 + 1
    cloud.number_of_returns = np.full(count, 3)
    cloud.key_point = np.arange(count) % 2
    cloud.overlap = np.arange(count) % 2
    cloud.scanner_channel = np.arange(count) % 4
    cloud.gps_time = np.arange(count) + 123456.789
    cloud.red = np.arange(count) * 100
    cloud.green = np.arange(count) * 200
    cloud.blue = np.arange(count) * 300
    cloud.add_extra_dim(laspy.ExtraBytesParams(name="planeid", type=np.uint8))
    cloud.planeid = np.zeros(count, dtype=np.uint8)
    cloud.write(path)
    return cloud


def test_edit_merge_delete_undo_and_zero(tmp_path):
    path = tmp_path / "roof.las"
    source_file(path)
    model = RoofPlanes(path)
    assert model.new_plane() == 1
    model.selection[:4] = True
    model.assign(1)
    model.selection[4:8] = True
    model.assign(2)
    model.merge(2, 1)
    np.testing.assert_array_equal(model.labels[:8], 1)
    model.undo()
    np.testing.assert_array_equal(model.labels[4:8], 2)
    model.delete(1)
    np.testing.assert_array_equal(model.labels[:4], 0)
    model.undo()
    np.testing.assert_array_equal(model.labels[:4], 1)
    model.assign(0, scope="all")  # 主动解除已标注保护后纠错。
    np.testing.assert_array_equal(model.labels[:4], 0)
    assert model.dirty


@pytest.mark.parametrize("suffix", [".las", ".laz"])
def test_save_resume_preserves_all_original_fields(tmp_path, suffix):
    path = tmp_path / ("roof" + suffix)
    source = source_file(path)
    original_bytes = path.read_bytes()
    model = RoofPlanes(path)
    model.selection[[1, 4, 7]] = True
    model.assign(300)  # 不是 uint8；超过 9 的 Plane 也能保存。
    model.save()
    output = model.export()
    saved = laspy.read(output)
    assert path.read_bytes() == original_bytes
    for field in source.points.array.dtype.names:
        np.testing.assert_array_equal(source.points.array[field], saved.points.array[field])
    assert saved.header.version == source.header.version
    assert saved.point_format.id == source.point_format.id
    np.testing.assert_array_equal(saved.header.scales, source.header.scales)
    np.testing.assert_array_equal(saved.header.offsets, source.header.offsets)
    assert saved.header.vlrs[0].record_data_bytes() == b"keep me"
    assert saved.plane_id.dtype == np.dtype("uint32")
    assert not model.dirty
    resumed = RoofPlanes(path)
    np.testing.assert_array_equal(resumed.labels, model.labels)
    model.undo()
    assert model.dirty
    assert not model.labels.any()


def test_failed_save_does_not_replace_previous_gt(tmp_path, monkeypatch):
    import labelCloud.model.roof_planes as module
    path = tmp_path / "roof.las"
    source_file(path)
    model = RoofPlanes(path)
    output = model.save()
    original = output.read_bytes()
    model.selection[:] = True
    model.assign(1)
    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError):
        model.save()
    assert output.read_bytes() == original
    assert model.dirty
    assert not list(tmp_path.glob(".roof-plane-*"))


def test_reject_mismatched_existing_output(tmp_path):
    path = tmp_path / "roof.las"
    source_file(path)
    source_file(tmp_path / "roof_gt.las", count=5)
    with pytest.raises(ValueError, match="不匹配"):
        RoofPlanes(path)


def test_projection_clipping_and_concave_lasso():
    points = np.array([[0, 0, 0], [-.5, .5, 0], [2, 0, 0], [0, 0, 2]])
    screen, valid = project_points(points, np.eye(4), np.eye(4), 400, 200)
    np.testing.assert_allclose(screen[:2], [[200, 100], [100, 50]])
    np.testing.assert_array_equal(valid, [True, True, False, False])
    samples = np.array([[1, 1], [3, 3], [1, 3], [0, 0], [5, 5]])
    polygon = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
    np.testing.assert_array_equal(polygon_mask(samples, polygon), [True, False, True, True, False])
    # OpenGL 查询出的矩阵以转置布局存放；本例平移到视锥之外。
    matrix = np.eye(4)
    matrix[3, 0] = 3
    _, valid = project_points(points[:1], matrix, np.eye(4), 400, 200)
    assert not valid.any()
