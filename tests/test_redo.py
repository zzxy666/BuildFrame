import numpy as np
from test_roof_planes import source_file
from labelCloud.model.scene_planes import RoofPlanes


def test_redo_labels_hidden_delete_and_branch(tmp_path):
    path=tmp_path/'roof.las';source_file(path)
    m=RoofPlanes(path)
    assert not m.redo()
    m.new_plane();m.selection[:4]=True;m.assign(1)
    labelled=m.labels.copy()
    m.selection[4:7]=True;m.set_hidden_ids(np.arange(4,7),True)
    assert m.undo() and not m.hidden_mask.any()
    assert m.redo() and m.hidden_count==3
    assert not m.selection.any()
    m.set_hidden_ids(np.arange(4,7),False)
    m.undo();assert m.hidden_count==3
    m.redo();assert m.hidden_count==0
    m.delete(1);assert 1 not in m.plane_ids
    m.undo();np.testing.assert_array_equal(m.labels,labelled)
    m.redo();assert 1 not in m.plane_ids and not m.labels.any()
    m.undo();m.new_plane();assert not m.redo()


def test_multiple_redo_empty_planes_merge_and_save(tmp_path):
    path=tmp_path/'roof.las';source_file(path)
    m=RoofPlanes(path)
    m.new_plane();m.selection[:3]=True;m.assign(1)
    m.new_plane();m.selection[3:6]=True;m.assign(2)
    m.merge(2,1);expected=m.labels.copy()
    for _ in range(5): assert m.undo()
    assert not m.labels.any()
    for _ in range(5): assert m.redo()
    np.testing.assert_array_equal(m.labels,expected)
    assert 2 not in m.plane_ids and m.plane_counts[1]==6
    m.save();np.testing.assert_array_equal(RoofPlanes(path).labels,expected)
    m.undo();m.redo();np.testing.assert_array_equal(m.labels,expected)
    assert m.history_bytes+m.redo_bytes<=m.UNDO_LIMIT
