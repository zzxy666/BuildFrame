import numpy as np
from test_roof_planes import source_file
from labelCloud.model.scene_planes import RoofPlanes


def test_snapshot_preserves_new_edits_and_revision(tmp_path):
    path=tmp_path/'roof.las';source_file(path)
    m=RoofPlanes(path);m.save()
    m.selection[:3]=True;m.assign(1)
    snapshot=m.session.snapshot(m.labels,m.session_metadata)
    revision=m.revision
    m.selection[1:5]=True;m.assign(2,scope='all')
    m.session.write_snapshot(snapshot)
    del m.session.pending[:snapshot[4]]
    m.saved_revision=revision
    assert m.dirty and m.session.pending
    disk=RoofPlanes(path)
    assert np.all(disk.labels[:3]==1) and not disk.labels[3:].any()
    assert np.all(m.labels[1:5]==2)
    m.save()
    np.testing.assert_array_equal(RoofPlanes(path).labels,m.labels)


def test_initial_snapshot_is_independent(tmp_path):
    path=tmp_path/'roof.las';source_file(path)
    m=RoofPlanes(path)
    snapshot=m.session.snapshot(m.labels,m.session_metadata)
    m.selection[:3]=True;m.assign(1)
    m.session.write_snapshot(snapshot)
    assert not RoofPlanes(path).labels.any()
    assert m.session.pending
    m.save()
    np.testing.assert_array_equal(RoofPlanes(path).labels,m.labels)
