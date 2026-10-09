import numpy as np
from labelCloud.model.preview_edits import PreviewEdits


def test_edits_survive_levels_and_undo_branch():
    edits=PreviewEdits()
    edits.edit(np.array([9]),True)
    edits.edit(np.array([2,4]),False)
    yellow,orange=edits.compose(np.array([1,2]),np.array([3,4,5]))
    np.testing.assert_array_equal(yellow,[1,9])
    np.testing.assert_array_equal(orange,[3,5])
    yellow,orange=edits.compose(np.array([1,2]),np.array([],dtype=int))
    np.testing.assert_array_equal(yellow,[1,9])
    assert edits.undo()
    np.testing.assert_array_equal(edits.compose(np.array([1,2]),np.array([4]))[0],[1,2,9])
    assert edits.redo()
    assert 2 not in edits.compose(np.array([1,2]),np.array([4]))[0]
    edits.undo();edits.edit(np.array([7]),True)
    assert not edits.redo()
