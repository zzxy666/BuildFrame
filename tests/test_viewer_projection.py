from labelCloud.view.viewer import calculate_clipping_planes


def test_dynamic_near_plane_shrinks_during_close_zoom():
    far_near, far_plane = calculate_clipping_planes(100.0, 50.0)
    close_near, close_far = calculate_clipping_planes(0.01, 50.0)

    assert close_near < far_near
    assert 0 < close_near < 0.01 < close_far
    assert far_near < 100.0 < far_plane


def test_clipping_planes_are_valid_for_degenerate_extent():
    near, far = calculate_clipping_planes(0.0, 0.0)
    assert 0 < near < far
