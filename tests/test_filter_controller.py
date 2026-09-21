from labelCloud.control.filter_controller import PointCloudFilterController


class _Button:
    def __init__(self):
        self.text = None
        self.enabled = None

    def setText(self, text):
        self.text = text

    def setEnabled(self, enabled):
        self.enabled = enabled


class _View:
    def __init__(self):
        self.button_point_cloud_filtering = _Button()


class _Owner:
    pcd_manager = object()


def test_switching_pointcloud_resets_filter_state():
    controller = PointCloudFilterController(_Owner())
    controller.set_view(_View())
    controller.filtered_pointcloud = object()

    controller.reset_for_pointcloud()

    assert controller.filtered_pointcloud is None
    assert controller.view.button_point_cloud_filtering.text == "点云滤波"
    assert controller.view.button_point_cloud_filtering.enabled
