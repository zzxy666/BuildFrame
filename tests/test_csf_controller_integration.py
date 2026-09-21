"""Exercise real QProcess lifecycle without needing an OpenGL display."""

import os
import subprocess
import sys


def test_async_filter_success_cancel_switch_and_failure():
    script = r'''
from pathlib import Path
from types import SimpleNamespace
import time
from unittest.mock import patch
import numpy as np
from PyQt5 import QtCore, QtWidgets
from labelCloud.control.filter_controller import PointCloudFilterController
from labelCloud.control.config_manager import config

app = QtWidgets.QApplication([])
view = QtWidgets.QWidget()
view.button_point_cloud_filtering = QtWidgets.QPushButton(view)
view.gl_widget = SimpleNamespace(makeCurrent=lambda: None, update=lambda: None)
view.status_manager = SimpleNamespace(set_message=lambda *a, **k: None)

def cloud():
    x, y = np.meshgrid(np.arange(11.), np.arange(11.))
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    points = np.concatenate((points, [[5., 5., 4.], [6., 5., 4.]]))
    colors = np.random.default_rng(2).random(points.shape).astype(np.float32)
    # Exercise undoing display normalization, not just native direct calls.
    return SimpleNamespace(points=points / 100, applied_scale=100., colors=colors,
                           create_buffers=lambda: None)

pc = cloud()
owner = SimpleNamespace(pcd_manager=SimpleNamespace(pointcloud=pc))
controller = PointCloudFilterController(owner)
controller.set_view(view)
errors = []
controller._show_error = errors.append

def wait_done():
    deadline = time.monotonic() + 20
    while controller.worker is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert controller.worker is None, 'worker did not finish'
    app.processEvents()

original = pc.colors.copy()
original_points = pc.points.copy()
controller.toggle()
job = controller.worker
assert job is not None
wait_done()
assert not errors, errors
assert not Path(job.directory.name).exists()
assert controller.filtered_pointcloud is pc
np.testing.assert_array_equal(pc.ground_labels[:-2], 2)
np.testing.assert_array_equal(pc.ground_labels[-2:], 1)
np.testing.assert_array_equal(pc.points, original_points)
np.testing.assert_array_equal(pc.backup_colors, original)
controller.toggle()
np.testing.assert_array_equal(pc.colors, original)
assert not hasattr(pc, 'ground_labels')

# Cancel immediately, including QProcess's Starting state.
controller.toggle()
job = controller.worker
controller.toggle()
app.processEvents()
assert controller.worker is None
assert not Path(job.directory.name).exists()
assert not errors, errors
assert controller.filtered_pointcloud is None
np.testing.assert_array_equal(pc.colors, original)

# Switching clouds must cancel and never apply an old result to the new cloud.
controller.toggle()
job = controller.worker
other = cloud()
owner.pcd_manager.pointcloud = other
controller.reset_for_pointcloud()
app.processEvents()
assert controller.worker is None
assert not Path(job.directory.name).exists()
controller._apply_result(pc, np.ones(len(pc.points), dtype=np.int8))
assert not hasattr(other, 'ground_labels')
assert not hasattr(pc, 'ground_labels')

# A native validation error returns to the UI without changing any data.
other.points[0, 0] = np.nan
controller.toggle()
wait_done()
assert errors and 'NaN' in errors[-1], errors
assert not hasattr(other, 'ground_labels')
assert view.button_point_cloud_filtering.isEnabled()

# FailedToStart has no finished signal: it still needs cleanup and reset.
errors.clear()
owner.pcd_manager.pointcloud = cloud()
with patch.object(controller, '_cleanup', wraps=controller._cleanup) as cleanup:
    with patch('labelCloud.control.filter_controller.sys', SimpleNamespace(executable='__missing_csf_python__')):
        controller.toggle()
        wait_done()
    assert cleanup.called
    job = cleanup.call_args.args[0]
assert errors
assert not Path(job.directory.name).exists()

# Closing while filtering stops the process and keeps the original cloud.
errors.clear()
controller.toggle()
job = controller.worker
controller.shutdown()
app.processEvents()
assert controller.worker is None
assert not Path(job.directory.name).exists()
assert not errors
print('QProcess integration OK')
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen", "OMP_NUM_THREADS": "2",
             "BUILDFRAME_LOG_PATH": "NUL" if os.name == "nt" else os.devnull,
             "PYTHONIOENCODING": "utf-8"},
        capture_output=True, text=True, encoding="utf-8", timeout=70,
    )
    assert result.returncode == 0, result.stdout + result.stderr
