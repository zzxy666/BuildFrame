"""Asynchronous CSF classification with cancellable native-process isolation."""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import sys
import tempfile

import numpy as np
from PyQt5 import QtCore, QtWidgets

from PointCloudFilter.csf_filter import CSFOptions
from .config_manager import config
from labelCloud.view.i18n import tr


@dataclass
class _FilterJob:
    process: QtCore.QProcess
    directory: tempfile.TemporaryDirectory
    pointcloud: object
    cancelled: bool = False
    output: str = ""
    cleaned: bool = False


class PointCloudFilterController:
    def __init__(self, owner) -> None:
        self.owner = owner
        self.view = None
        self.worker = None
        self.filtered_pointcloud = None

    def set_view(self, view) -> None:
        self.view = view

    def reset_for_pointcloud(self) -> None:
        self.cancel(announce=False)
        self.filtered_pointcloud = None
        if self.view is not None:
            self.view.button_point_cloud_filtering.setText(tr("点云滤波"))
            self.view.button_point_cloud_filtering.setEnabled(True)

    def _status(self, source, **values):
        if self.view is not None and hasattr(self.view, "status_manager"):
            self.view.status_manager.set_message(source, **values)

    def toggle(self) -> None:
        if self.worker is not None:
            self.cancel()
            return
        pointcloud = self.owner.pcd_manager.pointcloud
        if pointcloud is None:
            return
        if self.filtered_pointcloud is pointcloud:
            self._restore(pointcloud)
            return
        if pointcloud.points is None or len(pointcloud.points) == 0:
            QtWidgets.QMessageBox.warning(self.view, tr("滤波失败"), tr("当前点云为空。"))
            return

        directory = None
        try:
            options = CSFOptions.from_config(config)
            directory = tempfile.TemporaryDirectory(prefix="buildframe-csf-")
            points_path = Path(directory.name) / "points.npy"
            # Snapshot the point order; normalize/validate in the child process.
            np.save(points_path, pointcloud.points, allow_pickle=False)
            parent = self.view if isinstance(self.view, QtCore.QObject) else None
            process = QtCore.QProcess(parent)
            process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
            env = QtCore.QProcessEnvironment.systemEnvironment()
            env.insert("PYTHONIOENCODING", "utf-8")
            if not env.contains("OMP_NUM_THREADS"):
                env.insert("OMP_NUM_THREADS", "4")
            process.setProcessEnvironment(env)
            process.setWorkingDirectory(str(Path(__file__).resolve().parents[2]))
            process.setProgram(sys.executable)
            process.setArguments([
                "-m", "PointCloudFilter.csf_worker", str(points_path),
                str(Path(directory.name) / "labels.npy"),
                json.dumps(asdict(options)), str(pointcloud.applied_scale),
            ])
            job = _FilterJob(process, directory, pointcloud)
            self.worker = job
            process.readyReadStandardOutput.connect(lambda: self._read_output(job))
            process.finished.connect(lambda code, status: self._complete(job, code, status))
            process.errorOccurred.connect(lambda error: self._process_error(job, error))
            self.retranslate()
            self._status("开始 CSF 地面分离，可再次点击按钮取消。")
            logging.info("Starting CSF ground filtering: %s", asdict(options))
            process.start()
        except Exception as exc:
            self.worker = None
            if directory is not None:
                directory.cleanup()
            self.retranslate()
            self._show_error(str(exc))

    def _read_output(self, job):
        output = bytes(job.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        job.output = (job.output + output)[-6000:]

    def _process_error(self, job, error):
        # FailedToStart has no finished signal; crashes do.
        if error == QtCore.QProcess.FailedToStart:
            job.output += job.process.errorString()
            self._complete(job, -1, QtCore.QProcess.CrashExit)

    def _complete(self, job, exit_code, exit_status):
        if job.cleaned:
            return
        self._read_output(job)
        error = None
        try:
            if job.cancelled or self.worker is not job:
                return
            if self.owner.pcd_manager.pointcloud is not job.pointcloud:
                return
            if exit_code != 0 or exit_status != QtCore.QProcess.NormalExit:
                raise RuntimeError(job.output.strip() or job.process.errorString())
            labels = np.load(Path(job.directory.name) / "labels.npy", allow_pickle=False)
            self._apply_result(job.pointcloud, labels)
        except Exception as exc:
            error = str(exc)
        finally:
            if self.worker is job:
                self.worker = None
            self._cleanup(job)
            self.retranslate()
        if error:
            self._show_error(error)

    def _cleanup(self, job):
        if job.cleaned:
            return
        job.cleaned = True
        try:
            job.directory.cleanup()
        except OSError:
            logging.warning("Could not clean CSF temporary files", exc_info=True)
        job.process.deleteLater()

    def cancel(self, announce=True):
        job = self.worker
        if job is None:
            return
        job.cancelled = True
        self.worker = None
        if job.process.state() != QtCore.QProcess.NotRunning:
            job.process.kill()
            # Wait briefly for Windows to release its mapped input file.
            job.process.waitForFinished(3000)
        if not job.cleaned and job.process.state() == QtCore.QProcess.NotRunning:
            self._cleanup(job)
        self.retranslate()
        if announce:
            self._status("已取消地面滤波。")

    def shutdown(self):
        self.cancel(announce=False)

    def _refresh_cloud(self, pointcloud):
        self.view.gl_widget.makeCurrent()
        pointcloud.create_buffers()
        self.view.gl_widget.update()

    def _apply_result(self, pointcloud, ground_labels) -> None:
        if self.owner.pcd_manager.pointcloud is not pointcloud:
            return
        labels = np.asarray(ground_labels)
        if labels.shape != (len(pointcloud.points),) or not np.isin(labels, [1, 2]).all():
            raise ValueError("CSF returned invalid labels")
        original_colors = pointcloud.colors.copy()
        colors = np.zeros_like(pointcloud.points, dtype=np.float32)
        colors[labels == 2] = [0.0, 1.0, 0.0]
        colors[labels == 1] = [1.0, 0.0, 0.0]
        pointcloud.colors = colors
        try:
            self._refresh_cloud(pointcloud)
        except Exception:
            pointcloud.colors = original_colors
            raise
        pointcloud.backup_colors = original_colors
        pointcloud.ground_labels = labels.astype(np.int8, copy=True)
        self.filtered_pointcloud = pointcloud
        ground_count = int(np.count_nonzero(labels == 2))
        logging.info("CSF completed: ground=%d non-ground=%d", ground_count, len(labels) - ground_count)
        self._status(
            "地面分离完成：地面 {ground} 点，非地面 {other} 点（绿色 / 红色）。",
            ground=ground_count, other=len(labels) - ground_count,
        )
        self.retranslate()

    def _restore(self, pointcloud) -> None:
        backup = getattr(pointcloud, "backup_colors", None)
        if backup is None:
            self.reset_for_pointcloud()
            return
        pointcloud.colors = backup
        self._refresh_cloud(pointcloud)
        del pointcloud.backup_colors
        if hasattr(pointcloud, "ground_labels"):
            del pointcloud.ground_labels
        self.filtered_pointcloud = None
        self._status("")
        self.retranslate()

    def _show_error(self, details: str) -> None:
        logging.error("Ground filtering failed:\n%s", details)
        summary = details.splitlines()[-1] if details else "CSF"
        self._status("地面滤波失败：{error}", error=summary)
        message = QtWidgets.QMessageBox(self.view)
        message.setIcon(QtWidgets.QMessageBox.Critical)
        message.setWindowTitle(tr("滤波失败"))
        message.setText(tr("地面滤波失败：{error}", error=summary))
        message.setDetailedText(details)
        message.exec_()

    def retranslate(self) -> None:
        if self.view is None:
            return
        if self.worker is not None:
            text = tr("取消滤波")
        elif self.filtered_pointcloud is not None and self.filtered_pointcloud is self.owner.pcd_manager.pointcloud:
            text = tr("恢复原始颜色")
        else:
            text = tr("点云滤波")
        self.view.button_point_cloud_filtering.setText(text)
        self.view.button_point_cloud_filtering.setEnabled(True)
