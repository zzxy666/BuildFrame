"""SAM 只生成二维候选；所有 GT 写入继续走现有三维确认。"""
import copy
import logging
from dataclasses import replace
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from .scene_worker import SceneTask
from ..model.segmentation.engine import SegmentationEngine
from ..model.segmentation.patch import SAMPatch, PromptHistory, SAMState
from ..model.segmentation.sam2_backend import availability
from ..view.sam_panel import SAMPanel, load_config, configure, save_config


class SAMController(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, str)

    def __init__(self, rgb):
        super().__init__(rgb.panel)
        self.rgb = rgb;self.viewer = rgb.panel.viewer
        self.panel = SAMPanel(rgb.panel);rgb.panel.controls_layout.insertWidget(2, self.panel)
        self.config = load_config();self.engine = SegmentationEngine()
        self.patch = None;self.prompts = PromptHistory();self.prediction = None;self.mask_index = 0
        self.state = SAMState.IDLE;self.generation = 0;self.task = None;self.pending = None
        self.target = None;self.drag = None;self.source = None;self.guard = False
        self.mask_item = self.viewer.scene().addPixmap(QtGui.QPixmap());self.mask_item.setZValue(5)
        pen = QtGui.QPen(QtGui.QColor("#00ff66"), 2, QtCore.Qt.DashLine);pen.setCosmetic(True)
        self.drag_item = self.viewer.scene().addRect(QtCore.QRectF(), pen);self.drag_item.setZValue(8);self.drag_item.hide()
        self.items = []
        rgb.panel.tool.addItems(["SAM Point", "SAM Box"])
        self.available, self.reason = availability()
        self.update_availability()
        self.panel.settings.clicked.connect(self.settings)
        self.panel.reset.clicked.connect(self.reset_prompts)
        self.panel.cancel.clicked.connect(self.reset_prompts)
        self.panel.region.clicked.connect(self.reset_region)
        self.panel.accept.clicked.connect(self.accept)
        self.panel.next_mask.clicked.connect(self.next_mask)
        self.panel.cpu.clicked.connect(self.retry_cpu)
        rgb.panel.tool.currentIndexChanged.connect(self.mode_changed)
        self.progress.connect(self.show_progress)
        self.viewer.installEventFilter(self);self.viewer.viewport().installEventFilter(self)

    @property
    def active(self):
        from PyQt5 import sip
        # 窗口销毁期间仍可能收到事件，不能再访问已销毁的下拉框。
        tool = self.rgb.panel.tool
        return not sip.isdeleted(tool) and tool.currentIndex() in (2, 3)

    def update_availability(self):
        for index in (2, 3):
            item = self.rgb.panel.tool.model().item(index)
            item.setEnabled(self.available);item.setToolTip(self.reason)
        self.panel.status.setText("SAM: IDLE" if self.available else self.reason)

    def settings(self):
        if self.rgb.roof.candidate is not None: return
        result = configure(self.rgb.panel, self.config, self.reason)
        if result is not None and result != self.config:
            if not self.discard_question(): return
            self.reset_region();self.config = result;save_config(result)
        self.available, self.reason = availability();self.update_availability()
        if self.active and self.available: self.schedule()

    def discard_question(self):
        if not self.prompts.events: return True
        return QtWidgets.QMessageBox.question(self.rgb.panel, "SAM 尚未接受", "当前 SAM 提示 / mask 尚未接受，是否丢弃？", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No) == QtWidgets.QMessageBox.Yes

    def mode_changed(self, index):
        self.panel.controls.setVisible(index in (2, 3))
        self.rgb.panel.hint.setText("SAM：左键正点 / 拖拽 Box；右键负点；中键平移；Backspace 删除提示，R 重置，Enter 接受 mask，Esc 取消。" if index in (2, 3) else "RGB：滚轮缩放；中/右键平移。Polygon：左键加点，双击/Enter 闭合，Backspace 删顶点，Esc 取消。")
        if index in (2, 3):
            if not self.available: return
            self.viewer.clear_polygon();self.rgb.polygon = None;self.rgb.panel.map_button.setEnabled(False)
            self.viewer.setFocus()
            if self.engine.backend is None and not self.task: self.schedule()
        else:
            if self.prompts.events and not self.discard_question():
                blocker = QtCore.QSignalBlocker(self.rgb.panel.tool);self.rgb.panel.tool.setCurrentIndex(2)
                self.viewer.drawing = False;self.panel.controls.show();return
            self.reset_prompts()

    def show_progress(self, generation, value):
        if generation == self.generation: self.panel.status.setText("SAM: "+value)

    def invalidate(self):
        self.generation += 1;self.pending = None;self.prediction = None
        self.panel.accept.setEnabled(False);self.panel.next_mask.setEnabled(False)
        self.mask_item.setPixmap(QtGui.QPixmap())

    def reset_prompts(self):
        if self.rgb.roof.candidate is not None: return
        self.invalidate();self.prompts.clear();self.target = None
        self.drag = None;self.drag_item.hide()
        self.state = SAMState.PATCH_READY if self.patch else SAMState.IDLE
        self.panel.status.setText("SAM: "+self.state.value if self.available else self.reason);self.draw()

    def reset_region(self):
        if self.rgb.roof.candidate is not None: return
        self.reset_prompts()
        if self.patch is not None: self.patch.encoded = False
        self.patch = None;self.drag = None;self.drag_item.hide();self.state = SAMState.IDLE;self.draw()
        # 清缓存与推理串行，不能在 GUI 线程修改正在使用的 predictor。
        self.pending = (self.generation, None)
        self.launch()

    def sync(self):
        source = (id(self.rgb.photo), id(self.rgb.roof.model))
        if self.source != source:
            self.source = source;self.reset_region()
        model = self.rgb.roof.model
        if model and self.target is not None and model.current != self.target and self.prompts.events and self.state != SAMState.LIDAR_PREVIEW and not self.guard:
            self.guard = True
            if self.discard_question(): self.reset_prompts()
            else:
                model.current = self.target
                QtCore.QTimer.singleShot(0, self.rgb.roof.refresh)
            self.guard = False
        locked = self.rgb.roof.candidate is not None or bool(self.rgb.roof.task)
        self.panel.controls.setEnabled(not locked)
        self.panel.settings.setEnabled(not locked)
        self.rgb.panel.tool.setEnabled(not locked)

    def ensure_patch(self, point):
        photo = self.rgb.photo
        if photo is None or self.rgb.roof.model is None: raise ValueError("请先加载 LiDAR 和 GeoTIFF")
        if not (0 <= point[0] < photo.width and 0 <= point[1] < photo.height): raise ValueError("点击位于影像外")
        if self.patch is not None and not self.patch.contains(point):
            if QtWidgets.QMessageBox.question(self.rgb.panel, "切换 SAM 区域", "点击位于当前 patch 外，是否丢弃旧提示并切换区域？", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No) != QtWidgets.QMessageBox.Yes: return False
            self.reset_region()
        if self.patch is None:
            self.patch = SAMPatch.around(point, photo.width, photo.height, self.config.patch_size)
            self.state = SAMState.PATCH_READY
        if self.target is None: self.target = self.rgb.roof.model.current
        return True

    def add_point(self, point, positive):
        try:
            if not self.ensure_patch(point): return
            self.prompts.events.append((1 if positive else 0, tuple(point)))
            self.schedule()
        except Exception as exc: self.error(exc)

    def add_box(self, box):
        try:
            x1, y1, x2, y2 = box
            box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            if box[0] >= box[2] or box[1] >= box[3]: raise ValueError("Box 必须有面积")
            if not self.ensure_patch(((box[0]+box[2])/2, (box[1]+box[3])/2)): return
            self.patch.local_box(box)
            self.prompts.events.append(("box", box));self.schedule()
        except Exception as exc: self.error(exc)

    def schedule(self):
        self.invalidate();self.state = SAMState.PROMPT_EDITING if self.prompts.events else (SAMState.PATCH_READY if self.patch else SAMState.IDLE)
        self.draw()
        self.pending = (self.generation, (self.config, self.rgb.photo, self.patch, copy.deepcopy(self.prompts)))
        self.launch()

    def launch(self):
        if self.task is not None or self.pending is None: return
        generation, args = self.pending;self.pending = None
        model=self.rgb.roof.model
        protection=model.dual_gt.state if model is not None else None
        target=model.current if model is not None else 0
        allow_current=self.rgb.roof.dual.replace.isChecked() if hasattr(self.rgb.roof,'dual') else False
        def work():
            if args is None: self.engine.clear_image();return None
            result=self.engine.run(*args, progress=lambda value:self.progress.emit(generation, value))
            if result is not None and model is not None:
                from ..model.dual_gt import rgb_identity
                from ..model.coordinate_mapper import PixelMask
                from ..model.segmentation.base_backend import Prediction
                patch=args[2]
                available,_=model.dual_gt.unlabelled_mask(PixelMask(np.ones((patch.height,patch.width),bool),patch.col0,patch.row0),
                    rgb_identity(args[1].path),target,allow_current,state=protection)
                result=Prediction(result.masks & available.data[None,:,:],result.scores)
            return result
        task = SceneTask(work);self.task = task
        def done(value, error):
            self.task = None
            if generation == self.generation:
                if args is not None and model is not None and model.dual_gt.state is not protection:
                    self.schedule();return
                if error: self.error(error[0]);logging.debug("SAM worker error: %s", error[1])
                elif args is not None:
                    self.prediction = value
                    if value is not None:
                        self.mask_index = value.best();self.state = SAMState.MASK_PREVIEW
                    self.draw();self.ready_status()
            self.launch()
        task.signals.finished.connect(done)
        QtCore.QThreadPool.globalInstance().start(task)

    def ready_status(self):
        device = self.engine.backend.device if self.engine.backend else "unknown"
        count = 0 if self.prediction is None else int(self.prediction.masks[self.mask_index].sum())
        positive = sum(kind == 1 for _, kind in self.prompts.points)
        negative = sum(kind == 0 for _, kind in self.prompts.points)
        self.panel.status.setText(f"SAM: Ready / {device.upper()} | mask {count} pixels | +{positive} −{negative} | Box {'yes' if self.prompts.box is not None else 'no'}"+("\nCPU 推理可能较慢。" if device == "cpu" else ""))
        self.panel.accept.setEnabled(count > 0 and self.rgb.mapper is not None)
        self.panel.next_mask.setEnabled(self.prediction is not None and len(self.prediction.masks)>1)
        self.panel.cpu.hide()

    def error(self, error):
        self.panel.status.setText("SAM: Error — "+str(error)+"；可继续使用 Polygon。")
        self.panel.cpu.setVisible("out of memory" in str(error).lower() or "cuda" in str(error).lower())
        self.panel.accept.setEnabled(False)

    def retry_cpu(self):
        self.config = replace(self.config, device="cpu");self.schedule()

    def next_mask(self):
        if self.prediction is None: return
        self.mask_index = (self.mask_index+1)%len(self.prediction.masks)
        self.draw();self.ready_status()

    def draw(self):
        for item in self.items: self.viewer.scene().removeItem(item)
        self.items.clear()
        self.mask_item.setPixmap(QtGui.QPixmap())
        def path_item(path, color, dashed=False):
            pen = QtGui.QPen(QtGui.QColor(color), 2);pen.setCosmetic(True)
            if dashed: pen.setStyle(QtCore.Qt.DashLine)
            item = self.viewer.scene().addPath(path, pen);item.setZValue(7);self.items.append(item)
        if self.patch:
            p = QtGui.QPainterPath();p.addRect(self.patch.col0, self.patch.row0, self.patch.width, self.patch.height);path_item(p, "#ffff66", True)
        for point, positive in self.prompts.points:
            # 标记尺寸不随缩放变化，锚点始终为全图像素。
            p = QtGui.QPainterPath();p.moveTo(-5, 0);p.lineTo(5, 0)
            if positive: p.moveTo(0, -5);p.lineTo(0, 5)
            path_item(p, "#00ff66" if positive else "#ff3333")
            self.items[-1].setPos(*point);self.items[-1].setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
        if self.prompts.box is not None:
            x1, y1, x2, y2 = self.prompts.box
            p = QtGui.QPainterPath();p.addRect(x1, y1, x2-x1, y2-y1);path_item(p, "#00ff66")
        if self.prediction is not None and self.patch is not None:
            mask = self.prediction.masks[self.mask_index]
            interior = mask.copy();interior[0] = False;interior[-1] = False;interior[:, 0] = False;interior[:, -1] = False
            interior[1:-1, 1:-1] &= mask[:-2, 1:-1]&mask[2:, 1:-1]&mask[1:-1, :-2]&mask[1:-1, 2:]
            rgba = np.zeros((*mask.shape, 4), np.uint8);rgba[mask] = [0, 235, 255, 65];rgba[mask & ~interior] = [0, 255, 255, 230]
            self.viewer.put_image(self.mask_item, rgba, self.patch.bounds)

    def accept(self):
        if self.state != SAMState.MASK_PREVIEW or self.prediction is None or self.rgb.roof.preview_pending(): return
        if self.rgb.mapper is None: self.error(ValueError("RGB / LiDAR 坐标映射未就绪"));return
        mask = self.patch.pixel_mask(self.prediction.masks[self.mask_index])
        if not mask.data.any(): return
        metadata = self.proposal_metadata()
        if self.rgb.roof.model.current == 0:
            self.target = None;self.rgb.roof.new_plane();self.target = self.rgb.roof.model.current
        generation = self.generation
        self.rgb.map_mask(lambda:mask, metadata, save_mask=True,
            valid=lambda:generation==self.generation, on_preview=self.enter_candidate)

    def proposal_metadata(self):
        return {"annotation_source":"rgb_sam", "patch_origin":[self.patch.col0, self.patch.row0],
            "patch_window":list(self.patch.bounds), "positive_global_pixels":[list(p) for p, k in self.prompts.points if k==1],
            "negative_global_pixels":[list(p) for p, k in self.prompts.points if k==0], "box_global_pixels":self.prompts.box,
            "sam_model_config":self.config.model_config, "sam_checkpoint":self.config.checkpoint,
            "sam_score":float(self.prediction.scores[self.mask_index]), "sam_mask_index":self.mask_index}

    def enter_candidate(self):
        self.state = SAMState.LIDAR_PREVIEW;self.mask_item.hide();self.panel.accept.setEnabled(False)

    def finish_candidate(self, confirmed=False):
        if self.state != SAMState.LIDAR_PREVIEW: return
        self.invalidate();self.prompts.clear();self.target = None
        self.state = SAMState.CONFIRMED if confirmed else SAMState.PATCH_READY
        self.mask_item.show();self.draw();self.panel.status.setText("SAM: "+self.state.value)

    def eventFilter(self, obj, event):
        if not self.active or self.viewer.locked: return False
        if self.rgb.roof.task: return False
        if event.type() == QtCore.QEvent.ShortcutOverride and event.key() in (QtCore.Qt.Key_R, QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Escape, QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            event.accept();return True
        if event.type() == QtCore.QEvent.KeyPress:
            key = event.key()
            if key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                if not event.isAutoRepeat(): self.accept()
            elif key in (QtCore.Qt.Key_Escape, QtCore.Qt.Key_R): self.reset_prompts()
            elif key == QtCore.Qt.Key_Backspace:
                self.prompts.undo()
                if self.prompts.events: self.schedule()
                else: self.reset_prompts()
            else: return False
            return True
        if obj is not self.viewer.viewport(): return False
        if event.type() == QtCore.QEvent.MouseButtonDblClick: return True
        if event.type() == QtCore.QEvent.MouseButtonPress and event.button() in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
            self.viewer.setFocus();self.rgb.activity()
            point = self.viewer.mapToScene(event.pos());point = (point.x(), point.y())
            if self.rgb.panel.tool.currentIndex() == 3 and event.button() == QtCore.Qt.LeftButton:
                self.drag = point
            else: self.add_point(point, event.button() == QtCore.Qt.LeftButton)
            return True
        if event.type() == QtCore.QEvent.MouseMove and self.drag is not None:
            end = self.viewer.mapToScene(event.pos())
            self.drag_item.setRect(QtCore.QRectF(QtCore.QPointF(*self.drag), end).normalized());self.drag_item.show()
            return True  # 只在完成 Box 时预测，鼠标移动不触发模型。
        if event.type() == QtCore.QEvent.MouseButtonRelease and event.button() in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
            if self.drag is not None:
                end = self.viewer.mapToScene(event.pos());start = self.drag;self.drag = None
                self.drag_item.hide()
                self.add_box((*start, end.x(), end.y()))
            return True
        return False
