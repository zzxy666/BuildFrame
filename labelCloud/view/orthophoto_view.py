"""QGraphicsView 正射视图；场景坐标始终为全图像素坐标，不随缩放变化。"""
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from ..control.scene_worker import SceneTask


def pixmap(array):
    array=np.ascontiguousarray(array,np.uint8)
    h,w=array.shape[:2]
    return QtGui.QPixmap.fromImage(QtGui.QImage(array.data,w,h,w*4,QtGui.QImage.Format_RGBA8888).copy())


class OrthophotoView(QtWidgets.QGraphicsView):
    polygon_ready=QtCore.pyqtSignal(object)
    pixel_clicked=QtCore.pyqtSignal(object)
    activity=QtCore.pyqtSignal()
    undo_requested=QtCore.pyqtSignal()
    redo_requested=QtCore.pyqtSignal()

    def __init__(self,parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setMinimumSize(180,180)
        self.setBackgroundBrush(QtGui.QColor('#333333'))
        self.setTransformationAnchor(self.AnchorUnderMouse)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMouseTracking(True)
        self.photo=None;self.vertices=[];self.finished_polygon=None
        self.polygon_future=None;self.drawing=False;self.locked=False;self.pan=None
        self.base=self.scene().addPixmap(QtGui.QPixmap())
        self.detail=self.scene().addPixmap(QtGui.QPixmap());self.detail.setZValue(1)
        self.overlay=self.scene().addPixmap(QtGui.QPixmap());self.overlay.setZValue(2)
        self.path=self.scene().addPath(QtGui.QPainterPath());self.path.setZValue(3)
        pen=QtGui.QPen(QtGui.QColor('#00ffff'),2);pen.setCosmetic(True);self.path.setPen(pen)
        self.path.setBrush(QtGui.QColor(0,220,255,50))
        self.cursor_point=None
        self.guide_path=QtGui.QPainterPath();self.show_cursor=False
        self.cross=self.scene().addPath(QtGui.QPainterPath(),pen);self.cross.setZValue(4)
        self.read_task=None;self.generation=0
        self.detail_timer=QtCore.QTimer(self);self.detail_timer.setSingleShot(True)
        self.detail_timer.setInterval(180);self.detail_timer.timeout.connect(self.read_detail)
        self.horizontalScrollBar().valueChanged.connect(self.request_detail)
        self.verticalScrollBar().valueChanged.connect(self.request_detail)

    def set_photo(self,photo,overview):
        self.generation+=1;self.photo=photo
        self.clear_polygon();self.cross.setPath(QtGui.QPainterPath())
        self.detail.setPixmap(QtGui.QPixmap());self.overlay.setPixmap(QtGui.QPixmap())
        self.scene().setSceneRect(0,0,photo.width,photo.height)
        self.put_image(self.base,*overview)
        self.fit_image()

    def put_image(self,item,array,bounds):
        left,top,right,bottom=bounds
        item.setPixmap(pixmap(array));item.setPos(left,top)
        item.setTransform(QtGui.QTransform.fromScale((right-left)/array.shape[1],(bottom-top)/array.shape[0]))

    def set_overlay(self,array):
        if self.photo is not None:
            self.put_image(self.overlay,array,(0,0,self.photo.width,self.photo.height))

    def fit_image(self):
        if self.photo: self.fitInView(self.sceneRect(),QtCore.Qt.KeepAspectRatio);self.request_detail()

    def fit_bounds(self,bounds):
        left,top,right,bottom=bounds
        rect=QtCore.QRectF(left,top,max(right-left,1),max(bottom-top,1))
        self.fitInView(rect,QtCore.Qt.KeepAspectRatio);self.request_detail()

    def locate(self,col,row):
        self.centerOn(col,row)
        size=8/max(self.transform().m11(),1e-6)
        path=QtGui.QPainterPath();path.moveTo(col-size,row);path.lineTo(col+size,row)
        path.moveTo(col,row-size);path.lineTo(col,row+size);self.cross.setPath(path)

    def request_detail(self,*_):
        if self.photo is not None and self.isVisible(): self.detail_timer.start()

    def read_detail(self):
        if self.read_task is not None:
            self.detail_timer.start();return
        if self.photo is None or not self.isVisible(): return
        rect=self.mapToScene(self.viewport().rect()).boundingRect().intersected(self.sceneRect())
        if rect.isEmpty(): return
        bounds=(rect.left(),rect.top(),rect.right(),rect.bottom())
        photo=self.photo;generation=self.generation
        task=SceneTask(lambda:photo.read_window(bounds,max_size=1800))
        def done(result,error):
            self.read_task=None
            if error is None and self.photo is photo and generation==self.generation:
                self.put_image(self.detail,*result)
        task.signals.finished.connect(done);self.read_task=task
        QtCore.QThreadPool.globalInstance().start(task)

    def wheelEvent(self,event):
        self.activity.emit()
        factor=1.25 if event.angleDelta().y()>0 else .8
        if 1e-5 < self.transform().m11()*factor < 100:
            self.scale(factor,factor)
        self.request_detail();event.accept()

    def mousePressEvent(self,event):
        self.activity.emit();self.setFocus()
        if event.button() in (QtCore.Qt.MiddleButton,QtCore.Qt.RightButton):
            self.pan=event.pos();event.accept();return
        if event.button()==QtCore.Qt.LeftButton and self.photo:
            point=self.mapToScene(event.pos())
            if not self.sceneRect().contains(point): return
            if self.drawing and not self.locked:
                if self.finished_polygon is not None: self.clear_polygon()
                self.polygon_ready.emit(None)
                self.cursor_point=point
                self.vertices.append([point.x(),point.y()]);self.draw_polygon()
            elif not self.locked: self.pixel_clicked.emit([point.x(),point.y()])
            event.accept();return
        super().mousePressEvent(event)

    def mouseMoveEvent(self,event):
        if self.pan is not None:
            self.activity.emit();delta=event.pos()-self.pan;self.pan=event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value()-delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value()-delta.y())
            event.accept();return
        if self.drawing and not self.locked and self.finished_polygon is None:
            point=self.mapToScene(event.pos())
            self.cursor_point=point if self.sceneRect().contains(point) else None
            self.draw_guide()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self,event):
        if event.button() in (QtCore.Qt.MiddleButton,QtCore.Qt.RightButton): self.pan=None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self,event):
        if event.button()==QtCore.Qt.LeftButton and self.drawing and not self.locked:
            self.finish_polygon();event.accept();return
        super().mouseDoubleClickEvent(event)

    def finish_polygon(self):
        if len(self.vertices)<3 or self.locked: return
        vertices=np.asarray(self.vertices,float)
        if abs(np.dot(vertices[:,0],np.roll(vertices[:,1],1))-np.dot(vertices[:,1],np.roll(vertices[:,0],1)))<1e-6: return
        self.finished_polygon=vertices.copy();self.polygon_future=None
        self.draw_polygon();self.polygon_ready.emit(vertices)

    def draw_polygon(self):
        path=QtGui.QPainterPath()
        if self.vertices:
            path.moveTo(*self.vertices[0])
            for vertex in self.vertices[1:]: path.lineTo(*vertex)
            if self.finished_polygon is not None: path.closeSubpath()
        self.path.setPath(path)
        self.draw_guide()

    def draw_guide(self):
        self.guide_path=QtGui.QPainterPath()
        self.show_cursor=self.drawing and not self.locked and self.finished_polygon is None and self.cursor_point is not None
        if self.show_cursor and self.vertices:
            self.guide_path.moveTo(*self.vertices[-1]);self.guide_path.lineTo(self.cursor_point)
            if len(self.vertices)>1: self.guide_path.lineTo(*self.vertices[0])
        self.viewport().update()

    def paintEvent(self,event):
        super().paintEvent(event)
        # 在视窗坐标绘制固定大小的标记，不添加需要单独销毁的场景对象。
        painter=QtGui.QPainter(self.viewport());painter.setRenderHint(QtGui.QPainter.Antialiasing)
        if self.show_cursor and self.drawing and not self.locked:
            painter.setPen(QtGui.QPen(QtGui.QColor('#ffdd33'),2,QtCore.Qt.DashLine))
            painter.drawPath(self.viewportTransform().map(self.guide_path))
        for i,vertex in enumerate(self.vertices):
            painter.setPen(QtGui.QPen(QtGui.QColor('#222222'),1))
            painter.setBrush(QtGui.QColor('#66ff88' if i==0 else '#00dcff'))
            painter.drawEllipse(self.mapFromScene(QtCore.QPointF(*vertex)),4,4)
        if self.show_cursor and self.drawing and not self.locked and self.cursor_point is not None:
            painter.setPen(QtGui.QPen(QtGui.QColor('#222222'),1));painter.setBrush(QtGui.QColor('#ffdd33'))
            painter.drawEllipse(self.mapFromScene(self.cursor_point),4,4)
        painter.end()

    def leaveEvent(self,event):
        self.cursor_point=None;self.draw_guide();super().leaveEvent(event)

    def set_polygon_mode(self,enabled):
        self.drawing=enabled;self.cursor_point=None;self.draw_guide()

    def clear_polygon(self):
        self.vertices=[];self.finished_polygon=None;self.polygon_future=None;self.cursor_point=None;self.draw_polygon()

    def event(self,event):
        # RGB 焦点内拦截窗口级 Roof 快捷键，避免 Enter 同时闭合和确认标签。
        keys=(QtCore.Qt.Key_Return,QtCore.Qt.Key_Enter,QtCore.Qt.Key_Escape,QtCore.Qt.Key_Backspace,QtCore.Qt.Key_Z,QtCore.Qt.Key_Y)
        if event.type()==QtCore.QEvent.ShortcutOverride and event.key() in keys and not self.locked:
            event.accept();return True
        return super().event(event)

    def keyPressEvent(self,event):
        if self.locked: return super().keyPressEvent(event)
        if event.key() in (QtCore.Qt.Key_Return,QtCore.Qt.Key_Enter): self.finish_polygon()
        elif event.key()==QtCore.Qt.Key_Escape: self.clear_polygon();self.polygon_ready.emit(None)
        elif event.key()==QtCore.Qt.Key_Backspace:
            if self.vertices: self.vertices.pop();self.finished_polygon=None;self.draw_polygon();self.polygon_ready.emit(None)
        elif event.modifiers() & QtCore.Qt.ControlModifier and event.key()==QtCore.Qt.Key_Z and self.finished_polygon is not None:
            self.polygon_future=self.finished_polygon.copy();self.vertices=[];self.finished_polygon=None;self.draw_polygon();self.polygon_ready.emit(None)
        elif event.modifiers() & QtCore.Qt.ControlModifier and event.key()==QtCore.Qt.Key_Y and self.polygon_future is not None:
            self.vertices=self.polygon_future.tolist();self.finish_polygon()
        elif event.modifiers() & QtCore.Qt.ControlModifier and event.key()==QtCore.Qt.Key_Z:
            self.undo_requested.emit()
        elif event.modifiers() & QtCore.Qt.ControlModifier and event.key()==QtCore.Qt.Key_Y:
            self.redo_requested.emit()
        else: super().keyPressEvent(event)

    def resizeEvent(self,event):
        super().resizeEvent(event);self.request_detail()


class OrthophotoPanel(QtWidgets.QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        outer=QtWidgets.QVBoxLayout(self);outer.setContentsMargins(0,0,0,0)
        # RGB 操作区独立容器；隐藏时不改变内部 SAM 控件的可见状态。
        self.controls=QtWidgets.QWidget(self)
        self.controls_layout=QtWidgets.QVBoxLayout(self.controls)
        layout=self.controls_layout;layout.setContentsMargins(0,0,0,0)
        outer.addWidget(self.controls)
        row=QtWidgets.QHBoxLayout()
        self.open_button=QtWidgets.QPushButton("打开正射影像")
        self.tool=QtWidgets.QComboBox();self.tool.addItems(["浏览 / 查询 Plane","Polygon 标注"])
        self.map_button=QtWidgets.QPushButton("映射到 LiDAR（预览）");self.map_button.setEnabled(False)
        self.tool.setSizePolicy(QtWidgets.QSizePolicy.Ignored,QtWidgets.QSizePolicy.Fixed)
        for widget in (self.open_button,self.tool): row.addWidget(widget)
        layout.addLayout(row)
        layout.addWidget(self.map_button)
        row=QtWidgets.QHBoxLayout()
        self.fit_button=QtWidgets.QPushButton("Fit Image")
        self.roi_button=QtWidgets.QPushButton("Fit Current ROI")
        self.overlay_mode=QtWidgets.QComboBox()
        self.overlay_mode.addItems(["RGB 原始影像","RGB + 已标注 Plane 像素"])
        self.overlay_mode.setCurrentIndex(1)
        self.opacity=QtWidgets.QSlider(QtCore.Qt.Horizontal);self.opacity.setRange(10,100);self.opacity.setValue(70)
        self.opacity.setMaximumWidth(80);self.opacity.setToolTip("已标注 Plane 像素颜色不透明度")
        for widget in (self.fit_button,self.roi_button): row.addWidget(widget)
        layout.addLayout(row)
        row=QtWidgets.QHBoxLayout()
        self.overlay_mode.setSizePolicy(QtWidgets.QSizePolicy.Ignored,QtWidgets.QSizePolicy.Fixed)
        for widget in (self.overlay_mode,self.opacity): row.addWidget(widget)
        layout.addLayout(row)
        self.info=QtWidgets.QLabel("打开带 CRS 的 GeoTIFF，Polygon 闭合后先映射预览，再在 LiDAR 中确认。")
        self.info.setWordWrap(True);layout.addWidget(self.info)
        self.info.setSizePolicy(QtWidgets.QSizePolicy.Ignored,QtWidgets.QSizePolicy.Preferred)
        self.viewer=OrthophotoView(self);outer.addWidget(self.viewer,1)
        hint=QtWidgets.QLabel("RGB：滚轮缩放；中/右键平移。Polygon：左键加点，双击/Enter 闭合，Backspace 删顶点，Esc 取消。")
        self.hint=hint
        hint.setWordWrap(True);outer.addWidget(hint)
