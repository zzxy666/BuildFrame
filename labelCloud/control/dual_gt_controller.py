"""双 GT 的小型界面协调层；不接管既有点选择/候选算法。"""
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from .scene_worker import SceneTask
from ..model.dual_gt import rgb_identity
from ..model.dual_gt_export import DualGTExporter, validation_text


class DualGTController:
    def __init__(self, roof):
        self.roof = roof; self.overlay_key = None; self.overlay_task = None
        self.bound_model = None; self.summary_key = None; self.summary = ''
        rgb = roof.rgb
        self.overlay = rgb.panel.viewer.scene().addPixmap(QtGui.QPixmap()); self.overlay.setZValue(2.5)
        row = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(row); layout.setContentsMargins(0,0,0,0)
        self.confirm = QtWidgets.QPushButton('确认当前 RGB Mask 为 GT')
        self.confirm.clicked.connect(self.confirm_rgb)
        self.replace = QtWidgets.QCheckBox('替换当前 GT')
        self.replace.setToolTip('默认排除所有已标注像素；替换时允许重选当前 Plane，其他 Plane 仍受保护。')
        note=QtWidgets.QLabel("像素保护：自动排除已标注区域；替换时仅放开当前 Plane。")
        note.setWordWrap(True);layout.addWidget(note)
        self.replace.toggled.connect(self.protection_changed)
        layout.addWidget(self.confirm)
        options=QtWidgets.QHBoxLayout();options.addWidget(self.replace);layout.addLayout(options)
        rgb.panel.controls_layout.insertWidget(2,row)
        self.overlay.setOpacity(rgb.panel.opacity.value()/100)
        rgb.panel.opacity.valueChanged.connect(lambda value:self.overlay.setOpacity(value/100))
        self.status = QtWidgets.QLabel(); self.status.setWordWrap(True)
        self.review = QtWidgets.QPushButton('已检查：确认当前 RGB GT')
        self.review.clicked.connect(self.confirm_review)
        self.export_button = QtWidgets.QPushButton('导出双 GT（RGB + LiDAR）')
        self.export_button.clicked.connect(self.export_dialog)
        tools = roof.panel.tools.layout()
        tools.insertWidget(tools.indexOf(roof.panel.selection_info),self.status)
        for widget in (self.review,self.export_button): tools.addWidget(widget)
        self.timer = QtCore.QTimer(roof.view); self.timer.setSingleShot(True); self.timer.setInterval(100)
        self.timer.timeout.connect(self.refresh_overlay)
        rgb.panel.viewer.detail_timer.timeout.connect(self.refresh_overlay)
        self.sync()

    def protection_changed(self):
        sam=self.roof.rgb.sam
        if sam.active and sam.prompts.events and self.roof.candidate is None: sam.schedule()

    def sync(self):
        r = self.roof; model = r.model; rgb = r.rgb
        if model is not self.bound_model:
            self.bound_model=model; rgb.gt_pending=None; self.overlay_key=None
            self.overlay.setPixmap(QtGui.QPixmap())
        ready = model is not None and not r.task and not r._saving and r.candidate is None
        self.confirm.setEnabled(ready and model.current>0 and rgb.photo is not None)
        self.replace.setEnabled(ready)
        self.export_button.setEnabled(ready and rgb.photo is not None)
        record = model.dual_gt.state['active'].get(str(model.current),{}) if model else {}
        self.review.setEnabled(bool(ready and rgb.photo is not None and record.get('fragments')))
        if model is None: self.status.setText('Dual-GT：未加载点云'); return
        key = (id(model),id(model.dual_gt.state),model.labels_revision)
        if key != self.summary_key:
            values=model.dual_gt.summaries(); total=sum(v['lidar_point_count']>0 for v in values)
            complete=sum(v['lidar_point_count']>0 and v['rgb_gt_status']=='CONFIRMED' for v in values)
            review=sum(v['rgb_gt_status']=='NEEDS_REVIEW' for v in values)
            missing=sum(v['lidar_point_count']>0 and v['rgb_gt_status']=='MISSING' for v in values)
            self.summary=f'LiDAR Planes {total} | Dual confirmed {complete}\nRGB Missing {missing} | Needs Review {review}'
            self.summary_key=key
        rgb_status=record.get('rgb_gt_status','MISSING')
        if rgb.gt_pending and rgb.gt_pending['plane_id']==model.current: rgb_status='PENDING'
        lidar_status='CONFIRMED' if model.plane_counts.get(model.current,0) and model.current else 'MISSING'
        warning=' | 跨模态待复核' if record.get('cross_modal_status')=='NEEDS_REVIEW' else ''
        self.status.setText(f'Plane {model.current}: LiDAR {lidar_status}\nRGB {rgb_status}{warning}\n{self.summary}\n颜色覆盖：已确认 RGB 像素 GT')
        self.timer.start()

    def validate_pending(self):
        pending=self.roof.rgb.gt_pending
        if pending is None: return True
        try:
            record=self.roof.model.dual_gt.state['active'].get(str(pending['plane_id']))
            if record is None or record['plane_uid']!=pending['plane_uid']: raise ValueError('Plane 生命周期已变化，请重新预览')
            if rgb_identity(self.roof.rgb.photo.path)!=pending['fragment']['source']: raise ValueError('RGB 文件已变化，请重新预览')
            self.roof.model.dual_gt.store.read(pending['fragment'])
        except (ValueError,OSError,KeyError) as exc:
            self.roof.view.status_manager.set_message(str(exc)); return False
        return True

    def confirm_rgb(self):
        r=self.roof;rgb=r.rgb;model=r.model
        if model is None or model.current==0 or rgb.photo is None or r.preview_pending(): return
        pid=model.current; uid=model.dual_gt.ensure(pid)['plane_uid'];photo=rgb.photo
        replace=self.replace.isChecked()
        if rgb.sam.active:
            sam=rgb.sam
            from ..model.segmentation.patch import SAMState
            if sam.task is not None or sam.state!=SAMState.MASK_PREVIEW or sam.prediction is None or sam.patch is None:
                self.status.setText('请先生成并检查 SAM mask');return
            mask=sam.patch.pixel_mask(sam.prediction.masks[sam.mask_index])
            metadata=sam.proposal_metadata(); factory=lambda:mask
        else:
            if rgb.polygon is None or rgb.mapper is None:
                self.status.setText('请先闭合 Polygon，或生成 SAM mask');return
            vertices=rgb.polygon.copy(); mapper=rgb.mapper
            factory=lambda:mapper.polygon_mask(vertices)
            metadata=dict(annotation_source='rgb_polygon',pixel_polygon=vertices.tolist())
        def work():
            source=rgb_identity(photo.path)
            mask,blocked=model.dual_gt.unlabelled_mask(factory(),source,pid,replace)
            if not mask.data.any(): raise ValueError('该区域全部为已标注像素，没有可新增的 RGB GT；重画当前 Plane 请勾选替换当前 GT')
            return model.dual_gt.store.write(mask,source,metadata['annotation_source']),blocked
        def done(result):
            fragment,blocked=result
            if r.model is not model or rgb.photo is not photo or model.current!=pid or model.dual_gt.ensure(pid)['plane_uid']!=uid: return
            model.dual_gt.confirm(pid,fragment,replace=replace,annotation=dict(metadata,plane_id=pid,operation='rgb_only_confirm',protected_pixels_excluded=blocked))
            rgb.panel.viewer.clear_polygon();rgb.polygon=None
            if rgb.sam.active: rgb.sam.reset_prompts()
            r.refresh();r.view.status_manager.set_message(f'RGB GT 已确认；排除 {blocked} 个已标注像素；LiDAR 标签不变，可 Ctrl+Z 撤销')
        r.start_task('正在保存局部 RGB GT…',work,done)

    def confirm_review(self):
        r=self.roof
        if r.model is None or r.rgb.photo is None or r.preview_pending(): return
        record=r.model.dual_gt.state['active'].get(str(r.model.current))
        if not record or not record['fragments']: return
        try:
            identity=rgb_identity(r.rgb.photo.path)
            for fragment in record['fragments']:
                if fragment['source']!=identity: raise ValueError('RGB 来源不一致')
                r.model.dual_gt.store.read(fragment)
            r.model.dual_gt.confirm(r.model.current)
            r.refresh()
        except (ValueError,OSError) as exc: r.view.status_manager.set_message(str(exc))

    def overlay_view_key(self):
        r=self.roof;model=r.model;photo=r.rgb.photo
        if model is None or photo is None: return None
        viewer=r.rgb.panel.viewer
        rect=viewer.mapToScene(viewer.viewport().rect()).boundingRect().intersected(viewer.sceneRect())
        bounds=(max(0,int(np.floor(rect.left()))),max(0,int(np.floor(rect.top()))),
                min(photo.width,int(np.ceil(rect.right()))),min(photo.height,int(np.ceil(rect.bottom()))))
        return (id(model),id(photo),id(model.dual_gt.state),bounds)

    def refresh_overlay(self):
        r=self.roof;model=r.model;photo=r.rgb.photo
        mixed=r.rgb.panel.overlay_mode.currentIndex()==1
        r.rgb.panel.opacity.setEnabled(mixed)
        if model is None or photo is None or not mixed:
            self.overlay.hide();return
        self.overlay.show()
        key=self.overlay_view_key()
        if key==self.overlay_key: return
        if self.overlay_task is not None: self.timer.start();return
        records=[v for v in model.dual_gt.state['active'].values() if v['rgb_gt_status']=='CONFIRMED' and v['fragments']]
        self.overlay.setPixmap(QtGui.QPixmap())
        left,top,right,bottom=key[-1]
        if not records or right<=left or bottom<=top:
            self.overlay_key=key;return
        def work():
            from ..model.roof_planes import plane_color
            identity=rgb_identity(photo.path)
            # 只生成当前视窗的有界预览；缩放/平移后重新采样正式 mask。
            factor=max((right-left)/1800,(bottom-top)/1800,1)
            width=max(1,int(np.ceil((right-left)/factor)));height=max(1,int(np.ceil((bottom-top)/factor)))
            cols=left+(np.arange(width)+.5)*(right-left)/width;rows=top+(np.arange(height)+.5)*(bottom-top)/height
            rgba=np.zeros((height,width,4),np.uint8);occupied=np.zeros((height,width),bool)
            for record in records:
                union=np.zeros((height,width),bool)
                for f in record['fragments']:
                    cc=np.flatnonzero((cols>=f['col0'])&(cols<f['col0']+f['width']))
                    rr=np.flatnonzero((rows>=f['row0'])&(rows<f['row0']+f['height']))
                    if not len(cc) or not len(rr): continue
                    if f['source']!=identity: raise ValueError('RGB GT 来源不匹配')
                    mask=model.dual_gt.store.read(f)
                    union[np.ix_(rr,cc)] |= mask[np.ix_((rows[rr]-f['row0']).astype(int),(cols[cc]-f['col0']).astype(int))]
                conflict=union & occupied
                rgba[union,:3]=(np.asarray(plane_color(record['plane_id']))*255).astype(np.uint8)
                rgba[union,3]=255
                rgba[conflict,:3]=[255,0,0]  # 冲突仅作红色显示警示，正式导出仍会阻止。
                occupied |= union
            import rasterio
            from rasterio.windows import Window
            with rasterio.open(photo.path) as ds:
                valid=np.all(ds.read_masks(window=Window(left,top,right-left,bottom-top),out_shape=(ds.count,height,width))>0,axis=0)
                rgba[~valid,3]=0
            return rgba,(left,top,right,bottom)
        def done(value,error):
            self.overlay_task=None
            if self.overlay_view_key()!=key: self.timer.start();return
            if error is None:
                self.overlay_key=key;r.rgb.panel.viewer.put_image(self.overlay,*value)
            else:
                self.overlay.setPixmap(QtGui.QPixmap());self.overlay_key=key
                self.status.setText(self.status.text()+'\nRGB GT 显示失败：'+str(error[0]))
            self.overlay.setVisible(r.rgb.panel.overlay_mode.currentIndex()==1)
        self.overlay_task=SceneTask(work);self.overlay_task.signals.finished.connect(done)
        QtCore.QThreadPool.globalInstance().start(self.overlay_task)

    def export_dialog(self):
        r=self.roof
        if r.model is None or r.rgb.photo is None or r.preview_pending(): return
        text,ok=QtWidgets.QInputDialog.getItem(r.view,'双 GT 导出模式','默认严格检查两种 GT：',['Strict Dual-GT','Allow Partial'],0,False)
        if not ok: return
        strict=text=='Strict Dual-GT'; model=r.model;photo=r.rgb.photo
        exporter=DualGTExporter(model,photo.path,r.cancel_task)
        def done(report):
            message=validation_text(report)
            dialog=QtWidgets.QMessageBox(r.view)
            dialog.setWindowTitle('双 GT 校验报告' if report['ok'] else '双 GT 校验未通过')
            dialog.setText(f"LiDAR Planes: {report['lidar_planes']} | RGB: {report['rgb_planes']} | Complete: {report['complete_planes']}\n"
                f"Missing RGB: {len(report['missing_rgb_planes'])} | Missing LiDAR: {len(report['missing_lidar_planes'])}\n"
                f"RGB Needs Review: {len(report['needs_review_planes'])} | Conflicts: {len(report['conflicts'])} | Errors: {len(report['errors'])}\n"
                f"跨模态复核警告: {len(report['cross_modal_review_planes'])} | Dirty Geometry: {report['dirty_geometry']}\n"
                +('继续导出？' if report['ok'] else '已阻止导出，请展开详细信息查看 Plane ID 和原因。'))
            dialog.setDetailedText(message)
            dialog.setStandardButtons(QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No if report['ok'] else QtWidgets.QMessageBox.Ok)
            if report['ok']: dialog.setDefaultButton(QtWidgets.QMessageBox.No)
            if dialog.exec_()!=QtWidgets.QMessageBox.Yes or not report['ok']: return
            parent=QtWidgets.QFileDialog.getExistingDirectory(r.view,'选择输出父目录（创建新的 scene_dual_gt 子目录）')
            if not parent: return
            from pathlib import Path
            destination=Path(parent)/(model.path.stem+'_dual_gt')
            if r._autosave_task is not None and not r.save(force=True): return
            def export_work():
                exporter.cancel=r.cancel_task  # start_task 会为每次任务创建新的取消事件。
                return exporter.export(destination,strict)
            r.start_task('正在分块导出双 GT…',export_work,
                lambda path:r.view.status_manager.set_message('双 GT 已导出：'+str(path)))
        def validate_work():
            exporter.cancel=r.cancel_task
            return exporter.validate(strict)
        r.start_task('正在校验双 GT、mask 冲突与文件完整性…',validate_work,done)
