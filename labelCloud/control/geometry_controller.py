"""几何过滤预览协调；数学计算在 GeometryRefiner，所有写标签仍由确认入口负责。"""
from dataclasses import asdict
import threading
import numpy as np
from PyQt5 import QtCore, QtWidgets
from .scene_worker import SceneTask
from ..model.geometry_refiner import GeometryRefiner, RefineSettings, plane_geometry
from ..model.scene_spatial import coordinate_scale


class GeometryController:
    def __init__(self,roof):
        self.roof=roof;self.task=None;self.cancel=threading.Event();self.generation=0
        self.last=None;self.original=None;self.refined=None;self.settings=None;self.confirm_ready=False
        self.bound_model=None
        panel=roof.panel;layout=panel.tools.layout()
        box=QtWidgets.QWidget();rows=QtWidgets.QVBoxLayout(box);rows.setContentsMargins(0,0,0,0)
        self.run=QtWidgets.QPushButton('几何过滤（先预览）');self.run.clicked.connect(self.refine)
        rows.addWidget(self.run)
        self.manual=QtWidgets.QPushButton('选中点 → 几何候选预览');self.manual.clicked.connect(self.manual_candidate);rows.addWidget(self.manual)
        row=QtWidgets.QHBoxLayout()
        self.original_button=QtWidgets.QPushButton('恢复过滤前候选');self.filtered_button=QtWidgets.QPushButton('使用过滤结果')
        self.original_button.clicked.connect(lambda:self.show_result(False));self.filtered_button.clicked.connect(lambda:self.show_result(True))
        row.addWidget(self.original_button);row.addWidget(self.filtered_button);rows.addLayout(row)
        toggle=QtWidgets.QToolButton();toggle.setText('几何过滤参数 ▸');toggle.setCheckable(True);rows.addWidget(toggle)
        settings=QtWidgets.QWidget();form=QtWidgets.QFormLayout(settings);form.setContentsMargins(0,0,0,0)
        note=QtWidgets.QLabel('以下参数与“局部扩展参数”同步；\n预览中调整后，再点击几何过滤。');note.setWordWrap(True);form.addRow(note)
        self.parameter_fields=[]
        for name,label in [('neighbor_radius','邻接半径 (m)'),('plane_distance','平面距离 (m)'),('normal_angle','法向夹角 (°)')]:
            source=getattr(panel,name);field=QtWidgets.QDoubleSpinBox()
            field.setDecimals(source.decimals());field.setRange(source.minimum(),source.maximum());field.setValue(source.value())
            field.setKeyboardTracking(False);field.setToolTip(source.toolTip())
            field.valueChanged.connect(source.setValue);source.valueChanged.connect(field.setValue)
            form.addRow(label,field);self.parameter_fields.append(field)
        units=QtWidgets.QComboBox();units.setModel(panel.coordinate_units.model());units.setCurrentIndex(panel.coordinate_units.currentIndex())
        units.currentIndexChanged.connect(panel.coordinate_units.setCurrentIndex);panel.coordinate_units.currentIndexChanged.connect(units.setCurrentIndex)
        form.addRow('原始 XYZ 单位',units)
        self.normals=QtWidgets.QCheckBox('使用已有法向缓存（无缓存则跳过）');self.normals.setChecked(True);form.addRow(self.normals)
        self.connectivity=QtWidgets.QCheckBox('连通性过滤');self.connectivity.setChecked(True);form.addRow(self.connectivity)
        self.mode=QtWidgets.QComboBox();self.mode.addItem('Main Component（主连通分量）','main');self.mode.addItem('Keep Coplanar Components（保留共面分量）','coplanar');form.addRow(self.mode)
        self.mode.setSizePolicy(QtWidgets.QSizePolicy.Ignored,QtWidgets.QSizePolicy.Fixed)
        self.mode.setToolTip('默认只保留主连通分量。保留共面分量适合遮挡断开的同一屋面，但也可能包含另一栋共面建筑，需人工检查。')
        def integer(label,value,low,high):
            widget=QtWidgets.QSpinBox();widget.setRange(low,high);widget.setValue(value);form.addRow(label,widget);return widget
        self.trials=integer('RANSAC 最大次数',300,1,5000)
        self.minimum=integer('最少候选 / 内点',20,3,100000)
        self.component_min=integer('最小分量点数',10,1,100000)
        self.seed=integer('随机种子',0,0,2147483647)
        self.ratio=QtWidgets.QDoubleSpinBox();self.ratio.setRange(.01,1);self.ratio.setSingleStep(.05);self.ratio.setValue(.3);form.addRow('最低内点比例',self.ratio)
        rows.addWidget(settings);settings.hide();toggle.toggled.connect(settings.setVisible)
        self.info=QtWidgets.QLabel('几何过滤仅处理当前候选，不写标签。');self.info.setWordWrap(True);rows.addWidget(self.info)
        index=layout.indexOf(panel.expansion_info);layout.insertWidget(index,box)
        self.sync()

    def sync(self):
        roof=self.roof
        if roof.model is not self.bound_model:
            self.reset();self.bound_model=roof.model
        ready=roof.candidate is not None and not roof.task and not self.task
        self.run.setEnabled(ready)
        self.manual.setEnabled(roof.model is not None and roof.candidate is None and not roof.task)
        self.original_button.setEnabled(ready and self.original is not None)
        self.filtered_button.setEnabled(ready and self.refined is not None)
        if roof.model is not None and roof.candidate is None:
            geometry=roof.model.session_metadata.get('plane_geometry',{}).get(str(roof.model.current),{})
            if geometry.get('warning'): self.info.setText(f"Plane {roof.model.current}: "+geometry['warning'])

    def reset(self):
        self.cancel.set();self.generation+=1
        self.last=self.original=self.refined=self.settings=None;self.confirm_ready=False

    def manual_candidate(self):
        r=self.roof
        if r.model is None or r.preview_pending(): return
        ids=np.flatnonzero(r.model.selection);allowed=r.model.selectable('unlabelled',ids);roi=r.roi_mask()
        if roi is not None: allowed &= roi[ids]
        ids=ids[allowed]
        if not len(ids): self.info.setText('请先选择未标注、未隐藏且在 ROI 内的点。');return
        if r.model.current==0: r.new_plane()
        from ..model.preview_edits import PreviewEdits
        r.candidate=ids;r.expansion_target=r.model.current;r.highlight=None
        r.strict_candidate=ids.copy();r.preview_edits=PreviewEdits()
        r.preview_edge_levels=[np.empty(0,np.int64) for _ in range(11)]
        r.preview_edge_thresholds=[(0.,0.)]*11;r.edge_level=0
        r.preview_fit_info='手工 LiDAR 候选；尚未写标签。\n'
        r.rgb_annotation_pending={'plane_id':r.model.current,'annotation_source':'lidar_manual_candidate'}
        r.panel.set_preview(True);r.adjust_edge_level(0)

    def version(self):
        r=self.roof;m=r.model
        return (id(m),id(r.candidate),m.labels_revision,m.hidden_revision,r.workspace.roi_revision,r.expansion_target,m.current)

    def refine(self,seed_point_ids=None):
        r=self.roof
        if r.candidate is None or r.task or self.task: return
        if len(r.candidate)>200000:
            self.info.setText('候选超过 20 万点，请缩小到单个屋面后过滤；原候选保持不变。');return
        if isinstance(seed_point_ids,bool): seed_point_ids=None
        p=r.panel;options=p.growth_options()
        try: scale=coordinate_scale(r.model.header,p.coordinate_units.currentData())
        except Exception as exc: self.info.setText(str(exc));return
        settings=RefineSettings(options.plane_distance,options.neighbor_radius,options.normal_angle,
            self.normals.isChecked(),self.connectivity.isChecked(),self.mode.currentData(),
            self.minimum.value(),self.component_min.value(),self.ratio.value(),self.trials.value(),self.seed.value())
        ids=r.candidate.copy();allowed=r.model.selectable('unlabelled',ids);roi=r.roi_mask()
        if roi is not None: allowed &= roi[ids]
        ids=ids[allowed];version=self.version();generation=self.generation;model=r.model
        # 只使用条件一致的现有局部法向缓存，不为过滤建立全场树/法向。
        key=(tuple(scale*r.cloud.applied_scale),options.normal_k,options.neighbor_radius,model.hidden_revision,r.workspace.roi_revision)
        normals=None
        if r.spatial.normal_key==key:
            normals=np.asarray([r.spatial.normals.get(int(i),np.zeros(3)) for i in ids])
        engine=GeometryRefiner(model.point_reader.read,scale, None if normals is None else lambda _:normals)
        self.cancel=threading.Event();cancel=self.cancel
        self.info.setText(f'正在几何过滤 {len(ids)} 点… 可继续修改预览，旧结果将失效。')
        task=SceneTask(lambda:engine.refine_plane(ids,settings,seed_point_ids,cancel));self.task=task;self.sync()
        def done(result,error):
            self.task=None
            if generation!=self.generation or r.model is not model or version!=self.version() or cancel.is_set():
                if generation==self.generation: self.info.setText('候选或场景已变化，旧几何结果已忽略。')
                self.sync();return
            if error or not result.success:
                self.info.setText('几何过滤失败，保留原候选。\n'+(str(error[0]) if error else result.failure_reason));self.sync();return
            self.original=r.candidate.copy();self.refined=result.final_point_ids.copy();self.last=result;self.settings=settings
            self.show_result(True)
            self.info.setText(f'候选 {result.input_count} → {result.final_count}\nRANSAC {result.ransac_inlier_count} ({result.inlier_ratio:.1%}) | 距离 {len(result.distance_filtered_ids)} | 法向 {len(result.normal_filtered_ids)} | 连通 {len(result.connectivity_filtered_ids)}\nRMS {result.rms_distance:.4f} m | Median {result.median_distance:.4f} m | P95 {result.p95_distance:.4f} m\n'+result.warning)
            self.sync()
        task.signals.finished.connect(done);QtCore.QThreadPool.globalInstance().start(task)

    def show_result(self,filtered):
        r=self.roof;desired=self.refined if filtered else self.original
        if desired is None or r.candidate is None or r.task: return
        # 一次过滤是一个候选差分历史，不拷贝全场 labels / mask。
        r.preview_edits.replace(r.candidate,desired)
        r.adjust_edge_level(0)

    def prepare_confirm(self):
        """最终确认前在后台拟合目标 Plane 的全部最终点，而非早期 RANSAC 内点。"""
        r=self.roof
        if self.confirm_ready: self.confirm_ready=False;return False
        if self.last is None and getattr(r.rgb,"gt_pending",None) is None: return False
        self.cancel.set();self.generation+=1
        ids=r.candidate;allowed=r.model.selectable('unlabelled',ids);roi=r.roi_mask()
        if roi is not None: allowed &= roi[ids]
        pid=r.expansion_target;final=np.union1d(r.model.plane_indices(pid),ids[allowed]);model=r.model
        try: scale=coordinate_scale(model.header,r.panel.coordinate_units.currentData())
        except Exception as exc: self.info.setText(str(exc));return True
        version=self.version()
        distance_m=self.settings.distance_m if self.settings is not None else r.panel.plane_distance.value()
        def work():
            try:
                geometry=plane_geometry(model.point_reader.read(final),scale,pid)
                geometry['distance_threshold_m']=distance_m
                if geometry['p95_m']>distance_m: geometry['warning']='最终人工点可能不是单一平面'
                return geometry
            except ValueError as exc:
                return dict(plane_id=pid,point_count=len(final),geometry_dirty=False,fit_failed=str(exc),unit_to_m=scale.tolist())
        def done(geometry):
            if r.model is not model or version!=self.version(): return
            self.final_geometry=geometry;self.confirm_ready=True;r.confirm_expansion()
        r.start_task('正在根据最终确认点重拟合几何参数…',work,done)
        return True

    def attach_confirmed(self):
        r=self.roof
        if self.last is None:
            if getattr(r.rgb,"gt_pending",None) is not None: r.model.attach_plane_geometry(self.final_geometry)
            return
        r.model.attach_plane_geometry(self.final_geometry)
        annotation=r.rgb_annotation_pending
        if annotation is None:
            annotation={'plane_id':r.expansion_target,'annotation_source':'lidar_manual_candidate'}
            r.rgb_annotation_pending=annotation
        rejected=np.setdiff1d(self.original,self.refined)
        applied=not len(rejected) or bool(len(np.setdiff1d(rejected,r.candidate)))
        if applied: annotation['annotation_source']+='_geometry_refined'
        annotation['geometry_refine']={**asdict(self.settings),'input_count':self.last.input_count,
            'refined_count':self.last.final_count,'ransac_inlier_ratio':self.last.inlier_ratio,
            'preview_final_count':len(r.candidate),'applied':applied,'timings_ms':self.last.timings_ms}
        if self.final_geometry.get('warning') or self.final_geometry.get('fit_failed'):
            self.info.setText(self.final_geometry.get('warning') or self.final_geometry['fit_failed'])
