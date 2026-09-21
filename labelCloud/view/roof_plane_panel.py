"""复用主窗口右侧区域，Wireframe 控件保持独立。"""
from PyQt5 import QtCore, QtGui, QtWidgets
from ..model.local_plane_growth import GrowthOptions
from .plane_table import PlaneTableModel


class RoofPlanePanel(QtWidgets.QWidget):
    def __init__(self, control, parent):
        super().__init__(parent)
        self.control = control
        self.setMinimumWidth(285)
        self.setMaximumWidth(360)
        layout = QtWidgets.QVBoxLayout(self)
        title = QtWidgets.QLabel("Roof Plane 标注")
        title.setStyleSheet("font-size: 18px; font-weight: bold")
        layout.addWidget(title)
        self.message = QtWidgets.QLabel()
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.tools = QtWidgets.QWidget()
        tools = QtWidgets.QVBoxLayout(self.tools)
        tools.setContentsMargins(0, 0, 0, 0)
        tools.setSpacing(4)
        self.tool = QtWidgets.QComboBox()
        for text, value in [("浏览 / 旋转视角 [V]", "navigate"), ("矩形选择 [B]", "rectangle"), ("套索选择 [L]（按住左键拖动）", "lasso")]:
            self.tool.addItem(text, value)
        self.tool.currentIndexChanged.connect(control.cancel_gesture)
        tools.addWidget(self.tool)
        hint = QtWidgets.QLabel("选择穿透当前视图，包含遮挡点。\nV 浏览 / B 矩形 / L 套索；中键临时旋转。\nShift 追加；Alt 减选；右键平移，滚轮缩放。\n青色：选区 / 当前 Plane；黄色：扩展候选。")
        hint.setWordWrap(True)
        tools.addWidget(hint)
        self.display = QtWidgets.QComboBox()
        self.display.addItems(["Plane ID 颜色", "RGB 原始颜色"])
        self.display.currentIndexChanged.connect(control.refresh_colors)
        tools.addWidget(self.display)
        self.selection_scope = QtWidgets.QComboBox()
        for text, value in [("仅未标注点（默认保护）", "unlabelled"),
                            ("当前 Plane", "current"), ("所有可见点（允许纠错）", "all")]:
            self.selection_scope.addItem(text, value)
        self.selection_scope.setToolTip("选择范围：默认不能修改已有标签；纠错时主动切换范围。")
        self.selection_scope.currentIndexChanged.connect(control.change_selection_scope)
        tools.addWidget(QtWidgets.QLabel("选择范围"))
        tools.addWidget(self.selection_scope)
        self.plane_search = QtWidgets.QLineEdit()
        self.plane_search.setPlaceholderText("搜索 / 跳转 Plane ID")
        self.plane_search.textChanged.connect(control.search_planes)
        self.plane_search.returnPressed.connect(control.jump_plane)
        self.plane_search.installEventFilter(self)
        tools.addWidget(self.plane_search)
        navigation = QtWidgets.QHBoxLayout()
        for text, fn in [("上一 Plane", lambda: control.step_plane(-1)),
                         ("下一 Plane", lambda: control.step_plane(1)), ("跳转 ID", control.jump_plane)]:
            button = QtWidgets.QPushButton(text); button.clicked.connect(fn); navigation.addWidget(button)
        tools.addLayout(navigation)
        self.list = QtWidgets.QTableView()
        self.table_model = PlaneTableModel(self.list)
        self.list.setModel(self.table_model)
        self.list.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list.verticalHeader().hide()
        self.list.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        self.list.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        # 列表独立滚动，为下方标注按钮保留可见空间。
        self.list.setMinimumHeight(80)
        self.list.setMaximumHeight(160)
        self.list.clicked.connect(control.select_plane)
        tools.addWidget(self.list, 1)
        self.selection_info = QtWidgets.QLabel("已选择 0 points")
        tools.addWidget(self.selection_info)
        visibility = QtWidgets.QGridLayout()
        for i, (text, callback) in enumerate([
            ("隐藏选中点 [H]", control.hide_selected),
            ("恢复全部 [Shift+H]", control.restore_hidden),
            ("只显示选中点", control.isolate_selected),
        ]):
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda checked=False, fn=callback: fn())
            visibility.addWidget(button, i // 2, i % 2)
        self.show_hidden = QtWidgets.QCheckBox("仅显示隐藏点（只读）")
        self.show_hidden.toggled.connect(control.change_hidden_view)
        visibility.addWidget(self.show_hidden, 1, 1)
        tools.addLayout(visibility)
        self.assign_button = QtWidgets.QPushButton("赋给当前 Plane")
        self.assign_button.clicked.connect(lambda: control.assign())
        tools.addWidget(self.assign_button)
        self.growth_settings_toggle = QtWidgets.QToolButton()
        self.growth_settings_toggle.setText("局部扩展参数（米） ▸")
        self.growth_settings_toggle.setCheckable(True)
        tools.addWidget(self.growth_settings_toggle)
        self.growth_settings = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.growth_settings)
        form.setContentsMargins(0, 0, 0, 0)
        self.normal_k = QtWidgets.QSpinBox()
        self.normal_k.setRange(6, 500)
        self.normal_k.setValue(20)
        form.addRow("法向邻点数", self.normal_k)
        for name, label, value, maximum in [
            ("neighbor_radius", "邻接半径 (m)", .6, 100),
            ("plane_distance", "平面距离 (m)", .15, 100),
            ("normal_angle", "法向夹角 (°)", 10., 89.9),
        ]:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(3)
            spin.setRange(.001, maximum)
            spin.setSingleStep(.05 if name != "normal_angle" else 1)
            spin.setValue(value)
            setattr(self, name, spin)
            form.addRow(label, spin)
        self.coordinate_units = QtWidgets.QComboBox()
        for label, value in [("自动读取 CRS", "auto"), ("XYZ 均为米", "m"),
                             ("XYZ 均为 US survey foot", "ftUS"), ("XYZ 均为国际英尺", "ft")]:
            self.coordinate_units.addItem(label, value)
        form.addRow("原始坐标单位", self.coordinate_units)
        self.edge_completion = QtWidgets.QCheckBox("边缘补选（不继续向外生长）")
        self.edge_completion.setChecked(True)
        self.edge_completion.setToolTip("补选靠近严格区域且贴合种子平面的漏点；仍需检查预览，尤其是屋脊附近。")
        form.addRow(self.edge_completion)
        self.edge_radius = QtWidgets.QDoubleSpinBox()
        self.edge_radius.setDecimals(3); self.edge_radius.setRange(.01,2); self.edge_radius.setValue(.25)
        self.edge_distance = QtWidgets.QDoubleSpinBox()
        self.edge_distance.setDecimals(3); self.edge_distance.setRange(.001,1); self.edge_distance.setValue(.08)
        form.addRow("补选邻距 (m)", self.edge_radius)
        form.addRow("补选面距 (m)", self.edge_distance)
        for spin in (self.normal_k, self.neighbor_radius, self.plane_distance, self.normal_angle, self.edge_radius, self.edge_distance):
            spin.setKeyboardTracking(False)
            spin.installEventFilter(self)
            spin.lineEdit().installEventFilter(self)
        tools.addWidget(self.growth_settings)
        self.growth_settings.hide()
        self.growth_settings_toggle.toggled.connect(self.growth_settings.setVisible)
        self.expand_button = QtWidgets.QPushButton("局部扩展 [E]（先预览）")
        self.expand_button.clicked.connect(control.expand_local)
        tools.addWidget(self.expand_button)
        row = QtWidgets.QHBoxLayout()
        self.confirm_expansion = QtWidgets.QPushButton("确认扩展 [Enter]")
        self.cancel_expansion = QtWidgets.QPushButton("取消扩展 [Esc]")
        self.confirm_expansion.clicked.connect(control.confirm_expansion)
        self.cancel_expansion.clicked.connect(control.cancel_expansion)
        row.addWidget(self.confirm_expansion)
        row.addWidget(self.cancel_expansion)
        tools.addLayout(row)
        self.expansion_info = QtWidgets.QLabel("先选当前 Plane，再圈选至少 6 个相邻种子点。")
        self.expansion_info.setWordWrap(True)
        tools.addWidget(self.expansion_info)
        self.set_preview(False)
        for text, callback in [
            ("新建 Plane [N]", control.new_plane),
            ("设为未分配 / Background", lambda: control.assign(0)),
            ("合并当前 Plane 到…", control.merge),
            ("删除当前 Plane（点恢复为 0）", control.delete),
            ("撤销 [Ctrl+Z]", control.undo),
            ("清除选择 [Esc]", control.clear_selection),
            ("保存工作标注 [Ctrl+S]", lambda: control.owner.save(force_plane=True)),
            ("导出最终 LAS / LAZ", control.export_final),
        ]:
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda checked=False, fn=callback: fn())
            tools.addWidget(button)
        advanced_toggle = QtWidgets.QToolButton()
        advanced_toggle.setText("工作区 / 检查进度 / 书签 ▸")
        advanced_toggle.setCheckable(True)
        tools.addWidget(advanced_toggle)
        advanced = QtWidgets.QWidget()
        advanced_layout = QtWidgets.QGridLayout(advanced)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        for i, (text, callback) in enumerate([
            ("视域设为工作区", lambda: control.set_work_roi(False)),
            ("选区设为工作区", lambda: control.set_work_roi(True)),
            ("清除工作区", control.clear_work_roi),
            ("建立检查网格", control.make_review_grid),
            ("工作区已检查", control.mark_reviewed),
            ("下一未检查区", control.next_unreviewed),
            ("添加位置书签", control.add_bookmark),
            ("删除当前书签", control.delete_bookmark),
            ("上一书签", lambda: control.step_bookmark(-1)),
            ("下一书签", lambda: control.step_bookmark(1)),
        ]):
            button=QtWidgets.QPushButton(text)
            button.clicked.connect(lambda checked=False, fn=callback: fn())
            advanced_layout.addWidget(button, i//2, i%2)
        self.roi_boundary=QtWidgets.QCheckBox("显示工作区边界")
        self.roi_boundary.setChecked(True)
        self.roi_boundary.toggled.connect(lambda _: control.view.gl_widget.update())
        advanced_layout.addWidget(self.roi_boundary,5,0,1,2)
        self.grid_size=QtWidgets.QDoubleSpinBox()
        self.grid_size.setRange(1,1000); self.grid_size.setValue(50)
        self.grid_size.setSuffix(" m / grid")
        self.grid_size.installEventFilter(self); self.grid_size.lineEdit().installEventFilter(self)
        advanced_layout.addWidget(self.grid_size,6,0,1,2)
        self.grid_status=QtWidgets.QLabel("检查网格尚未建立")
        advanced_layout.addWidget(self.grid_status,7,0,1,2)
        self.interactive_lod=QtWidgets.QCheckBox("旋转期间 1/8 LOD（仅显示）")
        advanced_layout.addWidget(self.interactive_lod,8,0,1,2)
        self.debug_perf=QtWidgets.QCheckBox("性能日志")
        advanced_layout.addWidget(self.debug_perf,9,0,1,2)
        tools.addWidget(advanced); advanced.hide()
        advanced_toggle.toggled.connect(advanced.setVisible)
        # 展开参数时允许侧栏滚动，避免小屏幕下确认/保存按钮被挤出窗口。
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(self.tools)
        layout.addWidget(scroll, 1)
        self.shortcuts = []
        actions = [("A", lambda: control.assign()), ("E", control.expand_local)]
        for number, name in enumerate(("top", "front", "left", "bottom", "back", "right"), 1):
            actions.append((f"Alt+{number}", lambda name=name: control.owner.set_standard_view(name)))
        actions += [("N", control.new_plane), ("Ctrl+Z", control.undo),
                    ("H", control.hide_selected), ("Shift+H", control.restore_hidden),
                    ("V", lambda: control.set_tool("navigate")),
                    ("B", lambda: control.set_tool("rectangle")),
                    ("L", lambda: control.set_tool("lasso")),
                    ("Return", control.confirm_expansion),
                    ("Enter", control.confirm_expansion),
                    ("Escape", control.clear_selection),
                    ("Ctrl+S", lambda: control.owner.save(force_plane=True))]
        for key, callback in actions:
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(key), parent)
            shortcut.setContext(QtCore.Qt.WindowShortcut)
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(callback)
            shortcut.setEnabled(False)
            self.shortcuts.append(shortcut)

    def eventFilter(self, obj, event):
        # 参数输入中的数字、回车和撤销交给输入框，不能触发逐点标注。
        if event.type() == QtCore.QEvent.ShortcutOverride:
            event.accept()
            return True
        return super().eventFilter(obj, event)

    def growth_options(self):
        return GrowthOptions(self.normal_k.value(), self.neighbor_radius.value(),
                             self.plane_distance.value(), self.normal_angle.value(),
                             self.edge_completion.isChecked(), self.edge_radius.value(), self.edge_distance.value())

    def set_preview(self, pending):
        self.confirm_expansion.setEnabled(pending)
        self.cancel_expansion.setEnabled(pending)
        self.expand_button.setEnabled(not pending)
        self.growth_settings.setEnabled(not pending)
        self.selection_scope.setEnabled(not pending)
        self.show_hidden.setEnabled(not pending)

    def set_available(self, enabled):
        self.tools.setEnabled(enabled)
        for shortcut in self.shortcuts:
            shortcut.setEnabled(enabled)

    def refresh(self, model):
        self.table_model.refresh(model)
        if model.current in self.table_model.ids:
            row = self.table_model.ids.index(model.current)
            self.list.selectRow(row)
        self.assign_button.setText(f"赋给当前 Plane {model.current} [A]")
        self.selection_info.setText(f"已选择 {len(self.control.selected_ids)} points" + ("  • 未保存" if model.dirty else ""))
        if model.undo_discarded:
            self.selection_info.setText(self.selection_info.text()+"\n撤销历史已按内存上限截断")



def layout_widgets(layout):
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item.widget() is not None:
            yield item.widget()
        elif item.layout() is not None:
            yield from layout_widgets(item.layout())
