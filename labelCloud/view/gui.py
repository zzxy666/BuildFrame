import logging
import os
import re
import sys
import traceback
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Set

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import QEvent
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QColorDialog,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMessageBox,
)

from ..control.config_manager import config , config_manager
#from ..definitions import Color3f, LabelingMode
#from ..io.labels.config import LabelConfig
from ..io.pointclouds import BasePointCloudHandler
#from ..labeling_strategies import PickingStrategy, SpanningStrategy
from ..model.point_cloud import PointCloud
from .i18n import LANGUAGE_ENGLISH, language_manager, tr
from .settings_dialog import SettingsDialog  # type: ignore
#from .startup.dialog import StartupDialog
from .status_manager import StatusManager
from .viewer import GLWidget

if TYPE_CHECKING:
    from ..control.controller import Controller


def string_is_float(string: str, recect_negative: bool = False) -> bool:
    """Returns True if string can be converted to float"""
    try:
        decimal = float(string)
    except ValueError:
        return False
    if recect_negative and decimal < 0:
        return False
    return True


def set_floor_visibility(state: bool) -> None:
    logging.info(
        "%s floor grid (SHOW_FLOOR: %s).",
        "Activated" if state else "Deactivated",
        state,
    )
    config.set("USER_INTERFACE", "show_floor", str(state))


def set_orientation_visibility(state: bool) -> None:
    config.set("USER_INTERFACE", "show_orientation", str(state))


def set_zrotation_only(state: bool) -> None:
    config.set("USER_INTERFACE", "z_rotation_only", str(state))


def set_color_with_label(state: bool) -> None:
    config.set("POINTCLOUD", "color_with_label", str(state))


# def set_keep_perspective(state: bool) -> None:
#     config.set("USER_INTERFACE", "keep_perspective", str(state))


def set_propagate_labels(state: bool) -> None:
    config.set("LABEL", "propagate_labels", str(state))


# CSS file paths need to be set dynamically
STYLESHEET = """
    * {{
        background-color: #FFF;
        font-family: "DejaVu Sans", Arial;
    }}

    QMenu::item:selected {{
        background-color: #0000DD;
    }}

    QListWidget#label_list::item {{
        padding-left: 22px;
        padding-top: 7px;
        padding-bottom: 7px;
        background: url("{icons_dir}/cube-outline.svg") center left no-repeat;
    }}

    QListWidget#label_list::item:selected {{
        color: #FFF;
        border: none;
        background: rgb(0, 0, 255);
        background: url("{icons_dir}/cube-outline_white.svg") center left no-repeat, #0000ff;
    }}

    QComboBox#current_class_dropdown::item:checked{{
        color: gray;
    }}

    QComboBox#current_class_dropdown::item:selected {{
        color: #FFFFFF;
    }}

    QComboBox#current_class_dropdown{{
        selection-background-color: #0000FF;
    }}
"""

_MAIN_WINDOW_UI, _ = uic.loadUiType(
    str(
        resources.files("labelCloud.resources.interfaces").joinpath(
            "interface_roof.ui"
        )
    )
)


class GUI(QtWidgets.QMainWindow, _MAIN_WINDOW_UI):
    def __init__(self, control: "Controller") -> None:
        super(GUI, self).__init__()
        self.setupUi(self)
        self._current_pcd_name = None
        self.resize(1500, 900)
        self.setWindowTitle(tr("BuildFrame 点云标注"))
        self.setStyleSheet(
            STYLESHEET.format(
                icons_dir=str(
                    Path(__file__)
                    .resolve()
                    .parent.parent.joinpath("resources")
                    .joinpath("icons")
                    .as_posix()
                )
            )
        )

        # MENU BAR
        # File
        self.act_set_pcd_folder: QtWidgets.QAction
        self.act_set_label_folder: QtWidgets.QAction

        # Labels
        self.act_delete_all_labels: QtWidgets.QAction
        #self.act_set_default_class: QtWidgets.QMenu
        #self.actiongroup_default_class = QActionGroup(self.act_set_default_class)
        self.act_propagate_labels: QtWidgets.QAction

        # Settings
        #self.act_z_rotation_only: QtWidgets.QAction
        #self.act_color_with_label: QtWidgets.QAction
        self.act_show_floor: QtWidgets.QAction
        #self.act_show_orientation: QtWidgets.QAction
        #self.act_save_perspective: QtWidgets.QAction
        #self.act_align_pcd: QtWidgets.QAction
        self.act_change_settings: QtWidgets.QAction

        # STATUS BAR
        self.status_bar: QtWidgets.QStatusBar
        self.status_manager = StatusManager(self.status_bar)

        # CENTRAL WIDGET
        self.gl_widget: GLWidget

        # LEFT PANEL
        # point cloud management
        self.label_current_pcd: QtWidgets.QLabel
        self.button_prev_pcd: QtWidgets.QPushButton
        self.button_next_pcd: QtWidgets.QPushButton
        self.button_set_pcd: QtWidgets.QPushButton
        self.progressbar_pcds: QtWidgets.QProgressBar

        # bbox control section
        """self.button_bbox_up: QtWidgets.QPushButton
        self.button_bbox_down: QtWidgets.QPushButton
        self.button_bbox_left: QtWidgets.QPushButton
        self.button_bbox_right: QtWidgets.QPushButton
        self.button_bbox_forward: QtWidgets.QPushButton
        self.button_bbox_backward: QtWidgets.QPushButton
        self.dial_bbox_z_rotation: QtWidgets.QDial
        self.button_bbox_decrease_dimension: QtWidgets.QPushButton
        self.button_bbox_increase_dimension: QtWidgets.QPushButton"""

        #添加Add vertices lines and save labels
        self.button_add_vertices: QtWidgets.QPushButton
        self.connect_two_vertices: QtWidgets.QPushButton
        self.button_add_lines: QtWidgets.QPushButton
        self.button_save_label: QtWidgets.QPushButton

        self.button_point_cloud_filtering: QtWidgets.QPushButton

        #点微调按钮
        self.button_startpt_forward: QtWidgets.QPushButton
        self.button_startpt_backward: QtWidgets.QPushButton
        self.button_startpt_left: QtWidgets.QPushButton
        self.button_startpt_right: QtWidgets.QPushButton
        self.button_startpt_up: QtWidgets.QPushButton
        self.button_startpt_down: QtWidgets.QPushButton
        #视角按钮
        self.button_view_top: QtWidgets.QPushButton
        self.button_view_bottom: QtWidgets.QPushButton
        self.button_view_front: QtWidgets.QPushButton
        self.button_view_back: QtWidgets.QPushButton
        self.button_view_left: QtWidgets.QPushButton
        self.button_view_right: QtWidgets.QPushButton

        #按钮高亮显示
        self.button_add_vertices.setCheckable(True)
        self.connect_two_vertices.setCheckable(True)
        self.button_add_lines.setCheckable(True)


        # 2d image viewer
        """self.button_show_image: QtWidgets.QPushButton
        self.button_show_image.setVisible(
            config.getboolean("USER_INTERFACE", "show_2d_image")
        )"""

        # label mode selection
        """self.button_pick_bbox: QtWidgets.QPushButton
        self.button_span_bbox: QtWidgets.QPushButton
        self.button_save_label: QtWidgets.QPushButton"""

        # RIGHT PANEL
        self.label_list: QtWidgets.QListWidget
        #self.current_class_dropdown: QtWidgets.QComboBox
        self.button_deselect_label: QtWidgets.QPushButton
        self.button_delete_label: QtWidgets.QPushButton
        #self.button_assign_label: QtWidgets.QPushButton

        # label list actions
        # self.act_rename_class = QtWidgets.QAction("Rename class") #TODO: Implement!
        #self.act_change_class_color = QtWidgets.QAction("Change class color")
        #self.act_delete_class = QtWidgets.QAction("Delete label")
        #self.act_crop_pointcloud_inside = QtWidgets.QAction("Save points inside as")
        # self.label_list.addActions(
        #     [
        #         self.act_change_class_color,
        #         self.act_delete_class,
        #         self.act_crop_pointcloud_inside,
        #     ]
        # )
        self.label_list.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self._setup_coordinate_panel()
        self._setup_language_sensitive_fonts()

        # BOUNDING BOX PARAMETER EDITS
        """self.edit_pos_x: QtWidgets.QLineEdit
        self.edit_pos_y: QtWidgets.QLineEdit
        self.edit_pos_z: QtWidgets.QLineEdit

        self.edit_length: QtWidgets.QLineEdit
        self.edit_width: QtWidgets.QLineEdit
        self.edit_height: QtWidgets.QLineEdit

        self.edit_rot_x: QtWidgets.QLineEdit
        self.edit_rot_y: QtWidgets.QLineEdit
        self.edit_rot_z: QtWidgets.QLineEdit

        self.all_line_edits = [
            self.edit_pos_x,
            self.edit_pos_y,
            self.edit_pos_z,
            self.edit_length,
            self.edit_width,
            self.edit_height,
            self.edit_rot_x,
            self.edit_rot_y,
            self.edit_rot_z,
        ]

        self.label_volume: QtWidgets.QLabel"""

        # 坐标定义


        self.controller = control

        # Connect all events to functions
        self.connect_events()
        self.set_checkbox_states()  # tick in menu

        # Run startup dialog
        # self.startup_dialog = StartupDialog()
        # if self.startup_dialog.exec():
        #     pass
        # else:
        #     sys.exit()
        # Segmentation only functionalities
        # if LabelConfig().type == LabelingMode.OBJECT_DETECTION:
        #     self.button_assign_label.setVisible(False)
        #     self.act_color_with_label.setVisible(False)

        # Connect with controller
        self.controller.startup(self)

        # Start event cycle
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(20)  # period, in milliseconds
        self.timer.timeout.connect(self.controller.loop_gui)
        self.timer.start()

    def _setup_coordinate_panel(self) -> None:
        """Turn the compact coordinate boxes into readable property grids."""
        self.coordinate_precision = max(
            0,
            min(
                12,
                config.getint(
                    "USER_INTERFACE", "coordinate_precision", fallback=3
                ),
            ),
        )

        coordinate_font = QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.FixedFont
        )
        coordinate_font.setPointSize(9)

        coordinate_edits = (
            self.start_pt_x,
            self.start_pt_y,
            self.start_pt_z,
            self.end_pt_x,
            self.end_pt_y,
            self.end_pt_z,
            self.current_pt_x,
            self.current_pt_y,
            self.current_pt_z,
        )
        for edit in coordinate_edits:
            edit.setReadOnly(True)
            edit.setFrame(True)
            edit.setFont(coordinate_font)
            edit.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            edit.setMinimumHeight(24)
            edit.setToolTip(tr("选择坐标后按 Ctrl+C 可复制"))
            edit.setStyleSheet(
                "QLineEdit { background: #f7f7f7; padding: 1px 4px; "
                "border: 1px solid #c8c8c8; border-radius: 2px; }"
                "QLineEdit:focus { border: 1px solid #3478d4; "
                "background: white; }"
            )

        # A narrow side panel is easier to scan as an X/Y/Z property grid than
        # as six short fields laid out in a single row.
        self.label.hide()
        self.label_2.hide()
        line_layout = QtWidgets.QGridLayout(self.groupBox)
        line_layout.setContentsMargins(8, 20, 8, 8)
        line_layout.setHorizontalSpacing(6)
        line_layout.setVerticalSpacing(4)
        self._line_coordinate_labels = []
        for row, (label_text, edit) in enumerate(
            (
                ("起点 X", self.start_pt_x),
                ("起点 Y", self.start_pt_y),
                ("起点 Z", self.start_pt_z),
                ("终点 X", self.end_pt_x),
                ("终点 Y", self.end_pt_y),
                ("终点 Z", self.end_pt_z),
            )
        ):
            label = QLabel(tr(label_text))
            self._line_coordinate_labels.append((label, label_text))
            line_layout.addWidget(label, row, 0)
            line_layout.addWidget(edit, row, 1)
        line_layout.setColumnStretch(1, 1)
        self.groupBox.setMinimumHeight(212)

        self.label_3.hide()
        point_layout = QtWidgets.QGridLayout(self.current_pt)
        point_layout.setContentsMargins(8, 20, 8, 8)
        point_layout.setHorizontalSpacing(6)
        point_layout.setVerticalSpacing(4)
        self._point_coordinate_labels = []
        for row, (axis, edit) in enumerate(
            (
                ("X", self.current_pt_x),
                ("Y", self.current_pt_y),
                ("Z", self.current_pt_z),
            )
        ):
            label = QLabel(axis)
            self._point_coordinate_labels.append(label)
            point_layout.addWidget(label, row, 0, alignment=QtCore.Qt.AlignCenter)
            point_layout.addWidget(edit, row, 1)
        point_layout.setColumnStretch(1, 1)
        self.current_pt.setMinimumHeight(108)

        # Let the list yield vertical space on smaller displays instead of
        # forcing the coordinate fields outside the window.
        self.label_list.setMinimumHeight(140)
        self.label_list.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding
        )
        self._retranslate_coordinate_panel()

    def _retranslate_coordinate_panel(self) -> None:
        self.groupBox.setTitle(tr("选中线坐标"))
        self.current_pt.setTitle(tr("选中点坐标"))
        for label, source_text in self._line_coordinate_labels:
            label.setText(tr(source_text))
        for edit in (
            self.start_pt_x,
            self.start_pt_y,
            self.start_pt_z,
            self.end_pt_x,
            self.end_pt_y,
            self.end_pt_z,
            self.current_pt_x,
            self.current_pt_y,
            self.current_pt_z,
        ):
            coordinate_value = edit.property("coordinate_value")
            coordinate_axis = edit.property("coordinate_axis")
            if coordinate_value is not None and coordinate_axis:
                edit.setToolTip(
                    tr(
                        "{axis} = {value:.12f}\n点击字段后按 Ctrl+A、Ctrl+C 可复制完整显示值",
                        axis=tr(coordinate_axis),
                        value=float(coordinate_value),
                    )
                )
            else:
                edit.setToolTip(tr("选择坐标后按 Ctrl+C 可复制"))

    def _setup_language_sensitive_fonts(self) -> None:
        self._compact_english_buttons = (
            self.button_view_top,
            self.button_view_bottom,
            self.button_view_front,
            self.button_view_back,
            self.button_view_left,
            self.button_view_right,
        )
        for button in self._compact_english_buttons:
            button.setProperty("base_font", button.font())
        self._apply_language_sensitive_fonts()

    def _apply_language_sensitive_fonts(self) -> None:
        if not hasattr(self, "_compact_english_buttons"):
            return
        for button in self._compact_english_buttons:
            font = QtGui.QFont(button.property("base_font"))
            if language_manager.language == LANGUAGE_ENGLISH:
                font.setPointSize(min(font.pointSize(), 9))
            button.setFont(font)

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() != QtCore.QEvent.LanguageChange or not hasattr(
            self, "groupBox"
        ):
            return
        self.retranslateUi(self)
        if hasattr(self, "act_ground_filter_settings"):
            self.act_ground_filter_settings.setText(tr("地面滤波设置…"))
        if hasattr(self, "_line_coordinate_labels"):
            self._retranslate_coordinate_panel()
        self._apply_language_sensitive_fonts()
        if hasattr(self, "status_manager"):
            self.status_manager.retranslate()
        if self._current_pcd_name is not None:
            self.set_pcd_label(self._current_pcd_name)
        if hasattr(self, "controller"):
            self.controller.filter_controller.retranslate()

    def _set_coordinate_value(
        self, edit: QtWidgets.QLineEdit, axis: str, value: float
    ) -> None:
        numeric_value = float(value)
        edit.setText(f"{numeric_value:.{self.coordinate_precision}f}")
        edit.setCursorPosition(0)
        edit.setProperty("coordinate_axis", axis)
        edit.setProperty("coordinate_value", numeric_value)
        edit.setToolTip(
            tr(
                "{axis} = {value:.12f}\n点击字段后按 Ctrl+A、Ctrl+C 可复制完整显示值",
                axis=tr(axis),
                value=numeric_value,
            )
        )

    def show_vertex_coordinates(self, point) -> None:
        """Display a selected world-space vertex in the coordinate panel."""
        for axis, edit, value in zip(
            "XYZ",
            (self.current_pt_x, self.current_pt_y, self.current_pt_z),
            point,
        ):
            self._set_coordinate_value(edit, axis, value)

    def show_line_coordinates(self, start, end) -> None:
        """Display both endpoints of a selected world-space line."""
        for axis, start_edit, end_edit, start_value, end_value in zip(
            "XYZ",
            (self.start_pt_x, self.start_pt_y, self.start_pt_z),
            (self.end_pt_x, self.end_pt_y, self.end_pt_z),
            start,
            end,
        ):
            self._set_coordinate_value(start_edit, f"起点 {axis}", start_value)
            self._set_coordinate_value(end_edit, f"终点 {axis}", end_value)

    # Event connectors
    def connect_events(self) -> None:
        # POINTCLOUD CONTROL
        self.button_next_pcd.clicked.connect(
            lambda: self.controller.next_pcd(save=True)
        )
        self.button_prev_pcd.clicked.connect(self.controller.prev_pcd)

        # BBOX CONTROL
        """self.button_bbox_up.pressed.connect(
            lambda: self.controller.bbox_controller.translate_along_z()
        )
        self.button_bbox_down.pressed.connect(
            lambda: self.controller.bbox_controller.translate_along_z(down=True)
        )
        self.button_bbox_left.pressed.connect(
            lambda: self.controller.bbox_controller.translate_along_x(left=True)
        )
        self.button_bbox_right.pressed.connect(
            self.controller.bbox_controller.translate_along_x
        )
        self.button_bbox_forward.pressed.connect(
            lambda: self.controller.bbox_controller.translate_along_y(forward=True)
        )"""
        self.button_set_pcd.pressed.connect(lambda: self.ask_custom_index())
        """self.button_bbox_backward.pressed.connect(
            lambda: self.controller.bbox_controller.translate_along_y()
        )

        self.dial_bbox_z_rotation.valueChanged.connect(
            lambda x: self.controller.bbox_controller.rotate_around_z(x, absolute=True)
        )
        self.button_bbox_decrease_dimension.clicked.connect(
            lambda: self.controller.bbox_controller.scale(decrease=True)
        )
        self.button_bbox_increase_dimension.clicked.connect(
            lambda: self.controller.bbox_controller.scale()
        )"""

        # LABELING CONTROL
        # self.current_class_dropdown.currentTextChanged.connect(
        #     self.controller.bbox_controller.set_classname
        # )
        # self.button_deselect_label.clicked.connect(
        #     self.controller.bbox_controller.deselect_bbox
        # )
        # self.button_delete_label.clicked.connect(
        #     self.controller.bbox_controller.delete_current_bbox
        # )
        # self.label_list.currentRowChanged.connect(   
        #     self.controller.bbox_controller.set_active_bbox
        # )
        # self.button_assign_label.clicked.connect(
        #     self.controller.bbox_controller.assign_point_label_in_active_box
        # )
        # context menu
        # self.act_delete_class.triggered.connect(
        #     self.controller.bbox_controller.delete_current_bbox
        # )
        # self.act_crop_pointcloud_inside.triggered.connect(
        #     self.controller.crop_pointcloud_inside_active_bbox
        # )
        # self.act_change_class_color.triggered.connect(self.change_label_color)

        # open_2D_img
        #self.button_show_image.pressed.connect(lambda: self.show_2d_image())

        # LABEL CONTROL
        """self.button_pick_bbox.clicked.connect(
            lambda: self.controller.drawing_mode.set_drawing_strategy(
                PickingStrategy(self)
            )
        )
        self.button_span_bbox.clicked.connect(
            lambda: self.controller.drawing_mode.set_drawing_strategy(
                SpanningStrategy(self)
            )
        )
        self.button_save_label.clicked.connect(self.controller.save)"""

        # BOUNDING BOX PARAMETER
        """self.edit_pos_x.editingFinished.connect(
            lambda: self.update_bbox_parameter("pos_x")
        )
        self.edit_pos_y.editingFinished.connect(
            lambda: self.update_bbox_parameter("pos_y")
        )
        self.edit_pos_z.editingFinished.connect(
            lambda: self.update_bbox_parameter("pos_z")
        )

        self.edit_length.editingFinished.connect(
            lambda: self.update_bbox_parameter("length")
        )
        self.edit_width.editingFinished.connect(
            lambda: self.update_bbox_parameter("width")
        )
        self.edit_height.editingFinished.connect(
            lambda: self.update_bbox_parameter("height")
        )

        self.edit_rot_x.editingFinished.connect(
            lambda: self.update_bbox_parameter("rot_x")
        )
        self.edit_rot_y.editingFinished.connect(
            lambda: self.update_bbox_parameter("rot_y")
        )
        self.edit_rot_z.editingFinished.connect(
            lambda: self.update_bbox_parameter("rot_z")
        )"""

        # MENU BAR
        self.act_set_pcd_folder.triggered.connect(self.change_pointcloud_folder)
        self.act_set_label_folder.triggered.connect(self.change_label_folder)
        # self.actiongroup_default_class.triggered.connect(
        #     self.change_default_object_class
        # )
        # self.act_delete_all_labels.triggered.connect(
        #     self.controller.bbox_controller.reset
        # )
        self.act_propagate_labels.toggled.connect(set_propagate_labels)
        #self.act_z_rotation_only.toggled.connect(set_zrotation_only)
        #self.act_color_with_label.toggled.connect(set_color_with_label)
        self.act_show_floor.toggled.connect(set_floor_visibility)
        #self.act_show_orientation.toggled.connect(set_orientation_visibility)
        #self.act_save_perspective.toggled.connect(set_keep_perspective)
        #self.act_align_pcd.toggled.connect(self.controller.align_mode.change_activation)
        self.act_change_settings.triggered.connect(self.show_settings_dialog)
        self.act_ground_filter_settings = QtWidgets.QAction(tr("地面滤波设置…"), self)
        self.menuSettings.addAction(self.act_ground_filter_settings)
        self.act_ground_filter_settings.triggered.connect(self.show_ground_filter_settings)

     
        self.button_add_vertices.clicked.connect(lambda: self.toggle_roof_mode("point"))
        self.connect_two_vertices.clicked.connect(self.toggle_connect_mode)
        self.button_add_lines.clicked.connect(lambda: self.toggle_roof_mode("line"))

        #self.label_list.itemClicked.connect(self.controller.on_point_item_clicked)
        self.label_list.itemClicked.connect(self.controller.on_label_item_clicked)

        self.button_point_cloud_filtering.clicked.connect((self.controller.filter_pointcloud))

        # 删除按钮连接
        self.button_delete_label.clicked.connect(self.controller.delete_selected_label)

        #点微调按钮信号连接
        self.button_startpt_forward.clicked.connect(self.controller.vertex_forward)
        self.button_startpt_backward.clicked.connect(self.controller.vertex_backward)
        self.button_startpt_left.clicked.connect(self.controller.vertex_left)
        self.button_startpt_right.clicked.connect(self.controller.vertex_right)
        self.button_startpt_up.clicked.connect(self.controller.vertex_up)
        self.button_startpt_down.clicked.connect(self.controller.vertex_down)
        # 键盘快捷键
        QtWidgets.QShortcut(QtGui.QKeySequence("W"), self).activated.connect(self.controller.vertex_forward)
        QtWidgets.QShortcut(QtGui.QKeySequence("S"), self).activated.connect(self.controller.vertex_backward)
        QtWidgets.QShortcut(QtGui.QKeySequence("A"), self).activated.connect(self.controller.vertex_left)
        QtWidgets.QShortcut(QtGui.QKeySequence("D"), self).activated.connect(self.controller.vertex_right)
        QtWidgets.QShortcut(QtGui.QKeySequence("Q"), self).activated.connect(self.controller.vertex_up)
        QtWidgets.QShortcut(QtGui.QKeySequence("E"), self).activated.connect(self.controller.vertex_down)
        #视角按钮连接
        self.button_view_top.clicked.connect(lambda: self.controller.set_standard_view("top"))
        self.button_view_bottom.clicked.connect(lambda: self.controller.set_standard_view("bottom"))
        self.button_view_front.clicked.connect(lambda: self.controller.set_standard_view("front"))
        self.button_view_back.clicked.connect(lambda: self.controller.set_standard_view("back"))
        self.button_view_left.clicked.connect(lambda: self.controller.set_standard_view("left"))
        self.button_view_right.clicked.connect(lambda: self.controller.set_standard_view("right"))

        self.button_save_label.clicked.connect(self.save_roof_annotations)

    def set_checkbox_states(self) -> None:
        # self.act_propagate_labels.setChecked(
        #     config.getboolean("LABEL", "propagate_labels")
        # )
        self.act_show_floor.setChecked(
            config.getboolean("USER_INTERFACE", "show_floor")
        )
        # self.act_show_orientation.setChecked(
        #     config.getboolean("USER_INTERFACE", "show_orientation")
        # )
        # self.act_z_rotation_only.setChecked(
        #     config.getboolean("USER_INTERFACE", "z_rotation_only")
        # )
        # self.act_color_with_label.setChecked(
        #     config.getboolean("POINTCLOUD", "color_with_label")
        # )

    # Collect, filter and forward events to viewer
    def eventFilter(self, event_object, event) -> bool:
        roof = self.controller.roof_plane_controller
        if event_object == self.gl_widget and roof.handle_mouse(event):
            return True
        if roof.active and event.type() == QEvent.KeyPress and event_object == self.gl_widget:
            self.controller.key_press_event(event)
            self.gl_widget.update()
            return True
        # Keyboard Events
        if (event.type() == QEvent.KeyPress) and event_object in [
            self,
            self.label_list,  # otherwise steals focus for keyboard shortcuts
        ]:
            self.controller.key_press_event(event)
            #self.update_bbox_stats(self.controller.bbox_controller.get_active_bbox())
            return True  # TODO: Recheck pyqt behaviour
        # elif event.type() == QEvent.KeyRelease:
        #     self.controller.key_release_event(event)

        # Mouse Events
        elif (event.type() == QEvent.MouseMove) and (event_object == self.gl_widget):
            self.controller.mouse_move_event(event)
            #self.update_bbox_stats(self.controller.bbox_controller.get_active_bbox())
        elif (event.type() == QEvent.Wheel) and (event_object == self.gl_widget):
            self.controller.mouse_scroll_event(event)
            #self.update_bbox_stats(self.controller.bbox_controller.get_active_bbox())
        elif event.type() == QEvent.MouseButtonDblClick and (
            event_object == self.gl_widget
        ):
            self.controller.mouse_double_clicked(event)
            return True
        # 鼠标按下：记录初始位置
        elif (event.type() == QEvent.MouseButtonPress) and (event_object == self.gl_widget):
            self.controller.mouse_pressed(event)
        
        elif (event.type() == QEvent.MouseButtonRelease) and (event_object == self.gl_widget):
            self.controller.mouse_released(event)

        # elif (event.type() == QEvent.MouseButtonPress) and (
        #     event_object == self.gl_widget
        # ):
        #     self.controller.mouse_clicked(event)

            #self.update_bbox_stats(self.controller.bbox_controller.get_active_bbox())
        # elif (event.type() == QEvent.MouseButtonPress) and (
        #     event_object != self.current_class_dropdown
        # ):
        #     self.current_class_dropdown.clearFocus()
        #     self.update_bbox_stats(self.controller.bbox_controller.get_active_bbox())
        return False

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        if not self.controller.save():
            a0.ignore()
            return
        self.controller.filter_controller.shutdown()
        logging.info("Closing window after saving ...")
        self.timer.stop()
        a0.accept()

    def show_settings_dialog(self) -> None:
        dialog = SettingsDialog(self)
        dialog.exec()

    def show_ground_filter_settings(self) -> None:
        from .ground_filter_dialog import GroundFilterDialog

        GroundFilterDialog(self).exec_()

    def show_2d_image(self):
        """Searches for a 2D image with the point cloud name and displays it in a new window."""
        image_folder = config.getpath("FILE", "image_folder")

        # Look for image files with the name of the point cloud
        pcd_name = self.controller.pcd_manager.pcd_path.stem
        image_file_pattern = re.compile(
            f"{pcd_name}+(\\.(?i:(jpe?g|png|gif|bmp|tiff)))"
        )

        try:
            image_name = next(
                filter(image_file_pattern.search, os.listdir(image_folder))
            )
        except StopIteration:
            QMessageBox.information(
                self,
                tr("未找到二维图像"),
                tr(
                    "在图像文件夹（{folder}）中未找到对应图像。\n请检查文件夹路径以及是否存在与当前点云同名的图像。",
                    folder=image_folder,
                ),
                QMessageBox.Ok,
            )
        else:
            image_path = image_folder.joinpath(image_name)
            image = QtGui.QImage(QtGui.QImageReader(str(image_path)).read())
            self.imageLabel = QLabel()
            self.imageLabel.setWindowTitle(tr("二维图像（{name}）", name=image_name))
            self.imageLabel.setPixmap(QPixmap.fromImage(image))
            self.imageLabel.show()

    def show_no_pointcloud_dialog(
        self, pcd_folder: Path, pcd_extensions: Set[str]
    ) -> None:
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setText(
            tr("<b>指定文件夹中没有找到有效的点云文件。</b>")
        )
        msg.setInformativeText(
            tr(
                "请将点云文件放入 <code>{folder}</code>，或重新设置点云文件夹。当前支持以下点云格式：\n{formats}。",
                folder=pcd_folder.resolve(),
                formats=", ".join(sorted(pcd_extensions)),
            )
        )
        msg.setWindowTitle(tr("未找到点云文件"))
        msg.exec_()

    # VISUALIZATION METHODS

    def set_pcd_label(self, pcd_name: str) -> None:
        self._current_pcd_name = pcd_name
        self.label_current_pcd.setText(tr("当前：<em>{name}</em>", name=pcd_name))

    def init_progress(self, min_value, max_value):
        self.progressbar_pcds.setMinimum(min_value)
        self.progressbar_pcds.setMaximum(max_value)

    def update_progress(self, value) -> None:
        self.progressbar_pcds.setValue(value)

    def update_current_class_dropdown(self) -> None:
        self.controller.pcd_manager.populate_class_dropdown()

    def update_bbox_stats(self, bbox) -> None:
        viewing_precision = config.getint("USER_INTERFACE", "viewing_precision")
        if bbox and not self.line_edited_activated():
            self.edit_pos_x.setText(str(round(bbox.get_center()[0], viewing_precision)))
            self.edit_pos_y.setText(str(round(bbox.get_center()[1], viewing_precision)))
            self.edit_pos_z.setText(str(round(bbox.get_center()[2], viewing_precision)))

            self.edit_length.setText(
                str(round(bbox.get_dimensions()[0], viewing_precision))
            )
            self.edit_width.setText(
                str(round(bbox.get_dimensions()[1], viewing_precision))
            )
            self.edit_height.setText(
                str(round(bbox.get_dimensions()[2], viewing_precision))
            )

            self.edit_rot_x.setText(str(round(bbox.get_x_rotation(), 1)))
            self.edit_rot_y.setText(str(round(bbox.get_y_rotation(), 1)))
            self.edit_rot_z.setText(str(round(bbox.get_z_rotation(), 1)))

            self.label_volume.setText(str(round(bbox.get_volume(), viewing_precision)))

    def update_bbox_parameter(self, parameter: str) -> None:
        str_value = None
        self.setFocus()  # Changes the focus from QLineEdit to the window

        if parameter == "pos_x":
            str_value = self.edit_pos_x.text()
        if parameter == "pos_y":
            str_value = self.edit_pos_y.text()
        if parameter == "pos_z":
            str_value = self.edit_pos_z.text()
        # if str_value and string_is_float(str_value):
        #     self.controller.bbox_controller.update_position(parameter, float(str_value))
        #     return

        if parameter == "length":
            str_value = self.edit_length.text()
        if parameter == "width":
            str_value = self.edit_width.text()
        if parameter == "height":
            str_value = self.edit_height.text()
        # if str_value and string_is_float(str_value, recect_negative=True):
        #     self.controller.bbox_controller.update_dimension(
        #         parameter, float(str_value)
        #     )
        #     return

        if parameter == "rot_x":
            str_value = self.edit_rot_x.text()
        if parameter == "rot_y":
            str_value = self.edit_rot_y.text()
        if parameter == "rot_z":
            str_value = self.edit_rot_z.text()
        # if str_value and string_is_float(str_value):
        #     self.controller.bbox_controller.update_rotation(parameter, float(str_value))
        #     return

    # Enables, disables the draw mode
    def activate_draw_modes(self, state: bool) -> None:
        self.button_pick_bbox.setEnabled(state)
        self.button_span_bbox.setEnabled(state)

    def line_edited_activated(self) -> bool:
        for line_edit in self.all_line_edits:
            if line_edit.hasFocus():
                return True
        return False

    def change_pointcloud_folder(self) -> None:
        path_to_folder = Path(
            QFileDialog.getExistingDirectory(
                self,
                tr("选择点云文件夹"),
                directory=config.get("FILE", "pointcloud_folder"),
            )
        )
        if not path_to_folder.is_dir():
            logging.warning("Please specify a valid folder path.")
        else:
            if not self.controller.save():
                return
            self.controller.pcd_manager.pcd_folder = path_to_folder
            self.controller.pcd_manager.read_pointcloud_folder()
            self.controller.roof_plane_controller.on_pointcloud_changed()
            self.controller.next_pcd(save=False)
            logging.info("Changed point cloud folder to %s!" % path_to_folder)

    def change_label_folder(self) -> None:
        path_to_folder = Path(
            QFileDialog.getExistingDirectory(
                self,
                tr("选择标注文件夹"),
                directory=config.get("FILE", "label_folder"),
            )
        )
        if path_to_folder:
            path_to_folder = Path(path_to_folder)
            config["FILE"]["label_folder"] = str(path_to_folder)
            config_manager.write_into_file()
            logging.info(f"Label folder changed to: {path_to_folder}")

        # if not path_to_folder.is_dir():
        #     logging.warning("Please specify a valid folder path.")
        # else:
        #     self.controller.pcd_manager.label_manager.label_folder = path_to_folder
        #     self.controller.pcd_manager.label_manager.label_strategy.update_label_folder(
        #         path_to_folder
        #     )
        #     logging.info("Changed label folder to %s!" % path_to_folder)

    # def update_default_object_class_menu(
    #     self, new_classes: Optional[Set[str]] = None
    # ) -> None:
    #     object_classes = set(LabelConfig().get_classes())

    #     object_classes.update(new_classes or [])
    #     existing_classes = {
    #         action.text() for action in self.actiongroup_default_class.actions()
    #     }
    #     for object_class in object_classes.difference(existing_classes):
    #         action = self.actiongroup_default_class.addAction(
    #             object_class
    #         )  # TODO: Add limiter for number of classes
    #         action.setCheckable(True)
    #         if object_class == LabelConfig().get_default_class_name():
    #             action.setChecked(True)

    #     self.act_set_default_class.addActions(self.actiongroup_default_class.actions())

    # def change_default_object_class(self, action: QAction) -> None:
    #     LabelConfig().set_default_class(action.text())
    #     logging.info("Changed default object class to %s.", action.text())

    def ask_custom_index(self):
        input_d = QInputDialog(self)
        self.input_pcd = input_d
        input_d.setInputMode(QInputDialog.IntInput)
        input_d.setWindowTitle(tr("跳转到点云"))
        input_d.setLabelText(tr("输入点云序号："))
        input_d.setIntMaximum(len(self.controller.pcd_manager.pcds) - 1)
        input_d.intValueChanged.connect(lambda val: self.update_dialog_pcd(val))
        input_d.intValueSelected.connect(lambda val: self.controller.custom_pcd(val))
        input_d.open()
        self.update_dialog_pcd(0)

    def update_dialog_pcd(self, value: int) -> None:
        pcd_path = self.controller.pcd_manager.pcds[value]
        self.input_pcd.setLabelText(
            tr("输入点云序号（{name}）：", name=pcd_path.name)
        )

    # def change_label_color(self):
    #     bbox = self.controller.bbox_controller.get_active_bbox()
    #     LabelConfig().set_class_color(
    #         bbox.classname, Color3f.from_qcolor(QColorDialog.getColor())
    #     )

    @staticmethod
    def save_point_cloud_as(pointcloud: PointCloud) -> None:
        extensions = BasePointCloudHandler.get_supported_extensions()
        make_filter = " ".join(["*" + extension for extension in extensions])
        file_filter = tr("点云文件（{formats}）", formats=make_filter)
        file_name, _ = QFileDialog.getSaveFileName(
            caption=tr("选择点云保存位置"),
            directory=str(pointcloud.path.parent),
            filter=file_filter,
            initialFilter=file_filter,
        )
        if file_name == "":
            logging.warning("No file path provided. Ignored.")
            return

        try:
            path = Path(file_name)
            handler = BasePointCloudHandler.get_handler(path.suffix)
            handler.write_point_cloud(path, pointcloud)
        except Exception as e:
            msg = QMessageBox()
            msg.setWindowTitle(tr("点云保存失败"))
            msg.setText(e.__class__.__name__)
            msg.setInformativeText(traceback.format_exc())
            msg.setIcon(QMessageBox.Critical)
            msg.setStandardButtons(QMessageBox.Close)
            msg.button(QMessageBox.Close).setText(tr("关闭"))
            msg.exec_()
    
    def save_roof_annotations(self):
        """将屋顶标注以原始点云的世界坐标导出为 OBJ。"""
        if self.controller.roof_plane_controller.active:
            self.controller.save(force_plane=True)
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, tr("保存屋顶标注"), "", tr("OBJ 文件 (*.obj)")
        )
        if filepath:
            try:
                self.controller.roof_drawing_manager.save_to_obj(
                    filepath, self.controller.pcd_manager.pointcloud
                )
            except Exception:
                logging.exception("Failed to export roof OBJ to %s", filepath)
                QMessageBox.critical(
                    self,
                    tr("导出失败"),
                    tr(
                        "无法导出 OBJ：\n{path}\n\n请检查目录权限和磁盘空间。",
                        path=filepath,
                    ),
                )
    
    def toggle_roof_mode(self, mode: str):
        current = self.controller.roof_drawing_mode
        if current == mode:  # 点击已激活的模式 → 取消
            self.controller.activate_roof_drawing_mode(None)  # 或 "" 表
        else:
            self.controller.activate_roof_drawing_mode(mode)
        #self.controller.activate_roof_drawing_mode(mode)

        active = self.controller.roof_drawing_mode
        self.button_add_vertices.setChecked(active == "point")
        self.button_add_lines.setChecked(active == "line")

        self.button_add_vertices.setStyleSheet(
            "background-color: lightgreen;" if active == "point" else ""
        )
        self.button_add_lines.setStyleSheet(
            "background-color: lightgreen;" if active == "line" else ""
        )
        if active == "point":
            self.status_manager.update_status(
                "点模式：点击拾取屋顶关键点，黄色高亮显示",
                mode_text="点模式"
            )
        elif active == "line":
            self.status_manager.update_status(
                "线模式：点击连接线段，最后一点靠近起点时自动闭合",
                mode_text="线模式"
            )
        else:  # 导航模式（取消屋顶模式）
            self.status_manager.update_status(
                "导航模式：左键旋转视角，右键平移，滚轮缩放",
                mode_text="导航模式"
            )
    def toggle_connect_mode(self):
        checked = self.connect_two_vertices.isChecked()
        self.controller.activate_connect_mode(checked)

        self.connect_two_vertices.setStyleSheet(
            "background-color: lightblue;" if checked else ""
        )

        if checked:
            self.status_manager.update_status(
                "连接模式：依次点击两个已有顶点进行连接（第二次点击后自动完成一条边）",
                mode_text="连接两点"
            )
        else:
            self.status_manager.update_status(
                "导航模式：左键旋转视角，右键平移，滚轮缩放",
                mode_text="导航模式"
            )
