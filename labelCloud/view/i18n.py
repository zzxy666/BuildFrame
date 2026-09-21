"""Small runtime translation layer for the Chinese and English interfaces."""

from PyQt5 import QtCore


LANGUAGE_CHINESE = "zh_CN"
LANGUAGE_ENGLISH = "en_US"
SUPPORTED_LANGUAGES = (LANGUAGE_CHINESE, LANGUAGE_ENGLISH)


ENGLISH_TRANSLATIONS = {
    # Main window
    "BuildFrame 点云标注": "BuildFrame Point Cloud Annotation",
    "点云文件": "Point Clouds",
    "保存当前标注并加载上一个点云。": "Save the current annotations and load the previous point cloud.",
    "上一个": "Prev",
    "跳转": "Go",
    "保存当前标注并加载下一个点云。": "Save the current annotations and load the next point cloud.",
    "下一个": "Next",
    "点位微调": "Vertex Adjustment",
    "视角控制": "View Controls",
    "上视图": "Top",
    "下视图": "Bottom",
    "前视图": "Front",
    "后视图": "Back",
    "左视图": "Left",
    "右视图": "Right",
    "添加顶点": "Add Vertices",
    "连接两点": "Connect Vertices",
    "绘制边线": "Draw Edges",
    "点云滤波": "Filter Point Cloud",
    "正在滤波…": "Filtering…",
    "恢复原始颜色": "Restore Colors",
    "保存标注": "Save Annotations",
    "标注列表": "Annotations",
    "选中线坐标": "Selected Edge Coordinates",
    "选中点坐标": "Selected Vertex Coordinates",
    "起点": "Start",
    "终点": "End",
    "起点 X": "Start X",
    "起点 Y": "Start Y",
    "起点 Z": "Start Z",
    "终点 X": "End X",
    "终点 Y": "End Y",
    "终点 Z": "End Z",
    "取消当前选择。": "Clear the current selection.",
    "取消选择": "Clear Selection",
    "删除当前选中的点或线。[Del]": "Delete the selected vertex or edge. [Del]",
    "删除": "Delete",
    "文件": "File",
    "标注": "Annotations",
    "设置": "Settings",
    "设置点云文件夹…": "Set Point Cloud Folder…",
    "设置标注文件夹…": "Set Annotation Folder…",
    "加载单个点云…": "Load Point Cloud…",
    "仅绕 Z 轴旋转": "Z-axis Rotation Only",
    "仅允许标注对象绕 Z 轴旋转。": "Only rotate annotations around the Z axis.",
    "按标注着色": "Color by Annotation",
    "删除当前全部标注": "Delete All Annotations",
    "设置默认边界框尺寸…": "Set Default Bounding Box Dimensions…",
    "设置默认对象类别…": "Set Default Object Class…",
    "设置默认变换步长…": "Set Default Transformation Steps…",
    "点大小": "Point Size",
    "显示地面网格": "Show Ground Grid",
    "在 X-Y 平面（Z=0）显示网格。": "Show a grid on the X-Y plane (Z=0).",
    "显示方向指示": "Show Orientation",
    "保留视角": "Keep View",
    "保存当前视角，并在下次打开该点云时恢复。": "Save and restore the current view for this point cloud.",
    "对齐点云": "Align Point Cloud",
    "变换点云，使地面与 X-Y 平面对齐。": "Transform the point cloud so its ground aligns with the X-Y plane.",
    "修改设置…": "Preferences…",
    "测试": "Test",
    "延续标注": "Propagate Annotations",
    "当下一个点云尚无标注时，将当前标注延续过去。": "Propagate annotations when the next point cloud has none.",
    # Settings dialog
    "包含点云文件的文件夹路径。": "Folder containing the point cloud files.",
    "点云文件夹": "Point Cloud Folder",
    "点云视图的背景颜色（R,G,B）。": "Point-cloud viewer background color (R,G,B).",
    "背景颜色": "Background Color",
    "鼠标滚轮缩放时使用的标准步长。": "Standard mouse-wheel zoom step.",
    "缩放步长": "Zoom Step",
    "界面设置": "Interface",
    "点云中每个点在屏幕上的显示直径。": "Rendered diameter of each point.",
    "无颜色点云使用的点颜色（R,G,B）。": "Point color for point clouds without colors (R,G,B).",
    "根据高度为无颜色点云着色。": "Color point clouds without colors by height.",
    "按高度为无颜色点云着色": "Color Colorless Clouds by Height",
    "恢复默认设置": "Defaults",
    "保存标注文件的文件夹路径。": "Folder where annotation files are saved.",
    "文件设置": "Files",
    "标注文件夹": "Annotation Folder",
    "以网格形式显示点云的 X-Y 地面平面。": "Display the point-cloud X-Y ground plane as a grid.",
    "显示点云地面网格（X-Y 平面）": "Show Ground Grid (X-Y Plane)",
    "提示：部分设置需要重启程序后才能生效。": "Note: Some settings require an application restart.",
    "点颜色\n（用于无颜色点云）": "Point Color\n(for colorless point clouds)",
    "点云设置": "Point Cloud",
    "平移点云时使用的标准步长。": "Standard point-cloud translation step.",
    "平移步长": "Translation Step",
    "界面语言": "Interface Language",
    "选择界面、状态提示和弹窗使用的语言。": "Language used by the interface, status messages, and dialogs.",
    "中文": "Chinese",
    "英文": "English",
    "保存": "Save",
    "取消": "Cancel",
    "关闭": "Close",
    # Dynamic status and dialogs
    "导航模式": "Navigation",
    "点模式": "Vertex Mode",
    "线模式": "Edge Mode",
    "连接两点": "Connect Vertices",
    "点模式：点击拾取屋顶关键点，黄色高亮显示": "Vertex mode: click to pick roof vertices; picked vertices are highlighted in yellow.",
    "线模式：点击连接线段，最后一点靠近起点时自动闭合": "Edge mode: click to draw edges; click near the first vertex to close the polygon.",
    "导航模式：左键旋转视角，右键平移，滚轮缩放": "Navigation: left-drag to rotate, right-drag to pan, and use the wheel to zoom.",
    "连接模式：依次点击两个已有顶点进行连接（第二次点击后自动完成一条边）": "Connect mode: click two existing vertices to create an edge.",
    "旋转中心：{x:.3f}, {y:.3f}, {z:.3f}": "Rotation center: {x:.3f}, {y:.3f}, {z:.3f}",
    "局部旋转": "Local Orbit",
    "已在点云文件夹中找到 {count} 个点云文件。": "Found {count} point cloud file(s) in the configured folder.",
    "请将点云文件夹设置为包含有效点云文件的目录。": "Choose a folder containing supported point cloud files.",
    " —（请选择文件夹）": " — (select a folder)",
    "当前：<em>{name}</em>": "Current: <em>{name}</em>",
    "滤波失败": "Filtering Failed",
    "当前点云为空。": "The current point cloud is empty.",
    "地面滤波过程中发生错误，详情已写入日志。": "An error occurred during ground filtering. Details were written to the log.",
    "保存失败": "Save Failed",
    "无法保存屋顶标注：\n{path}\n\n请检查目录权限和磁盘空间。": "Could not save roof annotations:\n{path}\n\nCheck folder permissions and available disk space.",
    "加载标注失败": "Annotation Load Failed",
    "无法读取屋顶标注：\n{path}": "Could not load roof annotations:\n{path}",
    "未找到二维图像": "2D Image Not Found",
    "在图像文件夹（{folder}）中未找到对应图像。\n请检查文件夹路径以及是否存在与当前点云同名的图像。": "No matching image was found in {folder}.\nCheck the folder and whether an image with the point-cloud name exists.",
    "二维图像（{name}）": "2D Image ({name})",
    "<b>指定文件夹中没有找到有效的点云文件。</b>": "<b>No valid point cloud files were found in the selected folder.</b>",
    "请将点云文件放入 <code>{folder}</code>，或重新设置点云文件夹。当前支持以下点云格式：\n{formats}。": "Place point clouds in <code>{folder}</code>, or choose another folder. Supported formats:\n{formats}.",
    "未找到点云文件": "No Point Clouds Found",
    "选择点云文件夹": "Select Point Cloud Folder",
    "选择标注文件夹": "Select Annotation Folder",
    "跳转到点云": "Go to Point Cloud",
    "输入点云序号：": "Point cloud number:",
    "输入点云序号（{name}）：": "Point cloud number ({name}):",
    "点云文件（{formats}）": "Point Cloud Files ({formats})",
    "选择点云保存位置": "Save Point Cloud",
    "点云保存失败": "Point Cloud Save Failed",
    "保存屋顶标注": "Save Roof Annotations",
    "OBJ 文件 (*.obj)": "OBJ Files (*.obj)",
    "导出失败": "Export Failed",
    "无法导出 OBJ：\n{path}\n\n请检查目录权限和磁盘空间。": "Could not export OBJ:\n{path}\n\nCheck folder permissions and available disk space.",
    "选择坐标后按 Ctrl+C 可复制": "Select a coordinate and press Ctrl+C to copy.",
    "{axis} = {value:.12f}\n点击字段后按 Ctrl+A、Ctrl+C 可复制完整显示值": "{axis} = {value:.12f}\nClick the field, then press Ctrl+A and Ctrl+C to copy the displayed value.",
}

ENGLISH_TRANSLATIONS.update({
    # Ground/non-ground filtering options
    "取消滤波": "Cancel Filtering",
    "地面分离完成：地面 {ground} 点，非地面 {other} 点（绿色 / 红色）。": "Ground separation complete: {ground} ground points, {other} non-ground points (green / red).",
    "已取消地面滤波。": "Ground filtering cancelled.",
    "地面滤波失败：{error}": "Ground filtering failed: {error}",
    "开始 CSF 地面分离，可再次点击按钮取消。": "CSF ground separation started. Click the button again to cancel.",
    "地面滤波设置…": "Ground Filter Settings…",
    "地面滤波设置": "Ground Filter Settings",
    "使用 CSF 布料模拟滤波分离地面和非地面点。保存后，下次执行地面滤波时使用新参数。": "Separate ground and non-ground points using CSF (Cloth Simulation Filter). Saved settings apply to the next ground-filter run.",
    "单位换算系数（原始单位 / 米）": "Source Units per Metre",
    "1 米对应多少个原始坐标单位：米填 1，厘米填 100，毫米填 1000。若采用未知的自定义比例，请先核实实际尺寸，程序不会自动猜测。": "Number of source coordinate units in one metre: enter 1 for metres, 100 for centimetres, or 1000 for millimetres. For an unknown custom scale, confirm the actual dimensions first; the application will not guess.",
    "布料分辨率（米）": "Cloth Resolution (metres)",
    "以米输入布料网格间距。减小可保留更多地形细节，但会增加内存和计算时间；大范围点云宜适当增大。": "Enter cloth grid spacing in metres. Smaller values retain more terrain detail but use more memory and time. Consider larger values for wide-area clouds.",
    "分类距离阈值（米）": "Classification Threshold (metres)",
    "以米输入点到拟合地面的距离阈值。增大会将更多点归为地面，也可能误分低矮物体。": "Enter the maximum point-to-fitted-ground distance in metres. Larger values classify more points as ground but may include low objects.",
    "地形预设": "Terrain Preset",
    "城市 / 平坦地形": "Urban / Flat Terrain",
    "缓坡地形": "Gentle Slopes",
    "陡坡 / 起伏地形": "Steep / Rugged Terrain",
    "仅调整布料硬度：平坦为 3、缓坡为 2、陡坡为 1。仅切换地形不够，还需结合密度和分类预览调整分辨率、距离阈值，必要时调整时间步长。": "Changes cloth rigidity only: 3 for flat terrain, 2 for gentle slopes, 1 for steep slopes. A preset alone is insufficient: adjust resolution and threshold based on point density and the classification preview, and time step when needed.",
    "启用坡面后处理": "Enable Slope Post-processing",
    "补充识别陡坡附近的地面点；建议结合分类预览判断是否启用。": "Recover additional ground points near steep slopes. Check the classification preview to decide whether to enable this.",
    "最大迭代数": "Maximum Iterations",
    "布料模拟的最大迭代次数；通常从 500 开始。": "Maximum cloth-simulation iterations; 500 is a typical starting point.",
    "模拟时间步长": "Simulation Time Step",
    "布料模拟的时间步长，默认 0.65。调整后需重新检查分类结果，并结合分辨率和迭代次数判断。": "Cloth simulation time step; default 0.65. Recheck classification after changes, considering resolution and iteration count together.",
    "先确认坐标单位：Z 为高程，XY 对应地面平面，三个轴使用一致的长度单位。单位换算系数：米填 1，厘米填 100，毫米填 1000；未知比例须先核实实际尺寸，程序不会自动猜测。\n\n分辨率和阈值始终以米输入，默认 1.0 / 0.5 只是调参起点。仅切换地形不够，需结合密度和预览调整分辨率、阈值，陡坡可能还需调整时间步长。缺少地面、室内、多层结构等场景不能保证正确分离，请检查分类预览。": "Confirm coordinate units first: Z is elevation, XY is the ground plane, and all axes must use the same length unit. Source units per metre: enter 1 for metres, 100 for centimetres, or 1000 for millimetres. For an unknown scale, confirm actual dimensions first; the application will not guess.\n\nResolution and threshold are always entered in metres. Defaults 1.0 / 0.5 are only a starting point. A terrain preset alone is insufficient: adjust resolution and threshold using point density and the preview; steep slopes may also need a different time step. Correct separation is not guaranteed for missing ground, indoor scenes, or multi-level structures. Check the classification preview.",
    "当前滤波配置无效，已在表单中显示默认值；保存前不会修改配置。": "The current filter configuration is invalid. The form shows defaults; the configuration is unchanged until you save.",
    "滤波参数无效": "Invalid Filter Parameters",
    "请检查滤波参数：\n{error}": "Check the filter parameters:\n{error}",
    "无法保存地面滤波设置，请检查配置文件的写入权限和磁盘空间。": "Could not save ground-filter settings. Check configuration-file write permissions and available disk space.",
})

CHINESE_QT_TRANSLATIONS = {
    "OK": "确定",
    "Cancel": "取消",
    "Save": "保存",
    "Close": "关闭",
    "Open": "打开",
    "Yes": "是",
    "No": "否",
    "Apply": "应用",
    "Reset": "重置",
    "&OK": "确定",
    "&Cancel": "取消",
    "&Save": "保存",
    "&Close": "关闭",
}


def normalize_language(language: str) -> str:
    return language if language in SUPPORTED_LANGUAGES else LANGUAGE_CHINESE


class _DictionaryTranslator(QtCore.QTranslator):
    def __init__(self, translations, parent=None) -> None:
        super().__init__(parent)
        self._translations = translations

    def isEmpty(self) -> bool:  # noqa: N802 - Qt API spelling
        return False

    def translate(self, context, source_text, disambiguation=None, n=-1):
        return self._translations.get(source_text, source_text)


class LanguageManager(QtCore.QObject):
    language_changed = QtCore.pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.language = LANGUAGE_CHINESE
        self._translators = {
            LANGUAGE_CHINESE: _DictionaryTranslator(
                CHINESE_QT_TRANSLATIONS, self
            ),
            LANGUAGE_ENGLISH: _DictionaryTranslator(
                ENGLISH_TRANSLATIONS, self
            ),
        }
        self._active_translator = None
        self._installed = False

    def set_language(self, language: str) -> None:
        language = normalize_language(language)
        app = QtCore.QCoreApplication.instance()
        if language == self.language and self._installed:
            return
        if app is not None and self._installed and self._active_translator:
            app.removeTranslator(self._active_translator)
            self._installed = False
        self.language = language
        self._active_translator = self._translators[language]
        if app is not None:
            self._installed = app.installTranslator(self._active_translator)
        self.language_changed.emit(language)


language_manager = LanguageManager()


def tr(source_text: str, **values) -> str:
    translated = QtCore.QCoreApplication.translate("BuildFrame", source_text)
    return translated.format(**values) if values else translated
