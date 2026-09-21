# BuildFrame

BuildFrame 是基于 labelCloud 二次开发的城市点云屋顶结构标注工具。它使用
PyQt5 和 OpenGL 显示点云，支持屋顶顶点、边、多边形标注以及 LAS/LAZ、PLY、
PCD 等常见点云格式。

## 功能

- 点模式和线模式屋顶标注，支持吸附、预览及闭合
- 已有顶点连接、删除、选中和六方向微调
- 标准顶视、底视、前后左右视图
- 左键开始旋转时自动以鼠标下点为中心；双击可显式设置，`Home`/`P` 恢复整体中心
- 滚轮缩放自动朝向鼠标下的可见点，并动态调整裁剪面，避免近距离点云消失
- CSF 地面/非地面分类预览，独立进程计算，支持取消及中英文参数设置
- 自动保存版本化 `.roof.json` 项目标注
- 以原始点云世界坐标导出 OBJ 顶点和线
- 兼容旧版本生成的局部坐标 `.roof.obj` 自动标注

## Conda 安装（推荐）

```powershell
git clone https://github.com/zzxy666/BuildFrame.git
cd BuildFrame
conda env create -f environment.yml
conda activate buildframe
python -m labelCloud
```

Windows 下 Conda 提供的 `freeglut.dll` 名称与部分 PyOpenGL 版本的默认查找名称
不同。BuildFrame 会自动加载 Conda 环境中的 DLL；如果 GLUT 仍不可用，点云照常
显示，仅关闭 3D 顶点/边文字。

也可以在已有 Python 3.9–3.11 环境中安装：

```powershell
python -m pip install -e .
python -m labelCloud
```

## 配置和数据

程序从当前工作目录读取 `config.ini`。默认点云目录为：

```ini
[FILE]
pointcloud_folder = pointclouds/
label_folder = pointclouds/

[USER_INTERFACE]
language = zh_CN
auto_pick_rotation_center = True
zoom_to_cursor = True
```

界面语言也可以在“设置 → 修改设置”中切换，支持中文和 English，保存后立即生效。

把 `.las`、`.laz`、`.ply`、`.pcd` 等文件放到 `pointclouds/`，程序会递归发现。
自动标注保存在对应点云旁边，例如：

```text
building.las
building.roof.json
```

内部 JSON 使用世界坐标并带有格式版本。工具栏“保存”导出的 OBJ 同样使用世界
坐标，可与原始点云对齐。

## 地面 / 非地面分离（CSF）

主界面的“点云滤波”使用 [jianboqi/CSF](https://github.com/jianboqi/CSF)
官方布料模拟算法，通过 `cloth-simulation-filter==1.1.7` 集成。
该算法面向机载点云地面提取，也被 CloudCompare 集成。
旧 `PointCloudFilter/ground_filter.py` 保留作为历史实现，主界面不再调用它。

选型考虑：CSF 符合当前城市屋顶/地形点云的工作流，并提供 Windows Python 二进制包。
[PDAL SMRF](https://pdal.io/en/stable/stages/filters.smrf.html) 是值得对照的形态学地面滤波方案；
[Patchwork++](https://github.com/url-kaist/patchwork-plusplus) 更面向车载/机器人 LiDAR 扫描。
这里的选择是部署和场景匹配，不是所有数据集上的精度排名。

已有环境补装（新建 Conda 环境或 `pip install -e .` 会自动安装）：

```powershell
conda activate buildframe
python -m pip install cloth-simulation-filter==1.1.7
python begin.py
```

1. 打开点云，进入“设置 → 地面滤波设置…”。
2. 确认“单位换算系数”：**1 米等于多少原始坐标单位**。米填 `1`，厘米填 `100`，毫米填 `1000`；自定义比例必须按实际尺寸确认，程序不自动猜测。Z 必须是高程，XY 是地面平面，三个轴单位一致；经纬度坐标应先投影。
3. 分辨率和分类阈值始终以**米**填写。城市平地可从分辨率 `1.0`、阈值 `0.5`、城市/平坦地形、迭代 `500`、时间步长 `0.65` 开始。
4. 保存后点击“点云滤波”。**绿色为地面，红色为非地面**，状态栏显示点数。计算期间再次点击可取消；完成后再次点击恢复原色。修改参数后需恢复原色并重新滤波。

地形预设仅调整布料硬度（平坦 3、缓坡 2、陡坡 1）。更细的分辨率保留更多地形细节，
但消耗更多内存；增大分类阈值会接纳更多地面点，也可能误接纳低矮物体。
陡坡不能只切换预设，需联合检查分辨率、阈值、时间步长与坡面后处理。
本项目的 45°合成坡面测试中，默认参数识别很差，改为硬度 1、分辨率 0.5 米、
时间步长 1.0 后明显改善；这不是对真实点云精度的保证。

```ini
[GROUND_FILTER]
units_per_meter = 1.0
resolution = 1.0
threshold = 0.5
rigidness = 3
slope_smooth = True
iterations = 500
time_step = 0.65
```

运行时撤销显示用缩放，再按单位系数转换到米，不改原文件坐标或屋顶标注。
当前功能是分类预览和内存中的 `pointcloud.ground_labels`（2=地面，1=非地面），
**不会自动导出两份点云文件**，也不会删除原始点。切换点云或退出时取消旧任务。
计算前检查无效坐标及布料网格大小；超过 100 万网格节点会拒绝运行，提示检查单位、
增大分辨率或裁剪范围，不会偷偷降低精度。大量点的快照写盘仍可能短暂占用界面线程。

缺少真实地面、室内、多层结构、严重噪声和特殊地形都可能误分；请检查预览，
不要将分类结果当作人工确认的真值。不同文件单位可能不同，换文件后应检查设置。

## 开发和测试

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m compileall -q labelCloud PointCloudFilter
```

建议在功能分支开发：

```powershell
git switch -c fix/my-change
git add .
git commit -m "描述修改"
git push -u origin fix/my-change
```

## 许可证与来源

本项目基于 Christoph Sager 的
[labelCloud](https://github.com/ch-sa/labelCloud)，按 GNU GPL v3 or later 发布。
BuildFrame 开发者：Zhikang Yin、Shaobo Xia。

CSF 为外部依赖，按上游 Apache-2.0 许可证发布，源码和许可证见
[CSF 官方仓库](https://github.com/jianboqi/CSF)。

## 大场景 Roof Plane 标注

沿用 `python buildframe.py` 入口和现有 **PyQt5 + PyOpenGL** 框架。
Wireframe 保留原来的顶点/边线编辑和 `.roof.json` / OBJ 保存流程。
Roof Plane 可直接打开完整 LAS/LAZ；每个 `plane_id` 代表独立的屋顶平面基元，
不是建筑实例 ID。`0` 显示为 **Unassigned / Background**，不表示该区域已经检查过。

### 常用操作

- **N**：新建 Plane。ID 为会话历史最大值加一；删除或撤销不会回收号码。
  内部及导出均为 `uint32`，支持 `0–4294967295`，0 保留。
- **A / 赋给当前 Plane**：将选区赋给当前 ID。数字 0–9 不再直接赋值；
  清为 0 使用“设为未分配 / Background”按钮。
- **V / B / L**：浏览、矩形、套索；中键拖动临时旋转；右键平移；滚轮缩放。
  **Shift** 追加选区、**Alt** 减选、**Esc** 清除选择或取消候选/后台计算。
  选择穿透当前视图，包含遮挡点；青色为选区/当前 Plane，黄色为扩展候选。
- 搜索框按 ID 过滤轻量 Plane 表；输入完整 ID 后按回车或“跳转 ID”选择目标。
  “上一/下一 Plane”在现有 ID 中切换。
- 默认“仅未标注点”，保护已有标签。修改已有 Plane 前主动切换到“当前 Plane”
  或“所有可见点”；合并/删除也需要解除默认保护，且只作用于未隐藏的工作区内点。
- **H** 隐藏选区；**Shift+H** 恢复全部；“只显示选中点”隐藏其余点。
  “仅显示隐藏点”为只读检查视图。隐藏不会修改标签或原始点。
- **Ctrl+Z** 按顺序撤销标签、扩展确认、隐藏/恢复。撤销采用差分记录，默认内存上限 128 MiB；
  超过上限时丢弃最旧记录。大到单步超过上限的操作也可能无法保留撤销记录。

### 局部平面扩展

快捷键 **E** 启动局部扩展，**Enter** 确认、**Esc** 取消。
默认开启“边缘补选”：严格生长完成后，仅补一圈贴合种子平面的邻近漏点，
默认邻距 0.25 m、面距 0.08 m，且至少有 3 个严格区域点支撑；补入的点不继续向外生长。
这能缓解屋脊/檐口混合法向引起的漏选，但仍需检查黄色预览，必要时关闭该选项或缩小阈值。
隐藏、已标注点保护和工作区限制仍有效。

Roof Plane 模式的六个视角按钮可正常使用，切换方向保留当前位置和缩放。
**Alt+1 / 2 / 3 / 4 / 5 / 6** 分别切换上、前、左、下、后、右视图。
后台计算时，**V + 左键拖动**或**中键拖动**仍可旋转视角。

圈选同一平面内部至少 6 个相邻、具有面积的种子，点击“局部扩展”。
默认参数为 `normal_k=20`、邻接半径 `0.6 m`、平面距离 `0.15 m`、法向夹角 `10°`。
原 LAS 是英尺时选择自动识别 CRS 或明确指定 US survey foot；参数始终按米计算。
无垂直 CRS 时 Z 暂按 XY 单位解释。

当前场景的 KDTree 在首次扩展时后台建立，此后重复使用；隐藏/标签/单位参数变化不重建树。
仅沿连续 frontier 计算需要的法向，不进行全场相似法向聚类。法向缓存按单位、邻点参数、
隐藏版本和 ROI 版本区分；采用局部内存缓存，不强制生成全局 `normals.npy`。
所有新增候选均满足 `plane_id=0`、未隐藏、位于 Work ROI（若启用）。
已有当前 Plane 的点可在“当前 Plane”选择范围下手选为种子，但其他已标注点不作为桥梁。
**Enter** 确认，**Esc** 取消。远处不连通的共面区域不会自动并入；几何上连续的共面结构仍需人工检查。

### 工作区、检查进度和书签

展开“工作区 / 检查进度 / 书签”：

1. 框选局部点后用“选区设为工作区”，或用“视域设为工作区”。形成 local-render XY 包围盒，
   保留全场高度；只限制交互，不裁剪/删除原始点。可清除工作区或切换边界显示。
2. 设置网格尺寸（默认 50 米），建立检查网格。网格只包含有点的 XY 单元。
   “下一未检查区”设置一个网格工作区并移动视角；“工作区已检查”只标记被工作区完整覆盖的格子。
   这与 `plane_id=0` 完全独立；界面显示已检查/总区域数。
3. 书签保存相机位置/焦点/up、现有 OpenGL 的旋转/平移/旋转中心、当前 Plane 和可选 ROI。
   可以添加、前后跳转及删除；关闭后可恢复。当前视角也会随工作保存写入 session。

### 工作保存与最终导出

**Ctrl+S / 保存工作标注**不再重写整个 LAS/LAZ。停止编辑约 2 秒后自动保存；
切换文件或关闭时也保存工作标注，失败则阻止切换。后台计算期间请等待完成或按 Esc 取消后再切换。

```text
465902CA.laz                 原始文件，只读
465902CA.planar/
    plane_id.npy            uint32，长度与原始点数一致
    session.json            源文件指纹、最大 ID、相机、ROI、坐标约定
    reviewed_grid.json      检查进度
    bookmarks.json          书签
    pending.npz             仅保存/异常恢复期间存在的重做日志
```

首次保存初始化 `plane_id.npy`；之后只将变化索引写入 memmap，并用日志支持中断恢复。
启动时验证源文件大小、mtime、点数、头信息及首尾摘要；不匹配时停止加载，避免标签错位。
原有匹配的 `*_gt.las` 会在没有侧车时导入；此后以侧车为准。
兼容输入的 `plane_id` / `planeid` / `PlaneID` / `PlaneId`，但新标签统一使用 `plane_id`。

只有点击 **“导出最终 LAS / LAZ”** 才生成完整点云：先恢复临时隐藏，后台按原始顺序分块复制
全部 XYZ、RGB、Intensity、Classification、Returns、GPS Time、其他 Extra Bytes 及 VLR/EVLR，
新增/更新 `uint32 plane_id`。隐藏状态、ROI 和检查进度不写入最终点云，不会丢弃任何点。
禁止覆盖原始文件。LAZ 导出需要已安装的压缩后端；较新压缩版本优先使用 Laszip。

### 性能与坐标

套索移动时仅记录间隔至少 2 px 的二维顶点和绘制轮廓；松开后后台选择。
屏幕坐标以 25 万点分块、NumPy 矩阵计算；相机矩阵、投影、视窗尺寸或点云变化才失效。
相机停稳后后台预热缓存；套索先做 bbox / hidden / plane / ROI 过滤，再只对候选做 polygon test。
主点云 VBO 长期保留，选区仅上传原始点 ID 高亮，标签颜色仅在标签变化或显示模式切换时更新。

渲染使用居中的 local coordinates；原始坐标通过 `render_origin` / `render_scale` 对应。
法向和距离按米解释，最终导出直接复制原始整数 XYZ，不反算渲染坐标。
可选“旋转期间 1/8 LOD”只影响显示，停止后恢复全量；选择、扩展和导出仍使用全部原始点。
可勾选“性能日志”记录 bbox、polygon、overlay 和局部扩展各阶段耗时。

详细代码审查与测试结果见 [大场景改造报告](docs/LARGE_SCENE_REFACTOR.md)。
