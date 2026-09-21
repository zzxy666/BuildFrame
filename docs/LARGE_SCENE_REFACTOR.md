# BuildFrame 大场景 Roof Plane 改造报告

## 框架与原始问题

当前项目实际为 PyQt5 + QGLWidget/PyOpenGL，没有 PyVista PolyData、VTK actor 或 `plotter.clear()`。
因此按现有框架实现等价机制，没有迁移 GUI/3D 框架，没有替换 Wireframe 标注流程。

原来的 `plane_id` 已是 uint32，0–9 只是界面快捷赋值设计，并非存储限制。
鼠标移动原本也仅记录套索顶点；真正的问题是松开后重复全场投影、同步选择，继而每次刷新全场
颜色、全场 `np.unique` 统计和重建 QListWidget。原有全量刷新计时器还会在空闲时反复绘制大场景。
局部扩展每次创建全场 KDTree，计算所有接近种子数学平面的候选法向；保存每次复制并写完整 LAS。
标签撤销已有差分，但 dirty 每次比较全量数组，隐藏操作和筛选也有不必要的全量复制。
点云初始化/切换文件在主线程执行，并有在非当前 OpenGL 上下文创建缓冲的风险。

## 核心实现对应

| 用户问题 | 当前实现 |
|---|---|
| dtype / 最大值 | `np.uint32`，0–4,294,967,295；0 保留，越界明确报错。 |
| 新 Plane ID | `max_plane_id + 1`；删除和撤销都不回收号码，最大值存入 session。 |
| 大 ID 管理 | `PlaneTableModel` + QTableView；搜索、跳转、前后切换；名称不补零；A 赋当前 ID，取消数字赋值快捷键。 |
| 点数统计 | 加载时一次 unique 初始化 `plane_counts`；编辑仅统计变化点旧/新 ID 并更新字典。 |
| Lasso 执行 | 鼠标移动仅记录二维轮廓；释放后 QRunnable 运行投影缓存检查 → 分块 bbox/保护/ROI 过滤 → 候选 polygon test → 返回 original IDs → 主线程高亮。 |
| 是否仍全场遍历 | 首次/失效投影和 bbox 粗筛需分块扫描全场；polygon test 只处理候选，鼠标移动不扫描点。选区布尔状态查询仍有轻量线性扫描，不宣称完全 O(选中点数)。 |
| 屏幕缓存 | `ScreenProjectionCache` 保存 float32 xy/depth、valid；相机 modelview/projection、尺寸、点云身份变化失效，含 clipping 改变；250 ms 静止后预热。 |
| Base Actor | 在本框架中为持久位置/颜色 VBO，不因选择重建；overlay 用 original ID 的 glDrawElements，恒定高亮色。 |
| 原始 RGB | 保留原始颜色；标注着色与原始数据分离，选择不生成 N×3 RGB；标签改变按涉及的颜色块更新 GPU。 |
| KDTree | 每个场景的 `SceneSpatialIndex` 首次扩展时后台建立一次，后续复用；单位变化用查询半径和精确米距离调整，不重建树。小种子树仅检查种子连通性。 |
| 法向 | 仅计算 frontier 邻近候选；缓存按单位/normal_k/半径/隐藏版本/ROI 版本区分，非全场法向。未生成磁盘 normals.npy。 |
| 局部约束 | PCA 种子面、米距离、法向角、精确半径连接、hidden/ROI/标签保护同时生效；已有标签不重新赋值，也不自动作桥梁。 |
| Undo | changed IDs + old values + 少量 Plane 元信息；隐藏保存 changed IDs + old hidden states + 原选区 IDs；总计 128 MiB 上限。最大历史 ID 不撤销。 |
| 工作保存 | `<stem>.planar/plane_id.npy` + session/reviewed_grid/bookmarks JSON；首次初始化，后续差分 memmap 写入；2 秒 debounce，重做日志支持中断恢复。 |
| 最终导出 | 显式按钮选择 LAS/LAZ；后台按 25 万点块复制原始点记录及元信息，更新 uint32 plane_id；与工作保存分开。 |
| hidden_mask | 仅内存会话状态，不进入最终 LAS；隐藏点仍保留原始索引和标签，导出前恢复显示。 |
| original point ID | 渲染数组不因 ROI/隐藏/选择重排；缓存、overlay、frontier 和标签始终使用原始 ID；导出按源文件顺序逐块写入。 |
| Work ROI | 从视域或选区建立 local-render XY bbox，保留全场高度；限制选择/扩展/编辑，不限制最终输出。边界可开关。 |
| reviewed grid | 以米设定网格尺寸，只纳入有点的格子；仅完整位于 ROI 内的格子标记 reviewed；跳下一未检查格子时移动相机。 |
| Bookmark | 保存位置、焦点、up 和等价的 OpenGL 旋转/平移/pivot，以及当前 Plane、ROI；写入 sidecar。 |
| LOD | 可选旋转中 1/8 显示抽样，停止恢复全部；计算和输出不抽样。 |
| Wireframe | 保留原来的模型/编辑/保存；切回时清除 Roof Plane 专用颜色、overlay、隐藏绘制索引；共享加载改为后台、缓冲创建移到当前绘制上下文。 |

## 文件与新增类

- `model/scene_planes.py`：大场景 `RoofPlanes`，旧 `model/roof_planes.py` 保留投影/多边形工具及兼容导出。
- `model/scene_session.py`：`SceneSession`，侧车校验、差分持久化、分块最终导出。
- `model/screen_projection.py`：`ScreenProjectionCache`。
- `model/scene_spatial.py`：`SceneSpatialIndex`。
- `model/scene_workspace.py`：`SceneWorkspace`。
- `control/scene_worker.py`：`SceneTask`、`TaskSignals`、`SceneProgressDialog`。
- `control/scene_roof_controller.py`：大场景 `RoofPlaneController`，继承保留的 `LegacyRoofPlaneController` 手势和基础模式交互。
- `view/plane_table.py`：`PlaneTableModel`。
- 修改 `model/point_cloud.py`、`io/pointclouds/las_handler.py`、`control/pcd_manager.py`、
  `control/controller.py`、`control/roof_plane_controller.py`、`view/roof_plane_panel.py`、`view/viewer.py`、`view/gui.py`。
- 新增大场景模型/界面测试，更新原 Roof Plane 测试以适配“工作保存与最终导出分离”、异步选择及取消数字快捷键。

## 验证与实际规模

真实文件：`D:/dataset/USA-Pennsylvania/45v1/465902CA.laz`，**17,918,619 点**。
只读测试，没有对该场景生成标注输出或更改原始 LAZ。GPU：NVIDIA GeForce RTX 3060。

| 本次测试操作 | 时间 |
|---|---:|
| 加载及初始化窗口（含原始点读取） | 12.91 秒 |
| 首次投影全部点 | 1.66 秒 |
| 使用缓存框选 9025 点 | 0.310 秒 |
| overlay 更新 | 2.74 毫秒 |
| 一次真实 GL 绘制并等待完成 | 26.8 毫秒 |

这是该机器、视角、区域下的测量，不是所有场景的帧率保证；大选区复杂套索仍可能更慢，但选择计算在后台。
已覆盖 uint32 边界、历史 ID、统计/撤销一致性、sidecar 恢复和源文件校验、LAS/LAZ 全字段与 EVLR 导出、
投影失效、保护过滤、隐藏/恢复、ROI、书签/网格、自动保存、现有 Wireframe 和 Roof Plane 交互。
完整测试集：**135 项通过**；另完成上述真实 1792 万点 OpenGL 与投影/选择检查。

## 使用与边界

旧 `_gt.las` 在没有侧车时导入；此后以 sidecar 为权威。要交付完整点云请使用“导出最终 LAS/LAZ”，
不要将 `plane_id.npy` 当作 LAS。0 表示未分配，不等同于 reviewed。
选择为穿透式，ROI 为 XY 包围盒；用户仍需检查几何上连续的共面区域。
内存依赖完整渲染坐标、颜色和屏幕缓存；空间树按需建立，千万点场景仍需要足够 RAM/显存。
单步撤销若超过配置的 128 MiB 上限会被丢弃；这是有界内存策略，界面会提示。
