# BuildFrame

城市点云屋顶标注工具，支持屋顶顶点、边线和独立屋顶面的编辑，以及 RGB 正射影像辅助标注。基于 PyQt5、OpenGL 和 labelCloud 开发。

## 安装与运行

推荐使用 Conda，Python 3.10：

```powershell
git clone https://github.com/zzxy666/BuildFrame.git
cd BuildFrame
conda env create -f environment.yml
conda activate buildframe
python buildframe.py
```

正射影像功能需要额外安装：

```powershell
python -m pip install -e ".[rgb]"
```

SAM 辅助分割为可选功能，使用前安装 `.[sam]` 并在界面配置模型权重。真实模型推理仍需在本机验证。

## 使用

程序从当前目录读取 `config.ini`。默认点云目录为 `pointclouds/`，也可以在界面中选择其他目录。支持 LAS/LAZ、PLY、PCD 等格式，界面语言可在设置中切换。

- **Wireframe**：编辑屋顶顶点、边线和多边形，保存为 `.roof.json`，支持导出世界坐标 OBJ。
- **Roof Plane**：为每个屋顶面分配独立 `plane_id`。`0` 表示未分配；一个建筑可包含多个面。
- **RGB**：打开带地理坐标的 GeoTIFF，绘制屋面轮廓并映射到点云候选。在三维视图检查、修正后确认。几何过滤可用于剔除候选中的杂点。

Roof Plane 常用快捷键：

| 按键 | 操作 |
| --- | --- |
| N / A | 新建屋顶面 / 将选区赋给当前面 |
| V / B / L | 浏览 / 矩形选择 / 套索选择 |
| Shift / Alt | 追加 / 减去选区 |
| E | 局部平面扩展 |
| Enter / Esc | 确认 / 取消候选 |
| P | 点击查询已标注屋顶面 |
| H / Shift+H | 隐藏选区 / 恢复全部 |
| Ctrl+Z / Ctrl+Y | 撤销 / 重做 |
| Ctrl+S | 保存工作标注 |

选择默认保护已有标签。局部扩展时，先选同一屋面内部的一小块点，再检查扩展预览；可用 Shift/Alt 手工修正。大场景可设置工作区、记录检查进度和书签。

点云和影像需核对坐标系及单位。RGB 轮廓映射可能同时包含屋顶、树木和地面，应在三维视图确认。当前 RGB 标签来自人工确认的轮廓或 SAM mask，点云标签改变后需要复核对应影像区域。

## 保存与导出

Roof Plane 工作标注保存在原点云旁的 `<文件名>.planar/` 目录中，支持自动保存。请保留该目录，并保持原始点云不变。

- **导出最终 LAS/LAZ**：保留原始点序和属性，添加 `uint32 plane_id`。
- **导出双 GT**：输出点云标签、与原影像网格一致的 `plane_id` GeoTIFF、有效区域 mask 和元数据。默认检查两侧标签完整性及像素冲突。

## 地面滤波

地面分离使用 [CSF](https://github.com/jianboqi/CSF)。在“设置 → 地面滤波设置”中确认单位换算系数：米为 `1`，厘米为 `100`，毫米为 `1000`。分辨率和距离阈值按米填写。

绿色为地面，红色为非地面；可取消计算或恢复原色。网格过大时增大分辨率或缩小处理范围。坡地及复杂结构需要调整参数并检查结果。

## 开发

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

## 来源与许可证

基于 Christoph Sager 的 [labelCloud](https://github.com/ch-sa/labelCloud)，按 GPL-3.0-or-later 发布，详见 [LICENSE](LICENSE)。CSF 使用上游 Apache-2.0 许可证。

开发者：Zhikang Yin、Shaobo Xia。
