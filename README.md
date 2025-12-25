# BuildFrame - 屋顶点云标注工具

本项目是 [labelCloud](https://github.com/ch-sa/labelCloud) 的一个二次开发版本，用于建筑屋顶的关键点/线段标注。

原项目作者：Christoph Sager  
原项目许可证：GNU General Public License v3 or later (GPLv3+)

## 主要修改与新增功能
- 移除原有的3D bounding box标注功能
- 新增屋顶点/线绘制模式（支持实时预览、闭合判断）
- 支持将屋顶轮廓保存为OBJ文件（仅顶点+线）
- 新增LAS/LAZ点云读取支持
- 优化大点云深度缓冲问题（near/far plane调整建议）
- UI全面调整为屋顶标注专用（interface_roof.ui）
- 其他优化：标准视图快捷键、地面滤波等

## 安装与运行
```bash
git clone https://github.com/zzxy666/BuildFrame.git
cd BuildFrame
pip install -r requirements.txt  
python -m labelCloud