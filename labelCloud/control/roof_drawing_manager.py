# roof_drawing_manager.py

import logging
import numpy as np
from scipy.spatial import cKDTree
from typing import TYPE_CHECKING, List, Tuple
import os

if TYPE_CHECKING:
    from ..view.gui import GUI

class RoofDrawingManager:
    def __init__(self, controller):
        self.controller = controller
        self.view = None

        # 所有已确认的顶点（来自点模式或线模式）
        self.vertices: List[Tuple[float, float, float]] = []      # list of (x,y,z)
        self.vertex_info: List[dict] = []                         # 带名字的顶点信息

        # 线模式专用：正在构建的多边形临时点
        self.temp_points: List[Tuple[float, float, float]] = []

        # 所有已确认的线段（包含闭合边）
        self.lines: List[dict] = []                               # 每条线是 {"name": str, "coord": ((x1,y1,z1), (x2,y2,z2))}

        self.closed = False

        # 预览
        self.preview_point = None
        self.preview_line = None

        self.mode = "point"          # "point" 或 "line"
        self.active_vertex_index = None
        self.active_line_index = None
        self.active_line_start_idx = None
        self.active_line_end_idx = None

        self.connect_mode = False
        self.connect_first_vertex_idx = None


    def set_view(self, view):
        self.view = view

    def set_mode(self, mode: str) -> None:
        if mode is None:
            # 取消模式：清空临时状态，回到导航
            self.mode = None
            self.reset_temp_state()  # 清空预览等
            print("[RoofDrawing] roof mode OFF")
            return

        if mode not in ("point", "line"):
            raise ValueError(f"Invalid mode: {mode}, must be 'point' or 'line'")

        self.mode = mode
        self.reset_temp_state()  # 切换模式时清空临时点
        print(f"[RoofDrawing] {mode} mode ON")

    def reset_temp_state(self):
        self.temp_points.clear()
        self.preview_point = None
        self.preview_line = None
        self.closed = False

    def register_point(self, x: float, y: float, z: float):
        world_point = np.array([x, y, z])
        nearest_point = self._snap_to_nearest_point(world_point)

        if self.mode == "point":
            self._add_vertex(nearest_point)
        elif self.mode == "line":
            self._add_line_point(nearest_point)

        if self.view and self.view.gl_widget:
            self.view.gl_widget.update()

    def _snap_to_nearest_point(self, point):
        pcd_points = np.asarray(self.view.gl_widget.pcd_manager.pointcloud.points)
        if len(pcd_points) == 0:
            return tuple(point)

        tree = cKDTree(pcd_points)
        dist, idx = tree.query(point, k=1)
        if dist < 5.0:  # 可调阈值
            return tuple(pcd_points[idx])
        return tuple(point)

    def _add_vertex(self, p):
        name = f"Vertex{len(self.vertices) + 1}"
        self.vertices.append(p)
        self.vertex_info.append({"name": name, "coord": p})
        self.controller.update_point_list()

    def _add_line_point(self, p):
        if self.closed:
            return

        self.temp_points.append(p)

        # 每添加一个点，就把“上一个点 → 当前点”这条边加进去
        if len(self.temp_points) >= 2:
            edge_name = f"Edge{len(self.lines) + 1}"
            self.lines.append({
                "name": edge_name,
                "coord": (self.temp_points[-2], self.temp_points[-1])
            })

        # 当添加第N个点（N>=3）且当前点离第一个点足够近 → 自动闭合

        if len(self.temp_points) >= 3 and self._near(p, self.temp_points[0]):
            self.closed = True
            for i, pt in enumerate(self.temp_points):
                name = f"Vertex{len(self.vertices) + 1}"
                self.vertices.append(pt)
                self.vertex_info.append({"name": name, "coord": pt})

            if self.controller:
                self.controller.update_point_list()

            self.reset_temp_state()
            print("[RoofDrawing] 多边形自动闭合，点和线已更新到右侧列表")

    def close_polygon_manually(self):
        """手动闭合多边形（鼠标右键）"""
        if len(self.temp_points) < 3:
            print("[RoofDrawing] 至少需要3个点才能闭合")
            return
        edge_name = f"Edge{len(self.lines) + 1}"
        # 添加闭合边
        self.lines.append({
            "name": edge_name,
            "coord": (self.temp_points[-1], self.temp_points[0])
        })
        self.closed = True

        for i, pt in enumerate(self.temp_points):
            name = f"Vertex{len(self.vertices) + 1}"
            self.vertices.append(pt)
            self.vertex_info.append({"name": name, "coord": pt})


        if self.controller:
            self.controller.update_point_list()

        # 重置临时状态
        self.reset_temp_state()

        print("[RoofDrawing] 多边形已手动闭合，点已显示在右侧")

    def _near(self, p, p0, threshold=0.5):
        return np.linalg.norm(np.array(p) - np.array(p0)) < threshold

    def update_preview(self, pos3d):
        self.preview_point = pos3d

        if self.mode == "line" and len(self.temp_points) > 0 and not self.closed:
            start = self.temp_points[-1]
            self.preview_line = (start, pos3d)
        else:
            self.preview_line = None

    def clear_preview(self):
        self.preview_point = None
        self.preview_line = None

    def save_to_obj(self, filepath: str):
        """保存为 OBJ 文件（顶点 + 线）"""
        with open(filepath, 'w') as f:
            # 收集所有唯一顶点
            all_points = set()
            for p in self.vertices:
                all_points.add(p)
            for line in self.lines:
                all_points.add(line["coord"][0])
                all_points.add(line["coord"][1])

            # 写入顶点
            vertex_map = {}
            vertex_index = 1
            for p in all_points:
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
                vertex_map[p] = vertex_index
                vertex_index += 1


            # 写入线元素
            for line in self.lines:
                p1, p2 = line["coord"]
                if p1 in vertex_map and p2 in vertex_map:
                    f.write(f"l {vertex_map[p1]} {vertex_map[p2]}\n")
    
    def load_from_obj(self, filepath: str):
        """从 OBJ 文件加载顶点和线（只加载 v 和 l 行）"""
        if not os.path.exists(filepath):
            return

        vertices = []
        vertex_map = {}  # 顶点坐标 -> 索引（防止重复）
        lines = []

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if parts[0] == 'v':
                    x, y, z = map(float, parts[1:4])
                    p = (x, y, z)
                    if p not in vertex_map:
                        vertex_map[p] = len(vertices)
                        vertices.append(p)
                elif parts[0] == 'l' and len(parts) == 3:
                    idx1 = int(parts[1]) - 1
                    idx2 = int(parts[2]) - 1
                    p1 = vertices[idx1]
                    p2 = vertices[idx2]
                    lines.append({
                        "name": f"Edge{len(lines)+1}",
                        "coord": (p1, p2)
                    })

        # 填充到 manager
        self.vertices = vertices
        self.vertex_info = [{"name": f"Vertex{i+1}", "coord": p} for i, p in enumerate(vertices)]
        self.lines = lines
        self.closed = len(lines) > 0 and np.linalg.norm(np.array(lines[-1]["coord"][1]) - np.array(lines[0]["coord"][0])) < 0.5

        self.reset_temp_state()
        self.mode = "point"  # 或 "line"，根据你需要
        print(f"[RoofDrawing] 从 {filepath} 成功加载 {len(vertices)} 个顶点，{len(lines)} 条边")


    def cancel_current_polygon(self):
        """取消当前正在画的多边形（清空临时点和预览）"""
        if not self.temp_points and not self.preview_line:
            print("[RoofDrawing] 没有正在绘制的多边形")
            return

        self.reset_temp_state()  # 清空临时点、预览、closed状态
        self.controller.update_point_list()  # 刷新右侧列表（去掉未完成的边）
        print("[RoofDrawing] 当前屋顶轮廓已取消")

        if self.view and self.view.gl_widget:
            self.view.gl_widget.update()

    def toggle_connect_mode(self, enabled: bool):
        """开启/关闭连接两点模式"""
        if self.connect_mode == enabled:
            return

        self.connect_mode = enabled
        if not enabled:
            self.connect_first_vertex_idx = None
            print("[RoofDrawing] 连接模式已关闭")
        else:
            self.connect_first_vertex_idx = None
            print("[RoofDrawing] 连接模式已开启：请依次点击两个已有顶点进行连接")

        if self.view and self.view.gl_widget:
            self.view.gl_widget.update()

    def try_handle_left_click(self, world_point: Tuple[float, float, float]) -> bool:
        """尝试处理左键点击（仅连接模式下），返回True表示已处理"""
        if not self.connect_mode:
            return False

        if not self.vertices:
            print("[RoofDrawing] 还没有顶点，无法连接")
            return True

        # 找最近的已有顶点
        distances = [np.linalg.norm(np.array(world_point) - np.array(v)) for v in self.vertices]
        nearest_idx = int(np.argmin(distances))
        min_dist = distances[nearest_idx]

        # 距离阈值（可根据点云密度调整，单位米）
        if min_dist > 1.0:
            print("[RoofDrawing] 未点击到已有顶点")
            return True

        if self.connect_first_vertex_idx is None:
            # 第一次点击：选中起点
            self.connect_first_vertex_idx = nearest_idx
            print(f"[RoofDrawing] 已选中起点 Vertex{nearest_idx + 1}")
        else:
            # 第二次点击：添加线段
            start_idx = self.connect_first_vertex_idx
            end_idx = nearest_idx
            if start_idx == end_idx:
                print("[RoofDrawing] 不能连接同一个点，已取消选择")
                self.connect_first_vertex_idx = None
            else:
                p1 = self.vertices[start_idx]
                p2 = self.vertices[end_idx]
                self.lines.append({
                    "name": f"Edge{len(self.lines)+1}",
                    "coord": (p1, p2)
                })
                print(f"[RoofDrawing] 已连接 Vertex{start_idx + 1} → Vertex{end_idx + 1}")

                # 刷新右侧列表
                if self.controller:
                    self.controller.update_point_list()

            # 重置准备下一次连接
            self.connect_first_vertex_idx = None

        if self.view and self.view.gl_widget:
            self.view.gl_widget.update()
        return True