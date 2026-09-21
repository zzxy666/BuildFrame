# roof_drawing_manager.py

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

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
        self._point_tree: Optional[cKDTree] = None
        self._point_tree_source_id: Optional[int] = None


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

    def reset_temp_state(self, clear_closed: bool = True):
        self.temp_points.clear()
        self.preview_point = None
        self.preview_line = None
        if clear_closed:
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

        source_id = id(pcd_points)
        if self._point_tree is None or self._point_tree_source_id != source_id:
            self._point_tree = cKDTree(pcd_points)
            self._point_tree_source_id = source_id
        dist, idx = self._point_tree.query(point, k=1)
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
            self.closed = False

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
            # Use exactly the first point so the graph is topologically closed,
            # instead of merely looking closed within a distance threshold.
            self.temp_points[-1] = self.temp_points[0]
            self.lines[-1]["coord"] = (
                self.lines[-1]["coord"][0],
                self.temp_points[0],
            )
            self.closed = True
            for pt in self.temp_points[:-1]:
                name = f"Vertex{len(self.vertices) + 1}"
                self.vertices.append(pt)
                self.vertex_info.append({"name": name, "coord": pt})

            if self.controller:
                self.controller.update_point_list()

            self.reset_temp_state(clear_closed=False)
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
        self.reset_temp_state(clear_closed=False)

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

    @staticmethod
    def _atomic_write(path: Path, writer: Callable) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                writer(stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, path)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise

    def _graph(self):
        """Return deterministic vertices and edge indices for serialization."""
        vertices = [tuple(map(float, point)) for point in self.vertices]

        def index_for(point):
            candidate = tuple(map(float, point))
            for index, vertex in enumerate(vertices):
                if np.allclose(vertex, candidate, atol=1e-7):
                    return index
            vertices.append(candidate)
            return len(vertices) - 1

        edges = [
            (index_for(line["coord"][0]), index_for(line["coord"][1]))
            for line in self.lines
        ]
        return vertices, edges

    def save_to_obj(self, filepath: str, pointcloud=None):
        """Atomically export deterministic OBJ vertices/edges in world space."""
        path = Path(filepath)
        vertices, edges = self._graph()
        if pointcloud is not None and vertices:
            vertices = [
                tuple(pointcloud.to_world_coordinates(vertex)) for vertex in vertices
            ]

        def write_obj(stream):
            stream.write("# BuildFrame roof annotation (world coordinates)\n")
            for point in vertices:
                stream.write(f"v {point[0]:.9f} {point[1]:.9f} {point[2]:.9f}\n")
            for start, end in edges:
                stream.write(f"l {start + 1} {end + 1}\n")

        self._atomic_write(path, write_obj)

    def save_project(self, filepath: str, pointcloud) -> None:
        """Save the versioned internal annotation format in world coordinates."""
        path = Path(filepath)
        vertices, edges = self._graph()
        world_vertices = (
            pointcloud.to_world_coordinates(vertices).tolist() if vertices else []
        )
        payload = {
            "format": "buildframe-roof",
            "version": 1,
            "coordinate_space": "world",
            "source": pointcloud.path.name,
            "vertices": world_vertices,
            "edges": [list(edge) for edge in edges],
        }

        def write_json(stream):
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

        self._atomic_write(path, write_json)

    def load_project(self, filepath: str, pointcloud) -> None:
        path = Path(filepath)
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("format") != "buildframe-roof" or payload.get("version") != 1:
            raise ValueError(f"Unsupported BuildFrame annotation format: {path}")
        world_vertices = np.asarray(payload.get("vertices", []), dtype=np.float64)
        local_vertices = (
            pointcloud.to_local_coordinates(world_vertices) if len(world_vertices) else []
        )
        self.vertices = [tuple(point) for point in local_vertices]
        self.vertex_info = [
            {"name": f"Vertex{i + 1}", "coord": point}
            for i, point in enumerate(self.vertices)
        ]
        self.lines = []
        for start, end in payload.get("edges", []):
            if not (0 <= start < len(self.vertices) and 0 <= end < len(self.vertices)):
                raise ValueError(f"Invalid edge ({start}, {end}) in {path}")
            self.lines.append(
                {
                    "name": f"Edge{len(self.lines) + 1}",
                    "coord": (self.vertices[start], self.vertices[end]),
                }
            )
        self.reset_temp_state()
        self._update_closed_state()
        self.mode = "point"
    
    def load_from_obj(self, filepath: str, pointcloud=None, coordinates="local"):
        """Load OBJ vertices/edges, optionally converting world to local space."""
        if not os.path.exists(filepath):
            return

        vertices = []
        vertex_map = {}  # 顶点坐标 -> 索引（防止重复）
        lines = []

        with open(filepath, 'r', encoding="utf-8") as f:
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

        if coordinates == "world" and pointcloud is not None and vertices:
            vertices = [
                tuple(point) for point in pointcloud.to_local_coordinates(vertices)
            ]
            lines = [
                {
                    "name": line["name"],
                    "coord": (
                        tuple(pointcloud.to_local_coordinates(line["coord"][0])),
                        tuple(pointcloud.to_local_coordinates(line["coord"][1])),
                    ),
                }
                for line in lines
            ]

        self.vertices = vertices
        self.vertex_info = [{"name": f"Vertex{i+1}", "coord": p} for i, p in enumerate(vertices)]
        self.lines = lines
        self.reset_temp_state()
        self._update_closed_state()
        self.mode = "point"  # 或 "line"，根据你需要
        print(f"[RoofDrawing] 从 {filepath} 成功加载 {len(vertices)} 个顶点，{len(lines)} 条边")

    def _update_closed_state(self) -> None:
        self.closed = bool(
            self.lines
            and np.allclose(
                self.lines[-1]["coord"][1], self.lines[0]["coord"][0], atol=1e-7
            )
        )

    def invalidate_point_index(self) -> None:
        self._point_tree = None
        self._point_tree_source_id = None

    def delete_vertex(self, index: int) -> Optional[int]:
        if not 0 <= index < len(self.vertices):
            return None
        deleted_point = self.vertices.pop(index)
        self.vertex_info.pop(index)
        original_count = len(self.lines)
        self.lines = [
            line
            for line in self.lines
            if not (
                np.allclose(line["coord"][0], deleted_point, atol=1e-6)
                or np.allclose(line["coord"][1], deleted_point, atol=1e-6)
            )
        ]
        self._renumber()
        self._clear_selection()
        self._update_closed_state()
        return original_count - len(self.lines)

    def delete_edge(self, index: int) -> bool:
        if not 0 <= index < len(self.lines):
            return False
        del self.lines[index]
        self._renumber()
        self._clear_selection()
        self._update_closed_state()
        return True

    def move_vertex(self, index: int, offset) -> Tuple[float, float, float]:
        if not 0 <= index < len(self.vertices):
            raise IndexError(index)
        old_point = self.vertices[index]
        new_point = tuple(np.asarray(old_point) + np.asarray(offset, dtype=float))
        self.vertices[index] = new_point
        self.vertex_info[index]["coord"] = new_point
        for line in self.lines:
            start, end = line["coord"]
            if np.allclose(start, old_point, atol=1e-6):
                start = new_point
            if np.allclose(end, old_point, atol=1e-6):
                end = new_point
            line["coord"] = (start, end)
        return new_point

    def clear(self) -> None:
        self.vertices.clear()
        self.vertex_info.clear()
        self.lines.clear()
        self.reset_temp_state()
        self._clear_selection()
        self.invalidate_point_index()
        self.mode = "point"

    def _renumber(self) -> None:
        for index, info in enumerate(self.vertex_info):
            info["name"] = f"Vertex{index + 1}"
        for index, line in enumerate(self.lines):
            line["name"] = f"Edge{index + 1}"

    def _clear_selection(self) -> None:
        self.active_vertex_index = None
        self.active_line_index = None
        self.active_line_start_idx = None
        self.active_line_end_idx = None


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
