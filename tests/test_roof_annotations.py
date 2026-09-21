import json
from pathlib import Path

import numpy as np

from labelCloud.control.roof_drawing_manager import RoofDrawingManager


class _Controller:
    def update_point_list(self):
        pass


class _PointCloud:
    path = Path("source.las")
    center = np.array([100.0, 200.0, 300.0])
    scale = 10.0

    def to_world_coordinates(self, points):
        return np.asarray(points, dtype=float) * self.scale + self.center

    def to_local_coordinates(self, points):
        return (np.asarray(points, dtype=float) - self.center) / self.scale


def _closed_triangle():
    manager = RoofDrawingManager(_Controller())
    manager.set_mode("line")
    manager._add_line_point((0.0, 0.0, 0.0))
    manager._add_line_point((1.0, 0.0, 0.0))
    manager._add_line_point((0.0, 1.0, 0.0))
    manager.close_polygon_manually()
    return manager


def test_project_save_load_round_trip_uses_world_coordinates(tmp_path):
    path = tmp_path / "source.roof.json"
    source = _closed_triangle()
    source.save_project(path, _PointCloud())

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["coordinate_space"] == "world"
    assert payload["vertices"][0] == [100.0, 200.0, 300.0]
    assert not list(tmp_path.glob("*.tmp"))

    loaded = RoofDrawingManager(_Controller())
    loaded.load_project(path, _PointCloud())
    np.testing.assert_allclose(loaded.vertices, source.vertices)
    assert [line["coord"] for line in loaded.lines] == [
        line["coord"] for line in source.lines
    ]


def test_obj_export_is_deterministic_and_in_world_coordinates(tmp_path):
    manager = _closed_triangle()
    first = tmp_path / "first.obj"
    second = tmp_path / "second.obj"

    manager.save_to_obj(first, _PointCloud())
    manager.save_to_obj(second, _PointCloud())

    assert first.read_bytes() == second.read_bytes()
    assert "v 100.000000000 200.000000000 300.000000000" in first.read_text()


def test_move_and_delete_vertex_keep_edges_consistent():
    manager = _closed_triangle()
    moved = manager.move_vertex(0, (0.5, 0.0, 0.0))

    assert moved == (0.5, 0.0, 0.0)
    assert any(moved in line["coord"] for line in manager.lines)

    removed_edges = manager.delete_vertex(0)
    assert removed_edges == 2
    assert all(moved not in line["coord"] for line in manager.lines)


def test_auto_close_uses_exact_first_vertex():
    manager = RoofDrawingManager(_Controller())
    manager.set_mode("line")
    manager._add_line_point((0.0, 0.0, 0.0))
    manager._add_line_point((1.0, 0.0, 0.0))
    manager._add_line_point((0.1, 0.1, 0.0))

    assert manager.closed
    assert manager.lines[-1]["coord"][1] == manager.lines[0]["coord"][0]

