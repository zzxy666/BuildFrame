"""Regression tests for the CSF adapter and the pinned upstream native engine."""

from configparser import ConfigParser
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from PointCloudFilter.csf_filter import CSFGroundFilter, CSFOptions, prepare_points


@pytest.fixture
def native_csf(monkeypatch):
    # Limit native OpenMP work in tests, particularly on many-core machines.
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    return pytest.importorskip("CSF")


def _terrain_with_roof(slope=0.0):
    x, y = np.meshgrid(np.arange(0, 20.5, 0.5), np.arange(0, 20.5, 0.5))
    roof = ((x >= 8) & (x <= 12) & (y >= 8) & (y <= 12)).ravel()
    points = np.column_stack((x.ravel(), y.ravel(), slope * x.ravel()))
    # There are no ground returns underneath the roof, as in airborne data.
    points[roof, 2] += 5.0
    return points, roof


def test_flat_ground_and_roof_are_separated_in_original_point_order(native_csf):
    points, roof = _terrain_with_roof()
    order = np.random.default_rng(42).permutation(len(points))
    labels = CSFGroundFilter().process(points[order])
    assert labels.shape == (len(points),)
    assert labels.dtype == np.int8
    np.testing.assert_array_equal(labels, np.where(roof[order], 1, 2))


def test_gently_sloping_ground_and_roof(native_csf):
    points, roof = _terrain_with_roof(slope=0.25)
    labels = CSFGroundFilter(CSFOptions(rigidness=2)).process(points)
    # Allow a narrow transition region around the synthetic roof boundary.
    assert np.mean(labels[~roof] == 2) >= 0.95
    assert np.all(labels[roof] == 1)


@pytest.mark.parametrize(
    "options",
    [CSFOptions(rigidness=1), CSFOptions(rigidness=1, resolution=0.5, time_step=1.0)],
)
def test_steep_terrain_preserves_valid_complete_partition(native_csf, options):
    # A steep-terrain preset alone is not an accuracy guarantee: resolution,
    # time step, sampling density, and the scene all affect cloth convergence.
    points, _ = _terrain_with_roof(slope=1.0)
    original = points.copy()
    labels = CSFGroundFilter(options).process(points)
    assert labels.shape == (len(points),)
    assert labels.dtype == np.int8
    assert np.isin(labels, [1, 2]).all()
    assert np.count_nonzero(labels == 1) + np.count_nonzero(labels == 2) == len(points)
    np.testing.assert_array_equal(points, original)


def test_translation_and_display_scale_do_not_change_classification(native_csf):
    points, _ = _terrain_with_roof(slope=0.25)
    original = points.copy()
    engine = CSFGroundFilter(CSFOptions(rigidness=2))
    expected = engine.process(points)
    translated = points + np.array([600_000, 4_000_000, 300])
    # Model display normalization is undone before choosing a cloth grid.
    local = (points - points.mean(axis=0)) / 128.0
    np.testing.assert_array_equal(engine.process(translated), expected)
    np.testing.assert_array_equal(engine.process(local, applied_scale=128), expected)
    np.testing.assert_array_equal(points, original)


@pytest.mark.parametrize("units_per_meter", [1000.0, 100.0, 1.0 / 0.3048, 2.5])
def test_explicit_source_units_preserve_native_classification(native_csf, units_per_meter):
    points_m, roof = _terrain_with_roof()
    source_points = (points_m + np.array([1000, 2000, 10])) * units_per_meter
    original = source_points.copy()
    options = CSFOptions(units_per_meter=units_per_meter)
    labels = CSFGroundFilter(options).process(source_points)
    np.testing.assert_array_equal(labels, np.where(roof, 1, 2))
    # Display normalization and source-unit conversion are independent steps.
    local_points = (source_points - source_points.mean(axis=0)) / 128.0
    np.testing.assert_array_equal(
        CSFGroundFilter(options).process(local_points, applied_scale=128), labels,
    )
    np.testing.assert_array_equal(source_points, original)


def test_millimeter_source_units_preserve_gentle_slope_classification(native_csf):
    points_m, _ = _terrain_with_roof(slope=0.25)
    baseline = CSFGroundFilter(CSFOptions(rigidness=2)).process(points_m)
    labels = CSFGroundFilter(CSFOptions(rigidness=2, units_per_meter=1000)).process(
        points_m * 1000,
    )
    np.testing.assert_array_equal(labels, baseline)


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        ([[3, 4, 5]], [2]),
        ([[1, 1, 0], [1, 1, 0.05], [1, 1, 10]], [2, 2, 1]),
        ([[0, 0, 0], [1, 0, 0], [5, 0, 0], [5, 0, 4]], [2, 2, 2, 1]),
        ([[0, 0, 0], [0, 1, 0], [0, 5, 0], [0, 5, 4]], [2, 2, 2, 1]),
    ],
)
def test_native_degenerate_extents_in_isolated_process(native_csf, points, expected):
    # Native code failures must not take down the whole test runner.
    code = (
        "import json, sys; "
        "from PointCloudFilter.csf_filter import CSFGroundFilter; "
        "labels = CSFGroundFilter().process(json.loads(sys.argv[1])); "
        "print('LABELS_JSON=' + json.dumps(labels.tolist()))"
    )
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="2", BUILDFRAME_LOG_PATH="NUL", PYTHONIOENCODING="utf-8")
    result = subprocess.run(
        [sys.executable, "-c", code, json.dumps(points)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result_line = next(line for line in result.stdout.splitlines() if line.startswith("LABELS_JSON="))
    assert json.loads(result_line.partition("=")[2]) == expected


def test_empty_cloud_does_not_require_native_library(monkeypatch):
    monkeypatch.setitem(sys.modules, "CSF", None)
    labels = CSFGroundFilter().process(np.empty((0, 3)))
    assert labels.shape == (0,)
    assert labels.dtype == np.int8


@pytest.mark.parametrize("points", [[], [1, 2, 3], [[1, 2]], [[1, 2, 3, 4]], np.zeros((2, 3, 1))])
def test_invalid_coordinate_shape_is_rejected_before_native_import(monkeypatch, points):
    monkeypatch.setitem(sys.modules, "CSF", None)
    with pytest.raises(ValueError, match="Nx3"):
        CSFGroundFilter().process(points)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_coordinates_are_rejected(monkeypatch, value):
    monkeypatch.setitem(sys.modules, "CSF", None)
    with pytest.raises(ValueError, match="NaN or infinite"):
        CSFGroundFilter().process([[0, value, 0]])


@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf, -np.inf])
def test_invalid_display_scale_is_rejected(scale):
    with pytest.raises(ValueError, match="applied_scale"):
        prepare_points([[0, 0, 0]], CSFOptions(), scale)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("resolution", 0), ("resolution", -1), ("resolution", np.inf),
        ("resolution", np.nan), ("resolution", True),
        ("threshold", 0), ("threshold", -1), ("threshold", np.nan),
        ("time_step", 0), ("time_step", 1.01), ("time_step", np.inf),
        ("rigidness", 0), ("rigidness", 4), ("rigidness", 1.0), ("rigidness", True),
        ("iterations", 0), ("iterations", 10001), ("iterations", 2.5), ("iterations", False),
        ("slope_smooth", 1), ("slope_smooth", "true"),
        ("units_per_meter", 0), ("units_per_meter", -1), ("units_per_meter", np.nan),
        ("units_per_meter", np.inf), ("units_per_meter", -np.inf),
        ("units_per_meter", True), ("units_per_meter", False),
    ],
)
def test_invalid_options_are_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        replace(CSFOptions(), **{name: value}).validate()


def test_prepare_points_copies_readonly_input_and_restores_units():
    points = np.asfortranarray([[10, 20, -2], [11, 23, 5]], dtype=np.float32)
    original = points.copy()
    points.setflags(write=False)
    prepared = prepare_points(points, CSFOptions(), applied_scale=4)
    np.testing.assert_array_equal(prepared, [[0, 0, 0], [4, 12, 28]])
    np.testing.assert_array_equal(points, original)
    assert prepared.dtype == np.float64
    assert prepared.flags.c_contiguous
    assert not np.shares_memory(points, prepared)


def test_prepare_points_restores_display_scale_then_converts_to_meters():
    # The local display uses 4 source units per local unit; the file uses mm.
    points = np.array([[10, 20, -2], [260, 520, 248]], dtype=np.float64)
    prepared = prepare_points(points, CSFOptions(units_per_meter=1000), applied_scale=4)
    np.testing.assert_array_equal(prepared, [[0, 0, 0], [1, 2, 1]])


def test_cloth_grid_guard_uses_explicit_meters_without_guessing_units():
    points = [[0, 0, 0], [1000, 1000, 0]]
    with pytest.raises(ValueError, match="grid"):
        prepare_points(points, CSFOptions())
    prepared = prepare_points(points, CSFOptions(units_per_meter=1000))
    np.testing.assert_array_equal(prepared, [[0, 0, 0], [1, 1, 0]])


def test_source_unit_conversion_overflow_is_rejected():
    with pytest.raises(ValueError, match="extent"):
        prepare_points([[0, 0, 0], [2, 2, 0]], CSFOptions(units_per_meter=1e-320))


@pytest.mark.parametrize(
    ("points", "scale"),
    [
        ([[0, 0, 0], [1_000_000, 0, 0]], 1),
        ([[0, 0, 0], [1000, 1000, 0]], 1),
        ([[0, 0, 0], [2, 2, 0]], 1000),
    ],
)
def test_cloth_grid_memory_guard_runs_before_native_import(monkeypatch, points, scale):
    monkeypatch.setitem(sys.modules, "CSF", None)
    with pytest.raises(ValueError, match="grid"):
        CSFGroundFilter().process(points, applied_scale=scale)


def test_coordinate_extent_overflow_is_rejected():
    maximum = np.finfo(np.float64).max
    with pytest.raises(ValueError, match="extent"):
        prepare_points([[-maximum, 0, 0], [maximum, 0, 0]], CSFOptions())


def test_resolution_division_overflow_is_rejected():
    with pytest.raises(ValueError, match="grid"):
        prepare_points([[0, 0, 0], [2, 2, 0]], CSFOptions(resolution=1e-320))


def test_configuration_defaults_and_round_trip_preserve_unrelated_values():
    config = ConfigParser()
    config.read_dict({"USER_INTERFACE": {"language": "en_US"}})
    assert CSFOptions.from_config(config) == CSFOptions()
    custom = CSFOptions(
        resolution=0.25, threshold=0.15, rigidness=1,
        slope_smooth=False, iterations=900, time_step=0.5, units_per_meter=1000,
    )
    custom.to_config(config)
    assert CSFOptions.from_config(config) == custom
    assert config.get("USER_INTERFACE", "language") == "en_US"


def test_invalid_configuration_is_not_silently_used_or_written():
    config = ConfigParser()
    config.read_dict({"GROUND_FILTER": {"resolution": "-1"}})
    with pytest.raises(ValueError, match="resolution"):
        CSFOptions.from_config(config)
    with pytest.raises(ValueError, match="threshold"):
        CSFOptions(threshold=-1).to_config(config)
    assert dict(config["GROUND_FILTER"]) == {"resolution": "-1"}


def test_partial_configuration_uses_defaults_for_missing_fields():
    config = ConfigParser()
    config.read_dict({"GROUND_FILTER": {"resolution": "0.25"}})
    assert CSFOptions.from_config(config) == CSFOptions(resolution=0.25)
    assert CSFOptions.from_config(config).units_per_meter == 1.0


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "not-a-number"])
def test_invalid_source_units_in_configuration_are_rejected(value):
    config = ConfigParser()
    config.read_dict({"GROUND_FILTER": {"units_per_meter": value}})
    with pytest.raises(ValueError):
        CSFOptions.from_config(config)


def test_options_are_forwarded_and_native_indices_keep_input_order(monkeypatch):
    options = CSFOptions(
        resolution=0.25, threshold=0.15, rigidness=1,
        slope_smooth=False, iterations=900, time_step=0.5, units_per_meter=1000,
    )
    captured = SimpleNamespace(params=SimpleNamespace())

    class FakeCSF:
        params = captured.params

        def setPointCloud(self, points):
            captured.points = points.copy()

        def do_filtering(self, ground_result, other_result, export_cloth):
            assert export_cloth is False
            ground_result.extend([2, 0])
            other_result.append(1)

    monkeypatch.setitem(sys.modules, "CSF", SimpleNamespace(CSF=FakeCSF, VecInt=list))
    labels = CSFGroundFilter(options).process(
        [[10000, 20000, 5000], [11000, 20000, 6000], [12000, 20000, 5000]],
    )
    np.testing.assert_array_equal(labels, [2, 1, 2])
    np.testing.assert_array_equal(captured.points, [[0, 0, 0], [1, 0, 1], [2, 0, 0]])
    assert vars(captured.params) == {
        "cloth_resolution": 0.25, "class_threshold": 0.15, "rigidness": 1,
        "bSloopSmooth": False, "interations": 900, "time_step": 0.5,
    }


@pytest.mark.parametrize(
    ("ground", "other"),
    [([0, 0], [2]), ([0], [1]), ([-1], [1, 2]), ([3], [1, 2]), ([0, 1], [1])],
)
def test_invalid_native_partitions_are_rejected(monkeypatch, ground, other):
    class FakeCSF:
        params = SimpleNamespace()

        def setPointCloud(self, points):
            pass

        def do_filtering(self, ground_result, other_result, export_cloth):
            ground_result.extend(ground)
            other_result.extend(other)

    monkeypatch.setitem(sys.modules, "CSF", SimpleNamespace(CSF=FakeCSF, VecInt=list))
    with pytest.raises(RuntimeError, match="invalid point partition"):
        CSFGroundFilter().process([[0, 0, 0], [1, 0, 0], [2, 0, 0]])


def test_missing_native_dependency_has_installation_guidance(monkeypatch):
    monkeypatch.setitem(sys.modules, "CSF", None)
    with pytest.raises(RuntimeError, match="pip install cloth-simulation-filter"):
        CSFGroundFilter().process([[0, 0, 0]])
