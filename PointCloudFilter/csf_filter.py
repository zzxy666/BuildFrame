"""Adapter for the upstream jianboqi/CSF cloth simulation ground filter.

Coordinates must be Cartesian XYZ, with Z up. CSF distances use meters;
units_per_meter explicitly converts source units before native simulation.
Labels preserve input order: 2 = ground, 1 = non-ground. No points are removed.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CSFOptions:
    resolution: float = 1.0
    threshold: float = 0.5
    rigidness: int = 3
    slope_smooth: bool = True
    iterations: int = 500
    time_step: float = 0.65
    units_per_meter: float = 1.0

    def validate(self) -> None:
        for name in ("resolution", "threshold", "time_step", "units_per_meter"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if self.time_step > 1:
            raise ValueError("time_step must not exceed 1")
        if type(self.rigidness) is not int or self.rigidness not in (1, 2, 3):
            raise ValueError("rigidness must be 1, 2 or 3")
        if type(self.iterations) is not int or not 1 <= self.iterations <= 10000:
            raise ValueError("iterations must be an integer between 1 and 10000")
        if type(self.slope_smooth) is not bool:
            raise ValueError("slope_smooth must be a boolean")

    @classmethod
    def from_config(cls, config):
        defaults = cls()
        values = {}
        for name, default in asdict(defaults).items():
            getter = config.getboolean if type(default) is bool else (
                config.getint if type(default) is int else config.getfloat
            )
            values[name] = getter("GROUND_FILTER", name, fallback=default)
        options = cls(**values)
        options.validate()
        return options

    def to_config(self, config) -> None:
        self.validate()
        if not config.has_section("GROUND_FILTER"):
            config.add_section("GROUND_FILTER")
        for name, value in asdict(self).items():
            config.set("GROUND_FILTER", name, str(value))


# A cloth particle has neighbors and other native allocations; this is a safety
# cap, NOT a promise about peak RAM. Avoid multi-GB grids from bad units/outliers.
MAX_CLOTH_NODES = 1_000_000


def prepare_points(points, options: CSFOptions, applied_scale: float = 1.0):
    """Validate before entering native code and undo display normalization."""
    options.validate()
    if not math.isfinite(applied_scale) or applied_scale <= 0:
        raise ValueError("applied_scale must be finite and positive")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an Nx3 XYZ array")
    if not np.isfinite(points).all():
        raise ValueError("points contain NaN or infinite coordinates; clean them first")
    if not len(points):
        return np.empty((0, 3), dtype=np.float64)
    if len(points) > np.iinfo(np.int32).max:
        raise ValueError("Too many points for CSF; split the cloud into tiles")
    # Remove large global offsets before any scaling. Translation does not
    # affect classification; leaving units intact does affect it.
    with np.errstate(over="ignore", invalid="ignore"):
        centered = (points - points.min(axis=0)) * (applied_scale / options.units_per_meter)
    if not np.isfinite(centered).all():
        raise ValueError("Coordinate extent is too large")
    extent = centered.max(axis=0)
    with np.errstate(over="ignore"):
        cells = extent[:2] / options.resolution
    # Upstream CSF.cpp uses floor(extent / resolution) + 4 along each XY axis.
    if not np.isfinite(cells).all() or np.any(cells > MAX_CLOTH_NODES):
        raise ValueError("CSF grid is too large; increase cloth resolution or crop the cloud")
    nodes = (math.floor(cells[0]) + 4) * (math.floor(cells[1]) + 4)
    if nodes > MAX_CLOTH_NODES:
        raise ValueError(
            f"CSF grid needs {nodes:,} nodes (limit {MAX_CLOTH_NODES:,}); "
            "increase cloth resolution in Ground Filter Settings or crop the cloud. "
            "Resolution is in meters: check units_per_meter for the source cloud."
        )
    return np.ascontiguousarray(centered, dtype=np.float64)


class CSFGroundFilter:
    def __init__(self, options=None):
        self.options = options or CSFOptions()

    def process(self, points, applied_scale: float = 1.0):
        points = prepare_points(points, self.options, applied_scale)
        if not len(points):
            return np.empty(0, dtype=np.int8)
        try:
            import CSF
        except ImportError as exc:
            raise RuntimeError(
                "CSF is not installed in this Python environment. Run: "
                "python -m pip install cloth-simulation-filter==1.1.7"
            ) from exc
        csf = CSF.CSF()
        csf.params.cloth_resolution = self.options.resolution
        csf.params.class_threshold = self.options.threshold
        csf.params.rigidness = self.options.rigidness
        csf.params.bSloopSmooth = self.options.slope_smooth
        csf.params.interations = self.options.iterations  # upstream spelling
        csf.params.time_step = self.options.time_step
        csf.setPointCloud(points)
        ground, other = CSF.VecInt(), CSF.VecInt()
        csf.do_filtering(ground, other, False)
        ground = np.asarray(ground, dtype=np.int64)
        other = np.asarray(other, dtype=np.int64)
        indices = np.concatenate((ground, other))
        if (
            len(indices) != len(points)
            or np.any(indices < 0)
            or np.any(indices >= len(points))
            or len(np.unique(indices)) != len(points)
        ):
            raise RuntimeError("CSF returned an invalid point partition")
        labels = np.ones(len(points), dtype=np.int8)
        labels[ground] = 2
        return labels
