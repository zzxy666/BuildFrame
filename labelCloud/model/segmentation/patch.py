"""原生像素 patch 与提示；不保存全图、不复制点标签。"""
from dataclasses import dataclass
from enum import Enum
import numpy as np
from ..coordinate_mapper import PixelMask


class SAMState(Enum):
    IDLE = "IDLE"
    PATCH_READY = "SAM_PATCH_READY"
    PROMPT_EDITING = "SAM_PROMPT_EDITING"
    MASK_PREVIEW = "SAM_MASK_PREVIEW"
    LIDAR_PREVIEW = "LIDAR_CANDIDATE_PREVIEW"
    CONFIRMED = "CONFIRMED"


@dataclass
class SAMPatch:
    col0: int
    row0: int
    width: int
    height: int
    image: object = None
    encoded: bool = False

    @classmethod
    def around(cls, point, width, height, size=1024):
        if size not in (1024, 1536, 2048): raise ValueError("Patch 尺寸必须为 1024、1536 或 2048")
        x, y = point
        if not (0 <= x < width and 0 <= y < height): raise ValueError("点击位于影像外")
        left, top = int(np.floor(x))-size//2, int(np.floor(y))-size//2
        right, bottom = min(width, left+size), min(height, top+size)
        left, top = max(0, left), max(0, top)
        return cls(left, top, right-left, bottom-top)

    @property
    def bounds(self): return (self.col0, self.row0, self.col0+self.width, self.row0+self.height)
    def contains(self, point):
        x, y = point
        return self.col0 <= x < self.col0+self.width and self.row0 <= y < self.row0+self.height
    def local(self, point): return np.asarray(point, float)-[self.col0, self.row0]
    def global_pixel(self, point): return np.asarray(point, float)+[self.col0, self.row0]
    def local_box(self, box):
        x1, y1, x2, y2 = box
        if not (self.col0 <= x1 < x2 <= self.col0+self.width and self.row0 <= y1 < y2 <= self.row0+self.height):
            raise ValueError("Box 必须有面积且完整位于当前 SAM patch 内；请重置区域或缩小 Box")
        return np.asarray(box, float)-[self.col0, self.row0, self.col0, self.row0]
    def pixel_mask(self, mask):
        data = np.asarray(mask, bool)
        if data.shape != (self.height, self.width): raise ValueError("SAM mask 尺寸与 patch 不一致")
        return PixelMask(data.copy(), self.col0, self.row0)


class PromptHistory:
    def __init__(self): self.events = []
    def clear(self): self.events.clear()
    def undo(self):
        if self.events: self.events.pop()
    @property
    def points(self): return [(value, kind) for kind, value in self.events if kind in (0, 1)]
    @property
    def box(self): return next((value for kind, value in reversed(self.events) if kind == "box"), None)
    def apply(self, backend, patch):
        backend.reset_prompts()
        for point, positive in self.points:
            (backend.add_positive_point if positive else backend.add_negative_point)(patch.local(point))
        if self.box is not None: backend.set_box(patch.local_box(self.box))
