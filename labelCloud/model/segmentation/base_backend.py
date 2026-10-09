"""与 GUI、模型框架无关的局部图像分割协议。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class Prediction:
    masks: np.ndarray
    scores: np.ndarray

    def best(self):
        if len(self.masks) == 0 or len(self.masks) != len(self.scores):
            raise ValueError("分割后端没有返回有效 mask")
        return int(np.nanargmax(self.scores))


class BaseSegmentationBackend(ABC):
    device = "unknown"

    def __init__(self):
        self.reset_prompts()

    def add_positive_point(self, point): self.points.append((tuple(point), 1))
    def add_negative_point(self, point): self.points.append((tuple(point), 0))
    def set_box(self, box): self.box = tuple(box)
    def reset_prompts(self): self.points, self.box = [], None

    @abstractmethod
    def set_image(self, image): pass

    @abstractmethod
    def predict(self): pass

    @abstractmethod
    def clear_image(self): pass
