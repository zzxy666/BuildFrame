"""SAM 2.1 适配器：仅显式使用时导入依赖、加载权重。"""
from contextlib import nullcontext
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
import numpy as np
from .base_backend import BaseSegmentationBackend, Prediction


@dataclass(frozen=True)
class SAMConfig:
    checkpoint: str = ""
    model_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml"
    device: str = "auto"
    patch_size: int = 1024


def availability():
    missing = [name for name in ("torch", "sam2") if find_spec(name) is None]
    return (not missing, "" if not missing else "SAM 组件未安装："+", ".join(missing)+"。请安装可选依赖 [sam]；Polygon 仍可使用。")


def choose_device(torch, requested):
    if requested not in ("auto", "cpu", "cuda"): raise ValueError("设备必须为 auto、cpu 或 cuda")
    return "cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu"


class SAM2Backend(BaseSegmentationBackend):
    def __init__(self, config):
        super().__init__()
        available, reason = availability()
        if not available: raise RuntimeError(reason)
        if not Path(config.checkpoint).is_file(): raise ValueError("SAM checkpoint 不存在，请在 SAM 设置中选择权重文件")
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.torch = torch
        self.device = choose_device(torch, config.device)
        model = build_sam2(config.model_config, config.checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(model)

    def context(self):
        return self.torch.autocast("cuda", dtype=self.torch.bfloat16) if self.device == "cuda" and self.torch.cuda.is_bf16_supported() else nullcontext()

    def set_image(self, image):
        with self.torch.inference_mode(), self.context(): self.predictor.set_image(image)

    def predict(self):
        coords = np.asarray([p for p, _ in self.points], np.float32) if self.points else None
        labels = np.asarray([v for _, v in self.points], np.int32) if self.points else None
        box = None if self.box is None else np.asarray(self.box, np.float32)
        with self.torch.inference_mode(), self.context():
            masks, scores, _ = self.predictor.predict(point_coords=coords, point_labels=labels, box=box, multimask_output=True)
        return Prediction(np.asarray(masks, bool), np.asarray(scores, float))

    def clear_image(self):
        self.predictor.reset_predictor()
        self.reset_prompts()
