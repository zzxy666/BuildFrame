"""单一 worker 调用；模型复用、仅缓存当前 patch embedding。"""
import logging
import time
from .sam2_backend import SAM2Backend


class SegmentationEngine:
    def __init__(self, factory=SAM2Backend):
        self.factory = factory
        self.backend = self.config = self.patch_key = None
        self.current_patch = None

    def run(self, config, photo, patch, prompts, progress=lambda value: None):
        if self.backend is None or self.config != config:
            if self.backend is not None: self.backend.clear_image()
            self.backend = None
            self.patch_key = None
            progress("Loading")
            self.backend = self.factory(config)
            self.config = config
        if patch is None:
            self.clear_image()
            return None
        key = (id(photo), id(patch))
        if self.patch_key != key:
            if self.current_patch is not None:
                self.current_patch.image = None;self.current_patch.encoded = False
            self.backend.clear_image();self.patch_key = None
            progress("Encoding")
            start = time.perf_counter()
            rgba, bounds = photo.read_window(patch.bounds, max_size=2048)
            if tuple(bounds) != patch.bounds or rgba.shape[:2] != (patch.height, patch.width):
                raise ValueError("SAM patch 必须使用 GeoTIFF 原生像素")
            patch.image = rgba[:, :, :3].copy()
            self.backend.set_image(patch.image)
            self.patch_key = key
            self.current_patch = patch
            patch.encoded = True
            logging.debug("SAM patch=%sx%s device=%s image encode=%.1f ms", patch.width, patch.height, self.backend.device, (time.perf_counter()-start)*1000)
        if not prompts.events: return None
        prompts.apply(self.backend, patch)
        progress("Predicting")
        start = time.perf_counter()
        prediction = self.backend.predict()
        if prediction.masks.ndim != 3 or prediction.masks.shape[1:] != (patch.height, patch.width):
            raise ValueError("SAM 输出 mask 形状无效")
        prediction.best()
        logging.debug("SAM prompt decode=%.1f ms positive=%s negative=%s mask pixels=%s", (time.perf_counter()-start)*1000, sum(k == 1 for _, k in prompts.points), sum(k == 0 for _, k in prompts.points), int(prediction.masks[prediction.best()].sum()))
        return prediction

    def clear_image(self):
        if self.backend is not None: self.backend.clear_image()
        if self.current_patch is not None:
            self.current_patch.image = None;self.current_patch.encoded = False
        self.current_patch = None
        self.patch_key = None
