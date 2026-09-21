"""大场景侧车存储：原文件只读；保存差分，导出时分块复制原始记录。"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import laspy
import numpy as np


def las_backend():
    return laspy.LazBackend.Laszip if laspy.LazBackend.Laszip.is_available() else None


def atomic_json(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".session-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class SceneSession:
    def __init__(self, path):
        self.path = Path(path)
        self.folder = self.path.with_suffix(".planar")
        with laspy.open(self.path, laz_backend=las_backend()) as reader:
            self.header = reader.header.copy()
        stat = self.path.stat()
        with self.path.open("rb") as stream:
            first = stream.read(65536)
            stream.seek(max(0, stat.st_size - 65536))
            digest = hashlib.sha256(first + stream.read()).hexdigest()
        self.fingerprint = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                            "sample_sha256": digest, "points": int(self.header.point_count),
                            "scales": self.header.scales.tolist(), "offsets": self.header.offsets.tolist()}
        self.metadata = {}
        self.map = None
        self.pending = []
        self.recovered = False

    def verify_source(self):
        stat = self.path.stat()
        if stat.st_size != self.fingerprint["size"] or stat.st_mtime_ns != self.fingerprint["mtime_ns"]:
            raise ValueError("原始点云在会话期间被修改，已停止保存/导出，避免标签错位")

    def load(self):
        metadata = self.folder / "session.json"
        labels_path = self.folder / "plane_id.npy"
        if not metadata.exists():
            if labels_path.exists():
                raise ValueError("侧车标注缺少 session.json，不能安全对应原始点序")
            return None
        self.metadata = json.loads(metadata.read_text(encoding="utf-8"))
        if self.metadata.get("source") != self.fingerprint:
            raise ValueError("侧车标注与当前原文件不匹配；请检查文件是否被替换")
        self.map = np.load(labels_path, mmap_mode="r+", allow_pickle=False)
        if self.map.dtype != np.uint32 or self.map.shape != (self.header.point_count,):
            raise ValueError("侧车 plane_id 的类型或点数错误")
        journal = self.folder / "pending.npz"
        if journal.exists():
            with np.load(journal, allow_pickle=False) as data:
                ids, values = data["ids"], data["values"]
                if np.any(ids < 0) or np.any(ids >= len(self.map)) or len(ids) != len(values):
                    raise ValueError("侧车恢复日志索引错误")
                self.map[ids] = values
            self.map.flush()
            self.recovered = True
        return np.array(self.map, dtype=np.uint32)

    def queue(self, ids, values):
        if len(ids):
            self.pending.append((np.asarray(ids, dtype=np.int64).copy(), np.asarray(values, dtype=np.uint32).copy()))

    def save(self, labels, metadata):
        self.verify_source()
        self.folder.mkdir(parents=True, exist_ok=True)
        labels_path = self.folder / "plane_id.npy"
        if self.map is None:
            temporary = self.folder / "plane_id.initial.npy"
            arr = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint32, shape=labels.shape)
            for start in range(0, len(labels), 250000):
                arr[start:start+250000] = labels[start:start+250000]
            arr.flush(); del arr
            os.replace(temporary, labels_path)
            self.map = np.load(labels_path, mmap_mode="r+", allow_pickle=False)
        elif self.pending:
            # 先落盘重做日志，再更新 memmap，崩溃后重复应用仍然安全。
            ids = np.unique(np.concatenate([item[0] for item in self.pending]))
            temporary = self.folder / "pending.tmp.npz"
            with temporary.open("wb") as stream:
                np.savez(stream, ids=ids, values=labels[ids])
                stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, self.folder / "pending.npz")
            self.map[ids] = labels[ids]
            self.map.flush()
        metadata = dict(metadata, source=self.fingerprint, version=1)
        atomic_json(self.folder / "session.json", metadata)
        self.metadata = metadata
        self.pending.clear()
        (self.folder / "pending.npz").unlink(missing_ok=True)

    def export(self, labels, destination, cancel=None):
        """始终从原始文件顺序复制全部字段，隐藏/ROI 不影响导出。"""
        destination = Path(destination)
        self.verify_source()
        if destination.resolve() == self.path.resolve():
            raise ValueError("不能覆盖原始 LAS/LAZ")
        if destination.suffix.lower() not in (".las", ".laz"):
            raise ValueError("请选择 .las 或 .laz 输出")
        fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix=destination.suffix, prefix=".plane-export-")
        os.close(fd)
        try:
            with laspy.open(self.path, laz_backend=las_backend()) as reader:
                header = reader.header.copy()
                if "plane_id" in header.point_format.extra_dimension_names:
                    header.remove_extra_dim("plane_id")
                header.add_extra_dim(laspy.ExtraBytesParams(name="plane_id", type=np.uint32,
                                                           description="Roof planar primitive ID"))
                with laspy.open(temporary, mode="w", header=header,
                                do_compress=destination.suffix.lower() == ".laz", laz_backend=las_backend()) as writer:
                    offset = 0
                    for chunk in reader.chunk_iterator(250000):
                        if cancel is not None and cancel.is_set():
                            raise InterruptedError("导出已取消，未替换目标文件")
                        converted = laspy.ScaleAwarePointRecord.zeros(len(chunk), header=header)
                        for name in chunk.array.dtype.names:
                            if name != "plane_id":
                                converted.array[name] = chunk.array[name]
                        converted.plane_id = labels[offset:offset+len(chunk)]
                        writer.write_points(converted)
                        offset += len(chunk)
                    if offset != len(labels):
                        raise ValueError("原始点数与标签不匹配")
                    if reader.evlrs:
                        writer.write_evlrs(reader.evlrs)
            if cancel is not None and cancel.is_set():
                raise InterruptedError("导出已取消，未替换目标文件")
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return destination
