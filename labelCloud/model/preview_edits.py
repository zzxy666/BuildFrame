"""候选区的手工修正，独立于持久标签及其撤销记录。"""
import numpy as np


class PreviewStateChange:
    """预览算法切换的状态记录，与手工点差分共用撤销顺序。"""
    def __init__(self, before, after, restore):
        self.before, self.after, self.restore = before, after, restore
        def size(value):
            if isinstance(value, np.ndarray): return value.nbytes
            if isinstance(value, dict): return sum(size(v) for v in value.values())
            if isinstance(value, (list, tuple)): return sum(size(v) for v in value)
            return 0
        self.nbytes = size(before) + size(after)


class PreviewEdits:
    def __init__(self):
        self.added = np.empty(0, np.int64)
        self.removed = np.empty(0, np.int64)
        self.history = []
        self.future = []

    def edit(self, ids, add):
        ids = np.unique(ids)
        old = np.zeros(len(ids), np.int8)
        old[np.isin(ids,self.added)] = 1
        old[np.isin(ids,self.removed)] = -1
        value = 1 if add else -1
        changed = old != value
        ids, old = ids[changed], old[changed]
        if not len(ids): return
        self.history.append((ids,old,np.full(len(ids),value,np.int8)))
        self.future.clear()
        self._apply(ids,np.full(len(ids),value,np.int8))
        # 差分历史受限，避免长时间修正无限占用内存。
        while sum(r.nbytes if isinstance(r, PreviewStateChange) else sum(a.nbytes for a in r) for r in self.history)>128*1024*1024:
            self.history.pop(0)

    def _apply(self, ids, values):
        self.added=np.union1d(np.setdiff1d(self.added,ids),ids[values==1])
        self.removed=np.union1d(np.setdiff1d(self.removed,ids),ids[values==-1])

    def replace(self,before,after):
        """过滤/恢复候选作为一次差分操作，仍可与手工增删交替 Undo。"""
        added=np.setdiff1d(after,before);removed=np.setdiff1d(before,after)
        ids=np.union1d(added,removed)
        if not len(ids): return
        old=np.zeros(len(ids),np.int8);old[np.isin(ids,self.added)]=1;old[np.isin(ids,self.removed)]=-1
        values=np.where(np.isin(ids,added),1,-1).astype(np.int8)
        self.history.append((ids,old,values));self.future.clear();self._apply(ids,values)
        while sum(r.nbytes if isinstance(r, PreviewStateChange) else sum(a.nbytes for a in r) for r in self.history)>128*1024*1024:
            self.history.pop(0)

    def record_state(self, before, after, restore):
        self.history.append(PreviewStateChange(before, after, restore))
        self.future.clear()
        while len(self.history)>1 and sum(r.nbytes if isinstance(r, PreviewStateChange) else sum(a.nbytes for a in r) for r in self.history)>128*1024*1024:
            self.history.pop(0)

    def undo(self):
        if not self.history: return False
        record=self.history.pop()
        if isinstance(record, PreviewStateChange): record.restore(record.before)
        else: self._apply(record[0],record[1])
        self.future.append(record)
        return True

    def redo(self):
        if not self.future: return False
        record=self.future.pop()
        if isinstance(record, PreviewStateChange): record.restore(record.after)
        else: self._apply(record[0],record[2])
        self.history.append(record)
        return True

    def compose(self, strict, edge):
        yellow=np.setdiff1d(np.union1d(strict,self.added),self.removed)
        orange=np.setdiff1d(edge,np.union1d(yellow,self.removed))
        return yellow,orange
