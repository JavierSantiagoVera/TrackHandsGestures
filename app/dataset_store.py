# app/dataset_store.py
import os
import numpy as np
from . import config as cfg

class DatasetStore:
    def __init__(self, path: str, seq_len: int):
        self.path = path
        self.seq_len = int(seq_len)
        self.fdim = int(getattr(cfg, "FEATURE_DIM", 64))
        self.X = []
        self.y = []
        if os.path.exists(self.path):
            self.load()

    def load(self):
        d = np.load(self.path, allow_pickle=False)
        X = d["X"].astype(np.float32)
        y = d["y"].astype(np.int64)

        if X.shape[1:] != (self.seq_len, self.fdim):
            raise ValueError(
                f"Dataset {self.path} tiene {X.shape} pero se espera (*,{self.seq_len},{self.fdim})."
            )

        self.X = [X[i] for i in range(X.shape[0])]
        self.y = [int(y[i]) for i in range(y.shape[0])]

    def add(self, seq_TxF, label: int):
        seq = np.asarray(seq_TxF, dtype=np.float32)
        if seq.shape != (self.seq_len, self.fdim):
            raise ValueError(f"Seq shape {seq.shape} inválida (esperado {(self.seq_len, self.fdim)}).")
        self.X.append(seq)
        self.y.append(int(label))

    def save(self):
        if len(self.y) == 0:
            raise ValueError("Dataset vacío.")
        X = np.stack(self.X, axis=0).astype(np.float32)
        y = np.asarray(self.y, dtype=np.int64)
        np.savez_compressed(self.path, X=X, y=y)

    def counts(self, num_classes: int):
        c = [0] * num_classes
        for yi in self.y:
            if 0 <= yi < num_classes:
                c[yi] += 1
        return c

    def __len__(self):
        return len(self.y)