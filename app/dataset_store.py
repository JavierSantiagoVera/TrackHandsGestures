import os
import numpy as np
from . import config as cfg
from .crypto_store import load_encrypted, save_encrypted


class DatasetStore:
    def __init__(self, enc_path: str, seq_len: int, password: str = ""):
        self.enc_path = enc_path
        self.path = enc_path          # kept so other code can read the path
        self.seq_len = int(seq_len)
        self.fdim = int(getattr(cfg, "FEATURE_DIM", 64))
        self._password = password
        self._load_error: str = ""
        self.X: list = []
        self.y: list = []

        if password and os.path.exists(self.enc_path):
            try:
                self._load()
            except Exception as e:
                self._load_error = str(e)

    def _load(self):
        X, y = load_encrypted(self.enc_path, self._password)
        if X.shape[1:] != (self.seq_len, self.fdim):
            raise ValueError(
                f"Dataset {self.enc_path} tiene {X.shape}, "
                f"esperado (*,{self.seq_len},{self.fdim})."
            )
        self.X = [X[i] for i in range(X.shape[0])]
        self.y = [int(y[i]) for i in range(y.shape[0])]

    def add(self, seq_TxF, label: int):
        seq = np.asarray(seq_TxF, dtype=np.float32)
        if seq.shape != (self.seq_len, self.fdim):
            raise ValueError(
                f"Seq shape {seq.shape} inválida (esperado {(self.seq_len, self.fdim)})."
            )
        self.X.append(seq)
        self.y.append(int(label))

    def save(self):
        if len(self.y) == 0:
            raise ValueError("Dataset vacío.")
        X = np.stack(self.X, axis=0).astype(np.float32)
        y = np.asarray(self.y, dtype=np.int64)
        save_encrypted(self.enc_path, X, y, self._password)

    def counts(self, num_classes: int) -> list:
        c = [0] * num_classes
        for yi in self.y:
            if 0 <= yi < num_classes:
                c[yi] += 1
        return c

    def __len__(self) -> int:
        return len(self.y)
