# app/train_worker.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PySide6.QtCore import QThread, Signal

from . import config as cfg
from .model import TemporalCNN

class _NpzSeqDataset(Dataset):
    def __init__(self, npz_path: str, seq_len: int, fdim: int):
        d = np.load(npz_path, allow_pickle=False)
        self.X = d["X"].astype(np.float32)  # (N,T,F)
        self.y = d["y"].astype(np.int64)    # (N,)
        if self.X.shape[1:] != (seq_len, fdim):
            raise ValueError(f"NPZ X tiene {self.X.shape}, esperado (*,{seq_len},{fdim}).")

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

class TrainWorker(QThread):
    status = Signal(str)
    done = Signal(bool, str)

    def __init__(self, npz_path: str, ckpt_path: str, num_classes: int = 3, parent=None):
        super().__init__(parent)
        self.npz_path = npz_path
        self.ckpt_path = ckpt_path
        self.num_classes = int(num_classes)

    def run(self):
        try:
            seq_len = int(getattr(cfg, "SEQ_LEN", 30))
            fdim = int(getattr(cfg, "FEATURE_DIM", 64))

            ds = _NpzSeqDataset(self.npz_path, seq_len=seq_len, fdim=fdim)
            if len(ds) < 6:
                self.done.emit(False, f"Dataset muy pequeño (N={len(ds)}). Graba más muestras.")
                return

            batch = int(getattr(cfg, "TRAIN_BATCH", 64))
            epochs = int(getattr(cfg, "TRAIN_EPOCHS", 40))
            lr = float(getattr(cfg, "TRAIN_LR", 1e-3))
            wd = float(getattr(cfg, "TRAIN_WD", 1e-4))
            patience = int(getattr(cfg, "TRAIN_PATIENCE", 6))

            dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)

            device = torch.device("cpu")
            torch.set_num_threads(1)  # sube a 2–4 si quieres más velocidad, depende tu CPU

            model = TemporalCNN(feature_dim=fdim, num_classes=self.num_classes, hidden=48, dropout=0.15).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            crit = nn.CrossEntropyLoss()

            best_loss = float("inf")
            best_state = None
            bad = 0

            self.status.emit(f"Entrenando CNN temporal en CPU... N={len(ds)} | batch={batch} | T={seq_len} | F={fdim}")

            model.train()
            for ep in range(epochs):
                loss_sum, total, correct = 0.0, 0, 0

                for X, y in dl:
                    X = torch.from_numpy(X.numpy()) if hasattr(X, "numpy") else X
                    X = X.to(device)
                    y = y.to(device)

                    opt.zero_grad(set_to_none=True)
                    logits = model(X)
                    loss = crit(logits, y)
                    loss.backward()
                    opt.step()

                    bs = y.size(0)
                    loss_sum += float(loss.item()) * bs
                    total += bs
                    correct += int((torch.argmax(logits, dim=-1) == y).sum().item())

                avg_loss = loss_sum / max(1, total)
                acc = correct / max(1, total)
                self.status.emit(f"Epoch {ep+1}/{epochs} | loss={avg_loss:.4f} acc={acc:.3f}")

                # early stopping por loss de train (simple y rápido)
                if avg_loss + 1e-5 < best_loss:
                    best_loss = avg_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        self.status.emit(f"Early stop: sin mejora en {patience} epochs.")
                        break

            if best_state is not None:
                model.load_state_dict(best_state)

            torch.save(
                {
                    "model": model.state_dict(),
                    "meta": {
                        "seq_len": seq_len,
                        "feature_dim": fdim,
                        "num_classes": self.num_classes,
                        "arch": "TemporalCNN",
                    },
                },
                self.ckpt_path,
            )
            self.done.emit(True, f"Modelo guardado: {self.ckpt_path}")

        except Exception as e:
            self.done.emit(False, f"Error entrenando: {e}")