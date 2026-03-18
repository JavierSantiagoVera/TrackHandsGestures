# app/classifier.py
import numpy as np
import torch
import torch.nn.functional as F

from . import config as cfg
from .model import TemporalCNN

class RealtimeClassifier:
    def __init__(self, ckpt_path: str, seq_len: int, class_names, device: str = "cpu"):
        self.seq_len = int(seq_len)
        self.fdim = int(getattr(cfg, "FEATURE_DIM", 64))
        self.class_names = list(class_names)

        self.buf = np.zeros((self.seq_len, self.fdim), dtype=np.float32)
        self.ptr = 0
        self.filled = 0

        self.device = torch.device(device)
        self.model = TemporalCNN(feature_dim=self.fdim, num_classes=len(self.class_names), hidden=48, dropout=0.0)
        self.model.to(self.device).eval()

        self.ckpt_path = ckpt_path
        self.loaded = False
        self.load_checkpoint(ckpt_path)

        # hold
        fps = int(getattr(cfg, "FPS_EST", 30))
        hold_s = float(getattr(cfg, "HOLD_SECONDS", 2.0))
        self.hold_frames = max(1, int(fps * hold_s))
        self.hold_left = 0
        self.last_idx = None
        self.last_probs = None

    def reset(self):
        self.ptr = 0
        self.filled = 0
        self.buf.fill(0)
        self.hold_left = 0
        self.last_idx = None
        self.last_probs = None

    def set_class_names(self, class_names):
        if not class_names:
            return
        names = list(class_names)
        if len(names) != len(self.class_names):
            names = names[: len(self.class_names)]
        self.class_names = names

    def push(self, feat_F: np.ndarray):
        f = np.asarray(feat_F, dtype=np.float32).reshape(-1)
        if f.shape[0] != self.fdim:
            # blindaje: recorta/pad
            out = np.zeros((self.fdim,), dtype=np.float32)
            n = min(self.fdim, f.shape[0])
            out[:n] = f[:n]
            f = out

        self.buf[self.ptr] = f
        self.ptr = (self.ptr + 1) % self.seq_len
        self.filled = min(self.seq_len, self.filled + 1)

    def ready(self):
        return self.filled == self.seq_len

    def _sequence(self):
        if self.ptr == 0:
            x_np = self.buf
        else:
            x_np = np.concatenate((self.buf[self.ptr:], self.buf[:self.ptr]), axis=0)
        return x_np  # (T,F)

    def load_checkpoint(self, ckpt_path: str = None):
        if ckpt_path:
            self.ckpt_path = ckpt_path

        path = self.ckpt_path
        self.loaded = False
        if not path:
            return False

        try:
            state = torch.load(path, map_location=self.device)
            sd = state["model"] if isinstance(state, dict) and "model" in state else state
            self.model.load_state_dict(sd, strict=True)
            self.loaded = True
        except Exception:
            self.loaded = False
        return self.loaded

    @torch.inference_mode()
    def predict(self):
        if not self.ready():
            return None, "sin prediccion", None

        x_np = self._sequence()
        x = torch.from_numpy(x_np).unsqueeze(0).to(self.device)  # (1,T,F)
        logits = self.model(x)  # (1,C)
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()  # (1,C)

        pred_idx = int(np.argmax(probs[0]))
        conf = float(probs[0, pred_idx])
        thr = float(getattr(cfg, "CONF_THRESH", 0.75))

        if conf >= thr:
            self.last_idx = pred_idx
            self.last_probs = probs
            self.hold_left = self.hold_frames
            return pred_idx, self.class_names[pred_idx], probs

        # baja confianza => hold si hay pred previa reciente
        if self.hold_left > 0 and self.last_idx is not None:
            self.hold_left -= 1
            return self.last_idx, self.class_names[self.last_idx], self.last_probs

        return None, "sin prediccion", None
