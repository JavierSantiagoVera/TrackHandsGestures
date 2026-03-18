import time
import cv2
import mediapipe as mp
import numpy as np

from PySide6.QtCore import QThread, Signal, Slot, QMutex, QMutexLocker
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from . import config as cfg
from .mp_draw import draw_landmarks_filtered
from .features import result_to_feature
from .classifier import RealtimeClassifier


def _get_class_names_fallback():
    # Si cfg.CLASS_NAMES existe, úsalo. Si no, crea nombres genéricos.
    names = getattr(cfg, "CLASS_NAMES", None)
    if isinstance(names, (list, tuple)) and len(names) > 0:
        return list(names)
    return [f"Clase_{i}" for i in range(3)]


class HandWorker(QThread):
    frame_ready = Signal(object)              # np.ndarray BGR
    status = Signal(str)
    pred_ready = Signal(str, float, object)   # (label, conf, probs)

    # grabación de features
    rec_state = Signal(bool, str)             # (is_recording, msg)
    sample_ready = Signal(int, object)        # (label:int, seq:(T,126))

    # NUEVO: contador/progreso
    rec_progress = Signal(int, int)           # (remaining, total)

    def __init__(self, model_task_path: str, camera_index: int = 0, parent=None):
        super().__init__(parent)
        self.model_task_path = model_task_path
        self.camera_index = camera_index
        self._running = False

        # máscara (21)
        self._mask_mutex = QMutex()
        self._enabled_mask = [True] * 21

        # Config (con fallback)
        self.seq_len = int(getattr(cfg, "SEQ_LEN", 60))
        self.max_consec_lost = int(getattr(cfg, "MAX_CONSEC_LOST", 999999))
        self.min_valid_frames = int(getattr(cfg, "MIN_VALID_FRAMES", 5))
        self.detect_w = int(getattr(cfg, "DETECT_W", 0))  # 0 => no downscale
        self.predict_every = int(getattr(cfg, "PREDICT_EVERY", 2))

        self.class_names = _get_class_names_fallback()

        # LSTM realtime (cpu por defecto)
        ckpt_path = getattr(cfg, "LSTM_CKPT_PATH", "lstm_3gestures.pt")
        self.classifier = RealtimeClassifier(
            ckpt_path=ckpt_path,
            seq_len=self.seq_len,
            class_names=self.class_names
        )

        self._frame_count = 0

        # --- recording (features) ---
        self._rec_mutex = QMutex()
        self._recording = False
        self._rec_label = 0
        self._rec_buf = []              # list of (126,)
        self._last_feat = None          # last valid (126,)
        self._consec_lost = 0
        self._valid_frames = 0

        self.fdim = int(getattr(cfg, "FEATURE_DIM", 64))

    # ---------------- MASK ----------------

    @Slot(object)
    def set_enabled_mask(self, mask):
        if mask is None or len(mask) != 21:
            return
        with QMutexLocker(self._mask_mutex):
            self._enabled_mask = list(bool(x) for x in mask)

    def _get_mask_copy(self):
        with QMutexLocker(self._mask_mutex):
            return list(self._enabled_mask)

    @Slot(object)
    def set_class_names(self, names):
        if not names:
            return
        names = list(names)
        self.class_names = names
        if hasattr(self.classifier, "set_class_names"):
            self.classifier.set_class_names(names)

    # ---------------- RECORDING ----------------

    @Slot(int)
    def start_sample(self, label: int):
        """Empieza a grabar UNA muestra de SEQ_LEN frames para la clase label."""
        with QMutexLocker(self._rec_mutex):
            self._recording = True
            self._rec_label = int(label)
            self._rec_buf = []
            self._last_feat = None
            self._consec_lost = 0
            self._valid_frames = 0

        name = self.class_names[label] if 0 <= label < len(self.class_names) else str(label)
        self.rec_state.emit(True, f"Grabando muestra: {name} ({self.seq_len} frames)")
        self.rec_progress.emit(self.seq_len, self.seq_len)

    @Slot()
    def cancel_sample(self):
        with QMutexLocker(self._rec_mutex):
            self._recording = False
            self._rec_buf = []
        self.rec_progress.emit(0, self.seq_len)
        self.rec_state.emit(False, "Grabación cancelada")

    @Slot()
    def reload_classifier(self):
        ckpt_path = getattr(cfg, "LSTM_CKPT_PATH", self.classifier.ckpt_path)
        ok = self.classifier.load_checkpoint(ckpt_path)
        self.classifier.reset()
        if ok:
            self.status.emit("Modelo LSTM recargado tras el entrenamiento.")
        else:
            self.status.emit("No pude recargar el checkpoint recién entrenado.")

    def stop(self):
        self._running = False

    # ---------------- CAMERA ----------------

    def _open_camera_best_effort(self):
        backends = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("DEFAULT", 0)]
        indices = [self.camera_index] + [i for i in range(4) if i != self.camera_index]
        # Resoluciones a intentar en orden; si la cámara no soporta la primera, pasa a la siguiente
        resolutions = [(1280, 720), (960, 540), (640, 480)]
        for idx in indices:
            for name, api in backends:
                cap = cv2.VideoCapture(idx, api) if api != 0 else cv2.VideoCapture(idx)
                if cap is None or not cap.isOpened():
                    if cap:
                        cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FPS, 30)
                # Intenta resoluciones hasta obtener una válida
                for rw, rh in resolutions:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, rw)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)
                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if actual_w > 0 and actual_h > 0:
                        actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30
                        return cap, idx, name, actual_w, actual_h, actual_fps
                cap.release()
        return None, None, None, 0, 0, 0

    # ---------------- MAIN LOOP ----------------

    def run(self):
        self._running = True
        self.status.emit("Cargando modelo de manos...")

        base_options = python.BaseOptions(model_asset_path=self.model_task_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        detector = vision.HandLandmarker.create_from_options(options)

        if self.classifier.loaded:
            self.status.emit("LSTM cargada. Abriendo cámara...")
        else:
            self.status.emit("⚠️ No encontré checkpoint (pred sin entrenar). Abriendo cámara...")

        cap, used_idx, used_backend, actual_w, actual_h, actual_fps = self._open_camera_best_effort()
        if cap is None:
            self.status.emit(
                "No pude abrir la cámara.\n"
                "• Cierra Zoom/Teams/OBS.\n"
                "• Windows: Privacidad → Cámara → permitir apps de escritorio."
            )
            self._running = False
            return

        self.status.emit(
            f"Cámara OK (idx={used_idx}, backend={used_backend}, "
            f"{actual_w}×{actual_h} @{actual_fps:.0f}fps)"
        )
        t0 = time.perf_counter()

        while self._running:
            ok, frame_bgr = cap.read()
            if not ok:
                self.status.emit("Cámara: no pude leer frame.")
                break

            frame_bgr = cv2.flip(frame_bgr, 1)

            # Full-res para UI
            frame_rgb_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Downscale opcional para MediaPipe (más FPS)
            if self.detect_w and frame_rgb_full.shape[1] > self.detect_w:
                h0, w0, _ = frame_rgb_full.shape
                scale = self.detect_w / float(w0)
                frame_rgb_det = cv2.resize(
                    frame_rgb_full,
                    (self.detect_w, int(h0 * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                frame_rgb_det = frame_rgb_full

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb_det)
            timestamp_ms = int((time.perf_counter() - t0) * 1000)

            result = detector.detect_for_video(mp_image, timestamp_ms)
            has_hand = bool(result and getattr(result, "hand_landmarks", None) and len(result.hand_landmarks) > 0)

            # máscara UNA vez
            mask = self._get_mask_copy()

            # Feature enmascarado (para LSTM y grabación)
            feat_now = result_to_feature(result, enabled_mask=mask, target_dim=self.fdim)  # (F,)

            # --- Clasificación en vivo optimizada
            if has_hand:
                self.classifier.push(feat_now)
                self._frame_count += 1

                if self.classifier.ready() and (self._frame_count % self.predict_every == 0):
                    pred_idx, pred_name, probs = self.classifier.predict()

                    if probs is None or pred_idx is None:
                        # estado base
                        self.pred_ready.emit("BASE", 0.0, None)
                    else:
                        conf = float(probs[0][pred_idx])
                        self.pred_ready.emit(pred_name, conf, probs)
            else:
                # No hay mano -> estado base inmediato
                self.classifier.reset()
                self.pred_ready.emit("BASE", 0.0, None)

            # --- Grabación (contador incluido)
            with QMutexLocker(self._rec_mutex):
                if self._recording:
                    if has_hand:
                        self._last_feat = feat_now.copy()
                        self._consec_lost = 0
                        self._valid_frames += 1
                        feat_rec = feat_now
                    else:
                        # HOLD-LAST
                        self._consec_lost += 1
                        if self._last_feat is not None:
                            feat_rec = self._last_feat
                        else:
                            feat_rec = np.zeros((self.fdim,), dtype=np.float32)
                            """feat_rec = np.asarray(feat_rec, dtype=np.float32).reshape(-1)
                            if feat_rec.shape[0] != self.fdim:
                                out = np.zeros((self.fdim,), dtype=np.float32)
                                n = min(self.fdim, feat_rec.shape[0])
                                out[:n] = feat_rec[:n]
                                feat_rec = out
                            self._rec_buf.append(feat_rec)"""

                    if self._consec_lost > self.max_consec_lost:
                        self._recording = False
                        self._rec_buf = []
                        self.rec_progress.emit(0, self.seq_len)
                        self.rec_state.emit(False, "⚠️ Demasiados frames sin mano. Muestra descartada.")
                    else:
                        
                        self._rec_buf.append(feat_rec.astype(np.float32))

                        remaining = max(0, self.seq_len - len(self._rec_buf))
                        self.rec_progress.emit(remaining, self.seq_len)

                        if len(self._rec_buf) >= self.seq_len:
                            seq = np.stack(self._rec_buf[: self.seq_len], axis=0)  # (T,126)
                            ok_keep = (self._valid_frames >= self.min_valid_frames)

                            self._recording = False
                            self._rec_buf = []
                            self.rec_progress.emit(0, self.seq_len)

                            if ok_keep:
                                self.sample_ready.emit(self._rec_label, seq)
                                name = (
                                    self.class_names[self._rec_label]
                                    if 0 <= self._rec_label < len(self.class_names)
                                    else str(self._rec_label)
                                )
                                self.rec_state.emit(False, f"✅ Muestra guardada ({name}).")
                            else:
                                self.rec_state.emit(False, "⚠️ Muy pocos frames con mano real. Muestra descartada.")

            # --- Dibujo (usa frame full-res para UI; landmarks son normalizados)
            if result and result.hand_landmarks:
                annotated_rgb = draw_landmarks_filtered(frame_rgb_full, result, mask)
                out_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            else:
                out_bgr = frame_bgr
                # aviso visual si está grabando y no hay mano
                with QMutexLocker(self._rec_mutex):
                    if self._recording:
                        cv2.putText(
                            out_bgr,
                            "TRACKING LOST (HOLD LAST)",
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 0, 255),
                            3,
                            cv2.LINE_AA,
                        )

            self.frame_ready.emit(out_bgr)

        cap.release()
