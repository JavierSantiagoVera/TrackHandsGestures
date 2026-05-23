import os

import numpy as np
from PySide6.QtCore import Qt, Slot, QMetaObject, Q_ARG, QTimer, QEvent
from PySide6.QtGui import QFont, QGuiApplication, QPixmap
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget, QFrame,
)

from .config import (
    MODEL_TASK_PATH, SEQ_LEN, DATASET_ENC_PATH,
    LSTM_CKPT_PATH, DEFAULT_CLASSES,
    load_ui_classes, class_names_from_meta, save_ui_classes,
)
from .qt_utils import bgr_to_pixmap
from .landmark_widget import LandmarkToggleWidget
from .rec_border import RecBorderWidget, BORDER_W
from .worker import HandWorker
from .dataset_store import DatasetStore
from .train_worker import TrainWorker


class MainWindow(QWidget):
    def __init__(self, password: str = ""):
        super().__init__()
        self._password = password
        self.setWindowTitle("IA para Inclusión · LSC")

        screen = QGuiApplication.primaryScreen().availableGeometry()
        sw, sh = screen.width(), screen.height()
        self._scale = min(sw / 1920.0, sh / 1080.0)
        s = self._scale

        right_w  = max(220, int(sw * 0.20))
        lm_size  = max(160, int(sh * 0.22))
        icon_sz  = max(80,  int(sh * 0.13))

        # ── Dataset ────────────────────────────────────────────
        self.ds = DatasetStore(DATASET_ENC_PATH, seq_len=SEQ_LEN, password=password)

        # ── Video area with progress border ───────────────────
        self.rec_border = RecBorderWidget()
        self.rec_border.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.rec_border.installEventFilter(self)

        self.video = self.rec_border.video_label

        # Prediction icon overlay (top-right corner of video)
        self.icon_label = QLabel("", self.rec_border)
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setFixedSize(icon_sz, icon_sz)
        self.icon_label.setStyleSheet(
            "border: 2px solid rgba(255,255,255,160);"
            "background: rgba(0,0,0,140);"
            "border-radius: 8px;"
        )
        self.icon_label.setVisible(False)
        self.icon_label.raise_()

        # Countdown overlay (full-size, centered)
        self.countdown_overlay = QLabel("", self.rec_border)
        self.countdown_overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.countdown_overlay.setAlignment(Qt.AlignCenter)
        self.countdown_overlay.setStyleSheet(
            "background: rgba(0,0,0,160);"
            "color: #F59E0B;"
            "font-size: 96px;"
            "font-weight: bold;"
        )
        self.countdown_overlay.setVisible(False)
        self.countdown_overlay.raise_()

        # ── Right panel ────────────────────────────────────────
        self.landmarks_view = LandmarkToggleWidget()
        self.landmarks_view.setFixedWidth(right_w - 20)
        self.landmarks_view.setFixedHeight(lm_size)

        # Prediction display
        self.pred_label = QLabel("—")
        self.pred_label.setAlignment(Qt.AlignCenter)
        f_pred = QFont("Segoe UI", max(18, int(24 * s)), QFont.Bold)
        self.pred_label.setFont(f_pred)
        self.pred_label.setStyleSheet("color: #10B981; padding: 4px 0;")

        self.conf_label = QLabel("")
        self.conf_label.setAlignment(Qt.AlignCenter)
        self.conf_label.setStyleSheet("color: #64748B; font-size: 12px;")

        # ── Class meta / names ─────────────────────────────────
        self.class_meta = load_ui_classes()
        self._ensure_class_meta(3)
        self.class_meta = self.class_meta[:3]
        self.class_names = class_names_from_meta(self.class_meta)[:3]
        self._icon_cache: dict = {}
        self.class_controls: list = []

        while len(self.class_names) < 3:
            self.class_names.append(f"gesto {len(self.class_names) + 1}")

        # ── Dataset count label ────────────────────────────────
        self.counts_label = QLabel(self._counts_text())
        self.counts_label.setAlignment(Qt.AlignCenter)
        self.counts_label.setWordWrap(True)
        self.counts_label.setStyleSheet("color: #64748B; font-size: 12px; padding: 2px 0;")

        # ── Status label ───────────────────────────────────────
        self.status = QLabel("")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #94A3B8; font-size: 12px;")

        # ── Pre-record timer state ─────────────────────────────
        self.pre_record_timer = None
        self.pre_record_target = None
        self.pre_record_seconds = 0

        # ── Buttons ────────────────────────────────────────────
        btn_h = max(28, int(36 * s))

        self.btn_add0 = QPushButton(f"⏺  {self.class_names[0]}  ({SEQ_LEN} f)")
        self.btn_add1 = QPushButton(f"⏺  {self.class_names[1]}  ({SEQ_LEN} f)")
        self.btn_add2 = QPushButton(f"⏺  {self.class_names[2]}  ({SEQ_LEN} f)")
        self.btn_add0.setObjectName("btn_add0")
        self.btn_add1.setObjectName("btn_add1")
        self.btn_add2.setObjectName("btn_add2")

        self.btn_cancel = QPushButton("✕  Cancelar grabación")
        self.btn_cancel.setObjectName("btn_cancel")
        self.btn_cancel.setEnabled(False)

        self.btn_reset_ds = QPushButton("🗑  Resetear dataset")
        self.btn_reset_ds.setObjectName("btn_danger")

        self.btn_train = QPushButton("⚡  Entrenar modelo")
        self.btn_train.setObjectName("btn_train")

        for btn in (
            self.btn_add0, self.btn_add1, self.btn_add2,
            self.btn_cancel, self.btn_reset_ds, self.btn_train,
        ):
            btn.setMinimumHeight(btn_h)

        # ── Right panel layout ─────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(6)
        right.setContentsMargins(8, 10, 8, 10)

        # Landmarks
        self.lm_title = QLabel("Landmarks activos")
        self.lm_title.setAlignment(Qt.AlignCenter)
        self.lm_title.setStyleSheet("color: #64748B; font-size: 11px; font-weight: bold; letter-spacing: 1px;")
        right.addWidget(self.lm_title)
        right.addWidget(self.landmarks_view, 0, Qt.AlignHCenter)

        # Separator
        sep1 = QFrame(); sep1.setFrameShape(QFrame.HLine)
        right.addWidget(sep1)

        # Prediction
        right.addWidget(self.pred_label)
        right.addWidget(self.conf_label)

        # Separator
        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine)
        right.addWidget(sep2)

        # Class editors
        self.class_title = QLabel("Clases")
        self.class_title.setAlignment(Qt.AlignCenter)
        self.class_title.setStyleSheet("color: #64748B; font-size: 11px; font-weight: bold; letter-spacing: 1px;")
        right.addWidget(self.class_title)

        for idx in range(3):
            editor = self._create_class_editor(idx)
            right.addLayout(editor["layout"])
            self.class_controls.append(editor)

        right.addWidget(self.counts_label)

        # Separator
        sep3 = QFrame(); sep3.setFrameShape(QFrame.HLine)
        right.addWidget(sep3)

        # Record buttons
        right.addWidget(self.btn_add0)
        right.addWidget(self.btn_add1)
        right.addWidget(self.btn_add2)
        right.addWidget(self.btn_cancel)
        right.addWidget(self.btn_reset_ds)

        # Separator
        sep4 = QFrame(); sep4.setFrameShape(QFrame.HLine)
        right.addWidget(sep4)

        right.addWidget(self.btn_train)
        right.addWidget(self.status, 1)

        self.right_panel = QWidget()
        self.right_panel.setLayout(right)
        self.right_panel.setFixedWidth(right_w)
        self.right_panel.setStyleSheet("background-color: #0B1120; border-left: 1px solid #1E293B;")

        # ── Root layout ────────────────────────────────────────
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.rec_border, 1)
        root.addWidget(self.right_panel, 0)

        # ── Worker ─────────────────────────────────────────────
        self.worker = HandWorker(model_task_path=MODEL_TASK_PATH, camera_index=0)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status.connect(self.on_status)
        self.worker.pred_ready.connect(self.on_pred)
        self.landmarks_view.mask_changed.connect(self.worker.set_enabled_mask)
        self.worker.set_enabled_mask(self.landmarks_view.get_mask())
        self.worker.sample_ready.connect(self.on_sample_ready)
        self.worker.rec_state.connect(self.on_rec_state)
        self.worker.rec_progress.connect(self.on_rec_progress)

        # ── Button connections ─────────────────────────────────
        self.btn_add0.clicked.connect(lambda: self._schedule_recording(0))
        self.btn_add1.clicked.connect(lambda: self._schedule_recording(1))
        self.btn_add2.clicked.connect(lambda: self._schedule_recording(2))
        self.btn_cancel.clicked.connect(self.on_cancel_clicked)
        self.btn_reset_ds.clicked.connect(self.reset_dataset)

        self.train_worker = None
        self.btn_train.clicked.connect(self.train_model)

        self.worker.start()
        self._push_class_names_to_worker()

    # ── Responsive scaling ─────────────────────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if getattr(self, "right_panel", None):
            self._apply_scale()

    def _apply_scale(self):
        W, H = self.width(), self.height()
        if W < 100 or H < 100:
            return
        s = min(W / 1280.0, H / 720.0)

        right_w = max(180, int(W * 0.20))
        lm_size = max(130, int(H * 0.22))
        icon_sz = max(80,  int(H * 0.13))
        btn_h   = max(24,  int(34 * s))

        self.right_panel.setFixedWidth(right_w)
        self.landmarks_view.setFixedWidth(right_w - 20)
        self.landmarks_view.setFixedHeight(lm_size)
        self.icon_label.setFixedSize(icon_sz, icon_sz)

        pred_sz = max(14, int(22 * s))
        self.pred_label.setFont(QFont("Segoe UI", pred_sz, QFont.Bold))

        sm = max(9,  int(11 * s))
        xs = max(8,  int(10 * s))
        cd = max(48, int(96 * s))

        self.conf_label.setStyleSheet(f"color: #64748B; font-size: {sm}px;")
        self.status.setStyleSheet(f"color: #94A3B8; font-size: {sm}px;")
        self.counts_label.setStyleSheet(f"color: #64748B; font-size: {sm}px; padding: 2px 0;")
        self.lm_title.setStyleSheet(
            f"color: #64748B; font-size: {sm}px; font-weight: bold; letter-spacing: 1px;"
        )
        self.class_title.setStyleSheet(
            f"color: #64748B; font-size: {sm}px; font-weight: bold; letter-spacing: 1px;"
        )
        self.countdown_overlay.setStyleSheet(
            f"background: rgba(0,0,0,160); color: #F59E0B;"
            f"font-size: {cd}px; font-weight: bold;"
        )

        for btn in (self.btn_add0, self.btn_add1, self.btn_add2,
                    self.btn_cancel, self.btn_reset_ds, self.btn_train):
            btn.setMinimumHeight(btn_h)

        for ctrl in self.class_controls:
            ctrl["icon_label"].setStyleSheet(f"font-size: {xs}px; color: #475569;")
            ctrl["icon_btn"].setStyleSheet(f"font-size: {xs}px; padding: 2px 6px;")
            ctrl["clear_btn"].setStyleSheet(f"font-size: {xs}px; padding: 2px 6px;")
            ctrl["reset_btn"].setFixedWidth(max(24, int(34 * s)))

        self._reposition_icon()

    # ── Overlay positioning ────────────────────────────────────

    def eventFilter(self, obj, event):
        if obj is self.rec_border and event.type() == QEvent.Resize:
            self._reposition_icon()
            self.countdown_overlay.setGeometry(0, 0, obj.width(), obj.height())
        return super().eventFilter(obj, event)

    def _reposition_icon(self):
        iw = self.icon_label.width()
        margin = 8

        pm = self.video.pixmap()
        if pm and not pm.isNull():
            # Actual displayed image size (scaled with KeepAspectRatio)
            lw, lh = self.video.width(), self.video.height()
            pm_w, pm_h = pm.width(), pm.height()
            # Centering offset of the image within video_label (Qt.AlignCenter)
            dx = (lw - pm_w) // 2
            dy = (lh - pm_h) // 2
            # Translate to rec_border coordinates using video_label's position
            img_right = self.video.x() + dx + pm_w
            img_top   = self.video.y() + dy
        else:
            img_right = self.rec_border.width() - BORDER_W
            img_top   = BORDER_W

        self.icon_label.move(img_right - iw - margin, img_top + margin)

    # ── Dataset reset ──────────────────────────────────────────

    @Slot()
    def reset_dataset(self):
        reply = QMessageBox.question(
            self,
            "Resetear dataset",
            "¿Seguro que quieres borrar TODAS las muestras?\nEsta acción no se puede deshacer.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.ds.X = []
        self.ds.y = []
        if os.path.exists(DATASET_ENC_PATH):
            os.remove(DATASET_ENC_PATH)
        self.counts_label.setText(self._counts_text())
        self.status.setText("Dataset reseteado.")

    def _counts_text(self) -> str:
        if self.ds._load_error:
            return f"⚠️ Dataset no cargado: {self.ds._load_error}"
        counts = self.ds.counts(num_classes=3)
        names = self.class_names
        return (
            f"Total: {len(self.ds)} muestras  ·  "
            f"{names[0]}: {counts[0]}  {names[1]}: {counts[1]}  {names[2]}: {counts[2]}"
        )

    # ── Record buttons state ───────────────────────────────────

    def _set_record_buttons_enabled(self, enabled: bool):
        self.btn_add0.setEnabled(enabled)
        self.btn_add1.setEnabled(enabled)
        self.btn_add2.setEnabled(enabled)

    # ── Class meta helpers ─────────────────────────────────────

    def _ensure_class_meta(self, n: int):
        if not isinstance(self.class_meta, list):
            self.class_meta = []
        for i in range(len(self.class_meta)):
            if not isinstance(self.class_meta[i], dict):
                self.class_meta[i] = {"name": f"gesto {i + 1}", "icon": ""}
            self.class_meta[i].setdefault("name", f"gesto {i + 1}")
            self.class_meta[i].setdefault("icon", "")
        while len(self.class_meta) < n:
            self.class_meta.append({"name": f"gesto {len(self.class_meta) + 1}", "icon": ""})

    def _default_name(self, idx: int) -> str:
        if 0 <= idx < len(DEFAULT_CLASSES):
            return DEFAULT_CLASSES[idx].get("name", f"gesto {idx + 1}")
        return f"gesto {idx + 1}"

    def _icon_path_display(self, idx: int) -> str:
        icon_path = ""
        if 0 <= idx < len(self.class_meta):
            icon_path = self.class_meta[idx].get("icon", "") or ""
        return os.path.basename(icon_path) if icon_path else "Sin icono"

    def _create_class_editor(self, idx: int) -> dict:
        s = self._scale
        wrapper = QVBoxLayout()
        wrapper.setContentsMargins(0, 2, 0, 2)
        wrapper.setSpacing(2)

        name_row = QHBoxLayout()
        name_lbl = QLabel(f"{idx + 1}.")
        name_lbl.setFixedWidth(16)
        name_lbl.setStyleSheet("color: #64748B;")
        name_edit = QLineEdit(self.class_names[idx] if idx < len(self.class_names) else f"Clase_{idx}")
        name_edit.setMaxLength(40)
        reset_btn = QPushButton("↺")
        reset_btn.setObjectName("btn_reset_name")
        reset_btn.setFixedWidth(max(28, int(34 * s)))
        reset_btn.clicked.connect(lambda _, i=idx: self._reset_class_name(i))
        name_edit.editingFinished.connect(
            lambda idx=idx, edit=name_edit: self._on_class_name_changed(idx, edit.text())
        )
        name_row.addWidget(name_lbl)
        name_row.addWidget(name_edit, 1)
        name_row.addWidget(reset_btn)

        icon_row = QHBoxLayout()
        icon_row.setContentsMargins(20, 0, 0, 0)
        icon_btn = QPushButton("Icono…")
        icon_btn.setStyleSheet("font-size: 11px; padding: 2px 6px;")
        icon_btn.clicked.connect(lambda _, i=idx: self._choose_icon(i))
        clear_btn = QPushButton("✕")
        clear_btn.setStyleSheet("font-size: 11px; padding: 2px 6px;")
        clear_btn.setFixedWidth(28)
        clear_btn.clicked.connect(lambda _, i=idx: self._clear_icon(i))
        icon_info = QLabel(self._icon_path_display(idx))
        icon_info.setWordWrap(True)
        icon_info.setStyleSheet("font-size: 10px; color: #475569;")
        icon_row.addWidget(icon_btn)
        icon_row.addWidget(clear_btn)
        icon_row.addWidget(icon_info, 1)

        wrapper.addLayout(name_row)
        wrapper.addLayout(icon_row)

        return {
            "layout": wrapper, "name_edit": name_edit, "icon_label": icon_info,
            "icon_btn": icon_btn, "clear_btn": clear_btn, "reset_btn": reset_btn,
        }

    def _on_class_name_changed(self, idx: int, new_value: str):
        if not (0 <= idx < len(self.class_meta)):
            return
        name = (new_value or "").strip() or self._default_name(idx)
        self.class_meta[idx]["name"] = name
        self.class_controls[idx]["name_edit"].setText(name)
        self._on_class_meta_updated()

    def _reset_class_name(self, idx: int):
        if not (0 <= idx < len(self.class_meta)):
            return
        name = self._default_name(idx)
        self.class_meta[idx]["name"] = name
        self.class_controls[idx]["name_edit"].setText(name)
        self._on_class_meta_updated()

    def _choose_icon(self, idx: int):
        if not (0 <= idx < len(self.class_meta)):
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar icono", "",
            "Imágenes (*.png *.jpg *.jpeg *.bmp *.gif);;Todos los archivos (*)",
        )
        if not file_path:
            return
        current = self.class_meta[idx].get("icon", "")
        self._icon_cache.pop(current, None)
        self.class_meta[idx]["icon"] = file_path
        self.class_controls[idx]["icon_label"].setText(self._icon_path_display(idx))
        self._on_class_meta_updated()

    def _clear_icon(self, idx: int):
        if not (0 <= idx < len(self.class_meta)):
            return
        current = self.class_meta[idx].get("icon", "")
        self._icon_cache.pop(current, None)
        self.class_meta[idx]["icon"] = ""
        self.class_controls[idx]["icon_label"].setText(self._icon_path_display(idx))
        self._on_class_meta_updated()

    def _on_class_meta_updated(self):
        self.class_names = class_names_from_meta(self.class_meta)[:3]
        self.btn_add0.setText(f"⏺  {self.class_names[0]}  ({SEQ_LEN} f)")
        self.btn_add1.setText(f"⏺  {self.class_names[1]}  ({SEQ_LEN} f)")
        self.btn_add2.setText(f"⏺  {self.class_names[2]}  ({SEQ_LEN} f)")
        self.counts_label.setText(self._counts_text())
        self._push_class_names_to_worker()
        try:
            save_ui_classes(self.class_meta)
        except Exception as exc:
            self.status.setText(f"No pude guardar las clases: {exc}")

    def _push_class_names_to_worker(self):
        if not getattr(self, "worker", None):
            return
        names = list(self.class_names)
        try:
            QMetaObject.invokeMethod(
                self.worker, "set_class_names",
                Qt.QueuedConnection, Q_ARG(object, names),
            )
        except Exception:
            try:
                self.worker.set_class_names(names)
            except Exception:
                pass

    # ── Pre-record countdown ───────────────────────────────────

    def _schedule_recording(self, label_idx: int):
        if self.pre_record_timer is not None:
            self.status.setText("Ya hay una grabación programada.")
            return
        if not (self.worker and self.worker.isRunning()):
            QMessageBox.warning(self, "Grabación", "El worker de cámara no está activo.")
            return
        if not (0 <= label_idx < len(self.class_names)):
            return

        self.pre_record_target = label_idx
        self.pre_record_seconds = 3
        self._set_record_buttons_enabled(False)
        self.btn_cancel.setEnabled(True)
        self.rec_border.set_countdown()
        self._show_countdown(self.pre_record_seconds)
        self.status.setText(f"Iniciando en {self.pre_record_seconds}s…")

        self.pre_record_timer = QTimer(self)
        self.pre_record_timer.timeout.connect(self._handle_pre_record_tick)
        self.pre_record_timer.start(1000)

    def _show_countdown(self, n: int):
        self.countdown_overlay.setText(str(n))
        self.countdown_overlay.setGeometry(0, 0, self.rec_border.width(), self.rec_border.height())
        self.countdown_overlay.setVisible(True)
        self.countdown_overlay.raise_()

    def _handle_pre_record_tick(self):
        if self.pre_record_timer is None:
            return
        self.pre_record_seconds -= 1
        if self.pre_record_seconds > 0:
            self._show_countdown(self.pre_record_seconds)
            self.status.setText(f"Iniciando en {self.pre_record_seconds}s…")
            return

        target = self.pre_record_target
        self.pre_record_target = None
        self.countdown_overlay.setVisible(False)
        self._stop_pre_record_timer()

        if target is not None and self.worker:
            self.status.setText("Grabando…")
            self.worker.start_sample(target)

    def _stop_pre_record_timer(self):
        if self.pre_record_timer is not None:
            self.pre_record_timer.stop()
            self.pre_record_timer.deleteLater()
            self.pre_record_timer = None

    def _abort_pre_record(self, show_status: bool = True):
        if self.pre_record_timer is None:
            return
        self._stop_pre_record_timer()
        self.pre_record_target = None
        self.pre_record_seconds = 0
        self.countdown_overlay.setVisible(False)
        self.rec_border.set_idle()
        if show_status:
            self.status.setText("Grabación cancelada.")
        self._set_record_buttons_enabled(True)
        self.btn_cancel.setEnabled(False)

    @Slot()
    def on_cancel_clicked(self):
        if self.pre_record_timer is not None:
            self._abort_pre_record()
        else:
            self.worker.cancel_sample()

    # ── Worker slots ───────────────────────────────────────────

    @Slot(object)
    def on_frame(self, frame_bgr):
        self.video.setPixmap(bgr_to_pixmap(frame_bgr, self.video.size()))

    @Slot(str)
    def on_status(self, msg: str):
        self.status.setText(msg)

    @Slot(str, float, object)
    def on_pred(self, label: str, conf: float, probs):
        if label == "sin prediccion":
            self.pred_label.setText("—")
            self.pred_label.setStyleSheet("color: #475569; padding: 4px 0;")
            self.conf_label.setText("")
            self.icon_label.setPixmap(QPixmap())
            self.icon_label.setVisible(False)
            return

        self.pred_label.setText(label)
        self.pred_label.setStyleSheet("color: #10B981; padding: 4px 0;")
        self.conf_label.setText(f"Conf: {conf:.0%}")

        icon_path = ""
        for c in self.class_meta:
            if c.get("name") == label:
                icon_path = c.get("icon", "")
                break

        if icon_path and os.path.exists(icon_path) and conf >= 0.50:
            if icon_path not in self._icon_cache:
                self._icon_cache[icon_path] = QPixmap(icon_path)
            pm = self._icon_cache[icon_path]
            self.icon_label.setPixmap(
                pm.scaled(self.icon_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            self._reposition_icon()
            self.icon_label.setVisible(True)
            self.icon_label.raise_()
        else:
            self.icon_label.setPixmap(QPixmap())
            self.icon_label.setVisible(False)

    @Slot(bool, str)
    def on_rec_state(self, is_recording: bool, msg: str):
        self.status.setText(msg)
        if is_recording:
            self._set_record_buttons_enabled(False)
            self.btn_cancel.setEnabled(True)
        else:
            self.countdown_overlay.setVisible(False)
            self.rec_border.set_idle()
            if self.pre_record_timer is None:
                self._set_record_buttons_enabled(True)
                self.btn_cancel.setEnabled(False)
            else:
                self.btn_cancel.setEnabled(True)

    @Slot(int, int)
    def on_rec_progress(self, remaining: int, total: int):
        if remaining <= 0:
            return
        done = total - remaining
        self.rec_border.set_progress(done, total)

    @Slot(int, object)
    def on_sample_ready(self, label: int, seq):
        try:
            self.ds.add(seq, label)
            self.ds.save()
            self.counts_label.setText(self._counts_text())
        except Exception as e:
            QMessageBox.warning(self, "Dataset", f"No pude guardar muestra: {e}")

    # ── Training ───────────────────────────────────────────────

    def train_model(self):
        if self.train_worker is not None and self.train_worker.isRunning():
            QMessageBox.information(self, "Entrenar", "Ya hay un entrenamiento en curso.")
            return
        if len(self.ds) < 1:
            QMessageBox.information(self, "Entrenar", "Graba al menos una muestra primero.")
            return

        self.status.setText("Iniciando entrenamiento…")
        self.btn_train.setEnabled(False)

        X = np.stack(self.ds.X, axis=0).astype(np.float32)
        y = np.asarray(self.ds.y, dtype=np.int64)

        self.train_worker = TrainWorker(X, y, LSTM_CKPT_PATH, num_classes=3)
        self.train_worker.status.connect(self.on_status)
        self.train_worker.done.connect(self.on_train_done)
        self.train_worker.start()

    @Slot(bool, str)
    def on_train_done(self, ok: bool, msg: str):
        self.btn_train.setEnabled(True)
        self.status.setText(msg)
        if ok:
            if self.worker:
                QMetaObject.invokeMethod(self.worker, "reload_classifier", Qt.QueuedConnection)
            QMessageBox.information(self, "Entrenar", msg)
        else:
            QMessageBox.warning(self, "Entrenar", msg)

    # ── Close ──────────────────────────────────────────────────

    def closeEvent(self, event):
        if self.pre_record_timer is not None:
            self._abort_pre_record(show_status=False)
        if self.train_worker and self.train_worker.isRunning():
            self.train_worker.wait(500)
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1500)
        super().closeEvent(event)
