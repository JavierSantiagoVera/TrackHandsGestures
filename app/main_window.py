from PySide6.QtCore import Qt, Slot, QMetaObject, Q_ARG, QTimer
from PySide6.QtWidgets import (
    QWidget, QLabel, QHBoxLayout, QVBoxLayout, QSizePolicy,
    QPushButton, QMessageBox, QProgressBar, QLineEdit, QFileDialog
)

import os
from PySide6.QtGui import QPixmap

from .config import MODEL_TASK_PATH, SEQ_LEN, DATASET_PATH, LSTM_CKPT_PATH, DEFAULT_CLASSES
from .qt_utils import bgr_to_pixmap
from .landmark_widget import LandmarkToggleWidget
from .worker import HandWorker
from .dataset_store import DatasetStore
from .train_worker import TrainWorker

from .config import load_ui_classes, class_names_from_meta, save_ui_classes


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Tracker (LSC) - LSTM")

        # Dataset (features)
        self.ds = DatasetStore(DATASET_PATH, seq_len=SEQ_LEN)

        # -----------------------------
        # UI: Video grande
        # -----------------------------
        self.video = QLabel("Iniciando...")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(900, 600)
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # -----------------------------
        # UI: Panel derecho
        # -----------------------------
        self.landmarks_view = LandmarkToggleWidget()
        self.landmarks_view.setFixedWidth(330)
        self.landmarks_view.setFixedHeight(380)

        self.pred_label = QLabel("Pred: -")
        self.pred_label.setAlignment(Qt.AlignCenter)
        self.pred_label.setStyleSheet("font-size: 26px; font-weight: bold;")

        self.conf_label = QLabel("Conf: -")
        self.conf_label.setAlignment(Qt.AlignCenter)
        self.conf_label.setStyleSheet("font-size: 16px;")

        # Labels/clases (dinámico)
        self.class_meta = load_ui_classes()
        self._ensure_class_meta(3)
        self.class_meta = self.class_meta[:3]
        self.class_names = class_names_from_meta(self.class_meta)[:3]
        self._icon_cache = {}  # path -> QPixmap
        self.class_controls = []

        self.icon_label = QLabel("")
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setFixedSize(140, 140)
        self.icon_label.setStyleSheet("border: 1px solid #ccc;")
        self.icon_label.setVisible(False)

        # Dataset counts
        self.counts_label = QLabel(self._counts_text())
        self.counts_label.setAlignment(Qt.AlignCenter)
        self.counts_label.setWordWrap(True)

        # Status
        self.status = QLabel("")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setWordWrap(True)

        # -----------------------------
        # NUEVO: Contador + barra de grabación
        # -----------------------------
        self.countdown_label = QLabel("")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setStyleSheet("font-size: 18px; font-weight: bold;")

        self.rec_bar = QProgressBar()
        self.rec_bar.setRange(0, SEQ_LEN)
        self.rec_bar.setValue(0)
        self.rec_bar.setTextVisible(False)
        self.rec_bar.setFixedHeight(10)

        # Botones: añadir muestra (una secuencia de T frames)
        # (asegura que existan 3 nombres)
        while len(self.class_names) < 3:
            self.class_names.append(f"Clase_{len(self.class_names)}")

        self.btn_add0 = QPushButton(f"Añadir muestra: {self.class_names[0]} ({SEQ_LEN}f)")
        self.btn_add1 = QPushButton(f"Añadir muestra: {self.class_names[1]} ({SEQ_LEN}f)")
        self.btn_add2 = QPushButton(f"Añadir muestra: {self.class_names[2]} ({SEQ_LEN}f)")
        self.btn_cancel = QPushButton("Cancelar grabación")
        self.btn_cancel.setEnabled(False)

        # Entrenar
        self.btn_train = QPushButton("Entrenar modelo (LSTM)")

        # Pre-grabación
        self.pre_record_timer = None
        self.pre_record_target = None
        self.pre_record_seconds = 0

        # -----------------------------
        # Layout derecho (IMPORTANTE: primero crear right)
        # -----------------------------
        right = QVBoxLayout()
        right.addWidget(self.landmarks_view, 0, Qt.AlignTop)
        right.addWidget(self.icon_label, 0, Qt.AlignRight)

        right.addWidget(self.pred_label, 0)
        right.addWidget(self.conf_label, 0)

        self.class_title = QLabel("Clases (nombre e icono)")
        self.class_title.setAlignment(Qt.AlignCenter)
        self.class_title.setStyleSheet("font-weight: bold; margin-top: 6px;")
        right.addWidget(self.class_title, 0)

        class_edit_box = QVBoxLayout()
        class_edit_box.setSpacing(4)
        class_edit_box.setContentsMargins(0, 0, 0, 0)

        for idx in range(3):
            editor = self._create_class_editor(idx)
            class_edit_box.addLayout(editor["layout"])
            self.class_controls.append(editor)

        right.addLayout(class_edit_box, 0)

        right.addWidget(self.counts_label, 0)

        # Contador/barra (debajo de counts, antes de botones)
        right.addSpacing(8)
        right.addWidget(self.countdown_label, 0)
        right.addWidget(self.rec_bar, 0)

        right.addSpacing(10)
        right.addWidget(self.btn_add0, 0)
        right.addWidget(self.btn_add1, 0)
        right.addWidget(self.btn_add2, 0)
        right.addWidget(self.btn_cancel, 0)

        right.addSpacing(10)
        right.addWidget(self.btn_train, 0)
        right.addWidget(self.status, 1)

        root = QHBoxLayout(self)
        root.addWidget(self.video, 1)
        root.addLayout(right, 0)

        # -----------------------------
        # Worker cámara
        # -----------------------------
        self.worker = HandWorker(model_task_path=MODEL_TASK_PATH, camera_index=0)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.status.connect(self.on_status)
        self.worker.pred_ready.connect(self.on_pred)

        # máscara -> worker
        self.landmarks_view.mask_changed.connect(self.worker.set_enabled_mask)
        self.worker.set_enabled_mask(self.landmarks_view.get_mask())

        # grabación features
        self.worker.sample_ready.connect(self.on_sample_ready)
        self.worker.rec_state.connect(self.on_rec_state)

        # NUEVO: progreso de grabación (remaining, total)
        self.worker.rec_progress.connect(self.on_rec_progress)

        self.btn_add0.clicked.connect(lambda: self._schedule_recording(0))
        self.btn_add1.clicked.connect(lambda: self._schedule_recording(1))
        self.btn_add2.clicked.connect(lambda: self._schedule_recording(2))
        self.btn_cancel.clicked.connect(self.on_cancel_clicked)

        # Entrenamiento
        self.train_worker = None
        self.btn_train.clicked.connect(self.train_model)

        self.worker.start()
        self._push_class_names_to_worker()

    def _counts_text(self):
        counts = self.ds.counts(num_classes=3)
        return (
            f"Dataset: {len(self.ds)} muestras\n"
            f"{self.class_names[0]}: {counts[0]} | {self.class_names[1]}: {counts[1]} | {self.class_names[2]}: {counts[2]}"
        )

    def _set_record_buttons_enabled(self, enabled: bool):
        self.btn_add0.setEnabled(enabled)
        self.btn_add1.setEnabled(enabled)
        self.btn_add2.setEnabled(enabled)

    def _ensure_class_meta(self, required_len: int):
        if not isinstance(self.class_meta, list):
            self.class_meta = []
        for i in range(len(self.class_meta)):
            if not isinstance(self.class_meta[i], dict):
                self.class_meta[i] = {"name": f"Clase_{i}", "icon": ""}
            self.class_meta[i].setdefault("name", f"Clase_{i}")
            self.class_meta[i].setdefault("icon", "")
        while len(self.class_meta) < required_len:
            self.class_meta.append({"name": f"Clase_{len(self.class_meta)}", "icon": ""})

    def _create_class_editor(self, idx: int):
        wrapper = QVBoxLayout()
        wrapper.setContentsMargins(0, 0, 0, 0)

        name_row = QHBoxLayout()
        name_label = QLabel(f"Clase {idx + 1}:")
        name_edit = QLineEdit(self.class_names[idx] if idx < len(self.class_names) else f"Clase_{idx}")
        name_edit.setMaxLength(40)
        name_row.addWidget(name_label, 0)
        name_row.addWidget(name_edit, 1)
        reset_btn = QPushButton("Reset")
        reset_btn.setFixedWidth(60)
        reset_btn.clicked.connect(lambda _, i=idx: self._reset_class_name(i))
        name_row.addWidget(reset_btn, 0)
        name_edit.editingFinished.connect(
            lambda idx=idx, edit=name_edit: self._on_class_name_changed(idx, edit.text())
        )

        icon_row = QHBoxLayout()
        icon_row.setContentsMargins(16, 0, 0, 6)
        icon_button = QPushButton("Icono...")
        icon_button.clicked.connect(lambda _, i=idx: self._choose_icon(i))
        clear_button = QPushButton("Quitar")
        clear_button.setFixedWidth(70)
        clear_button.clicked.connect(lambda _, i=idx: self._clear_icon(i))
        icon_info = QLabel(self._icon_path_display(idx))
        icon_info.setWordWrap(True)
        icon_info.setStyleSheet("font-size: 11px; color: #555;")
        icon_row.addWidget(icon_button, 0)
        icon_row.addWidget(clear_button, 0)
        icon_row.addWidget(icon_info, 1)

        wrapper.addLayout(name_row)
        wrapper.addLayout(icon_row)

        return {
            "layout": wrapper,
            "name_edit": name_edit,
            "icon_label": icon_info,
        }

    def _icon_path_display(self, idx: int):
        icon_path = ""
        if 0 <= idx < len(self.class_meta):
            icon_path = self.class_meta[idx].get("icon", "") or ""
        return os.path.basename(icon_path) if icon_path else "Sin icono"

    def _default_name(self, idx: int):
        if 0 <= idx < len(DEFAULT_CLASSES):
            base = DEFAULT_CLASSES[idx].get("name", f"Clase_{idx}")
        else:
            base = f"Clase_{idx}"
        return base

    def _on_class_name_changed(self, idx: int, new_value: str):
        if not (0 <= idx < len(self.class_meta)):
            return
        name = (new_value or "").strip()
        if not name:
            name = self._default_name(idx)
            self.class_controls[idx]["name_edit"].setText(name)
        self.class_meta[idx]["name"] = name
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
            self,
            "Seleccionar icono",
            "",
            "Imágenes (*.png *.jpg *.jpeg *.bmp *.gif);;Todos los archivos (*)",
        )
        if not file_path:
            return

        current = self.class_meta[idx].get("icon", "")
        if current and current in self._icon_cache:
            self._icon_cache.pop(current, None)

        self.class_meta[idx]["icon"] = file_path
        self.class_controls[idx]["icon_label"].setText(self._icon_path_display(idx))
        self._on_class_meta_updated()

    def _clear_icon(self, idx: int):
        if not (0 <= idx < len(self.class_meta)):
            return
        current = self.class_meta[idx].get("icon", "")
        if current and current in self._icon_cache:
            self._icon_cache.pop(current, None)
        self.class_meta[idx]["icon"] = ""
        self.class_controls[idx]["icon_label"].setText(self._icon_path_display(idx))
        self._on_class_meta_updated()

    def _on_class_meta_updated(self):
        self.class_names = class_names_from_meta(self.class_meta)[:3]

        button_texts = [
            (self.btn_add0, 0),
            (self.btn_add1, 1),
            (self.btn_add2, 2),
        ]
        for btn, idx in button_texts:
            btn.setText(f"Añadir muestra: {self.class_names[idx]} ({SEQ_LEN}f)")

        self.counts_label.setText(self._counts_text())
        self._push_class_names_to_worker()
        self._save_class_meta()

    def _save_class_meta(self):
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
                self.worker,
                "set_class_names",
                Qt.QueuedConnection,
                Q_ARG(object, names),
            )
        except Exception:
            try:
                self.worker.set_class_names(names)
            except Exception:
                pass

    def _schedule_recording(self, label_idx: int):
        if self.pre_record_timer is not None:
            self.status.setText("Ya hay una grabación programada. Espera unos segundos.")
            return
        if not (self.worker and self.worker.isRunning()):
            QMessageBox.warning(self, "Grabación", "El worker de cámara no está activo.")
            return
        if not (0 <= label_idx < len(self.class_names)):
            return

        self.pre_record_target = label_idx
        self.pre_record_seconds = 3
        self.rec_bar.setValue(0)
        self.countdown_label.setText(f"Grabación inicia en {self.pre_record_seconds}s")
        self.status.setText(f"Grabación inicia en {self.pre_record_seconds} segundos...")
        self._set_record_buttons_enabled(False)
        self.btn_cancel.setEnabled(True)

        self.pre_record_timer = QTimer(self)
        self.pre_record_timer.timeout.connect(self._handle_pre_record_tick)
        self.pre_record_timer.start(1000)

    def _handle_pre_record_tick(self):
        if self.pre_record_timer is None:
            return
        self.pre_record_seconds -= 1
        if self.pre_record_seconds > 0:
            self.countdown_label.setText(f"Grabación inicia en {self.pre_record_seconds}s")
            self.status.setText(f"Grabación inicia en {self.pre_record_seconds} segundos...")
            return

        target = self.pre_record_target
        self.pre_record_target = None
        self.countdown_label.setText("Iniciando grabación...")
        self.status.setText("Iniciando grabación...")
        self._stop_pre_record_timer()

        if target is not None and self.worker:
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
        self.countdown_label.setText("")
        if show_status:
            self.status.setText("Grabación cancelada antes de comenzar.")
        self._set_record_buttons_enabled(True)
        self.btn_cancel.setEnabled(False)

    @Slot()
    def on_cancel_clicked(self):
        if self.pre_record_timer is not None:
            self._abort_pre_record()
        else:
            self.worker.cancel_sample()

    @Slot(object)
    def on_frame(self, frame_bgr):
        self.video.setPixmap(bgr_to_pixmap(frame_bgr, self.video.size()))

    @Slot(str)
    def on_status(self, msg: str):
        self.status.setText(msg)

    @Slot(str, float, object)
    def on_pred(self, label: str, conf: float, probs):
        self.pred_label.setText(f"Pred: {label}")
        self.conf_label.setText(f"Conf: {conf:.2f}")

        # busca icono por nombre
        icon_path = ""
        for c in self.class_meta:
            if c.get("name") == label:
                icon_path = c.get("icon", "")
                break

        if icon_path and os.path.exists(icon_path) and conf >= 0.50:
            if icon_path not in self._icon_cache:
                pm = QPixmap(icon_path)
                self._icon_cache[icon_path] = pm
            pm = self._icon_cache[icon_path]
            self.icon_label.setPixmap(
                pm.scaled(self.icon_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            self.icon_label.setVisible(True)
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
            if self.pre_record_timer is None:
                self._set_record_buttons_enabled(True)
                self.btn_cancel.setEnabled(False)
            else:
                self.btn_cancel.setEnabled(True)

        # si dejó de grabar, limpiar UI del contador
        if not is_recording:
            self.countdown_label.setText("")
            self.rec_bar.setValue(0)

    @Slot(int, int)
    def on_rec_progress(self, remaining: int, total: int):
        # remaining: faltan N frames
        if remaining <= 0:
            self.countdown_label.setText("")
            self.rec_bar.setValue(0)
            return

        done = total - remaining
        self.countdown_label.setText(f"Grabando… faltan {remaining}/{total} frames")
        self.rec_bar.setValue(done)

    @Slot(int, object)
    def on_sample_ready(self, label: int, seq):
        try:
            self.ds.add(seq, label)
            self.ds.save()
            self.counts_label.setText(self._counts_text())
        except Exception as e:
            QMessageBox.warning(self, "Dataset", f"No pude guardar muestra: {e}")

    def train_model(self):
        if self.train_worker is not None and self.train_worker.isRunning():
            QMessageBox.information(self, "Train", "Ya hay un entrenamiento en curso.")
            return

        if len(self.ds) < 1:
            QMessageBox.information(self, "Train", "Necesitas más muestras (al menos 1).")
            return

        self.status.setText("Iniciando entrenamiento...")
        self.btn_train.setEnabled(False)

        self.train_worker = TrainWorker(DATASET_PATH, LSTM_CKPT_PATH, num_classes=3)
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
            QMessageBox.information(self, "Train", msg)
        else:
            QMessageBox.warning(self, "Train", msg)

    def closeEvent(self, event):
        if self.pre_record_timer is not None:
            self._abort_pre_record(show_status=False)
        if self.train_worker and self.train_worker.isRunning():
            self.train_worker.wait(500)

        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1500)
        super().closeEvent(event)
