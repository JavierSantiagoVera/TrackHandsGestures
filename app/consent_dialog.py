import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QVBoxLayout,
)

from . import config as cfg


class ConsentDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("IA para Inclusión · LSC")
        self.setModal(True)
        self.setMinimumWidth(540)
        self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self._password = ""

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(36, 32, 36, 28)

        # ── Title ──────────────────────────────────────────────
        title = QLabel("🤟  IA para Inclusión")
        title.setAlignment(Qt.AlignCenter)
        f = QFont("Segoe UI", 22, QFont.Bold)
        title.setFont(f)
        layout.addWidget(title)

        sub = QLabel("Reconocimiento de señas LSC · Colombia")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color: #64748B; font-size: 13px;")
        layout.addWidget(sub)

        # ── Separator ──────────────────────────────────────────
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #334155; margin: 4px 0;")
        layout.addWidget(line)

        # ── Info text ──────────────────────────────────────────
        info = QLabel(
            "<b>¿Qué datos se recopilan?</b><br>"
            "Esta herramienta captura <b>secuencias de puntos geométricos de manos</b> "
            "(landmarks) para entrenar un modelo de reconocimiento de señas de la LSC.<br><br>"
            "<b>¿Para qué se usan?</b><br>"
            "• Únicamente para esta actividad educativa.<br>"
            "• No se guarda video ni imagen de los participantes.<br>"
            "• No se comparten con terceros ni se conservan después de la sesión.<br>"
            "• Los datos se guardan <b>cifrados</b>; solo el docente puede acceder con su contraseña.<br>"
            "• El docente puede borrar toda la información en cualquier momento."
        )
        info.setWordWrap(True)
        info.setStyleSheet("font-size: 13px; color: #CBD5E1; line-height: 1.6;")
        layout.addWidget(info)

        # ── Consent checkbox ───────────────────────────────────
        self.check = QCheckBox(
            "Entiendo y acepto el uso de mis datos de mano para esta actividad educativa."
        )
        self.check.setStyleSheet("font-size: 13px; padding: 4px 0;")
        layout.addWidget(self.check)

        # ── Password ───────────────────────────────────────────
        _has_dataset = os.path.exists(cfg.DATASET_ENC_PATH)

        pwd_row = QHBoxLayout()
        pwd_lbl = QLabel("Contraseña del docente:")
        pwd_lbl.setFixedWidth(200)
        self.pwd_edit = QLineEdit()
        self.pwd_edit.setEchoMode(QLineEdit.Password)
        self.pwd_edit.setPlaceholderText("Mínimo 4 caracteres")
        pwd_row.addWidget(pwd_lbl)
        pwd_row.addWidget(self.pwd_edit, 1)
        layout.addLayout(pwd_row)

        if _has_dataset:
            self.confirm_edit = None
            hint = QLabel(
                "⚠️  Ya existe un dataset de esta sesión. "
                "Ingresa la misma contraseña para continuar."
            )
            hint.setWordWrap(True)
            hint.setStyleSheet("color: #F59E0B; font-size: 12px;")
            layout.addWidget(hint)
        else:
            confirm_row = QHBoxLayout()
            confirm_lbl = QLabel("Confirmar contraseña:")
            confirm_lbl.setFixedWidth(200)
            self.confirm_edit = QLineEdit()
            self.confirm_edit.setEchoMode(QLineEdit.Password)
            self.confirm_edit.setPlaceholderText("Repite la contraseña")
            confirm_row.addWidget(confirm_lbl)
            confirm_row.addWidget(self.confirm_edit, 1)
            layout.addLayout(confirm_row)
            new_hint = QLabel(
                "🔑  Nueva sesión: elige una contraseña para proteger los datos."
            )
            new_hint.setWordWrap(True)
            new_hint.setStyleSheet("color: #34D399; font-size: 12px;")
            layout.addWidget(new_hint)

        # ── Buttons ────────────────────────────────────────────
        btn_row = QHBoxLayout()

        self.btn_reset = QPushButton("🗑  Borrar todo")
        self.btn_reset.setObjectName("btn_danger")
        self.btn_reset.setToolTip(
            "Borra el dataset cifrado y el modelo guardados (requiere contraseña)."
        )

        self.btn_continue = QPushButton("Continuar  →")
        self.btn_continue.setObjectName("btn_continue")
        self.btn_continue.setEnabled(False)
        self.btn_continue.setDefault(True)

        btn_row.addWidget(self.btn_reset)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_continue)
        layout.addLayout(btn_row)

        # ── Connections ────────────────────────────────────────
        self.check.stateChanged.connect(self._update_btn)
        self.pwd_edit.textChanged.connect(self._update_btn)
        if self.confirm_edit is not None:
            self.confirm_edit.textChanged.connect(self._update_btn)
        self.btn_continue.clicked.connect(self._on_continue)
        self.btn_reset.clicked.connect(self._on_reset_all)

    # ──────────────────────────────────────────────────────────

    def _update_btn(self):
        pwd = self.pwd_edit.text()
        ok = self.check.isChecked() and len(pwd.strip()) >= 4
        if ok and self.confirm_edit is not None:
            ok = pwd == self.confirm_edit.text() and len(self.confirm_edit.text().strip()) >= 4
        self.btn_continue.setEnabled(ok)

    def _on_continue(self):
        pwd = self.pwd_edit.text().strip()
        if self.confirm_edit is not None and self.pwd_edit.text() != self.confirm_edit.text():
            QMessageBox.warning(self, "Contraseña", "Las contraseñas no coinciden.")
            return
        enc = cfg.DATASET_ENC_PATH
        if os.path.exists(enc):
            try:
                load_encrypted(enc, pwd)
            except Exception:
                QMessageBox.warning(
                    self,
                    "Contraseña incorrecta",
                    "La contraseña no coincide con el dataset guardado.\n\n"
                    "• Si recuerdas la contraseña, ingrésala de nuevo.\n"
                    "• Si la olvidaste, usa el botón «🗑 Borrar todo» para\n"
                    "  eliminar los datos y empezar con una contraseña nueva.",
                )
                return
        self._password = pwd
        self.accept()

    def _on_reset_all(self):
        enc = cfg.DATASET_ENC_PATH

        has_data = os.path.exists(enc) or os.path.exists(cfg.LSTM_CKPT_PATH)
        if not has_data:
            QMessageBox.information(self, "Sin datos", "No hay datos guardados para borrar.")
            return

        reply = QMessageBox.question(
            self,
            "Borrar todo",
            "¿Seguro que quieres borrar TODO el dataset y el modelo?\n"
            "Esta acción NO se puede deshacer.\n\n"
            "Después podrás comenzar una sesión nueva con una contraseña diferente.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        for path in (enc, cfg.LSTM_CKPT_PATH):
            if os.path.exists(path):
                os.remove(path)

        QMessageBox.information(
            self,
            "Borrar todo",
            "Datos borrados correctamente.\n"
            "Puedes comenzar una nueva sesión con una contraseña nueva.",
        )

    def password(self) -> str:
        return self._password
