"""Pantalla de inicio: consentimiento y contraseña.

- Sesión nueva (no hay datos): se pide una contraseña dos veces y se crea el
  verificador auth.enc.
- Sesión existente: se comprueba la contraseña contra auth.enc. Sin la
  contraseña correcta no se entra a la app.
- "Borrar todo": elimina dataset, verificador y modelo, y vuelve a modo sesión nueva.
"""
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFrame, QHBoxLayout, QLabel, QWidget,
    QLineEdit, QMessageBox, QPushButton, QVBoxLayout,
)

from cryptography.fernet import InvalidToken

from . import config as cfg
from .crypto_store import check_verifier, create_verifier, load_encrypted


PWD_MIN, PWD_MAX = 4, 32


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
            "• Los datos se guardan <b>cifrados</b> con una contraseña que elige cada grupo o persona; "
            "son solo para su propio uso y nadie más puede abrirlos.<br>"
            "• Quien creó la sesión puede borrar toda su información en cualquier momento."
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
        pwd_row = QHBoxLayout()
        pwd_lbl = QLabel("Contraseña del grupo:")
        pwd_lbl.setFixedWidth(200)
        self.pwd_edit = QLineEdit()
        self.pwd_edit.setEchoMode(QLineEdit.Password)
        self.pwd_edit.setMaxLength(PWD_MAX)
        self.pwd_edit.setPlaceholderText(f"Entre {PWD_MIN} y {PWD_MAX} caracteres")
        pwd_row.addWidget(pwd_lbl)
        pwd_row.addWidget(self.pwd_edit, 1)
        layout.addLayout(pwd_row)

        # Confirm row: only shown when a new password is being created
        self.confirm_row = QWidget()
        confirm_layout = QHBoxLayout(self.confirm_row)
        confirm_layout.setContentsMargins(0, 0, 0, 0)
        confirm_lbl = QLabel("Confirmar contraseña:")
        confirm_lbl.setFixedWidth(200)
        self.confirm_edit = QLineEdit()
        self.confirm_edit.setEchoMode(QLineEdit.Password)
        self.confirm_edit.setMaxLength(PWD_MAX)
        self.confirm_edit.setPlaceholderText("Repite la contraseña")
        confirm_layout.addWidget(confirm_lbl)
        confirm_layout.addWidget(self.confirm_edit, 1)
        layout.addWidget(self.confirm_row)

        self.mode_hint = QLabel()
        self.mode_hint.setWordWrap(True)
        layout.addWidget(self.mode_hint)

        # ── Buttons ────────────────────────────────────────────
        btn_row = QHBoxLayout()

        self.btn_reset = QPushButton("🗑  Borrar todo")
        self.btn_reset.setObjectName("btn_danger")
        self.btn_reset.setToolTip(
            "Borra el dataset cifrado, la contraseña y el modelo guardados."
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
        self.confirm_edit.textChanged.connect(self._update_btn)
        self.btn_continue.clicked.connect(self._on_continue)
        self.btn_reset.clicked.connect(self._on_reset_all)

        self._refresh_mode()

    # ──────────────────────────────────────────────────────────

    def _refresh_mode(self):
        """Existing session → ask for the password; otherwise create a new one."""
        self._new_session = not (
            os.path.exists(cfg.AUTH_PATH) or os.path.exists(cfg.DATASET_ENC_PATH)
        )
        self.confirm_row.setVisible(self._new_session)
        self.pwd_edit.clear()
        self.confirm_edit.clear()
        if self._new_session:
            self.mode_hint.setText("🔑  Nueva sesión: elijan una contraseña para proteger sus datos. Es solo para su grupo; no la compartan.")
            self.mode_hint.setStyleSheet("color: #34D399; font-size: 12px;")
        else:
            self.mode_hint.setText(
                "⚠️  Ya existe una sesión guardada. "
                "Ingresa la contraseña de tu grupo para continuar."
            )
            self.mode_hint.setStyleSheet("color: #F59E0B; font-size: 12px;")
        self._update_btn()
        self.adjustSize()

    # ──────────────────────────────────────────────────────────

    def _update_btn(self):
        pwd = self.pwd_edit.text()
        ok = self.check.isChecked() and len(pwd.strip()) >= PWD_MIN
        if ok and self._new_session:
            ok = pwd == self.confirm_edit.text() and len(self.confirm_edit.text().strip()) >= PWD_MIN
        self.btn_continue.setEnabled(ok)

    def _on_continue(self):
        pwd = self.pwd_edit.text().strip()
        if self._new_session and self.pwd_edit.text() != self.confirm_edit.text():
            QMessageBox.warning(self, "Contraseña", "Las contraseñas no coinciden.")
            return
        enc = cfg.DATASET_ENC_PATH
        wrong_pwd_msg = (
            "La contraseña no coincide con los datos guardados.\n\n"
            "• Si recuerdas la contraseña, ingrésala de nuevo.\n"
            "• Si la olvidaste, usa el botón «🗑 Borrar todo» para\n"
            "  eliminar los datos y empezar con una contraseña nueva."
        )
        if os.path.exists(cfg.AUTH_PATH):
            try:
                ok = check_verifier(cfg.AUTH_PATH, pwd)
            except Exception as e:
                QMessageBox.warning(self, "Verificador dañado",
                                    f"No se pudo leer {cfg.AUTH_PATH}:\n{e}")
                return
            if not ok:
                QMessageBox.warning(self, "Contraseña incorrecta", wrong_pwd_msg)
                return
        elif os.path.exists(enc):
            # Dataset from before the verifier existed: check by decrypting it
            try:
                load_encrypted(enc, pwd)
            except InvalidToken:
                QMessageBox.warning(self, "Contraseña incorrecta", wrong_pwd_msg)
                return
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Dataset dañado",
                    f"No se pudo leer el dataset guardado:\n{e}\n\n"
                    "Usa «🗑 Borrar todo» para empezar de nuevo.",
                )
                return

        if not os.path.exists(cfg.AUTH_PATH):
            try:
                create_verifier(cfg.AUTH_PATH, pwd)
            except Exception as e:
                QMessageBox.warning(self, "Contraseña", f"No se pudo guardar la contraseña:\n{e}")
                return
        self._password = pwd
        self.accept()

    def _on_reset_all(self):
        enc = cfg.DATASET_ENC_PATH

        has_data = any(os.path.exists(p) for p in (enc, cfg.LSTM_CKPT_PATH, cfg.AUTH_PATH))
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

        for path in (enc, cfg.LSTM_CKPT_PATH, cfg.AUTH_PATH):
            if os.path.exists(path):
                os.remove(path)

        QMessageBox.information(
            self,
            "Borrar todo",
            "Datos borrados correctamente.\n"
            "Puedes comenzar una nueva sesión con una contraseña nueva.",
        )
        self._refresh_mode()

    def password(self) -> str:
        return self._password
