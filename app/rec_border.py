from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

BORDER_W = 10  # pixels; also the layout margin so video sits inside

_C_IDLE = QColor("#1E293B")
_C_REC = QColor("#10B981")       # emerald — fills as recording progresses
_C_DONE = QColor("#34D399")      # brighter green when complete
_C_COUNTDOWN = QColor("#F59E0B") # amber — solid full border during countdown


class RecBorderWidget(QWidget):
    """Video container that draws an animated rectangular progress border."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._done = 0
        self._total = 0
        self._state = "idle"  # "idle" | "countdown" | "recording"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(BORDER_W, BORDER_W, BORDER_W, BORDER_W)
        layout.setSpacing(0)

        self.video_label = QLabel("Iniciando cámara…")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video_label)

    # ── Public API ─────────────────────────────────────────────

    def set_idle(self):
        self._state = "idle"
        self._done = 0
        self._total = 0
        self.update()

    def set_countdown(self):
        self._state = "countdown"
        self.update()

    def set_progress(self, done: int, total: int):
        self._state = "recording"
        self._done = done
        self._total = max(1, total)
        self.update()

    # ── Paint ──────────────────────────────────────────────────

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)  # crisp rectangular edges
        W, H = self.width(), self.height()
        hw = BORDER_W // 2

        if self._state == "idle":
            pen = QPen(_C_IDLE, 2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(0, 0, W - 1, H - 1)

        elif self._state == "countdown":
            pen = QPen(_C_COUNTDOWN, BORDER_W)
            pen.setJoinStyle(Qt.MiterJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(hw, hw, W - BORDER_W, H - BORDER_W)

        elif self._state == "recording" and self._total > 0:
            # Dark background border
            pen_bg = QPen(_C_IDLE, BORDER_W)
            pen_bg.setJoinStyle(Qt.MiterJoin)
            painter.setPen(pen_bg)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(hw, hw, W - BORDER_W, H - BORDER_W)

            # Clockwise progress fill from top-left
            frac = min(1.0, self._done / self._total)
            color = _C_DONE if frac >= 1.0 else _C_REC
            pen_fg = QPen(color, BORDER_W)
            pen_fg.setCapStyle(Qt.FlatCap)
            painter.setPen(pen_fg)
            painter.setBrush(Qt.NoBrush)

            iw = W - BORDER_W   # inner segment width
            ih = H - BORDER_W   # inner segment height

            # (x1, y1, x2, y2, segment_length)
            segments = [
                (hw,      hw,      hw + iw, hw,      iw),  # top    L→R
                (hw + iw, hw,      hw + iw, hw + ih, ih),  # right  T→B
                (hw + iw, hw + ih, hw,      hw + ih, iw),  # bottom R→L
                (hw,      hw + ih, hw,      hw,      ih),  # left   B→T
            ]

            remaining = frac * 2 * (iw + ih)
            for x1, y1, x2, y2, seg_len in segments:
                if remaining <= 0:
                    break
                if remaining >= seg_len:
                    painter.drawLine(int(x1), int(y1), int(x2), int(y2))
                    remaining -= seg_len
                else:
                    t = remaining / seg_len
                    painter.drawLine(
                        int(x1), int(y1),
                        int(round(x1 + t * (x2 - x1))),
                        int(round(y1 + t * (y2 - y1))),
                    )
                    remaining = 0

        painter.end()
