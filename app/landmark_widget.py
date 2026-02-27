from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QPainter, QPen, QBrush
from PySide6.QtWidgets import QWidget
import mediapipe as mp

mp_hands = mp.tasks.vision.HandLandmarksConnections

def _normalize_connections(connections):
    out = []
    for c in connections:
        a = b = None
        if isinstance(c, (tuple, list)) and len(c) == 2:
            a, b = c[0], c[1]
        else:
            for a_key, b_key in [("start", "end"), ("from_landmark", "to_landmark"), ("from_", "to")]:
                if hasattr(c, a_key) and hasattr(c, b_key):
                    a, b = getattr(c, a_key), getattr(c, b_key)
                    break
            if a is None:
                try:
                    a, b = c[0], c[1]
                except Exception:
                    pass
        if a is None or b is None:
            continue
        out.append((int(a), int(b)))
    return out

CONNS = _normalize_connections(mp_hands.HAND_CONNECTIONS)

class LandmarkToggleWidget(QWidget):
    mask_changed = Signal(object)  # list[bool]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.enabled = [True] * 21
        self.setMinimumWidth(280)
        self.setMinimumHeight(280)

        # Esquema mano (mejor proporcionado)
        self.pos = [
            (0.50, 0.88),
            (0.40, 0.78), (0.33, 0.70), (0.28, 0.62), (0.25, 0.54),
            (0.46, 0.74), (0.44, 0.60), (0.43, 0.45), (0.42, 0.30),
            (0.53, 0.72), (0.53, 0.56), (0.53, 0.40), (0.53, 0.22),
            (0.60, 0.74), (0.62, 0.60), (0.64, 0.45), (0.66, 0.30),
            (0.67, 0.78), (0.71, 0.66), (0.75, 0.54), (0.78, 0.42),
        ]

    def get_mask(self):
        return list(self.enabled)

    def _content_rect(self) -> QRectF:
        m = 18
        r = self.rect().adjusted(m, m, -m, -m)
        side = min(r.width(), r.height())
        c = r.center()
        return QRectF(c.x() - side/2, c.y() - side/2, side, side)

    def _to_screen(self, i) -> QPointF:
        x, y = self.pos[i]
        r = self._content_rect()
        return QPointF(r.left() + x * r.width(), r.top() + y * r.height())

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        try:
            # fondo default (no pintamos negro)

            # conexiones
            for a, b in CONNS:
                pa = self._to_screen(a)
                pb = self._to_screen(b)
                both_on = self.enabled[a] and self.enabled[b]
                pen = QPen(Qt.black if both_on else Qt.gray, 2 if both_on else 1)
                pen.setCapStyle(Qt.RoundCap)
                p.setPen(pen)
                p.drawLine(pa, pb)

            # puntos
            for i in range(21):
                pt = self._to_screen(i)
                if self.enabled[i]:
                    p.setPen(QPen(Qt.black, 2))
                    p.setBrush(QBrush(Qt.black))
                    r = 7
                else:
                    p.setPen(QPen(Qt.gray, 2))
                    p.setBrush(QBrush(Qt.gray))
                    r = 6
                p.drawEllipse(pt, r, r)
        finally:
            p.end()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        click = event.position()
        best_i = None
        best_d2 = 1e18

        for i in range(21):
            pt = self._to_screen(i)
            dx = pt.x() - click.x()
            dy = pt.y() - click.y()
            d2 = dx*dx + dy*dy
            if d2 < best_d2:
                best_d2 = d2
                best_i = i

        if best_i is not None and best_d2 <= (18 * 18):
            self.enabled[best_i] = not self.enabled[best_i]
            self.update()
            self.mask_changed.emit(self.get_mask())
