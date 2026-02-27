import cv2
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

def bgr_to_pixmap(frame_bgr, target_size):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    return pix.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
