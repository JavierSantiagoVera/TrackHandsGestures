import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QGuiApplication
from app.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    geo = QGuiApplication.primaryScreen().availableGeometry()
    w.setGeometry(geo)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
