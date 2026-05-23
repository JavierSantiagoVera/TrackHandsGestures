import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QGuiApplication

from app.theme import DARK_QSS
from app.consent_dialog import ConsentDialog
from app.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)

    dialog = ConsentDialog()
    if dialog.exec() != ConsentDialog.Accepted:
        sys.exit(0)

    password = dialog.password()

    w = MainWindow(password=password)
    geo = QGuiApplication.primaryScreen().availableGeometry()
    w.setGeometry(geo)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
