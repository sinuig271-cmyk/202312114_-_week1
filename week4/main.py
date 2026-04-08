"""ChaosLab entry point."""

import sys
import matplotlib
matplotlib.use("QtAgg")

from PySide6.QtWidgets import QApplication

import os
sys.path.insert(0, os.path.dirname(__file__))

from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ChaosLab")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
