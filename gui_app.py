import sys

from PySide6.QtQuickControls2 import QQuickStyle

from gui.app import main


if __name__ == "__main__":
    QQuickStyle.setStyle("Fusion")
    sys.exit(main())