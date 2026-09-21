import argparse
import logging
from labelCloud import __version__


def main():
    parser = argparse.ArgumentParser(
        description="Annotate roof vertices and edges in urban point clouds."
    )
    parser.add_argument(
        "-v", "--version", action="version", version="%(prog)s " + __version__
    )
    args = parser.parse_args()

    start_gui()


def start_gui():
    import sys

    from PyQt5.QtWidgets import QApplication, QDesktopWidget

    from labelCloud.control.config_manager import config
    from labelCloud.control.controller import Controller
    from labelCloud.view.gui import GUI
    from labelCloud.view.i18n import LANGUAGE_CHINESE, language_manager

    app = QApplication(sys.argv)
    language_manager.set_language(
        config.get("USER_INTERFACE", "language", fallback=LANGUAGE_CHINESE)
    )

    # Setup Model-View-Control structure
    control = Controller()
    view = GUI(control)

    # Install event filter to catch user interventions
    app.installEventFilter(view)

    # Start GUI
    view.show()

    app.setStyle("Fusion")
    desktop = QDesktopWidget().availableGeometry()
    width = (desktop.width() - view.width()) // 2
    height = (desktop.height() - view.height()) // 2
    view.move(width, height)

    logging.info("Showing GUI...")
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
