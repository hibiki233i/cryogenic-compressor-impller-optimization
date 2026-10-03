"""Shared visual tokens for the BOUNDYR engineering workspace."""

from PySide6.QtGui import QColor, QPalette

PALETTE = {
    "bg": "#10171f", "panel": "#17212c", "input": "#101923",
    "raised": "#22313f", "border": "#344657", "text": "#e6edf3",
    "muted": "#a3b3c2", "accent": "#5ee0ca", "accent_soft": "#203f40",
    "danger": "#ffaaa5", "warning": "#f4cf82", "blue": "#91c6ff",
}


def build_palette() -> QPalette:
    """Give native arrows and popups the same readable dark colors as QSS."""
    palette = QPalette()
    for role, token in ((QPalette.Window, "bg"), (QPalette.WindowText, "text"),
                        (QPalette.Base, "input"), (QPalette.Text, "text"),
                        (QPalette.Button, "raised"), (QPalette.ButtonText, "text"),
                        (QPalette.Highlight, "accent_soft"), (QPalette.HighlightedText, "accent")):
        palette.setColor(role, QColor(PALETTE[token]))
    return palette


def build_stylesheet() -> str:
    p = PALETTE
    return f"""
    QWidget {{ color: {p['text']}; font-size: 13px; }}
    QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget {{ background: {p['bg']}; }}
    QLabel {{ background: transparent; }}
    QWidget#sidebar {{ background: {p['panel']}; border-right: 1px solid {p['border']}; }}
    QLabel#brand {{ color: {p['accent']}; font-size: 23px; font-weight: 700; letter-spacing: 3px; }}
    QLabel#pageTitle {{ font-size: 26px; font-weight: 700; }}
    QLabel#subtitle, QLabel#sidebarFooter {{ color: {p['muted']}; font-size: 12px; }}
    QLabel#pageIntro {{ color: {p['muted']}; font-size: 14px; padding: 4px 0; }}
    QLabel#sectionTitle {{ font-weight: 700; color: {p['muted']}; }}
    QListWidget#workflowNavigation {{ background: transparent; border: none; outline: none; }}
    QListWidget#workflowNavigation::item {{ color: {p['muted']}; padding-left: 16px; border-radius: 6px; border-left: 3px solid transparent; }}
    QListWidget#workflowNavigation::item:hover {{ background: {p['raised']}; }}
    QListWidget#workflowNavigation::item:selected {{ background: {p['accent_soft']}; color: {p['accent']}; border-left: 3px solid {p['accent']}; font-weight: 700; }}
    QListWidget#workflowNavigation:focus {{ border: 1px solid {p['border']}; }}
    QStackedWidget, QScrollArea {{ border: none; }}
    QGroupBox {{ background: {p['panel']}; border: 1px solid {p['border']}; border-radius: 8px; margin-top: 12px; padding: 22px 18px 16px; font-weight: 600; }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 14px; padding: 0 6px; color: {p['accent']}; }}
    QGroupBox#statCard {{ padding: 5px 10px; margin-top: 0; }}
    QGroupBox#liveTaskGroup {{ padding: 10px 12px 6px; }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {{ background: {p['input']}; color: {p['text']}; border: 1px solid {p['border']}; border-radius: 5px; padding: 7px 9px; selection-background-color: {p['accent_soft']}; selection-color: {p['accent']}; }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{ border: 1px solid {p['accent']}; }}
    QComboBox QAbstractItemView {{ background: {p['panel']}; color: {p['text']}; selection-background-color: {p['accent_soft']}; }}
    QPlainTextEdit {{ font-family: Consolas, 'Microsoft YaHei UI'; font-size: 12px; }}
    QWidget#activityPanel {{ background: {p['panel']}; border: 1px solid {p['border']}; border-radius: 8px; }}
    QPushButton {{ background: {p['raised']}; color: {p['text']}; border: 1px solid {p['border']}; border-radius: 5px; padding: 8px 14px; font-weight: 600; }}
    QPushButton:hover {{ border-color: {p['accent']}; }}
    QPushButton:focus {{ border: 2px solid {p['accent']}; }}
    QPushButton:pressed {{ background: {p['accent_soft']}; }}
    QPushButton[role="primary"] {{ background: {p['accent']}; color: #102d2b; border-color: {p['accent']}; }}
    QPushButton[role="primary"]:hover {{ background: #89eddb; }}
    QPushButton#quietButton {{ background: transparent; color: {p['accent']}; border: none; }}
    QPushButton#stopButton {{ background: #38252b; color: {p['danger']}; border-color: #72434a; }}
    QPushButton:disabled, QPushButton#stopButton:disabled {{ background: {p['panel']}; color: #758697; border-color: {p['border']}; }}
    QLabel#statusLabel {{ color: {p['muted']}; padding: 5px 10px; background: {p['panel']}; border-radius: 4px; }}
    QLabel#statusLabel[state="status_running"] {{ color: {p['blue']}; }}
    QLabel#statusLabel[state="status_done"] {{ color: {p['accent']}; }}
    QLabel#statusLabel[state="status_failed"] {{ color: {p['danger']}; }}
    QLabel#statusLabel[state="status_stopping"] {{ color: {p['warning']}; }}
    QCheckBox {{ spacing: 8px; }}
    QCheckBox:focus {{ color: {p['accent']}; }}
    QProgressBar {{ background: {p['input']}; color: {p['text']}; border: 1px solid {p['border']}; border-radius: 4px; text-align: center; min-height: 18px; }}
    QProgressBar::chunk {{ background: #276f66; border-radius: 3px; }}
    QTableWidget, QTableView {{ background: {p['input']}; alternate-background-color: {p['panel']}; gridline-color: {p['border']}; border: 1px solid {p['border']}; selection-background-color: {p['accent_soft']}; }}
    QHeaderView::section {{ background: {p['raised']}; color: {p['text']}; padding: 7px; border: 1px solid {p['border']}; }}
    QTabWidget::pane {{ border: 1px solid {p['border']}; background: {p['bg']}; }}
    QTabBar::tab {{ background: {p['panel']}; color: {p['muted']}; padding: 9px 14px; border-bottom: 2px solid transparent; }}
    QTabBar::tab:selected {{ background: {p['raised']}; color: {p['accent']}; border-bottom-color: {p['accent']}; }}
    QSplitter::handle {{ background: {p['bg']}; height: 8px; }}
    QScrollBar:vertical {{ background: {p['bg']}; width: 10px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: {p['border']}; min-height: 30px; border-radius: 5px; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
    QToolTip {{ background: {p['raised']}; color: {p['text']}; border: 1px solid {p['border']}; padding: 6px; }}
    QMessageBox, QFileDialog {{ background: {p['panel']}; }}
    """
