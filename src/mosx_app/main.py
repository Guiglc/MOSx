from __future__ import annotations

import math
import io
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import font_manager, rcParams
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QAction, QGuiApplication, QImage, QPainter, QPalette, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class GroupBoxSeparator(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(10)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        try:
            palette = self.palette()
            pen = QPen(palette.color(QPalette.Mid))
            pen.setWidth(1)
            painter.setPen(pen)
            y = self.height() // 2
            painter.drawLine(0, y, self.width(), y)
        finally:
            painter.end()


def _row_widget(*widgets: QWidget) -> QWidget:
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(12)
    for widget in widgets:
        layout.addWidget(widget)
    layout.addStretch(1)
    container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    container.setFixedHeight(32)
    return container


CURRENT_UNITS_TO_AMP = {
    "pA": 1e-12,
    "nA": 1e-9,
    "uA": 1e-6,
    "mA": 1e-3,
    "A": 1.0,
}

VOLTAGE_UNITS_TO_VOLT = {
    "mV": 1e-3,
    "V": 1.0,
}

NONE_OPTION = "(None)"


def configure_matplotlib_fonts() -> None:
    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "PingFang SC",
        "Heiti SC",
        "Arial Unicode MS",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_fonts = [font for font in preferred_fonts if font in available_fonts]
    if selected_fonts:
        rcParams["font.sans-serif"] = selected_fonts + list(rcParams.get("font.sans-serif", []))
    rcParams["axes.unicode_minus"] = False


def guess_column(columns: list[str], keywords: list[str]) -> str | None:
    lowered = [(column, column.lower()) for column in columns]
    for keyword in keywords:
        for original, lowered_name in lowered:
            if keyword in lowered_name:
                return original
    return columns[0] if columns else None


def combo_text(combo: QComboBox) -> str | None:
    text = combo.currentText().strip()
    return None if not text or text == NONE_OPTION else text


def selected_list_texts(widget: QListWidget) -> list[str]:
    return [item.text() for item in widget.selectedItems()]


def info_message(text: str) -> None:
    box = QMessageBox()
    box.setText(text)
    box.exec()


def follow_system_color_scheme() -> None:
    set_app_color_scheme("system")


def set_app_color_scheme(mode: str) -> bool:
    style_hints = QGuiApplication.styleHints()
    set_color_scheme = getattr(style_hints, "setColorScheme", None)
    color_scheme = getattr(Qt, "ColorScheme", None)
    if not callable(set_color_scheme) or color_scheme is None:
        return False

    target_map = {
        "system": "Unknown",
        "light": "Light",
        "dark": "Dark",
    }
    target_name = target_map.get(mode)
    if target_name is None or not hasattr(color_scheme, target_name):
        return False

    set_color_scheme(getattr(color_scheme, target_name))
    return True


@dataclass
class DeviceResult:
    device_key: tuple[Any, ...]
    device_id: str
    display_key: str
    width_nm: float | None
    length_nm: float | None
    idoff_pa: float | None
    idoff_pa_per_nm: float | None
    ids_ua: float | None
    ids_ua_per_nm: float | None
    idl_ua: float | None
    idl_ua_per_nm: float | None
    vts_v: float | None
    vtl_v: float | None
    vtgm_sat_v: float | None
    vtgm_lin_v: float | None
    details: dict[str, Any]


class DataFrameModel(QAbstractTableModel):
    def __init__(self) -> None:
        super().__init__()
        self._frame = pd.DataFrame()

    def set_frame(self, frame: pd.DataFrame) -> None:
        self.beginResetModel()
        self._frame = frame.copy()
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._frame.index)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._frame.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.EditRole):
            return None
        value = self._frame.iat[index.row(), index.column()]
        return self._format_value(value)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole) -> Any:
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._frame.columns[section])
        return str(section + 1)

    def _format_value(self, value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (int, float, np.integer, np.floating)):
            number = float(value)
            return f"{number:.2e}" if number != 0 and (abs(number) >= 1e4 or abs(number) < 1e-2) else f"{number:.2f}"
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return ""
            try:
                numeric = float(text)
            except ValueError:
                return value
            return f"{numeric:.2e}" if numeric != 0 and (abs(numeric) >= 1e4 or abs(numeric) < 1e-2) else f"{numeric:.2f}"
        return str(value)


class WlDialog(QDialog):
    def __init__(
        self,
        device_keys: list[tuple[Any, ...]],
        existing: dict[tuple[Any, ...], tuple[float, float]],
        device_id_by_key: dict[tuple[Any, ...], str],
        vdd_by_device: dict[tuple[Any, ...], float],
        high_vd_by_device: dict[tuple[Any, ...], float],
        low_vd_by_device: dict[tuple[Any, ...], float],
        threshold_constant_by_device: dict[tuple[Any, ...], float],
        polarity_by_device: dict[tuple[Any, ...], str],
        current_direction_by_device: dict[tuple[Any, ...], float],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Device Setting")
        self.resize(1480, 620)
        self._device_keys = device_keys

        self._table = QTableWidget(len(device_keys), 11)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.ExtendedSelection)
        self._table.setHorizontalHeaderLabels(
            [
                "ID",
                "Device",
                "Width (nm)",
                "Length (nm)",
                "Vdd target (V)",
                "High Vd (V)",
                "Low Vd (V)",
                "Vth Const. (nA)",
                "Vth Current (nA)",
                "Polarity",
                "Current Direction",
            ]
        )
        self._table.setColumnWidth(0, 80)
        self._table.setColumnWidth(1, 360)
        self._table.setColumnWidth(2, 110)
        self._table.setColumnWidth(3, 110)
        self._table.setColumnWidth(4, 115)
        self._table.setColumnWidth(5, 95)
        self._table.setColumnWidth(6, 95)
        self._table.setColumnWidth(7, 135)
        self._table.setColumnWidth(8, 145)
        self._table.setColumnWidth(9, 110)
        self._table.setColumnWidth(10, 140)
        for row, key in enumerate(device_keys):
            label = " | ".join(str(part) for part in key)
            self._table.setItem(row, 0, QTableWidgetItem(device_id_by_key.get(key, f"D{row + 1}")))
            self._table.setItem(row, 1, QTableWidgetItem(label))
            if key in existing:
                width, length = existing[key]
                width_text = str(width)
                length_text = str(length)
            else:
                width_text = ""
                length_text = ""
            self._table.setItem(row, 2, QTableWidgetItem(width_text))
            self._table.setItem(row, 3, QTableWidgetItem(length_text))
            self._table.setItem(row, 4, QTableWidgetItem(str(vdd_by_device.get(key, 1.2))))
            self._table.setItem(row, 5, QTableWidgetItem(str(high_vd_by_device.get(key, 1.2))))
            self._table.setItem(row, 6, QTableWidgetItem(str(low_vd_by_device.get(key, 0.05))))
            self._table.setItem(row, 7, QTableWidgetItem(str(threshold_constant_by_device.get(key, 40.0))))
            self._table.setItem(row, 8, QTableWidgetItem(""))
            self._table.setCellWidget(row, 9, self._make_choice_combo(["NMOS", "PMOS"], polarity_by_device.get(key, "NMOS")))
            direction_sign = current_direction_by_device.get(key)
            if direction_sign is None:
                direction_sign = -1.0 if polarity_by_device.get(key, "NMOS") == "PMOS" else 1.0
            self._table.setCellWidget(row, 10, self._make_choice_combo(["Positive", "Negative"], "Negative" if direction_sign < 0 else "Positive"))
            self._update_threshold_current_for_row(row)

        self._table.itemChanged.connect(self._on_item_changed)
        self._table.resizeColumnsToContents()

        apply_button = QPushButton("Apply First W/L/VDD/High/Low/Vt Const. To All")
        apply_button.clicked.connect(self.apply_first_row_to_all)
        set_selected_wl_button = QPushButton("Set Selected W/L/VDD/High/Low/Th. Const.")
        set_selected_wl_button.clicked.connect(self.set_selected_device_parameters)
        clear_button = QPushButton("Clear All W/L")
        clear_button.clicked.connect(self.clear_all_dimensions)
        set_nmos_button = QPushButton("Set Selected NMOS")
        set_nmos_button.clicked.connect(lambda: self.set_selected_polarity("NMOS"))
        set_pmos_button = QPushButton("Set Selected PMOS")
        set_pmos_button.clicked.connect(lambda: self.set_selected_polarity("PMOS"))
        set_positive_button = QPushButton("Set Selected Positive Id")
        set_positive_button.clicked.connect(lambda: self.set_selected_current_direction(1.0))
        set_negative_button = QPushButton("Set Selected Negative Id")
        set_negative_button.clicked.connect(lambda: self.set_selected_current_direction(-1.0))
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        wl_group = QGroupBox("W/L/VDD/High Vd/Low Vd/Vt Const.")
        wl_layout = QHBoxLayout(wl_group)
        wl_layout.addWidget(apply_button)
        wl_layout.addWidget(set_selected_wl_button)
        wl_layout.addWidget(clear_button)

        polarity_group = QGroupBox("Polarity")
        polarity_layout = QHBoxLayout(polarity_group)
        polarity_layout.addWidget(set_nmos_button)
        polarity_layout.addWidget(set_pmos_button)

        current_group = QGroupBox("Id Direction")
        current_layout = QHBoxLayout(current_group)
        current_layout.addWidget(set_positive_button)
        current_layout.addWidget(set_negative_button)

        first_row = QHBoxLayout()
        first_row.addWidget(wl_group)
        first_row.addStretch(1)

        second_row = QHBoxLayout()
        second_row.addWidget(polarity_group)
        second_row.addWidget(current_group)
        second_row.addStretch(1)
        second_row.addWidget(save_button)
        second_row.addWidget(cancel_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self._table)
        layout.addLayout(first_row)
        layout.addLayout(second_row)

    def _make_choice_combo(self, values: list[str], current: str) -> QComboBox:
        combo = QComboBox()
        combo.addItems(values)
        combo.setCurrentText(current if current in values else values[0])
        combo.setEditable(False)
        return combo

    def apply_first_row_to_all(self) -> None:
        if self._table.rowCount() == 0:
            return
        width = self._table.item(0, 2).text() if self._table.item(0, 2) else ""
        length = self._table.item(0, 3).text() if self._table.item(0, 3) else ""
        vdd = self._table.item(0, 4).text() if self._table.item(0, 4) else ""
        high_vd = self._table.item(0, 5).text() if self._table.item(0, 5) else ""
        low_vd = self._table.item(0, 6).text() if self._table.item(0, 6) else ""
        threshold = self._table.item(0, 7).text() if self._table.item(0, 7) else ""
        for row in range(1, self._table.rowCount()):
            self._table.setItem(row, 2, QTableWidgetItem(width))
            self._table.setItem(row, 3, QTableWidgetItem(length))
            self._table.setItem(row, 4, QTableWidgetItem(vdd))
            self._table.setItem(row, 5, QTableWidgetItem(high_vd))
            self._table.setItem(row, 6, QTableWidgetItem(low_vd))
            self._table.setItem(row, 7, QTableWidgetItem(threshold))

    def _prompt_device_parameters(self) -> tuple[float, float, float, float, float, float] | None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Set Device Parameters")
        layout = QFormLayout(dialog)

        width_spin = QDoubleSpinBox(dialog)
        width_spin.setRange(0.0, 1_000_000_000.0)
        width_spin.setDecimals(6)

        length_spin = QDoubleSpinBox(dialog)
        length_spin.setRange(0.0, 1_000_000_000.0)
        length_spin.setDecimals(6)

        vdd_spin = QDoubleSpinBox(dialog)
        vdd_spin.setRange(-1_000_000.0, 1_000_000.0)
        vdd_spin.setDecimals(6)
        vdd_spin.setValue(1.2)

        high_vd_spin = QDoubleSpinBox(dialog)
        high_vd_spin.setRange(-1_000_000.0, 1_000_000.0)
        high_vd_spin.setDecimals(6)
        high_vd_spin.setValue(1.2)

        low_vd_spin = QDoubleSpinBox(dialog)
        low_vd_spin.setRange(-1_000_000.0, 1_000_000.0)
        low_vd_spin.setDecimals(6)
        low_vd_spin.setValue(0.05)

        threshold_spin = QDoubleSpinBox(dialog)
        threshold_spin.setRange(0.0, 1_000_000.0)
        threshold_spin.setDecimals(6)
        threshold_spin.setValue(40.0)

        layout.addRow("Width (nm)", width_spin)
        layout.addRow("Length (nm)", length_spin)
        layout.addRow("Vdd target (V)", vdd_spin)
        layout.addRow("High Vd (V)", high_vd_spin)
        layout.addRow("Low Vd (V)", low_vd_spin)
        layout.addRow("Vth Const. (nA)", threshold_spin)

        buttons = QHBoxLayout()
        ok_button = QPushButton("OK", dialog)
        ok_button.clicked.connect(dialog.accept)
        cancel_button = QPushButton("Cancel", dialog)
        cancel_button.clicked.connect(dialog.reject)
        buttons.addStretch(1)
        buttons.addWidget(ok_button)
        buttons.addWidget(cancel_button)
        layout.addRow(buttons)

        if dialog.exec() != QDialog.Accepted:
            return None
        return (
            width_spin.value(),
            length_spin.value(),
            vdd_spin.value(),
            high_vd_spin.value(),
            low_vd_spin.value(),
            threshold_spin.value(),
        )

    def set_selected_device_parameters(self) -> None:
        selected_rows = sorted({index.row() for index in self._table.selectionModel().selectedRows()})
        if not selected_rows:
            return
        values = self._prompt_device_parameters()
        if values is None:
            return
        width, length, vdd, high_vd, low_vd, threshold = values
        width_text = f"{width:g}"
        length_text = f"{length:g}"
        vdd_text = f"{vdd:g}"
        high_vd_text = f"{high_vd:g}"
        low_vd_text = f"{low_vd:g}"
        threshold_text = f"{threshold:g}"
        for row in selected_rows:
            self._table.setItem(row, 2, QTableWidgetItem(width_text))
            self._table.setItem(row, 3, QTableWidgetItem(length_text))
            self._table.setItem(row, 4, QTableWidgetItem(vdd_text))
            self._table.setItem(row, 5, QTableWidgetItem(high_vd_text))
            self._table.setItem(row, 6, QTableWidgetItem(low_vd_text))
            self._table.setItem(row, 7, QTableWidgetItem(threshold_text))

    def clear_all_dimensions(self) -> None:
        for row in range(self._table.rowCount()):
            self._table.setItem(row, 2, QTableWidgetItem(""))
            self._table.setItem(row, 3, QTableWidgetItem(""))

    def set_selected_polarity(self, polarity: str) -> None:
        selected_rows = sorted({index.row() for index in self._table.selectionModel().selectedRows()})
        if not selected_rows:
            selected_rows = list(range(self._table.rowCount()))
        for row in selected_rows:
            combo = self._table.cellWidget(row, 9)
            if isinstance(combo, QComboBox):
                combo.setCurrentText(polarity)

    def set_selected_current_direction(self, sign: float) -> None:
        selected_rows = sorted({index.row() for index in self._table.selectionModel().selectedRows()})
        if not selected_rows:
            selected_rows = list(range(self._table.rowCount()))
        label = "Negative" if sign < 0 else "Positive"
        for row in selected_rows:
            combo = self._table.cellWidget(row, 10)
            if isinstance(combo, QComboBox):
                combo.setCurrentText(label)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() in (2, 3, 7):
            self._update_threshold_current_for_row(item.row())

    def _update_threshold_current_for_row(self, row: int) -> None:
        width_text = self._table.item(row, 2).text().strip() if self._table.item(row, 2) else ""
        length_text = self._table.item(row, 3).text().strip() if self._table.item(row, 3) else ""
        threshold_text = self._table.item(row, 7).text().strip() if self._table.item(row, 7) else ""
        result_text = ""
        try:
            if width_text and length_text and threshold_text:
                width = float(width_text)
                length = float(length_text)
                threshold = float(threshold_text)
                if not math.isclose(length, 0.0):
                    result_text = f"{threshold * width / length:.6g}"
        except ValueError:
            result_text = ""

        self._table.blockSignals(True)
        self._table.setItem(row, 8, QTableWidgetItem(result_text))
        self._table.blockSignals(False)

    def values(self) -> dict[tuple[Any, ...], tuple[float, float]]:
        result: dict[tuple[Any, ...], tuple[float, float]] = {}
        for row, key in enumerate(self._device_keys):
            width_text = self._table.item(row, 2).text().strip()
            length_text = self._table.item(row, 3).text().strip()
            if not width_text and not length_text:
                continue
            if not width_text or not length_text:
                raise ValueError("Width and Length must both be filled or both left empty.")
            result[key] = (float(width_text), float(length_text))
        return result

    def vdd_by_device(self) -> dict[tuple[Any, ...], float]:
        result: dict[tuple[Any, ...], float] = {}
        for row, key in enumerate(self._device_keys):
            item = self._table.item(row, 4)
            text = item.text().strip() if item else ""
            result[key] = float(text) if text else 1.2
        return result

    def high_vd_by_device(self) -> dict[tuple[Any, ...], float]:
        result: dict[tuple[Any, ...], float] = {}
        for row, key in enumerate(self._device_keys):
            item = self._table.item(row, 5)
            text = item.text().strip() if item else ""
            result[key] = float(text) if text else 1.2
        return result

    def low_vd_by_device(self) -> dict[tuple[Any, ...], float]:
        result: dict[tuple[Any, ...], float] = {}
        for row, key in enumerate(self._device_keys):
            item = self._table.item(row, 6)
            text = item.text().strip() if item else ""
            result[key] = float(text) if text else 0.05
        return result

    def threshold_constant_by_device(self) -> dict[tuple[Any, ...], float]:
        result: dict[tuple[Any, ...], float] = {}
        for row, key in enumerate(self._device_keys):
            item = self._table.item(row, 7)
            text = item.text().strip() if item else ""
            result[key] = float(text) if text else 40.0
        return result

    def device_id_by_device(self) -> dict[tuple[Any, ...], str]:
        result: dict[tuple[Any, ...], str] = {}
        for row, key in enumerate(self._device_keys):
            item = self._table.item(row, 0)
            device_id = item.text().strip() if item else ""
            result[key] = device_id or f"D{row + 1}"
        return result

    def polarity_by_device(self) -> dict[tuple[Any, ...], str]:
        result: dict[tuple[Any, ...], str] = {}
        for row, key in enumerate(self._device_keys):
            combo = self._table.cellWidget(row, 9)
            polarity = combo.currentText().strip().upper() if isinstance(combo, QComboBox) else "NMOS"
            result[key] = "PMOS" if polarity == "PMOS" else "NMOS"
        return result

    def current_direction_by_device(self) -> dict[tuple[Any, ...], float]:
        result: dict[tuple[Any, ...], float] = {}
        for row, key in enumerate(self._device_keys):
            combo = self._table.cellWidget(row, 10)
            direction = combo.currentText().strip().lower() if isinstance(combo, QComboBox) else "positive"
            result[key] = -1.0 if direction.startswith("neg") else 1.0
        return result


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(6, 4))
        super().__init__(self.figure)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MOSx Parameter Calculator")
        self.resize(1280, 820)

        self.dataframe = pd.DataFrame()
        self.preview_model = DataFrameModel()
        self.results_model = DataFrameModel()
        self.device_lookup_model = DataFrameModel()
        self.device_dimensions: dict[tuple[Any, ...], tuple[float, float]] = {}
        self.device_polarity_by_key: dict[tuple[Any, ...], str] = {}
        self.device_current_direction_by_key: dict[tuple[Any, ...], float] = {}
        self.device_id_by_key: dict[tuple[Any, ...], str] = {}
        self.device_vdd_by_key: dict[tuple[Any, ...], float] = {}
        self.device_high_vd_by_key: dict[tuple[Any, ...], float] = {}
        self.device_low_vd_by_key: dict[tuple[Any, ...], float] = {}
        self.device_threshold_constant_by_key: dict[tuple[Any, ...], float] = {}
        self.results_by_key: dict[tuple[Any, ...], DeviceResult] = {}
        self.results_frame = pd.DataFrame()
        self.device_lookup_frame = pd.DataFrame()

        self._build_ui()
        self._apply_plot_defaults()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        toolbar_row = QHBoxLayout()
        import_button = QPushButton("Import CSV")
        import_button.clicked.connect(self.import_csv)
        self.calculate_button = QPushButton("Calculate")
        self.calculate_button.clicked.connect(self.calculate_results)
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        self.copy_button = QPushButton("Copy All Results")
        self.copy_button.clicked.connect(self.copy_results)
        self.system_theme_button = QPushButton("System")
        self.system_theme_button.clicked.connect(lambda: self.apply_color_scheme("system"))
        self.light_theme_button = QPushButton("Light")
        self.light_theme_button.clicked.connect(lambda: self.apply_color_scheme("light"))
        self.dark_theme_button = QPushButton("Dark")
        self.dark_theme_button.clicked.connect(lambda: self.apply_color_scheme("dark"))

        for button in [import_button, self.calculate_button, self.export_button, self.copy_button]:
            toolbar_row.addWidget(button)
        toolbar_row.addStretch(1)
        for button in [self.system_theme_button, self.light_theme_button, self.dark_theme_button]:
            toolbar_row.addWidget(button)
        root_layout.addLayout(toolbar_row)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        config_panel = QWidget()
        config_panel.setMinimumWidth(430)
        config_panel.setMaximumWidth(500)
        config_layout = QVBoxLayout(config_panel)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.setSpacing(8)

        mapping_group = QGroupBox("Column Mapping")
        mapping_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        mapping_layout = QFormLayout(mapping_group)

        self.group_columns_list = QListWidget()
        self.group_columns_list.setSelectionMode(QListWidget.MultiSelection)
        self.group_columns_list.setFixedHeight(118)
        mapping_layout.addRow("Device key columns", self.group_columns_list)
        self.dimensions_button = QPushButton("Device Setting")
        self.dimensions_button.clicked.connect(self.open_dimensions_dialog)
        mapping_layout.addRow("", self.dimensions_button)

        mapping_layout.addRow(GroupBoxSeparator(mapping_group))
        self.curve_mode_with_type_radio = QRadioButton("Long Table Mode")
        self.curve_mode_idvg_only_radio = QRadioButton("Wide Table Mode")
        self.curve_mode_with_type_radio.setChecked(True)
        mapping_layout.addRow("Curve mode", _row_widget(self.curve_mode_with_type_radio, self.curve_mode_idvg_only_radio))

        self.curve_mode_stack = QStackedWidget()
        self.curve_mode_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        curve_page_with_type = QWidget()
        curve_page_with_type_layout = QFormLayout(curve_page_with_type)
        curve_page_with_type_layout.setContentsMargins(0, 0, 0, 0)
        self.curve_type_column_combo = QComboBox()
        self.idvg_value_list = QListWidget()
        self.idvg_value_list.setSelectionMode(QListWidget.MultiSelection)
        self.idvg_value_list.setFixedHeight(118)
        curve_page_with_type_layout.addRow("Curve type column", self.curve_type_column_combo)
        curve_page_with_type_layout.addRow("IdVg curve", self.idvg_value_list)
        self.curve_mode_stack.addWidget(curve_page_with_type)

        curve_page_idvg_only = QWidget()
        curve_page_idvg_only_layout = QVBoxLayout(curve_page_idvg_only)
        curve_page_idvg_only_layout.setContentsMargins(0, 0, 0, 0)
        curve_page_idvg_only_layout.addWidget(QLabel("No curve selection needed (IdVg only)."))
        self.curve_mode_stack.addWidget(curve_page_idvg_only)

        mapping_layout.addRow(self.curve_mode_stack)

        mapping_layout.addRow(GroupBoxSeparator(mapping_group))
        self.bias_mode_column_radio = QRadioButton("Long Table Mode")
        self.bias_mode_columns_radio = QRadioButton("Wide Table Mode")
        self.bias_mode_column_radio.setChecked(True)
        mapping_layout.addRow("Drain bias mode", _row_widget(self.bias_mode_column_radio, self.bias_mode_columns_radio))

        self.bias_mode_stack = QStackedWidget()
        self.bias_mode_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        bias_page_by_column = QWidget()
        bias_page_by_column_layout = QFormLayout(bias_page_by_column)
        bias_page_by_column_layout.setContentsMargins(0, 0, 0, 0)
        self.drain_bias_column_combo = QComboBox()
        self.high_bias_value_combo = QComboBox()
        self.low_bias_value_combo = QComboBox()
        self.gate_voltage_column_combo = QComboBox()
        self.drain_current_column_combo = QComboBox()
        bias_page_by_column_layout.addRow("Drain bias column", self.drain_bias_column_combo)
        bias_page_by_column_layout.addRow("Value for high Vd", self.high_bias_value_combo)
        bias_page_by_column_layout.addRow("Value for low Vd", self.low_bias_value_combo)
        bias_page_by_column_layout.addRow("Gate voltage column", self.gate_voltage_column_combo)
        bias_page_by_column_layout.addRow("Drain current column", self.drain_current_column_combo)
        self.bias_mode_stack.addWidget(bias_page_by_column)

        bias_page_by_columns = QWidget()
        bias_page_by_columns_layout = QFormLayout(bias_page_by_columns)
        bias_page_by_columns_layout.setContentsMargins(0, 0, 0, 0)
        self.high_gate_voltage_column_combo = QComboBox()
        self.high_drain_current_column_combo = QComboBox()
        self.low_gate_voltage_column_combo = QComboBox()
        self.low_drain_current_column_combo = QComboBox()
        bias_page_by_columns_layout.addRow("High Vd gate voltage", self.high_gate_voltage_column_combo)
        bias_page_by_columns_layout.addRow("High Vd drain current", self.high_drain_current_column_combo)
        bias_page_by_columns_layout.addRow("Low Vd gate voltage", self.low_gate_voltage_column_combo)
        bias_page_by_columns_layout.addRow("Low Vd drain current", self.low_drain_current_column_combo)
        self.bias_mode_stack.addWidget(bias_page_by_columns)

        mapping_layout.addRow(self.bias_mode_stack)

        mapping_layout.addRow(GroupBoxSeparator(mapping_group))
        self.gate_voltage_unit_combo = QComboBox()
        self.gate_voltage_unit_combo.addItems([NONE_OPTION, "V", "mV"])
        self.gate_voltage_unit_combo.setCurrentText("V")
        self.drain_current_unit_combo = QComboBox()
        self.drain_current_unit_combo.addItems([NONE_OPTION, "uA", "nA", "pA", "mA", "A"])
        self.drain_current_unit_combo.setCurrentText("A")
        unit_row = QWidget()
        unit_layout = QGridLayout(unit_row)
        unit_layout.setContentsMargins(0, 0, 0, 0)
        unit_layout.setHorizontalSpacing(8)
        unit_layout.setVerticalSpacing(0)
        unit_row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        unit_row.setFixedHeight(32)
        self.gate_voltage_unit_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.drain_current_unit_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        unit_layout.addWidget(QLabel("Gate voltage unit"), 0, 0)
        unit_layout.addWidget(self.gate_voltage_unit_combo, 0, 1)
        unit_layout.addWidget(QLabel("Drain current unit"), 0, 2)
        unit_layout.addWidget(self.drain_current_unit_combo, 0, 3)
        unit_layout.setColumnStretch(0, 2)
        unit_layout.setColumnStretch(1, 1)
        unit_layout.setColumnStretch(2, 2)
        unit_layout.setColumnStretch(3, 1)
        mapping_layout.addRow(unit_row)
        config_layout.addWidget(mapping_group)
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setPlaceholderText("Imported file info, validation messages, and calculation notes will appear here.")
        self.status_text.setMinimumHeight(170)
        self.status_text.setMaximumHeight(220)
        config_layout.addWidget(self.status_text, 1)

        splitter.addWidget(config_panel)
        splitter.addWidget(right_panel := QTabWidget())
        splitter.setSizes([470, 830])
        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        self.preview_table = QTableView()
        self.preview_table.setModel(self.preview_model)
        self.preview_table.setAlternatingRowColors(True)
        preview_layout.addWidget(self.preview_table)
        right_panel.addTab(preview_tab, "Source Preview")

        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        self.results_table = QTableView()
        self.results_table.setModel(self.results_model)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self.open_results_context_menu)
        self.results_table.clicked.connect(self.sync_plot_device_with_selection)
        results_layout.addWidget(self.results_table)
        self.device_lookup_table = QTableView()
        self.device_lookup_table.setModel(self.device_lookup_model)
        self.device_lookup_table.setAlternatingRowColors(True)
        self.device_lookup_table.setMaximumHeight(160)
        self.device_lookup_table.setColumnWidth(0, 80)
        self.device_lookup_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.device_lookup_table)
        right_panel.addTab(results_tab, "Results")

        plot_tab = QWidget()
        plot_tab_layout = QVBoxLayout(plot_tab)

        plot_controls = QGroupBox("Plot Settings")
        plot_controls_layout = QVBoxLayout(plot_controls)
        plot_controls_layout.setContentsMargins(8, 8, 8, 8)
        plot_controls_layout.setSpacing(6)
        self.plot_device_combo = QComboBox()
        self.plot_vd_both_radio = QRadioButton("Both Vd")
        self.plot_vd_high_radio = QRadioButton("Sat Vd")
        self.plot_vd_low_radio = QRadioButton("Lin Vd")
        self.plot_vd_both_radio.setChecked(True)
        self.plot_abs_log_checkbox = QCheckBox("log10(|Y|)")
        device_row = QWidget()
        device_layout = QHBoxLayout(device_row)
        device_layout.setContentsMargins(0, 0, 0, 0)
        device_layout.setSpacing(8)
        device_layout.addWidget(QLabel("Device"))
        device_layout.addWidget(self.plot_device_combo, 1)
        self.plot_prev_button = QPushButton("Prev")
        self.plot_prev_button.clicked.connect(lambda: self._step_plot_device(-1))
        device_layout.addWidget(self.plot_prev_button)
        self.plot_next_button = QPushButton("Next")
        self.plot_next_button.clicked.connect(lambda: self._step_plot_device(1))
        device_layout.addWidget(self.plot_next_button)
        plot_controls_layout.addWidget(device_row)

        options_row = QWidget()
        options_layout = QHBoxLayout(options_row)
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(8)
        options_layout.addWidget(QLabel("Vd selection"))
        options_layout.addWidget(self.plot_vd_both_radio)
        options_layout.addWidget(self.plot_vd_high_radio)
        options_layout.addWidget(self.plot_vd_low_radio)
        options_layout.addSpacing(16)
        options_layout.addWidget(QLabel("Y axis"))
        options_layout.addWidget(self.plot_abs_log_checkbox)
        options_layout.addStretch(1)
        plot_controls_layout.addWidget(options_row)

        sg_row = QWidget()
        sg_layout = QHBoxLayout(sg_row)
        sg_layout.setContentsMargins(0, 0, 0, 0)
        sg_layout.setSpacing(8)
        sg_layout.addWidget(QLabel("SG window"))
        self.sg_window_spin = QSpinBox()
        self.sg_window_spin.setRange(5, 101)
        self.sg_window_spin.setSingleStep(2)
        self.sg_window_spin.setValue(11)
        sg_layout.addWidget(self.sg_window_spin)
        sg_layout.addWidget(QLabel("SG polyorder"))
        self.sg_polyorder_spin = QSpinBox()
        self.sg_polyorder_spin.setRange(1, 7)
        self.sg_polyorder_spin.setValue(3)
        sg_layout.addWidget(self.sg_polyorder_spin)
        self.vtgm_viewer_button = QPushButton("Vtgm Viewer")
        self.vtgm_viewer_button.setCheckable(True)
        self.vtgm_viewer_button.setChecked(False)
        self.vtgm_viewer_button.toggled.connect(self._toggle_vtgm_viewer)
        sg_layout.addWidget(self.vtgm_viewer_button)
        sg_layout.addStretch(1)
        plot_controls_layout.addWidget(sg_row)

        self.vtgm_viewer_group = QGroupBox("Vtgm Viewer")
        vtgm_viewer_layout = QFormLayout(self.vtgm_viewer_group)
        self.vtgm_viewer_mode_label = QLabel("-")
        self.vtgm_viewer_value_label = QLabel("-")
        self.vtgm_viewer_point_label = QLabel("-")
        self.vtgm_viewer_gm_label = QLabel("-")
        self.vtgm_viewer_intercept_label = QLabel("-")
        for label in [
            self.vtgm_viewer_mode_label,
            self.vtgm_viewer_value_label,
            self.vtgm_viewer_point_label,
            self.vtgm_viewer_gm_label,
            self.vtgm_viewer_intercept_label,
        ]:
            label.setTextFormat(Qt.PlainText)
            label.setStyleSheet("font-family: Consolas, 'Courier New', monospace;")
        vtgm_viewer_layout.addRow("Curve", self.vtgm_viewer_mode_label)
        vtgm_viewer_layout.addRow("Vtgm", self.vtgm_viewer_value_label)
        vtgm_viewer_layout.addRow("Selected point", self.vtgm_viewer_point_label)
        vtgm_viewer_layout.addRow("gm", self.vtgm_viewer_gm_label)
        vtgm_viewer_layout.addRow("X intercept", self.vtgm_viewer_intercept_label)
        plot_controls_layout.addWidget(self.vtgm_viewer_group)

        plot_controls.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        plot_controls.setMaximumHeight(plot_controls.sizeHint().height())
        plot_tab_layout.addWidget(plot_controls, 0)
        self.canvas = MplCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self.open_plot_context_menu)
        plot_tab_layout.addWidget(self.canvas, 1)
        right_panel.addTab(plot_tab, "Plot")

        self.curve_type_column_combo.currentTextChanged.connect(self.refresh_curve_value_choices)
        self.drain_bias_column_combo.currentTextChanged.connect(self.refresh_bias_value_choices)
        self.curve_mode_with_type_radio.toggled.connect(self._update_curve_mode_ui)
        self.bias_mode_column_radio.toggled.connect(self._update_bias_mode_ui)
        self.plot_vd_both_radio.toggled.connect(self.plot_selected_device)
        self.plot_vd_high_radio.toggled.connect(self.plot_selected_device)
        self.plot_vd_low_radio.toggled.connect(self.plot_selected_device)
        self.plot_abs_log_checkbox.toggled.connect(self.plot_selected_device)
        self.plot_device_combo.currentTextChanged.connect(self.plot_selected_device)
        self.sg_window_spin.valueChanged.connect(self._on_sg_settings_changed)
        self.sg_polyorder_spin.valueChanged.connect(self._on_sg_settings_changed)
        self._update_curve_mode_ui()
        self._update_bias_mode_ui()
        self.vtgm_viewer_group.setVisible(True)
        self._toggle_vtgm_viewer(False)

    def _apply_plot_defaults(self) -> None:
        self.canvas.figure.set_facecolor("#ffffff")
        for ax in self.canvas.figure.axes:
            ax.set_facecolor("#ffffff")
            ax.xaxis.label.set_color("#000000")
            ax.yaxis.label.set_color("#000000")
            ax.title.set_color("#000000")
            ax.tick_params(colors="#000000")
            for spine in ax.spines.values():
                spine.set_color("#000000")
            ax.grid(True, color="#d0d0d0", alpha=0.4)
        self.canvas.draw_idle()

    def _on_sg_settings_changed(self) -> None:
        window = self.sg_window_spin.value()
        if window % 2 == 0:
            self.sg_window_spin.blockSignals(True)
            self.sg_window_spin.setValue(window + 1)
            self.sg_window_spin.blockSignals(False)
            return
        max_poly = max(1, window - 1)
        if self.sg_polyorder_spin.value() > max_poly:
            self.sg_polyorder_spin.blockSignals(True)
            self.sg_polyorder_spin.setValue(max_poly)
            self.sg_polyorder_spin.blockSignals(False)
        if not self.results_frame.empty and not self.dataframe.empty:
            error = self.validate_config()
            if error is None:
                self.calculate_results()
                return
        self.plot_selected_device()

    def _toggle_vtgm_viewer(self, checked: bool) -> None:
        if not checked:
            self._update_vtgm_viewer(None)
        self.plot_selected_device()

    def _step_plot_device(self, step: int) -> None:
        count = self.plot_device_combo.count()
        if count <= 0:
            return
        current_index = self.plot_device_combo.currentIndex()
        if current_index < 0:
            current_index = 0
        next_index = (current_index + step) % count
        self.plot_device_combo.setCurrentIndex(next_index)
        self.plot_selected_device()

    def _fmt_viewer(self, value: float | None, unit: str = "") -> str:
        if value is None or pd.isna(value):
            return "-"
        number = float(value)
        if math.isclose(number, 0.0, abs_tol=1e-15):
            text = "0.000"
        elif abs(number) >= 1e4 or abs(number) < 1e-2:
            text = f"{number:.3e}"
        else:
            text = f"{number:.3f}"
        return f"{text} {unit}".rstrip()

    def _format_sat_lin_pair(self, sat_text: str, lin_text: str, sat_width: int) -> str:
        return f"{sat_text.ljust(max(sat_width, 1))} | {lin_text}"

    def _set_vtgm_viewer_details(self, title: str, value_v: float | None, details: dict[str, Any] | None) -> None:
        self.vtgm_viewer_mode_label.setText(title)
        self.vtgm_viewer_value_label.setText(self._fmt_viewer(value_v, "V") if value_v is not None else "-")
        if not details:
            self.vtgm_viewer_point_label.setText("-")
            self.vtgm_viewer_gm_label.setText("-")
            self.vtgm_viewer_intercept_label.setText("-")
            return
        vg_v = details.get("vg_v")
        original_id_a = details.get("original_id_a")
        gm_a_per_v = details.get("gm_a_per_v")
        x_intercept_v = details.get("x_intercept_v")
        point_text = "-"
        if vg_v is not None and original_id_a is not None:
            point_text = f"Vg = {self._fmt_viewer(vg_v, 'V')}, Id = {self._fmt_viewer(original_id_a, 'A')}"
        self.vtgm_viewer_point_label.setText(point_text)
        self.vtgm_viewer_gm_label.setText(self._fmt_viewer(gm_a_per_v, "A/V") if gm_a_per_v is not None else "-")
        self.vtgm_viewer_intercept_label.setText(self._fmt_viewer(x_intercept_v, "V") if x_intercept_v is not None else "-")

    def _update_vtgm_viewer(self, result: DeviceResult | None) -> None:
        if not self.vtgm_viewer_button.isChecked():
            self._set_vtgm_viewer_details("-", None, None)
            return
        if result is None:
            self._set_vtgm_viewer_details("-", None, None)
            return
        if self.plot_vd_high_radio.isChecked():
            self._set_vtgm_viewer_details("Sat Vd", result.vtgm_sat_v, result.details.get("vtgm_sat"))
            return
        if self.plot_vd_low_radio.isChecked():
            self._set_vtgm_viewer_details("Lin Vd", result.vtgm_lin_v, result.details.get("vtgm_lin"))
            return
        sat_text = f"Sat: {self._fmt_viewer(result.vtgm_sat_v, 'V')}" if result.vtgm_sat_v is not None else "Sat=-"
        lin_text = f"Lin: {self._fmt_viewer(result.vtgm_lin_v, 'V')}" if result.vtgm_lin_v is not None else "Lin=-"
        self.vtgm_viewer_mode_label.setText("Both Vd")
        pair_rows: list[tuple[str, str]] = [(sat_text, lin_text)]
        sat_details = result.details.get("vtgm_sat")
        lin_details = result.details.get("vtgm_lin")
        point_sat = "Sat=-"
        point_lin = "Lin=-"
        gm_sat = "Sat=-"
        gm_lin = "Lin=-"
        intercept_sat = "Sat=-"
        intercept_lin = "Lin=-"
        if sat_details:
            point_sat = (
                f"Sat: Vg = {self._fmt_viewer(sat_details.get('vg_v'), 'V')}, "
                f"Id = {self._fmt_viewer(sat_details.get('original_id_a'), 'A')}"
            )
            gm_sat = f"Sat: {self._fmt_viewer(sat_details.get('gm_a_per_v'), 'A/V')}"
            intercept_sat = f"Sat: {self._fmt_viewer(sat_details.get('x_intercept_v'), 'V')}"
        if lin_details:
            point_lin = (
                f"Lin: Vg = {self._fmt_viewer(lin_details.get('vg_v'), 'V')}, "
                f"Id = {self._fmt_viewer(lin_details.get('original_id_a'), 'A')}"
            )
            gm_lin = f"Lin: {self._fmt_viewer(lin_details.get('gm_a_per_v'), 'A/V')}"
            intercept_lin = f"Lin: {self._fmt_viewer(lin_details.get('x_intercept_v'), 'V')}"
        pair_rows.extend(
            [
                (point_sat, point_lin),
                (gm_sat, gm_lin),
                (intercept_sat, intercept_lin),
            ]
        )
        sat_width = max(len(left) for left, _ in pair_rows)
        self.vtgm_viewer_value_label.setText(self._format_sat_lin_pair(sat_text, lin_text, sat_width))
        self.vtgm_viewer_point_label.setText(self._format_sat_lin_pair(point_sat, point_lin, sat_width))
        self.vtgm_viewer_gm_label.setText(self._format_sat_lin_pair(gm_sat, gm_lin, sat_width))
        self.vtgm_viewer_intercept_label.setText(self._format_sat_lin_pair(intercept_sat, intercept_lin, sat_width))

    def _current_savgol_settings(self, point_count: int) -> tuple[int, int]:
        window_length = self.sg_window_spin.value() if hasattr(self, "sg_window_spin") else 11
        polyorder = self.sg_polyorder_spin.value() if hasattr(self, "sg_polyorder_spin") else 3
        if window_length % 2 == 0:
            window_length += 1
        if point_count % 2 == 0:
            point_limit = point_count - 1
        else:
            point_limit = point_count
        window_length = min(window_length, point_limit)
        if window_length < 3:
            return 0, polyorder
        polyorder = min(polyorder, window_length - 1)
        return window_length, polyorder

    def _update_curve_mode_ui(self) -> None:
        # 0 = needs curve type column, 1 = IdVg-only
        if self.curve_mode_with_type_radio.isChecked():
            self.curve_mode_stack.setCurrentIndex(0)
        else:
            self.curve_mode_stack.setCurrentIndex(1)

    def _update_bias_mode_ui(self) -> None:
        # 0 = bias by column, 1 = separate columns
        if self.bias_mode_column_radio.isChecked():
            self.bias_mode_stack.setCurrentIndex(0)
        else:
            self.bias_mode_stack.setCurrentIndex(1)

    def apply_color_scheme(self, mode: str) -> None:
        labels = {
            "system": "system",
            "light": "light",
            "dark": "dark",
        }
        if set_app_color_scheme(mode):
            self.log(f"Requested {labels.get(mode, mode)} color scheme via Qt official API.")
        else:
            self.log("Qt color scheme override API is not available in this PySide6/Qt build.")

    def log(self, text: str) -> None:
        lines = str(text).splitlines() or [""]
        formatted_lines = []
        for index, line in enumerate(lines):
            prefix = "> " if index == 0 else "  "
            formatted_lines.append(f"{prefix}{line}")
        if self.status_text.toPlainText():
            self.status_text.append("")
        self.status_text.append("\n".join(formatted_lines))

    def import_csv(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv);;All Files (*.*)")
        if not file_path:
            return
        try:
            frame = pd.read_csv(file_path)
        except Exception as exc:
            info_message(f"Failed to read CSV:\n{exc}")
            return

        self.dataframe = frame
        self.preview_model.set_frame(frame.head(500))
        self._populate_mapping_controls(frame.columns.tolist())
        self.log(f"Loaded file: {file_path}")
        self.log(f"Rows: {len(frame)}, Columns: {len(frame.columns)}")

    def _populate_mapping_controls(self, columns: list[str]) -> None:
        self.group_columns_list.clear()
        for column in columns:
            item = QListWidgetItem(column)
            self.group_columns_list.addItem(item)

        for combo in [
            self.curve_type_column_combo,
            self.drain_bias_column_combo,
            self.gate_voltage_column_combo,
            self.drain_current_column_combo,
            self.high_gate_voltage_column_combo,
            self.high_drain_current_column_combo,
            self.low_gate_voltage_column_combo,
            self.low_drain_current_column_combo,
        ]:
            combo.clear()
            combo.addItem(NONE_OPTION)
            combo.addItems(columns)

        self.refresh_curve_value_choices()
        self.refresh_bias_value_choices()

    def selected_group_columns(self) -> list[str]:
        return [item.text() for item in self.group_columns_list.selectedItems()]

    def refresh_curve_value_choices(self) -> None:
        self.idvg_value_list.clear()

        column = combo_text(self.curve_type_column_combo)
        if self.dataframe.empty or not column or column not in self.dataframe.columns:
            return

        values = sorted(self.dataframe[column].dropna().astype(str).unique().tolist())
        for value in values:
            item = QListWidgetItem(value)
            self.idvg_value_list.addItem(item)

    def refresh_bias_value_choices(self) -> None:
        self.high_bias_value_combo.clear()
        self.low_bias_value_combo.clear()

        column = combo_text(self.drain_bias_column_combo)
        if self.dataframe.empty or not column or column not in self.dataframe.columns:
            self.high_bias_value_combo.addItem(NONE_OPTION)
            self.low_bias_value_combo.addItem(NONE_OPTION)
            return

        values = sorted(self.dataframe[column].dropna().astype(str).unique().tolist())
        self.high_bias_value_combo.addItem(NONE_OPTION)
        self.low_bias_value_combo.addItem(NONE_OPTION)
        self.high_bias_value_combo.addItems(values)
        self.low_bias_value_combo.addItems(values)

    def _device_keys(self) -> list[tuple[Any, ...]]:
        group_columns = self.selected_group_columns()
        if not group_columns:
            return []
        grouped = self.dataframe[group_columns].drop_duplicates().fillna("")
        return [tuple(row[column] for column in group_columns) for _, row in grouped.iterrows()]

    def _ensure_device_ids(self, device_keys: list[tuple[Any, ...]]) -> None:
        for index, key in enumerate(device_keys, start=1):
            if not self.device_id_by_key.get(key):
                self.device_id_by_key[key] = f"D{index}"

    def open_dimensions_dialog(self) -> None:
        if self.dataframe.empty:
            info_message("Please import a CSV file first.")
            return

        group_columns = self.selected_group_columns()
        if not group_columns:
            info_message("Please select the columns used to distinguish each transistor.")
            return

        device_keys = self._device_keys()
        if not device_keys:
            info_message("No device keys found under the current grouping columns.")
            return
        self._ensure_device_ids(device_keys)
        dialog = WlDialog(
            device_keys,
            self.device_dimensions,
            self.device_id_by_key,
            self.device_vdd_by_key,
            self.device_high_vd_by_key,
            self.device_low_vd_by_key,
            self.device_threshold_constant_by_key,
            self.device_polarity_by_key,
            self.device_current_direction_by_key,
            self,
        )
        if dialog.exec() == QDialog.Accepted:
            try:
                self.device_dimensions = dialog.values()
            except ValueError:
                info_message("Width / Length must be valid numeric values.")
                return
            self.device_id_by_key = dialog.device_id_by_device()
            self.device_vdd_by_key = dialog.vdd_by_device()
            self.device_high_vd_by_key = dialog.high_vd_by_device()
            self.device_low_vd_by_key = dialog.low_vd_by_device()
            self.device_threshold_constant_by_key = dialog.threshold_constant_by_device()
            self.device_polarity_by_key = dialog.polarity_by_device()
            self.device_current_direction_by_key = dialog.current_direction_by_device()
            self.log(f"Saved device IDs for {len(self.device_id_by_key)} devices.")
            self.log(f"Saved Vdd target for {len(self.device_vdd_by_key)} devices.")
            self.log(f"Saved High Vd for {len(self.device_high_vd_by_key)} devices.")
            self.log(f"Saved Low Vd for {len(self.device_low_vd_by_key)} devices.")
            self.log(f"Saved threshold constant for {len(self.device_threshold_constant_by_key)} devices.")
            self.log(f"Saved Width / Length for {len(self.device_dimensions)} devices.")
            self.log(f"Saved polarity for {len(self.device_polarity_by_key)} devices.")
            self.log(f"Saved current direction for {len(self.device_current_direction_by_key)} devices.")
            self.refresh_plot_devices()

    def _config(self) -> dict[str, Any]:
        curve_mode = None
        if self.curve_mode_with_type_radio.isChecked():
            curve_mode = "with_type"
        elif self.curve_mode_idvg_only_radio.isChecked():
            curve_mode = "idvg_only"

        bias_mode = None
        if self.bias_mode_column_radio.isChecked():
            bias_mode = "by_column"
        elif self.bias_mode_columns_radio.isChecked():
            bias_mode = "by_columns"
        return {
            "group_columns": self.selected_group_columns(),
            "curve_mode": curve_mode,
            "bias_mode": bias_mode,
            "curve_type_column": combo_text(self.curve_type_column_combo),
            "idvg_values": selected_list_texts(self.idvg_value_list),
            "drain_bias_column": combo_text(self.drain_bias_column_combo),
            "high_bias_value": combo_text(self.high_bias_value_combo),
            "low_bias_value": combo_text(self.low_bias_value_combo),
            "gate_voltage_column": combo_text(self.gate_voltage_column_combo),
            "drain_current_column": combo_text(self.drain_current_column_combo),
            "high_gate_voltage_column": combo_text(self.high_gate_voltage_column_combo),
            "high_drain_current_column": combo_text(self.high_drain_current_column_combo),
            "low_gate_voltage_column": combo_text(self.low_gate_voltage_column_combo),
            "low_drain_current_column": combo_text(self.low_drain_current_column_combo),
            "gate_voltage_unit": combo_text(self.gate_voltage_unit_combo),
            "drain_current_unit": combo_text(self.drain_current_unit_combo),
        }

    def validate_config(self) -> str | None:
        config = self._config()
        if config["curve_mode"] not in ("with_type", "idvg_only"):
            return "Please select Curve mode."
        if config["bias_mode"] not in ("by_column", "by_columns"):
            return "Please select Drain bias mode."
        if not config["group_columns"]:
            return "Missing required mapping: group_columns"
        if not config["gate_voltage_unit"]:
            return "Missing required mapping: gate_voltage_unit"
        if not config["drain_current_unit"]:
            return "Missing required mapping: drain_current_unit"

        if config["curve_mode"] == "with_type":
            if not config["curve_type_column"]:
                return "Missing required mapping: curve_type_column"
            if not config["idvg_values"]:
                return "Missing required mapping: idvg_values"

        if config["bias_mode"] == "by_column":
            required = [
                "drain_bias_column",
                "gate_voltage_column",
                "drain_current_column",
            ]
            for key in required:
                if not config.get(key):
                    return f"Missing required mapping: {key}"
            if not config["high_bias_value"] and not config["low_bias_value"]:
                return "Please select at least one Drain bias value."
        else:
            has_high = bool(config["high_gate_voltage_column"] and config["high_drain_current_column"])
            has_low = bool(config["low_gate_voltage_column"] and config["low_drain_current_column"])
            if not has_high and not has_low:
                return "Please select at least one High/Low Vd column pair."
            if bool(config["high_gate_voltage_column"]) != bool(config["high_drain_current_column"]):
                return "High Vd gate/drain current columns must both be set or both be empty."
            if bool(config["low_gate_voltage_column"]) != bool(config["low_drain_current_column"]):
                return "Low Vd gate/drain current columns must both be set or both be empty."
        return None

    def _extract_idvg_curves(
        self,
        device_frame: pd.DataFrame,
        config: dict[str, Any],
        vg_scale: float,
        id_scale: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        base = device_frame.copy()

        if config["curve_mode"] == "with_type":
            curve_col = config["curve_type_column"]
            base[curve_col] = base[curve_col].astype(str)
            base = base[base[curve_col].isin([str(value) for value in config["idvg_values"]])].copy()

        if base.empty:
            return pd.DataFrame(), pd.DataFrame()

        if config["bias_mode"] == "by_column":
            bias_col = config["drain_bias_column"]
            base[bias_col] = base[bias_col].astype(str)

            base["_vg_v"] = pd.to_numeric(base[config["gate_voltage_column"]], errors="coerce") * vg_scale
            base["_id_a"] = pd.to_numeric(base[config["drain_current_column"]], errors="coerce") * id_scale
            base = base.dropna(subset=["_vg_v", "_id_a"])
            if base.empty:
                return pd.DataFrame(), pd.DataFrame()

            high_frame = pd.DataFrame()
            low_frame = pd.DataFrame()
            if config["high_bias_value"]:
                high_frame = base[base[bias_col] == str(config["high_bias_value"])].copy().sort_values("_vg_v")
            if config["low_bias_value"]:
                low_frame = base[base[bias_col] == str(config["low_bias_value"])].copy().sort_values("_vg_v")
            return high_frame, low_frame

        high = pd.DataFrame()
        low = pd.DataFrame()
        if (
            config["high_gate_voltage_column"]
            and config["high_drain_current_column"]
            and config["high_gate_voltage_column"] in base.columns
            and config["high_drain_current_column"] in base.columns
        ):
            high = base[[config["high_gate_voltage_column"], config["high_drain_current_column"]]].copy()
            high["_vg_v"] = pd.to_numeric(high[config["high_gate_voltage_column"]], errors="coerce") * vg_scale
            high["_id_a"] = pd.to_numeric(high[config["high_drain_current_column"]], errors="coerce") * id_scale
            high = high.dropna(subset=["_vg_v", "_id_a"]).sort_values("_vg_v")
        if (
            config["low_gate_voltage_column"]
            and config["low_drain_current_column"]
            and config["low_gate_voltage_column"] in base.columns
            and config["low_drain_current_column"] in base.columns
        ):
            low = base[[config["low_gate_voltage_column"], config["low_drain_current_column"]]].copy()
            low["_vg_v"] = pd.to_numeric(low[config["low_gate_voltage_column"]], errors="coerce") * vg_scale
            low["_id_a"] = pd.to_numeric(low[config["low_drain_current_column"]], errors="coerce") * id_scale
            low = low.dropna(subset=["_vg_v", "_id_a"]).sort_values("_vg_v")
        return high, low

    def calculate_results(self) -> None:
        if self.dataframe.empty:
            info_message("Please import a CSV file first.")
            return

        error = self.validate_config()
        if error:
            info_message(error)
            return

        config = self._config()
        vg_scale = VOLTAGE_UNITS_TO_VOLT[config["gate_voltage_unit"]]
        id_scale = CURRENT_UNITS_TO_AMP[config["drain_current_unit"]]

        results: list[DeviceResult] = []
        device_keys = []
        for device_key, device_frame in self.dataframe.groupby(config["group_columns"], dropna=False):
            if not isinstance(device_key, tuple):
                device_key = (device_key,)
            device_keys.append(device_key)
        self._ensure_device_ids(device_keys)

        for device_key, device_frame in self.dataframe.groupby(config["group_columns"], dropna=False):
            if not isinstance(device_key, tuple):
                device_key = (device_key,)
            dimensions = self.device_dimensions.get(device_key)
            if dimensions is None:
                self.log(f"Left result blank because W/L is not set: {device_key}")
                results.append(self._empty_device_result(device_key))
                continue

            width_nm, length_nm = dimensions
            if length_nm == 0:
                self.log(f"Left result blank because length is zero: {device_key}")
                results.append(self._empty_device_result(device_key, width_nm, length_nm))
                continue

            high_curve, low_curve = self._extract_idvg_curves(device_frame, config, vg_scale, id_scale)
            results.append(self._calculate_device(device_key, high_curve, low_curve, width_nm, length_nm))

        if not results:
            info_message("No valid devices were calculated. Please check your mappings and numeric columns.")
            return

        self.results_by_key = {result.device_key: result for result in results}
        parameter_rows = [
            ("Width (nm)", {result.device_id: result.width_nm for result in results}),
            ("Length (nm)", {result.device_id: result.length_nm for result in results}),
            ("Idoff (pA)", {result.device_id: result.idoff_pa for result in results}),
            ("Idoff/L (pA/nm)", {result.device_id: result.idoff_pa_per_nm for result in results}),
            ("Ids (uA)", {result.device_id: result.ids_ua for result in results}),
            ("Ids/L (uA/nm)", {result.device_id: result.ids_ua_per_nm for result in results}),
            ("Idl (uA)", {result.device_id: result.idl_ua for result in results}),
            ("Idl/L (uA/nm)", {result.device_id: result.idl_ua_per_nm for result in results}),
            ("Vts (V)", {result.device_id: result.vts_v for result in results}),
            ("Vtl (V)", {result.device_id: result.vtl_v for result in results}),
            ("Vtgm_sat (V)", {result.device_id: result.vtgm_sat_v for result in results}),
            ("Vtgm_lin (V)", {result.device_id: result.vtgm_lin_v for result in results}),
        ]
        self.results_frame = pd.DataFrame(
            [{"Parameter": parameter_name, **values_by_device} for parameter_name, values_by_device in parameter_rows]
        )
        self.device_lookup_frame = pd.DataFrame(
            [{"ID": result.device_id, "Device": result.display_key} for result in results]
        )
        self.results_model.set_frame(self.results_frame)
        self.device_lookup_model.set_frame(self.device_lookup_frame)
        self.log(
            "Calculation settings: "
            "Per-device Vdd / High Vd / Low Vd / Vth constant / current direction applied."
        )
        self.log(f"Calculated {len(results)} devices.")
        self.refresh_plot_devices()
        self.plot_selected_device()

    def _empty_device_result(
        self,
        device_key: tuple[Any, ...],
        width_nm: float | None = None,
        length_nm: float | None = None,
    ) -> DeviceResult:
        return DeviceResult(
            device_key=device_key,
            device_id=self.device_id_by_key.get(device_key, "N/A"),
            display_key=" | ".join(str(part) for part in device_key),
            width_nm=width_nm,
            length_nm=length_nm,
            idoff_pa=None,
            idoff_pa_per_nm=None,
            ids_ua=None,
            ids_ua_per_nm=None,
            idl_ua=None,
            idl_ua_per_nm=None,
            vts_v=None,
            vtl_v=None,
            vtgm_sat_v=None,
            vtgm_lin_v=None,
            details={},
        )

    def _calculate_device(
        self,
        device_key: tuple[Any, ...],
        high_frame: pd.DataFrame,
        low_frame: pd.DataFrame,
        width_nm: float,
        length_nm: float,
    ) -> DeviceResult:
        polarity = self.device_polarity_by_key.get(device_key, "NMOS")
        current_direction_sign = self.device_current_direction_by_key.get(
            device_key,
            -1.0 if polarity == "PMOS" else 1.0,
        )
        vdd_target_v = self.device_vdd_by_key.get(device_key, 1.2)
        high_vd_v = self.device_high_vd_by_key.get(device_key, 1.2)
        low_vd_v = self.device_low_vd_by_key.get(device_key, 0.05)
        threshold_constant_na = self.device_threshold_constant_by_key.get(device_key, 40.0)
        idoff_row = self._closest_row(high_frame, target=0.0, column="_vg_v")
        ids_row = self._closest_abs_row(high_frame, target=abs(vdd_target_v), column="_vg_v")
        idl_row = self._closest_abs_row(low_frame, target=abs(vdd_target_v), column="_vg_v")

        target_current_a = current_direction_sign * threshold_constant_na * 1e-9 * (width_nm / length_nm)
        vts, vts_details = self._interpolate_vg_for_current(high_frame, target_current_a)
        vtl, vtl_details = self._interpolate_vg_for_current(low_frame, target_current_a)
        vtgm_sat, vtgm_sat_details = self._calculate_vtgm(high_frame, high_vd_v, polarity)
        vtgm_lin, vtgm_lin_details = self._calculate_vtgm(low_frame, low_vd_v, polarity)

        idoff_pa = self._current_display_value(idoff_row, 1e12)
        ids_ua = self._current_display_value(ids_row, 1e6)
        idl_ua = self._current_display_value(idl_row, 1e6)

        idoff_pa_per_nm = idoff_pa / length_nm if idoff_pa is not None else None
        ids_ua_per_nm = ids_ua / length_nm if ids_ua is not None else None
        idl_ua_per_nm = idl_ua / length_nm if idl_ua is not None else None

        details = {
            "high_curve": high_frame,
            "low_curve": low_frame,
            "idoff_point": None if idoff_row is None else {"vg": float(idoff_row["_vg_v"]), "id": float(idoff_row["_id_a"]), "current_direction_sign": current_direction_sign},
            "ids_point": None if ids_row is None else {"vg": float(ids_row["_vg_v"]), "id": float(ids_row["_id_a"]), "current_direction_sign": current_direction_sign},
            "idl_point": None if idl_row is None else {"vg": float(idl_row["_vg_v"]), "id": float(idl_row["_id_a"]), "current_direction_sign": current_direction_sign},
            "vts": vts_details,
            "vtl": vtl_details,
            "vtgm_sat": vtgm_sat_details,
            "vtgm_lin": vtgm_lin_details,
            "target_current_a": target_current_a,
            "vdd_target_v": vdd_target_v,
            "high_vd_v": high_vd_v,
            "low_vd_v": low_vd_v,
            "threshold_constant_na": threshold_constant_na,
            "current_direction_sign": current_direction_sign,
            "polarity": polarity,
        }

        return DeviceResult(
            device_key=device_key,
            device_id=self.device_id_by_key.get(device_key, "N/A"),
            display_key=" | ".join(str(part) for part in device_key),
            width_nm=width_nm,
            length_nm=length_nm,
            idoff_pa=idoff_pa,
            idoff_pa_per_nm=idoff_pa_per_nm,
            ids_ua=ids_ua,
            ids_ua_per_nm=ids_ua_per_nm,
            idl_ua=idl_ua,
            idl_ua_per_nm=idl_ua_per_nm,
            vts_v=vts,
            vtl_v=vtl,
            vtgm_sat_v=vtgm_sat,
            vtgm_lin_v=vtgm_lin,
            details=details,
        )

    def _closest_row(self, frame: pd.DataFrame, target: float, column: str) -> pd.Series | None:
        if frame.empty or pd.isna(target):
            return None
        index = (frame[column] - target).abs().idxmin()
        return frame.loc[index]

    def _closest_abs_row(self, frame: pd.DataFrame, target: float, column: str) -> pd.Series | None:
        if frame.empty or pd.isna(target):
            return None
        index = (frame[column].abs() - abs(target)).abs().idxmin()
        return frame.loc[index]

    def _current_display_value(self, row: pd.Series | None, scale: float) -> float | None:
        if row is None:
            return None
        current_a = float(row["_id_a"])
        return current_a * scale

    def _interpolate_vg_for_current(
        self,
        frame: pd.DataFrame,
        target_current_a: float,
    ) -> tuple[float | None, dict[str, Any] | None]:
        if len(frame) < 2:
            return None, None

        work = frame[["_vg_v", "_id_a"]].copy().sort_values("_vg_v")
        work["_effective_id_a"] = work["_id_a"]
        values = work.to_dict("records")

        for left, right in zip(values, values[1:]):
            y1 = left["_effective_id_a"]
            y2 = right["_effective_id_a"]
            if y1 == target_current_a:
                return float(left["_vg_v"]), {"method": "exact", "left": left, "right": right}
            if (y1 - target_current_a) * (y2 - target_current_a) <= 0 and y1 != y2:
                ratio = (target_current_a - y1) / (y2 - y1)
                vg = left["_vg_v"] + ratio * (right["_vg_v"] - left["_vg_v"])
                return float(vg), {"method": "interpolated", "left": left, "right": right}

        work["_distance"] = (work["_effective_id_a"] - target_current_a).abs()
        nearest = work.nsmallest(2, "_distance").sort_values("_effective_id_a")
        if len(nearest) < 2:
            return None, None

        left = nearest.iloc[0].to_dict()
        right = nearest.iloc[1].to_dict()
        if math.isclose(left["_effective_id_a"], right["_effective_id_a"]):
            return float(left["_vg_v"]), {"method": "nearest", "left": left, "right": right}

        ratio = (target_current_a - left["_effective_id_a"]) / (right["_effective_id_a"] - left["_effective_id_a"])
        vg = left["_vg_v"] + ratio * (right["_vg_v"] - left["_vg_v"])
        return float(vg), {"method": "nearest-interpolated", "left": left, "right": right}

    def _savgol_window_length(self, point_count: int) -> int:
        window_length, _ = self._current_savgol_settings(point_count)
        return window_length

    def _savgol_smooth_and_derivative(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        window_length: int | None = None,
        polyorder: int = 3,
    ) -> tuple[np.ndarray, np.ndarray]:
        point_count = len(x_values)
        if point_count < 3:
            return y_values.copy(), np.full(point_count, np.nan, dtype=float)

        if window_length is None:
            window_length, polyorder = self._current_savgol_settings(point_count)
        if window_length <= 0:
            return y_values.copy(), np.full(point_count, np.nan, dtype=float)

        window_length = min(window_length, point_count if point_count % 2 == 1 else point_count - 1)
        if window_length < 3:
            return y_values.copy(), np.full(point_count, np.nan, dtype=float)

        polyorder = min(polyorder, window_length - 1)
        half_window = window_length // 2
        smoothed = np.empty(point_count, dtype=float)
        derivative = np.empty(point_count, dtype=float)

        for index in range(point_count):
            start = max(0, index - half_window)
            stop = min(point_count, index + half_window + 1)
            while stop - start < window_length:
                if start > 0:
                    start -= 1
                elif stop < point_count:
                    stop += 1
                else:
                    break

            local_x = x_values[start:stop] - x_values[index]
            local_y = y_values[start:stop]
            effective_order = min(polyorder, len(local_x) - 1)
            if len(local_x) <= effective_order:
                smoothed[index] = y_values[index]
                derivative[index] = np.nan
                continue

            coefficients = np.polyfit(local_x, local_y, effective_order)
            smoothed[index] = np.polyval(coefficients, 0.0)
            derivative_coefficients = np.polyder(coefficients)
            derivative[index] = np.polyval(derivative_coefficients, 0.0) if len(derivative_coefficients) else 0.0

        return smoothed, derivative

    def _calculate_vtgm(
        self,
        frame: pd.DataFrame,
        drain_bias_v: float,
        polarity: str,
    ) -> tuple[float | None, dict[str, Any] | None]:
        if len(frame) < 3:
            return None, None

        work = frame[["_vg_v", "_id_a"]].dropna().sort_values("_vg_v")
        if len(work) < 3:
            return None, None

        x_values = work["_vg_v"].to_numpy(dtype=float)
        y_values = work["_id_a"].to_numpy(dtype=float)
        window_length, polyorder = self._current_savgol_settings(len(x_values))
        smoothed_y, smoothed_gm = self._savgol_smooth_and_derivative(
            x_values,
            y_values,
            window_length=window_length,
            polyorder=polyorder,
        )
        valid_mask = np.isfinite(smoothed_gm) & ~np.isclose(smoothed_gm, 0.0)
        if not valid_mask.any():
            return None, None

        valid_indices = np.flatnonzero(valid_mask)
        best_index = int(valid_indices[np.argmax(np.abs(smoothed_gm[valid_mask]))])
        x0 = float(x_values[best_index])
        y0 = float(smoothed_y[best_index])
        original_y0 = float(y_values[best_index])
        gm0 = float(smoothed_gm[best_index])
        x_intercept = x0 - (y0 / gm0)
        vtgm_magnitude = abs(x_intercept) - abs(drain_bias_v) / 2.0
        vtgm_value = -vtgm_magnitude if str(polarity).upper() == "PMOS" else vtgm_magnitude

        details = {
            "method": "savgol-tangent",
            "vg_v": x0,
            "id_a": y0,
            "original_id_a": original_y0,
            "gm_a_per_v": gm0,
            "x_intercept_v": x_intercept,
            "drain_bias_v": drain_bias_v,
            "vtgm_v": vtgm_value,
            "window_length": window_length,
            "polyorder": polyorder,
        }
        return float(vtgm_value), details

    def _curve_derivative_series(
        self,
        curve: pd.DataFrame,
        y_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(curve) < 2:
            return np.array([]), np.array([]), np.array([]), np.array([])

        work = curve[["_vg_v", "_id_a"]].dropna().sort_values("_vg_v").copy()
        if len(work) < 2:
            return np.array([]), np.array([]), np.array([]), np.array([])

        x_values = work["_vg_v"].to_numpy(dtype=float)
        y_values = (work["_id_a"] * y_scale).to_numpy(dtype=float)
        delta_x = np.diff(x_values)
        valid = ~np.isclose(delta_x, 0.0)
        if not valid.any():
            return np.array([]), np.array([]), np.array([]), np.array([])

        mid_x = (x_values[:-1] + x_values[1:]) / 2
        slopes = np.diff(y_values) / delta_x
        window_length, polyorder = self._current_savgol_settings(len(x_values))
        smooth_y, smooth_slopes = self._savgol_smooth_and_derivative(
            x_values,
            y_values,
            window_length=window_length,
            polyorder=polyorder,
        )
        smooth_valid = np.isfinite(smooth_slopes)
        return mid_x[valid], slopes[valid], x_values[smooth_valid], smooth_slopes[smooth_valid]

    def _fmt(self, value: float | None) -> str:
        return "" if value is None or pd.isna(value) else f"{value:.6g}"

    def _axis_tick_label(self, value: float, _: float) -> str:
        if math.isclose(value, 0.0, abs_tol=1e-15):
            return "0"
        return f"{value:.2e}" if abs(value) >= 1e4 or abs(value) < 1e-2 else f"{value:.2f}"

    def refresh_plot_devices(self) -> None:
        current = self.plot_device_combo.currentText()
        self.plot_device_combo.clear()
        names = [result.device_id for result in self.results_by_key.values()]
        self.plot_device_combo.addItems(names)
        if current in names:
            self.plot_device_combo.setCurrentText(current)

    def sync_plot_device_with_selection(self, index: QModelIndex) -> None:
        if not index.isValid() or self.results_frame.empty:
            return
        if index.column() == 0:
            return
        device_id = str(self.results_frame.columns[index.column()])
        self.plot_device_combo.setCurrentText(device_id)

    def plot_selected_device(self) -> None:
        self.canvas.figure.clear()
        if not self.results_by_key:
            self._update_vtgm_viewer(None)
            self._apply_plot_defaults()
            self.canvas.draw()
            return

        selected_name = self.plot_device_combo.currentText()
        result = next((item for item in self.results_by_key.values() if item.device_id == selected_name), None)
        if result is None:
            self._update_vtgm_viewer(None)
            self._apply_plot_defaults()
            self.canvas.draw()
            return
        self._update_vtgm_viewer(result)

        config = self._config()
        if not config["gate_voltage_unit"] or not config["drain_current_unit"]:
            ax = self.canvas.figure.add_subplot(111)
            ax.set_title("Please select Gate/Drain current units before plotting")
            self._apply_plot_defaults()
            self.canvas.draw()
            return
        if not any(
            [
                self.plot_vd_both_radio.isChecked(),
                self.plot_vd_high_radio.isChecked(),
                self.plot_vd_low_radio.isChecked(),
            ]
        ):
            ax = self.canvas.figure.add_subplot(111)
            ax.set_title("Please select a Vd plot mode")
            self._apply_plot_defaults()
            self.canvas.draw()
            return
        vg_scale = VOLTAGE_UNITS_TO_VOLT[config["gate_voltage_unit"]]
        id_scale = CURRENT_UNITS_TO_AMP[config["drain_current_unit"]]

        frame = self.dataframe.copy()
        mask = np.ones(len(frame), dtype=bool)
        for column, value in zip(config["group_columns"], result.device_key):
            mask &= frame[column].astype(str).eq(str(value)).to_numpy()
        device_frame = frame.loc[mask].copy()

        high_curve, low_curve = self._extract_idvg_curves(device_frame, config, vg_scale, id_scale)
        if high_curve.empty and low_curve.empty:
            ax = self.canvas.figure.add_subplot(111)
            ax.set_title("No IdVg data available for selected device")
            self._apply_plot_defaults()
            self.canvas.draw()
            return

        polarity = self.device_polarity_by_key.get(result.device_key, "NMOS")
        y_unit = config["drain_current_unit"]
        y_scale = 1.0 / CURRENT_UNITS_TO_AMP[y_unit]
        use_abs_log = self.plot_abs_log_checkbox.isChecked()
        show_vtgm_overlay = self.vtgm_viewer_button.isChecked()

        ax = self.canvas.figure.add_subplot(111)
        gm_ax = ax.twinx() if show_vtgm_overlay else None
        gm_plotted = False

        def plot_curve(ax, curve: pd.DataFrame, label: str, vtgm_details: dict[str, Any] | None = None) -> bool:
            nonlocal gm_plotted
            if curve.empty:
                return False
            curve = curve.sort_values("_vg_v")
            y_values = curve["_id_a"] * y_scale
            if use_abs_log:
                y_values = y_values.abs()
                valid_mask = y_values > 0
                if not valid_mask.any():
                    return False
                x_values = curve.loc[valid_mask, "_vg_v"]
                y_values = y_values.loc[valid_mask]
            else:
                x_values = curve["_vg_v"]
            ax.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=1.2,
                label=label,
            )
            line_color = ax.lines[-1].get_color()
            if (
                show_vtgm_overlay
                and vtgm_details
                and vtgm_details.get("vg_v") is not None
                and vtgm_details.get("original_id_a") is not None
            ):
                highlight_y = float(vtgm_details["original_id_a"]) * y_scale
                if use_abs_log:
                    highlight_y = abs(highlight_y)
                    if highlight_y > 0:
                        ax.plot(
                            [float(vtgm_details["vg_v"])],
                            [highlight_y],
                            marker="o",
                            markersize=5,
                            color="red",
                            linestyle="None",
                            zorder=6,
                        )
                else:
                    ax.plot(
                        [float(vtgm_details["vg_v"])],
                        [highlight_y],
                        marker="o",
                        markersize=5,
                        color="red",
                        linestyle="None",
                        zorder=6,
                    )
                slope_display = float(vtgm_details.get("gm_a_per_v", 0.0)) * y_scale
                if not math.isclose(slope_display, 0.0, abs_tol=1e-30):
                    point_x = float(vtgm_details["vg_v"])
                    intercept_x = float(vtgm_details.get("x_intercept_v", point_x))
                    tangent_x = np.linspace(point_x, intercept_x, 100)
                    tangent_y = highlight_y + slope_display * (tangent_x - point_x)
                    if use_abs_log:
                        tangent_y = np.abs(tangent_y)
                        valid_tangent = tangent_y > 0
                    else:
                        valid_tangent = np.isfinite(tangent_y)
                    if np.any(valid_tangent):
                        ax.plot(
                            tangent_x[valid_tangent],
                            tangent_y[valid_tangent],
                            color="red",
                            linewidth=0.9,
                            linestyle="-.",
                            alpha=0.9,
                            label="_nolegend_",
                        )

            if show_vtgm_overlay and gm_ax is not None:
                derivative_x, derivative_y, smooth_gm_x, smooth_gm_y = self._curve_derivative_series(curve, y_scale)
                if len(derivative_x) > 0:
                    gm_ax.plot(
                        derivative_x,
                        derivative_y,
                        linestyle="--",
                        linewidth=1.2,
                        color=line_color,
                        alpha=0.9,
                        label=f"{label} dId/dVg",
                    )
                    gm_plotted = True
                if len(smooth_gm_x) > 0:
                    gm_ax.plot(
                        smooth_gm_x,
                        smooth_gm_y,
                        linestyle="-",
                        linewidth=0.9,
                        color="#000000",
                        alpha=0.95,
                        label="_nolegend_",
                    )
                    gm_plotted = True
            return True

        high_label = f"Sat Vd: {config['high_bias_value']}" if config["bias_mode"] == "by_column" else "Sat Vd"
        low_label = f"Lin Vd: {config['low_bias_value']}" if config["bias_mode"] == "by_column" else "Lin Vd"

        plotted = False
        if self.plot_vd_both_radio.isChecked():
            plotted |= plot_curve(ax, high_curve, high_label, result.details.get("vtgm_sat"))
            plotted |= plot_curve(ax, low_curve, low_label, result.details.get("vtgm_lin"))
            mode_title = "Sat + Lin Vd"
        elif self.plot_vd_high_radio.isChecked():
            plotted |= plot_curve(ax, high_curve, high_label, result.details.get("vtgm_sat"))
            mode_title = "Sat Vd"
        elif self.plot_vd_low_radio.isChecked():
            plotted |= plot_curve(ax, low_curve, low_label, result.details.get("vtgm_lin"))
            mode_title = "Lin Vd"
        else:
            mode_title = "No Vd Selection"

        if not plotted:
            ax = self.canvas.figure.add_subplot(111)
            ax.set_title("No plottable IdVg curve for selected device")
            self._apply_plot_defaults()
            self.canvas.draw()
            return

        ax.set_xlabel("Gate Voltage (V)")
        ax.set_ylabel(f"{'|Drain Current|' if use_abs_log else 'Drain Current'} ({y_unit})")
        ax.set_title(mode_title)
        if use_abs_log:
            ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(self._axis_tick_label))
        ax.yaxis.offsetText.set_visible(False)
        if gm_plotted and gm_ax is not None:
            gm_ax.set_ylabel(f"dId/dVg ({y_unit}/V)")
            gm_ax.yaxis.set_major_formatter(FuncFormatter(self._axis_tick_label))
            gm_ax.yaxis.offsetText.set_visible(False)
            gm_ax.tick_params(colors="#000000")
            gm_ax.yaxis.label.set_color("#000000")
            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = gm_ax.get_legend_handles_labels()
            ax.legend(lines_1 + lines_2, labels_1 + labels_2)
        else:
            if gm_ax is not None:
                gm_ax.set_visible(False)
            ax.legend()
        self.canvas.figure.suptitle(f"{result.display_key} | IdVg")

        self.canvas.figure.tight_layout()
        self._apply_plot_defaults()
        self.canvas.draw()

    def _annotate_point(self, point: dict[str, float] | None, label: str) -> None:
        if not point:
            return

        x = point["vg"]
        polarity_sign = point.get("polarity_sign", 1.0)
        y = point["id"] * polarity_sign * 1e6
        self.canvas.axes.scatter([x], [y], color="crimson")
        self.canvas.axes.annotate(
            f"{label}\nVg={x:.4g} V\nI={y:.4g} uA",
            xy=(x, y),
            xytext=(10, 18),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "crimson"},
            fontsize=9,
        )

    def _annotate_threshold(self, value: float | None, details: dict[str, Any] | None, label: str, target_current_a: float) -> None:
        if value is None or not details:
            return

        left = details["left"]
        right = details["right"]
        polarity_sign = details.get("polarity_sign", 1.0)
        y = target_current_a * 1e6
        self.canvas.axes.scatter([value], [y], color="darkgreen")
        self.canvas.axes.plot(
            [left["_vg_v"], right["_vg_v"]],
            [left["_id_a"] * polarity_sign * 1e6, right["_id_a"] * polarity_sign * 1e6],
            linestyle="--",
            color="darkgreen",
        )
        self.canvas.axes.annotate(
            f"{label}\nVg={value:.4g} V\nTarget={y:.4g} uA",
            xy=(value, y),
            xytext=(12, -28),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "darkgreen"},
            fontsize=9,
        )

    def export_results(self) -> None:
        if self.results_frame.empty:
            info_message("No results to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Export Results", "mosx_results.csv", "CSV Files (*.csv)")
        if not file_path:
            return

        self.results_frame.to_csv(file_path, index=False, encoding="utf-8-sig")
        self.log(f"Exported results to: {file_path}")

    def copy_results(self) -> None:
        if self.results_frame.empty:
            info_message("No results to copy.")
            return

        QApplication.clipboard().setText(self.results_frame.to_csv(index=False, sep="\t"))
        self.log("Copied all result rows to clipboard.")

    def open_results_context_menu(self, position) -> None:
        menu = QMenu(self)
        copy_action = QAction("Copy All", self)
        copy_action.triggered.connect(self.copy_results)
        export_action = QAction("Export CSV", self)
        export_action.triggered.connect(self.export_results)
        menu.addAction(copy_action)
        menu.addAction(export_action)
        menu.exec(self.results_table.viewport().mapToGlobal(position))

    def open_plot_context_menu(self, position) -> None:
        menu = QMenu(self)
        copy_action = QAction("Copy Plot Image", self)
        copy_action.triggered.connect(self.copy_plot_image)
        menu.addAction(copy_action)
        menu.exec(self.canvas.mapToGlobal(position))

    def copy_plot_image(self) -> None:
        buffer = io.BytesIO()
        self.canvas.figure.savefig(
            buffer,
            format="png",
            dpi=200,
            facecolor=self.canvas.figure.get_facecolor(),
            bbox_inches="tight",
        )
        image = QImage.fromData(buffer.getvalue(), "PNG")
        if image.isNull():
            info_message("Failed to copy plot image.")
            return
        QApplication.clipboard().setImage(image)
        self.log("Copied plot image to clipboard.")


def run() -> None:
    configure_matplotlib_fonts()
    app = QApplication(sys.argv)
    follow_system_color_scheme()
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
