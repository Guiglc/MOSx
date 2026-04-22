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
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QAction, QGuiApplication, QImage, QPainter, QPalette, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
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
    for widget in widgets:
        layout.addWidget(widget)
    layout.addStretch(1)
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
    return text or None


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
    vtgm_v: float | None
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
        polarity_by_device: dict[tuple[Any, ...], str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Device Setting")
        self.resize(860, 520)
        self._device_keys = device_keys

        self._table = QTableWidget(len(device_keys), 4)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.ExtendedSelection)
        self._table.setHorizontalHeaderLabels(["Device", "Width (nm)", "Length (nm)", "Polarity"])
        for row, key in enumerate(device_keys):
            label = " | ".join(str(part) for part in key)
            self._table.setItem(row, 0, QTableWidgetItem(label))
            if key in existing:
                width, length = existing[key]
                width_text = str(width)
                length_text = str(length)
            else:
                width_text = ""
                length_text = ""
            self._table.setItem(row, 1, QTableWidgetItem(width_text))
            self._table.setItem(row, 2, QTableWidgetItem(length_text))
            self._table.setItem(row, 3, QTableWidgetItem(polarity_by_device.get(key, "NMOS")))

        apply_button = QPushButton("Apply First W/L To All")
        apply_button.clicked.connect(self.apply_first_row_to_all)
        clear_button = QPushButton("Clear All W/L")
        clear_button.clicked.connect(self.clear_all_dimensions)
        set_nmos_button = QPushButton("Set Selected NMOS")
        set_nmos_button.clicked.connect(lambda: self.set_selected_polarity("NMOS"))
        set_pmos_button = QPushButton("Set Selected PMOS")
        set_pmos_button.clicked.connect(lambda: self.set_selected_polarity("PMOS"))
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        button_row = QHBoxLayout()
        button_row.addWidget(apply_button)
        button_row.addWidget(clear_button)
        button_row.addWidget(set_nmos_button)
        button_row.addWidget(set_pmos_button)
        button_row.addStretch(1)
        button_row.addWidget(save_button)
        button_row.addWidget(cancel_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self._table)
        layout.addLayout(button_row)

    def apply_first_row_to_all(self) -> None:
        if self._table.rowCount() == 0:
            return
        width = self._table.item(0, 1).text() if self._table.item(0, 1) else ""
        length = self._table.item(0, 2).text() if self._table.item(0, 2) else ""
        for row in range(1, self._table.rowCount()):
            self._table.setItem(row, 1, QTableWidgetItem(width))
            self._table.setItem(row, 2, QTableWidgetItem(length))

    def clear_all_dimensions(self) -> None:
        for row in range(self._table.rowCount()):
            self._table.setItem(row, 1, QTableWidgetItem(""))
            self._table.setItem(row, 2, QTableWidgetItem(""))

    def set_selected_polarity(self, polarity: str) -> None:
        selected_rows = sorted({index.row() for index in self._table.selectionModel().selectedRows()})
        if not selected_rows:
            selected_rows = list(range(self._table.rowCount()))
        for row in selected_rows:
            self._table.setItem(row, 3, QTableWidgetItem(polarity))

    def values(self) -> dict[tuple[Any, ...], tuple[float, float]]:
        result: dict[tuple[Any, ...], tuple[float, float]] = {}
        for row, key in enumerate(self._device_keys):
            width_text = self._table.item(row, 1).text().strip()
            length_text = self._table.item(row, 2).text().strip()
            if not width_text and not length_text:
                continue
            if not width_text or not length_text:
                raise ValueError("Width and Length must both be filled or both left empty.")
            result[key] = (float(width_text), float(length_text))
        return result

    def polarity_by_device(self) -> dict[tuple[Any, ...], str]:
        result: dict[tuple[Any, ...], str] = {}
        for row, key in enumerate(self._device_keys):
            polarity_item = self._table.item(row, 3)
            polarity = polarity_item.text().strip().upper() if polarity_item else "NMOS"
            result[key] = "PMOS" if polarity == "PMOS" else "NMOS"
        return result


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(6, 4))
        super().__init__(self.figure)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MOSx Parameter Calculator")
        self.resize(1600, 950)

        self.dataframe = pd.DataFrame()
        self.preview_model = DataFrameModel()
        self.results_model = DataFrameModel()
        self.device_dimensions: dict[tuple[Any, ...], tuple[float, float]] = {}
        self.device_polarity_by_key: dict[tuple[Any, ...], str] = {}
        self.results_by_key: dict[tuple[Any, ...], DeviceResult] = {}
        self.results_frame = pd.DataFrame()

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
        config_layout = QVBoxLayout(config_panel)

        mapping_group = QGroupBox("Column Mapping")
        mapping_layout = QFormLayout(mapping_group)

        self.group_columns_list = QListWidget()
        self.group_columns_list.setSelectionMode(QListWidget.MultiSelection)
        mapping_layout.addRow("Device key columns", self.group_columns_list)
        self.dimensions_button = QPushButton("Device Setting")
        self.dimensions_button.clicked.connect(self.open_dimensions_dialog)
        mapping_layout.addRow("", self.dimensions_button)

        mapping_layout.addRow(GroupBoxSeparator(mapping_group))
        self.curve_mode_with_type_radio = QRadioButton("Need curve type column + IdVg selection")
        self.curve_mode_idvg_only_radio = QRadioButton("Table already contains IdVg only")
        self.curve_mode_with_type_radio.setChecked(True)
        mapping_layout.addRow("Curve mode", _row_widget(self.curve_mode_with_type_radio, self.curve_mode_idvg_only_radio))

        self.curve_mode_stack = QStackedWidget()
        self.curve_mode_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        curve_page_with_type = QWidget()
        curve_page_with_type_layout = QFormLayout(curve_page_with_type)
        curve_page_with_type_layout.setContentsMargins(0, 0, 0, 0)
        self.curve_type_column_combo = QComboBox()
        self.idvg_value_combo = QComboBox()
        curve_page_with_type_layout.addRow("Curve type column", self.curve_type_column_combo)
        curve_page_with_type_layout.addRow("IdVg curve", self.idvg_value_combo)
        self.curve_mode_stack.addWidget(curve_page_with_type)

        curve_page_idvg_only = QWidget()
        curve_page_idvg_only_layout = QVBoxLayout(curve_page_idvg_only)
        curve_page_idvg_only_layout.setContentsMargins(0, 0, 0, 0)
        curve_page_idvg_only_layout.addWidget(QLabel("No curve selection needed (IdVg only)."))
        self.curve_mode_stack.addWidget(curve_page_idvg_only)

        mapping_layout.addRow(self.curve_mode_stack)

        mapping_layout.addRow(GroupBoxSeparator(mapping_group))
        self.bias_mode_column_radio = QRadioButton("Drain bias identified by a column + values")
        self.bias_mode_columns_radio = QRadioButton("Drain bias uses separate Vg/Id columns")
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
        bias_page_by_column_layout.addRow("Value for VDD / high Vd", self.high_bias_value_combo)
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
        self.gate_voltage_unit_combo.addItems(["V", "mV"])
        self.drain_current_unit_combo = QComboBox()
        self.drain_current_unit_combo.addItems(["uA", "nA", "pA", "mA", "A"])
        self.drain_current_unit_combo.setCurrentText("A")
        self.threshold_current_positive_radio = QRadioButton("Positive")
        self.threshold_current_negative_radio = QRadioButton("Negative")
        self.threshold_current_positive_radio.setChecked(True)
        mapping_layout.addRow("Gate voltage unit", self.gate_voltage_unit_combo)
        mapping_layout.addRow("Drain current unit", self.drain_current_unit_combo)
        mapping_layout.addRow(
            "Drain current direction",
            _row_widget(self.threshold_current_positive_radio, self.threshold_current_negative_radio),
        )
        config_layout.addWidget(mapping_group)

        calc_group = QGroupBox("Calculation Settings")
        calc_layout = QFormLayout(calc_group)

        self.vdd_spin = QDoubleSpinBox()
        self.vdd_spin.setDecimals(2)
        self.vdd_spin.setRange(-1_000_000, 1_000_000)
        self.vdd_spin.setValue(1.2)
        self.vdd_spin.setSingleStep(0.01)

        self.threshold_constant_spin = QDoubleSpinBox()
        self.threshold_constant_spin.setDecimals(2)
        self.threshold_constant_spin.setRange(0.0, 1_000_000)
        self.threshold_constant_spin.setValue(40.0)
        self.threshold_constant_spin.setSuffix(" nA")
        self.threshold_constant_spin.setSingleStep(1.0)

        calc_layout.addRow("Vdd target (V)", self.vdd_spin)
        calc_layout.addRow("Threshold constant", self.threshold_constant_spin)
        config_layout.addWidget(calc_group)

        splitter.addWidget(config_panel)

        right_splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(right_splitter)
        splitter.setSizes([720, 880])

        right_panel = QTabWidget()
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
        right_panel.addTab(results_tab, "Results")

        plot_tab = QWidget()
        plot_tab_layout = QVBoxLayout(plot_tab)

        plot_controls = QGroupBox("Plot Settings")
        plot_controls_layout = QFormLayout(plot_controls)
        self.plot_device_combo = QComboBox()
        self.plot_vd_both_radio = QRadioButton("Both Vd")
        self.plot_vd_high_radio = QRadioButton("High Vd")
        self.plot_vd_low_radio = QRadioButton("Low Vd")
        self.plot_vd_both_radio.setChecked(True)
        plot_controls_layout.addRow("Device", self.plot_device_combo)
        plot_controls_layout.addRow(
            "Vd selection",
            _row_widget(self.plot_vd_both_radio, self.plot_vd_high_radio, self.plot_vd_low_radio),
        )
        self.plot_button = QPushButton("Plot Selected Device")
        self.plot_button.clicked.connect(self.plot_selected_device)
        plot_controls_layout.addRow("", self.plot_button)

        plot_controls.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        plot_controls.setMaximumHeight(plot_controls.sizeHint().height())
        plot_tab_layout.addWidget(plot_controls, 0)
        self.canvas = MplCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self.open_plot_context_menu)
        plot_tab_layout.addWidget(self.canvas, 1)
        right_panel.addTab(plot_tab, "Plot")
        right_splitter.addWidget(right_panel)

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setPlaceholderText("Imported file info, validation messages, and calculation notes will appear here.")
        right_splitter.addWidget(self.status_text)
        right_splitter.setSizes([760, 210])

        self.curve_type_column_combo.currentTextChanged.connect(self.refresh_curve_value_choices)
        self.drain_bias_column_combo.currentTextChanged.connect(self.refresh_bias_value_choices)
        self.vdd_spin.valueChanged.connect(self.auto_select_bias_values)
        self.curve_mode_with_type_radio.toggled.connect(self._update_curve_mode_ui)
        self.bias_mode_column_radio.toggled.connect(self._update_bias_mode_ui)
        self.plot_vd_both_radio.toggled.connect(self.plot_selected_device)
        self.plot_vd_high_radio.toggled.connect(self.plot_selected_device)
        self.plot_vd_low_radio.toggled.connect(self.plot_selected_device)
        self._update_curve_mode_ui()
        self._update_bias_mode_ui()

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

    def _update_curve_mode_ui(self) -> None:
        # 0 = needs curve type column, 1 = IdVg-only
        self.curve_mode_stack.setCurrentIndex(0 if self.curve_mode_with_type_radio.isChecked() else 1)

    def _update_bias_mode_ui(self) -> None:
        # 0 = bias by column, 1 = separate columns
        self.bias_mode_stack.setCurrentIndex(0 if self.bias_mode_column_radio.isChecked() else 1)

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
        self.status_text.append(text)

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
        for index in range(min(3, self.group_columns_list.count())):
            self.group_columns_list.item(index).setSelected(True)

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
            combo.addItems(columns)

        curve_guess = guess_column(columns, ["curve", "type", "mode", "source_name", "test", "name"])
        bias_guess = guess_column(columns, ["bias", "drain", "vd", "condition"])
        vg_guess = guess_column(columns, ["vg", "gate"])
        id_guess = guess_column(columns, ["id", "ids", "drain current", "current"])

        if curve_guess:
            self.curve_type_column_combo.setCurrentText(curve_guess)
        if bias_guess:
            self.drain_bias_column_combo.setCurrentText(bias_guess)
        if vg_guess:
            self.gate_voltage_column_combo.setCurrentText(vg_guess)
            self.high_gate_voltage_column_combo.setCurrentText(vg_guess)
            self.low_gate_voltage_column_combo.setCurrentText(vg_guess)
        if id_guess:
            self.drain_current_column_combo.setCurrentText(id_guess)
            self.high_drain_current_column_combo.setCurrentText(id_guess)
            self.low_drain_current_column_combo.setCurrentText(id_guess)

        self.refresh_curve_value_choices()
        self.refresh_bias_value_choices()

    def selected_group_columns(self) -> list[str]:
        return [item.text() for item in self.group_columns_list.selectedItems()]

    def refresh_curve_value_choices(self) -> None:
        self.idvg_value_combo.clear()

        column = combo_text(self.curve_type_column_combo)
        if self.dataframe.empty or not column or column not in self.dataframe.columns:
            return

        values = sorted(self.dataframe[column].dropna().astype(str).unique().tolist())
        self.idvg_value_combo.addItems(values)
        if "IdVg" in values:
            self.idvg_value_combo.setCurrentText("IdVg")

    def refresh_bias_value_choices(self) -> None:
        self.high_bias_value_combo.clear()
        self.low_bias_value_combo.clear()

        column = combo_text(self.drain_bias_column_combo)
        if self.dataframe.empty or not column or column not in self.dataframe.columns:
            return

        values = sorted(self.dataframe[column].dropna().astype(str).unique().tolist())
        self.high_bias_value_combo.addItems(values)
        self.low_bias_value_combo.addItems(values)
        self.auto_select_bias_values()

    def auto_select_bias_values(self) -> None:
        values = [self.high_bias_value_combo.itemText(i) for i in range(self.high_bias_value_combo.count())]
        if not values:
            return

        parsed_values: list[tuple[float, str]] = []
        for value in values:
            try:
                parsed_values.append((float(value), value))
            except ValueError:
                continue

        if parsed_values:
            # Bias columns are often signed (e.g. PMOS uses negative Vd). Match by magnitude.
            high_target = abs(self.vdd_spin.value())
            high_value = min(parsed_values, key=lambda item: abs(abs(item[0]) - high_target))[1]
            low_value = min(parsed_values, key=lambda item: abs(item[0]))[1]
            self.high_bias_value_combo.setCurrentText(high_value)
            self.low_bias_value_combo.setCurrentText(low_value)
            return

        self.high_bias_value_combo.setCurrentIndex(len(values) - 1)
        self.low_bias_value_combo.setCurrentIndex(0)

    def _device_keys(self) -> list[tuple[Any, ...]]:
        group_columns = self.selected_group_columns()
        if not group_columns:
            return []
        grouped = self.dataframe[group_columns].drop_duplicates().fillna("")
        return [tuple(row[column] for column in group_columns) for _, row in grouped.iterrows()]

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
        dialog = WlDialog(device_keys, self.device_dimensions, self.device_polarity_by_key, self)
        if dialog.exec() == QDialog.Accepted:
            try:
                self.device_dimensions = dialog.values()
            except ValueError:
                info_message("Width / Length must be valid numeric values.")
                return
            self.device_polarity_by_key = dialog.polarity_by_device()
            self.log(f"Saved Width / Length for {len(self.device_dimensions)} devices.")
            self.log(f"Saved polarity for {len(self.device_polarity_by_key)} devices.")
            self.refresh_plot_devices()

    def _config(self) -> dict[str, Any]:
        curve_mode = "with_type" if self.curve_mode_with_type_radio.isChecked() else "idvg_only"
        bias_mode = "by_column" if self.bias_mode_column_radio.isChecked() else "by_columns"
        return {
            "group_columns": self.selected_group_columns(),
            "curve_mode": curve_mode,
            "bias_mode": bias_mode,
            "curve_type_column": combo_text(self.curve_type_column_combo),
            "idvg_value": combo_text(self.idvg_value_combo),
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
            "vdd_target_v": self.vdd_spin.value(),
            "threshold_constant_na": self.threshold_constant_spin.value(),
            "threshold_current_sign": 1.0 if self.threshold_current_positive_radio.isChecked() else -1.0,
        }

    def validate_config(self) -> str | None:
        config = self._config()
        if not config["group_columns"]:
            return "Missing required mapping: group_columns"
        if not config["gate_voltage_unit"]:
            return "Missing required mapping: gate_voltage_unit"
        if not config["drain_current_unit"]:
            return "Missing required mapping: drain_current_unit"

        if config["curve_mode"] == "with_type":
            if not config["curve_type_column"]:
                return "Missing required mapping: curve_type_column"
            if not config["idvg_value"]:
                return "Missing required mapping: idvg_value"

        if config["bias_mode"] == "by_column":
            required = [
                "drain_bias_column",
                "high_bias_value",
                "low_bias_value",
                "gate_voltage_column",
                "drain_current_column",
            ]
        else:
            required = [
                "high_gate_voltage_column",
                "high_drain_current_column",
                "low_gate_voltage_column",
                "low_drain_current_column",
            ]
        for key in required:
            if not config.get(key):
                return f"Missing required mapping: {key}"
        if config["threshold_constant_na"] <= 0:
            return "Threshold constant must be greater than 0."
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
            base = base[base[curve_col] == str(config["idvg_value"])].copy()

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

            high_frame = base[base[bias_col] == str(config["high_bias_value"])].copy().sort_values("_vg_v")
            low_frame = base[base[bias_col] == str(config["low_bias_value"])].copy().sort_values("_vg_v")
            return high_frame, low_frame

        high = pd.DataFrame()
        low = pd.DataFrame()
        if config["high_gate_voltage_column"] in base.columns and config["high_drain_current_column"] in base.columns:
            high = base[[config["high_gate_voltage_column"], config["high_drain_current_column"]]].copy()
            high["_vg_v"] = pd.to_numeric(high[config["high_gate_voltage_column"]], errors="coerce") * vg_scale
            high["_id_a"] = pd.to_numeric(high[config["high_drain_current_column"]], errors="coerce") * id_scale
            high = high.dropna(subset=["_vg_v", "_id_a"]).sort_values("_vg_v")
        if config["low_gate_voltage_column"] in base.columns and config["low_drain_current_column"] in base.columns:
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
            results.append(self._calculate_device(device_key, high_curve, low_curve, width_nm, length_nm, config))

        if not results:
            info_message("No valid devices were calculated. Please check your mappings and numeric columns.")
            return

        self.results_by_key = {result.device_key: result for result in results}
        self.results_frame = pd.DataFrame(
            [
                {
                    "Device": result.display_key,
                    "Width (nm)": result.width_nm,
                    "Length (nm)": result.length_nm,
                    "Idoff (pA)": result.idoff_pa,
                    "Idoff/L (pA/nm)": result.idoff_pa_per_nm,
                    "Ids (uA)": result.ids_ua,
                    "Ids/L (uA/nm)": result.ids_ua_per_nm,
                    "Idl (uA)": result.idl_ua,
                    "Idl/L (uA/nm)": result.idl_ua_per_nm,
                    "Vts (V)": result.vts_v,
                    "Vtl (V)": result.vtl_v,
                    "Vtgm (V)": result.vtgm_v,
                }
                for result in results
            ]
        )
        self.results_model.set_frame(self.results_frame)
        self.log(
            "Calculation settings: "
            f"Vdd={config['vdd_target_v']:.6g} V, "
            f"Threshold={config['threshold_constant_na']:.6g} nA * W/L "
            f"({'positive' if config['threshold_current_sign'] > 0 else 'negative'} Id)"
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
            vtgm_v=None,
            details={},
        )

    def _calculate_device(
        self,
        device_key: tuple[Any, ...],
        high_frame: pd.DataFrame,
        low_frame: pd.DataFrame,
        width_nm: float,
        length_nm: float,
        config: dict[str, Any],
    ) -> DeviceResult:
        polarity = self.device_polarity_by_key.get(device_key, "NMOS")
        polarity_sign = -1.0 if polarity == "PMOS" else 1.0
        vg_vdd_target_v = abs(config["vdd_target_v"])

        idoff_row = self._closest_row(high_frame, target=0.0, column="_vg_v")
        ids_row = self._closest_abs_row(high_frame, target=vg_vdd_target_v, column="_vg_v")
        idl_row = self._closest_abs_row(low_frame, target=vg_vdd_target_v, column="_vg_v")

        target_current_a = config["threshold_current_sign"] * config["threshold_constant_na"] * 1e-9 * (width_nm / length_nm)
        vts, vts_details = self._interpolate_vg_for_current(high_frame, target_current_a)
        vtl, vtl_details = self._interpolate_vg_for_current(low_frame, target_current_a)
        vtgm, vtgm_details = self._calculate_vtgm(high_frame)

        idoff_pa = self._current_display_value(idoff_row, polarity_sign, 1e12)
        ids_ua = self._current_display_value(ids_row, polarity_sign, 1e6)
        idl_ua = self._current_display_value(idl_row, polarity_sign, 1e6)

        idoff_pa_per_nm = idoff_pa / length_nm if idoff_pa is not None else None
        ids_ua_per_nm = ids_ua / length_nm if ids_ua is not None else None
        idl_ua_per_nm = idl_ua / length_nm if idl_ua is not None else None

        details = {
            "high_curve": high_frame,
            "low_curve": low_frame,
            "idoff_point": None if idoff_row is None else {"vg": float(idoff_row["_vg_v"]), "id": float(idoff_row["_id_a"]), "polarity_sign": polarity_sign},
            "ids_point": None if ids_row is None else {"vg": float(ids_row["_vg_v"]), "id": float(ids_row["_id_a"]), "polarity_sign": polarity_sign},
            "idl_point": None if idl_row is None else {"vg": float(idl_row["_vg_v"]), "id": float(idl_row["_id_a"]), "polarity_sign": polarity_sign},
            "vts": vts_details,
            "vtl": vtl_details,
            "vtgm": vtgm_details,
            "target_current_a": target_current_a,
            "polarity_sign": polarity_sign,
            "polarity": polarity,
        }

        return DeviceResult(
            device_key=device_key,
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
            vtgm_v=vtgm,
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

    def _current_display_value(self, row: pd.Series | None, polarity_sign: float, scale: float) -> float | None:
        if row is None:
            return None
        current_a = float(row["_id_a"]) * polarity_sign
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

    def _calculate_vtgm(self, frame: pd.DataFrame) -> tuple[float | None, dict[str, Any] | None]:
        if len(frame) < 2:
            return None, None

        work = frame[["_vg_v", "_id_a"]].dropna().sort_values("_vg_v")
        if len(work) < 2:
            return None, None

        candidates: list[dict[str, Any]] = []
        values = work.to_dict("records")
        for left, right in zip(values, values[1:]):
            delta_vg = right["_vg_v"] - left["_vg_v"]
            if math.isclose(delta_vg, 0.0):
                continue
            slope = (right["_id_a"] - left["_id_a"]) / delta_vg
            candidates.append(
                {
                    "left": left,
                    "right": right,
                    "slope_a_per_v": slope,
                    "vtgm_v": (left["_vg_v"] + right["_vg_v"]) / 2,
                }
            )

        if not candidates:
            return None, None

        best = max(candidates, key=lambda item: abs(item["slope_a_per_v"]))
        return float(best["vtgm_v"]), best

    def _fmt(self, value: float | None) -> str:
        return "" if value is None or pd.isna(value) else f"{value:.6g}"

    def refresh_plot_devices(self) -> None:
        current = self.plot_device_combo.currentText()
        self.plot_device_combo.clear()
        names = [result.display_key for result in self.results_by_key.values()]
        self.plot_device_combo.addItems(names)
        if current in names:
            self.plot_device_combo.setCurrentText(current)

    def sync_plot_device_with_selection(self, index: QModelIndex) -> None:
        if not index.isValid() or self.results_frame.empty:
            return
        device_name = str(self.results_frame.iloc[index.row()]["Device"])
        self.plot_device_combo.setCurrentText(device_name)

    def plot_selected_device(self) -> None:
        self.canvas.figure.clear()
        if not self.results_by_key:
            self._apply_plot_defaults()
            self.canvas.draw()
            return

        selected_name = self.plot_device_combo.currentText()
        result = next((item for item in self.results_by_key.values() if item.display_key == selected_name), None)
        if result is None:
            self._apply_plot_defaults()
            self.canvas.draw()
            return

        config = self._config()
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
        polarity_sign = -1.0 if polarity == "PMOS" else 1.0
        y_unit = config["drain_current_unit"]
        y_scale = 1.0 / CURRENT_UNITS_TO_AMP[y_unit]

        def plot_curve(ax, curve: pd.DataFrame, label: str) -> bool:
            if curve.empty:
                return False
            curve = curve.sort_values("_vg_v")
            ax.plot(
                curve["_vg_v"],
                curve["_id_a"] * polarity_sign * y_scale,
                marker="o",
                linewidth=1.2,
                label=label,
            )
            return True

        high_label = f"High Vd: {config['high_bias_value']}" if config["bias_mode"] == "by_column" else "High Vd"
        low_label = f"Low Vd: {config['low_bias_value']}" if config["bias_mode"] == "by_column" else "Low Vd"

        ax = self.canvas.figure.add_subplot(111)
        plotted = False
        if self.plot_vd_both_radio.isChecked():
            plotted |= plot_curve(ax, high_curve, high_label)
            plotted |= plot_curve(ax, low_curve, low_label)
            mode_title = "High + Low Vd"
        elif self.plot_vd_high_radio.isChecked():
            plotted |= plot_curve(ax, high_curve, high_label)
            mode_title = "High Vd"
        else:
            plotted |= plot_curve(ax, low_curve, low_label)
            mode_title = "Low Vd"

        if not plotted:
            ax = self.canvas.figure.add_subplot(111)
            ax.set_title("No plottable IdVg curve for selected device")
            self._apply_plot_defaults()
            self.canvas.draw()
            return

        ax.set_xlabel("Gate Voltage (V)")
        ax.set_ylabel(f"Drain Current ({y_unit})")
        ax.set_title(mode_title)
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
