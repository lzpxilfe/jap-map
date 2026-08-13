"""Create sheet frames in projected X/Y or geographic DMS/DD coordinates."""

from __future__ import annotations

from qgis.PyQt.QtCore import QSettings, Qt
from qgis.PyQt.QtWidgets import QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QSpinBox, QStackedWidget, QVBoxLayout, QWidget
from qgis.core import QgsCoordinateReferenceSystem
from qgis.gui import QgsProjectionSelectionDialog, QgsProjectionSelectionWidget

from histcontour_core.models import MapSheet, MetadataError

from .core.coordinates import CoordinateParseError, dms_to_decimal, parse_angle
from .core.frame import Corner, CornerRole, FrameValidationError, SheetFrame
from .core.layer_manager import FrameLayerManager


PRESETS = (("Tokyo 1892 (EPSG:5132)", "EPSG:5132"), ("Tokyo / Tokyo 1918 (EPSG:4301)", "EPSG:4301"), ("WGS 84 (EPSG:4326)", "EPSG:4326"))
SETTINGS_KEY = "historical_map_tools/last_crs"


class _DmsWidget(QGroupBox):
    """Three-cell DMS input retaining the v0.1 map-transcription workflow."""
    def __init__(self, axis, parent=None):
        super().__init__("Longitude" if axis == "lon" else "Latitude", parent)
        self.axis = axis
        layout = QHBoxLayout(self); layout.setContentsMargins(4, 8, 4, 4)
        self.degrees = QSpinBox(); self.degrees.setRange(0, 180 if axis == "lon" else 90); self.degrees.setSuffix(" °")
        self.minutes = QSpinBox(); self.minutes.setRange(0, 59); self.minutes.setSuffix(" ′")
        self.seconds = QDoubleSpinBox(); self.seconds.setRange(0, 59.999); self.seconds.setDecimals(2); self.seconds.setSuffix(" ″")
        self.hemisphere = QComboBox(); self.hemisphere.addItems(("E", "W") if axis == "lon" else ("N", "S"))
        self.preview = QLabel(); self.preview.setMinimumWidth(94)
        for widget in (self.degrees, self.minutes, self.seconds, self.hemisphere, self.preview): layout.addWidget(widget)
        for signal in (self.degrees.valueChanged, self.minutes.valueChanged, self.seconds.valueChanged, self.hemisphere.currentIndexChanged): signal.connect(self._refresh)
        self._refresh()

    def _refresh(self):
        try:
            value = self.value()
            self.preview.setText(f"→ {value:.6f}°")
            self.preview.setStyleSheet("color: #1a7f37; font-size: 10px;")
        except CoordinateParseError as error:
            self.preview.setText(f"⚠ {error}")
            self.preview.setStyleSheet("color: #b42318; font-size: 10px;")

    def value(self):
        value = dms_to_decimal(self.degrees.value(), self.minutes.value(), self.seconds.value(), self.hemisphere.currentText())
        maximum = 180 if self.axis == "lon" else 90
        if abs(value) > maximum:
            raise CoordinateParseError("Coordinate is outside its axis range.")
        return value


class MapFrameDialog(QDialog):
    """CRS-independent frame editor; the old class name remains a public API."""
    def __init__(self, parent=None, iface=None):
        super().__init__(parent); self.iface = iface; self._metadata = {}; self._corners = {}
        self.setWindowTitle("Create Sheet Frame / 도곽 만들기"); self.setMinimumWidth(760)
        self._build_ui(); self._restore_crs()

    def _build_ui(self):
        root = QVBoxLayout(self)
        description = QLabel("Create a sheet frame in any CRS. Geographic input has DMS cells with a live decimal preview; projected input uses X/Y.")
        description.setWordWrap(True); root.addWidget(description)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); body = QWidget(); layout = QVBoxLayout(body)
        metadata_box = QGroupBox("Map sheet metadata / 지도 메타데이터"); form = QFormLayout(metadata_box)
        labels = {"sheet_id":"Sheet ID *", "source_title":"Source title *", "display_title":"Display title *", "series":"Series", "edition":"Edition", "producer":"Producer *", "survey_purpose":"Survey purpose", "survey_year":"Survey year", "publication_year":"Publication year", "scale":"Scale", "contour_interval_m":"Contour interval (m)", "source_language":"Source language", "source_script":"Source script", "vertical_datum":"Vertical datum", "scan_source":"Scan source *", "rights":"Rights *", "context_note":"Historical context note"}
        for key, label in labels.items():
            edit = QLineEdit(); edit.setObjectName(f"sheetMetadata_{key}"); form.addRow(label, edit); self._metadata[key] = edit
        layout.addWidget(metadata_box)
        crs_box = QGroupBox("Coordinate reference system / 좌표계"); crs_layout = QVBoxLayout(crs_box); self.crs_widget = QgsProjectionSelectionWidget(); crs_layout.addWidget(self.crs_widget)
        choices = QHBoxLayout()
        for label, authid in PRESETS:
            button = QPushButton(label); button.clicked.connect(lambda checked=False, value=authid: self._set_crs(value)); choices.addWidget(button)
        custom = QPushButton("Choose CRS… / 기타…"); custom.clicked.connect(self._choose_custom_crs); choices.addWidget(custom); crs_layout.addLayout(choices); layout.addWidget(crs_box)
        input_row = QHBoxLayout(); input_row.addWidget(QLabel("Input mode / 입력 방식")); self.mode = QComboBox(); self.mode.addItem("Geographic DMS / DD", "geographic"); self.mode.addItem("Projected X / Y", "projected"); self.mode.currentIndexChanged.connect(self._set_mode); input_row.addWidget(self.mode, 1); layout.addLayout(input_row)
        corner_box = QGroupBox("Frame corners / 도곽 모서리"); grid = QGridLayout(corner_box)
        for role, row, column in ((CornerRole.NW,0,0),(CornerRole.NE,0,1),(CornerRole.SW,1,0),(CornerRole.SE,1,1)):
            group = QGroupBox(role.value); group_layout = QVBoxLayout(group); stack = QStackedWidget()
            dms = QWidget(); dms_layout = QVBoxLayout(dms); latitude, longitude = _DmsWidget("lat"), _DmsWidget("lon"); dms_layout.addWidget(latitude); dms_layout.addWidget(longitude)
            projected = QWidget(); projected_form = QFormLayout(projected); x, y = QLineEdit(), QLineEdit(); projected_form.addRow("X", x); projected_form.addRow("Y", y)
            stack.addWidget(dms); stack.addWidget(projected); group_layout.addWidget(stack); grid.addWidget(group,row,column)
            self._corners[role] = (stack, latitude, longitude, x, y)
        layout.addWidget(corner_box); scroll.setWidget(body); root.addWidget(scroll)
        self.error = QLabel(); self.error.setStyleSheet("color: #b42318;"); self.error.setWordWrap(True); root.addWidget(self.error)
        buttons = QHBoxLayout(); buttons.addStretch(1); cancel, create = QPushButton("Cancel / 취소"), QPushButton("Create Sheet Frame / 도곽 만들기"); cancel.clicked.connect(self.reject); create.clicked.connect(self._create); buttons.addWidget(cancel); buttons.addWidget(create); root.addLayout(buttons)

    def _set_mode(self):
        for stack, *_rest in self._corners.values(): stack.setCurrentIndex(self.mode.currentIndex())

    def _restore_crs(self):
        authid = QSettings().value(SETTINGS_KEY, "")
        if authid:
            crs = QgsCoordinateReferenceSystem(str(authid))
            if crs.isValid(): self.crs_widget.setCrs(crs)

    def _set_crs(self, authid): self.crs_widget.setCrs(QgsCoordinateReferenceSystem(authid))
    def _choose_custom_crs(self):
        picker = QgsProjectionSelectionDialog(self); picker.setCrs(self.crs_widget.crs())
        if picker.exec() and picker.crs().isValid(): self.crs_widget.setCrs(picker.crs())

    def _sheet(self, crs_authid):
        values = {key: edit.text().strip() for key, edit in self._metadata.items()}; interval = values.pop("contour_interval_m")
        return MapSheet(**values, contour_interval_m=float(interval) if interval else None, horizontal_crs=crs_authid)

    def _create(self):
        crs = self.crs_widget.crs()
        if not crs.isValid(): self.error.setText("Choose the coordinates' CRS / 좌표계를 선택하세요."); return
        if self.mode.currentData() == "geographic" and not crs.isGeographic(): self.error.setText("Geographic input requires a geographic CRS / 경위도 입력에는 지리좌표계가 필요합니다."); return
        try:
            sheet, corners = self._sheet(crs.authid()), {}
            for role, (_stack, latitude, longitude, x_edit, y_edit) in self._corners.items():
                x, y = (longitude.value(), latitude.value()) if self.mode.currentData() == "geographic" else (float(x_edit.text()), float(y_edit.text()))
                corners[role] = Corner(role, x, y)
            FrameLayerManager(self.iface).add_frame(SheetFrame.create(sheet.display_title, crs.authid(), corners), crs, sheet)
        except (ValueError, MetadataError, CoordinateParseError, FrameValidationError, RuntimeError) as error:
            self.error.setText(str(error)); return
        QSettings().setValue(SETTINGS_KEY, crs.authid()); self.accept()
