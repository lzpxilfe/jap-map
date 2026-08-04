"""Korean-first dialog for creating a four-corner map frame."""

from __future__ import annotations

from qgis.PyQt.QtCore import QSettings, Qt
from qgis.PyQt.QtWidgets import (
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)
from qgis.core import QgsCoordinateReferenceSystem
from qgis.gui import QgsProjectionSelectionDialog, QgsProjectionSelectionWidget

from .core.coordinates import CoordinateParseError, parse_angle
from .core.frame import Corner, CornerRole, FrameValidationError, SheetFrame
from .core.layer_manager import FrameLayerManager


PRESETS = (
    ("Tokyo 1892 (EPSG:5132)", "EPSG:5132"),
    ("Tokyo / Tokyo 1918 (EPSG:4301)", "EPSG:4301"),
    ("WGS 84 (EPSG:4326)", "EPSG:4326"),
)
SETTINGS_KEY = "historical_map_tools/last_crs"


class MapFrameDialog(QDialog):
    def __init__(self, parent=None, iface=None):
        super().__init__(parent)
        self.iface = iface
        self.setWindowTitle("역사지형도 도곽 만들기")
        self.setMinimumWidth(620)
        self._fields = {}
        self._build_ui()
        self._restore_crs()

    def _build_ui(self):
        root = QVBoxLayout(self)

        intro = QLabel(
            "좌상·우상·우하·좌하 모서리의 위도와 경도를 입력하면 네 점을 그대로 연결한 임시 도곽을 만듭니다."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("도엽명"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("선택 사항 — 비우면 자동으로 이름을 붙입니다")
        name_row.addWidget(self.name_edit, 1)
        root.addLayout(name_row)

        crs_box = QGroupBox("입력 좌표의 CRS")
        crs_layout = QVBoxLayout(crs_box)
        self.crs_widget = QgsProjectionSelectionWidget()
        crs_layout.addWidget(self.crs_widget)
        preset_row = QHBoxLayout()
        for label, authid in PRESETS:
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, value=authid: self._set_crs(value))
            preset_row.addWidget(button)
        custom_button = QPushButton("기타…")
        custom_button.clicked.connect(self._choose_custom_crs)
        preset_row.addWidget(custom_button)
        crs_layout.addLayout(preset_row)
        help_label = QLabel(
            "조선 지형도는 Tokyo 1892를 먼저 시도하되, 판본에 따라 Tokyo/Tokyo 1918일 수 있습니다. "
            "두 결과를 현재 배경지도나 도엽 정보와 비교하세요. WGS 84는 이미 변환된 좌표에 사용합니다."
        )
        help_label.setWordWrap(True)
        help_label.setTextFormat(Qt.TextFormat.PlainText)
        crs_layout.addWidget(help_label)
        root.addWidget(crs_box)
        self.crs_widget.crsChanged.connect(self._on_crs_changed)

        corners_box = QGroupBox("도곽 모서리 좌표")
        corners_layout = QGridLayout(corners_box)
        roles = ((CornerRole.NW, 0, 0), (CornerRole.NE, 0, 1), (CornerRole.SW, 1, 0), (CornerRole.SE, 1, 1))
        for role, row, column in roles:
            group = QGroupBox(self._role_label(role))
            group_layout = QVBoxLayout(group)
            lat = QLineEdit()
            lat.setPlaceholderText("예: 37°30′00″N")
            lon = QLineEdit()
            lon.setPlaceholderText("예: 127°00′00″E")
            group_layout.addWidget(QLabel("위도"))
            group_layout.addWidget(lat)
            group_layout.addWidget(QLabel("경도"))
            group_layout.addWidget(lon)
            corners_layout.addWidget(group, row, column)
            self._fields[role] = (lat, lon)
        root.addWidget(corners_box)

        example = QLabel("십진도(37.5), 도분(37°30′), 도분초(37°30′00″N)와 ASCII 따옴표 표기를 모두 사용할 수 있습니다.")
        example.setWordWrap(True)
        root.addWidget(example)

        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #b42318;")
        root.addWidget(self.error_label)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("취소")
        cancel.clicked.connect(self.reject)
        self.create_button = QPushButton("도곽 만들기")
        self.create_button.setDefault(True)
        self.create_button.clicked.connect(self._create)
        buttons.addWidget(cancel)
        buttons.addWidget(self.create_button)
        root.addLayout(buttons)

    @staticmethod
    def _role_label(role):
        return {
            CornerRole.NW: "좌상 (NW)",
            CornerRole.NE: "우상 (NE)",
            CornerRole.SE: "우하 (SE)",
            CornerRole.SW: "좌하 (SW)",
        }[role]

    def _restore_crs(self):
        authid = QSettings().value(SETTINGS_KEY, "")
        if authid:
            crs = QgsCoordinateReferenceSystem(str(authid))
            if crs.isValid():
                self.crs_widget.setCrs(crs)
                return
        # Do not silently assume WGS 84 on first use: historical map sheets
        # often use a Tokyo datum, so the user must make the datum choice once.
        self.crs_widget.setCrs(QgsCoordinateReferenceSystem())

    def _set_crs(self, authid):
        self.crs_widget.setCrs(QgsCoordinateReferenceSystem(authid))

    def _choose_custom_crs(self):
        dialog = QgsProjectionSelectionDialog(self)
        current = self.crs_widget.crs()
        if current.isValid():
            dialog.setCrs(current)
        exec_method = getattr(dialog, "exec", None) or dialog.exec_
        accepted = getattr(QDialog, "Accepted", None)
        if accepted is None:
            accepted = QDialog.DialogCode.Accepted
        if exec_method() == accepted and dialog.crs().isValid():
            self.crs_widget.setCrs(dialog.crs())

    def _on_crs_changed(self, crs):
        if crs.isValid() and not crs.isGeographic():
            self.error_label.setText("위도·경도 입력에는 지리좌표계(각도 단위)만 사용할 수 있습니다.")
        elif crs.isValid():
            self.error_label.clear()

    def _create(self):
        crs = self.crs_widget.crs()
        if not crs.isValid():
            self.error_label.setText("입력 좌표의 CRS를 선택해 주세요.")
            return
        if not crs.isGeographic():
            self.error_label.setText("위도·경도 입력에는 지리좌표계(각도 단위)만 사용할 수 있습니다.")
            return

        corners = {}
        try:
            for role, (lat_edit, lon_edit) in self._fields.items():
                lat = parse_angle(lat_edit.text(), "lat")
                lon = parse_angle(lon_edit.text(), "lon")
                corners[role] = Corner(role, lat, lon)
            frame = SheetFrame.create(self.name_edit.text() or "도곽", crs.authid(), corners)
            FrameLayerManager(self.iface).add_frame(frame, crs)
        except (CoordinateParseError, FrameValidationError, RuntimeError) as error:
            self.error_label.setText(str(error))
            return

        QSettings().setValue(SETTINGS_KEY, crs.authid())
        self.accept()
