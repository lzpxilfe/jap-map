"""Korean-first dialog for creating a four-corner map frame.

좌표 입력 방식:
  - 위도/경도 각각 도(°)·분(′)·초(″) 세 칸과 방위(N/S, E/W) 드롭다운으로 입력합니다.
  - 입력 즉시 십진도로 변환하여 칸 아래에 표시하므로 사용자가 해석 결과를 확인할 수 있습니다.
  - 분·초 칸은 0–59 범위를 벗어나면 즉시 오류 색상으로 알립니다.
"""

from __future__ import annotations

from qgis.PyQt.QtCore import QSettings, Qt
from qgis.PyQt.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
)
from qgis.core import QgsCoordinateReferenceSystem
from qgis.gui import QgsProjectionSelectionDialog, QgsProjectionSelectionWidget

from .core.coordinates import CoordinateParseError, dms_to_decimal
from .core.frame import Corner, CornerRole, FrameValidationError, SheetFrame
from .core.layer_manager import FrameLayerManager


PRESETS = (
    ("Tokyo 1892 (EPSG:5132)", "EPSG:5132"),
    ("Tokyo / Tokyo 1918 (EPSG:4301)", "EPSG:4301"),
    ("WGS 84 (EPSG:4326)", "EPSG:4326"),
)
SETTINGS_KEY = "historical_map_tools/last_crs"

# 오류 색상 (빨강), 정상 색상 (기본 테두리)
_STYLE_ERROR = "border: 1.5px solid #b42318;"
_STYLE_OK = ""


class _DmsWidget(QGroupBox):
    """도·분·초 + 방위 입력 위젯 (위도 또는 경도 1개).

    사용자가 지도 귀퉁이에 인쇄된 값을 칸별로 그대로 옮겨 입력합니다.
    입력 즉시 십진도로 변환하여 아래에 미리보기로 표시합니다.
    """

    def __init__(self, label: str, axis: str, parent=None):
        """
        Parameters
        ----------
        label : str
            그룹 박스 제목 (예: "위도", "경도")
        axis : str
            ``"lat"`` 또는 ``"lon"``
        """
        super().__init__(label, parent)
        self._axis = axis
        self._build()

    def _build(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 8, 4, 4)
        layout.setSpacing(2)

        # ── 도 ─────────────────────────────────
        self._deg = QSpinBox()
        self._deg.setRange(0, 180 if self._axis == "lon" else 90)
        self._deg.setSuffix(" °")
        self._deg.setMinimumWidth(70)
        layout.addWidget(self._deg)

        # ── 분 ─────────────────────────────────
        self._min = QSpinBox()
        self._min.setRange(0, 59)
        self._min.setSuffix(" ′")
        self._min.setMinimumWidth(60)
        layout.addWidget(self._min)

        # ── 초 ─────────────────────────────────
        self._sec = QDoubleSpinBox()
        self._sec.setRange(0.0, 59.999)
        self._sec.setDecimals(1)
        self._sec.setSuffix(" ″")
        self._sec.setMinimumWidth(72)
        layout.addWidget(self._sec)

        # ── 방위 드롭다운 ─────────────────────
        self._hemi = QComboBox()
        if self._axis == "lat":
            self._hemi.addItems(["N", "S"])
        else:
            self._hemi.addItems(["E", "W"])
        self._hemi.setMinimumWidth(42)
        layout.addWidget(self._hemi)

        layout.addStretch(1)

        # ── 십진도 미리보기 ───────────────────
        self._preview = QLabel("—")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._preview.setStyleSheet("color: #555; font-size: 10px; padding-left: 4px;")
        layout.addWidget(self._preview)

        # 변경 시 미리보기 갱신
        self._deg.valueChanged.connect(self._update_preview)
        self._min.valueChanged.connect(self._update_preview)
        self._sec.valueChanged.connect(self._update_preview)
        self._hemi.currentIndexChanged.connect(self._update_preview)

        self._update_preview()

    def _update_preview(self):
        try:
            val = dms_to_decimal(
                self._deg.value(),
                self._min.value(),
                self._sec.value(),
                self._hemi.currentText(),
            )
            axis_max = 90.0 if self._axis == "lat" else 180.0
            if abs(val) > axis_max:
                raise CoordinateParseError(
                    f"{'위도' if self._axis == 'lat' else '경도'} 범위를 벗어났습니다."
                )
            label = "위도" if self._axis == "lat" else "경도"
            self._preview.setText(f"→ {val:.6f}°")
            self._preview.setStyleSheet("color: #1a7f37; font-size: 10px; padding-left: 4px;")
        except CoordinateParseError as e:
            self._preview.setText(f"⚠ {e}")
            self._preview.setStyleSheet("color: #b42318; font-size: 10px; padding-left: 4px;")

    def decimal_value(self) -> float:
        """현재 입력값을 십진도 float으로 반환합니다. 범위 오류 시 CoordinateParseError."""
        val = dms_to_decimal(
            self._deg.value(),
            self._min.value(),
            self._sec.value(),
            self._hemi.currentText(),
        )
        axis_max = 90.0 if self._axis == "lat" else 180.0
        if abs(val) > axis_max:
            label = "위도" if self._axis == "lat" else "경도"
            raise CoordinateParseError(f"{label} 범위를 벗어났습니다.")
        return val


class MapFrameDialog(QDialog):
    def __init__(self, parent=None, iface=None):
        super().__init__(parent)
        self.iface = iface
        self.setWindowTitle("역사지형도 도곽 만들기")
        self.setMinimumWidth(680)
        self._dms_widgets: dict[CornerRole, tuple[_DmsWidget, _DmsWidget]] = {}
        self._build_ui()
        self._restore_crs()

    def _build_ui(self):
        root = QVBoxLayout(self)

        intro = QLabel(
            "지도 귀퉁이에 인쇄된 위도·경도를 도(°)·분(′)·초(″) 칸에 그대로 옮겨 입력하세요. "
            "입력값이 십진도로 어떻게 해석되는지 즉시 확인할 수 있습니다."
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

        corners_box = QGroupBox("도곽 모서리 좌표  (도° 분′ 초″ 방위)")
        corners_layout = QGridLayout(corners_box)
        corners_layout.setSpacing(8)

        roles = (
            (CornerRole.NW, 0, 0),
            (CornerRole.NE, 0, 1),
            (CornerRole.SW, 1, 0),
            (CornerRole.SE, 1, 1),
        )
        for role, row, column in roles:
            group = QGroupBox(self._role_label(role))
            group_layout = QVBoxLayout(group)
            group_layout.setSpacing(4)

            lat_widget = _DmsWidget("위도", "lat")
            lon_widget = _DmsWidget("경도", "lon")
            group_layout.addWidget(lat_widget)
            group_layout.addWidget(lon_widget)

            corners_layout.addWidget(group, row, column)
            self._dms_widgets[role] = (lat_widget, lon_widget)

        root.addWidget(corners_box)

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
            for role, (lat_widget, lon_widget) in self._dms_widgets.items():
                lat = lat_widget.decimal_value()
                lon = lon_widget.decimal_value()
                corners[role] = Corner(role, lat, lon)
            frame = SheetFrame.create(self.name_edit.text() or "도곽", crs.authid(), corners)
            FrameLayerManager(self.iface).add_frame(frame, crs)
        except (CoordinateParseError, FrameValidationError, RuntimeError) as error:
            self.error_label.setText(str(error))
            return

        QSettings().setValue(SETTINGS_KEY, crs.authid())
        self.accept()
