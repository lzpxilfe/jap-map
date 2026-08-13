"""Register a raw scan to a selected sheet frame without resampling it."""

from __future__ import annotations

from qgis.PyQt.QtGui import QImage
from qgis.PyQt.QtWidgets import QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QPlainTextEdit, QVBoxLayout

from histcontour_core.registration import GroundControlPoint, RegistrationError, SheetRegistration

from .core.frame import CornerRole


class RegisterMapDialog(QDialog):
    """Four image corners are registered automatically; optional GCPs are text-based in v0.2."""

    def __init__(self, parent=None, iface=None):
        super().__init__(parent)
        self.iface = iface
        self.setWindowTitle("Register Map / 지도 맞추기")
        self.setMinimumWidth(620)
        root = QVBoxLayout(self)
        root.addWidget(QLabel("Select one sheet-frame feature, then select its original scan. The scan remains unwarped; this command writes a reusable registration JSON."))
        form = QFormLayout()
        self.image_path, self.output_path = QLineEdit(), QLineEdit()
        image_button, output_button = QPushButton("Browse…"), QPushButton("Save as…")
        image_button.clicked.connect(self._choose_image)
        output_button.clicked.connect(self._choose_output)
        image_row, output_row = QHBoxLayout(), QHBoxLayout()
        image_row.addWidget(self.image_path); image_row.addWidget(image_button)
        output_row.addWidget(self.output_path); output_row.addWidget(output_button)
        form.addRow("Original scan", image_row)
        form.addRow("Registration JSON", output_row)
        root.addLayout(form)
        root.addWidget(QLabel("Optional additional GCPs, one per line: pixel_x,pixel_y,map_x,map_y,label"))
        self.extra_gcps = QPlainTextEdit()
        self.extra_gcps.setPlaceholderText("1200,850,198765.4,443210.2,church")
        root.addWidget(self.extra_gcps)
        self.error = QLabel(); self.error.setStyleSheet("color: #b42318;"); self.error.setWordWrap(True); root.addWidget(self.error)
        buttons = QHBoxLayout(); buttons.addStretch(1)
        cancel, register = QPushButton("Cancel / 취소"), QPushButton("Register Map / 지도 맞추기")
        cancel.clicked.connect(self.reject); register.clicked.connect(self._register)
        buttons.addWidget(cancel); buttons.addWidget(register); root.addLayout(buttons)

    def _choose_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Original map scan", "", "Images (*.tif *.tiff *.png *.jpg *.jpeg *.jp2);;All files (*)")
        if path: self.image_path.setText(path)

    def _choose_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Registration file", "", "JSON files (*.json)")
        if path:
            self.output_path.setText(path if path.lower().endswith(".json") else path + ".json")

    def _frame_feature(self):
        layer = self.iface.activeLayer()
        if layer is None:
            raise RegistrationError("Select a sheet-frame feature first.")
        selected = layer.selectedFeatures()
        if len(selected) != 1:
            raise RegistrationError("Select exactly one sheet-frame feature first.")
        feature = selected[0]
        if not feature.fields().indexOf("sheet_id") >= 0:
            raise RegistrationError("The selected feature is not a Historical Map Tools sheet frame.")
        return feature, layer

    def _parse_extra_gcps(self):
        values = []
        for line in self.extra_gcps.toPlainText().splitlines():
            if not line.strip(): continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) not in (4, 5):
                raise RegistrationError("Additional GCPs must have pixel_x,pixel_y,map_x,map_y[,label].")
            values.append(GroundControlPoint(float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), parts[4] if len(parts) == 5 else ""))
        return values

    def _register(self):
        try:
            feature, layer = self._frame_feature()
            image = QImage(self.image_path.text())
            if image.isNull(): raise RegistrationError("Choose a readable original scan.")
            if not self.output_path.text(): raise RegistrationError("Choose where to save the registration JSON.")
            width, height = image.width(), image.height()
            gcps = []
            for role, pixel in ((CornerRole.NW, (0, 0)), (CornerRole.NE, (width - 1, 0)), (CornerRole.SE, (width - 1, height - 1)), (CornerRole.SW, (0, height - 1))):
                gcps.append(GroundControlPoint(pixel[0], pixel[1], float(feature[f"{role.value.lower()}_x"]), float(feature[f"{role.value.lower()}_y"]), role.value))
            gcps.extend(self._parse_extra_gcps())
            registration = SheetRegistration.create(str(feature["sheet_id"]), self.image_path.text(), width, height, layer.crs().authid(), gcps)
            registration.write_json(self.output_path.text())
            rmse_index = layer.fields().indexOf("registration_rmse")
            if rmse_index >= 0:
                layer.startEditing()
                layer.changeAttributeValue(feature.id(), rmse_index, registration.rmse)
                layer.commitChanges()
            self.iface.messageBar().pushSuccess("Historical Map Tools", f"Saved projective registration (RMSE {registration.rmse:.3f}).")
        except (ValueError, RegistrationError) as error:
            self.error.setText(str(error)); return
        self.accept()
