"""Interactive baseline contour extraction to a temporary review layer."""

from __future__ import annotations

import json

from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QImage
from qgis.PyQt.QtWidgets import QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout
from qgis.core import QgsFeature, QgsField, QgsGeometry, QgsPointXY, QgsProject, QgsVectorLayer

from histcontour_core.contours import extract_visible_contours, generate_link_candidates
from histcontour_core.models import MapProfile
from histcontour_core.registration import SheetRegistration


class ExtractContoursDialog(QDialog):
    def __init__(self, parent=None, iface=None):
        super().__init__(parent)
        self.iface = iface
        self.setWindowTitle("Extract Contours / 등고 추출하기")
        root = QVBoxLayout(self)
        root.addWidget(QLabel("Baseline colour extraction works on original image pixels, then transforms output lines using registration JSON."))
        form = QFormLayout(); self.image, self.registration, self.profile = QLineEdit(), QLineEdit(), QLineEdit()
        for label, edit, callback in (("Original scan", self.image, self._image), ("Registration JSON", self.registration, self._registration), ("Map profile JSON", self.profile, self._profile)):
            row = QHBoxLayout(); button = QPushButton("Browse…"); button.clicked.connect(callback); row.addWidget(edit); row.addWidget(button); form.addRow(label, row)
        root.addLayout(form)
        self.error = QLabel(); self.error.setStyleSheet("color: #b42318;"); self.error.setWordWrap(True); root.addWidget(self.error)
        buttons = QHBoxLayout(); buttons.addStretch(1); cancel, run = QPushButton("Cancel / 취소"), QPushButton("Extract visible contours / 가시 등고선 추출")
        cancel.clicked.connect(self.reject); run.clicked.connect(self._run); buttons.addWidget(cancel); buttons.addWidget(run); root.addLayout(buttons)

    def _select(self, edit, title, filter):
        path, _ = QFileDialog.getOpenFileName(self, title, "", filter)
        if path: edit.setText(path)
    def _image(self): self._select(self.image, "Original map scan", "Images (*.tif *.tiff *.png *.jpg *.jpeg)")
    def _registration(self): self._select(self.registration, "Registration JSON", "JSON files (*.json)")
    def _profile(self): self._select(self.profile, "Map profile JSON", "JSON files (*.json)")

    @staticmethod
    def _pixels(image):
        return [[(image.pixelColor(x, y).red(), image.pixelColor(x, y).green(), image.pixelColor(x, y).blue()) for x in range(image.width())] for y in range(image.height())]

    def _run(self):
        try:
            image = QImage(self.image.text())
            if image.isNull(): raise ValueError("Choose a readable original scan.")
            registration = SheetRegistration.read_json(self.registration.text())
            profile = MapProfile.from_dict(json.loads(open(self.profile.text(), encoding="utf-8").read()))
            if profile.contour_rgb is None: raise ValueError("Map profile requires contour_rgb.")
            lines = extract_visible_contours(self._pixels(image), profile.contour_rgb, profile.color_distance, profile.min_component_pixels)
            layer = QgsVectorLayer(f"LineString?crs={registration.crs_authid}", f"Visible contours — {registration.sheet_id}", "memory")
            provider = layer.dataProvider(); provider.addAttributes([QgsField("contour_id", QMetaType.Type.QString), QgsField("segment_kind", QMetaType.Type.QString), QgsField("seg_conf", QMetaType.Type.Double), QgsField("link_conf", QMetaType.Type.Double), QgsField("qc_status", QMetaType.Type.QString), QgsField("registration_rmse", QMetaType.Type.Double)]); layer.updateFields()
            features = []
            for line in lines:
                feature = QgsFeature(layer.fields()); feature.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(*registration.transform(x, y).as_tuple()) for x, y in line.points])); feature.setAttributes([line.line_id, "visible", line.confidence, None, "unreviewed", registration.rmse]); features.append(feature)
            provider.addFeatures(features); layer.updateExtents(); QgsProject.instance().addMapLayer(layer)
            candidates = generate_link_candidates(lines)
            candidate_layer = QgsVectorLayer(f"LineString?crs={registration.crs_authid}", f"Contour link proposals — {registration.sheet_id}", "memory")
            candidate_provider = candidate_layer.dataProvider(); candidate_provider.addAttributes([QgsField("candidate_id", QMetaType.Type.QString), QgsField("source_lines", QMetaType.Type.QString), QgsField("cost", QMetaType.Type.Double), QgsField("link_conf", QMetaType.Type.Double), QgsField("status", QMetaType.Type.QString)]); candidate_layer.updateFields()
            candidate_features = []
            for candidate in candidates:
                feature = QgsFeature(candidate_layer.fields()); feature.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(*registration.transform(x, y).as_tuple()) for x, y in candidate.points])); feature.setAttributes([candidate.candidate_id, ",".join(candidate.source_line_ids), candidate.cost, candidate.link_confidence, candidate.status]); candidate_features.append(feature)
            candidate_provider.addFeatures(candidate_features); candidate_layer.updateExtents(); QgsProject.instance().addMapLayer(candidate_layer)
            self.iface.messageBar().pushSuccess("Historical Map Tools", f"Created {len(lines)} visible contour lines and {len(candidates)} reconnection proposals. Review proposals in Processing output.")
        except (OSError, ValueError) as error:
            self.error.setText(str(error)); return
        self.accept()
