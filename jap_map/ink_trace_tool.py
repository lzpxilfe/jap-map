"""Interactive, explicit Ink tracing for editable QGIS line layers."""

from __future__ import annotations

import math

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QVBoxLayout
from qgis.core import QgsFeature, QgsGeometry, QgsPointXY, QgsProject, QgsRasterLayer, QgsRectangle, QgsVectorLayer, QgsWkbTypes
from qgis.gui import QgsMapToolEmitPoint, QgsRubberBand

from histcontour_core.ink import ink_centerline_candidates
from histcontour_core.livewire import LiveWireError, trace_ink_path
from histcontour_core.manual_gap_bridge import ManualGapBridgeError, build_manual_gap_bridge, sample_evidence_tangent
from histcontour_core.trace_guidance import guidance_from_boxes


class InkTraceDialog(QDialog):
    """Choose same-CRS raster and editable line layers before tracing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ink Trace / Ink 추적·라벨 공백 연결")
        root = QVBoxLayout(self)
        root.addWidget(QLabel("두 점을 찍어 Ink 경로를 미리 보고 Enter로 저장합니다. Alt+두 번 클릭은 글자·기호 회피 영역, G는 라벨 공백 연결, Esc는 취소입니다."))
        form = QFormLayout()
        self.raster = QComboBox()
        self.target = QComboBox()
        self._rasters = [layer for layer in QgsProject.instance().mapLayers().values() if isinstance(layer, QgsRasterLayer)]
        self._targets = [layer for layer in QgsProject.instance().mapLayers().values() if isinstance(layer, QgsVectorLayer) and QgsWkbTypes.geometryType(layer.wkbType()) == QgsWkbTypes.LineGeometry]
        for layer in self._rasters:
            self.raster.addItem(layer.name(), layer.id())
        for layer in self._targets:
            self.target.addItem(layer.name(), layer.id())
        form.addRow("Source raster", self.raster)
        form.addRow("Editable line layer", self.target)
        root.addLayout(form)
        self.error = QLabel()
        self.error.setStyleSheet("color: #b42318;")
        self.error.setWordWrap(True)
        root.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _accept(self):
        raster = QgsProject.instance().mapLayer(self.raster.currentData())
        target = QgsProject.instance().mapLayer(self.target.currentData())
        if raster is None or target is None:
            self.error.setText("Select a source raster and an editable line layer.")
            return
        if raster.crs() != target.crs():
            self.error.setText("This first Ink tool requires raster and target layers to share one CRS.")
            return
        if not target.isEditable() and not target.startEditing():
            self.error.setText("Could not start editing the target line layer.")
            return
        self.accept()

    def selected_layers(self):
        return QgsProject.instance().mapLayer(self.raster.currentData()), QgsProject.instance().mapLayer(self.target.currentData())


class InkTraceMapTool(QgsMapToolEmitPoint):
    """Two explicit clicks create a preview; Enter is the only commit action."""

    MAX_CACHE_PIXELS = 1000

    def __init__(self, canvas, iface, raster: QgsRasterLayer, target: QgsVectorLayer):
        super().__init__(canvas)
        self.iface, self.raster, self.target = iface, raster, target
        self.anchor_full = None
        self.end_full = None
        self.guide_corner = None
        self.guide_boxes = []
        self._evidence = None
        self._origin = (0, 0)
        self._preview_points = None
        self._generation = 0
        self.rubber = QgsRubberBand(canvas, QgsWkbTypes.LineGeometry)
        self.rubber.setColor(QColor("#16a34a"))
        self.rubber.setWidth(2)
        self.rubber.setLineStyle(Qt.DashLine)

    def _message(self, level, text):
        method = getattr(self.iface.messageBar(), f"push{level}", None)
        if method:
            method("Historical Map Tools", text)

    def _pixel(self, map_point):
        extent = self.raster.extent()
        width, height = self.raster.width(), self.raster.height()
        x = (map_point.x() - extent.xMinimum()) * width / extent.width() - 0.5
        y = (extent.yMaximum() - map_point.y()) * height / extent.height() - 0.5
        return x, y

    def _map(self, full_xy):
        extent = self.raster.extent()
        width, height = self.raster.width(), self.raster.height()
        return QgsPointXY(extent.xMinimum() + (full_xy[0] + 0.5) * extent.width() / width, extent.yMaximum() - (full_xy[1] + 0.5) * extent.height() / height)

    def _cache_bounds(self):
        first, second = self.anchor_full, self.end_full or self.anchor_full
        min_x, max_x = sorted((first[0], second[0]))
        min_y, max_y = sorted((first[1], second[1]))
        if max(max_x - min_x, max_y - min_y) > 320:
            raise LiveWireError("Ink tracing is limited to a 320 source-pixel window")
        padding = 64
        width, height = self.raster.width(), self.raster.height()
        x0, y0 = max(0, int(math.floor(min_x - padding))), max(0, int(math.floor(min_y - padding)))
        x1, y1 = min(width, int(math.ceil(max_x + padding + 1))), min(height, int(math.ceil(max_y + padding + 1)))
        if max(x1 - x0, y1 - y0) > self.MAX_CACHE_PIXELS:
            raise LiveWireError("Ink evidence cache would exceed 1000 source pixels")
        return x0, y0, x1, y1

    def _read_evidence(self):
        try:
            import numpy as np
        except ImportError as error:
            raise LiveWireError("Ink trace needs NumPy in the QGIS Python environment") from error
        x0, y0, x1, y1 = self._cache_bounds()
        extent = self.raster.extent()
        full_width, full_height = self.raster.width(), self.raster.height()
        crop_extent = QgsRectangle(
            extent.xMinimum() + x0 * extent.width() / full_width,
            extent.yMaximum() - y1 * extent.height() / full_height,
            extent.xMinimum() + x1 * extent.width() / full_width,
            extent.yMaximum() - y0 * extent.height() / full_height,
        )
        block = self.raster.dataProvider().block(1, crop_extent, x1 - x0, y1 - y0)
        source = np.array([[block.value(column, row) for column in range(x1 - x0)] for row in range(y1 - y0)], dtype=np.float32)
        finite = np.isfinite(source)
        if not finite.any():
            raise LiveWireError("raster crop has no finite source pixels")
        low, high = float(source[finite].min()), float(source[finite].max())
        source = np.full(source.shape, 255, dtype=np.uint8) if high <= low else np.clip((source - low) / (high - low) * 255, 0, 255).astype(np.uint8)
        self._origin = x0, y0
        self._evidence = ink_centerline_candidates(source, tile_origin=self._origin)

    def _local(self, full_xy):
        return full_xy[0] - self._origin[0], full_xy[1] - self._origin[1]

    def _guidance(self):
        if not self.guide_boxes:
            return None
        local_boxes = [(x0 - self._origin[0], y0 - self._origin[1], x1 - self._origin[0], y1 - self._origin[1]) for x0, y0, x1, y1 in self.guide_boxes]
        return guidance_from_boxes(self._evidence.centerline.shape, local_boxes)

    def _set_preview(self, points):
        self._preview_points = tuple(points)
        self.rubber.reset(QgsWkbTypes.LineGeometry)
        self.rubber.setToGeometry(QgsGeometry.fromPolylineXY([self._map(point) for point in points]), None)
        self.rubber.show()

    def _build_livewire_preview(self):
        self._generation += 1
        request = self._generation
        self._read_evidence()
        path = trace_ink_path(self._evidence, self._local(self.anchor_full), self._local(self.end_full), guidance=self._guidance())
        if request != self._generation:
            return
        self._set_preview([(x + self._origin[0], y + self._origin[1]) for x, y in path.points])
        self._message("Info", "Green Ink preview ready. Press Enter to add it, G for a label-gap preview, or Esc to cancel.")

    def _build_gap_preview(self):
        if self._evidence is None:
            self._read_evidence()
        first, second = self._local(self.anchor_full), self._local(self.end_full)
        first_tangent, second_tangent = sample_evidence_tangent(self._evidence, first), sample_evidence_tangent(self._evidence, second)
        if first_tangent is None or second_tangent is None:
            raise ManualGapBridgeError("Ink direction is ambiguous near one endpoint")
        bridge = build_manual_gap_bridge(first, second, first_tangent, second_tangent)
        self._set_preview([(x + self._origin[0], y + self._origin[1]) for x, y in bridge.points])
        cleanup = " Subpixel wiggle removed; clicked endpoints unchanged." if bridge.geometry_refinement and bridge.geometry_refinement["changed"] else ""
        self._message("Info", "Label-gap preview ready." + cleanup + " Press Enter to add it or Esc to cancel.")

    def _clear_preview(self, *, keep_guidance=True):
        self.anchor_full = self.end_full = None
        self._preview_points = None
        self._evidence = None
        if not keep_guidance:
            self.guide_boxes = []
        self.rubber.reset(QgsWkbTypes.LineGeometry)

    def canvasReleaseEvent(self, event):
        full = self._pixel(self.toMapCoordinates(event.pos()))
        if event.modifiers() & Qt.AltModifier:
            if self.guide_corner is None:
                self.guide_corner = full
                self._message("Info", "Select the opposite corner of the Ink avoidance area.")
            else:
                self.guide_boxes.append((*self.guide_corner, *full))
                self.guide_corner = None
                self._message("Info", "Soft avoidance area added. It raises path cost but never blocks a route.")
                if self.anchor_full is not None and self.end_full is not None:
                    try:
                        self._build_livewire_preview()
                    except LiveWireError as error:
                        self._message("Warning", str(error))
            return
        if self.anchor_full is None:
            self.anchor_full = full
            self._message("Info", "Select the trace endpoint.")
            return
        if self.end_full is None:
            self.end_full = full
            try:
                self._build_livewire_preview()
            except LiveWireError as error:
                self._clear_preview()
                self._message("Warning", str(error))
            return
        self._clear_preview()
        self.anchor_full = full
        self._message("Info", "New trace start selected. Select its endpoint.")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self._generation += 1
            self._clear_preview()
            self._message("Info", "Ink preview cancelled.")
            return
        if event.key() == Qt.Key_G and self.anchor_full is not None and self.end_full is not None:
            try:
                self._build_gap_preview()
            except (LiveWireError, ManualGapBridgeError) as error:
                self._message("Warning", str(error))
            return
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and self._preview_points:
            self._commit_preview()
            return
        super().keyPressEvent(event)

    def _commit_preview(self):
        feature = QgsFeature(self.target.fields())
        feature.setGeometry(QgsGeometry.fromPolylineXY([self._map(point) for point in self._preview_points]))
        self.target.beginEditCommand("Add user-confirmed Ink trace")
        if not self.target.addFeature(feature):
            self.target.destroyEditCommand()
            self._message("Warning", "Could not add the Ink trace to the editable target layer.")
            return
        self.target.endEditCommand()
        self.target.triggerRepaint()
        self._clear_preview()
        self._message("Success", "Ink trace added to the edit buffer. Use QGIS Undo to remove it before saving.")

    def deactivate(self):
        self._generation += 1
        self.rubber.reset(QgsWkbTypes.LineGeometry)
        super().deactivate()


__all__ = ["InkTraceDialog", "InkTraceMapTool"]
