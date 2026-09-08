"""QGIS plugin entry point and menu integration."""

from __future__ import annotations

import os

from qgis.PyQt.QtGui import QAction, QIcon
from qgis.PyQt.QtWidgets import QDialog
from qgis.core import QgsApplication

from .dialog import MapFrameDialog
from .extract_dialog import ExtractContoursDialog
from .ink_trace_tool import InkTraceDialog, InkTraceMapTool
from .processing_provider.provider import HistoricalMapToolsProvider
from .registration_dialog import RegisterMapDialog
from .review_actions import classify_selected_proposals


class HistoricalMapTools:
    def __init__(self, iface):
        self.iface = iface
        self.actions = []
        self.provider = None
        self.ink_trace_tool = None

    def initProcessing(self):
        self.provider = HistoricalMapToolsProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        icon_path = os.path.join(os.path.dirname(__file__), "icon.svg")
        self.initProcessing()
        for object_name, label, status, callback in (
            ("historicalMapToolsCreateFrame", "Create Sheet Frame / 도곽 만들기…", "Create a historical map sheet frame.", self.run_create_frame),
            ("historicalMapToolsRegisterMap", "Register Map / 지도 맞추기…", "Register an original scan to a sheet frame.", self.run_register_map),
            ("historicalMapToolsExtractContours", "Extract Contours / 등고 추출하기…", "Extract visible contours from an original scan.", self.run_extract_contours),
            ("historicalMapToolsInkTrace", "Ink Trace / Ink 추적·라벨 공백 연결…", "Trace an explicit Ink path and preview label-gap bridges before saving.", self.run_ink_trace),
        ):
            action = QAction(QIcon(icon_path), label, self.iface.mainWindow())
            action.setObjectName(object_name)
            action.setStatusTip(status)
            action.triggered.connect(callback)
            self.iface.addToolBarIcon(action)
            self.iface.addPluginToVectorMenu("Historical Map Tools", action)
            self.actions.append(action)
        for object_name, label, shortcut, review_status in (
            ("historicalMapToolsReviewContour", "Mark selected: Contour / 등고선", "Ctrl+1", "contour"),
            ("historicalMapToolsReviewText", "Mark selected: Text / 글자", "Ctrl+2", "text"),
            ("historicalMapToolsReviewRoadRiver", "Mark selected: Road or river / 도로·하천", "Ctrl+3", "road_river"),
            ("historicalMapToolsReviewSymbol", "Mark selected: Map symbol / 지도 기호", "Ctrl+4", "symbol"),
            ("historicalMapToolsReviewUnsure", "Mark selected: Unsure / 보류", "Ctrl+0", "unsure"),
        ):
            action = QAction(QIcon(icon_path), label, self.iface.mainWindow())
            action.setObjectName(object_name)
            action.setShortcut(shortcut)
            action.setStatusTip("Classify selected features in the development-only review queue and save immediately.")
            action.triggered.connect(lambda checked=False, value=review_status: classify_selected_proposals(self.iface, value))
            self.iface.addToolBarIcon(action)
            self.iface.addPluginToVectorMenu("Historical Map Tools", action)
            self.actions.append(action)

    def unload(self):
        for action in self.actions:
            self.iface.removePluginVectorMenu("Historical Map Tools", action)
            self.iface.removeToolBarIcon(action)
            action.deleteLater()
        self.actions = []
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None

    def _run_dialog(self, dialog_type):
        dialog = dialog_type(self.iface.mainWindow(), self.iface)
        exec_method = getattr(dialog, "exec", None) or dialog.exec_
        accepted = getattr(QDialog, "Accepted", None)
        if accepted is None:
            accepted = QDialog.DialogCode.Accepted
        if exec_method() == accepted:
            return

    def run_create_frame(self):
        self._run_dialog(MapFrameDialog)

    def run_register_map(self):
        self._run_dialog(RegisterMapDialog)

    def run_extract_contours(self):
        self._run_dialog(ExtractContoursDialog)

    def run_ink_trace(self):
        dialog = InkTraceDialog(self.iface.mainWindow())
        exec_method = getattr(dialog, "exec", None) or dialog.exec_
        accepted = getattr(QDialog, "Accepted", None)
        if accepted is None:
            accepted = QDialog.DialogCode.Accepted
        if exec_method() != accepted:
            return
        raster, target = dialog.selected_layers()
        self.ink_trace_tool = InkTraceMapTool(self.iface.mapCanvas(), self.iface, raster, target)
        self.iface.mapCanvas().setMapTool(self.ink_trace_tool)
