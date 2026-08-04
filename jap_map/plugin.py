"""QGIS plugin entry point and menu integration."""

from __future__ import annotations

import os

from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QDialog
from qgis.PyQt.QtGui import QAction

from .dialog import MapFrameDialog


class HistoricalMapTools:
    def __init__(self, iface):
        self.iface = iface
        self.action = None

    def initGui(self):
        icon_path = os.path.join(os.path.dirname(__file__), "icon.svg")
        self.action = QAction(QIcon(icon_path), "도곽 만들기…", self.iface.mainWindow())
        self.action.setObjectName("japMapCreateFrameAction")
        self.action.setStatusTip("네 귀퉁이 좌표로 역사 지형도 도곽을 만듭니다.")
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToVectorMenu("Historical Map Tools", self.action)

    def unload(self):
        if self.action is None:
            return
        self.iface.removePluginVectorMenu("Historical Map Tools", self.action)
        self.iface.removeToolBarIcon(self.action)
        self.action.deleteLater()
        self.action = None

    def run(self):
        dialog = MapFrameDialog(self.iface.mainWindow(), self.iface)
        exec_method = getattr(dialog, "exec", None) or dialog.exec_
        accepted = getattr(QDialog, "Accepted", None)
        if accepted is None:
            accepted = QDialog.DialogCode.Accepted
        if exec_method() == accepted:
            return
