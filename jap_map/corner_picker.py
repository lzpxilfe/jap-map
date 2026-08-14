"""Interactive selection of the printed map neatline inside a scan."""

from __future__ import annotations

from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor, QFont, QImage, QPen, QPixmap
from qgis.PyQt.QtWidgets import (
    QDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


LEFT_BUTTON = Qt.LeftButton if hasattr(Qt, "LeftButton") else Qt.MouseButton.LeftButton


class ScanGraphicsView(QGraphicsView):
    """Image view that reports scene coordinates and supports wheel zoom."""

    imageClicked = pyqtSignal(float, float)

    def __init__(self, scene, pixmap_item, parent=None):
        super().__init__(scene, parent)
        self.pixmap_item = pixmap_item
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if event.button() == LEFT_BUTTON:
            point = self.mapToScene(event.pos())
            if self.pixmap_item.contains(self.pixmap_item.mapFromScene(point)):
                self.imageClicked.emit(point.x(), point.y())
                event.accept()
                return
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        self.scale(1.25 if event.angleDelta().y() > 0 else 0.8, 1.25 if event.angleDelta().y() > 0 else 0.8)


class ImageCornerPickerDialog(QDialog):
    """Collect NW, NE, SE, SW pixel locations on a scan."""

    LABELS = ("NW", "NE", "SE", "SW")

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        image = QImage(image_path)
        if image.isNull():
            raise ValueError("Original scan could not be read")
        self.setWindowTitle("Pick Printed Map Corners / 도곽 모서리 선택")
        self.resize(1000, 720)
        self._points: list[tuple[float, float]] = []
        self._markers = []

        root = QVBoxLayout(self)
        self.instruction = QLabel()
        self.instruction.setWordWrap(True)
        root.addWidget(self.instruction)

        scene = QGraphicsScene(self)
        pixmap = QPixmap.fromImage(image)
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        scene.addItem(self.pixmap_item)
        self.view = ScanGraphicsView(scene, self.pixmap_item, self)
        self.view.imageClicked.connect(self._add_point)
        root.addWidget(self.view, 1)

        buttons = QHBoxLayout()
        reset = QPushButton("Reset / 초기화")
        undo = QPushButton("Undo / 실행 취소")
        cancel = QPushButton("Cancel / 취소")
        self.use_button = QPushButton("Use corners / 모서리 사용")
        reset.clicked.connect(self._reset)
        undo.clicked.connect(self._undo)
        cancel.clicked.connect(self.reject)
        self.use_button.clicked.connect(self.accept)
        buttons.addWidget(reset)
        buttons.addWidget(undo)
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        buttons.addWidget(self.use_button)
        root.addLayout(buttons)
        self._update_instruction()

    def showEvent(self, event):
        super().showEvent(event)
        self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio if hasattr(Qt, "KeepAspectRatio") else Qt.AspectRatioMode.KeepAspectRatio)

    def _update_instruction(self):
        if len(self._points) < 4:
            self.instruction.setText(
                f"Click the printed map neatline corner {self.LABELS[len(self._points)]}. "
                "Order: NW → NE → SE → SW. Use the mouse wheel to zoom."
            )
        else:
            self.instruction.setText("Four printed-map corners selected. Confirm or undo the last point.")
        self.use_button.setEnabled(len(self._points) == 4)

    def _add_point(self, x: float, y: float):
        if len(self._points) >= 4:
            return
        label = self.LABELS[len(self._points)]
        self._points.append((x, y))
        size = max(self.pixmap_item.pixmap().width(), self.pixmap_item.pixmap().height()) / 260
        pen = QPen(QColor("#e11d48"), max(2.0, size / 4))
        ellipse = self.view.scene().addEllipse(x - size, y - size, size * 2, size * 2, pen)
        text = self.view.scene().addSimpleText(label)
        font = QFont(); font.setPixelSize(max(18, round(size * 2.2))); text.setFont(font); text.setBrush(QColor("#e11d48")); text.setPos(x + size, y + size)
        self._markers.append((ellipse, text))
        self._update_instruction()

    def _undo(self):
        if not self._points:
            return
        self._points.pop()
        for item in self._markers.pop():
            self.view.scene().removeItem(item)
        self._update_instruction()

    def _reset(self):
        while self._points:
            self._undo()

    def selected_corners(self) -> tuple[tuple[float, float], ...]:
        return tuple(self._points)

