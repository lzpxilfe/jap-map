"""Creation and reuse of temporary QGIS frame layers."""

from __future__ import annotations

import uuid

from qgis.PyQt.QtCore import QMetaType
from qgis.core import (
    QgsFeature,
    QgsField,
    QgsFillSymbol,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsSingleSymbolRenderer,
    QgsVectorLayer,
)

from .frame import CornerRole, SheetFrame


GROUP_NAME = "역사지형도 도곽"
LAYER_ROLE = "historical_map_tools/frame_layer"
LAYER_ROLE_VALUE = "corner_frames"


def _field_names() -> list[tuple[str, object]]:
    fields = [
        ("frame_id", QMetaType.Type.QString),
        ("sheet_name", QMetaType.Type.QString),
        ("crs_authid", QMetaType.Type.QString),
    ]
    for role in (CornerRole.NW, CornerRole.NE, CornerRole.SE, CornerRole.SW):
        fields.extend(
            (
                (f"{role.value.lower()}_lon", QMetaType.Type.Double),
                (f"{role.value.lower()}_lat", QMetaType.Type.Double),
            )
        )
    return fields


class FrameLayerManager:
    def __init__(self, iface):
        self.iface = iface

    def add_frame(self, frame: SheetFrame, crs):
        layer = self._find_layer(crs)
        created = False
        if layer is None:
            layer = self._create_layer(crs)
            created = True

        points = [QgsPointXY(lon, lat) for lon, lat in frame.ring_xy()]
        feature = QgsFeature(layer.fields())
        feature.setGeometry(QgsGeometry.fromPolygonXY([points]))
        feature.setAttributes(self._attributes(frame, crs))
        ok, added = layer.dataProvider().addFeatures([feature])
        if not ok or not added:
            if created:
                QgsProject.instance().removeMapLayer(layer.id())
            raise RuntimeError("도곽 피처를 레이어에 추가하지 못했습니다.")

        layer.updateExtents()
        layer.triggerRepaint()
        feature_id = added[0].id()
        self.iface.setActiveLayer(layer)
        layer.selectByIds([feature_id])
        self.iface.mapCanvas().zoomToSelected(layer)
        layer.removeSelection()
        self.iface.messageBar().pushSuccess("Historical Map Tools", f"'{frame.sheet_name}' 도곽을 생성했습니다.")
        return layer, feature_id

    def _find_layer(self, crs):
        for layer in QgsProject.instance().mapLayers().values():
            if not isinstance(layer, QgsVectorLayer):
                continue
            if layer.customProperty(LAYER_ROLE) == LAYER_ROLE_VALUE and layer.crs() == crs:
                return layer
        return None

    def _create_layer(self, crs):
        authid = crs.authid()
        uri = "Polygon"
        if authid:
            uri += f"?crs={authid}"
        layer = QgsVectorLayer(uri, self._layer_name(crs), "memory")
        if not layer.isValid():
            raise RuntimeError("임시 도곽 레이어를 만들지 못했습니다.")
        if not authid:
            layer.setCrs(crs)
        provider = layer.dataProvider()
        provider.addAttributes([QgsField(name, kind) for name, kind in _field_names()])
        layer.updateFields()
        layer.setCustomProperty(LAYER_ROLE, LAYER_ROLE_VALUE)
        layer.setCustomProperty("historical_map_tools/crs", crs.authid() or crs.toWkt())
        symbol = QgsFillSymbol.createSimple(
            {"color": "255,255,255,0", "outline_color": "210,55,45", "outline_width": "0.8"}
        )
        layer.setRenderer(QgsSingleSymbolRenderer(symbol))

        project = QgsProject.instance()
        project.addMapLayer(layer, False)
        root = project.layerTreeRoot()
        group = root.findGroup(GROUP_NAME)
        if group is None:
            group = root.insertGroup(0, GROUP_NAME)
        group.addLayer(layer)
        return layer

    @staticmethod
    def _layer_name(crs):
        label = crs.authid() or crs.description() or "사용자 CRS"
        return f"도곽 — {label}"

    @staticmethod
    def _attributes(frame, crs):
        values = [str(uuid.uuid4()), frame.sheet_name, crs.authid()]
        for corner in frame.corners:
            values.extend((corner.lon, corner.lat))
        return values
