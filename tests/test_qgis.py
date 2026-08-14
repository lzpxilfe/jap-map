import sys

from qgis.testing import start_app, unittest

start_app()

from qgis.PyQt.QtCore import QMetaType
from qgis.core import QgsCoordinateReferenceSystem, QgsFeature, QgsField, QgsGeometry, QgsPointXY, QgsProject, QgsVectorLayer

from histcontour_core.models import MapSheet
from jap_map.core.frame import Corner, CornerRole, SheetFrame
from jap_map.core.layer_manager import FrameLayerManager, GROUP_NAME, LAYER_ROLE, LAYER_ROLE_VALUE
from jap_map.dialog import MapFrameDialog
from jap_map.registration_dialog import RegisterMapDialog


class _MessageBar:
    def pushSuccess(self, _title, _message):
        return None

    def pushWarning(self, _title, _message):
        return None


class _Canvas:
    def zoomToSelected(self, _layer):
        return None


class _Iface:
    def __init__(self):
        self._bar = _MessageBar()
        self._canvas = _Canvas()

    def messageBar(self):
        return self._bar

    def mapCanvas(self):
        return self._canvas

    def setActiveLayer(self, _layer):
        self._active = _layer

    def activeLayer(self):
        return getattr(self, "_active", None)

    def mainWindow(self):
        return None

    def addToolBarIcon(self, _action):
        return None

    def removeToolBarIcon(self, _action):
        return None

    def addPluginToVectorMenu(self, _menu, _action):
        return None

    def removePluginVectorMenu(self, _menu, _action):
        return None


class QgisIntegrationTest(unittest.TestCase):
    def setUp(self):
        QgsProject.instance().removeAllMapLayers()

    def tearDown(self):
        QgsProject.instance().removeAllMapLayers()

    def test_layer_uses_selected_crs_and_stores_frame(self):
        frame = SheetFrame.create(
            "테스트 도엽",
            "EPSG:5132",
            {
                CornerRole.NW: Corner(CornerRole.NW, 127, 38),
                CornerRole.NE: Corner(CornerRole.NE, 128.2, 38.1),
                CornerRole.SE: Corner(CornerRole.SE, 128, 37),
                CornerRole.SW: Corner(CornerRole.SW, 127, 37),
            },
        )
        sheet = MapSheet("test-sheet", "원제", "테스트 도엽", "Series", "1", "Survey", "Topographic", "1930", "1931", "1:50000", 20, "ja", "Jpan", "EPSG:5132", "Unknown", "Archive", "Public domain")
        crs = QgsCoordinateReferenceSystem("EPSG:5132")
        layer, _feature_id = FrameLayerManager(_Iface()).add_frame(frame, crs, sheet)
        self.assertTrue(layer.isValid())
        self.assertEqual(layer.crs().authid(), "EPSG:5132")
        self.assertEqual(layer.featureCount(), 1)
        self.assertEqual(layer.customProperty(LAYER_ROLE), LAYER_ROLE_VALUE)
        self.assertEqual(layer.fields().indexOf("nw_x") >= 0, True)
        self.assertEqual(layer.fields().indexOf("sheet_id") >= 0, True)
        self.assertIsNotNone(QgsProject.instance().layerTreeRoot().findGroup(GROUP_NAME))

    def test_dialog_can_be_constructed(self):
        dialog = MapFrameDialog(None, _Iface())
        self.assertTrue(dialog.windowTitle())
        dialog._set_crs("EPSG:5132")
        self.assertEqual(dialog.crs_widget.crs().authid(), "EPSG:5132")
        dialog.close()

    def test_registration_dialog_accepts_printed_map_pixel_corners(self):
        dialog = RegisterMapDialog(None, _Iface())
        dialog.pixel_corners.setText("10,20;110,21;109,220;11,219")
        self.assertEqual(dialog._parse_pixel_corners(), ((10.0, 20.0), (110.0, 21.0), (109.0, 220.0), (11.0, 219.0)))
        dialog.close()

    def test_plugin_action_lifecycle(self):
        from jap_map.plugin import HistoricalMapTools

        plugin = HistoricalMapTools(_Iface())
        plugin.initGui()
        self.assertEqual(len(plugin.actions), 7)
        self.assertEqual(len(plugin.provider.algorithms()), 4)
        plugin.unload()
        self.assertEqual(plugin.actions, [])

    def test_review_action_persists_selected_status(self):
        from jap_map.review_actions import REVIEW_LAYER_PROPERTY, classify_selected_proposals

        layer = QgsVectorLayer("LineString?crs=EPSG:5132", "Quick review queue — development only", "memory")
        layer.dataProvider().addAttributes([QgsField("review_status", QMetaType.Type.QString)])
        layer.updateFields()
        feature = QgsFeature(layer.fields())
        feature.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(0, 0), QgsPointXY(1, 1)]))
        feature.setAttributes(["unreviewed"])
        layer.dataProvider().addFeature(feature)
        layer.setCustomProperty(REVIEW_LAYER_PROPERTY, True)
        feature_id = next(layer.getFeatures()).id()
        layer.selectByIds([feature_id])
        iface = _Iface()
        iface.setActiveLayer(layer)
        self.assertEqual(classify_selected_proposals(iface, "contour"), 1)
        self.assertEqual(next(layer.getFeatures())["review_status"], "contour")


def run_all():
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    return unittest.TextTestRunner(verbosity=2).run(suite)
