import sys
import json
from pathlib import Path
import tempfile

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
        self.assertEqual(len(plugin.actions), 9)
        self.assertEqual(len(plugin.provider.algorithms()), 5)
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
        self.assertEqual(classify_selected_proposals(iface, "symbol"), 1)
        self.assertEqual(next(layer.getFeatures())["review_status"], "symbol")

    def test_ink_processing_filters_only_with_explicit_model_and_threshold(self):
        from qgis.PyQt.QtGui import QColor, QImage, QPainter, QPen
        from qgis.core import QgsProcessingContext, QgsProcessingException, QgsProcessingFeedback, QgsRasterLayer
        from histcontour_core.segment_review import FEATURE_NAMES, LogisticModel
        from histcontour_core.provenance import INK_ADAPTER_VERSION
        from jap_map.processing_provider.ink_extract import ExtractInkProposalsAlgorithm

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raster_path = root / "synthetic-ink.png"
            image = QImage(96, 96, QImage.Format.Format_RGB32)
            image.fill(QColor("white"))
            painter = QPainter(image)
            painter.setPen(QPen(QColor("black"), 4))
            painter.drawLine(10, 48, 85, 48)
            painter.end()
            self.assertTrue(image.save(str(raster_path)))
            raster_path.with_suffix(".pgw").write_text("1\n0\n0\n-1\n0.5\n95.5\n", encoding="ascii")
            raster = QgsRasterLayer(str(raster_path), "fixture")
            self.assertTrue(raster.isValid())
            context = QgsProcessingContext()
            context.setProject(QgsProject.instance())
            algorithm = ExtractInkProposalsAlgorithm()
            algorithm.initAlgorithm()
            parameters = {"RASTER": raster, "MINIMUM_LENGTH": 18, "MODEL": "", "MINIMUM_SCORE": 0, "OUTPUT": "memory:"}
            output = algorithm.processAlgorithm(parameters, context, QgsProcessingFeedback())
            lines = context.getMapLayer(output["OUTPUT"])
            self.assertGreater(lines.featureCount(), 0)
            feature = next(lines.getFeatures())
            self.assertEqual(feature["adapter_version"], INK_ADAPTER_VERSION)
            self.assertEqual(feature["review_status"], "unreviewed")
            with self.assertRaisesRegex(QgsProcessingException, "model is required"):
                algorithm.processAlgorithm({**parameters, "MINIMUM_SCORE": .5}, context, QgsProcessingFeedback())
            model = LogisticModel(FEATURE_NAMES, (0.,)*len(FEATURE_NAMES), (1.,)*len(FEATURE_NAMES), (0.,)*len(FEATURE_NAMES), -20.)
            model_path = root / "synthetic-test-model.json"
            model_path.write_text(json.dumps({"status": "trained", "model": model.to_dict()}), encoding="utf-8")
            output = algorithm.processAlgorithm({**parameters, "MODEL": str(model_path), "MINIMUM_SCORE": .5}, context, QgsProcessingFeedback())
            self.assertEqual(context.getMapLayer(output["OUTPUT"]).featureCount(), 0)
            parameters.clear()
            del raster

    def test_context_comparison_is_new_read_only_and_preserves_inputs(self):
        from qgis.PyQt.QtGui import QColor, QImage
        from histcontour_core.provenance import sha256_file
        from scripts.build_contour_comparison_project import build_context
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raster = root/"fixture.png"
            image = QImage(32, 32, QImage.Format.Format_RGB32)
            image.fill(QColor("white"))
            self.assertTrue(image.save(str(raster)))
            raster.with_suffix(".pgw").write_text("1\n0\n0\n-1\n0.5\n31.5\n", encoding="ascii")
            line = root/"line.geojson"
            line.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"review_status": "unreviewed"}, "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}}]}), encoding="utf-8")
            tiles = [{"tile_id": f"dev-{i}", "sheet_id": f"source-{i//3}", "split": "development", "raster_path": str(raster)} for i in range(9)]
            index_path, vector_path, context_path = root/"index.json", root/"vectors.json", root/"context.json"
            index_path.write_text(json.dumps({"tiles": tiles}), encoding="utf-8")
            vector_path.write_text(json.dumps({"tiles": [{"tile_id": tile["tile_id"], "source_raster_sha256": sha256_file(raster)} for tile in tiles]}), encoding="utf-8")
            snapshots = {}
            for name in ("balanced", "conservative"):
                path = root/f"{name}-experiment.json"
                path.write_text("{}", encoding="utf-8")
                snapshots[f"{name}_experiment_sha256"] = sha256_file(path)
            context = {"schema": "jap-map-contour-context-candidates/1", "source_index_sha256": sha256_file(index_path),
                       "source_vector_index_sha256": sha256_file(vector_path), **snapshots,
                       "tiles": [{"tile_id": tile["tile_id"], "source_raster_sha256": sha256_file(raster), "all_scores_path": str(line),
                                  "methods": {name: {"path": str(line)} for name in ("legacy", "balanced", "conservative", "uncertain")}} for tile in tiles]}
            context_path.write_text(json.dumps(context), encoding="utf-8")
            before = (sha256_file(raster), sha256_file(line))
            output = root/"new-comparison.qgz"
            result = build_context(index_path, vector_path, context_path, output)
            self.assertEqual(result["layers"], 54)
            project = QgsProject()
            self.assertTrue(project.read(str(output)))
            vector_layers = [layer for layer in project.mapLayers().values() if isinstance(layer, QgsVectorLayer)]
            self.assertEqual(len(vector_layers), 45)
            self.assertTrue(all(layer.readOnly() for layer in vector_layers))
            project.clear()
            self.assertEqual(before, (sha256_file(raster), sha256_file(line)))
            with self.assertRaises(FileExistsError):
                build_context(index_path, vector_path, context_path, output)





def run_all():
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    return unittest.TextTestRunner(verbosity=2).run(suite)
