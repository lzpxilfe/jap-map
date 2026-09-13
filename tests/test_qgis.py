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

    def test_assisted_drawing_review_is_new_portable_and_unapproved(self):
        import shutil
        from qgis.PyQt.QtGui import QColor, QImage
        from histcontour_core.provenance import sha256_file
        from scripts.build_assisted_contour_review import build
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/"source"
            source.mkdir()
            for name in ("sources","images","tile-previews"):
                (source/name).mkdir()
            image=QImage(64,64,QImage.Format.Format_RGB32)
            image.fill(QColor("white"))
            raster=source/"sources"/"fixture.png"
            self.assertTrue(image.save(str(raster)))
            raster.with_suffix(".pgw").write_text("1\n0\n0\n-1\n0.5\n63.5\n",encoding="ascii")
            digest=sha256_file(raster)
            tiles=[{"tile_id":f"synthetic-{index}","sheet_id":f"source-{index//3}","split":"development",
                    "crs_authid":"EPSG:3857","raster_path":"sources/fixture.png","source_raster_sha256":digest,
                    "bounds":[0,0,64,64],"pixel_bounds":[0,0,64,64]} for index in range(9)]
            row={"proposal_id":"A0001","tile_id":"synthetic-0","mode":"ink_livewire","question":"Synthetic only",
                 "dataset_role":"review_only_not_training","source_uid":"a","target_uid":"b","priority":1,
                 "gap_pixels":10.,"competing_endpoint_pair":False,"pixel_box":[0,0,64,64],"human_approved":False}
            report={"schema":"jap-map-assisted-contour-drawing/1","holdout_used":False,"human_approvals":0,
                    "tiles":tiles,"proposals":[row]}
            report_path=source/"drawing-report.json"
            report_path.write_text(json.dumps(report),encoding="utf-8")
            collection={"type":"FeatureCollection","features":[{"type":"Feature","properties":{"proposal_id":"A0001","mode":"ink_livewire"},
                        "geometry":{"type":"LineString","coordinates":[[10.5,43.5],[20.5,43.5]]}}]}
            for name in ("base-lines.geojson","ai-proposals.geojson"):
                (source/name).write_text(json.dumps(collection),encoding="utf-8")
            for name in ("configuration.json","upstream-pin.json","attempts.json"):
                (source/name).write_text("{}",encoding="utf-8")
            (source/"report.html").write_text("<html>fixture</html>",encoding="utf-8")
            output=root/"review"
            build(report_path,output)
            decisions=QgsVectorLayer(f"{output/'ai-drawing-review.gpkg'}|layername=review_cases","fixture","ogr")
            self.assertEqual(decisions.featureCount(),1)
            feature=next(decisions.getFeatures())
            self.assertEqual(feature["review_status"],"unreviewed")
            self.assertEqual(feature["human_approved"],0)
            del feature,decisions
            moved=root/"moved"
            shutil.copytree(output,moved)
            project=QgsProject()
            self.assertTrue(project.read(str(moved/"ai-drawing-review.qgz")))
            self.assertEqual(len(project.mapLayers()),13)
            self.assertEqual(len(project.bookmarkManager().bookmarks()),1)
            for layer in project.mapLayers().values():
                self.assertTrue(layer.isValid())
                self.assertTrue(Path(layer.source().split("|")[0]).resolve().is_relative_to(moved.resolve()))
                self.assertEqual(layer.crs().authid(), "EPSG:3857")
                if layer.source().endswith(".geojson"):
                    self.assertTrue(layer.readOnly())
                    self.assertEqual(layer.crs().authid(), "EPSG:3857")
            project.clear()
            self.assertEqual(sha256_file(raster),digest)
            with self.assertRaises(FileExistsError):
                build(report_path,output)

    def test_assisted_human_export_loads_only_explicitly_accepted_contours(self):
        from histcontour_core.assisted_review import build_review_outputs
        from tests.test_assisted_review import review_fixture
        output = build_review_outputs(*review_fixture())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"approved-contour.geojson"
            path.write_text(json.dumps(output["collections"]["approved_contour"]), encoding="utf-8")
            layer = QgsVectorLayer(str(path), "Synthetic reviewed additions", "ogr")
            self.assertTrue(layer.isValid())
            self.assertEqual(layer.crs().authid(), "EPSG:3857")
            self.assertEqual(layer.featureCount(), 2)
            ids = set()
            for feature in layer.getFeatures():
                ids.add(feature["proposal_id"])
                self.assertTrue(feature["human_approved"])
                self.assertEqual(feature["semantic_decision"], "contour")
                self.assertTrue(feature.geometry().isGeosValid())
            self.assertEqual(ids, {"A0000", "A0006-R1"})
            del feature, layer

    def test_short_gap_cleanup_is_only_a_preview_and_keeps_clicked_nodes(self):
        from unittest.mock import patch
        from qgis.PyQt.QtGui import QColor, QImage
        from qgis.core import QgsRasterLayer
        from qgis.gui import QgsMapCanvas
        from histcontour_core.gap_refinement import shape_audit
        from jap_map.ink_trace_tool import InkTraceMapTool
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"fixture.png"
            image = QImage(96,96,QImage.Format.Format_RGB32); image.fill(QColor("white"))
            self.assertTrue(image.save(str(path)))
            path.with_suffix(".pgw").write_text("1\n0\n0\n-1\n0.5\n95.5\n",encoding="ascii")
            raster = QgsRasterLayer(str(path),"fixture")
            target = QgsVectorLayer("LineString?crs=EPSG:3857","preview destination","memory")
            canvas = QgsMapCanvas(); tool = InkTraceMapTool(canvas,_Iface(),raster,target)
            tool.anchor_full,tool.end_full = (20.,40.),(30.,40.)
            tool._evidence = object()
            with patch("jap_map.ink_trace_tool.sample_evidence_tangent",return_value=(1.,.2)):
                tool._build_gap_preview()
            self.assertEqual(tool._preview_points[0],tool.anchor_full)
            self.assertEqual(tool._preview_points[-1],tool.end_full)
            self.assertEqual(shape_audit(tool._preview_points)["inflection_count"],0)
            self.assertEqual(target.featureCount(),0)
            self.assertFalse(target.isModified())
            tool._clear_preview(); self.assertIsNone(tool._preview_points)
            del tool,canvas,target,raster

    def test_feedback_vectorization_packet_is_portable_and_not_whole_line_approval(self):
        import shutil
        from qgis.PyQt.QtGui import QColor,QImage
        from histcontour_core.provenance import sha256_file
        from scripts.build_feedback_vectorization_project import build
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/"vectorized"; (output/"sources").mkdir(parents=True)
            raster = output/"sources/synthetic.png"
            image = QImage(32,32,QImage.Format.Format_RGB32);image.fill(QColor("white"))
            self.assertTrue(image.save(str(raster)))
            raster.with_suffix(".pgw").write_text("1\n0\n0\n-1\n0.5\n31.5\n",encoding="ascii")
            tile = {"tile_id":"synthetic","sheet_id":"synthetic","split":"development","crs_authid":"EPSG:3857",
                    "pixel_bounds":[0,0,32,32],"bounds":[0,0,32,32],"raster_path":"sources/synthetic.png","source_raster_sha256":sha256_file(raster)}
            feature = {"type":"Feature","geometry":{"type":"LineString","coordinates":[[2.5,20.5],[20.5,20.5]]},
                       "properties":{"tile_id":"synthetic","line_id":"C000001","proposal_id":"fixture","segment_uid":"base-fixture","human_approved":False,
                                     "whole_line_semantics_approved":False,"training_eligible":False}}
            collection = {"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"EPSG:3857"}},"features":[feature]}
            hashes = {}
            for name in ("contour-candidates","reviewed-connection-network","source-before","approved-connections","automatic-connections","needs-review"):
                content = json.loads(json.dumps(collection))
                if name == "approved-connections":content["features"][0]["properties"].update(human_approved=True,semantic_decision="contour")
                path = output/(name+".geojson");path.write_text(json.dumps(content),encoding="utf-8");hashes[path.name] = sha256_file(path)
            report = {"schema":"jap-map-feedback-vectorization/1","holdout_used":False,"new_human_approvals":0,
                      "whole_network_human_approved":False,"native_crs":"EPSG:3857","tiles":[tile],"output_sha256":hashes,
                      "counts":{"source_fragments":1,"enhanced_candidate_lines":1,"approved_connections":1,"automatic_connections":1}}
            (output/"vectorization-report.json").write_text(json.dumps(report),encoding="utf-8")
            result = build(output,previews=False)
            self.assertEqual(result["layers"],8)
            self.assertTrue(result["all_sources_inside_packet"])
            self.assertTrue(result["vector_layers_read_only"])
            layer = QgsVectorLayer(f"{output/'contour-vectorization.gpkg'}|layername=contour_candidates","network","ogr")
            self.assertEqual(layer.featureCount(),1)
            self.assertEqual(layer.crs().authid(),"EPSG:3857")
            for feature in layer.getFeatures():
                self.assertFalse(feature["human_approved"])
                self.assertFalse(feature["whole_line_semantics_approved"])
                self.assertTrue(feature.geometry().isGeosValid())
            del feature,layer
            moved = root/"relocated"; shutil.copytree(output,moved)
            project = QgsProject(); self.assertTrue(project.read(str(moved/"contour-vectorization.qgz")))
            for layer in project.mapLayers().values():
                self.assertTrue(layer.isValid())
                self.assertTrue(Path(layer.source().split('|')[0]).resolve().is_relative_to(moved.resolve()))
            project.clear(); del layer,project
            with self.assertRaises(FileExistsError):build(output,previews=False)

    def test_human_packet_round_trip_is_explicit_and_role_separated(self):
        import os
        import shutil
        import xml.etree.ElementTree as ET
        import zipfile
        from qgis.PyQt.QtGui import QColor, QImage
        from histcontour_core.human_feedback import geometry_digest
        from histcontour_core.provenance import sha256_file
        from scripts.build_contour_human_review import build
        from scripts.import_contour_human_feedback import import_feedback
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raster = root/"synthetic.png"
            image = QImage(64, 64, QImage.Format.Format_RGB32)
            image.fill(QColor("white"))
            self.assertTrue(image.save(str(raster)))
            raster.with_suffix(".pgw").write_text("1\n0\n0\n-1\n0.5\n63.5\n", encoding="ascii")
            digest = sha256_file(raster)
            tiles, cases = [], []
            for index in range(9):
                tile_id = f"synthetic-{index}"
                sample_id = "E001" if index == 8 else f"S{index+1:03}"
                role = "evaluation_only" if index == 8 else "training"
                tile = {"tile_id": tile_id, "sheet_id": f"source-{index//3}", "split": "development", "crs_authid": "EPSG:3857",
                        "raster_path": "synthetic.png", "source_raster_sha256": digest, "bounds": [0., 0., 64., 64.], "pixel_bounds": [0, 0, 64, 64]}
                tiles.append(tile)
                geometry = {"type": "LineString", "coordinates": [[10.5, 43.5], [30.5, 43.5]]}
                cases.append({"case_id": f"H{index+1:03}", "sample_id": sample_id, "dataset_role": role, "tile_id": tile_id, "sheet_id": tile["sheet_id"],
                              "segment_uid": sample_id, "source_raster_sha256": digest, "original_geometry": geometry, "original_geometry_sha256": geometry_digest(geometry),
                              "pixel_points": [[10., 20.], [30., 20.]], "pixel_box": [0, 0, 64, 64], "priority": 1, "prompt": "Synthetic contract fixture, not human evidence"})
            packet_path = root/"packet.json"
            packet_path.write_text(json.dumps({"schema": "jap-map-contour-human-packet/1", "holdout_used": False, "tiles": tiles, "cases": cases}), encoding="utf-8")
            result = build(Path(os.path.relpath(packet_path)))
            self.assertEqual(result["layers"], 13)
            self.assertEqual(result["bookmarks"], 9)
            with zipfile.ZipFile(root/"human-review.qgz") as archive:
                xml = ET.fromstring(archive.read(next(name for name in archive.namelist() if name.endswith(".qgs"))))
            sources = [node.text for node in xml.findall("./projectlayers/maplayer/datasource")]
            self.assertEqual(len(sources), 13)
            self.assertTrue(all(source.startswith("./") for source in sources))
            with tempfile.TemporaryDirectory() as moved_directory:
                moved = Path(moved_directory)/"relocated-packet"
                shutil.copytree(root, moved)
                relocated = QgsProject()
                self.assertTrue(relocated.read(str(moved/"human-review.qgz")))
                self.assertEqual(len(relocated.mapLayers()), 13)
                for layer in relocated.mapLayers().values():
                    self.assertTrue(layer.isValid())
                    self.assertTrue(Path(layer.source().split("|")[0]).resolve().is_relative_to(moved.resolve()))
                relocated.clear()
            gpkg = root/"human-review.gpkg"
            original_hash = sha256_file(gpkg)
            empty = import_feedback(packet_path, gpkg, root/"empty-feedback")
            self.assertEqual(empty["human_approval_count"], 0)
            self.assertEqual(sha256_file(gpkg), original_hash)
            # Deliberately edit ONLY a temporary synthetic fixture to exercise
            # the human-save path; no real review approvals are generated.
            decisions = QgsVectorLayer(f"{gpkg}|layername=review_cases", "fixture decisions", "ogr")
            self.assertTrue(decisions.startEditing())
            for feature in decisions.getFeatures():
                if str(feature["case_id"]) not in ("H001", "H009"):
                    continue
                values = {"review_status": "contour", "geometry_decision": "replace_with_trace" if str(feature["case_id"]) == "H001" else "accept_original",
                          "annotator": "Synthetic test reviewer", "human_approved": 1}
                for name, value in values.items():
                    self.assertTrue(decisions.changeAttributeValue(feature.id(), decisions.fields().indexOf(name), value))
            self.assertTrue(decisions.commitChanges())
            traces = QgsVectorLayer(f"{gpkg}|layername=human_traces", "fixture traces", "ogr")
            self.assertTrue(traces.startEditing())
            feature = QgsFeature(traces.fields())
            feature.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(11.5, 43.5), QgsPointXY(31.5, 43.5)]))
            feature["case_id"], feature["trace_kind"], feature["note"] = "H001", "observed_contour", "Synthetic one-pixel correction"
            self.assertTrue(traces.addFeature(feature))
            self.assertTrue(traces.commitChanges())
            del feature, traces, decisions
            feedback = import_feedback(packet_path, gpkg, root/"approved-fixture", previous_path=root/"empty-feedback"/"feedback.json")
            self.assertEqual(feedback["human_approval_count"], 2)
            self.assertEqual(len(feedback["training_labels"]), 0)
            self.assertEqual(len(feedback["evaluation_labels"]), 1)
            drawn = next(row for row in feedback["geometry_references"] if row["geometry_origin"] == "human_drawn")
            self.assertEqual(drawn["original_pixel_points"], [[10., 20.], [30., 20.]])
            self.assertEqual(drawn["human_pixel_points"], [[11., 20.], [31., 20.]])
            self.assertEqual(set(feedback["changed_decision_case_ids"]), {"H001", "H009"})
            with self.assertRaises(FileExistsError):
                build(packet_path)





def run_all():
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    return unittest.TextTestRunner(verbosity=2).run(suite)
