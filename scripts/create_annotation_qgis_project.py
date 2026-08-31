"""Create a QGIS annotation project and empty GeoPackage review layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qgis.PyQt.QtCore import QMetaType
from qgis.PyQt.QtGui import QColor
from qgis.core import (
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsField,
    QgsFeature,
    QgsProject,
    QgsRasterLayer,
    QgsCategorizedSymbolRenderer,
    QgsLineSymbol,
    QgsRendererCategory,
    QgsVectorFileWriter,
    QgsVectorLayer,
)

REVIEW_LAYER_NAME = "proposal_review"
REVIEW_QUEUE_PROPERTY = "historical_map_tools/review_queue"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path, help="Generated annotation-package index.json")
    parser.add_argument("--project", type=Path, help="Output QGIS project; defaults beside index.json")
    return parser.parse_args()


def memory_layer(name: str, geometry: str, fields: list[tuple[str, QMetaType.Type]]):
    layer = QgsVectorLayer(f"{geometry}?crs=EPSG:5132", name, "memory")
    layer.dataProvider().addAttributes([QgsField(field_name, field_type) for field_name, field_type in fields])
    layer.updateFields()
    return layer


def write_layers(package_path: Path, layers):
    if package_path.exists():
        return
    for index, layer in enumerate(layers):
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GPKG"
        options.layerName = layer.name()
        options.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteFile if index == 0 else QgsVectorFileWriter.CreateOrOverwriteLayer
        result = QgsVectorFileWriter.writeAsVectorFormatV3(layer, str(package_path), QgsProject.instance().transformContext(), options)
        if result[0] != QgsVectorFileWriter.NoError:
            raise RuntimeError(f"Could not create {layer.name()}: {result}")


def layer_exists(package_path: Path, layer_name: str) -> bool:
    return QgsVectorLayer(f"{package_path}|layername={layer_name}", layer_name, "ogr").isValid()


def review_queue_layer():
    fields = [
        ("proposal_uid", QMetaType.Type.QString),
        ("tile_id", QMetaType.Type.QString),
        ("sheet_id", QMetaType.Type.QString),
        ("split", QMetaType.Type.QString),
        ("proposal_kind", QMetaType.Type.QString),
        ("backend", QMetaType.Type.QString),
        ("pixel_length", QMetaType.Type.Double),
        ("point_count", QMetaType.Type.Int),
        ("confidence", QMetaType.Type.Double),
        ("review_status", QMetaType.Type.QString),
        ("review_note", QMetaType.Type.QString),
        ("elevation_m", QMetaType.Type.Double),
    ]
    return memory_layer(REVIEW_LAYER_NAME, "LineString", fields)


def create_review_queue(package_path: Path, candidate_vector_index: dict, repository: Path):
    """Create once; later project rebuilds must retain human review decisions."""
    if layer_exists(package_path, REVIEW_LAYER_NAME):
        return
    queue = review_queue_layer()
    features = []
    for candidate in candidate_vector_index["tiles"]:
        if candidate["split"] == "holdout_test":
            continue
        source_path = Path(candidate["candidate_vector_path"])
        if not source_path.is_absolute():
            source_path = repository / source_path
        source = QgsVectorLayer(str(source_path), source_path.stem, "ogr")
        if not source.isValid():
            raise RuntimeError(f"Invalid candidate vector: {source_path}")
        for feature in source.getFeatures():
            output = QgsFeature(queue.fields())
            output.setGeometry(feature.geometry())
            output.setAttributes([
                f"{feature['tile_id']}:{feature['proposal_id']}",
                feature["tile_id"], feature["sheet_id"], feature["split"], feature["proposal_kind"], feature["backend"],
                feature["pixel_length"], feature["point_count"], feature["confidence"], "unreviewed", None, None,
            ])
            features.append(output)
    queue.dataProvider().addFeatures(features)
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GPKG"
    options.layerName = REVIEW_LAYER_NAME
    options.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteLayer
    result = QgsVectorFileWriter.writeAsVectorFormatV3(queue, str(package_path), QgsProject.instance().transformContext(), options)
    if result[0] != QgsVectorFileWriter.NoError:
        raise RuntimeError(f"Could not create review queue: {result}")


def apply_review_renderer(layer):
    categories = []
    for value, label, colour, width in (
        ("unreviewed", "Unreviewed", "#64748b", "0.35"),
        ("contour", "Contour", "#e11d48", "0.95"),
        ("text", "Text", "#2563eb", "0.75"),
        ("road_river", "Road or river", "#9333ea", "0.75"),
        ("unsure", "Unsure", "#f59e0b", "0.75"),
    ):
        symbol = QgsLineSymbol.createSimple({"color": colour, "width": width})
        categories.append(QgsRendererCategory(value, symbol, label))
    layer.setRenderer(QgsCategorizedSymbolRenderer("review_status", categories))


def apply_completion_renderer(layer):
    categories = []
    for value, label, colour, width in (
        ("ink_path", "Ink-supported short path", "#16a34a", "0.95"),
        ("hermite_occlusion", "Text/symbol occlusion interpolation", "#f97316", "1.05"),
        ("hermite_gap", "Blank/long gap interpolation", "#9333ea", "1.05"),
    ):
        symbol = QgsLineSymbol.createSimple({"color": colour, "width": width})
        categories.append(QgsRendererCategory(value, symbol, label))
    layer.setRenderer(QgsCategorizedSymbolRenderer("mode", categories))


def main():
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    index_path = args.index.resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    candidate_index_path = index_path.parent / "candidates" / "candidate_index.json"
    candidate_index = json.loads(candidate_index_path.read_text(encoding="utf-8")) if candidate_index_path.exists() else None
    candidate_vector_index_path = index_path.parent / "candidate_vectors" / "candidate_vector_index.json"
    candidate_vector_index = json.loads(candidate_vector_index_path.read_text(encoding="utf-8")) if candidate_vector_index_path.exists() else None
    ink_vector_index_path = index_path.parent / "ink_candidate_vectors" / "ink_candidate_vector_index.json"
    ink_vector_index = json.loads(ink_vector_index_path.read_text(encoding="utf-8")) if ink_vector_index_path.exists() else None
    completion_index_path = index_path.parent / "contour_completion_candidates" / "completion_candidate_index.json"
    completion_index = json.loads(completion_index_path.read_text(encoding="utf-8")) if completion_index_path.exists() else None
    output_project = args.project.resolve() if args.project else index_path.with_name("annotation_project.qgz")
    package_path = index_path.with_name("contour_annotations.gpkg")

    application = QgsApplication([], False)
    application.initQgis()
    try:
        project = QgsProject.instance()
        project.clear()
        project.setCrs(QgsCoordinateReferenceSystem("EPSG:5132"))
        contour = memory_layer("contour_gt", "MultiLineString", [("tile_id", QMetaType.Type.QString), ("sheet_id", QMetaType.Type.QString), ("elevation_m", QMetaType.Type.Double), ("review_status", QMetaType.Type.QString)])
        hard_negative = memory_layer("hard_negative", "MultiLineString", [("tile_id", QMetaType.Type.QString), ("sheet_id", QMetaType.Type.QString), ("class_name", QMetaType.Type.QString), ("review_status", QMetaType.Type.QString)])
        ignore_area = memory_layer("ignore_area", "MultiPolygon", [("tile_id", QMetaType.Type.QString), ("sheet_id", QMetaType.Type.QString), ("reason", QMetaType.Type.QString)])
        write_layers(package_path, (contour, hard_negative, ignore_area))
        if candidate_vector_index:
            create_review_queue(package_path, candidate_vector_index, repository)

        root = project.layerTreeRoot()
        development_group = root.addGroup("Development tiles")
        holdout_group = root.addGroup("Holdout test tiles — do not train")
        candidate_group = root.insertGroup(1, "Automatic line candidates — review only") if candidate_index else None
        proposal_group = root.insertGroup(1, "Automatic vector proposals — review only") if candidate_vector_index else None
        ink_proposal_group = root.insertGroup(1, "Ink v2 vector proposals — A/B review only") if ink_vector_index else None
        completion_group = None
        if completion_index:
            completion_group_name = (
                "Contour completion proposals — reviewed anchors"
                if completion_index.get("anchor_status") == "contour"
                else "Contour completion proposals — research only"
            )
            completion_group = root.insertGroup(1, completion_group_name)
        for tile in index["tiles"]:
            raster_path = Path(tile["raster_path"])
            if not raster_path.is_absolute():
                raster_path = repository / raster_path
            layer = QgsRasterLayer(str(raster_path), tile["tile_id"])
            if not layer.isValid():
                raise RuntimeError(f"Invalid annotation raster: {raster_path}")
            project.addMapLayer(layer, False)
            (holdout_group if tile["split"] == "holdout_test" else development_group).addLayer(layer)

        if candidate_index:
            for candidate in candidate_index["tiles"]:
                raster_path = Path(candidate["candidate_raster_path"])
                if not raster_path.is_absolute():
                    raster_path = repository / raster_path
                layer = QgsRasterLayer(str(raster_path), f"candidate — {candidate['tile_id']}")
                if not layer.isValid():
                    raise RuntimeError(f"Invalid candidate raster: {raster_path}")
                project.addMapLayer(layer, False)
                candidate_group.addLayer(layer)
            candidate_group.setItemVisibilityChecked(False)

        if candidate_vector_index:
            for candidate in candidate_vector_index["tiles"]:
                vector_path = Path(candidate["candidate_vector_path"])
                if not vector_path.is_absolute():
                    vector_path = repository / vector_path
                layer = QgsVectorLayer(str(vector_path), f"proposal — {candidate['tile_id']}", "ogr")
                if not layer.isValid():
                    raise RuntimeError(f"Invalid candidate vector: {vector_path}")
                layer.setCrs(QgsCoordinateReferenceSystem("EPSG:5132"))
                symbol = layer.renderer().symbol()
                symbol.setColor(QColor("#06b6d4"))
                symbol.setWidth(0.55)
                project.addMapLayer(layer, False)
                proposal_group.addLayer(layer)
            proposal_group.setItemVisibilityChecked(False)

        if ink_vector_index:
            for candidate in ink_vector_index["tiles"]:
                vector_path = Path(candidate["ink_vector_path"])
                if not vector_path.is_absolute():
                    vector_path = repository / vector_path
                layer = QgsVectorLayer(str(vector_path), f"ink A/B — {candidate['tile_id']}", "ogr")
                if not layer.isValid():
                    raise RuntimeError(f"Invalid Ink candidate vector: {vector_path}")
                layer.setCrs(QgsCoordinateReferenceSystem("EPSG:5132"))
                symbol = layer.renderer().symbol()
                symbol.setColor(QColor("#16a34a"))
                symbol.setWidth(0.65)
                project.addMapLayer(layer, False)
                ink_proposal_group.addLayer(layer)
            ink_proposal_group.setItemVisibilityChecked(False)

        if completion_index:
            for candidate in completion_index["tiles"]:
                if not candidate.get("completion_count"):
                    continue
                vector_path = Path(candidate["completion_vector_path"])
                if not vector_path.is_absolute():
                    vector_path = repository / vector_path
                layer = QgsVectorLayer(str(vector_path), f"completion — {candidate['tile_id']}", "ogr")
                if not layer.isValid():
                    raise RuntimeError(f"Invalid contour completion vector: {vector_path}")
                layer.setCrs(QgsCoordinateReferenceSystem("EPSG:5132"))
                apply_completion_renderer(layer)
                project.addMapLayer(layer, False)
                completion_group.addLayer(layer)
            completion_group.setItemVisibilityChecked(False)

        labels_group = root.insertGroup(0, "Annotation layers")
        if candidate_vector_index:
            queue = QgsVectorLayer(f"{package_path}|layername={REVIEW_LAYER_NAME}", "Quick review queue — development only", "ogr")
            if not queue.isValid():
                raise RuntimeError("Invalid proposal review queue")
            queue.setCustomProperty(REVIEW_QUEUE_PROPERTY, True)
            apply_review_renderer(queue)
            project.addMapLayer(queue, False)
            labels_group.addLayer(queue)
        for layer_name, colour in (("contour_gt", "#e11d48"), ("hard_negative", "#2563eb"), ("ignore_area", "#f59e0b")):
            layer = QgsVectorLayer(f"{package_path}|layername={layer_name}", layer_name, "ogr")
            if not layer.isValid():
                raise RuntimeError(f"Invalid annotation layer: {layer_name}")
            layer.renderer().symbol().setColor(QColor(colour))
            project.addMapLayer(layer, False)
            labels_group.addLayer(layer)
        if not project.write(str(output_project)):
            raise RuntimeError(f"Could not write project: {output_project}")
        print(package_path)
        print(output_project)
    finally:
        application.exitQgis()


if __name__ == "__main__":
    main()
