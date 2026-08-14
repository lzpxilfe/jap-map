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
    QgsProject,
    QgsRasterLayer,
    QgsVectorFileWriter,
    QgsVectorLayer,
)


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


def main():
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    index_path = args.index.resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    candidate_index_path = index_path.parent / "candidates" / "candidate_index.json"
    candidate_index = json.loads(candidate_index_path.read_text(encoding="utf-8")) if candidate_index_path.exists() else None
    candidate_vector_index_path = index_path.parent / "candidate_vectors" / "candidate_vector_index.json"
    candidate_vector_index = json.loads(candidate_vector_index_path.read_text(encoding="utf-8")) if candidate_vector_index_path.exists() else None
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

        root = project.layerTreeRoot()
        development_group = root.addGroup("Development tiles")
        holdout_group = root.addGroup("Holdout test tiles — do not train")
        candidate_group = root.insertGroup(1, "Automatic line candidates — review only") if candidate_index else None
        proposal_group = root.insertGroup(1, "Automatic vector proposals — review only") if candidate_vector_index else None
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

        labels_group = root.insertGroup(0, "Annotation layers")
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
