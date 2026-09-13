#!/usr/bin/env python3
"""Build a new portable QGIS packet for accepting/rejecting AI-drawn routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from histcontour_core.provenance import sha256_file
from scripts.generate_assisted_contour_drawing import read, world


def build(report_path, output):
    from qgis.PyQt.QtCore import QMetaType
    from qgis.core import (
        Qgis, QgsBookmark, QgsCategorizedSymbolRenderer, QgsCoordinateReferenceSystem,
        QgsEditorWidgetSetup, QgsFeature, QgsField, QgsFillSymbol, QgsGeometry,
        QgsLineSymbol, QgsPointXY, QgsProject, QgsRasterLayer, QgsRectangle,
        QgsReferencedRectangle, QgsRendererCategory, QgsVectorFileWriter, QgsVectorLayer,
    )
    report_path, output = report_path.resolve(), output.resolve()
    report = read(report_path)
    if (report.get("schema") != "jap-map-assisted-contour-drawing/1" or report.get("holdout_used") is not False
            or report.get("human_approvals") != 0 or not report["proposals"]):
        raise ValueError("expected an unapproved development-only AI drawing report")
    if any(row.get("human_approved") is not False or row.get("dataset_role") != "review_only_not_training" for row in report["proposals"]):
        raise ValueError("AI routes must not contain human attestations or training labels")
    tiles = {tile["tile_id"]: tile for tile in report["tiles"]}
    crs_ids = {tile["crs_authid"] for tile in tiles.values()}
    if len(tiles) != 9 or len(crs_ids) != 1 or any(tile["split"] != "development" or tile["sheet_id"] == "178-gongju" for tile in tiles.values()):
        raise ValueError("review packet must contain the nine development sources in one CRS")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("sources", "images", "tile-previews"):
        shutil.copytree(report_path.parent/name, output/name)
    for name in ("drawing-report.json", "configuration.json", "upstream-pin.json", "report.html", "attempts.json"):
        shutil.copyfile(report_path.parent/name, output/name)
    for name in ("screening.json", "ai-visual-review.json"):
        if (report_path.parent/name).is_file():
            shutil.copyfile(report_path.parent/name, output/name)
    crs_id = next(iter(crs_ids))
    # These are native source-CRS coordinates, not RFC7946/WGS84 coordinates.
    # Preserve every vertex and declare the CRS in both files and the project.
    for name in ("base-lines.geojson", "ai-proposals.geojson"):
        collection = read(report_path.parent/name)
        collection["crs"] = {"type": "name", "properties": {"name": crs_id}}
        with (output/name).open("x", encoding="utf-8") as handle:
            json.dump(collection, handle, ensure_ascii=False, indent=2, allow_nan=False)
    project = QgsProject()
    project.setCrs(QgsCoordinateReferenceSystem(crs_id))
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    project.setTitle("AI가 먼저 그린 등고선 경로 — 사람은 채택·거절·수정 판단")
    gpkg = output/"ai-drawing-review.gpkg"
    project_path = output/"ai-drawing-review.qgz"
    string, integer, real = QMetaType.Type.QString, QMetaType.Type.Int, QMetaType.Type.Double

    def make(name, kind, fields):
        layer = QgsVectorLayer(f"{kind}?crs={crs_id}" if kind != "None" else "None", name, "memory")
        layer.dataProvider().addAttributes([QgsField(key, value) for key, value in fields])
        layer.updateFields()
        return layer

    fields = [(key, string) for key in ("proposal_id", "tile_id", "mode", "question", "dataset_role", "source_uid", "target_uid", "source_geometry_sha256")]
    fields += [("priority", integer), ("gap_pixels", real), ("competing_endpoint_pair", integer),
               ("review_status", string), ("annotator", string), ("human_approved", integer), ("review_note", string)]
    cases = make("review_cases", "Polygon", fields)
    corrections = make("human_corrections", "LineString", [("proposal_id", string), ("trace_kind", string), ("note", string)])
    metadata = make("review_metadata", "None", [("drawing_report_sha256", string), ("schema", string)])
    meta = QgsFeature(metadata.fields())
    meta.setAttributes([sha256_file(report_path), report["schema"]])
    metadata.dataProvider().addFeature(meta)
    from histcontour_core.human_feedback import geometry_digest
    original_proposals = {feature["properties"]["proposal_id"]: feature for feature in read(output/"ai-proposals.geojson")["features"]}
    for row in report["proposals"]:
        x1, y1, x2, y2 = row["pixel_box"]
        ring = world(tiles[row["tile_id"]], [(x1-.5, y1-.5), (x2-.5, y1-.5), (x2-.5, y2-.5), (x1-.5, y2-.5), (x1-.5, y1-.5)])
        feature = QgsFeature(cases.fields())
        feature.setGeometry(QgsGeometry.fromPolygonXY([[QgsPointXY(x, y) for x, y in ring]]))
        values = {**row, "source_geometry_sha256": geometry_digest(original_proposals[row["proposal_id"]]["geometry"]),
                  "review_status": "unreviewed", "human_approved": 0, "annotator": "", "review_note": "",
                  "competing_endpoint_pair": int(row["competing_endpoint_pair"])}
        feature.setAttributes([values[key] for key, _ in fields])
        cases.dataProvider().addFeature(feature)
    for number, layer in enumerate((cases, corrections, metadata)):
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName, options.layerName = "GPKG", layer.name()
        options.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteFile if number == 0 else QgsVectorFileWriter.CreateOrOverwriteLayer
        result = QgsVectorFileWriter.writeAsVectorFormatV3(layer, str(gpkg), project.transformContext(), options)
        if result[0] != QgsVectorFileWriter.NoError:
            raise RuntimeError(str(result))
    cases = QgsVectorLayer(f"{gpkg}|layername=review_cases", "1 · AI 경로 판단 — 채택·거절·수정 필요", "ogr")
    corrections = QgsVectorLayer(f"{gpkg}|layername=human_corrections", "2 · 필요한 경우만 수정선을 따로 그리기", "ogr")
    base = QgsVectorLayer(str(output/"base-lines.geojson"), "기존 후보 — 청록 · 읽기 전용", "ogr")
    proposed = QgsVectorLayer(str(output/"ai-proposals.geojson"), "AI가 추가로 그린 경로 — 읽기 전용", "ogr")
    for layer in (base, proposed):
        layer.setCrs(project.crs())
        layer.setReadOnly(True)
    base.renderer().setSymbol(QgsLineSymbol.createSimple({"line_color": "0,130,180,190", "line_width": "0.18"}))
    categories = []
    for mode, label, colour in (("ink_livewire", "잉크 지지 경로", "0,165,70,255"),
                               ("smart_recovery", "Recovery 제안", "180,0,200,255"),
                               ("contextual_gap", "공백 추론 — 사람 판단 필요", "235,110,0,255")):
        categories.append(QgsRendererCategory(mode, QgsLineSymbol.createSimple({"line_color": colour, "line_width": "0.6"}), label))
    proposed.setRenderer(QgsCategorizedSymbolRenderer("mode", categories))
    corrections.renderer().setSymbol(QgsLineSymbol.createSimple({"line_color": "220,30,60,255", "line_width": "0.5"}))
    cases.renderer().setSymbol(QgsFillSymbol.createSimple({"style": "no", "outline_style": "dash", "outline_color": "120,90,0,130", "outline_width": "0.12"}))
    form = cases.editFormConfig()
    editable = {"review_status", "annotator", "human_approved", "review_note"}
    for index, field in enumerate(cases.fields()):
        form.setReadOnly(index, field.name() not in editable)
        if field.name() in ("source_uid", "target_uid", "source_geometry_sha256"):
            cases.setEditorWidgetSetup(index, QgsEditorWidgetSetup("Hidden", {}))
    cases.setEditFormConfig(form)
    aliases = {"proposal_id": "경로 ID", "question": "판단할 질문", "review_status": "① 이 경로의 판단",
               "annotator": "② 판독자", "human_approved": "③ 직접 확인", "review_note": "④ 이유",
               "competing_endpoint_pair": "같은 끝점에 다른 연결 후보가 있음", "dataset_role": "용도 — 학습에 자동 투입하지 않음"}
    for field, label in aliases.items():
        cases.setFieldAlias(cases.fields().indexOf(field), label)
    mapping = [("unreviewed", "미검수"), ("accept", "이 경로 채택"), ("reject", "이 경로 거절"),
               ("needs_edit", "수정 필요"), ("unsure", "원본으로 판단 불가")]
    cases.setEditorWidgetSetup(cases.fields().indexOf("review_status"), QgsEditorWidgetSetup("ValueMap", {"map": [{label: key} for key, label in mapping]}))
    cases.setEditorWidgetSetup(cases.fields().indexOf("human_approved"), QgsEditorWidgetSetup("CheckBox", {"CheckedState": 1, "UncheckedState": 0}))
    cases.setEditorWidgetSetup(cases.fields().indexOf("review_note"), QgsEditorWidgetSetup("TextEdit", {"IsMultiline": True}))
    options = [{"사례 먼저 선택": ""}]+[{row["proposal_id"]: row["proposal_id"]} for row in report["proposals"]]
    corrections.setEditorWidgetSetup(corrections.fields().indexOf("proposal_id"), QgsEditorWidgetSetup("ValueMap", {"map": options}))
    corrections.setEditorWidgetSetup(corrections.fields().indexOf("trace_kind"), QgsEditorWidgetSetup("ValueMap", {"map": [{"보이는 등고선": "observed_contour"}, {"추론한 연결": "inferred_gap"}, {"제외할 획": "hard_negative"}]}))
    edit_group = project.layerTreeRoot().addGroup("AI 선을 먼저 보고 판단 — 자동 승인 없음")
    for layer in (corrections, proposed, cases, base):
        project.addMapLayer(layer, False)
        edit_group.addLayer(layer)
    source_group = project.layerTreeRoot().addGroup("동일 입력 원본 9타일")
    for tile in tiles.values():
        path = output/tile["raster_path"]
        if sha256_file(path) != tile["source_raster_sha256"]:
            raise ValueError("copied source changed")
        layer = QgsRasterLayer(str(path), tile["tile_id"])
        if not layer.isValid():
            raise ValueError("copied raster is not readable")
        provider_crs = layer.dataProvider().crs()
        if provider_crs.isValid() and provider_crs != project.crs():
            raise ValueError("raster CRS conflicts with the source tile metadata")
        layer.setCrs(project.crs())
        project.addMapLayer(layer, False)
        source_group.addLayer(layer)
    ordered = sorted(report["proposals"], key=lambda row: (row["priority"], row["proposal_id"]))
    case_geometries = {feature["proposal_id"]: feature.geometry() for feature in cases.getFeatures()}
    for index, row in enumerate(ordered):
        geometry = case_geometries[row["proposal_id"]]
        extent = geometry.boundingBox()
        extent.scale(1.1)
        bookmark = QgsBookmark()
        bookmark.setId(row["proposal_id"])
        bookmark.setName(f"{row['proposal_id']} · {row['mode']} · {row['tile_id']}")
        bookmark.setGroup(f"우선순위 {row['priority']}")
        bookmark.setExtent(QgsReferencedRectangle(extent, project.crs()))
        project.bookmarkManager().addBookmark(bookmark)
        if index == 0:
            project.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(extent, project.crs()))
    if not project.write(str(project_path)):
        raise RuntimeError("could not save the review project")
    project.clear()
    reopened = QgsProject()
    if not reopened.read(str(project_path)) or len(reopened.mapLayers()) != 13 or any(not layer.isValid() for layer in reopened.mapLayers().values()):
        raise RuntimeError("portable QGIS review project failed validation")
    if len(reopened.bookmarkManager().bookmarks()) != len(ordered):
        raise RuntimeError("proposal bookmarks are incomplete")
    reopened.clear()
    print({"project": str(project_path), "valid_layers": 13, "bookmarks": len(ordered), "human_approvals": 0}, flush=True)
    return project_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from qgis.testing import start_app
    app = start_app()
    build(args.report, args.output)


if __name__ == "__main__":
    main()
