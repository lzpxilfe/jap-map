#!/usr/bin/env python3
"""Create a NEW portable QGIS human-review project, with zero approvals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from histcontour_core.human_feedback import validate_packet
from histcontour_core.provenance import sha256_file


def case_ring(case, tile):
    west, south, east, north = tile["bounds"]
    width, height = tile["pixel_bounds"][2:]
    x1, y1, x2, y2 = case["pixel_box"]
    left, right = west+x1*(east-west)/width, west+x2*(east-west)/width
    top, bottom = north-y1*(north-south)/height, north-y2*(north-south)/height
    return [(left, top), (right, top), (right, bottom), (left, bottom), (left, top)]


def build(packet_path):
    from qgis.PyQt.QtCore import QMetaType
    from qgis.core import (
        Qgis, QgsBookmark, QgsCoordinateReferenceSystem, QgsEditorWidgetSetup, QgsFeature, QgsField,
        QgsFillSymbol, QgsGeometry, QgsLineSymbol, QgsPointXY, QgsProject, QgsRasterLayer, QgsRectangle,
        QgsReferencedRectangle, QgsVectorFileWriter, QgsVectorLayer,
    )
    # QGIS needs absolute provider paths before it can serialize project-relative
    # paths. Passing cwd-relative strings can appear valid only in that cwd.
    packet_path = packet_path.resolve()
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    cases, tiles = validate_packet(packet)
    root = packet_path.parent
    gpkg, project_path = root/"human-review.gpkg", root/"human-review.qgz"
    if gpkg.exists() or project_path.exists():
        raise FileExistsError("human review files already exist; never rebuild over human work")
    crs_ids = {tile["crs_authid"] for tile in tiles.values()}
    if len(crs_ids) != 1:
        raise ValueError("the review packet requires one declared source CRS")
    crs_id = next(iter(crs_ids))
    project = QgsProject()
    project.setCrs(QgsCoordinateReferenceSystem(crs_id))
    project.setTitle(f"사람의 등고선 판단·수정 — 핵심 {sum(row['priority'] == 1 for row in cases.values())} + 학습 비교 표본")
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    string, integer = QMetaType.Type.QString, QMetaType.Type.Int

    def layer(name, geometry, fields):
        value = QgsVectorLayer(f"{geometry}?crs={crs_id}" if geometry != "None" else "None", name, "memory")
        value.dataProvider().addAttributes([QgsField(key, kind) for key, kind in fields])
        value.updateFields()
        if not value.isValid():
            raise ValueError(f"could not construct review layer {name}")
        return value

    case_fields = [(key, string) for key in ("case_id", "sample_id", "dataset_role", "tile_id", "sheet_id", "segment_uid", "source_raster_sha256", "original_geometry_sha256")]
    case_fields += [("priority", integer), ("prompt", string), ("review_status", string), ("geometry_decision", string), ("annotator", string), ("human_approved", integer), ("review_note", string)]
    areas = layer("review_cases", "Polygon", case_fields)
    originals = layer("source_candidates", "LineString", [(key, string) for key in ("case_id", "sample_id", "tile_id", "segment_uid", "original_geometry_sha256")])
    traces = layer("human_traces", "LineString", [("case_id", string), ("trace_kind", string), ("note", string)])
    ignored = layer("human_ignore", "Polygon", [("case_id", string), ("reason", string)])
    metadata = layer("review_metadata", "None", [("packet_sha256", string), ("schema", string)])
    row = QgsFeature(metadata.fields())
    row.setAttributes([sha256_file(packet_path), packet["schema"]])
    metadata.dataProvider().addFeature(row)
    for case in cases.values():
        area = QgsFeature(areas.fields())
        area.setGeometry(QgsGeometry.fromPolygonXY([[QgsPointXY(x, y) for x, y in case_ring(case, tiles[case["tile_id"]])]]))
        values = {**case, "annotator": "", "human_approved": 0, "review_status": "unreviewed", "geometry_decision": "unreviewed", "review_note": ""}
        area.setAttributes([values[key] for key, _ in case_fields])
        areas.dataProvider().addFeature(area)
        target = QgsFeature(originals.fields())
        target.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(float(x), float(y)) for x, y in case["original_geometry"]["coordinates"]]))
        target.setAttributes([case[key] for key in ("case_id", "sample_id", "tile_id", "segment_uid", "original_geometry_sha256")])
        originals.dataProvider().addFeature(target)
    for number, source in enumerate((areas, originals, traces, ignored, metadata)):
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = "GPKG"
        options.layerName = source.name()
        options.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteFile if number == 0 else QgsVectorFileWriter.CreateOrOverwriteLayer
        result = QgsVectorFileWriter.writeAsVectorFormatV3(source, str(gpkg), project.transformContext(), options)
        if result[0] != QgsVectorFileWriter.NoError:
            raise RuntimeError(f"could not save review layer {source.name()}: {result}")
    loaded = {}
    for name in ("review_cases", "source_candidates", "human_traces", "human_ignore"):
        value = QgsVectorLayer(f"{gpkg}|layername={name}", name, "ogr")
        if not value.isValid():
            raise RuntimeError(f"saved human review layer is invalid: {name}")
        loaded[name] = value
    areas, originals, traces, ignored = (loaded[name] for name in ("review_cases", "source_candidates", "human_traces", "human_ignore"))
    areas.setName("1 · 사례 판정 — 표에서 상태·승인 입력")
    originals.setName("원래 대상 선 — 읽기 전용·파랑")
    traces.setName("2 · 사람이 그린 선 — 보라")
    ignored.setName("3 · 판독 불가 영역 — 사선")
    originals.setReadOnly(True)
    originals.renderer().setSymbol(QgsLineSymbol.createSimple({"line_color": "0,100,235,255", "line_width": "0.45"}))
    traces.renderer().setSymbol(QgsLineSymbol.createSimple({"line_color": "150,30,200,255", "line_width": "0.4"}))
    areas.renderer().setSymbol(QgsFillSymbol.createSimple({"style": "no", "outline_style": "dash", "outline_color": "170,120,0,180", "outline_width": "0.15"}))
    ignored.renderer().setSymbol(QgsFillSymbol.createSimple({"style": "b_diagonal", "color": "180,80,80,130", "outline_color": "180,80,80,200"}))
    editable = {"review_status", "geometry_decision", "annotator", "human_approved", "review_note"}
    form = areas.editFormConfig()
    for field in areas.fields():
        index = areas.fields().indexOf(field.name())
        if field.name() not in editable:
            form.setReadOnly(index, True)
        if field.name() in ("segment_uid", "source_raster_sha256", "original_geometry_sha256", "sheet_id"):
            areas.setEditorWidgetSetup(index, QgsEditorWidgetSetup("Hidden", {}))
    areas.setEditFormConfig(form)
    aliases = {"case_id": "사례 ID", "sample_id": "원 표본 ID", "dataset_role": "용도 — 평가 전용은 학습 금지", "priority": "우선순위", "tile_id": "원본 타일", "prompt": "판단할 내용",
               "review_status": "① 원래 선의 종류", "geometry_decision": "② 선 처리", "annotator": "③ 판독자 이름", "human_approved": "④ 직접 확인·승인", "review_note": "⑤ 판단 근거"}
    for name, alias in aliases.items():
        areas.setFieldAlias(areas.fields().indexOf(name), alias)

    def value_map(value, name, mapping):
        value.setEditorWidgetSetup(value.fields().indexOf(name), QgsEditorWidgetSetup("ValueMap", {"map": [{label: key} for key, label in mapping]}))

    value_map(areas, "review_status", [("unreviewed", "미검수"), ("contour", "등고선"), ("text", "문자"), ("road_river", "도로·하천·둑"), ("symbol", "지도 기호"), ("mixed", "여러 종류 혼합"), ("unsure", "판독 불가")])
    value_map(areas, "geometry_decision", [("unreviewed", "미결정"), ("accept_original", "원래 선 그대로 등고선 승인"), ("replace_with_trace", "정확한 선을 별도로 그리기"), ("reject", "원래 선 제외"), ("unsure", "불명확 — 학습 제외")])
    areas.setEditorWidgetSetup(areas.fields().indexOf("human_approved"), QgsEditorWidgetSetup("CheckBox", {"CheckedState": 1, "UncheckedState": 0}))
    areas.setEditorWidgetSetup(areas.fields().indexOf("review_note"), QgsEditorWidgetSetup("TextEdit", {"IsMultiline": True}))
    case_options = [("", "사례를 먼저 선택")]+[(row["case_id"], f"{row['case_id']} · {row['sample_id']} · {row['tile_id']}") for row in cases.values()]
    value_map(traces, "case_id", case_options)
    value_map(ignored, "case_id", case_options)
    value_map(traces, "trace_kind", [("", "미결정"), ("observed_contour", "실제로 보이는 등고선"), ("inferred_gap", "보이지 않지만 사람이 추론한 연결"), ("hard_negative", "도로·문자 등 제외할 획")])
    areas.setCustomProperty("historical_map_tools/review_queue", True)
    for name, value in loaded.items():
        value.setCustomProperty("jap_map/human_packet_sha256", sha256_file(packet_path))
        project.addMapLayer(value, False)
    edit_group = project.layerTreeRoot().addGroup("사람 검수·선 그리기 — 자동 승인 없음")
    for value in (traces, ignored, originals, areas):
        edit_group.addLayer(value)
    source_group = project.layerTreeRoot().addGroup("동일 입력 원본 — 9개 타일")
    for tile_id, tile in tiles.items():
        path = root/tile["raster_path"]
        if sha256_file(path) != tile["source_raster_sha256"]:
            raise ValueError("portable source tile changed")
        raster = QgsRasterLayer(str(path), tile_id)
        if not raster.isValid():
            raise RuntimeError(f"invalid source raster: {tile_id}")
        project.addMapLayer(raster, False)
        source_group.addLayer(raster)
    for case in cases.values():
        ring = case_ring(case, tiles[case["tile_id"]])
        xs, ys = zip(*ring)
        rectangle = QgsRectangle(min(xs), min(ys), max(xs), max(ys))
        rectangle.scale(1.18)
        bookmark = QgsBookmark()
        bookmark.setId(case["case_id"])
        bookmark.setName(f"{case['case_id']} · {case['sample_id']} · {case['tile_id']}")
        bookmark.setGroup(f"우선순위 {case['priority']}")
        bookmark.setExtent(QgsReferencedRectangle(rectangle, project.crs()))
        project.bookmarkManager().addBookmark(bookmark)
        if case is next(iter(cases.values())):
            project.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(rectangle, project.crs()))
    first_id = next(areas.getFeatures()).id()
    areas.selectByIds([first_id])
    project.writeEntry("jap-map", "human_review_instructions", "Use project spatial bookmarks H001…; select the matching case row. Edit decisions in review_cases, draw actual paths in human_traces, and mark unreadable polygons in human_ignore. Name the annotator and explicitly approve only after checking. Blue originals are immutable. E-series remains evaluation-only.")
    if not project.write(str(project_path)):
        raise RuntimeError("could not save the new human review project")
    project.clear()
    verified = QgsProject()
    if not verified.read(str(project_path)) or len(verified.mapLayers()) != 13 or any(not value.isValid() for value in verified.mapLayers().values()):
        raise RuntimeError("saved human review project failed 13-layer verification")
    bookmark_count = len(verified.bookmarkManager().bookmarks())
    if bookmark_count != len(cases):
        raise RuntimeError("saved case bookmarks are incomplete")
    verified.clear()
    return {"project": str(project_path), "gpkg": str(gpkg), "layers": 13, "cases": len(cases), "bookmarks": bookmark_count, "human_approvals": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    args = parser.parse_args()
    from qgis.testing import start_app
    app = start_app()
    print(json.dumps(build(args.packet), ensure_ascii=False))


if __name__ == "__main__":
    main()
