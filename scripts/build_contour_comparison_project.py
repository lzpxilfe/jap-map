#!/usr/bin/env python3
"""Build a NEW, read-only QGIS A/B project for the nine development tiles."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from histcontour_core.provenance import sha256_file


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def build(index_path, before_path, after_path, score_path, output):
    from qgis.core import Qgis, QgsLineSymbol, QgsProject, QgsRasterLayer, QgsReferencedRectangle, QgsVectorLayer

    if output.exists():
        raise FileExistsError("comparison project already exists; use a new output name")
    index, before, after, scored = (read(path) for path in (index_path, before_path, after_path, score_path))
    tiles = [row for row in index["tiles"] if row["split"] == "development"]
    if len(tiles) != 9 or any(row["sheet_id"] == "178-gongju" for row in tiles):
        raise ValueError("the comparison project is development-only; Gongju is excluded")
    ids = {row["tile_id"] for row in tiles}
    groups = [{row["tile_id"]: row for row in value["tiles"]} for value in (before, after, scored)]
    if any(set(group) != ids for group in groups) or any(value.get("holdout_included") for value in (before, after, scored)):
        raise ValueError("input runs do not cover the same development tiles")
    if scored["source_ink_index_sha256"] != sha256_file(after_path):
        raise ValueError("score candidates do not belong to the corrected vector run")
    project = QgsProject()
    project.setTitle("등고선 추출 비교 — 개발 타일·검수 후보")
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    first_extent = None
    total_features = 0
    for number, tile in enumerate(tiles):
        tile_id = tile["tile_id"]
        raster_path = resolve(tile["raster_path"])
        digest = sha256_file(raster_path)
        if any(group[tile_id]["source_raster_sha256"] != digest for group in groups[:2]):
            raise ValueError("before/after raster provenance differs")
        raster = QgsRasterLayer(str(raster_path), "원본")
        if not raster.isValid():
            raise ValueError(f"invalid original raster: {tile_id}")
        group = project.layerTreeRoot().addGroup(tile_id)
        alternatives = group.addGroup("추출 비교 — 하나씩 켜기")
        entries = (("수정 전", groups[0][tile_id]["ink_vector_path"], "0,125,190,255"),
                   ("변환 오류 수정 · 필터 전", groups[1][tile_id]["ink_vector_path"], "224,123,0,255"),
                   (f"문맥 필터 ≥ {scored['filter_threshold']} · 검수 후보", groups[2][tile_id]["retained_path"], "230,28,80,255"))
        for label, path, color in entries:
            layer = QgsVectorLayer(str(resolve(path)), label, "ogr")
            if not layer.isValid():
                raise ValueError(f"invalid comparison vector layer: {path}")
            layer.renderer().setSymbol(QgsLineSymbol.createSimple({"line_color": color, "line_width": "0.25"}))
            layer.setReadOnly(True)
            layer.setCustomProperty("jap_map/review_only", True)
            project.addMapLayer(layer, False)
            alternatives.addLayer(layer)
            total_features += layer.featureCount()
        alternatives.setIsMutuallyExclusive(True, 2)
        project.addMapLayer(raster, False)
        group.addLayer(raster)
        group.setItemVisibilityChecked(number == 0)
        if first_extent is None:
            project.setCrs(raster.crs())
            first_extent = QgsReferencedRectangle(raster.extent(), raster.crs())
    project.viewSettings().setDefaultViewExtent(first_extent)
    project.writeEntry("jap-map", "interpretation", "Review candidates only. No human approvals, elevation assignment, gap filling, or holdout use. Original annotation GeoPackage is not included or changed.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not project.write(str(output)):
        raise RuntimeError("could not write the new QGIS comparison project")
    project.clear()
    # Read back the actual saved project, not only its in-memory construction.
    verified = QgsProject()
    if not verified.read(str(output)) or len(verified.mapLayers()) != 36 or any(not layer.isValid() for layer in verified.mapLayers().values()):
        raise RuntimeError("saved comparison project failed layer verification")
    verified.clear()
    return {"layers": 36, "development_tiles": 9, "vector_features": total_features, "human_approvals": 0, "holdout_used": False}


def build_context(index_path, vector_path, context_path, output):
    """A separate 54-layer project; never mutates an annotation project."""
    from qgis.core import Qgis, QgsLineSymbol, QgsProject, QgsRasterLayer, QgsReferencedRectangle, QgsVectorLayer
    if output.exists():
        raise FileExistsError("context comparison project already exists; use a new output")
    index, vectors, context = (read(path) for path in (index_path, vector_path, context_path))
    tiles = [row for row in index["tiles"] if row["split"] == "development"]
    ids = {row["tile_id"] for row in tiles}
    raw = {row["tile_id"]: row for row in vectors["tiles"]}
    scored = {row["tile_id"]: row for row in context["tiles"]}
    if (len(tiles) != 9 or any(row["sheet_id"] == "178-gongju" for row in tiles) or set(raw) != ids or set(scored) != ids
            or vectors.get("holdout_included") or context.get("holdout_included")
            or context.get("schema") != "jap-map-contour-context-candidates/1"
            or context.get("source_index_sha256") != sha256_file(index_path) or context.get("source_vector_index_sha256") != sha256_file(vector_path)):
        raise ValueError("context project inputs must be the same nine development sources")
    for name in ("balanced", "conservative"):
        if context[f"{name}_experiment_sha256"] != sha256_file(context_path.parent/f"{name}-experiment.json"):
            raise ValueError("context model snapshot changed")
    project = QgsProject()
    project.setTitle("주변 문맥 등고선 비교 — 연구 후보·불확실 선 보존")
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    first_extent, total_features = None, 0
    for number, tile in enumerate(tiles):
        sid = tile["tile_id"]
        raster_path = resolve(tile["raster_path"])
        digest = sha256_file(raster_path)
        if raw[sid]["source_raster_sha256"] != digest or scored[sid]["source_raster_sha256"] != digest:
            raise ValueError("context raster provenance changed")
        raster = QgsRasterLayer(str(raster_path), "원본")
        if not raster.isValid():
            raise ValueError("context source raster is invalid")
        group = project.layerTreeRoot().addGroup(sid)
        alternatives = group.addGroup("선별 비교 — 하나씩 켜기")
        for name, label, color in (("legacy", "기존 합성 필터 0.1", "0,125,190,255"),
                                   ("balanced", "새 주변 문맥 · 선별 후보", "230,28,80,255"),
                                   ("conservative", "새 주변 문맥 · 보수 후보", "50,130,80,255"),
                                   ("uncertain", "불확실 선 · 별도 재검수", "230,160,0,255"),
                                   ("all", "원래 모든 후보 + 점수 · 보존", "110,110,110,255")):
            path = scored[sid]["all_scores_path"] if name == "all" else scored[sid]["methods"][name]["path"]
            layer = QgsVectorLayer(str(resolve(path)), label, "ogr")
            if not layer.isValid():
                raise ValueError(f"invalid context vector layer: {path}")
            layer.renderer().setSymbol(QgsLineSymbol.createSimple({"line_color": color, "line_width": "0.25"}))
            layer.setReadOnly(True)
            layer.setCustomProperty("jap_map/review_only", True)
            project.addMapLayer(layer, False)
            node = (alternatives if name in ("legacy", "balanced", "conservative") else group).addLayer(layer)
            if name == "all":
                node.setItemVisibilityChecked(False)
            total_features += layer.featureCount()
        alternatives.setIsMutuallyExclusive(True, 1)
        project.addMapLayer(raster, False)
        group.addLayer(raster)
        group.setItemVisibilityChecked(number == 0)
        if first_extent is None:
            project.setCrs(raster.crs())
            first_extent = QgsReferencedRectangle(raster.extent(), raster.crs())
    project.viewSettings().setDefaultViewExtent(first_extent)
    project.writeEntry("jap-map", "interpretation", "Red = balanced review candidates. Orange = uncertain differences, NOT contour approvals. Gray all-score layer preserves every original line. Each model excludes its displayed source sheet. No elevations or new geometry.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not project.write(str(output)):
        raise RuntimeError("could not save context comparison project")
    project.clear()
    verified = QgsProject()
    if not verified.read(str(output)) or len(verified.mapLayers()) != 54 or any(not layer.isValid() for layer in verified.mapLayers().values()):
        raise RuntimeError("saved context project failed 54-layer verification")
    verified.clear()
    return {"layers": 54, "development_tiles": 9, "vector_features": total_features, "human_approvals": 0, "holdout_used": False}


def render_project(project_path, output):
    from qgis.PyQt.QtCore import QSize
    from qgis.core import QgsMapRendererSequentialJob, QgsMapSettings, QgsProject
    if output.exists():
        raise FileExistsError("preview already exists; use a new output")
    project = QgsProject()
    if not project.read(str(project_path)):
        raise ValueError("cannot read comparison project for preview")
    settings = QgsMapSettings()
    settings.setLayers([node.layer() for node in project.layerTreeRoot().findLayers() if node.isVisible()])
    settings.setDestinationCrs(project.crs())
    settings.setExtent(project.viewSettings().defaultViewExtent())
    settings.setOutputSize(QSize(1024, 1024))
    job = QgsMapRendererSequentialJob(settings)
    job.start()
    job.waitForFinished()
    image = job.renderedImage()
    if image.isNull() or not image.save(str(output), "PNG"):
        raise RuntimeError("comparison preview failed")
    del job, settings, image
    project.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--context-index", type=Path, help="optional new out-of-sheet context comparison; --after must be its raw vector index")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, help="optional new PNG rendered by QGIS after saving")
    args = parser.parse_args()
    # The testing initializer isolates QGIS's profile/auth DB in a temporary
    # directory. This standalone process cannot alter the user's open project.
    from qgis.testing import start_app
    app = start_app()
    if args.context_index is not None:
        if args.before is not None or args.scores is not None:
            parser.error("--context-index does not use --before or --scores")
        result = build_context(args.index, args.after, args.context_index, args.output)
    else:
        if args.before is None or args.scores is None:
            parser.error("legacy comparison needs --before and --scores")
        result = build(args.index, args.before, args.after, args.scores, args.output)
    print(json.dumps(result))
    if args.preview is not None:
        render_project(args.output, args.preview)


if __name__ == "__main__":
    main()
