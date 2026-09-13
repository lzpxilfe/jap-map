#!/usr/bin/env python3
"""Package the assembled vector scenario as portable GeoPackage and QGIS.

Run with QGIS's Python. Never operates on the user's open QGIS instance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from histcontour_core.provenance import sha256_file
from scripts.generate_assisted_contour_drawing import read, write


def build(directory, *, previews=True):
    from qgis.PyQt.QtCore import QMetaType, QSize
    from qgis.PyQt.QtGui import QColor, QFont, QImage, QPainter
    from qgis.core import (Qgis,QgsBookmark,QgsCoordinateReferenceSystem,QgsFeature,QgsField,QgsGeometry,
        QgsLineSymbol,QgsMapRendererSequentialJob,QgsMapSettings,QgsPointXY,QgsProject,QgsRasterLayer,
        QgsRectangle,QgsReferencedRectangle,QgsVectorFileWriter,QgsVectorLayer)
    directory = Path(directory).resolve()
    report = read(directory/"vectorization-report.json")
    if (report.get("schema") != "jap-map-feedback-vectorization/1" or report.get("holdout_used") is not False
            or report.get("new_human_approvals") != 0 or report.get("whole_network_human_approved") is not False):
        raise ValueError("expected development-only unapproved vectorization scenario")
    if not report["tiles"] or any(t["split"] != "development" or t["sheet_id"] == "178-gongju" for t in report["tiles"]):
        raise ValueError("held-out sources cannot enter vectorization display")
    gpkg = directory/"contour-vectorization.gpkg"; destination = directory/"contour-vectorization.qgz"
    crs = QgsCoordinateReferenceSystem(report["native_crs"])
    for path in (gpkg,destination,directory/"project-validation.json",directory/"previews",directory/"START-HERE.md"):
        if path.exists():
            raise FileExistsError("project outputs must be new: "+path.name)
    for name,digest in report["output_sha256"].items():
        if sha256_file(directory/name) != digest:
            raise ValueError("vectorization output changed: "+name)
    for tile in report["tiles"]:
        source = (directory/tile["raster_path"]).resolve()
        if not source.is_relative_to(directory) or sha256_file(source) != tile["source_raster_sha256"]:
            raise ValueError("portable source raster changed")
        probe = QgsRasterLayer(str(source),"source alignment check")
        extent = probe.extent(); expected = tile["bounds"]
        actual = [extent.xMinimum(),extent.yMinimum(),extent.xMaximum(),extent.yMaximum()]
        pixel_size = min((expected[2]-expected[0])/tile["pixel_bounds"][2],(expected[3]-expected[1])/tile["pixel_bounds"][3])
        if (not probe.isValid() or (probe.width(),probe.height()) != tuple(tile["pixel_bounds"][2:])
                or max(abs(a-b) for a,b in zip(actual,expected)) > pixel_size*1e-4):
            raise ValueError("raster georeferencing or size differs from source pixel coordinates")
        provider_crs = probe.dataProvider().crs()
        if provider_crs.isValid() and provider_crs != crs:
            raise ValueError("raster source CRS differs from metadata")
        del probe
    project = QgsProject()
    project.setCrs(crs); project.setFilePathStorage(Qgis.FilePathType.Relative)
    project.setTitle("피드백 적용 등고선 벡터화 — 자동 후보와 승인 연결 구분")
    string,integer,real = QMetaType.Type.QString,QMetaType.Type.Int,QMetaType.Type.Double
    network_fields = [(key,string) for key in ("line_id","tile_id","review_status","dataset_role","elevation_status","geometry_sha256")]
    network_fields += [(key,integer) for key in ("part_count","source_part_count","approved_connection_count","automatic_connection_count","human_approved","whole_line_semantics_approved","training_eligible","closed")]
    network_fields += [("length_pixels",real)]
    connection_fields = [(key,string) for key in ("proposal_id","parent_proposal_id","original_proposal_id","tile_id","source_uid","target_uid",
        "semantic_decision","automatic_method","revision_kind","application_scope","not_assembled_reason")]
    connection_fields += [(key,integer) for key in ("human_approved","training_eligible","requires_source_tail_replacement","source_tail_replacement_applied_in_this_scenario")]
    definitions = [("contour_candidates","contour-candidates.geojson",network_fields),
                   ("reviewed_connection_network","reviewed-connection-network.geojson",network_fields),
                   ("source_before","source-before.geojson",[(key,string) for key in ("segment_uid","tile_id","review_status")]+[("forward_candidate_score",real)]),
                   ("approved_connections","approved-connections.geojson",connection_fields),
                   ("automatic_connections","automatic-connections.geojson",connection_fields),
                   ("needs_review","needs-review.geojson",connection_fields)]
    layer_counts = {}
    for number,(name,filename,fields) in enumerate(definitions):
        collection = read(directory/filename)
        if collection.get("crs",{}).get("properties",{}).get("name") != report["native_crs"]:
            raise ValueError("source coordinates lost their explicit CRS")
        if name == "approved_connections":
            if (len(collection["features"]) != report["counts"]["approved_connections"]
                    or any(f["properties"].get("human_approved") is not True
                           or f["properties"].get("semantic_decision") != "contour" for f in collection["features"])):
                raise ValueError("approved connector layer lacks explicit contour approvals")
        elif any(f["properties"].get("human_approved") is not False for f in collection["features"]):
            raise ValueError("candidate layers must not be human approvals")
        layer = QgsVectorLayer(f"LineString?crs={report['native_crs']}",name,"memory")
        layer.dataProvider().addAttributes([QgsField(key,kind) for key,kind in fields]); layer.updateFields()
        features = []
        for item in collection["features"]:
            feature = QgsFeature(layer.fields())
            feature.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(*p) for p in item["geometry"]["coordinates"]]))
            attrs = []
            for key,kind in fields:
                value = item["properties"].get(key)
                if isinstance(value,bool):value = int(value)
                attrs.append(value)
            feature.setAttributes(attrs); features.append(feature)
        if not layer.dataProvider().addFeatures(features)[0]:
            raise RuntimeError("could not add assembled vector features")
        options = QgsVectorFileWriter.SaveVectorOptions(); options.driverName = "GPKG"; options.layerName = name
        options.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteFile if number == 0 else QgsVectorFileWriter.CreateOrOverwriteLayer
        saved = QgsVectorFileWriter.writeAsVectorFormatV3(layer,str(gpkg),project.transformContext(),options)
        if saved[0] != QgsVectorFileWriter.NoError:
            raise RuntimeError(str(saved))
        layer_counts[name] = len(features)
        del features,layer
    # Read the actual stored data, not just OGR layer metadata. Also verify
    # every native-coordinate vertex so exported approvals cannot drift.
    verified_vertices = 0
    for name,filename,_fields in definitions:
        key = "line_id" if name in ("contour_candidates","reviewed_connection_network") else "segment_uid" if name == "source_before" else "proposal_id"
        expected = {f["properties"][key]:f for f in read(directory/filename)["features"]}
        check = QgsVectorLayer(f"{gpkg}|layername={name}","stored coordinate check","ogr")
        actual_count = 0
        for feature in check.getFeatures():
            identity = feature[key]
            coordinates = [[p.x(),p.y()] for p in feature.geometry().asPolyline()]
            if identity not in expected or coordinates != expected[identity]["geometry"]["coordinates"] or not feature.geometry().isGeosValid():
                raise RuntimeError("GeoPackage geometry round-trip changed or invalidated a feature")
            actual_count += 1; verified_vertices += len(coordinates)
        if not check.isValid() or actual_count != layer_counts[name]:
            raise RuntimeError("GeoPackage has missing stored features")
        del check
    (directory/"previews").mkdir()
    def vector(name,title,colour,width):
        layer = QgsVectorLayer(f"{gpkg}|layername={name}",title,"ogr")
        if not layer.isValid() or layer.crs() != crs:
            raise ValueError("saved GeoPackage layer is invalid or in another CRS")
        layer.setReadOnly(True); layer.setCustomProperty("jap_map/review_only",True)
        layer.renderer().setSymbol(QgsLineSymbol.createSimple({"line_color":colour,"line_width":str(width),"joinstyle":"round","capstyle":"round"}))
        project.addMapLayer(layer,False)
        return layer
    def render(layers,extent,size):
        settings = QgsMapSettings(); settings.setLayers(layers); settings.setDestinationCrs(crs)
        settings.setExtent(extent); settings.setOutputSize(size); settings.setBackgroundColor(QColor("white"))
        job = QgsMapRendererSequentialJob(settings); job.start(); job.waitForFinished()
        image = job.renderedImage().copy(); del job,settings
        if image.isNull():
            raise RuntimeError("empty map render")
        return image
    # Share source-wide vector providers across tiles. The map extent/spatial
    # index selects each tile, so bookmarks work without toggling 72 layers.
    root = project.layerTreeRoot()
    approved = vector("approved_connections","초록 · 사람 승인 연결만","0,145,65,255",.55)
    automatic = vector("automatic_connections","주황 · 자동 연결 후보 (미승인)","228,115,0,255",.45)
    pending = vector("needs_review","분홍 · 합치지 않은 검수 연결 (기본 숨김)","200,40,120,230",.45)
    root.addLayer(pending).setItemVisibilityChecked(False)
    alternatives = root.addGroup("선망 비교 · 하나씩 켜기")
    enhanced = vector("contour_candidates","강화 후 · 연속 벡터 후보","0,108,159,255",.22)
    human = vector("reviewed_connection_network","승인 연결만 합친 후보 선망","74,135,100,255",.22)
    before = vector("source_before","적용 전 · 기존 선분","125,125,125,255",.22)
    after_group = alternatives.addGroup("강화 후 · 자동 연결 포함 후보")
    for layer in (approved,automatic,enhanced):after_group.addLayer(layer)
    human_group = alternatives.addGroup("승인 연결만 적용 · 전체 선망은 미검수")
    human_approved = vector("approved_connections","초록 · 승인 연결","0,145,65,255",.55)
    human_group.addLayer(human_approved); human_group.addLayer(human)
    alternatives.addGroup("적용 전 · 원래 선분").addLayer(before)
    alternatives.setIsMutuallyExclusive(True,0)
    sources = root.addGroup("원본 지도 · 끄면 벡터만 보기")
    preview_files = []
    for number,tile in enumerate(report["tiles"]):
        tid = tile["tile_id"]
        raster = QgsRasterLayer(str(directory/tile["raster_path"]),tid)
        if not raster.isValid():raise ValueError("portable raster is unreadable")
        provider_crs = raster.dataProvider().crs()
        if provider_crs.isValid() and provider_crs != crs:raise ValueError("raster CRS differs from metadata")
        raster.setCrs(crs); project.addMapLayer(raster,False); sources.addLayer(raster)
        extent = QgsRectangle(*tile["bounds"])
        bookmark = QgsBookmark(); bookmark.setId(tid); bookmark.setName(tid+" · 강화된 벡터 후보")
        bookmark.setGroup(f"{len(report['tiles'])}개 개발 타일"); bookmark.setExtent(QgsReferencedRectangle(extent,crs))
        project.bookmarkManager().addBookmark(bookmark)
        if number == 0:project.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(extent,crs))
        if previews:
            width = 760; height = round(width*extent.height()/extent.width()); size = QSize(width,height)
            board = QImage(width*3+64,height+110,QImage.Format.Format_ARGB32); board.fill(QColor("white"))
            painter = QPainter(board); painter.setFont(QFont("Arial",16)); painter.setPen(QColor("#222222"))
            painter.drawText(16,28,tid+" | strengthened vectorization — review candidates")
            for index,(label,layers) in enumerate((("Source map",[raster]),("Vectors over source",[approved,automatic,enhanced,raster]),
                                                 ("Vector network only",[approved,automatic,enhanced]))):
                x = 16+index*(width+16); painter.drawText(x,58,label)
                painter.drawImage(x,70,render(layers,extent,size))
            painter.setFont(QFont("Arial",13)); painter.drawText(16,height+96,"Blue: candidate network   Green: approved connectors   Orange: automatic connectors   Elevation not assigned")
            painter.end(); path = directory/"previews"/(tid+".png")
            if not board.save(str(path),"PNG"):raise RuntimeError("could not save vector comparison")
            preview_files.append(str(path.relative_to(directory)))
    project.writeEntry("jap-map","interpretation","Assembled review-only scenario. Whole lines are not human approved. Green approval applies to connector geometry only; orange is inferred and unapproved. No elevation assignment or holdout processing.")
    if not project.write(str(destination)):raise RuntimeError("could not save vectorization project")
    project.clear()
    verified = QgsProject()
    if not verified.read(str(destination)) or len(verified.mapLayers()) != 7+len(report["tiles"]):
        raise RuntimeError("saved vectorization project has incomplete layers")
    for layer in verified.mapLayers().values():
        if (not layer.isValid() or layer.crs() != crs
                or not Path(layer.source().split('|')[0]).resolve().is_relative_to(directory)):
            raise RuntimeError("saved project is not portable or has an invalid layer")
    validation = {"schema":"jap-map-vectorization-project-validation/1","qgis_version":Qgis.QGIS_VERSION,
                  "layers":len(verified.mapLayers()),"bookmarks":len(verified.bookmarkManager().bookmarks()),
                  "gpkg_layer_counts":layer_counts,"native_crs":crs.authid(),"all_layers_valid":True,
                  "stored_coordinate_vertices_verified":verified_vertices,"all_stored_geometries_valid":True,
                  "all_sources_inside_packet":True,"vector_layers_read_only":all(l.readOnly() for l in verified.mapLayers().values() if isinstance(l,QgsVectorLayer)),
                  "project_sha256":sha256_file(destination),"gpkg_sha256":sha256_file(gpkg),"previews":preview_files}
    verified.clear(); write(directory/"project-validation.json",validation)
    counts = report["counts"]
    note = ("# 강화된 연결을 적용한 등고선 벡터 후보\n\n"
        f"기존 {len(report['tiles'])}개 개발 타일의 선분 {counts['source_fragments']:,}개를 실제 선망으로 합쳐 **{counts['enhanced_candidate_lines']:,}개 연속 벡터선**을 만들었습니다.\n\n"
        f"사람이 승인한 연결 {counts['approved_connections']}개와 미승인 자동 연결 {counts['automatic_connections']}개를 구분합니다. 전체 선망이 사람 검수 완료된 등고선이라는 뜻은 아닙니다.\n\n"
        "## 열기\n\n- [QGIS 프로젝트](contour-vectorization.qgz)\n- [GeoPackage 벡터 파일](contour-vectorization.gpkg)\n- [GeoJSON 연속 벡터 후보](contour-candidates.geojson)\n\n"
        "QGIS에서 첫 타일이 열립니다. 프로젝트 북마크로 다른 타일로 이동할 수 있습니다. "
        "파랑은 벡터 후보, 초록은 승인된 연결만, 주황은 자동 추론 연결입니다. 분홍색 검수 대상은 기본으로 숨겼습니다. "
        "‘선망 비교’에서 적용 전·승인 연결만·강화 후를 바꾸고 ‘원본 지도’를 끄면 벡터만 볼 수 있습니다.\n\n"
        "## 원본 보존과 한계\n\n"
        "이미 승인된 도형과 원본 패킷은 그대로입니다. 자동 연결은 이번 작업 사본에만 적용했고 사람 승인을 만들지 않았습니다. "
        "국소 수정은 지정한 원선 끝부분을 먼저 교체한 뒤 합쳤으며, 이미 잘린 끝부분을 중복으로 덧붙이지 않았습니다. "
        "근거가 약한 수정안과 문맥 검수 대상은 합치지 않았습니다. 등고선·하천·문자 혼입과 흐린 선 누락은 남아 있고, 표고는 부여하지 않았습니다. "
        "전체 도엽 재추출이 아니라 앞서 사용한 개발 타일의 벡터 선망 구성 결과입니다. 공주 홀드아웃은 제외했습니다.\n\n"
        "`network-membership.json`은 합쳐진 선에서 원래 선분·승인 연결·자동 연결의 출처를 추적합니다. "
        "`assembled-parts.geojson`과 `source-tail-replacements.geojson`은 이 결과를 만든 구성 요소입니다. 이미 적용된 결과에 다시 덧붙이지 마세요.\n\n"
        "## 타일별 원본·겹침·벡터만 비교\n\n"+"\n".join(f"- [{Path(p).stem}]({p})" for p in preview_files)+"\n")
    with (directory/"START-HERE.md").open("x",encoding="utf-8") as handle:handle.write(note)
    print(json.dumps(validation,ensure_ascii=False),flush=True)
    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("directory",type=Path)
    parser.add_argument("--no-previews",action="store_true"); args = parser.parse_args()
    from qgis.testing import start_app
    app = start_app()
    try:
        build(args.directory,previews=not args.no_previews)
    except (OSError,ValueError,RuntimeError) as error:
        parser.exit(2,f"Vectorization project stopped: {error}\n")
