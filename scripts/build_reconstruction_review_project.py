#!/usr/bin/env python3
"""Portable QGIS comparison of reconstruction stages and explicit local feedback."""
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from histcontour_core.provenance import sha256_file
from histcontour_core.human_feedback import validate_pixel_points
from scripts.prepare_contour_reconstruction_review import read,write,validate_regions
from histcontour_core.recovery_export import export_recovery_additions
from histcontour_core.context_gap_preview import resolve_straight_feedback


def verify_feedback_lineage(application,drawing_path):
    """An old local approval belongs to its frozen preview, not today's rerun."""
    hashes=application['input_sha256']
    if len(hashes)>32 or hashes.get(str(drawing_path))!=sha256_file(drawing_path):
        raise ValueError('feedback source drawing changed')
    reports=[Path(p) for p in hashes if Path(p).name=='gap-context-report.json']
    if len(reports)!=1 or sha256_file(reports[0])!=hashes[str(reports[0])]:
        raise ValueError('original approved preview changed or is missing')
    decisions=[]
    for p,digest in hashes.items():
        path=Path(p)
        if path.suffix!='.json' or path in (drawing_path,reports[0]):continue
        if sha256_file(path)!=digest:raise ValueError('feedback decision input changed')
        record=read(path)
        if record.get('schema')=='jap-map-context-gap-feedback/1':decisions.append((path,record))
    if len(decisions)!=1:raise ValueError('explicit feedback decision not uniquely found')
    resolved=resolve_straight_feedback(read(reports[0]),decisions[0][1],hashes[str(reports[0])])
    if resolved!=application['revisions']:raise ValueError('saved local revisions differ from explicit feedback')
    return [reports[0],decisions[0][0]]


def build(packet,manifest_path,separation,curves,feedback,output,*,short_recovery=None):
    from qgis.core import (Qgis,QgsProject,QgsVectorLayer,QgsRasterLayer,QgsCoordinateReferenceSystem,
        QgsLineSymbol,QgsRectangle,QgsReferencedRectangle,QgsBookmark,QgsMapSettings,QgsMapRendererSequentialJob)
    from qgis.PyQt.QtCore import QSize
    from qgis.PyQt.QtGui import QColor
    packet,manifest_path,separation,curves,feedback,output=map(lambda p:Path(p).resolve(),
        (packet,manifest_path,separation,curves,feedback,output))
    if output.exists():raise FileExistsError('project output must be new')
    if any(p==output or p in output.parents for p in (packet,separation,curves,feedback)):
        raise ValueError('project output must be outside original snapshots')
    manifest=read(manifest_path);drawing_path=packet/'drawing-report.json';drawing=read(drawing_path)
    tiles=validate_regions(manifest,drawing)
    if sha256_file(drawing_path)!=manifest['drawing_report_sha256']:raise ValueError('source drawing report changed')
    sr=read(separation/'numeric-separation-report.json');cr=read(curves/'gap-context-report.json')
    fr=read(feedback/'feedback-application.json')
    if sr.get('holdout_used') is not False or cr.get('connections_applied')!=0 or cr.get('human_approvals')!=0:
        raise ValueError('expected separate unapproved reconstruction stages')
    if cr['input_and_code_sha256'].get(str(separation/'numeric-separation-report.json'))!=sha256_file(separation/'numeric-separation-report.json'):
        raise ValueError('curve context belongs to another numeric separation snapshot')
    feedback_lineage=verify_feedback_lineage(fr,drawing_path)
    local_features=read(feedback/'reviewed-local-connections.geojson')['features']
    if len(local_features)!=len(fr['revisions']):raise ValueError('local feedback feature count differs')
    for feature,revision in zip(local_features,fr['revisions']):
        tile=tiles[revision['tile_id']];west,south,east,north=tile['bounds'];w,h=tile['pixel_bounds'][2:]
        expected=[[west+(x+.5)*(east-west)/w,north-(y+.5)*(north-south)/h] for x,y in revision['points']]
        if (feature['geometry']!={'type':'LineString','coordinates':expected}
                or feature['properties']!={k:v for k,v in revision.items() if k not in ('points','previous_points')}):
            raise ValueError('local feedback geometry or approval scope changed')
    source_files=[drawing_path,manifest_path,separation/'numeric-separation-report.json',curves/'gap-context-report.json',
                  feedback/'feedback-application.json',feedback/'reviewed-local-connections.geojson',Path(__file__),
                  ROOT/'histcontour_core/recovery_export.py']
    source_files+=feedback_lineage+[ROOT/'histcontour_core/context_gap_preview.py']
    recovery_collection=None
    if short_recovery is not None:
        short_recovery=Path(short_recovery).resolve()
        if short_recovery.parent==output or short_recovery.parent in output.parents:raise ValueError('output inside short recovery snapshot')
        short_report=read(short_recovery)
        for path in (drawing_path,separation/'numeric-separation-report.json'):
            if short_report['input_sha256'].get(str(path))!=sha256_file(path):raise ValueError('short recovery source snapshot differs')
        recovery_collection=export_recovery_additions(short_report,tiles);source_files.append(short_recovery)
    for name,digest in sr['output_sha256'].items():
        p=(separation/name).resolve()
        if p.parent!=separation or sha256_file(p)!=digest:raise ValueError('separation output hash differs')
        source_files.append(p)
    for tid in tiles:
        p=(packet/tiles[tid]['raster_path']).resolve()
        if packet not in p.parents or sha256_file(p)!=tiles[tid]['source_raster_sha256']:
            raise ValueError('source raster hash differs')
        source_files.append(p)
    before={str(p):sha256_file(p) for p in source_files}
    output.mkdir(parents=True);(output/'sources').mkdir()
    for name in sr['output_sha256']:shutil.copyfile(separation/name,output/name)
    shutil.copyfile(feedback/'reviewed-local-connections.geojson',output/'reviewed-local-connections.geojson')
    # Serialize curves separately, without silently substituting the new straight
    # geometry into the frozen original candidate history.
    features=[]
    for case in cr['cases']:
        tile=tiles[case['tile_id']];west,south,east,north=tile['bounds'];w,h=tile['pixel_bounds'][2:]
        for curve in case['curve_previews']['previews']:
            if not curve['points']:continue
            validate_pixel_points(curve['points'],tile)
            coords=[[west+(x+.5)*(east-west)/w,north-(y+.5)*(north-south)/h] for x,y in curve['points']]
            features.append({'type':'Feature','properties':{'case_id':case['case_id'],'candidate_id':curve['candidate_id'],
                'tile_id':case['tile_id'],'human_approved':False,'training_eligible':False,
                'geometry_checks_passed':curve['geometry_checks_passed'],'inferred_gap':True},
                'geometry':{'type':'LineString','coordinates':coords}})
    authid=next(iter(tiles.values()))['crs_authid'];crs=QgsCoordinateReferenceSystem(authid)
    write(output/'curve-history.geojson',{'type':'FeatureCollection','crs':{'type':'name','properties':{'name':authid}},'features':features})
    project=QgsProject();project.setCrs(crs);project.setFilePathStorage(Qgis.FilePathType.Relative)
    project.setTitle('등고선 강화 검수 · 숫자 분리와 승인 연결 구분')
    root=project.layerTreeRoot();counts={};vector_layers={};verified_vertices=0
    definitions=[('reviewed-local-connections.geojson','청록 · 새 연결 승인 / 직선 허용','0,150,150',.55,True),
                 ('approved-connections-unchanged.geojson','초록 · 기존 승인 연결','0,140,50',.5,True),
                 ('curve-history.geojson','빨강 · 이전 곡선 후보 / 미승인 / 기본 숨김','220,40,40',.4,False),
                 ('numeric-candidates.geojson','자홍 · 숫자 획 후보 / 기본 숨김','200,30,150',.3,False),
                 ('retained.geojson','파랑 · 숫자 분리 후 관측선 후보','20,100,200',.18,True),
                 ('before.geojson','회색 · 숫자 분리 전 관측선 / 기본 숨김','120,120,120',.18,False)]
    if recovery_collection is not None:
        write(output/'short-recovery-additions.geojson',recovery_collection)
        definitions.insert(2,('short-recovery-additions.geojson','보라 · 짧은 복원 새 구간 / 미승인 / 기본 숨김','135,45,210',.5,False))
    for name,title,color,width,visible in definitions:
        data=read(output/name);layer=QgsVectorLayer(str(output/name),title,'ogr')
        if not layer.isValid() or layer.crs()!=crs or layer.featureCount()!=len(data['features']):
            raise ValueError('invalid portable vector layer')
        for actual,expected in zip(layer.getFeatures(),data['features']):
            coords=[[p.x(),p.y()] for p in actual.geometry().vertices()]
            if coords!=expected['geometry']['coordinates'] or not actual.geometry().isGeosValid():
                raise ValueError('invalid or changed stored native geometry')
            verified_vertices+=len(coords)
        counts[name]=layer.featureCount();layer.setReadOnly(True)
        layer.renderer().setSymbol(QgsLineSymbol.createSimple({'line_color':color,'line_width':str(width),'joinstyle':'round'}))
        layer.setCustomProperty('jap_map/review_only',True);project.addMapLayer(layer,False)
        root.addLayer(layer).setItemVisibilityChecked(visible);vector_layers[name]=layer
    sources=root.addGroup('원본 지도');rasters={}
    for tid,tile in tiles.items():
        path=output/'sources'/(tid+'.tif');shutil.copyfile(packet/tile['raster_path'],path)
        raster=QgsRasterLayer(str(path),tid)
        if not raster.isValid() or raster.crs()!=crs or [raster.width(),raster.height()]!=tile['pixel_bounds'][2:]:
            raise ValueError('portable raster grid or CRS differs')
        extent=raster.extent();actual=[extent.xMinimum(),extent.yMinimum(),extent.xMaximum(),extent.yMaximum()]
        if max(abs(a-b) for a,b in zip(actual,tile['bounds']))>1e-9:raise ValueError('portable raster extent differs')
        project.addMapLayer(raster,False);sources.addLayer(raster);rasters[tid]=raster
    initial=None
    for region in manifest['regions']:
        tile=tiles[region['tile_id']];west,south,east,north=tile['bounds'];w,h=tile['pixel_bounds'][2:]
        l,t,r,b=region['box'];extent=QgsRectangle(west+l*(east-west)/w,north-b*(north-south)/h,
                                              west+r*(east-west)/w,north-t*(north-south)/h)
        bookmark=QgsBookmark();bookmark.setId(region['region_id']);bookmark.setName(region['region_id']+' · '+region['tile_id'])
        bookmark.setExtent(QgsReferencedRectangle(extent,crs));project.bookmarkManager().addBookmark(bookmark)
        if region['region_id']=='R004':initial=(region,extent)
    region,extent=initial or (manifest['regions'][0],next(iter(rasters.values())).extent())
    project.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(extent,crs))
    project.writeEntry('jap-map','interpretation','Review-only observed drafts, not a completed contour map. Numeric candidates are not erased source pixels. New straight approval is local connection geometry only. No elevation or training labels assigned.')
    destination=output/'contour-reconstruction-review.qgz'
    if not project.write(str(destination)):raise RuntimeError('project save failed')
    settings=QgsMapSettings();settings.setDestinationCrs(crs);settings.setExtent(extent);settings.setOutputSize(QSize(1000,700))
    settings.setBackgroundColor(QColor('white'));settings.setLayers([vector_layers['reviewed-local-connections.geojson'],
        vector_layers['approved-connections-unchanged.geojson'],vector_layers['retained.geojson'],rasters[region['tile_id']]])
    job=QgsMapRendererSequentialJob(settings);job.start();job.waitForFinished()
    if not job.renderedImage().save(str(output/'R004-qgis-preview.png')):raise RuntimeError('preview save failed')
    check=QgsProject()
    if not check.read(str(destination)) or len(check.mapLayers())!=len(definitions)+len(tiles):raise ValueError('saved project lost layers')
    for layer in check.mapLayers().values():
        if not layer.isValid() or layer.crs()!=crs or not Path(layer.source().split('|')[0]).resolve().is_relative_to(output):
            raise ValueError('saved layer invalid or not portable')
        if isinstance(layer,QgsVectorLayer) and not layer.readOnly():raise ValueError('saved vector is not read-only')
    if len(check.bookmarkManager().bookmarks())!=len(manifest['regions']):raise ValueError('saved bookmarks lost')
    if any(sha256_file(Path(p))!=digest for p,digest in before.items()):raise ValueError('original input changed')
    validation={'schema':'jap-map-reconstruction-project-validation/1','qgis_version':Qgis.QGIS_VERSION,
        'layer_counts':counts,'layers':len(check.mapLayers()),'bookmarks':len(check.bookmarkManager().bookmarks()),
        'native_vertices_verified':verified_vertices,'input_sha256':before,'project_sha256':sha256_file(destination),
        'all_layer_paths_inside_output':True,'original_inputs_unchanged':True,'vector_layers_read_only':True,
        'whole_network_approved':False,'new_automatic_connections_applied':0}
    write(output/'project-validation.json',validation);print(json.dumps(validation,ensure_ascii=False))
    check.clear();project.clear();return validation


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','manifest','separation','curves','feedback'):p.add_argument(name,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--short-recovery',type=Path);a=p.parse_args()
    from qgis.core import QgsApplication
    app=QgsApplication([],False);app.initQgis()
    build(a.packet,a.manifest,a.separation,a.curves,a.feedback,a.output,short_recovery=a.short_recovery)
