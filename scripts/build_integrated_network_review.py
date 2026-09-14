#!/usr/bin/env python3
"""Portable comparison of observed, assembled, smoothed and semantic-review stages."""
import argparse
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from histcontour_core.provenance import sha256_file
from scripts.prepare_contour_reconstruction_review import read,write


def build(review,assembled,smoothed,semantic,output,*,topology_validation):
    from qgis.core import QgsProject,QgsVectorLayer,QgsLineSymbol,Qgis
    review,assembled,smoothed,semantic,output=map(lambda p:Path(p).resolve(),(review,assembled,smoothed,semantic,output))
    if output.exists():raise FileExistsError('integrated review output must be new')
    if any(p==output or p in output.parents for p in (review,assembled,smoothed,semantic)):raise ValueError('output inside input snapshot')
    project_path=review/'contour-reconstruction-review.qgz';validation=read(review/'project-validation.json')
    if sha256_file(project_path)!=validation['project_sha256']:raise ValueError('source project changed')
    assembly=read(assembled/'assembly-report.json');smoothing=read(smoothed/'smoothing-report.json');semantics=read(semantic/'semantic-overlap-report.json')
    assembly_path=assembled/'assembled-candidate-network.geojson';smooth_path=smoothed/'smoothed-candidate-network.geojson'
    topology_path=Path(topology_validation).resolve();topology=read(topology_path)
    if (topology.get('schema')!='jap-map-network-topology-validation/1' or topology.get('passed') is not True
            or topology.get('existing_pairs_checked_for_added_or_moved_contacts') is not True
            or topology.get('before_sha256')!=sha256_file(assembly_path) or topology.get('after_sha256')!=sha256_file(smooth_path)
            or any(topology.get(k)!=[] for k in ('invalid_after_line_ids','moved_endpoint_line_ids',
                'new_self_intersection_line_ids','new_line_intersection_pairs'))):
        raise ValueError('missing, stale or failed smoothing topology validation')
    for report,path in ((assembly,review/'retained.geojson'),(smoothing,assembly_path),(semantics,smooth_path)):
        if report['input_sha256'].get(str(path))!=sha256_file(path):raise ValueError('pipeline stage input identity differs')
    if assembly.get('whole_network_human_approved') is not False or semantics.get('whole_line_labels_assigned')!=0:
        raise ValueError('expected unapproved network and local semantic review')
    inputs=[p for p in review.rglob('*') if p.is_file()]+[assembled/'assembly-report.json',assembly_path,
        smoothed/'smoothing-report.json',smooth_path,semantic/'semantic-overlap-report.json',semantic/'needs-semantic-review.geojson',Path(__file__),topology_path]
    before={str(p):sha256_file(p) for p in inputs}
    shutil.copytree(review,output)
    shutil.copyfile(topology_path,output/'topology-validation.json')
    for folder,names in ((assembled,['assembled-candidate-network.geojson','assembly-report.json','network-membership.json','source-and-addition-parts.geojson']),
                         (smoothed,['smoothed-candidate-network.geojson','smoothing-report.json']),
                         (semantic,['needs-semantic-review.geojson','semantic-overlap-report.json'])):
        for name in names:
            if (output/name).exists():raise ValueError('output copy name collision')
            shutil.copyfile(folder/name,output/name)
    project=QgsProject()
    if not project.read(str(output/'contour-reconstruction-review.qgz')):raise ValueError('copied project unreadable')
    root=project.layerTreeRoot();crs=project.crs()
    for layer in project.mapLayers().values():
        if not Path(layer.source().split('|')[0]).resolve().is_relative_to(output):raise ValueError('copied project still points outside packet')
        if layer.source().endswith('retained.geojson'):root.findLayer(layer.id()).setItemVisibilityChecked(False)
    group=root.insertGroup(2,'연속 벡터 단계 비교 · 하나씩 켜기');counts={};vertices=0
    definitions=[('smoothed-candidate-network.geojson','3 · 조립 후 원본 근거 평활화','20,100,190'),
                 ('assembled-candidate-network.geojson','2 · 복원 구간을 합친 연속선','100,100,100'),
                 ('retained.geojson','1 · 숫자 분리 후 원래 관측선','20,140,150'),
                 ('needs-semantic-review.geojson','국소 비등고선 참조 겹침 · 전체선 미판정','240,120,0')]
    for i,(name,title,color) in enumerate(definitions):
        expected=read(output/name)['features'];layer=QgsVectorLayer(str(output/name),title,'ogr')
        if not layer.isValid() or layer.crs()!=crs or layer.featureCount()!=len(expected):raise ValueError('invalid comparison layer')
        for f,e in zip(layer.getFeatures(),expected):
            coordinates=[[p.x(),p.y()] for p in f.geometry().vertices()]
            if coordinates!=e['geometry']['coordinates'] or not f.geometry().isGeosValid():raise ValueError('comparison geometry changed or invalid')
            vertices+=len(coordinates)
            if e['properties'].get('human_approved') is not False:raise ValueError('whole-line approval must not propagate')
        layer.setReadOnly(True);layer.renderer().setSymbol(QgsLineSymbol.createSimple({'line_color':color,'line_width':'.25' if i<3 else '.5'}))
        project.addMapLayer(layer,False);counts[name]=len(expected)
        if i<3:group.addLayer(layer)
        else:root.insertLayer(0,layer).setItemVisibilityChecked(False)
    group.setIsMutuallyExclusive(True,0)
    project.setTitle('등고선 강화 · 관측–연결–평활화 비교 (미검수 후보)')
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    destination=output/'contour-network-review.qgz'
    if not project.write(str(destination)):raise RuntimeError('integrated project save failed')
    check=QgsProject()
    if not check.read(str(destination)):raise ValueError('integrated project reopen failed')
    for layer in check.mapLayers().values():
        if not layer.isValid() or layer.crs()!=crs or not Path(layer.source().split('|')[0]).resolve().is_relative_to(output):raise ValueError('integrated layer not portable')
        if isinstance(layer,QgsVectorLayer) and not layer.readOnly():raise ValueError('integrated vector lost read-only flag')
    if any(sha256_file(Path(p))!=digest for p,digest in before.items()):raise ValueError('original stage or review packet changed')
    write(output/'integrated-project-validation.json',{'schema':'jap-map-integrated-review-project/1','input_sha256':before,
          'qgis_version':Qgis.QGIS_VERSION,'layers':len(check.mapLayers()),'stage_counts':counts,
          'stage_vertices_verified':vertices,'bookmarks':len(check.bookmarkManager().bookmarks()),
          'project_sha256':sha256_file(destination),'whole_network_human_approved':False,'source_modified':False})
    print({'layers':len(check.mapLayers()),'stage_counts':counts,'vertices_verified':vertices})
    check.clear();project.clear()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('review','assembled','smoothed','semantic'):p.add_argument(name,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--topology-validation',required=True,type=Path);a=p.parse_args()
    from qgis.core import QgsApplication
    app=QgsApplication([],False);app.initQgis();build(a.review,a.assembled,a.smoothed,a.semantic,a.output,topology_validation=a.topology_validation)
