#!/usr/bin/env python3
"""Assemble an explicit copy of audited recovery candidates at exact endpoints."""
import argparse
import copy
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from histcontour_core.provenance import sha256_file
from histcontour_core.contour_network import merge_parts
from histcontour_core.human_feedback import map_to_pixel
from histcontour_core.recovery_export import export_recovery_additions
from scripts.prepare_contour_reconstruction_review import read,write,validate_regions


def build(packet,manifest_path,review,output):
    import numpy as np
    from scipy.spatial import cKDTree
    packet,manifest_path,review,output=map(lambda p:Path(p).resolve(),(packet,manifest_path,review,output))
    if output.exists():raise FileExistsError('assembled network output must be new')
    if any(p==output or p in output.parents for p in (packet,review)):raise ValueError('output inside source snapshot')
    drawing=packet/'drawing-report.json';manifest=read(manifest_path);tiles=validate_regions(manifest,read(drawing))
    validation=read(review/'project-validation.json');hashes=validation['input_sha256']
    if sha256_file(drawing)!=manifest['drawing_report_sha256']:raise ValueError('drawing snapshot differs')
    checked=[drawing,manifest_path,review/'project-validation.json',Path(__file__),ROOT/'histcontour_core/contour_network.py',ROOT/'histcontour_core/recovery_export.py']
    def verified_copy(name):
        original=[Path(p) for p in hashes if Path(p).name==name]
        if len(original)!=1 or sha256_file(review/name)!=hashes[str(original[0])]:raise ValueError('project source copy differs: '+name)
        checked.append(review/name);return read(review/name)
    retained=verified_copy('retained.geojson');local=verified_copy('reviewed-local-connections.geojson')
    approved=verified_copy('approved-connections-unchanged.geojson')
    reports=[Path(p) for p in hashes if Path(p).name=='short-recovery-report.json']
    if len(reports)!=1 or sha256_file(reports[0])!=hashes[str(reports[0])]:raise ValueError('recovery report changed')
    checked.append(reports[0]);checked.append(review/'short-recovery-additions.geojson')
    additions=read(review/'short-recovery-additions.geojson')
    if additions!=export_recovery_additions(read(reports[0]),tiles):raise ValueError('recovery additions differ from audited export')
    before={str(p):sha256_file(p) for p in checked}
    parts=[];endpoints={tid:[] for tid in tiles}
    for f in retained['features']:
        item=copy.deepcopy(f);p=item['properties'];tid=p['tile_id']
        if p['source_raster_sha256']!=tiles[tid]['source_raster_sha256']:raise ValueError('source feature identity differs')
        p.update(part_id=tid+':source:'+p['path_id'],part_kind='retained_source')
        parts.append(item);coords=item['geometry']['coordinates'];endpoints[tid]+=map_to_pixel(tiles[tid],[coords[0],coords[-1]])
    trees={tid:cKDTree(np.asarray(points).reshape((-1,2))) for tid,points in endpoints.items()}
    baseline=merge_parts(parts,tiles);unattached=[];accepted=[]
    for collection,kind in ((additions,'automatic_connection'),(local,'approved_connection')):
        for i,f in enumerate(collection['features']):
            item=copy.deepcopy(f);p=item['properties'];tid=p['tile_id'];coords=item['geometry']['coordinates']
            points=map_to_pixel(tiles[tid],[coords[0],coords[-1]])
            distance,_=trees[tid].query(points)
            identity=p.get('recovery_id') or p['case_id']+':'+p['candidate_id']
            if any(float(d)>1e-5 for d in distance):
                unattached.append({'identity':identity,'tile_id':tid,'reason':'no_exact_retained_endpoint',
                                   'endpoint_distances_pixels':[float(d) for d in distance]});continue
            p.update(part_id=tid+':'+kind+':'+identity,part_kind=kind)
            parts.append(item);accepted.append({'part_id':p['part_id'],'kind':kind})
    assembled=merge_parts(parts,tiles)
    by_id={p['properties']['part_id']:p for p in parts};used=[]
    for feature,membership in zip(assembled['features'],assembled['memberships']):
        rebuilt=[]
        for member in membership['members']:
            used.append(member['part_id']);points=by_id[member['part_id']]['geometry']['coordinates']
            if member['reversed']:points=list(reversed(points))
            if rebuilt and rebuilt[-1]==points[0]:points=points[1:]
            rebuilt+=points
        if rebuilt!=feature['geometry']['coordinates']:raise ValueError('assembled vertices differ from source parts')
    if len(used)!=len(set(used)) or set(used)!=set(by_id):raise ValueError('parts missing or used twice')
    output.mkdir(parents=True);crs=retained['crs']
    write(output/'assembled-candidate-network.geojson',{'type':'FeatureCollection','crs':crs,'features':assembled['features']})
    write(output/'source-and-addition-parts.geojson',{'type':'FeatureCollection','crs':crs,'features':parts})
    write(output/'network-membership.json',assembled['memberships'])
    shutil.copyfile(review/'approved-connections-unchanged.geojson',output/'original-approved-connections.geojson')
    shutil.copyfile(review/'reviewed-local-connections.geojson',output/'reviewed-local-connections.geojson')
    if any(sha256_file(Path(p))!=digest for p,digest in before.items()):raise ValueError('source changed during assembly')
    summary={'schema':'jap-map-reconstruction-network/1','input_sha256':before,'accepted_parts':accepted,'unattached':unattached,
             'input_source_parts':len(retained['features']),'assembled_lines':assembled['line_count'],'all_parts_used_once':True,
             'all_original_vertices_preserved':True,'baseline_open_endpoints':baseline['open_endpoint_nodes'],
             'assembled_open_endpoints':assembled['open_endpoint_nodes'],'tile_audits':assembled['tiles'],
             'human_approvals_added':0,'whole_network_human_approved':False,'original_source_modified':False,
             'original_approved_23_used_for_new_endpoint_snapping':False}
    write(output/'assembly-report.json',summary);print({k:v for k,v in summary.items() if k not in ('input_sha256','tile_audits','accepted_parts','unattached')})
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','manifest','review'):p.add_argument(name,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args();build(a.packet,a.manifest,a.review,a.output)
