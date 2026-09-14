#!/usr/bin/env python3
"""Fit source-supported joined routes while locking approved and inferred spans."""
import argparse
import copy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from histcontour_core.human_feedback import map_to_pixel,geometry_digest
from histcontour_core.provenance import sha256_file
from histcontour_core.map_text_detection import inspect_raster
from scripts.prepare_contour_reconstruction_review import read,write,validate_regions
from scripts.reconstruct_observed_contours import smooth_path


def source_locks(membership,parts,expected):
    reconstructed=[];locks=[]
    for member in membership['members']:
        part=parts[member['part_id']];ordered=part['geometry']['coordinates']
        if member['reversed']:ordered=list(reversed(ordered))
        start=len(reconstructed)
        if reconstructed and reconstructed[-1]==ordered[0]:
            start-=1;ordered=ordered[1:]
        reconstructed+=ordered
        props=part['properties']
        if props['part_kind']=='approved_connection' or props.get('inferred_gap') is True:
            locks.append((start,len(reconstructed)-1))
    if reconstructed!=expected:raise ValueError('membership does not reconstruct exact source geometry')
    return locks


def run(packet,manifest_path,assembled,output):
    import numpy as np
    from PIL import Image
    packet,manifest_path,assembled,output=map(lambda p:Path(p).resolve(),(packet,manifest_path,assembled,output))
    if output.exists():raise FileExistsError('smoothing output must be new')
    if any(p==output or p in output.parents for p in (packet,assembled)):raise ValueError('output inside source snapshot')
    drawing_path=packet/'drawing-report.json';manifest=read(manifest_path);tiles=validate_regions(manifest,read(drawing_path))
    if sha256_file(drawing_path)!=manifest['drawing_report_sha256']:raise ValueError('source drawing changed')
    network_path=assembled/'assembled-candidate-network.geojson';parts_path=assembled/'source-and-addition-parts.geojson'
    members_path=assembled/'network-membership.json';network=read(network_path)
    parts={f['properties']['part_id']:f for f in read(parts_path)['features']}
    members={m['line_id']:m for m in read(members_path)}
    sources={tid:(packet/tile['raster_path']).resolve() for tid,tile in tiles.items()}
    for tid,path in sources.items():
        if packet not in path.parents:raise ValueError('source raster escapes packet')
        inspect_raster(path,tiles[tid])
    inputs=[drawing_path,manifest_path,network_path,parts_path,members_path,*sources.values(),Path(__file__),
            ROOT/'scripts/reconstruct_observed_contours.py',ROOT/'histcontour_core/observed_curve.py']
    before={str(p):sha256_file(p) for p in inputs};result=copy.deepcopy(network);cache={};audits=[]
    for feature in result['features']:
        props=feature['properties'];lid=props['line_id'];tid=props['tile_id'];original=feature['geometry']['coordinates']
        locks=source_locks(members[lid],parts,original)
        if not props['automatic_connection_count'] and not props['approved_connection_count']:continue
        tile=tiles[tid];w,h=tile['pixel_bounds'][2:];west,south,east,north=tile['bounds']
        if tid not in cache:
            with Image.open(sources[tid]) as image:gray=np.asarray(image.convert('L'))
            skeleton=np.zeros((h,w),bool)
            for part in parts.values():
                if part['properties']['tile_id']!=tid:continue
                p=np.asarray(map_to_pixel(tile,part['geometry']['coordinates']))
                for a,b in zip(p,p[1:]):
                    count=max(2,int(np.ceil(np.linalg.norm(b-a)*2))+1)
                    xy=np.rint(np.linspace(a,b,count)).astype(int)
                    skeleton[np.clip(xy[:,1],0,h-1),np.clip(xy[:,0],0,w-1)]=True
            cache[tid]=(gray,skeleton)
        gray,skeleton=cache[tid]
        points=np.clip(np.asarray(map_to_pixel(tile,original)),[0,0],[w-1,h-1]).tolist()
        boxes=[r['box'] for r in manifest['regions'] if r['tile_id']==tid]
        fitted,audit=smooth_path(gray,skeleton,points,boxes,locked_source_spans=locks)
        audits.append({'line_id':lid,'tile_id':tid,'locked_original_index_spans':locks,**audit})
        if not audit['adopted_in_draft']:continue
        coordinates=[[west+(x+.5)*(east-west)/w,north-(y+.5)*(north-south)/h] for x,y in fitted]
        for index,position in enumerate(audit['original_vertex_indices']):
            if np.linalg.norm(np.asarray(fitted[position])-points[index])<1e-8:coordinates[position]=original[index]
        for start,end in locks:
            positions=audit['original_vertex_indices'][start:end+1]
            if [coordinates[i] for i in positions]!=original[start:end+1]:raise ValueError('locked geometry moved')
        feature['geometry']['coordinates']=coordinates
        props.update(geometry_sha256=geometry_digest(feature['geometry']),
                     length_pixels=sum(float(np.linalg.norm(np.asarray(b)-a)) for a,b in zip(fitted,fitted[1:])),
                     source_supported_postassembly_fit=True)
    if any(sha256_file(Path(p))!=digest for p,digest in before.items()):raise ValueError('smoothing source changed')
    output.mkdir(parents=True);write(output/'smoothed-candidate-network.geojson',result)
    report={'schema':'jap-map-postassembly-smoothing/1','input_sha256':before,'audits':audits,
            'attempted_lines':len(audits),'adopted_lines':sum(a['adopted_in_draft'] for a in audits),
            'approved_and_inferred_spans_locked':True,'human_approvals_added':0,'source_modified':False}
    write(output/'smoothing-report.json',report);print({k:v for k,v in report.items() if k not in ('audits','input_sha256')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','manifest','assembled'):p.add_argument(name,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args();run(a.packet,a.manifest,a.assembled,a.output)
