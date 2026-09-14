#!/usr/bin/env python3
"""Prioritize new network lines touching explicit local non-contour references."""
import argparse
import copy
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from histcontour_core.human_feedback import map_to_pixel
from histcontour_core.provenance import sha256_file
from histcontour_core.semantic_reference_overlap import reference_overlap
from scripts.prepare_contour_reconstruction_review import read,write,validate_regions


def run(packet,manifest_path,network_path,references_path,output):
    packet,manifest_path,network_path,references_path,output=map(lambda p:Path(p).resolve(),(packet,manifest_path,network_path,references_path,output))
    if output.exists():raise FileExistsError('semantic review output must be new')
    if any(p==output or p in output.parents for p in (packet,network_path.parent,references_path.parent)):raise ValueError('output inside source snapshot')
    manifest=read(manifest_path);drawing=packet/'drawing-report.json';tiles=validate_regions(manifest,read(drawing))
    if sha256_file(drawing)!=manifest['drawing_report_sha256']:raise ValueError('source drawing changed')
    network,references=read(network_path),read(references_path)
    crs={'type':'name','properties':{'name':next(iter(tiles.values()))['crs_authid']}}
    if network['crs']!=crs or references['crs']!=crs:raise ValueError('semantic reference CRS differs')
    inputs=[drawing,manifest_path,network_path,references_path,Path(__file__),ROOT/'histcontour_core/semantic_reference_overlap.py']
    before={str(p):sha256_file(p) for p in inputs};refs=[]
    for f in references['features']:
        p=f['properties'];tid=p['tile_id']
        if (p.get('semantic_decision')!='non_contour' or p.get('review_origin')!='explicit_user_chat'
                or p.get('human_geometry_accepted') is not True or p.get('training_eligible') is not False
                or p['source_raster_sha256']!=tiles[tid]['source_raster_sha256']):raise ValueError('reference lacks explicit local non-contour feedback')
        refs.append((p['proposal_id'],tid,map_to_pixel(tiles[tid],f['geometry']['coordinates'])))
    selected=[];evidence=[]
    for f in network['features']:
        tid=f['properties']['tile_id'];points=map_to_pixel(tiles[tid],f['geometry']['coordinates']);matches=[]
        for rid,rtid,reference in refs:
            if tid!=rtid:continue
            overlap=reference_overlap(points,reference)
            if overlap['aligned_overlap_length_px']>=3:matches.append({'reference_id':rid,**overlap})
        if matches:
            item=copy.deepcopy(f);item['properties'].update(semantic_review_priority='overlaps_explicit_non_contour_reference',
                reference_ids=[m['reference_id'] for m in matches],whole_line_semantics_approved=False,
                human_approved=False,training_eligible=False)
            selected.append(item);evidence.append({'line_id':f['properties']['line_id'],'tile_id':tid,'local_matches':matches})
    if any(sha256_file(Path(p))!=h for p,h in before.items()):raise ValueError('semantic reference inputs changed')
    output.mkdir(parents=True);write(output/'needs-semantic-review.geojson',{'type':'FeatureCollection','crs':crs,'features':selected})
    write(output/'semantic-overlap-report.json',{'schema':'jap-map-semantic-overlap-review/1','evidence':evidence,
        'input_sha256':before,'whole_line_labels_assigned':0,'model_fitted':False,'geometry_modified':False,
        'limitations':['A sparse non-contour reference does not classify the rest of an assembled line.',
                       'Parallel proximity is review evidence, not exact semantic pixel truth.']})
    print({'lines_requiring_local_semantic_review':len(selected),'whole_line_labels_assigned':0})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','manifest','network','references'):p.add_argument(name,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args();run(a.packet,a.manifest,a.network,a.references,a.output)
