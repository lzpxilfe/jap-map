#!/usr/bin/env python3
"""Preserve explicit local connection feedback as separate native geometry."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from histcontour_core.context_gap_preview import resolve_straight_feedback
from histcontour_core.provenance import sha256_file
from histcontour_core.human_feedback import map_to_pixel,validate_pixel_points
from histcontour_core.map_text_detection import inspect_raster
from scripts.prepare_contour_reconstruction_review import read,write


def run(packet,report_path,feedback_path,output):
    from PIL import Image,ImageDraw
    packet,report_path,feedback_path,output=map(lambda p:Path(p).resolve(),(packet,report_path,feedback_path,output))
    if output.exists():raise FileExistsError('feedback output must be new')
    if any(p==output or p in output.parents for p in (packet,report_path.parent,feedback_path.parent)):
        raise ValueError('output must be outside source inputs')
    report=read(report_path);feedback=read(feedback_path)
    revisions=resolve_straight_feedback(report,feedback,sha256_file(report_path))
    drawing=read(packet/'drawing-report.json')
    tiles={t['tile_id']:t for t in drawing['tiles']}
    inputs=[report_path,feedback_path,packet/'drawing-report.json',Path(__file__),ROOT/'histcontour_core/context_gap_preview.py']
    if report['input_and_code_sha256'].get(str(packet/'drawing-report.json'))!=sha256_file(packet/'drawing-report.json'):
        raise ValueError('source drawing report changed')
    features=[];images=[]
    for row in revisions:
        tile=tiles[row['tile_id']];source=(packet/tile['raster_path']).resolve()
        if packet not in source.parents:raise ValueError('source escapes packet')
        inspect_raster(source,tile);inputs.append(source)
        validate_pixel_points(row['points'],tile)
        west,south,east,north=tile['bounds'];width,height=tile['pixel_bounds'][2:]
        coords=[[west+(x+.5)*(east-west)/width,north-(y+.5)*(north-south)/height] for x,y in row['points']]
        if max(__import__('math').dist(a,b) for a,b in zip(row['points'],map_to_pixel(tile,coords)))>1e-5:
            raise ValueError('native coordinate round trip failed')
        features.append({'type':'Feature','properties':{k:v for k,v in row.items() if k not in ('points','previous_points')},
                         'geometry':{'type':'LineString','coordinates':coords}})
        case=next(c for c in report['cases'] if c['case_id']==row['case_id'])
        region_id=case['region_id']
        # Show a local source crop around both ends with generous context.
        xs,ys=zip(*row['points']);box=[max(0,int(min(xs))-40),max(0,int(min(ys))-35),
                                    min(width,int(max(xs))+41),min(height,int(max(ys))+36)]
        with Image.open(source) as image:crop=image.crop(box).convert('RGB')
        panels=[]
        for label,points in [('Source',[]),('Previous curve',row['previous_points']),('User: straight acceptable',row['points'])]:
            panel=crop.resize((crop.width*4,crop.height*4));draw=ImageDraw.Draw(panel)
            if points:draw.line([((x-box[0])*4,(y-box[1])*4) for x,y in points],fill=(0,170,80),width=3)
            draw.text((4,4),label,fill=(180,0,0));panels.append(panel)
        board=Image.new('RGB',(panels[0].width*3,panels[0].height),'white')
        for i,panel in enumerate(panels):board.paste(panel,(i*panel.width,0))
        images.append((region_id+'-straight-feedback.png',board))
    before={str(p):sha256_file(p) for p in inputs}
    output.mkdir(parents=True)
    for name,image in images:image.save(output/name)
    write(output/'reviewed-local-connections.geojson',{'type':'FeatureCollection',
          'crs':{'type':'name','properties':{'name':tiles[revisions[0]['tile_id']]['crs_authid']}},'features':features})
    if any(sha256_file(Path(p))!=digest for p,digest in before.items()):raise ValueError('feedback input changed')
    write(output/'feedback-application.json',{'schema':'jap-map-context-feedback-application/1','revisions':revisions,
          'input_sha256':before,'source_pixels_modified':False,'whole_network_modified':False,
          'semantic_approvals_added':0,'connection_approvals_recorded':len(revisions)})
    return revisions


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','report','feedback'):p.add_argument(name,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    run(a.packet,a.report,a.feedback,a.output)
