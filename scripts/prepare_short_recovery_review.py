#!/usr/bin/env python3
"""Freeze two-axis human review cards for explicit short recovery candidates."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from histcontour_core.provenance import sha256_file
from histcontour_core.map_text_detection import inspect_raster
from histcontour_core.human_feedback import geometry_digest,validate_pixel_points
from scripts.prepare_contour_reconstruction_review import read,write


def build(packet,report_path,output,selections):
    from PIL import Image,ImageDraw
    packet,report_path,output=map(lambda p:Path(p).resolve(),(packet,report_path,output))
    if output.exists():raise FileExistsError('review output must be new')
    if any(p==output or p in output.parents for p in (packet,report_path.parent)):raise ValueError('review output inside source snapshot')
    if not 1<=len(selections)<=3 or len(set(selections))!=len(selections):raise ValueError('select one to three distinct candidates')
    report=read(report_path);drawing_path=packet/'drawing-report.json';drawing=read(drawing_path)
    if report.get('holdout_used') is not False or report.get('human_approvals')!=0 or report.get('numeric_candidate_masks_checked') is not True:
        raise ValueError('expected numerically audited development recovery')
    if report['input_sha256'].get(str(drawing_path))!=sha256_file(drawing_path):raise ValueError('source drawing report differs')
    tiles={t['tile_id']:t for t in drawing['tiles']};cards=[];images=[];inputs=[drawing_path,report_path,Path(__file__)]
    available={c['candidate_id']:(r,c) for r in report['regions'] for c in r['weak_sensitivity']['candidates']}
    for index,identity in enumerate(selections,1):
        if identity not in available:raise ValueError('unknown recovery candidate')
        region,candidate=available[identity]
        if (candidate['status']!='observed' or candidate['topology_audit']['status']!='passed'
                or candidate['numeric_ink_audit']['status']!='no_numeric_candidate_overlap'
                or candidate['edge_partition']['requires_topology_resolution']):raise ValueError('candidate has unresolved automatic checks')
        tile=tiles[region['tile_id']];source=(packet/tile['raster_path']).resolve()
        if packet not in source.parents or report['input_sha256'].get(str(source))!=sha256_file(source):raise ValueError('candidate source differs')
        inspect_raster(source,tile);inputs.append(source);points=candidate['points'];validate_pixel_points(points,tile)
        xs,ys=zip(*points);width,height=tile['pixel_bounds'][2:]
        box=[max(0,int(min(xs))-42),max(0,int(min(ys))-42),min(width,int(max(xs))+43),min(height,int(max(ys))+43)]
        with Image.open(source) as image:crop=image.crop(box).convert('RGB')
        scale=4;panel=crop.resize((crop.width*scale,crop.height*scale),Image.Resampling.NEAREST)
        board=Image.new('RGB',(panel.width*2,panel.height+32),'white');board.paste(panel,(0,32));board.paste(panel,(panel.width,32))
        draw=ImageDraw.Draw(board);review_id=f'F{index:03d}'
        draw.text((8,8),review_id+' | Source',fill='black');draw.text((panel.width+8,8),'Recovery hypothesis | UNAPPROVED',fill='black')
        display=[(panel.width+(x-box[0])*scale,32+(y-box[1])*scale) for x,y in points]
        draw.line(display,fill=(0,160,60),width=3)
        for x,y in (display[0],display[-1]):draw.ellipse((x-4,y-4,x+4,y+4),outline=(0,100,220),width=2)
        geometry={'type':'LineString','coordinates':points}
        cards.append({'review_id':review_id,'candidate_id':identity,'region_id':region['region_id'],'tile_id':region['tile_id'],
            'source_path_ids':candidate['source_path_ids'],'source_raster_sha256':tile['source_raster_sha256'],
            'crop_box':box,'pixel_geometry':geometry,'pixel_geometry_sha256':geometry_digest(geometry),
            'source_recovery_report_sha256':sha256_file(report_path),'human_approved':False,'training_eligible':False,
            'semantic_question':'등고선으로 보이나요? 아니면 하천·도로·기호이거나 판단이 어려운가요?',
            'geometry_question':'초록색 연결 대상과 경로가 맞나요? 더 곧게 또는 더 굽혀야 하나요?',
            'decision_status':'awaiting_human','image':review_id+'.png'})
        images.append((review_id+'.png',board))
    before={str(p):sha256_file(p) for p in inputs};output.mkdir(parents=True)
    for name,image in images:image.save(output/name)
    if any(sha256_file(Path(p))!=digest for p,digest in before.items()):raise ValueError('source changed during review rendering')
    write(output/'questions.json',{'schema':'jap-map-short-recovery-human-review/1','cards':cards,'input_sha256':before,
          'human_approvals':0,'training_promotion':False,'selection_policy':'explicit development cases, not independent evaluation'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','report'):p.add_argument(name,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--candidates',required=True,nargs='+')
    a=p.parse_args();build(a.packet,a.report,a.output,a.candidates)
