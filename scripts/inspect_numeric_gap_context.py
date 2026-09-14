#!/usr/bin/env python3
"""Source-coupled gap matching diagnostic; no connections are applied."""
import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from histcontour_core.gap_endpoint_context import terminal_context, endpoint_banks
from histcontour_core.contextual_gap_ranking import rank_gap_context
from histcontour_core.context_gap_preview import build_context_gap_previews
from histcontour_core.regional_gap_matching import match_endpoint_banks, _intersect
from histcontour_core.human_feedback import map_to_pixel
from histcontour_core.map_text_detection import inspect_raster
from histcontour_core.provenance import sha256_file
from scripts.prepare_contour_reconstruction_review import read, write, validate_regions


def run(packet, manifest_path, ownership_path, vectors_path, output):
    import numpy as np
    from PIL import Image, ImageDraw
    packet, manifest_path, ownership_path, vectors_path, output = map(lambda p: Path(p).resolve(),
        (packet, manifest_path, ownership_path, vectors_path, output))
    if output.exists(): raise FileExistsError('diagnostic output must be new')
    if any(p == output or p in output.parents for p in (packet, ownership_path.parent, vectors_path.parent)):
        raise ValueError('output must be outside source snapshots')
    report_path = packet/'drawing-report.json'
    manifest, report, ownership, vectors = map(read, (manifest_path, report_path, ownership_path, vectors_path))
    tiles = validate_regions(manifest, report)
    if sha256_file(report_path) != manifest['drawing_report_sha256']:
        raise ValueError('source packet changed')
    if ownership.get('human_approvals') != 0 or ownership.get('holdout_used') is not False:
        raise ValueError('expected development-only numeric evidence')
    if ownership.get('schema') != 'jap-map-numeric-separation-run/1':
        raise ValueError('unsupported numeric evidence schema')
    regions = {r['region_id']: r for r in manifest['regions']}
    paths, images, input_paths = {}, {}, [report_path, manifest_path, ownership_path, vectors_path]
    if len(vectors['features']) > 150000: raise ValueError('vector budget exceeded')
    for tid in sorted({r['tile_id'] for r in ownership['regions']}):
        tile = tiles[tid]
        source = (packet/tile['raster_path']).resolve()
        if packet not in source.parents: raise ValueError('source path escapes packet')
        inspect_raster(source, tile)
        with Image.open(source) as image: images[tid] = image.convert('RGB')
        input_paths.append(source); paths[tid] = []
    expected_crs = {'type':'name', 'properties':{'name':next(iter(tiles.values()))['crs_authid']}}
    if vectors.get('crs') != expected_crs: raise ValueError('vector CRS differs')
    for f in vectors['features']:
        p = f['properties']; tid = p['tile_id']
        if tid not in paths: continue
        if (p.get('inferred_gap') is not False or p.get('source_raster_sha256') != tiles[tid]['source_raster_sha256']
                or f['geometry']['type'] != 'LineString' or 'numeric_partition_role' in p):
            raise ValueError('only source-matched unsplit observed paths may supply endpoints')
        paths[tid].append({'source_path_id':p['path_id'], 'points':map_to_pixel(tiles[tid],f['geometry']['coordinates'])})
    code_paths = [Path(__file__), ROOT/'histcontour_core/gap_endpoint_context.py', ROOT/'histcontour_core/regional_gap_matching.py',
                  ROOT/'histcontour_core/contextual_gap_ranking.py',
                  ROOT/'histcontour_core/context_gap_preview.py', ROOT/'histcontour_core/manual_gap_bridge.py',
                  ROOT/'histcontour_core/gap_refinement.py',
                  ROOT/'histcontour_core/human_feedback.py', ROOT/'histcontour_core/map_text_detection.py',
                  ROOT/'scripts/prepare_contour_reconstruction_review.py']
    before = {str(p):sha256_file(p) for p in input_paths+code_paths}
    output.mkdir(parents=True)
    cases, seen = [], set()
    for row in ownership['regions']:
        rid, tid = row['region_id'], row['tile_id']
        if rid in seen or rid not in regions or any(row[k] != regions[rid][k] for k in ('tile_id','box')):
            raise ValueError('numeric region identity differs')
        seen.add(rid)
        if row['source_raster_sha256'] != tiles[tid]['source_raster_sha256']: raise ValueError('numeric source changed')
        for h in row['hypotheses']:
            if not h['glyph_candidate_component_ids']: continue
            quad = np.asarray(h['reference_quad_crop_pixel_centers'])+row['box'][:2]
            box = [float(v) for v in (*quad.min(axis=0), *quad.max(axis=0))]
            context = terminal_context(paths[tid], box, image_shape=(images[tid].height,images[tid].width))
            alternatives = []
            for angle in (0,45,90,135):
                theta = math.radians(angle); axis = [math.cos(theta),math.sin(theta)]
                a,b = endpoint_banks(context['endpoints'],box,axis)
                if max(len(a),len(b)) > 64:
                    alternatives.append({'angle_degrees':angle,'status':'bank_cap_abstained'}); continue
                match = match_endpoint_banks(a,b,ordering_axis=[-axis[1],axis[0]])
                # Check the display chords against other observed strokes. These
                # are still not routed curves and cannot be promoted to results.
                for hypothesis in (match['best'],match['runner_up']):
                    if hypothesis is None: continue
                    collisions = set()
                    for m in hypothesis['matches']:
                        for path in paths[tid]:
                            points = path['points']
                            for c,d in zip(points,points[1:]):
                                if any(math.dist(p,q)<1e-5 for p in (m['start'],m['end']) for q in (c,d)): continue
                                if _intersect(m['start'],m['end'],c,d): collisions.add(path['source_path_id']); break
                    hypothesis['observed_chord_collision_path_ids'] = sorted(collisions)
                    if collisions:
                        hypothesis['eligible_for_routing'] = False
                        hypothesis['ambiguous'] = True
                        hypothesis['ambiguity_reasons'].append('chord_crosses_observed_linework')
                match['ambiguous'] = match['best']['ambiguous']
                match['ambiguity_reasons'] = list(match['best']['ambiguity_reasons'])
                alternatives.append({'angle_degrees':angle,'banks':[a,b],'matching':match})
            pool=[m for a in alternatives if 'matching' in a
                  for hypothesis in (a['matching']['best'],a['matching']['runner_up']) if hypothesis
                  for m in hypothesis['matches']]
            ranking=rank_gap_context(pool,paths[tid],box)
            curves=build_context_gap_previews(ranking,context['endpoints'],paths[tid])
            identity = f'{rid}-{h["hypothesis_id"]}'
            l,t,r,b = row['box']; scale=3
            canvas=Image.new('RGB',((r-l)*scale*2,(b-t)*scale*2),'white')
            for i,alternative in enumerate(alternatives):
                panel=images[tid].crop(row['box']).resize(((r-l)*scale,(b-t)*scale))
                draw=ImageDraw.Draw(panel)
                def pt(p):return ((p[0]-l)*scale,(p[1]-t)*scale)
                draw.rectangle([pt(box[:2]),pt(box[2:])],outline='orange',width=2)
                match=alternative.get('matching')
                if match:
                    for endpoint in context['endpoints']:
                        x,y=pt(endpoint['point']);draw.ellipse((x-3,y-3,x+3,y+3),outline='blue',width=2)
                    for m in match['best']['matches']:
                        draw.line([pt(m['start']),pt(m['end'])],fill='red',width=2)
                    draw.text((5,5),f'{alternative["angle_degrees"]} deg | {len(match["best"]["matches"])} pairs | UNAPPROVED',fill='red')
                canvas.paste(panel,((i%2)*(r-l)*scale,(i//2)*(b-t)*scale))
            canvas.save(output/(identity+'.png'))
            for curve in curves['previews']:
                if not curve['points']:continue
                panel=images[tid].crop(row['box']).resize(((r-l)*scale,(b-t)*scale))
                draw=ImageDraw.Draw(panel)
                draw.line([((p[0]-l)*scale,(p[1]-t)*scale) for p in curve['points']],
                          fill='green' if curve['geometry_checks_passed'] else 'red',width=3)
                draw.text((5,5),curve['candidate_id']+' | UNAPPROVED',fill='red')
                curve['preview_image']=identity+'-'+curve['candidate_id']+'.png'
                panel.save(output/curve['preview_image'])
            cases.append({'case_id':identity,'region_id':rid,'tile_id':tid,'label_context_box_pixel_centers':box,
                'numeric_value_not_assigned':True,'terminal_context':context,'orientation_alternatives':alternatives,
                'contextual_ranking':ranking,
                'curve_previews':curves,
                'global_orientation_selected':False,'eligible_for_automatic_connection':False,
                'preview':identity+'.png','human_approved':False,'training_eligible':False})
            print(rid,[(a['angle_degrees'],len(a.get('matching',{}).get('best',{}).get('matches',[]))) for a in alternatives],flush=True)
    if any(sha256_file(Path(p))!=digest for p,digest in before.items()): raise ValueError('source or code changed during inspection')
    result={'schema':'jap-map-numeric-gap-context/1','cases':cases,'input_and_code_sha256':before,
            'human_approvals':0,'connections_applied':0,'routing_performed':False,'holdout_used':False,
            'limitations':['Boxes are search context, not true occlusion boundaries.',
                'Orientation alternatives are not globally selected; endpoint reuse across alternatives is not an approved network.',
                'Red chords are matching diagnostics, not smoothed contour geometry.']}
    write(output/'gap-context-report.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','manifest','ownership','vectors'):parser.add_argument(name,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    run(args.packet,args.manifest,args.ownership,args.vectors,args.output)
