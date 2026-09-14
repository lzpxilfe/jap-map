#!/usr/bin/env python3
"""Inspect short observed-ink gaps on fixed faint development regions."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from histcontour_core.provenance import sha256_file
from histcontour_core.human_feedback import map_to_pixel
from histcontour_core.map_text_detection import inspect_raster
from histcontour_core.ink import ink_centerline_candidates
from histcontour_core.gap_endpoint_context import terminal_context
from histcontour_core.short_ink_recovery import recover_short_ink,contrast_guarded_evidence,audit_short_recovery_topology,audit_recovery_numeric_ink
from histcontour_core.directional_trace import DirectionalTraceConfig
from histcontour_core.recovery_edge_partition import partition_recovery_edges
from histcontour_core.fragment_chain_recovery import recover_via_source_fragment
from histcontour_core.short_ink_recovery import recover_short_ink_batched
from scripts.prepare_contour_reconstruction_review import read,write,validate_regions


def run(packet,manifest_path,vectors_path,output,*,all_regions=False,numeric_separation=None,complete_batches=False):
    import numpy as np
    from PIL import Image,ImageDraw
    packet,manifest_path,vectors_path,output=map(lambda p:Path(p).resolve(),(packet,manifest_path,vectors_path,output))
    if output.exists():raise FileExistsError('output must be new')
    if any(p==output or p in output.parents for p in (packet,vectors_path.parent)):raise ValueError('output inside source snapshot')
    manifest=read(manifest_path);drawing=read(packet/'drawing-report.json');tiles=validate_regions(manifest,drawing)
    if sha256_file(packet/'drawing-report.json')!=manifest['drawing_report_sha256']:raise ValueError('source report changed')
    vectors=read(vectors_path)
    if vectors['crs']['properties']['name']!=next(iter(tiles.values()))['crs_authid']:raise ValueError('CRS differs')
    if type(all_regions) is not bool:raise ValueError('all_regions must be explicit boolean')
    regions=[r for r in manifest['regions'] if all_regions or r['region_id'] in ('R009','R010','R011','R012')]
    source_paths=[packet/'drawing-report.json',manifest_path,vectors_path,Path(__file__),
                  ROOT/'histcontour_core/short_ink_recovery.py',ROOT/'histcontour_core/gap_endpoint_context.py',
                  ROOT/'histcontour_core/directional_trace.py',ROOT/'histcontour_core/ink.py',ROOT/'histcontour_core/observed_linework.py',
                  ROOT/'histcontour_core/recovery_edge_partition.py',ROOT/'histcontour_core/regional_gap_matching.py',
                  ROOT/'histcontour_core/fragment_chain_recovery.py',ROOT/'histcontour_core/numeric_line_partition.py']
    numeric_masks={}
    if numeric_separation is not None:
        import zipfile
        numeric_separation=Path(numeric_separation).resolve()
        if output==numeric_separation or numeric_separation in output.parents:raise ValueError('output inside numeric snapshot')
        numeric_report_path=numeric_separation/'numeric-separation-report.json';numeric_report=read(numeric_report_path)
        if (numeric_report.get('schema')!='jap-map-numeric-separation-run/1' or numeric_report.get('holdout_used') is not False
                or numeric_report.get('human_approvals')!=0):raise ValueError('invalid numeric snapshot')
        source_paths.append(numeric_report_path)
        for row in numeric_report['merged_masks']:
            tid=row['tile_id'];path=(numeric_separation/row['path']).resolve()
            if (tid not in tiles or tid in numeric_masks or numeric_separation not in path.parents
                    or row['source_raster_sha256']!=tiles[tid]['source_raster_sha256'] or sha256_file(path)!=row['sha256']):
                raise ValueError('numeric mask source identity differs')
            width,height=tiles[tid]['pixel_bounds'][2:]
            with zipfile.ZipFile(path) as archive:
                if sum(i.file_size for i in archive.infolist())>width*height*8+8192:raise ValueError('numeric mask archive exceeds grid budget')
            with np.load(path,allow_pickle=False) as arrays:
                mask=arrays['glyph'].copy()
                if mask.shape!=(height,width) or mask.dtype.kind!='b':raise ValueError('numeric mask grid differs')
                if np.any(mask & arrays['protected']):raise ValueError('numeric mask overrides protected ink')
            numeric_masks[tid]=mask;source_paths.append(path)
        if any(r['tile_id'] not in numeric_masks for r in regions):raise ValueError('missing numeric mask for selected tile')
    sources={tid:(packet/tiles[tid]['raster_path']).resolve() for tid in {r['tile_id'] for r in regions}}
    for tid,source in sources.items():
        if packet not in source.parents:raise ValueError('source escapes packet')
        inspect_raster(source,tiles[tid]);source_paths.append(source)
    before={str(p):sha256_file(p) for p in source_paths};output.mkdir(parents=True)
    results=[];cache={}
    for region in regions:
        tid=region['tile_id'];tile=tiles[tid]
        if tid not in cache:
            with Image.open(sources[tid]) as source:image=source.convert('L')
            evidence=ink_centerline_candidates(np.asarray(image),tile_origin=tuple(tile['pixel_bounds'][:2]))
            paths=[]
            for f in vectors['features']:
                p=f['properties']
                if p['tile_id']!=tid:continue
                if p.get('inferred_gap') is not False or p['source_raster_sha256']!=tile['source_raster_sha256']:
                    raise ValueError('expected original observed paths')
                paths.append({'source_path_id':p['path_id'],'points':map_to_pixel(tile,f['geometry']['coordinates'])})
            guarded,guard_audit=contrast_guarded_evidence(np.asarray(image),evidence)
            cache[tid]=(image,evidence,paths,guarded,guard_audit)
        image,evidence,paths,guarded,guard_audit=cache[tid]
        context=terminal_context(paths,region['box'],image_shape=(image.height,image.width),exclude_context_interior=False)
        l,t,r,b=region['box']
        endpoints=[e for e in context['endpoints'] if l+4<=e['point'][0]<r-4 and t+4<=e['point'][1]<b-4]
        recover=recover_short_ink_batched if complete_batches else recover_short_ink
        result=recover(evidence,endpoints)
        sensitivity=recover(guarded,endpoints,
            trace_config=DirectionalTraceConfig(corridor_radius_px=2.,max_length_ratio=1.6,minimum_support=.025))
        result=audit_short_recovery_topology(result,paths)
        sensitivity=audit_short_recovery_topology(sensitivity,paths)
        result=partition_recovery_edges(result,paths)
        sensitivity=partition_recovery_edges(sensitivity,paths)
        fragment_chains=recover_via_source_fragment(sensitivity,paths,guarded)
        if tid in numeric_masks:
            result=audit_recovery_numeric_ink(result,numeric_masks[tid])
            sensitivity=audit_recovery_numeric_ink(sensitivity,numeric_masks[tid])
            fragment_chains['alternatives']=audit_recovery_numeric_ink({'candidates':fragment_chains['alternatives']},numeric_masks[tid])['candidates']
            for alternative in fragment_chains['alternatives']:
                if 'supported_subgap_previews' in alternative:
                    alternative['supported_subgap_previews']=audit_recovery_numeric_ink(
                        {'candidates':alternative['supported_subgap_previews']},numeric_masks[tid])['candidates']
        panel=image.crop(region['box']).convert('RGB').resize(((r-l)*3,(b-t)*3));draw=ImageDraw.Draw(panel)
        for i,candidate in enumerate(result['candidates'],1):
            candidate['candidate_id']=f'{region["region_id"]}-short-{i:03d}'
            if candidate['points']:
                draw.line([((x-l)*3,(y-t)*3) for x,y in candidate['points']],
                          fill='green' if candidate['status']=='observed' else 'orange',width=3)
        panel.save(output/(region['region_id']+'.png'))
        weak_panel=image.crop(region['box']).convert('RGB').resize(((r-l)*3,(b-t)*3));weak_draw=ImageDraw.Draw(weak_panel)
        for i,candidate in enumerate(sensitivity['candidates'],1):
            candidate['candidate_id']=f'{region["region_id"]}-weak-{i:03d}'
            if candidate['points']:
                weak_draw.line([((x-l)*3,(y-t)*3) for x,y in candidate['points']],
                    fill='red' if candidate['topology_audit']['status']!='passed' or candidate.get('numeric_ink_audit',{}).get('status')=='held_for_numeric_review' else 'green' if candidate['status']=='observed' else 'orange',width=3)
        weak_panel.save(output/(region['region_id']+'-weak-sensitivity.png'))
        results.append({'region_id':region['region_id'],'tile_id':tid,'endpoints':endpoints,'recovery':result,
                        'weak_sensitivity':sensitivity,'source_contrast_guard':guard_audit,'fragment_chains':fragment_chains})
        print(region['region_id'],[(c['status'],c['gap_distance_px']) for c in result['candidates']],flush=True)
    if any(sha256_file(Path(p))!=h for p,h in before.items()):raise ValueError('input or code changed')
    write(output/'short-recovery-report.json',{'schema':'jap-map-short-recovery-run/1','regions':results,'input_sha256':before,
          'all_fixed_development_regions':all_regions,
          'numeric_candidate_masks_checked':numeric_separation is not None,
          'complete_batches_requested':complete_batches,
          'connections_applied':0,'human_approvals':0,'holdout_used':False,'source_modified':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','manifest','vectors'):p.add_argument(name,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--all-regions',action='store_true',help='Use all 24 fixed development regions, never holdout sheets')
    p.add_argument('--numeric-separation',type=Path,help='Frozen numeric separation packet; audit every traversed source pixel cell')
    p.add_argument('--complete-batches',action='store_true',help='Complete all candidate batches under bounded per-batch search limits')
    a=p.parse_args();run(a.packet,a.manifest,a.vectors,a.output,all_regions=a.all_regions,numeric_separation=a.numeric_separation,
                         complete_batches=a.complete_batches)
