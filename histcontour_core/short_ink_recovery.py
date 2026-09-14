"""Propose short terminal-to-terminal observed-ink recovery, without merging."""
from dataclasses import asdict,replace
import math
from histcontour_core.directional_trace import trace_observed_ink,DirectionalTraceConfig


def audit_recovery_numeric_ink(recovery,glyph_mask):
    """Keep rejected traces but block reintroduction of numeric ink candidates.

    Test every traversed pixel cell, not only vertices. A negative result means
    no overlap with this supplied mask, not proof of contour semantics.
    """
    import copy
    import numpy as np
    from histcontour_core.numeric_line_partition import rasterize_line_cells
    mask=np.asarray(glyph_mask)
    if mask.ndim!=2 or mask.dtype.kind!='b' or not all(mask.shape) or mask.size>4_000_000:
        raise ValueError('numeric exclusion must be bounded boolean source grid')
    result=copy.deepcopy(recovery)
    for row in result['candidates']:
        row['eligible_for_automatic_merge']=False
        if not row['points']:
            row['numeric_ink_audit']={'status':'no_route_to_check'};continue
        cells=rasterize_line_cells(mask.shape,[row['points']])
        overlap=int(np.count_nonzero(cells & mask))
        row['numeric_ink_audit']={'status':'held_for_numeric_review' if overlap else 'no_numeric_candidate_overlap',
                                 'overlapping_pixel_cells':overlap,'contour_semantics_proven':False}
    return result


def audit_short_recovery_topology(recovery,paths,*,minimum_foreign_clearance=.75):
    """Audit complete segments, not just sampled vertices; do not change ink status."""
    import copy
    import numpy as np
    from histcontour_core.regional_gap_matching import _intersect
    if not 0<=minimum_foreign_clearance<=2 or len(paths)>20000 or len(recovery['candidates'])>32640:
        raise ValueError('topology audit budget or clearance invalid')
    segments=[];owners=[]
    for path in paths:
        for a,b in zip(path['points'],path['points'][1:]):
            segments.append([a,b]);owners.append(path['source_path_id'])
            if len(segments)>1000000:raise ValueError('source segment budget exceeded')
    array=np.asarray(segments,dtype=float).reshape((-1,2,2))
    if not np.isfinite(array).all():raise ValueError('nonfinite source segment')
    lo=array.min(axis=1);hi=array.max(axis=1)
    result=copy.deepcopy(recovery)
    def point_distance(p,a,b):
        d=b-a;den=float(d@d)
        t=max(0.,min(1.,float((p-a)@d)/den)) if den else 0.
        return float(np.linalg.norm(p-a-t*d))
    for row in result['candidates']:
        row['eligible_for_automatic_merge']=False
        if not row['points']:
            row['topology_audit']={'status':'no_route_to_check'};continue
        collisions=set();nearby=set()
        for a,b in zip(row['points'],row['points'][1:]):
            a,b=np.asarray(a),np.asarray(b)
            lower=np.minimum(a,b)-minimum_foreign_clearance;upper=np.maximum(a,b)+minimum_foreign_clearance
            for index in np.flatnonzero(np.all(hi>=lower,axis=1)&np.all(lo<=upper,axis=1)):
                c,d=array[index];owner=owners[index]
                crosses=_intersect(a,b,c,d)
                if crosses:
                    legitimate_terminal_touch=False
                    for pid,point in zip(row['source_path_ids'],(row['start'],row['end'])):
                        if pid!=owner:continue
                        if any(math.dist(p,point)<1e-5 for p in (a,b)) and any(math.dist(p,point)<1e-5 for p in (c,d)):
                            # Re-entering a parent's first edge is an overlap,
                            # not the legitimate single terminal touch.
                            va=(b if math.dist(a,point)<1e-5 else a)-point
                            vb=(d if math.dist(c,point)<1e-5 else c)-point
                            cross=float(va[0]*vb[1]-va[1]*vb[0])
                            legitimate_terminal_touch=abs(cross)>1e-8 or float(va@vb)<=0
                    if not legitimate_terminal_touch:collisions.add(owner)
                if owner not in row['source_path_ids']:
                    distance=0. if crosses else min(point_distance(a,c,d),point_distance(b,c,d),point_distance(c,a,b),point_distance(d,a,b))
                    if distance<minimum_foreign_clearance:nearby.add(owner)
        row['topology_audit']={'status':'passed' if not collisions and not nearby else 'held_for_review',
            'intersected_source_path_ids':sorted(collisions),'insufficient_clearance_path_ids':sorted(nearby),
            'minimum_foreign_clearance_px':minimum_foreign_clearance}
    conflicts=[]
    traced=[(i,r) for i,r in enumerate(result['candidates']) if r['points']]
    for number,(i,a) in enumerate(traced):
        for j,b in traced[number+1:]:
            if any(_intersect(p,q,r,s) for p,q in zip(a['points'],a['points'][1:]) for r,s in zip(b['points'],b['points'][1:])):
                conflicts.append([i,j])
                for row in (a,b):row['topology_audit']['status']='held_for_review'
    result['route_conflict_candidate_indices']=conflicts
    result['topology_checked']=True
    return result


def contrast_guarded_evidence(gray,evidence):
    """Gate weak support with the extractor's independent source-normal contrast.

    This is a sensitivity input, not a newly calibrated contour probability.
    All geometry and source pixels are unchanged.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter,map_coordinates
    from histcontour_core.observed_linework import _gray
    image=_gray(gray,2_000_000)
    if image.shape!=evidence.support_score.shape:raise ValueError('source and evidence grids differ')
    yy,xx=np.indices(image.shape,dtype=np.float32)
    blurred=gaussian_filter(image,.55);tx,ty=evidence.tangent_x,evidence.tangent_y
    left=map_coordinates(blurred,[yy-2*tx,xx+2*ty],order=1,mode='nearest')
    right=map_coordinates(blurred,[yy+2*tx,xx-2*ty],order=1,mode='nearest')
    contrast=.5*(left+right)-blurred
    support=np.where(contrast>=.008,evidence.support_score,0).astype(np.float32)
    return replace(evidence,support_score=support),{'normal_contrast_floor':.008,
        'supported_pixels_before':int(np.count_nonzero(evidence.support_score>=.025)),
        'supported_pixels_after':int(np.count_nonzero(support>=.025)),
        'source_pixels_modified':False,'sensitivity_only':True}


def recover_short_ink(evidence,endpoints,*,maximum_pairs=24,first_pair=0,
                      trace_config=DirectionalTraceConfig(corridor_radius_px=2.,max_length_ratio=1.6)):
    if len(endpoints)>2048 or type(maximum_pairs) is not int or not 1<=maximum_pairs<=64:
        raise ValueError('short recovery work budget exceeded')
    if type(first_pair) is not int or not 0<=first_pair<=32640:raise ValueError('invalid first pair offset')
    import numpy as np
    from scipy.spatial import cKDTree
    points=np.asarray([e['point'] for e in endpoints],dtype=float).reshape((-1,2))
    if not np.isfinite(points).all():raise ValueError('nonfinite endpoint')
    nearby=sorted(cKDTree(points).query_pairs(12.)) if len(points) else []
    if len(nearby)>32640:raise ValueError('short recovery nearby-pair budget exceeded')
    candidates=[]
    for i,j in nearby:
        a,b=endpoints[i],endpoints[j]
        if a['source_path_id']==b['source_path_id']:continue
        distance=math.dist(a['point'],b['point'])
        if not 2<=distance<=12:continue
        direction=[(q-p)/distance for p,q in zip(a['point'],b['point'])]
        if sum(x*y for x,y in zip(direction,a['outward_tangent']))<.8:continue
        if -sum(x*y for x,y in zip(direction,b['outward_tangent']))<.8:continue
        candidates.append((distance,i,j))
    candidates.sort()
    output=[]
    for rank,(distance,i,j) in enumerate(candidates):
        a,b=endpoints[i],endpoints[j]
        row={'start':list(a['point']),'end':list(b['point']),
             'source_path_ids':[a['source_path_id'],b['source_path_id']],
             'gap_distance_px':distance,'human_approved':False,'training_eligible':False,
             'contour_semantics_assigned':False,'points':[]}
        output.append(row)
        if rank<first_pair or rank>=first_pair+maximum_pairs:
            row['status']='pair_cap_not_run';continue
        if any((i in (u,v) or j in (u,v)) and (u,v)!=(i,j) and abs(other-distance)<=2
               for other,u,v in candidates):
            row['status']='competing_endpoints_abstained';continue
        # Keep all unsupported spans explicit; a white route is not recovered
        # observed ink. The frozen support threshold is not lowered per case.
        result=trace_observed_ink(evidence,a['point'],b['point'],reference_path=[a['point'],b['point']],
            config=trace_config)
        row.update(status=result.status,points=[list(p) for p in result.points],audit=asdict(result.audit),
                   inferred_gap=result.status!='observed',merged_into_source=False)
    return {'schema':'jap-map-short-ink-recovery/1','candidates':output,'pairs_before_cap':len(candidates),
            'trace_configuration':asdict(trace_config),'endpoint_count':len(endpoints),'nearby_pairs_examined':len(nearby),
            'batch_first_pair':first_pair,'batch_size':maximum_pairs,
            'connections_applied':0,'human_approvals':0,'source_modified':False}


def recover_short_ink_batched(evidence,endpoints,*,maximum_pairs=24,maximum_batches=64,
                              trace_config=DirectionalTraceConfig(corridor_radius_px=2.,max_length_ratio=1.6)):
    """Complete bounded batches while comparing competing endpoints globally."""
    if type(maximum_batches) is not int or not 1<=maximum_batches<=64:raise ValueError('invalid batch count budget')
    first=recover_short_ink(evidence,endpoints,maximum_pairs=maximum_pairs,trace_config=trace_config)
    total=first['pairs_before_cap'];batches=[]
    if total>maximum_pairs*maximum_batches:raise ValueError('complete recovery exceeds total batch budget')
    for offset in range(0,total,maximum_pairs):
        batch=first if offset==0 else recover_short_ink(evidence,endpoints,maximum_pairs=maximum_pairs,
            first_pair=offset,trace_config=trace_config)
        if batch['pairs_before_cap']!=total:raise ValueError('candidate enumeration changed between batches')
        end=min(total,offset+maximum_pairs)
        for i in range(offset,end):
            if first['candidates'][i]['start']!=batch['candidates'][i]['start'] or first['candidates'][i]['end']!=batch['candidates'][i]['end']:
                raise ValueError('candidate ordering changed between batches')
            first['candidates'][i]=batch['candidates'][i]
        batches.append({'first_pair':offset,'stop_pair_exclusive':end})
    if any(c['status']=='pair_cap_not_run' for c in first['candidates']):raise ValueError('batch merge left an unexamined candidate')
    first.update(completed_batches=batches,all_candidates_examined=True)
    return first
