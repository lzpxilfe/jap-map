"""Independent neighboring-stroke evidence for unapproved gap hypotheses.

Votes are capped per observed path, exclude both candidate parents and label
context, and use axial (sign-invariant) orientation. Ranking is not routing,
contour classification, or selection of a globally consistent network.
"""
import math
from histcontour_core.regional_gap_matching import _intersect


def rank_gap_context(matches, paths, box, *, radius=24.):
    import numpy as np
    if len(matches)>256 or len(paths)>20000 or not 4 <= radius <= 64:
        raise ValueError('context ranking exceeds bounded input')
    b=np.asarray(box,dtype=float)
    if b.shape!=(4,) or not np.isfinite(b).all() or np.any(b[:2]>=b[2:]):
        raise ValueError('invalid context box')
    references=[]; identities=set(); vertices=0
    for path in paths:
        pid=path['source_path_id']; p=np.asarray(path['points'],dtype=float)
        if not isinstance(pid,str) or not pid or pid in identities:
            raise ValueError('reference path identities must be unique')
        identities.add(pid)
        if p.ndim!=2 or p.shape[1:]!=(2,) or len(p)<2 or not np.isfinite(p).all():
            raise ValueError('invalid reference path')
        vertices+=len(p)
        if vertices>1000000:raise ValueError('reference vertex budget exceeded')
        if float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())<12:continue
        local=[]
        for i in range(3,len(p)-3,3):
            center=p[i]; delta=p[i+3]-p[i-3]; length=float(np.linalg.norm(delta))
            if length<4 or (b[0]<=center[0]<=b[2] and b[1]<=center[1]<=b[3]):continue
            # The whole tangent window must avoid the label context, not just
            # its centre: a stroke through text cannot vote as a neighbor.
            window=p[i-3:i+4]
            if np.any(np.all((window>=b[:2]) & (window<=b[2:]),axis=1)):continue
            local.append((center,delta/length))
        if local:references.append((pid,np.asarray([x[0] for x in local]),np.asarray([x[1] for x in local])))
    output=[]; unique=set()
    if any(isinstance(m.get('cost'),bool) or not isinstance(m.get('cost'),(int,float)) or not math.isfinite(m['cost']) or m['cost']<0 for m in matches):
        raise ValueError('invalid endpoint cost')
    # Different bank axes can offer the same pair. Retain its lowest proposal
    # cost deterministically, never multiply evidence by orientation count.
    for m in sorted(matches,key=lambda m:m['cost']):
        start,end=np.asarray(m['start'],dtype=float),np.asarray(m['end'],dtype=float)
        if start.shape!=(2,) or end.shape!=(2,) or not np.isfinite([start,end]).all():raise ValueError('invalid endpoints')
        length=float(np.linalg.norm(end-start))
        if length<.5 or length>512:raise ValueError('invalid gap length')
        source=m['source_path_ids']
        if len(source)!=2 or any(not isinstance(s,str) for s in source):raise ValueError('invalid source IDs')
        key=tuple(sorted(((source[0],tuple(start)),(source[1],tuple(end)))))
        if key in unique:continue
        unique.add(key)
        direction=(end-start)/length; samples=[]
        enter,leave=0.,1.
        for k in (0,1):
            delta=end[k]-start[k]
            if abs(delta)<1e-9:
                if not b[k]<=start[k]<=b[k+2]:leave=-1.
            else:
                lo,hi=sorted(((b[k]-start[k])/delta,(b[k+2]-start[k])/delta))
                enter,leave=max(enter,lo),min(leave,hi)
        box_span=max(0.,leave-enter)*length
        for fraction in (.25,.5,.75):
            point=start+fraction*(end-start); neighbors=[]
            for pid,centers,axes in references:
                if pid in source:continue
                distances=np.linalg.norm(centers-point,axis=1); index=int(np.argmin(distances))
                if distances[index]<=radius:
                    neighbors.append((float(distances[index]),pid,axes[index]))
            neighbors.sort(key=lambda n:(n[0],n[1]));neighbors=neighbors[:8]
            tensor=np.zeros((2,2)); weight=0.
            for distance,_pid,axis in neighbors:
                w=1/(distance+2);tensor+=w*np.outer(axis,axis);weight+=w
            coherence=error=None
            if weight:
                eigenvalues,eigenvectors=np.linalg.eigh(tensor/weight)
                coherence=float(eigenvalues[-1]-eigenvalues[0])
                error=float(math.degrees(math.acos(min(1.,abs(float(direction@eigenvectors[:,-1]))))))
            reliable=len(neighbors)>=3 and coherence is not None and coherence>=.65
            samples.append({'fraction':fraction,'neighbor_path_ids':[n[1] for n in neighbors],
                'neighbor_distances_px':[n[0] for n in neighbors],'axial_coherence':coherence,
                'axis_error_degrees':error,'reliable_local_axis':reliable})
        reliable=[s for s in samples if s['reliable_local_axis']]
        sufficient=len(reliable)>=2
        penalty=sum(s['axis_error_degrees']/90 for s in reliable)/len(reliable) if sufficient else None
        output.append({'start':start.tolist(),'end':end.tolist(),'source_path_ids':list(source),
            'context_samples':samples,'context_evidence_sufficient':sufficient,
            'context_direction_penalty':penalty,'original_endpoint_cost':float(m['cost']),
            'label_context_intersection_length_px':float(box_span),
            'ranking_cost':float(m['cost'])+.75*penalty if sufficient and box_span>=1. else None,
            'ranking_abstention_reasons':(['insufficient_neighbor_axis'] if not sufficient else [])+
                (['chord_does_not_traverse_label_context'] if box_span<1. else []),
            'human_approved':False,'training_eligible':False,'inferred_gap':True,
            'contour_semantics_assigned':False,'eligible_for_automatic_connection':False})
    # Unknown context is not scored as a perfect direction match.
    output.sort(key=lambda r:(r['ranking_cost'] is None,r['ranking_cost'] or 0,r['source_path_ids'],r['start']))
    for i,row in enumerate(output):
        row['candidate_id']=f'context-gap-{i+1:03d}'
    conflicts=[]
    for i,a in enumerate(output):
        for c in output[i+1:]:
            shared=any(math.dist(p,q)<1e-5 for p in (a['start'],a['end']) for q in (c['start'],c['end']))
            if shared or _intersect(a['start'],a['end'],c['start'],c['end']):
                conflicts.append({'candidate_ids':[a['candidate_id'],c['candidate_id']],
                                  'reason':'endpoint_reuse' if shared else 'chords_cross_or_touch'})
    return {'schema':'jap-map-contextual-gap-ranking/1','candidates':output,'conflicts':conflicts,
            'reference_path_count':len(references),'radius_px':radius,'global_network_selected':False,
            'connections_applied':0,'human_approvals':0,
            'limitations':['Neighbor strokes are not proven contours; coherent roads can also vote.',
                'Local axis penalties describe straight chords; they do not validate curved routes.',
                'Insufficient context is unscored, not positive evidence.']}
