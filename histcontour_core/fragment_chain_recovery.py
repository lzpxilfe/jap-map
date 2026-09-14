"""Recover two missing spans around one preserved intermediate source fragment."""
from dataclasses import asdict
import math
from histcontour_core.directional_trace import trace_observed_ink,DirectionalTraceConfig
from histcontour_core.recovery_edge_partition import _pixel,partition_recovery_edges
from histcontour_core.short_ink_recovery import audit_short_recovery_topology


def recover_via_source_fragment(recovery,paths,evidence):
    import numpy as np
    lookup={p['source_path_id']:p for p in paths}
    if len(lookup)!=len(paths) or len(paths)>20000 or len(recovery['candidates'])>256:
        raise ValueError('invalid fragment chain identity or budget')
    mask=np.zeros(evidence.support_score.shape,bool)
    for path in paths:
        for p in path['points']:
            x,y=_pixel(p)
            if not 0<=y<mask.shape[0] or not 0<=x<mask.shape[1]:raise ValueError('source outside evidence')
            mask[y,x]=True
    output=[]
    for index,row in enumerate(recovery['candidates']):
        partition=row.get('edge_partition',{})
        if not partition.get('requires_topology_resolution'):continue
        reused={ref['source_path_id'] for part in partition['parts'] for refs in part['source_edge_references'] for ref in refs}
        middle_ids=reused-set(row['source_path_ids'])
        record={'parent_candidate_index':index,'status':'abstained','points':[],'human_approved':False,
                'training_eligible':False,'connections_applied':0,'source_modified':False}
        output.append(record)
        if len(middle_ids)!=1:
            record['reason']='requires_more_than_one_intermediate_fragment';continue
        middle_id=next(iter(middle_ids));middle=[list(_pixel(p)) for p in lookup[middle_id]['points']]
        if not 2<=len(middle)<=25 or middle[0]==middle[-1]:
            record['reason']='intermediate_fragment_not_short_open_path';continue
        a,b=row['start'],row['end']
        if math.dist(a,middle[-1])+math.dist(b,middle[0])<math.dist(a,middle[0])+math.dist(b,middle[-1]):middle.reverse()
        if any(not .5<=math.dist(x,y)<=12 for x,y in ((a,middle[0]),(middle[-1],b))):
            record['reason']='subgap_distance_outside_bounds';continue
        bridges=[]
        for start,end,parents in ((a,middle[0],[row['source_path_ids'][0],middle_id]),
                                  (middle[-1],b,[middle_id,row['source_path_ids'][1]])):
            exclusion=mask.copy()
            for x,y in (start,end):exclusion[int(y),int(x)]=False
            trace=trace_observed_ink(evidence,start,end,reference_path=[start,end],exclusion_mask=exclusion,
                config=DirectionalTraceConfig(corridor_radius_px=2.,max_length_ratio=1.6,minimum_support=.025))
            bridges.append({'start':list(start),'end':list(end),'points':[list(p) for p in trace.points],
                            'source_path_ids':parents,'status':trace.status,'audit':asdict(trace.audit)})
        record['bridge_attempts']=bridges;record['intermediate_source_path_id']=middle_id
        partial_checked=audit_short_recovery_topology({'candidates':[b for b in bridges if b['points']]},paths)
        record['supported_subgap_previews']=[b for b in partial_checked['candidates'] if b['topology_audit']['status']=='passed']
        if any(not bridge['points'] for bridge in bridges):
            record['reason']='subgap_has_no_supported_route'
            if record['supported_subgap_previews']:record['status']='partial_subgap_draft_not_complete_chain'
            continue
        checked=audit_short_recovery_topology({'candidates':bridges},paths)
        record['bridge_topology']=checked
        if any(b['topology_audit']['status']!='passed' for b in checked['candidates']):
            record['reason']='subgap_topology_conflict';continue
        points=bridges[0]['points'][:-1]+middle+bridges[1]['points'][1:]
        if len(set(map(tuple,points)))!=len(points):
            record['reason']='combined_route_revisits_source_pixel';continue
        combined=partition_recovery_edges({'candidates':[{'points':points,'source_path_ids':row['source_path_ids']}]},paths)['candidates'][0]
        record['edge_partition']=combined['edge_partition']
        if combined['edge_partition']['requires_topology_resolution']:
            record['reason']='combined_route_still_creates_branch';continue
        record.update(status='source_fragment_preserving_draft',points=points,reason='two_supported_subgaps_and_exact_source_fragment',
                      preserved_source_fragment_points=middle,
                      inferred_gap=any(b['status']!='observed' for b in bridges),eligible_for_automatic_merge=False)
    return {'schema':'jap-map-fragment-chain-recovery/1','alternatives':output,'connections_applied':0,
            'human_approvals':0,'source_modified':False,'semantic_classification_performed':False}
