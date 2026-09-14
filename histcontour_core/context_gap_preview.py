"""Bounded smooth previews for ranked hypotheses, never approved connections."""
import math
from histcontour_core.manual_gap_bridge import build_manual_gap_bridge, ManualGapBridgeError
from histcontour_core.regional_gap_matching import _intersect


def resolve_straight_feedback(report, feedback, report_sha256):
    """Resolve explicit local geometry feedback against an immutable preview."""
    import copy
    if feedback.get('schema')!='jap-map-context-gap-feedback/1' or feedback.get('source_report_sha256')!=report_sha256:
        raise ValueError('feedback does not match frozen preview report')
    results=[];seen=set()
    for decision in feedback['decisions']:
        key=(decision['case_id'],decision['candidate_id'])
        if key in seen:raise ValueError('duplicate local decision')
        seen.add(key)
        cases=[c for c in report['cases'] if c['case_id']==key[0]]
        if len(cases)!=1:raise ValueError('feedback case not uniquely found')
        case=cases[0]
        curves=[c for c in case['curve_previews']['previews'] if c['candidate_id']==key[1]]
        candidates=[c for c in case['contextual_ranking']['candidates'] if c['candidate_id']==key[1]]
        if len(curves)!=1 or len(candidates)!=1:raise ValueError('feedback candidate not uniquely found')
        curve,candidate=curves[0],candidates[0]
        if (decision.get('scope')!='this_connection_only' or decision.get('connection_decision')!='approved'
                or decision.get('geometry_decision')!='straight_acceptable' or not decision.get('user_quote')
                or decision.get('training_eligible') is not False):raise ValueError('unsupported explicit local decision')
        if (case['tile_id']!=decision['tile_id'] or candidate['source_path_ids']!=decision['source_path_ids']
                or curve['original_endpoints']!=decision['expected_endpoints'] or not curve['points']):
            raise ValueError('feedback geometry or source identity differs')
        points=copy.deepcopy(decision['expected_endpoints'])
        results.append({'case_id':key[0],'candidate_id':key[1],'tile_id':case['tile_id'],
            'source_path_ids':copy.deepcopy(candidate['source_path_ids']), 'points':points,
            'previous_points':copy.deepcopy(curve['points']), 'user_quote':decision['user_quote'],
            'connection_human_approved':True,'geometry_authorized_by_user':True,
            'contour_semantics_assigned':False,'semantic_decision':decision['semantic_decision'],
            'training_eligible':False,'inferred_gap':True,'source_tails_modified':False,
            'whole_network_human_approved':False,'global_smoothing_policy_changed':False})
    return results


def build_context_gap_previews(ranking, endpoints, paths):
    import numpy as np
    if len(ranking['candidates'])>256 or len(endpoints)>512 or len(paths)>20000:
        raise ValueError('preview input exceeds budget')
    lookup={(e['source_path_id'],tuple(e['point'])):e for e in endpoints}
    segments=[]; owners=[]
    for path in paths:
        for a,b in zip(path['points'],path['points'][1:]):
            segments.append([a,b]);owners.append(path['source_path_id'])
            if len(segments)>1000000:raise ValueError('preview segment budget exceeded')
    array=np.asarray(segments,dtype=float).reshape((-1,2,2))
    lo=array.min(axis=1);hi=array.max(axis=1)
    results=[]
    for candidate in ranking['candidates']:
        row={'candidate_id':candidate['candidate_id'],'human_approved':False,'training_eligible':False,
             'inferred_gap':True,'eligible_for_automatic_connection':False,'points':[],
             'status':'not_constructed','reasons':[]}
        results.append(row)
        if candidate['ranking_cost'] is None:
            row['reasons']=['context_ranking_abstained'];continue
        a,b=candidate['start'],candidate['end'];delta=np.asarray(b)-a;length=np.linalg.norm(delta)
        direction=delta/length
        ends=[lookup.get((pid,tuple(point))) for pid,point in zip(candidate['source_path_ids'],(a,b))]
        if any(e is None for e in ends):
            row['reasons']=['source_endpoint_not_found'];continue
        ta,tb=[np.asarray(e['outward_tangent'],dtype=float) for e in ends]
        if float(ta@direction)<=0 or float(tb@direction)>=0:
            row['reasons']=['outward_endpoint_direction_incompatible'];continue
        try:
            bridge=build_manual_gap_bridge(a,b,ta,tb)
        except ManualGapBridgeError as error:
            row['reasons']=[str(error)];continue
        points=[list(p) for p in bridge.points]
        # Endpoints must remain exact, even when a short-gap regularizer runs.
        if points[0]!=list(a) or points[-1]!=list(b):raise ValueError('bridge moved an endpoint')
        row.update(points=points,status='unapproved_curve_preview',detour_ratio=bridge.detour_ratio,
                   original_endpoints=[list(a),list(b)],source_tails_modified=False)
        errors=[]
        for actual,expected in ((np.asarray(points[1])-points[0],ta),(np.asarray(points[-2])-points[-1],tb)):
            cosine=float(actual@expected/np.linalg.norm(actual)/np.linalg.norm(expected))
            errors.append(math.degrees(math.acos(max(-1.,min(1.,cosine)))))
        row['sampled_endpoint_tangent_errors_degrees']=errors
        if max(errors)>12:row['reasons'].append('bounded_curve_does_not_match_source_tangent')
        collided=set()
        for c,d in zip(points,points[1:]):
            lower=np.minimum(c,d);upper=np.maximum(c,d)
            candidates=np.flatnonzero(np.all(hi>=lower-1e-8,axis=1)&np.all(lo<=upper+1e-8,axis=1))
            for index in candidates:
                e,f=array[index]
                # Exempt only a parent's terminal edge sharing the curve end.
                # Other paths touching that endpoint remain collisions.
                parent_touch=any(owners[index]==pid and any(math.dist(p,endpoint)<1e-5 for p in (e,f))
                    and any(math.dist(p,endpoint)<1e-5 for p in (c,d))
                    for pid,endpoint in zip(candidate['source_path_ids'],(a,b)))
                if not parent_touch and _intersect(c,d,e,f):collided.add(owners[index])
        row['observed_collision_path_ids']=sorted(collided)
        if collided:row['reasons'].append('curve_crosses_observed_linework')
        row['geometry_checks_passed']=not row['reasons']
    conflicts=[]
    for i,a in enumerate(results):
        if not a['points']:continue
        for b in results[i+1:]:
            if b['points'] and any(_intersect(c,d,e,f) for c,d in zip(a['points'],a['points'][1:])
                                  for e,f in zip(b['points'],b['points'][1:])):
                conflicts.append([a['candidate_id'],b['candidate_id']])
    return {'schema':'jap-map-context-gap-preview/1','previews':results,'curve_conflicts':conflicts,
            'connections_applied':0,'human_approvals':0,'global_network_selected':False,
            'limitations':['Geometry checks do not prove contour identity or the selected route.',
                           'Tangent slope limits can change endpoint direction; deviations are reported.']}
