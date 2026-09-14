"""Separate recovered pixel edges from reused raw observed edges.

The original source remains authoritative. Only integer adjacent-pixel paths
are supported, with tolerance solely for native-coordinate round trips.
"""
import math
from collections import defaultdict


def _pixel(p):
    if len(p)!=2 or any(not math.isfinite(v) or abs(v-round(v))>1e-5 for v in p):
        raise ValueError('edge partition requires raw integer pixel coordinates')
    return tuple(int(round(v)) for v in p)


def _edge(a,b):
    if a==b or max(abs(x-y) for x,y in zip(a,b))>1:
        raise ValueError('edge partition requires nonzero adjacent pixel edges')
    return tuple(sorted((a,b)))


def partition_recovery_edges(recovery,paths):
    import copy
    if len(paths)>20000 or len(recovery['candidates'])>32640:raise ValueError('edge partition input budget exceeded')
    owners=defaultdict(list);incident=defaultdict(set);count=0
    for path in paths:
        points=[_pixel(p) for p in path['points']]
        for index,(a,b) in enumerate(zip(points,points[1:])):
            key=_edge(a,b);count+=1
            if count>1000000:raise ValueError('edge partition source budget exceeded')
            owners[key].append({'source_path_id':path['source_path_id'],'source_edge_index':index})
            incident[a].add(key);incident[b].add(key)
    result=copy.deepcopy(recovery)
    for row in result['candidates']:
        if not row['points']:continue
        pixels=[_pixel(p) for p in row['points']];parts=[];new_edges=set();duplicate_route_edges=[]
        seen=set()
        for index,(a,b) in enumerate(zip(pixels,pixels[1:])):
            key=_edge(a,b)
            if key in seen:duplicate_route_edges.append(index)
            seen.add(key)
            references=owners.get(key,[])
            kind='existing_observed_edge' if references else 'new_recovery_edge'
            if not references:new_edges.add(key)
            if not parts or parts[-1]['kind']!=kind:
                parts.append({'kind':kind,'points':[list(a)],'route_edge_indices':[],
                              'source_edge_references':[],'length_px':0.})
            part=parts[-1];part['points'].append(list(b));part['route_edge_indices'].append(index)
            part['source_edge_references'].append(copy.deepcopy(references));part['length_px']+=math.dist(a,b)
        added_incident=defaultdict(set)
        for key in new_edges:
            for point in key:added_incident[point].add(key)
        attachments=[]
        for point,added in sorted(added_incident.items()):
            if incident.get(point):
                degree_before=len(incident[point]);degree_after=len(incident[point]|added)
                attachments.append({'pixel':list(point),'source_degree':degree_before,'combined_degree':degree_after,
                                    'introduces_or_extends_branch':degree_after>2})
        row['edge_partition']={'parts':parts,'new_edge_count':len(new_edges),
            'reused_edge_count':sum(len(p['route_edge_indices']) for p in parts if p['kind']=='existing_observed_edge'),
            'attachments':attachments,'duplicate_route_edge_indices':duplicate_route_edges,
            'requires_topology_resolution':bool(duplicate_route_edges or any(a['introduces_or_extends_branch'] for a in attachments)),
            'source_geometry_modified':False,'connections_applied':0}
        # Existing collision audits stay intact: removing duplicate edges alone
        # is not proof of a legitimate network connection.
        row['eligible_for_automatic_merge']=False
    return result
