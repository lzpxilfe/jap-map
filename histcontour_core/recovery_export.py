"""Native-coordinate exports of new, audited recovery edges, not merged maps."""
from histcontour_core.human_feedback import validate_pixel_points


def export_recovery_additions(report,tiles):
    if (report.get('schema')!='jap-map-short-recovery-run/1' or report.get('holdout_used') is not False
            or report.get('human_approvals')!=0 or report.get('numeric_candidate_masks_checked') is not True):
        raise ValueError('expected frozen numerically audited development recovery')
    features=[];seen={};omitted=[]
    for region in report['regions']:
        tid=region['tile_id'];tile=tiles[tid]
        west,south,east,north=tile['bounds'];width,height=tile['pixel_bounds'][2:]
        for candidate in region['weak_sensitivity']['candidates']:
            cid=candidate['candidate_id']
            if not candidate['points']:continue
            if (candidate['topology_audit']['status']!='passed'
                    or candidate['numeric_ink_audit']['status']!='no_numeric_candidate_overlap'
                    or candidate['edge_partition']['requires_topology_resolution']):
                omitted.append({'candidate_id':cid,'reason':'failed_geometry_numeric_or_branch_audit'});continue
            if candidate.get('human_approved') is not False or candidate.get('training_eligible') is not False:
                raise ValueError('cannot reinterpret human or training geometry as automatic recovery')
            for part_index,part in enumerate(candidate['edge_partition']['parts']):
                if part['kind']!='new_recovery_edge':continue
                points=part['points'];validate_pixel_points(points,tile)
                key=(tid,min(tuple(map(tuple,points)),tuple(reversed(tuple(map(tuple,points))))))
                if key in seen:
                    props=features[seen[key]]['properties']
                    props['candidate_ids'].append(cid)
                    props['inferred_gap']=props['inferred_gap'] or candidate['status']!='observed'
                    continue
                coordinates=[[west+(x+.5)*(east-west)/width,north-(y+.5)*(north-south)/height] for x,y in points]
                seen[key]=len(features)
                features.append({'type':'Feature','properties':{'recovery_id':f'{cid}-part-{part_index:03d}',
                    'candidate_ids':[cid],'tile_id':tid,'source_path_ids':candidate['source_path_ids'],
                    'source_raster_sha256':tile['source_raster_sha256'],'human_approved':False,'training_eligible':False,
                    'contour_semantics_assigned':False,'inferred_gap':candidate['status']!='observed',
                    'dataset_role':'review_only_not_training','existing_source_edges_excluded':True},
                    'geometry':{'type':'LineString','coordinates':coordinates}})
    return {'type':'FeatureCollection','crs':{'type':'name','properties':{'name':next(iter(tiles.values()))['crs_authid']}},
            'features':features,'omitted_candidates':omitted,'whole_network_approved':False,'connections_applied':0}
