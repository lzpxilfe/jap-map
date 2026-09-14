#!/usr/bin/env python3
"""QGIS/GEOS validation gate for newly introduced smoothing intersections."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from histcontour_core.provenance import sha256_file
from scripts.prepare_contour_reconstruction_review import write


def compare_networks(before_layer,after_layer):
    from qgis.core import QgsSpatialIndex
    if not before_layer.isValid() or not after_layer.isValid() or before_layer.crs()!=after_layer.crs():
        raise ValueError('invalid or different network CRS')
    old={f['line_id']:f for f in before_layer.getFeatures()};new={f['line_id']:f for f in after_layer.getFeatures()}
    if len(old)!=before_layer.featureCount() or len(new)!=after_layer.featureCount() or old.keys()!=new.keys():
        raise ValueError('duplicate or changed network line identities')
    oi=QgsSpatialIndex(before_layer.getFeatures());ni=QgsSpatialIndex(after_layer.getFeatures())
    old_ids={f.id():lid for lid,f in old.items()};new_ids={f.id():lid for lid,f in new.items()}
    changed=[];invalid=[];moved=[];self_crossings=[];pairs=set()
    for lid,a in old.items():
        b=new[lid]
        if a['tile_id']!=b['tile_id']:raise ValueError('line tile identity changed')
        x,y=a.geometry(),b.geometry()
        if not y.isGeosValid():invalid.append(lid)
        if x.asWkb()==y.asWkb():continue
        changed.append(lid)
        px=list(x.vertices());py=list(y.vertices())
        if not px or not py or (px[0].x(),px[0].y(),px[-1].x(),px[-1].y())!=(py[0].x(),py[0].y(),py[-1].x(),py[-1].y()):moved.append(lid)
        if x.isSimple() and not y.isSimple():self_crossings.append(lid)
        previous={old_ids[i] for i in oi.intersects(x.boundingBox()) if old_ids[i]!=lid and x.intersects(old[old_ids[i]].geometry())}
        current={new_ids[i] for i in ni.intersects(y.boundingBox()) if new_ids[i]!=lid and y.intersects(new[new_ids[i]].geometry())}
        pairs.update(tuple(sorted((lid,other))) for other in current-previous)
        for other in current & previous:
            # An already-intersecting pair can gain another contact, or move
            # its original junction. Pair-set comparison alone misses both.
            old_contact=x.intersection(old[other].geometry())
            new_contact=y.intersection(new[other].geometry())
            added_contact=new_contact.difference(old_contact)
            if not added_contact.isEmpty():pairs.add(tuple(sorted((lid,other))))
    return {'line_count':len(old),'changed_line_ids':changed,'invalid_after_line_ids':invalid,
            'moved_endpoint_line_ids':moved,'new_self_intersection_line_ids':self_crossings,
            'new_line_intersection_pairs':[list(p) for p in sorted(pairs)],
            'existing_pairs_checked_for_added_or_moved_contacts':True,
            'passed':not(invalid or moved or self_crossings or pairs),
            'semantic_accuracy_established':False}


def run(before_path,after_path,output):
    from qgis.core import QgsVectorLayer,Qgis
    before_path,after_path,output=map(lambda p:Path(p).resolve(),(before_path,after_path,output))
    if output.exists():raise FileExistsError('validation output must be new')
    if any(p.parent==output or p.parent in output.parents for p in (before_path,after_path)):
        raise ValueError('validation output inside geometry snapshot')
    before_hash=sha256_file(before_path);after_hash=sha256_file(after_path)
    a=QgsVectorLayer(str(before_path),'before','ogr');b=QgsVectorLayer(str(after_path),'after','ogr')
    result=compare_networks(a,b)
    if sha256_file(before_path)!=before_hash or sha256_file(after_path)!=after_hash:raise ValueError('network changed during validation')
    result.update(schema='jap-map-network-topology-validation/1',before_sha256=before_hash,after_sha256=after_hash,
                  qgis_version=Qgis.QGIS_VERSION,implementation_sha256=sha256_file(Path(__file__)),source_modified=False)
    output.mkdir(parents=True);write(output/'topology-validation.json',result)
    print({'passed':result['passed'],'lines':result['line_count'],'changed':len(result['changed_line_ids']),
           'new_intersections':len(result['new_line_intersection_pairs'])})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('before',type=Path);p.add_argument('after',type=Path)
    p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    from qgis.core import QgsApplication
    app=QgsApplication([],False);app.initQgis()
    if not run(args.before,args.after,args.output)['passed']:p.exit(2,'New topology defects: not validated for delivery.\n')
