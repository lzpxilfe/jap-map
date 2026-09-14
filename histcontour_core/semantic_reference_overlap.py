"""Locate aligned overlaps with sparse semantic references, never label a whole route."""
import math


def reference_overlap(points,reference,*,tolerance_px=1.):
    import numpy as np
    p,q=np.asarray(points,dtype=float),np.asarray(reference,dtype=float)
    if any(a.ndim!=2 or a.shape[1:]!=(2,) or len(a)<2 or not np.isfinite(a).all() for a in (p,q)):
        raise ValueError('invalid semantic reference geometry')
    if len(p)>100000 or len(q)>1024 or not 0<tolerance_px<=2:raise ValueError('semantic overlap budget exceeded')
    if np.any(p.max(axis=0)<q.min(axis=0)-tolerance_px) or np.any(p.min(axis=0)>q.max(axis=0)+tolerance_px):
        return {'aligned_overlap_length_px':0.,'sampled_spans':[],'whole_line_semantics_assigned':False}
    qa,qb=q[:-1],q[1:];vectors=qb-qa;norm2=np.sum(vectors*vectors,axis=1);valid=norm2>1e-12
    qa,vectors,norm2=qa[valid],vectors[valid],norm2[valid]
    if not len(qa):raise ValueError('reference has no positive length')
    spans=[];position=0.;work=0
    for a,b in zip(p,p[1:]):
        delta=b-a;length=float(np.linalg.norm(delta))
        if not length:continue
        count=max(1,math.ceil(length*2));work+=count*len(qa)
        if work>8000000:raise ValueError('semantic overlap comparison budget exceeded')
        samples=a+((np.arange(count)+.5)/count)[:,None]*delta
        rel=samples[:,None,:]-qa[None,:,:]
        t=np.clip(np.sum(rel*vectors,axis=2)/norm2,0,1)
        distances=np.linalg.norm(rel-t[:,:,None]*vectors,axis=2)
        nearest=np.argmin(distances,axis=1)
        alignment=np.abs(np.sum(vectors[nearest]*(delta/length),axis=1)/np.sqrt(norm2[nearest]))
        covered=(distances[np.arange(count),nearest]<=tolerance_px)&(alignment>=math.cos(math.radians(30)))
        for i,is_covered in enumerate(covered):
            if not is_covered:continue
            start=position+i*length/count;end=position+(i+1)*length/count
            if spans and abs(spans[-1][1]-start)<1e-8:spans[-1][1]=end
            else:spans.append([start,end])
        position+=length
    return {'aligned_overlap_length_px':sum(b-a for a,b in spans),'sampled_spans':spans,
            'sample_spacing_max_px':.5,'distance_tolerance_px':tolerance_px,'maximum_axial_error_degrees':30,
            'whole_line_semantics_assigned':False,'measurement_is_sampled_not_exact_intersection':True}
