#!/usr/bin/env python3
"""Run the source-coupled review pipeline from existing frozen OCR evidence.

No model download, training, git push, or promotion to final contour truth.
Each run uses a new directory; failed stage outputs remain for diagnosis.
"""
import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from histcontour_core.provenance import sha256_file
from scripts.prepare_contour_reconstruction_review import read,write,validate_regions


@dataclass(frozen=True)
class Stage:
    name:str
    command:tuple
    required_output:Path


def stages(args,regions):
    out=args.output
    def stage(name,script,parameters,required,qgis=False):
        interpreter=args.qgis_python if qgis else Path(sys.executable)
        return Stage(name,tuple(str(v) for v in [interpreter,ROOT/'scripts'/script,*parameters]),out/required)
    observed=out/'observed';numeric=out/'numeric';recovery=out/'recovery';gaps=out/'gaps';review=out/'review'
    assembled=out/'assembled';smoothed=out/'smoothed';semantic=out/'semantic';topology=out/'topology'
    return [
        stage('observed','reconstruct_observed_contours.py',[args.packet,args.manifest,args.detection,args.approved,'--output',observed,'--regions',*regions],'observed/reconstruction-report.json'),
        stage('numeric','separate_numeric_ink.py',[args.packet,args.manifest,args.recognition,observed/'candidate-observed.geojson',args.approved,'--output',numeric],'numeric/numeric-separation-report.json'),
        stage('recovery','recover_short_observed_gaps.py',[args.packet,args.manifest,observed/'raw-observed.geojson','--all-regions','--complete-batches','--numeric-separation',numeric,'--output',recovery],'recovery/short-recovery-report.json'),
        stage('gaps','inspect_numeric_gap_context.py',[args.packet,args.manifest,numeric/'numeric-separation-report.json',observed/'raw-observed.geojson','--output',gaps],'gaps/gap-context-report.json'),
        stage('review','build_reconstruction_review_project.py',[args.packet,args.manifest,numeric,gaps,args.feedback,'--short-recovery',recovery/'short-recovery-report.json','--output',review],'review/project-validation.json',True),
        stage('assembled','assemble_reconstruction_network.py',[args.packet,args.manifest,review,'--output',assembled],'assembled/assembly-report.json'),
        stage('smoothed','smooth_assembled_reconstruction.py',[args.packet,args.manifest,assembled,'--output',smoothed],'smoothed/smoothing-report.json'),
        stage('semantic','locate_semantic_review_overlaps.py',[args.packet,args.manifest,smoothed/'smoothed-candidate-network.geojson',args.non_contour_references,'--output',semantic],'semantic/semantic-overlap-report.json'),
        stage('topology','validate_network_topology.py',[assembled/'assembled-candidate-network.geojson',smoothed/'smoothed-candidate-network.geojson','--output',topology],'topology/topology-validation.json',True),
        stage('integrated','build_integrated_network_review.py',[review,assembled,smoothed,semantic,'--topology-validation',topology/'topology-validation.json','--output',out/'integrated'],'integrated/integrated-project-validation.json',True)]


def execute(plan,output,*,run_command=subprocess.run):
    records=[];env=dict(os.environ);env['QT_QPA_PLATFORM']='offscreen'
    for stage in plan:
        print('Starting '+stage.name,flush=True);started=time.perf_counter();log=output/(stage.name+'.log')
        try:
            with log.open('x') as stream:
                result=run_command(list(stage.command),stdout=stream,stderr=subprocess.STDOUT,env=env,cwd=ROOT,check=False)
            ok=result.returncode==0 and stage.required_output.is_file()
            row={'stage':stage.name,'exit_code':result.returncode,'required_output':str(stage.required_output),
                 'log':str(log),'wall_seconds':time.perf_counter()-started,'passed':ok}
        except OSError as error:
            row={'stage':stage.name,'passed':False,'error':str(error),'log':str(log)}
        records.append(row);write(output/(stage.name+'-result.json'),row)
        if not row['passed']:
            write(output/'run-status.json',{'status':'failed','stages':records,'whole_contour_quality_established':False})
            raise RuntimeError('Pipeline stopped at '+stage.name+'; see '+str(log))
        print('Completed '+stage.name,flush=True)
    return records


def run(args):
    for name in ('packet','manifest','detection','recognition','approved','feedback','non_contour_references','output'):
        setattr(args,name,Path(getattr(args,name)).resolve())
    # QGIS's macOS wrapper resolves sibling binaries relative to argv[0].
    # Resolving this launcher symlink changes its behavior and breaks startup.
    args.qgis_python=Path(args.qgis_python).absolute()
    if args.output.exists():raise FileExistsError('pipeline output must be new')
    source_dirs=(args.packet,args.feedback,args.recognition.parent,args.detection.parent)
    if any(p==args.output or p in args.output.parents for p in source_dirs):raise ValueError('pipeline output inside input snapshot')
    manifest=read(args.manifest);report_path=args.packet/'drawing-report.json';report=read(report_path)
    validate_regions(manifest,report)
    if sha256_file(report_path)!=manifest['drawing_report_sha256']:raise ValueError('source manifest changed')
    plan=stages(args,[r['region_id'] for r in manifest['regions']])
    inputs=[report_path,args.manifest,args.detection,args.recognition,args.approved,args.non_contour_references,
            args.feedback/'feedback-application.json',args.feedback/'reviewed-local-connections.geojson']
    input_hashes={str(p):sha256_file(p) for p in inputs}
    code_hashes={str(p):sha256_file(p) for folder in (ROOT/'histcontour_core',ROOT/'scripts') for p in folder.glob('*.py')}
    if args.plan_only:
        for stage in plan:print(stage.name,list(stage.command))
        return
    args.output.mkdir(parents=True)
    records=execute(plan,args.output)
    if any(sha256_file(Path(p))!=digest for p,digest in {**input_hashes,**code_hashes}.items()):
        write(args.output/'run-status.json',{'status':'failed_input_or_code_changed','stages':records});raise ValueError('input or code changed during pipeline')
    write(args.output/'run-status.json',{'schema':'jap-map-reconstruction-pipeline/1','status':'completed_review_pipeline',
        'stages':records,'input_sha256':input_hashes,'implementation_sha256':code_hashes,
        'project':str(args.output/'integrated/contour-network-review.qgz'),'whole_contour_quality_established':False,
        'human_approvals_added':0,'model_fitted':False,'holdout_used':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','manifest','detection','recognition','approved','feedback','non-contour-references','qgis-python','output'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--plan-only',action='store_true');run(p.parse_args())
