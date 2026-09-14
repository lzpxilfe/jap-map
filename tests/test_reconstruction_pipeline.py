from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from scripts.run_reconstruction_pipeline import Stage,execute,stages,run


class ReconstructionPipelineTests(unittest.TestCase):
    def test_qgis_launcher_symlink_is_not_resolved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);target=root/'actual';target.write_text('launcher');link=root/'launcher';link.symlink_to(target)
            args=SimpleNamespace(**{k:root/'inputs'/k for k in ('packet','manifest','detection','recognition','approved','feedback','non_contour_references')},
                                 qgis_python=link,output=root/'output',plan_only=True)
            with patch('scripts.run_reconstruction_pipeline.read',side_effect=[{'drawing_report_sha256':'digest','regions':[]},{}]), \
                 patch('scripts.run_reconstruction_pipeline.validate_regions'), \
                 patch('scripts.run_reconstruction_pipeline.sha256_file',return_value='digest'), \
                 patch('scripts.run_reconstruction_pipeline.stages',return_value=[]):run(args)
            self.assertEqual(args.qgis_python,link.absolute());self.assertNotEqual(args.qgis_python,target)

    def test_failed_stage_prevents_downstream_execution(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);calls=[]
            def fail(command,**kwargs):calls.append(command);return SimpleNamespace(returncode=2)
            plan=[Stage('a',('python','first'),out/'a.json'),Stage('b',('python','second'),out/'b.json')]
            with self.assertRaises(RuntimeError):execute(plan,out,run_command=fail)
            self.assertEqual(len(calls),1);self.assertFalse((out/'b.log').exists())

    def test_exit_zero_without_required_artifact_is_failure(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            with self.assertRaises(RuntimeError):execute([Stage('a',('python','stage'),out/'missing')],out,
                run_command=lambda *a,**k:SimpleNamespace(returncode=0))

    def test_successive_stage_records_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            plan=[Stage(name,('python',name),out/(name+'.json')) for name in ('a','b')]
            def succeed(command,**kwargs):
                (out/(command[1]+'.json')).write_text('{}');return SimpleNamespace(returncode=0)
            result=execute(plan,out,run_command=succeed)
            self.assertEqual(len(result),2)
            self.assertTrue((out/'a-result.json').is_file());self.assertTrue((out/'b-result.json').is_file())
            self.assertFalse((out/'run-status.json').exists())

    def test_topology_precedes_integrated_project_and_arguments_are_separate(self):
        args=SimpleNamespace(**{k:Path('/example with spaces')/k for k in ('packet','manifest','detection','recognition','approved','feedback','non_contour_references','qgis_python','output')})
        plan=stages(args,['R001','R002'])
        self.assertEqual([s.name for s in plan][-2:],['topology','integrated'])
        self.assertIn(str(args.packet),plan[0].command)
        self.assertIn('--complete-batches',plan[2].command)
        self.assertIn('--topology-validation',plan[-1].command)


if __name__=='__main__':unittest.main()
