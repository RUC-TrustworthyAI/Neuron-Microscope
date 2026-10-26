"""Exercise run.sh with fake stage work; no Docker, model or API calls."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from common import read
from test_session import BASE,STAGES,prepare_batch

DRIVER = r'''import json,sys
from pathlib import Path
base,folder,stage,previous,mode=sys.argv[1:]
sys.path[:0]=[base+'/tools',base+'/tests']
from common import read,write
from launch_files import main
from session import Session
from stage_session import StageSession
from test_session import fake_measure
folder=Path(folder)
main(['attach',str(folder),previous])
with (folder.parent/'order.txt').open('a') as stream:stream.write(stage+'\n')
if mode=='fail' and stage=='02-experiments':raise SystemExit(7)
def fake_publish(self,arguments):
    paper=self.root/'paper';paper.mkdir()
    bundle={'neuron':{'id':'L3_N7'},'distribution':{'n_tokens':100},**arguments}
    page='<script id="bundle" type="application/json">'+json.dumps(bundle)+'</script>'
    page+="<script>const B = JSON.parse(document.getElementById('bundle').textContent); const POS = 1;</script>"
    (paper/'index.html').write_text(page)
    write(paper/'report.json',{'identity':self.config['neuron']})
    return {'status':'complete','report':'paper/index.html'}
Session.finish=fake_publish
runner=(lambda group,cfg:{'error':'Worker did not start'}) if mode=='unmeasured' else fake_measure
config=read(folder/'host-session.json');config['full_paper']=False
session=StageSession(config,runner=runner)
if stage=='01-interpretation':
    session.call('submit_interpretation',{'claim':{'title':'Initial'},'criticism':[]})
elif stage=='02-experiments':
    session.call('run_experiments',{'sweeps':[{'template':'{slot}','replacements':['a','b']}]})
    session.call('complete_experiments',{'summary':'Measured contrast'})
else:
    session.call('finish',{'claim':{'title':'Revised'},'criticism':[]})
'''

class ThreeStageTests(unittest.TestCase):
    def execute(self,tmp,mode='success'):
        batch=prepare_batch(Path(tmp));driver=Path(tmp)/'fake_stage.py';driver.write_text(DRIVER)
        command=r'''source "$1"
launch_stage() { "$TASK_PYTHON" "$TEST_DRIVER" "$TEST_BASE" "$1" "$3" "$5" "$TEST_MODE"; }
execute_unit "$TEST_BATCH" L3_N7
'''
        env={**os.environ,'NEURON_MICROSCOPE_ROOT':tmp,'NEURON_PAPER_PYTHON':sys.executable,
             'TEST_DRIVER':str(driver),'TEST_BASE':str(BASE),'TEST_MODE':mode,'TEST_BATCH':str(batch)}
        result=subprocess.run(['bash','-c',command,'test',str(BASE/'run.sh')],env=env,capture_output=True,text=True)
        return batch,result

    def test_serial_handoff_json_html_and_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch,result=self.execute(tmp)
            self.assertEqual(result.returncode,0,result.stderr)
            unit=batch/'units/L3_N7'
            self.assertEqual((unit/'order.txt').read_text().splitlines(),[s for s,k in STAGES])
            first=read(unit/'01-interpretation/handoff/handoff.json')
            second=read(unit/'02-experiments/handoff/handoff.json')
            final=read(unit/'03-revision/handoff/handoff.json')
            self.assertEqual(first['interpretation']['claim']['title'],'Initial')
            self.assertEqual(second['interpretation']['claim']['title'],'Initial')
            self.assertEqual(final['interpretation']['claim']['title'],'Revised')
            self.assertTrue((unit/'03-revision/handoff/measurements/sweep-001/result.json').exists())
            self.assertEqual(read(unit/'PIPELINE_TERMINAL.json')['status'],'complete')

    def test_container_mounts_one_skill_and_readonly_large_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch=prepare_batch(Path(tmp));names=[]
            for stage,skill in STAGES:
                folder=batch/'units/L3_N7'/stage
                previous='' if stage=='01-interpretation' else str(Path(tmp)/'previous')
                command=r'''source "$1"
NETWORK=net; GATEWAY=172.1.0.1; METER_PORT=1234; IMAGE=test-image
build_container_command "$2" L3_N7 "$3" "$4" "$5" "$6"
printf '%s\0' "${CMD[@]}"
'''
                env={**os.environ,'NEURON_MICROSCOPE_ROOT':tmp,'NEURON_PAPER_PYTHON':sys.executable}
                result=subprocess.run(['bash','-c',command,'test',str(BASE/'run.sh'),str(folder),stage,skill,previous,str(Path(tmp)/'socket')],env=env,check=True,capture_output=True)
                cmd=result.stdout.decode().rstrip('\0').split('\0');names.append(cmd[cmd.index('--name')+1])
                mounts=[cmd[i+1] for i,v in enumerate(cmd) if v=='--mount']
                skills=[m for m in mounts if 'dst=/skills/' in m]
                self.assertEqual(len(skills),1)
                self.assertTrue(skills[0].endswith('dst=/skills/'+skill+',readonly'))
                if stage=='01-interpretation':
                    self.assertFalse(any('dst=/inputs/activations' in m for m in mounts))
                    inputs=[m for m in mounts if 'dst=/inputs/' in m]
                    self.assertEqual(inputs,['type=bind,src='+str(folder/'input/report')+',dst=/inputs/report,readonly'])
                else:
                    self.assertTrue(any('dst=/inputs/activations,readonly' in m for m in mounts))
                models=[m for m in mounts if 'dst=/inputs/model' in m]
                self.assertEqual(len(models),int(stage=='02-experiments'))
                if models:self.assertTrue(models[0].endswith(',readonly'))
                self.assertTrue(any('dst=/workspace' in m and not m.endswith(',readonly') for m in mounts))
                if previous:self.assertTrue(any('dst=/inputs/handoff,readonly' in m for m in mounts))
            self.assertEqual(len(set(names)),3)

    def test_first_stage_package_excludes_old_answers_and_other_neurons(self):
        from launch_files import stage_input,INITIAL_DATA_FILES
        with tempfile.TemporaryDirectory() as tmp:
            batch=prepare_batch(Path(tmp));folder=batch/'units/L3_N7/01-interpretation'
            cfg=read(folder/'host-session.json')
            self.assertNotIn('activations',cfg)
            self.assertEqual(cfg['report'],str(folder/'input/report'))
            stage_input(folder);target=Path(cfg['report'])
            self.assertEqual({p.name for p in target.iterdir()},set(INITIAL_DATA_FILES)|{'index.html'})
            exported=''.join(p.read_text() for p in target.iterdir())
            self.assertIn('TARGET_ONLY',exported)
            self.assertNotIn('OLD_ANSWER',exported)
            self.assertNotIn('OTHER_NEURON',exported)
            self.assertFalse(any(p.is_symlink() for p in target.iterdir()))
            self.assertIn('OLD_ANSWER',(Path(cfg['source_report'])/'index.html').read_text())

    def test_stage_service_and_container_cleanup_without_real_docker(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch=prepare_batch(Path(tmp));folder=batch/'units/L3_N7/01-interpretation'
            bindir=Path(tmp)/'bin';bindir.mkdir();docker=bindir/'docker'
            docker.write_text('#!/bin/sh\nprintf "%s\\n" "$1" >> "$TEST_DOCKER_LOG"\nif [ "$1" = run ]; then exit "$TEST_DOCKER_EXIT"; fi\n')
            docker.chmod(0o755);log=Path(tmp)/'docker.log'
            command=r'''source "$1"
NETWORK=net; GATEWAY=172.1.0.1; METER_PORT=1234; IMAGE=test-image; AGENT_WALL=30
launch_stage "$2" L3_N7 01-interpretation interpret ""
'''
            for code in [0,7]:
                env={**os.environ,'NEURON_MICROSCOPE_ROOT':tmp,'NEURON_PAPER_PYTHON':sys.executable,
                     'PATH':str(bindir)+os.pathsep+os.environ['PATH'],
                     'TEST_DOCKER_LOG':str(log),'TEST_DOCKER_EXIT':str(code)}
                result=subprocess.run(['bash','-c',command,'test',str(BASE/'run.sh'),str(folder)],
                                      env=env,capture_output=True,text=True,timeout=45)
                self.assertEqual(result.returncode,code,result.stderr)
                self.assertEqual(read(folder/'CONTROL_TERMINAL.json')['exit_code'],code)
            self.assertEqual(log.read_text().splitlines(),['run','stop','rm']*2)
            self.assertEqual(list((Path(tmp)/'.local/np-sockets').iterdir()),[])

    def test_process_failure_does_not_launch_next_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch,result=self.execute(tmp,'fail');unit=batch/'units/L3_N7'
            self.assertEqual(result.returncode,7,result.stderr)
            self.assertEqual((unit/'order.txt').read_text().splitlines(),['01-interpretation','02-experiments'])
            self.assertEqual(read(unit/'PIPELINE_TERMINAL.json')['exit_code'],7)
            self.assertFalse((unit/'03-revision/handoff').exists())

    def test_unmeasured_plans_are_carried_to_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            batch,result=self.execute(tmp,'unmeasured')
            self.assertEqual(result.returncode,0,result.stderr)
            folder=batch/'units/L3_N7/03-revision/handoff'
            self.assertTrue((folder/'measurements/sweep-001/failure.json').exists())
            data=read(folder/'handoff.json')
            self.assertEqual(data['measurement_summary'][0]['status'],'not_measured')

if __name__=='__main__':unittest.main()
