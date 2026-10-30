"""Packaging contracts: paths, frozen settings, credential and host-context boundaries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from settings import load, paths, effective, agent_settings
from api_meter import Budget
from common import write, read

BASE=Path(__file__).resolve().parents[1]

class PortabilityTests(unittest.TestCase):
    def test_prepare_freezes_paths_and_never_copies_host_metadata_to_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);settings=load()
            settings['paths']={'reports':'measurements/reports','activations':'measurements/native',
                               'model':'weights','state':'state'}
            settings['gpu_groups']=[[1,5,6,7]]
            file=root/'settings.json';write(file,settings)
            out=root/'batch'
            env={**os.environ,'NEURON_PAPER_PYTHON':sys.executable}
            subprocess.run([str(BASE/'run.sh'),'prepare','--output',str(out),
                            '--settings',str(file),'--neurons','L1_N2'],env=env,check=True,capture_output=True)
            manifest=read(out/'manifest.json')
            self.assertEqual(manifest['paths']['model'],str(root/'weights'))
            self.assertEqual(manifest['paths']['tokenizer'],str(root/'weights/tokenizer.json'))
            self.assertEqual(manifest['start_next_after'],'')
            self.assertEqual(manifest['stage2_lock'],str(root/'state/gpu-locks/stage2-1-5-6-7.lock'))
            settings['gpu_groups']=[[99]];write(file,settings)
            self.assertEqual(read(out/'framework-source/settings.json')['gpu_groups'],[[1,5,6,7]])
            self.assertFalse((out/'framework-source/.git').exists())
            self.assertFalse((out/'framework-source/runs').exists())
            self.assertFalse((out/'framework-source/settings.local.json').exists())
            for stage in (out/'units/L1_N2').iterdir():
                control=read(stage/'control/settings.json')
                self.assertFalse(set(control)&{'pricing','llm','paths','execution','gpu_groups'})
                cfg=read(stage/'host-session.json')
                self.assertFalse(set(cfg['settings'])&{'pricing','llm','paths','execution'})
            first=read(out/'units/L1_N2/01-interpretation/host-session.json')
            self.assertNotIn('activations',first)
            self.assertEqual(first['preparation_activations'],str(root/'measurements/native'))

    def test_key_is_external_and_never_written_in_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write(root/'manifest.json',{'settings':load(),'budget_usd':5})
            with patch.dict(os.environ,{'DEEPSEEK_API_KEY':'software-fixture-not-a-real-key'},clear=True):
                budget=Budget(root)
            self.assertEqual(budget.limit,5)
            self.assertNotIn(budget.key,(root/'logs/usage.json').read_text())
            self.assertEqual(budget.llm['reasoning_effort'],'high')
            with patch.dict(os.environ,{},clear=True):
                with self.assertRaisesRegex(ValueError,'Set DEEPSEEK_API_KEY'):Budget(root)

    def test_scientific_settings_exclude_operator_configuration(self):
        result=effective({'settings':load()})
        self.assertFalse(set(result)&{'paths','pricing','llm','execution'})
        self.assertIn('gpu_groups',result)
        self.assertNotIn('gpu_groups',agent_settings(result))

if __name__=='__main__':unittest.main()
