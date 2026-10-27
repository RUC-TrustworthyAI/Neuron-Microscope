"""Exercise evidence batching and display-only updates without API or GPU work."""
import copy
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from common import read,write
from report import source_bundle,merge
from stage_session import StageSession


class StageToolTests(unittest.TestCase):
    def fixture(self,root,stage='03-revision'):
        report=root/'source';report.mkdir()
        workspace=root/'workspace';workspace.mkdir()
        bundle={'neuron':{'id':'L1_N2','layer':1,'index':2},
                'claim':{'title':'A fixed explanation','negative_pole':'A fixed condition.'},
                'criticism':[{'verdict':'narrowed','result':'A saved limitation.'}],
                'agent_read':{'group':'unplaced'},'distribution':{'n_tokens':1000},
                'custom':{'figures':[{'id':'s1','title':'Old title','caption':'Old caption',
                                     'rows':[{'activation':12.5}]}]},
                'graphs_curated':{'picks':[{'id':'g1','title':'Old graph','graph':{'nodes':[]}}]},
                'anatomy':{'points':[],'explanation_version':'frozen'},
                'section_prose':{'circuit':{'summary':'Old introduction.'}}}
        text='<script id="bundle" type="application/json">'+json.dumps(bundle)+'</script>'
        text+="<script>const B = JSON.parse(document.getElementById('bundle').textContent); const POS = 1;</script>"
        text,bundle=merge(text,{})
        (report/'index.html').write_text(text)
        write(report/'annotations.json',{'figures':{'s1':{'title':'Old title','caption':'Old caption'}}})
        artifact=report/'artifacts/predictions';artifact.mkdir(parents=True)
        prefix='Complete observed prefix. '*1000
        records=[{'id':'wrong','activation':-20,'predicted_pole':'positive','group':'positive',
                  'correct':False,'observed_prefix':prefix,'anchor':{'position':2,'token':'anchor'}},
                 {'id':'right','activation':30,'predicted_pole':'positive','group':'positive',
                  'correct':True,'observed_prefix':'A correct case.','anchor':{'position':2,'token':'anchor'}}]
        write(artifact/'response-predictions.json',{'version':'frozen','explanation':bundle['claim'],
            'response_rules':[{'id':'positive','predicted_pole':'positive'}],
            'records':records,'summary':{'feedback':{'n':2,'correct':1}}})
        packet={'contexts_json':{'cases':[{'id':'wrong','groups':['bottom_tokens'],
                                          'observed_prefix':prefix,'anchor':{'product':-20}}],
                                 'trace_columns':['offset','token','gate','value','product']},
                'distribution_json':{'channels':{'product':{'max':30,'min':-20}}},
                'token_selectivity_json':{},'peak_census_json':{},'gate_value_json':{},'output_effects_json':{}}
        write(workspace/'agent-evidence.json',packet)
        measured=workspace/'measurements';measured.mkdir()
        write(measured/'result.json',{'status':'complete','sections':['search']})
        write(measured/'sections.json',{'search':{'n_texts':10,'best_negative':-22}})
        session=object.__new__(StageSession)
        session.root=workspace;session.stage=stage;session.lock=threading.Lock()
        session.config={'stage':stage,'report':str(report),'neuron':{'layer':1,'index':2}}
        session.prior={'interpretation':{'claim':bundle['claim'],'response_rules':[{'id':'positive'}]}}
        session.jobs=[{'tool':'paper_sections','request':{'id':'measured-search','output':str(measured)}}]
        return session,report,bundle,packet

    def test_brief_combines_measurements_feedback_and_section_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            session,report,bundle,packet=self.fixture(Path(tmp),'02-experiments')
            with patch('blind_tests.call',side_effect=AssertionError('Unexpected API call')):
                brief=session.evidence({'section':'brief','limit':1})
            self.assertEqual(brief['interpretation']['claim'],bundle['claim'])
            self.assertEqual(brief['sections']['circuit']['introduction']['summary'],'Old introduction.')
            self.assertTrue(brief['sections']['search']['available'])
            measured=brief['sections']['search']['measurements'][0]
            self.assertEqual(measured['statistics']['best_negative'],-22)
            self.assertEqual(brief['experiment_index'][0]['id'],'measured-search')
            self.assertEqual(brief['experiment_index'][0]['path'],'/workspace/measurements')
            self.assertIn('sections.json',brief['experiment_index'][0]['files'])
            self.assertEqual(brief['response_predictions']['cases'][0]['id'],'wrong')
            self.assertEqual(brief['response_predictions']['cases'][0]['observed_prefix'],
                             packet['contexts_json']['cases'][0]['observed_prefix'])
            self.assertEqual(brief['response_predictions']['next_offset'],1)

    def test_multi_section_read_preserves_complete_requested_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            session,_,_,packet=self.fixture(Path(tmp))
            result=session.evidence({'sections':['distribution','contexts'],'ids':['wrong']})
            self.assertEqual(result['sections']['distribution']['data'],packet['distribution_json'])
            self.assertEqual(result['sections']['contexts']['cases'],packet['contexts_json']['cases'])

    def test_handoff_measurement_paths_match_the_container_mount(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);session,report,_,_=self.fixture(root)
            incoming=root/'incoming';incoming.mkdir()
            session.config['handoff']=str(incoming)
            folder=Path(session.jobs[0]['request']['output'])
            moved=incoming/'measurements';folder.rename(moved)
            session.jobs[0]['request']['output']=str(moved)
            result=session.evidence({'section':'measurements'})
            self.assertEqual(result['measurements'][0]['path'],'/inputs/handoff/measurements')

    def test_cli_accepts_multi_section_flags_without_reading_stdin(self):
        from rpc_client import _arguments
        with patch.object(sys,'argv',['read_evidence.py','--sections','distribution','contexts','--ids','wrong']):
            _,arguments=_arguments('evidence')
        self.assertEqual(arguments,{'sections':['distribution','contexts'],'ids':['wrong']})

    def test_display_updates_preserve_claim_predictions_and_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            session,source,bundle,_=self.fixture(Path(tmp))
            original={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
            saved={'stage':session.stage,'interpretation':copy.deepcopy(session.prior['interpretation']),
                   'agent_output':{'claim':bundle['claim']},'measurements':[{'id':'measured-search'}],
                   'report':str(source/'index.html')}
            write(session.root/'stage-result.json',saved)
            with (patch('paper_reading.run',side_effect=AssertionError('Unexpected review')),
                 patch('blind_anatomy.finalize',side_effect=AssertionError('Unexpected predictions')),
                 patch('session.default_runner',side_effect=AssertionError('Unexpected GPU work'))):
                first=session.call('render_report',{'annotations':{
                    'figures':{'s1':{'caption':'A revised caption.','rows':[{'activation':999}]}},
                    'graphs':{'g1':{'why':'A revised graph explanation.'}}}})
                second=session.call('render_report',{'annotations':{'sections':{
                    'circuit':{'summary':'A revised introduction.'}}}})
            latest=read(session.root/'stage-result.json')
            output=Path(latest['report']).parent
            after,_=source_bundle((output/'index.html').read_text())
            for field in ('claim','criticism','agent_read','anatomy','distribution','tests'):
                self.assertEqual(after[field],bundle[field])
            self.assertEqual(after['custom']['figures'][0]['caption'],'A revised caption.')
            self.assertEqual(after['custom']['figures'][0]['title'],'Old title')
            self.assertEqual(after['custom']['figures'][0]['rows'],[{'activation':12.5}])
            self.assertEqual(after['graphs_curated']['picks'][0]['why'],'A revised graph explanation.')
            self.assertEqual(latest['interpretation'],saved['interpretation'])
            self.assertEqual(latest['measurements'],saved['measurements'])
            self.assertEqual(read(session.root/'latest-paper.json')['path'],latest['report'])
            self.assertEqual(read(output/'annotations.json')['figures']['s1']['caption'],'A revised caption.')
            self.assertEqual({str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()},original)
            for result in (first,second):
                self.assertEqual(result['llm_requests'],0)
                self.assertEqual(result['new_model_forwards'],0)
            self.assertEqual((output/'artifacts/predictions/response-predictions.json').read_bytes(),
                             original['artifacts/predictions/response-predictions.json'])


if __name__=='__main__':unittest.main()
