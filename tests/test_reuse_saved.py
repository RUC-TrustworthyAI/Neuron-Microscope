import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from common import read,write
from stage_session import StageSession
from blind_anatomy import mark_reused_samples


class ReuseSavedTests(unittest.TestCase):
    def test_import_keeps_current_feedback_and_omits_old_prediction_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';workspace=root/'workspace'
            source.mkdir();workspace.mkdir();entries=[]
            for name,extra in [('old-scan','anatomy-scan.json'),('old-labels','anatomy-semantics.json'),
                               ('old-prediction','response-predictions.json'),('sweep-1','conditions.json')]:
                folder=source/name;folder.mkdir();write(folder/extra,{})
                write(folder/'tool-request.json',{'id':name,'output':str(folder)})
                write(folder/'sections.json',{'tests':{'tested_explanation':'OLD','explanation_version':'v0'}})
                write(folder/'messages.json',{'old_request':'OLD'})
                entries.append({'id':name,'tool':'paper_sections','path':name})
            write(source/'handoff.json',{'measurements':entries})
            s=SimpleNamespace(config={'reuse_measurements':str(source),'reuse_completed_experiments':True},
                root=workspace,jobs=[{'tool':'paper_sections','request':{'id':'current-feedback'}}])
            StageSession.import_measurements(s)
            self.assertEqual([j['request']['id'] for j in s.jobs],['current-feedback','sweep-1'])
            copied=workspace/'reused-measurements/sweep-1'
            self.assertFalse((copied/'messages.json').exists())
            test=read(copied/'sections.json')['tests']
            self.assertNotIn('tested_explanation',test)
            self.assertEqual(test['explanation_version'],'v0')

    def test_complete_uses_current_claim_without_launching_experiments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';old.mkdir()
            write(old/'handoff.json',{'interpretation':{'claim':'OLD'},
                'agent_output':{'annotations':{'graph_ids':['graph-1']}}})
            def forbidden(*a,**k):raise AssertionError('Unexpected experiment')
            s=SimpleNamespace(stage='02-experiments',root=root,jobs=[],
                config={'existing_only':True,'reuse_completed_experiments':True,'reuse_measurements':str(old)},
                prior={'interpretation':{'claim':{'title':'CURRENT'},'response_rules':[{'id':'new-rule'}]}},
                run_experiments=forbidden,run_tests=forbidden)
            # Calling through the actual subclass keeps super().finish dispatch meaningful.
            session=object.__new__(StageSession);session.__dict__.update(s.__dict__)
            with patch('session.Session.finish',return_value={'report':'paper/index.html'}) as finish:
                StageSession.complete(session,{})
            payload=finish.call_args.args[0]
            self.assertEqual(payload['claim']['title'],'CURRENT')
            self.assertEqual(payload['annotations']['graph_ids'],['graph-1'])

    def test_reused_sample_is_preserved_but_not_counted_as_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);write(folder/'atlas-raw.json',[{'source_document':'exposed','text':'seen context'}])
            scan={'points':[{'id':'a','document':'exposed','evaluation_split':'final_holdout'},
                            {'id':'b','document':'new','evaluation_split':'final_holdout'}]}
            mark_reused_samples(scan,[{'request':{'output':str(folder)}}])
            self.assertEqual(len(scan['points']),2)
            self.assertEqual(scan['points'][0]['evaluation_split'],'seen_reused')
            self.assertEqual(scan['points'][1]['evaluation_split'],'final_holdout')


if __name__=='__main__':unittest.main()
