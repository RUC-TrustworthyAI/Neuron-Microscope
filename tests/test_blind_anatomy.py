import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from common import read,write
from blind_anatomy import context,frozen_explanation,predict,apply,reveal_scan,finalize,prior,prediction_points


class BlindAnatomyTests(unittest.TestCase):
    def output(self):
        return {'claim':{'title':'Test neuron','one_paragraph':'Predict negative at the end of a request.'},
            'response_rules':[{'id':'request','name':'Request ending','description':'Final token of a request.',
                               'predicted_pole':'negative'}]}

    def point(self):
        return {'id':'scan-SECRET','source':'corpus:scan','source_family':'corpus','document':'unseen',
                'position':2,'gate':10,'up':12,'activation':120,'bin':99,
                'cluster':'SECRET_TOPIC','peak_token':' please','observed_prefix':'Help me please',
                'label_reason':'SECRET_ANSWER','evaluation_split':'feedback'}

    def test_predictor_only_sees_fixed_explanation_full_prefix_and_anchor(self):
        point=self.point();calls=[]
        def llm(messages,*args):
            packet=json.loads(messages[1]['content']);calls.append(packet)
            self.assertEqual(set(packet),{'explanation','response_rules','points'})
            self.assertEqual(set(packet['points'][0]),{'id','observed_prefix','anchor'})
            self.assertEqual(packet['points'][0]['anchor'],{'position':2,'token':' please'})
            for forbidden in ('SECRET','activation','gate','evaluation_split','feedback'):
                self.assertNotIn(forbidden,json.dumps(packet))
            return {'predictions':[{'id':packet['points'][0]['id'],'group':'request',
                                   'predicted_pole':'negative','reason':'Request ending.'}]}
        with tempfile.TemporaryDirectory() as tmp:
            frozen=frozen_explanation(self.output())
            saved=predict([point],{},frozen,Path(tmp)/'v1',llm=llm)
            bundle={'anatomy':{'points':[point]}}
            apply(bundle,saved)
            self.assertEqual(point['predicted_pole'],'negative')
            self.assertEqual(point['activation'],120)
            self.assertEqual(bundle['anatomy']['quadrants']['request']['g+u+'],1)
            self.assertFalse(saved['records'][0]['correct'])
            # Truth changes do not enter the request or invalidate the text prediction.
            point['activation']=-1
            cached=predict([point],{},frozen,Path(tmp)/'cached',[saved],llm=llm)
            self.assertEqual(len(calls),1)
            self.assertTrue(cached['records'][0]['correct'])
            changed=self.output();changed['claim']['one_paragraph']='A revised condition.'
            new=predict([point],{},frozen_explanation(changed),Path(tmp)/'v2',[saved],llm=llm)
            self.assertEqual(len(calls),2)
            self.assertNotEqual(new['version'],saved['version'])
            self.assertEqual(read(Path(tmp)/'v1/response-predictions.json')['version'],saved['version'])

    def test_natural_prefix_keeps_all_prior_tokens_and_no_future(self):
        p={'document':'d','position':600,'source':'corpus:selected'}
        data={'documents':{'d':{'tokens':[str(i)+' ' for i in range(700)]}}}
        key,visible=context(p,data)
        self.assertTrue(visible['observed_prefix'].startswith('0 1 2 '))
        self.assertTrue(visible['observed_prefix'].endswith('600 '))
        self.assertNotIn('601 ',visible['observed_prefix'])

    def test_missing_prediction_stays_uncertain(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=predict([self.point()],{},frozen_explanation(self.output()),tmp,
                           llm=lambda *a:{'predictions':[]})
        self.assertEqual(result['records'][0]['predicted_pole'],'uncertain')
        self.assertEqual(result['summary']['feedback']['coverage'],0)
        self.assertEqual(result['records'][0]['group'],'no-visible-cue')

    def test_submission_has_one_no_visible_cue_bucket(self):
        output=self.output()
        output['response_rules']=[{'id':pole+str(i),'name':pole+str(i),'predicted_pole':pole}
                                  for pole in ('positive','negative') for i in range(3)]
        output['response_rules'] += [{'id':'ambiguous','predicted_pole':'uncertain'},
                                     {'id':'background','predicted_pole':'uncertain'}]
        rules=frozen_explanation(output)['response_rules']
        self.assertEqual(len(rules),7)
        self.assertEqual([r['id'] for r in rules if r['predicted_pole']=='uncertain'],['no-visible-cue'])
        self.assertEqual(rules[-1]['name'],'No visible cue')
        self.assertEqual(len(frozen_explanation({})['response_rules']),1)

    def test_uncertain_predictions_share_one_bucket(self):
        def llm(messages,*args):
            packet=json.loads(messages[1]['content'])
            return {'predictions':[{'id':p['id'],'group':'request','predicted_pole':'uncertain',
                                   'reason':'The direction is ambiguous.'} for p in packet['points']]}
        with tempfile.TemporaryDirectory() as tmp:
            result=predict([self.point()],{},frozen_explanation(self.output()),tmp,llm=llm)
        self.assertEqual(result['records'][0]['group'],'no-visible-cue')
        self.assertEqual(result['records'][0]['activation'],120)

    def test_spectrum_and_scatter_share_predictions_without_sign_based_recolouring(self):
        spectrum={'edges':[-10,0,10],'corpus_hist':{'counts':[700,300]},
                  'bins':[{'n_labeled':1,'groups':{'old-topic':1}},{'n_labeled':1,'groups':{'old-topic':1}}],
                  'groups':[{'id':'old-topic','pole':'positive'}],
                  'points':[{'id':'1','bin':0,'activation':-5},{'id':'2','bin':1,'activation':5}]}
        bundle={'spectrum':copy.deepcopy(spectrum),'anatomy':{'points':[]}}
        saved={'version':'fixed','response_rules':[{'id':'positive','predicted_pole':'positive'}],
               'records':[{'id':'spectrum-1','identity':'x','group':'positive','predicted_pole':'positive'}],
               'summary':{}}
        apply(bundle,saved)
        spec=bundle['spectrum']
        self.assertEqual(spec['corpus_hist'],spectrum['corpus_hist'])
        self.assertEqual([p['activation'] for p in spec['points']],[-5,5])
        self.assertEqual([p['group'] for p in spec['points']],['positive','no-visible-cue'])
        self.assertEqual(spec['bins'][0]['groups'],{'positive':1})
        self.assertEqual({g['id']:g['pole'] for g in spec['groups']},{'positive':'positive','no-visible-cue':'either'})
        self.assertNotIn('old-topic',json.dumps(spec))

    def test_joint_prediction_pool_only_adds_missing_spectrum_positions(self):
        point={'id':'1','source':'corpus:selected','source_family':'corpus','document':'d','position':0,'activation':3}
        bundle={'anatomy':{'points':[point]},'spectrum':{'points':[{'id':'1'},{'id':'2'}]}}
        data={'points':{'1':{'document':'d','position':0,'values':[1,3,3]},
                        '2':{'document':'d','position':1,'values':[1,-2,-2]}}}
        rows=prediction_points(bundle,data)
        self.assertEqual([p['id'] for p in rows],['1','spectrum-2'])
        self.assertEqual(rows[1]['position'],1)

    def test_ten_batches_run_together_and_keep_every_prediction(self):
        points=[{**self.point(),'id':str(i),'document':str(i),'observed_prefix':'Request %d please'%i}
                for i in range(306)]
        barrier=threading.Barrier(10,timeout=5);lock=threading.Lock();calls=[]
        def llm(messages,folder,route):
            packet=json.loads(messages[1]['content'])
            with lock:calls.append((str(folder),len(packet['points'])))
            barrier.wait()
            return {'predictions':[{'id':p['id'],'group':'request','predicted_pole':'negative'}
                                   for p in packet['points']]}
        with tempfile.TemporaryDirectory() as tmp:
            frozen=frozen_explanation(self.output())
            saved=predict(points,{},frozen,Path(tmp)/'first',llm=llm)
            self.assertEqual(len(calls),10)
            self.assertEqual(len({path for path,_ in calls}),10)
            self.assertEqual(sorted(size for _,size in calls),[18]+[32]*9)
            self.assertEqual(len(saved['assignments']),306)
            self.assertEqual(read(Path(tmp)/'first/predictions-checkpoint.json'),saved['assignments'])
            self.assertEqual([r['id'] for r in saved['records']],[p['id'] for p in points])
            cached=predict(points,{},frozen,Path(tmp)/'cached',[saved],llm=llm)
            self.assertEqual(cached['llm_requests'],0)
            self.assertEqual(cached['reused_predictions'],306)
            self.assertEqual(len(calls),10)

    def test_parallel_failure_preserves_successful_batches_without_retry(self):
        points=[{**self.point(),'id':str(i),'document':str(i),'observed_prefix':'Request %d please'%i}
                for i in range(10)]
        barrier=threading.Barrier(10,timeout=5);lock=threading.Lock();calls=[]
        def llm(messages,folder,route):
            packet=json.loads(messages[1]['content'])
            with lock:calls.append(str(folder))
            barrier.wait()
            if folder.parent.name=='batch-001':raise RuntimeError('upstream unavailable')
            return {'predictions':[{'id':p['id'],'group':'request','predicted_pole':'negative'}
                                   for p in packet['points']]}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'upstream unavailable'):
                predict(points,{},frozen_explanation(self.output()),tmp,llm=llm,batch_size=1)
            self.assertEqual(len(calls),10)
            self.assertEqual(len(read(Path(tmp)/'predictions-checkpoint.json')),9)
            self.assertFalse((Path(tmp)/'response-predictions.json').exists())

    def test_holdout_revealed_only_after_final_submission(self):
        from anatomy_scan import saved
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);private=root/'private';private.mkdir();workspace=root/'workspace';workspace.mkdir()
            p=self.point();q={**p,'id':'heldout','document':'reserved','observed_prefix':'SECRET_HOLDOUT','evaluation_split':'final_holdout'}
            scan={'points':[p,q],'edges':[-200,0,200],'seed':42}
            write(private/'selected-scan.json',scan)
            session=SimpleNamespace(config={'private_scan':str(private)},root=workspace,jobs=[])
            request={'report':str(root/'report'),'sections':[]}
            reveal_scan(session,request,False)
            paths=saved(request)
            self.assertEqual(len(read(paths[-1])['points']),1)
            self.assertNotIn('SECRET_HOLDOUT',json.dumps(read(paths[-1])))
            self.assertNotIn(str(private),json.dumps(session.jobs))
            # A prior report's feedback artifact must not override the final scan.
            artifact=root/'report/artifacts/old';artifact.mkdir(parents=True)
            write(artifact/'anatomy-scan.json',read(paths[-1]))
            reveal_scan(session,request,True)
            self.assertEqual(len(read(saved(request)[-1])['points']),2)

    def test_stage_hook_renders_predictions_without_publishing_private_scan(self):
        from test_initial_scan import InitialScanTests
        from report import run,source_bundle
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report,data,cfg=InitialScanTests().fixture(root)
            workspace=root/'workspace';workspace.mkdir();private=root/'private';private.mkdir()
            cfg['private_scan']=str(private)
            point={**self.point(),'position':0,'id':'scan-1','gate':1,'up':2,'activation':2}
            hidden={**point,'id':'scan-2','document':'reserved','observed_prefix':'HIDDEN_SENTINEL','evaluation_split':'final_holdout'}
            write(private/'selected-scan.json',{'points':[point,hidden],'edges':[-10,0,10],
                'seed':42,'scan_seconds':1,'tokens_scanned':1000,'candidate_points_per_bin':40})
            interpretation=root/'interpretation';interpretation.mkdir()
            write(interpretation/'output.json',self.output());write(interpretation/'annotations.json',{})
            request={'report':str(report),'model':cfg['model'],'settings':cfg['settings'],'stage':'01-interpretation',
                'interpretation':str(interpretation),'annotations':str(interpretation/'annotations.json'),
                'sections':[],'graphs':[],'sweeps':[]}
            session=SimpleNamespace(config=cfg,root=workspace,jobs=[],stage='01-interpretation')
            def llm(messages,*args):
                packet=json.loads(messages[1]['content'])
                self.assertNotIn('HIDDEN_SENTINEL',json.dumps(packet))
                return {'predictions':[{'id':p['id'],'group':'request','predicted_pole':'negative'} for p in packet['points']]}
            with patch('blind_tests.call',llm),patch.dict('sys.modules',{'tokenizers':SimpleNamespace(Tokenizer=SimpleNamespace(from_file=lambda path:None))}):
                finalize(session,request)
            self.assertEqual(len(prior(request)[-1]['records']),2)
            output=root/'paper';output.mkdir()
            with patch('report.model_config',return_value=(root/'model',{})):
                run(request,output)
            text=(output/'index.html').read_text();bundle,_=source_bundle(text)
            self.assertNotIn('HIDDEN_SENTINEL',text)
            self.assertEqual(len(bundle['anatomy']['points']),2)
            self.assertEqual({p['predicted_pole'] for p in bundle['anatomy']['points']},{'negative'})
            self.assertTrue(list((output/'artifacts').glob('*/response-predictions.json')))
            from stage_session import StageSession
            revision=object.__new__(StageSession);revision.config={'report':str(output)};revision.jobs=[]
            feedback=revision.evidence({'section':'response_predictions','limit':4})
            self.assertEqual(feedback['version'],prior(request)[-1]['version'])
            self.assertIn('feedback',feedback['summary'])
            self.assertTrue(feedback['cases'])
            for path in (output/'artifacts').rglob('*.json'):
                self.assertNotIn('HIDDEN_SENTINEL',path.read_text())

    def test_renderer_keeps_prediction_colour_when_measurement_disagrees(self):
        from report_layout import anatomy_colours
        source="(() => { const A = B.anatomy; const figs = []; const pts = A.points || []; const dot={fill: clusterColour(p.cluster)}; })();"
        result=anatomy_colours(source)
        self.assertIn('fill: predictedColour(p.predicted_pole)',result)
        self.assertNotIn('measuredColour',result)
        self.assertEqual(anatomy_colours(result),result)


if __name__=='__main__':unittest.main()
