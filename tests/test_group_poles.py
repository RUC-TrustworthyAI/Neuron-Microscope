import copy,json,tempfile,unittest
from pathlib import Path
from group_poles import annotate,cluster_poles

class GroupPoleTests(unittest.TestCase):
    def test_measured_members_override_submitted_pole_without_api_call(self):
        data={'groups':[{'id':'mixed','pole':'negative'},{'id':'empty','pole':'positive'}],
              'points':[{'id':'1','group':'mixed','activation':300},
                        {'id':'2','group':'mixed','activation':-20}]}
        original=copy.deepcopy(data)
        def llm(*a,**k):raise AssertionError('No LLM needed for measured signs')
        with tempfile.TemporaryDirectory() as folder:
            result=annotate(data,folder,llm)
            status=json.loads((Path(folder)/'result.json').read_text())
        self.assertEqual(data,original)
        self.assertEqual(result['groups'][0]['pole'],'either')
        self.assertEqual(result['groups'][0]['response_summary']['positive'],1)
        self.assertEqual(result['groups'][1]['response_summary']['n'],0)
        self.assertEqual([p['response_pole'] for p in result['points']],['positive','negative'])
        self.assertEqual(status['llm_requests'],0)
        c=cluster_poles([{'id':'mixed','member_groups':['mixed'],'pole':'negative'}],result)
        self.assertEqual(c[0]['pole'],'either')

if __name__=='__main__':unittest.main()
