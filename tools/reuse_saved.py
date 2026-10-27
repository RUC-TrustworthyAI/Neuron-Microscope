"""Assemble a measured handoff without starting an experiment agent."""
import json
import sys
sys.path.insert(0,'/control')
from rpc_client import call

if __name__=='__main__':
    result=call('complete_experiments',{})
    print(json.dumps(result,ensure_ascii=False))
    if result.get('error_type'):
        raise RuntimeError(result.get('error','Saved experiment handoff failed.'))
