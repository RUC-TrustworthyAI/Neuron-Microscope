#!/usr/bin/env python3
"""Render synthetic saved activations using the shared analysis and report code."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'analysis')]
import test_batch_reports as fixture
import build_report as core
import build_reports as batch
from report_token_fields import TokenFields
from activation_report import initial_html
from report import merge


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    capture=out/'synthetic-input';capture.mkdir()
    run,inputs,targets,records,_,n_docs,raw,plan=fixture.fixture(capture)
    work=out/'statistics';work.mkdir()
    hist,quad,indices,scores=batch.first_pass(records,4,work,1)
    bands=batch.second_pass(records,4,hist,work,1,42)
    np.savez(out/'histograms.npz',counts=hist[0],gate_value_sign_counts=quad[0])
    extremes={key:[{'index':int(i),'value':float(s)} for i,s in zip(indices[0,k],scores[0,k])
                   if i!=np.iinfo(np.int64).max] for k,key in enumerate(batch.KEYS)}
    identity={'id':'L0_N0','layer':0,'neuron':0,'model':'Synthetic software fixture'}
    data=core.package(records,batch.CaptureView(records,0),hist[0],quad[0],extremes,bands[0],
                      inputs,fixture.TOKENIZER,identity,plan,42,n_docs)
    core.write(out/'report.json',data)
    batch.render_report(data,out)
    cache=out/'token-cache';cache.mkdir()
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(fixture.TOKENIZER))
    ids=np.load(inputs/'block-000000.ids.u32.npy')
    np.save(cache/'token-counts.i64.npy',np.bincount(ids,minlength=tok.get_vocab_size()).astype(np.int64))
    core.write(cache/'token-labels.json',[tok.decode([i],skip_special_tokens=False) for i in range(tok.get_vocab_size())])
    scan=TokenFields(cache).start(hist[0,2]);scan.update(ids,core.floats(raw[:,0,2]));scan.publish(out,'L0_N0')
    page=initial_html((out/'index.html').read_text(),data)
    page,bundle=merge(page,{'claim':{'title':'Synthetic demo · L0_N0',
        'lede':'Software fixture: synthetic channels, no model inference or LLM calls.',
        'one_paragraph':'—'}})
    (out/'index.html').write_text(page)
    core.write(out/'paper-data.json',{'bundle':bundle})
    core.write(out/'demo.json',{'synthetic':True,'model_forwards':0,'llm_requests':0,
                               'renderer':'shared saved-data and report modules'})
    print(out/'index.html')

if __name__=='__main__':main()
